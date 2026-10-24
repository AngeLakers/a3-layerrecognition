import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from a3_ultrasound.runtime_supervisor import (FileLease,RecoveryRequired,atomic_json,atomic_write,
    exclusive_json,guard_reason,identity,matching_process,receipt,recovery_action,validate_receipt,
    install_atomic_outputs,watchdog_loop,all_finite)


def test_failed_write_never_replaces_last_complete_result(tmp_path):
    target=tmp_path/'state.json';atomic_json(target,{'epoch':3})
    def crash(handle):handle.write(b'{"epoch":4');raise RuntimeError('simulated write failure')
    with pytest.raises(RuntimeError):atomic_write(target,crash)
    assert json.loads(target.read_text())=={'epoch':3}
    assert list(tmp_path.glob('*.pending_*'))


def test_validation_failure_never_promotes_partial_result(tmp_path):
    target=tmp_path/'state.json';atomic_json(target,{'done':True})
    with pytest.raises(json.JSONDecodeError):
        atomic_write(target,lambda f:f.write(b'{'),lambda p:json.loads(p.read_text()))
    assert json.loads(target.read_text())=={'done':True}


def test_exclusive_claim_and_os_lease_prevent_duplicate_start(tmp_path):
    claim=tmp_path/'claim.json';exclusive_json(claim,{'job':'one'})
    with pytest.raises(FileExistsError):exclusive_json(claim,{'job':'one'})
    lock=tmp_path/'queue.lock'
    with FileLease(lock):
        with pytest.raises(RecoveryRequired):
            with FileLease(lock):pass
    with FileLease(lock):pass


def test_process_crash_releases_lease(tmp_path):
    lock=tmp_path/'queue.lock';ready=tmp_path/'ready'
    import a3_ultrasound.runtime_supervisor as runtime
    source=Path(runtime.__file__)
    script="import sys,time,runpy;from pathlib import Path;r=runpy.run_path(sys.argv[1]);lease=r['FileLease'](sys.argv[2]);lease.__enter__();Path(sys.argv[3]).write_text('ready');time.sleep(30)"
    child=subprocess.Popen([sys.executable,'-c',script,str(source),str(lock),str(ready)])
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert ready.exists()
        with pytest.raises(RecoveryRequired):
            with FileLease(lock):pass
        child.kill();child.wait(timeout=5)
        with FileLease(lock):pass
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_process_killed_during_save_preserves_previous_checkpoint(tmp_path):
    target=tmp_path/'state.json';atomic_json(target,{'epoch':3})
    import a3_ultrasound.runtime_supervisor as runtime
    ready=tmp_path/'ready';source=Path(runtime.__file__)
    script="import sys,time,runpy;from pathlib import Path;r=runpy.run_path(sys.argv[1]);\ndef writing(f):\n f.write(b'{');f.flush();Path(sys.argv[3]).write_text('ready');time.sleep(30)\nr['atomic_write'](sys.argv[2],writing)"
    child=subprocess.Popen([sys.executable,'-c',script,str(source),str(target),str(ready)])
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert ready.exists();child.kill();child.wait(timeout=5)
        assert json.loads(target.read_text())=={'epoch':3}
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_atomic_tensor_and_numpy_roundtrip(tmp_path,monkeypatch):
    import numpy as np
    import torch
    import a3_ultrasound.stream_training as streaming
    for module,name in [(torch,'save'),(np,'save'),(np,'savez_compressed'),(streaming,'save'),(streaming,'progress')]:
        monkeypatch.setattr(module,name,getattr(module,name))
    install_atomic_outputs()
    value={'epoch':2,'updates_total':8,'model_state_dict':{'weight':torch.arange(6).reshape(2,3)}}
    torch.save(value,tmp_path/'epoch.pt');saved=torch.load(tmp_path/'epoch.pt',weights_only=True)
    assert torch.equal(saved['model_state_dict']['weight'],value['model_state_dict']['weight'])
    np.save(tmp_path/'array.npy',np.arange(4));assert np.array_equal(np.load(tmp_path/'array.npy'),np.arange(4))
    np.savez_compressed(tmp_path/'cache.npz',x=np.arange(7))
    with np.load(tmp_path/'cache.npz') as pack:assert np.array_equal(pack['x'],np.arange(7))


def test_owner_death_watchdog_stops_only_matching_child(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    active=tmp_path/'active.json';atomic_json(active,identity(psutil.Process(child.pid)))
    dead=identity(psutil.Process(os.getpid()));dead['created']-=10
    try:
        watchdog_loop(dead,active,tmp_path/'stop',tmp_path/'event.json',interval=.01)
        assert child.wait(timeout=5) is not None
        assert json.loads((tmp_path/'event.json').read_text())['child_stopped']
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_owner_heartbeat_timeout_stops_child_even_if_owner_alive(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    active=tmp_path/'active.json';heartbeat=tmp_path/'heartbeat.json'
    atomic_json(active,identity(psutil.Process(child.pid)));atomic_json(heartbeat,{'epoch':time.time()-180})
    os.utime(heartbeat,(time.time()-180,time.time()-180))
    try:
        watchdog_loop(identity(psutil.Process(os.getpid())),active,tmp_path/'stop',tmp_path/'event.json',
            interval=.01,heartbeat_file=heartbeat,heartbeat_limit=120)
        assert child.wait(timeout=5) is not None
        assert json.loads((tmp_path/'event.json').read_text())['child_stopped']
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_recovery_skips_completed_adopts_live_and_never_restarts_partial():
    assert recovery_action(completed_receipt={})=='VERIFY_AND_SKIP'
    assert recovery_action(claim_exists=True,live_process={})=='ADOPT_RUNNING'
    assert recovery_action(claim_exists=True)=='RECOVERY_REQUIRED'
    assert recovery_action(output_exists=True)=='RECOVERY_REQUIRED'
    assert recovery_action()=='START_NEW'


def test_changed_completed_artifact_is_rejected(tmp_path):
    target=tmp_path/'summary.json';atomic_json(target,{'status':'COMPLETED'})
    r=receipt(tmp_path,'job','manifest',[target]);assert validate_receipt(r,job_id='job',manifest_sha='manifest')
    target.write_text('{}')
    with pytest.raises(RecoveryRequired):validate_receipt(r,job_id='job',manifest_sha='manifest')


def test_receipt_rejects_empty_artifacts_wrong_root_and_omitted_required_file(tmp_path):
    target=tmp_path/'summary.json';atomic_json(target,{'status':'COMPLETED'})
    value=receipt(tmp_path,'job','manifest',[target])
    assert validate_receipt(value,job_id='job',manifest_sha='manifest',expected_root=tmp_path,required_files=['summary.json'])
    with pytest.raises(RecoveryRequired):
        validate_receipt(dict(value,artifacts={}),job_id='job',manifest_sha='manifest')
    with pytest.raises(RecoveryRequired):
        validate_receipt(value,job_id='job',manifest_sha='manifest',expected_root=tmp_path/'different')
    with pytest.raises(RecoveryRequired):
        validate_receipt(value,job_id='job',manifest_sha='manifest',required_files=['best.pt'])


def test_pid_reuse_identity_mismatch_cannot_be_adopted():
    saved=identity(psutil.Process(os.getpid()));assert matching_process(saved) is not None
    saved['created']-=10;assert matching_process(saved) is None


def test_vectorized_finite_guard_rejects_nan_and_inf_without_changing_values():
    import torch
    values=[torch.tensor([1.,-2.,3.]),torch.zeros(3)]
    originals=[v.clone() for v in values];assert all_finite(values)
    assert all(torch.equal(a,b) for a,b in zip(values,originals))
    for invalid in (float('nan'),float('inf'),-float('inf')):
        values[1][0]=invalid;assert not all_finite(values)


@pytest.mark.parametrize('kwargs,expected',[
    ({'idle_seconds':601,'elapsed_seconds':10,'free_bytes':20*1024**3},'NO_PROGRESS_TIMEOUT'),
    ({'idle_seconds':0,'elapsed_seconds':28801,'free_bytes':20*1024**3},'JOB_WALL_TIMEOUT'),
    ({'idle_seconds':0,'elapsed_seconds':10,'free_bytes':1024},'LOW_DISK_SPACE'),
    ({'idle_seconds':0,'elapsed_seconds':10,'free_bytes':20*1024**3,'stop_requested':True},'OPERATOR_STOP'),
    ({'idle_seconds':0,'elapsed_seconds':10,'free_bytes':20*1024**3},None)])
def test_guard_stop_conditions(kwargs,expected):assert guard_reason(**kwargs)==expected
