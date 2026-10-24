"""One bounded diagnostic job per explicit invocation, with existing ownership guards."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.runtime_supervisor import (FileLease,RecoveryRequired,atomic_json,exclusive_json,
    file_digest,identity,matching_process,terminate_owned,receipt,validate_receipt,watchdog_loop,activity_age,guard_reason)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def stop_spawned(proc,saved=None):
    """The live Popen handle is also an identity-safe fallback during setup."""
    if proc is None or proc.poll() is not None:return
    if saved is not None and terminate_owned(saved):return
    # Popen's Windows process handle names the child we created, not a guessed PID.
    proc.terminate()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait(timeout=10)


def verify(root):
    m=read(root/'manifest.json')
    if m['formal_training_permitted'] or m['outer_evaluation_permitted']:
        raise RecoveryRequired('Formal and outer must remain closed.')
    for name,digest in m['files'].items():
        if file_digest(root/name)!=digest:raise RecoveryRequired('Frozen deployment file changed: '+name)
    if m.get('protocol')=='V1.4-D2':
        from a3_ultrasound.d2_diagnostic import validate_config
        if m.get('candidate_only') or not m.get('development_training_permitted'):
            raise RecoveryRequired('D2 candidate is closed pending protocol review.')
        validate_config(read(root/'d2_config.json'))
    if m.get('protocol') in ('V1.4-freeze','V1.5-freeze'):
        from a3_ultrasound.freeze_protocol import validate_freeze_config,commit_identity,HEADS
        if m['protocol']=='V1.5-freeze':
            from a3_ultrasound.v15_protocol import validate_config as validate_freeze_config,HEADS,load_physical_artifacts
        from a3_ultrasound.training import _code_version
        cfg=read(root/'freeze_config.json')
        if not m.get('freeze_training_permitted'):
            raise RecoveryRequired('Waiting for the user to explicitly restart frozen training.')
        if set(m['jobs'])!={f'F1_{h}_s23' for h in HEADS}:
            raise RecoveryRequired('Freeze manifest differs from the approved Head set.')
        for head in HEADS:
            if m['jobs'][f'F1_{head}_s23']['head']!=head:raise RecoveryRequired('Freeze job-to-Head mapping differs.')
            validate_freeze_config(cfg,fold_name='F1',seed=23,head=head)
        commit_identity(cfg,_code_version())
        if m['protocol']=='V1.5-freeze':load_physical_artifacts(cfg)
    return m


def child(root,job):
    m=verify(root);session=root/'sessions'/job
    if matching_process(read(session/'owner.json')) is None or (root/'STOP').exists():
        raise RecoveryRequired('No owner or user stop.')
    import torch
    if '3080 Ti' not in torch.cuda.get_device_name(0):raise RecoveryRequired('Wrong target GPU.')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory))
    from a3_ultrasound.runtime_supervisor import install_atomic_outputs
    install_atomic_outputs()
    out=Path(m['jobs'][job]['output'])
    try:
        if m.get('protocol')=='V1.4-D2':
            if job!='d2':raise RecoveryRequired('D2 only accepts its registered pair job.')
            from a3_ultrasound.d2_diagnostic import main
            main(root/'d2_config.json',out)
        elif m.get('protocol') in ('V1.4-freeze','V1.5-freeze'):
            from a3_ultrasound.stream_training import train
            train(root/'freeze_config.json',out,fold_name='F1',seed=23,head=m['jobs'][job]['head'])
        elif job=='stage1':
            from a3_v14_cache import main
            main(root/'stage1_config.json',out)
        else:
            from a3_ultrasound.stream_training import train
            train(root/'stage2_config.json',out,fold_name='F1',seed=23,head=m['jobs'][job]['head'])
        atomic_json(out/'execution_witness.json',dict(job=job,manifest_sha256=file_digest(root/'manifest.json'),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            gpu=torch.cuda.get_device_name(0),deterministic=True,tf32=False,formal_result=False,outer_opened=False))
    except Exception as exc:
        atomic_json(session/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise


def supervise(root,job):
    m=verify(root)
    if job not in m['jobs']:raise RecoveryRequired('Unregistered job.')
    with FileLease(root/'diagnostic.lock'):
        session=root/'sessions'/job;session.mkdir(parents=True,exist_ok=True)
        out=Path(m['jobs'][job]['output']);receipt_path=session/'receipt.json'
        if receipt_path.exists():
            validate_receipt(read(receipt_path),job_id=job,manifest_sha=file_digest(root/'manifest.json'),expected_root=out)
            print('Verified completed diagnostic; skipped.');return
        if (session/'claim.json').exists() or out.exists():raise RecoveryRequired('Prior incomplete job; explicit recovery required.')
        if (root/'STOP').exists():raise RecoveryRequired('Operator stop.')
        exclusive_json(session/'claim.json',dict(job=job,started=time.time()))
        owner=identity(psutil.Process());atomic_json(session/'owner.json',owner)
        env=dict(os.environ,PYTHONHASHSEED='23',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONIOENCODING='utf-8')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',job]
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        started=time.time();proc=None;watchdog=None;saved=None
        try:
            with (session/'stdout.log').open('wb') as stdout,(session/'stderr.log').open('wb') as stderr:
                proc=subprocess.Popen(cmd+['--mode','child'],stdout=stdout,stderr=stderr,env=env,creationflags=flags)
                saved=identity(psutil.Process(proc.pid));atomic_json(session/'active.json',saved)
                atomic_json(session/'heartbeat.json',dict(epoch=time.time()))
                watchdog=subprocess.Popen(cmd+['--mode','watchdog'],stdout=subprocess.DEVNULL,stderr=stderr,env=env,creationflags=flags)
                atomic_json(session/'watchdog.json',identity(psutil.Process(watchdog.pid)))
                while proc.poll() is None:
                    reason=guard_reason(idle_seconds=activity_age(out,started),elapsed_seconds=time.time()-started,
                        free_bytes=shutil.disk_usage(root).free,stop_requested=(root/'STOP').exists(),idle_limit=1200,
                        wall_limit=(24 if m.get('protocol') in ('V1.4-freeze','V1.5-freeze') else 8)*3600)
                    if reason:
                        terminate_owned(saved);raise RecoveryRequired(reason)
                    atomic_json(session/'heartbeat.json',dict(epoch=time.time(),elapsed=time.time()-started,status='RUNNING',job=job))
                    time.sleep(10)
                if proc.returncode:raise RecoveryRequired('Child exit '+str(proc.returncode))
            required='d2_result.json' if m.get('protocol')=='V1.4-D2' else ('stage1_result.json' if job=='stage1' else 'summary.json')
            result=read(out/required)
            if result.get('status') not in ('COMPLETED','COMPLETED_DEVELOPMENT_DIAGNOSTIC','SCREENED_OUT'):
                raise RecoveryRequired('Diagnostic result incomplete.')
            files=[p for p in out.rglob('*') if p.is_file() and '.pending_' not in p.name]
            value=receipt(out,job,file_digest(root/'manifest.json'),files)
            validate_receipt(value,job_id=job,manifest_sha=file_digest(root/'manifest.json'),expected_root=out,required_files=[required,'execution_witness.json'])
            atomic_json(receipt_path,value)
            atomic_json(session/'status.json',dict(status='COMPLETED',scientific_status=result['status'],next_job_started=False))
        except BaseException as exc:
            stop_spawned(proc,saved)
            atomic_json(session/'status.json',dict(status='NEEDS_ATTENTION',error_type=type(exc).__name__,error=str(exc),automatic_retry=False))
            raise
        finally:
            if proc is None or proc.poll() is not None:
                atomic_json(session/'watchdog_stop.json',dict(epoch=time.time()))
                if watchdog is not None:watchdog.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--mode',choices=['supervise','child','watchdog'],default='supervise');a=p.parse_args()
    if a.mode=='child':child(a.root,a.job)
    elif a.mode=='supervise':supervise(a.root,a.job)
    else:
        s=a.root/'sessions'/a.job
        watchdog_loop(read(s/'owner.json'),s/'active.json',s/'watchdog_stop.json',s/'watchdog_event.json',heartbeat_file=s/'heartbeat.json')
