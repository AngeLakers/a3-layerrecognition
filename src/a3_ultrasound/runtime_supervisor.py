"""Durable artifacts and fail-closed job ownership for the Windows experiment queue."""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid

import psutil


class RecoveryRequired(RuntimeError):
    pass


def all_finite(tensors):
    """One host synchronization for finite checks; no tensor values are changed."""
    import torch
    values=[t.detach() for t in tensors if t is not None and t.is_floating_point()]
    if not values:return True
    if len({(t.device,t.dtype) for t in values})==1:
        norms=torch._foreach_norm(values,float('inf'))
        return bool(torch.isfinite(torch.stack(norms)).all())
    return bool(torch.stack([torch.isfinite(t).all().cpu() for t in values]).all())


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer, validator=None):
    """Keep the last complete target until the replacement has flushed and validated."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
    with pending.open('xb') as handle:
        writer(handle);handle.flush();os.fsync(handle.fileno())
    if validator is not None:validator(pending)
    # Readers can briefly hold the destination open on Windows.
    for attempt in range(51):
        try:os.replace(pending,path);break
        except PermissionError:
            if attempt==50:raise
            time.sleep(.1)
    return path


def atomic_json(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    return atomic_write(path,lambda handle:handle.write(data),lambda p:json.loads(p.read_text(encoding='utf-8')))


def exclusive_json(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with Path(path).open('xb') as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())


class FileLease:
    """OS-held lock: a crashed owner releases it without deleting any evidence."""
    def __init__(self,path):self.path=Path(path);self.handle=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.handle=self.path.open('a+b')
        self.handle.seek(0,os.SEEK_END)
        if self.handle.tell()==0:self.handle.write(b'0');self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.handle.close();self.handle=None
            raise RecoveryRequired('Another supervisor already owns this batch.')
        return self
    def __exit__(self,*_):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
            self.handle.close();self.handle=None


def identity(process):
    return dict(pid=process.pid,created=process.create_time(),exe=process.exe(),cmdline=process.cmdline())


def matching_process(saved):
    try:
        p=psutil.Process(saved['pid'])
        if (abs(p.create_time()-saved['created'])>.001
                or os.path.normcase(p.exe())!=os.path.normcase(saved['exe'])
                or p.cmdline()!=saved['cmdline']):return None
        return p if p.is_running() else None
    except (psutil.NoSuchProcess,psutil.AccessDenied):return None


def terminate_owned(saved):
    p=matching_process(saved)
    if p is None:return False
    p.terminate()
    try:p.wait(timeout=20)
    except psutil.TimeoutExpired:
        p=matching_process(saved)
        if p is not None:p.kill();p.wait(timeout=10)
    return True


def receipt(root,job_id,manifest_sha,files):
    root=Path(root)
    artifacts={str(Path(p).relative_to(root)):dict(bytes=Path(p).stat().st_size,sha256=file_digest(p)) for p in files}
    if not artifacts:raise RecoveryRequired('No completed artifacts to acknowledge.')
    return dict(status='COMPLETED',job=job_id,manifest_sha256=manifest_sha,root=str(root),
                committed_epoch=time.time(),artifacts=artifacts)


def validate_receipt(value,*,job_id,manifest_sha,expected_root=None,required_files=()):
    if value.get('status')!='COMPLETED' or value.get('job')!=job_id or value.get('manifest_sha256')!=manifest_sha:
        raise RecoveryRequired('Completion receipt does not match the frozen job.')
    root=Path(value['root']).resolve()
    if expected_root is not None and root!=Path(expected_root).resolve():
        raise RecoveryRequired('Completion receipt output root mismatch.')
    if not value.get('artifacts') or not set(required_files).issubset(value['artifacts']):
        raise RecoveryRequired('Completion receipt has no artifacts or omits required artifacts.')
    for name,expected in value['artifacts'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise RecoveryRequired('Completed artifact missing: '+name)
        if path.stat().st_size!=expected['bytes'] or file_digest(path)!=expected['sha256']:
            raise RecoveryRequired('Completed artifact changed: '+name)
    return True


def recovery_action(*,completed_receipt=None,claim_exists=False,output_exists=False,live_process=None):
    if completed_receipt is not None:return 'VERIFY_AND_SKIP'
    if live_process is not None:return 'ADOPT_RUNNING'
    if claim_exists or output_exists:return 'RECOVERY_REQUIRED'
    return 'START_NEW'


def activity_age(output,started,now=None):
    newest=started;output=Path(output)
    if output.exists():
        for path in output.rglob('*'):
            if path.is_file():newest=max(newest,path.stat().st_mtime)
    return (time.time() if now is None else now)-newest


def guard_reason(*,idle_seconds,elapsed_seconds,free_bytes,stop_requested=False,
                 idle_limit=600,wall_limit=8*3600,min_free_bytes=10*1024**3):
    if stop_requested:return 'OPERATOR_STOP'
    if free_bytes<min_free_bytes:return 'LOW_DISK_SPACE'
    if idle_seconds>idle_limit:return 'NO_PROGRESS_TIMEOUT'
    if elapsed_seconds>wall_limit:return 'JOB_WALL_TIMEOUT'
    return None


def install_atomic_outputs():
    """Only serialization changes; models, inputs, optimizer and update order stay intact."""
    import numpy as np
    import torch
    import a3_ultrasound.stream_training as streaming
    original_torch_save=torch.save
    original_save=np.save
    original_savez=np.savez_compressed
    def torch_save(value,path,*args,**kwargs):
        if hasattr(path,'write'):return original_torch_save(value,path,*args,**kwargs)
        def validate(p):
            saved=torch.load(p,map_location='cpu',weights_only=True)
            if isinstance(value,dict):
                for key in ('epoch','updates_total'):
                    if key in value and saved[key]!=value[key]:raise RecoveryRequired('Checkpoint readback metadata mismatch.')
        return atomic_write(path,lambda f:original_torch_save(value,f,*args,**kwargs),validate)
    def save_array(path,value,*args,**kwargs):
        if hasattr(path,'write'):return original_save(path,value,*args,**kwargs)
        return atomic_write(path,lambda f:original_save(f,value,*args,**kwargs),
                            lambda p:np.load(p,allow_pickle=False))
    def save_archive(path,*args,**kwargs):
        if hasattr(path,'write'):return original_savez(path,*args,**kwargs)
        def validate(p):
            with np.load(p,allow_pickle=False) as z:
                for key in z.files:z[key]
        return atomic_write(path,lambda f:original_savez(f,*args,**kwargs),validate)
    torch.save=torch_save;np.save=save_array;np.savez_compressed=save_archive
    streaming.save=atomic_json
    def progress(path,**values):
        if path is not None:atomic_json(path,dict(heartbeat_epoch=time.time(),**values))
    streaming.progress=progress
    return dict(torch=original_torch_save,numpy_save=original_save,numpy_savez=original_savez)


def watchdog_loop(owner,active_file,stop_file,event_file,*,interval=10,heartbeat_file=None,heartbeat_limit=120):
    """A separate CPU-only process stops this session's child if its owner dies."""
    while not Path(stop_file).exists():
        reason='SUPERVISOR_DIED' if matching_process(owner) is None else None
        if heartbeat_file and Path(heartbeat_file).exists():
            if time.time()-Path(heartbeat_file).stat().st_mtime>heartbeat_limit:reason='SUPERVISOR_HEARTBEAT_TIMEOUT'
        if reason:
            active=json.loads(Path(active_file).read_text(encoding='utf-8')) if Path(active_file).exists() else None
            stopped=terminate_owned(active) if active else False
            atomic_json(event_file,dict(reason=reason,child_stopped=stopped,epoch=time.time()))
            return
        time.sleep(interval)
