"""Local mirror of immutable checkpoints plus half-hour script-only records."""
from pathlib import Path
import argparse
import ctypes
import datetime
import json
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.runtime_supervisor import FileLease,atomic_json,file_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);p.add_argument('--remote-root',required=True)
    args=p.parse_args();root=args.local_root;backup=root/'mirrored_results';ledger_file=root/'backup_ledger.json'
    ledger=json.loads(ledger_file.read_text('utf8')) if ledger_file.exists() else {}
    api=ctypes.windll.kernel32.SetThreadExecutionState;api.argtypes=[ctypes.c_uint];api.restype=ctypes.c_uint
    with FileLease(root/'watch.lock'):
        atomic_json(root/'watch_launch.json',dict(pid=__import__('os').getpid(),started=time.time(),remote_root=args.remote_root))
        last_record=0
        try:
            api(0x80000001)
            while True:
                try:
                    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','gpu3080',
                        'E:/programming/anaconda/envs/aislee-a3-layercongition/python.exe','-X','utf8',args.remote_root+'/runtime.py','--snapshot']
                    result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
                    if result.returncode:raise RuntimeError('Remote snapshot failed.')
                    snapshot=json.loads(result.stdout);atomic_json(root/'batch_snapshot.json',snapshot)
                    for item in snapshot['backup_entries']:
                        dest=backup/item['relative'];dest.parent.mkdir(parents=True,exist_ok=True)
                        if ledger.get(item['relative'])==item['sha256'] and dest.exists():continue
                        tmp=dest.with_name(dest.name+'.downloading')
                        pull=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                            'gpu3080:'+item['path'],str(tmp)],capture_output=True,timeout=300,creationflags=subprocess.CREATE_NO_WINDOW)
                        if pull.returncode or file_digest(tmp)!=item['sha256']:raise RuntimeError('Mirror hash mismatch; retry snapshot.')
                        if dest.exists() and not item.get('mutable') and file_digest(dest)!=item['sha256']:
                            raise RuntimeError('Immutable mirror changed: '+item['relative'])
                        tmp.replace(dest);ledger[item['relative']]=item['sha256'];atomic_json(ledger_file,ledger)
                    state=snapshot['state'];status=dict(status='MONITORING',remote_status=state['status'],checked_epoch=time.time(),
                        mirrored_files=len(ledger),active_job=state.get('active_job'),model_calls=0)
                    atomic_json(root/'watch_state.json',status)
                    if time.time()-last_record>=1800:
                        last_record=time.time();record=dict(local_time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                            **status,completed=sum(j['status']=='COMPLETED' for j in state.get('jobs',[])),
                            action='verified remote snapshot and mirrored committed checkpoint artifacts')
                        with (root/'heartbeat_records.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if state['status'] in ('COMPLETED','NEEDS_ATTENTION'):
                        atomic_json(root/'watch_state.json',dict(status='COLLECTED',remote_status=state['status'],checked_epoch=time.time(),mirrored_files=len(ledger),model_calls=0));return
                except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
                    atomic_json(root/'watch_state.json',dict(status='RETRYING',checked_epoch=time.time(),error=str(exc),mirrored_files=len(ledger)))
                time.sleep(60)
        finally:api(0x80000000)


if __name__=='__main__':main()
