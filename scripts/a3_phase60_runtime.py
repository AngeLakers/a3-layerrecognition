"""Frozen single-GPU phase with ownership, durable receipts and separate outer stage."""
from pathlib import Path
import argparse
import ctypes
import datetime
import json
import importlib.metadata
import os
import socket
import shutil
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code/src' if (HERE/'code/src').is_dir() else HERE.parent/'src'))
from a3_ultrasound.runtime_supervisor import (FileLease,RecoveryRequired,atomic_json,exclusive_json,
    file_digest,identity,matching_process,terminate_owned,receipt,validate_receipt,watchdog_loop,activity_age,guard_reason)


def read(p):return json.loads(Path(p).read_text('utf8'))


def verify_files(root,m):
    for rel,digest in m['files'].items():
        if file_digest(root/rel)!=digest:raise RecoveryRequired('Frozen file changed: '+rel)


def gpu():
    result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=name,memory.free,memory.total,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
    fields=[s.strip() for s in result.stdout.strip().split(',')]
    return dict(name=fields[0],free_mib=float(fields[1]),total_mib=float(fields[2]),temperature_c=float(fields[3]),utilization=float(fields[4]))


def all_training_receipts(root,m):
    for job in m['jobs']:
        validate_job_receipt(root,m,job,False)


def validate_job_receipt(root,m,job,outer):
    token=job['id']+('_outer' if outer else '')
    output=Path(job['outer_output'] if outer else job['output'])
    value=read(root/'receipts'/f'{token}.json')
    validate_receipt(value,job_id=token,manifest_sha=file_digest(root/'manifest.json'),expected_root=output,
        required_files=['summary.json','execution_witness.json','joint_state_diagnostics.json',
            'test_predictions.jsonl' if outer else 'predictions.jsonl'])
    required=[p.name for p in completion_files(root,m,job,outer)]
    if not set(required).issubset(value['artifacts']):raise RecoveryRequired('Receipt omits completed job artifacts.')


def run_job(root,m,job,outer,session):
    import psutil
    atomic_json(session/'active.json',identity(psutil.Process()))
    if matching_process(read(session/'owner.json')) is None or (session/'stop.json').exists():
        raise RecoveryRequired('Owner absent before CUDA import.')
    verify_files(root,m)
    if os.environ.get('PYTHONHASHSEED')!=str(job['seed']):raise RecoveryRequired('Interpreter seed not set.')
    import torch
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if '3080 Ti' not in torch.cuda.get_device_name(0):raise RecoveryRequired('Wrong GPU.')
    from a3_ultrasound.runtime_supervisor import install_atomic_outputs
    install_atomic_outputs()
    from a3_ultrasound.stream_training import train,test_checkpoint
    from a3_ultrasound.phase60 import prediction_diagnostics
    output=Path(job['output'])
    if outer:
        all_training_receipts(root,m)
        output=Path(job['outer_output'])
        test_checkpoint(root/'config_outer.json',Path(job['output'])/'best.pt',output)
        prediction=output/'test_predictions.jsonl'
    else:
        train(root/'config_train.json',output,fold_name=job['fold'],head=job['head'],seed=job['seed'])
        prediction=output/'predictions.jsonl'
    diagnostics=prediction_diagnostics(prediction)
    diagnostics['scope']='outer_test' if outer else 'selected_validation_weighted_estimator_not_outer'
    atomic_json(output/'joint_state_diagnostics.json',diagnostics)
    atomic_json(output/'execution_witness.json',dict(protocol_signature=m['protocol_signature'],
        manifest_sha256=file_digest(root/'manifest.json'),label_version='V2',seed=job['seed'],head=job['head'],fold=job['fold'],
        diagnostic=False,outer_opened=outer,old_weights_reused=False,deterministic=True,tf32=False,
        pythonhashseed=os.environ.get('PYTHONHASHSEED'),precision='float32',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def completion_files(root,m,job,outer):
    output=Path(job['outer_output'] if outer else job['output'])
    required=['summary.json','execution_witness.json','joint_state_diagnostics.json',
              'test_predictions.jsonl' if outer else 'predictions.jsonl']
    if not outer:required+=['best.pt','resolved_config.json','history.json','training_selection.npy']
    if any(not (output/name).is_file() for name in required):raise RecoveryRequired('Missing completed job artifacts.')
    summary=read(output/'summary.json');w=read(output/'execution_witness.json')
    if summary['status']!=('EVALUATED' if outer else 'COMPLETED') or w['protocol_signature']!=m['protocol_signature']:
        raise RecoveryRequired('Completion protocol mismatch.')
    if outer:
        if summary.get('protocol_signature')!=m['protocol_signature']:raise RecoveryRequired('Outer protocol mismatch.')
    else:
        resolved=read(output/'resolved_config.json')
        if (resolved.get('protocol_signature')!=m['protocol_signature'] or resolved['seed']!=job['seed']
                or resolved['model']['head_type']!=job['head'] or resolved['fold']['fold']!=job['fold']):
            raise RecoveryRequired('Training identity mismatch.')
        if len(list(output.glob('epoch_*.pt')))!=summary['epochs']:raise RecoveryRequired('Per-epoch checkpoint gap.')
    return [p for p in output.iterdir() if p.is_file() and p.name not in ('progress.json','failure.json') and '.pending_' not in p.name]


def snapshot(root,m):
    state=read(root/'queue_state.json') if (root/'queue_state.json').exists() else dict(status='NOT_STARTED')
    entries={}
    for path in (root/'receipts').glob('*.json'):
        r=read(path)
        for name,info in r['artifacts'].items():
            source=Path(r['root'])/name;rel=source.relative_to(Path(m['output_root'])).as_posix()
            entries[rel]=dict(path=str(source),relative=rel,**info)
    for job in m['jobs']:
        checkpoints=sorted(Path(job['output']).glob('epoch_*.pt'))
        if checkpoints:
            source=checkpoints[-1];rel=source.relative_to(Path(m['output_root'])).as_posix()
            if rel not in entries:entries[rel]=dict(path=str(source),relative=rel,bytes=source.stat().st_size,sha256=file_digest(source))
    for name in ('queue_state.json','launch_gate.json','preflight.json','final_oof.json'):
        p=root/name
        if p.exists():
            rel='_operations/'+name
            entries[rel]=dict(path=str(p),relative=rel,bytes=p.stat().st_size,sha256=file_digest(p),mutable=True)
    print(json.dumps(dict(state=state,backup_entries=list(entries.values())),ensure_ascii=True),flush=True)


def preflight(root,m):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    import psutil
    from dataclasses import asdict
    from a3_ultrasound.stream_training import catalog_signature,validate_fold
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.training import ModelConfig,TrainingConfig,build_model,build_optimizer,evaluate,TensorDataset
    from a3_ultrasound.models.losses import head_training_loss
    from a3_ultrasound.phase60 import phase_cells,prediction_diagnostics
    verify_files(root,m);cfg=read(root/'config_train.json');catalog=read(cfg['catalog']);phase_cells(cfg)
    expected=read(root/'environment/expected.json')
    versions={name:importlib.metadata.version(name) for name in expected['versions']}
    if socket.gethostname().lower()!=expected['hostname'].lower() or sys.version.split()[0]!=expected['python'] or versions!=expected['versions']:
        raise RecoveryRequired('Warm-reuse environment differs from the recorded specification.')
    if catalog_signature(catalog)!=m['catalog_signature']:raise RecoveryRequired('Catalog signature drift.')
    checked=0
    for fold in cfg['folds']:
        validate_fold(fold,catalog)
        for seed in cfg['seeds']:
            d=RFShardDataset(cfg['catalog'],fold['train'],training=True,seed=seed,normal_ratio=cfg['normal_to_positive'])
            if not np.array_equal(d.refs,np.load(Path(cfg['frozen_selection_root'])/f"{fold['fold']}_s{seed}.npy")):
                raise RecoveryRequired('Training refs changed.')
            checked+=1
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    before=gpu()
    if '3080 Ti' not in before['name']:raise RecoveryRequired('Wrong GPU')
    d=RFShardDataset(cfg['catalog'],cfg['folds'][1]['train'],training=True,seed=23,normal_ratio=cfg['normal_to_positive'])
    lengths=np.concatenate([s['arrays']['lengths'][d.refs[d.refs[:,0]==i,1]] for i,s in enumerate(d.shards)])
    ids=np.argsort(lengths,kind='stable')[-16:];batch=d.make_batch(ids,'cuda');witness=[]
    eval_ids=np.argsort(lengths,kind='stable')[-cfg['evaluation_batch_size']:]
    evaluation_batch=d.make_batch(eval_ids,'cuda')
    for head in cfg['heads']:
        states=[];peak=0;reserved=0
        for repetition in range(2):
            torch.manual_seed(23);torch.cuda.manual_seed_all(23)
            model=build_model(ModelConfig(**dict(cfg['model'],head_type=head))).cuda()
            opt,_=build_optimizer(model,TrainingConfig(**cfg['training']))
            torch.cuda.reset_peak_memory_stats();model.train();opt.zero_grad(set_to_none=True)
            value=model(batch,batch.y)
            loss=head_training_loss(value.conditional,batch,head_type=head,lambda_local=cfg['training']['lambda_local'])
            loss.total.backward();opt.step()
            if not torch.isfinite(loss.total):raise RecoveryRequired('Nonfinite probe loss.')
            states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
            if repetition==1:
                names=[d.identity(i) for i in eval_ids]
                _,rows=evaluate(model,TensorDataset(evaluation_batch,tuple(v[0] for v in names),tuple(v[1] for v in names)),batch_size=cfg['evaluation_batch_size'],device='cuda')
                probe=root/f'preflight_predictions_{head}.jsonl'
                probe.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),'utf8')
                prediction_diagnostics(probe)
                peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
                reserved=max(reserved,torch.cuda.max_memory_reserved()/2**20)
            del model,opt,value,loss;torch.cuda.empty_cache()
        if any(not torch.equal(states[0][k],states[1][k]) for k in states[0]):raise RecoveryRequired('Seeded model/BN witness differs.')
        witness.append(dict(head=head,seeded_one_update_model_BN_equal=True,training_records=16,evaluation_records=len(eval_ids),
            peak_allocated_mib=peak,peak_reserved_mib=reserved,unusable_for_performance_or_pretraining=True))
    result=dict(status='PASS',manifest_sha256=file_digest(root/'manifest.json'),catalog_signature=m['catalog_signature'],
        train_selections_checked=checked,torch=torch.__version__,python=sys.version.split()[0],gpu_before=before,
        hostname=socket.gethostname(),versions=versions,environment_ledger_sha256=file_digest(root/'environment/ENVIRONMENT_LEDGER.md'),
        environment_spec_sha256=file_digest(root/'environment/env-spec.environment-only.json'),
        rf_training_lengths=lengths[ids].tolist(),rf_evaluation_lengths=lengths[eval_ids].tolist(),
        evaluation_entity_records=[dict(sample_id=d.identity(i)[0],entity=d.identity(i)[1]) for i in eval_ids],
        host_available_gib=psutil.virtual_memory().available/1024**3,
        probes=witness,outer_loaded=False,probe_weights_not_saved_or_reused=True)
    atomic_json(root/'preflight.json',result);print(json.dumps(result,ensure_ascii=True),flush=True)


def supervise(root,m):
    import psutil
    digest=file_digest(root/'manifest.json');gate=read(root/'launch_gate.json');pf=read(root/'preflight.json')
    if (gate.get('status')!='PASS' or gate.get('manifest_sha256')!=digest or not gate.get('external_review_adopted')
        or not gate.get('independent_codex_adopted') or not gate.get('agent_follows_doc_passed')
        or pf.get('status')!='PASS' or pf.get('manifest_sha256')!=digest):raise RecoveryRequired('Launch gate incomplete.')
    verify_files(root,m)
    from a3_ultrasound.phase60 import phase_cells
    if [(j['fold'],j['seed'],j['head']) for j in m['jobs']]!=phase_cells(read(root/'config_train.json')):
        raise RecoveryRequired('Wrong ordered phase cells.')
    old=read(m['original_queue'])
    if old['status'] not in ('STOPPED_AT_BOUNDARY','COMPLETED') or any(j['status']=='RUNNING' for j in old['jobs']):
        raise RecoveryRequired('Old queue not stopped.')
    with FileLease(root/'queue.lock'):
        session=root/'sessions'/uuid.uuid4().hex;session.mkdir(parents=True)
        atomic_json(session/'owner.json',identity(psutil.Process()));atomic_json(session/'owner_heartbeat.json',dict(epoch=time.time()))
        env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        guard=subprocess.Popen([m['python'],'-X','utf8',str(root/'runtime.py'),'--watchdog',str(session)],env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        atomic_json(session/'watchdog.json',identity(psutil.Process(guard.pid)))
        api=ctypes.windll.kernel32.SetThreadExecutionState;api.argtypes=[ctypes.c_uint];api.restype=ctypes.c_uint
        state=dict(status='RUNNING',manifest_sha256=digest,protocol_signature=m['protocol_signature'],label_version='V2',
                   started_epoch=time.time(),session=str(session),pid=os.getpid(),watchdog_pid=guard.pid,jobs=[],active_job=None)
        active=None
        try:
            if not api(0x80000001):raise ctypes.WinError()
            (root/'receipts').mkdir(exist_ok=True);(root/'claims').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
            stages=(False,True) if m.get('outer_after_all_training') else (False,)
            for outer in stages:
                if outer:all_training_receipts(root,m)
                for job in m['jobs']:
                    token=job['id']+('_outer' if outer else '')
                    done=root/'receipts'/f'{token}.json';claim=root/'claims'/f'{token}.json'
                    if done.exists():
                        validate_job_receipt(root,m,job,outer)
                        state['jobs'].append(dict(id=token,status='COMPLETED',recovered='VERIFIED_SKIP'));continue
                    output=Path(job['outer_output'] if outer else job['output'])
                    if claim.exists() or output.exists():raise RecoveryRequired('Interrupted output preserved; no automatic rerun: '+token)
                    if (root/'STOP').exists() or (root/'STOP_AFTER_CURRENT').exists():raise RecoveryRequired('OPERATOR_STOP')
                    while gpu()['free_mib']<m['minimum_gpu_free_mib']:
                        atomic_json(session/'owner_heartbeat.json',dict(epoch=time.time()))
                        state.update(status='WAITING_GPU',heartbeat_epoch=time.time());atomic_json(root/'queue_state.json',state)
                        if (root/'STOP').exists():raise RecoveryRequired('OPERATOR_STOP')
                        time.sleep(15)
                    exclusive_json(claim,dict(job=token,manifest_sha256=digest,started_epoch=time.time()))
                    started=time.time();env['PYTHONHASHSEED']=str(job['seed'])
                    command=[m['python'],'-X','utf8',str(root/'runtime.py'),'--job',job['id'],'--owner-session',str(session)]
                    if outer:command+=['--outer']
                    with (root/'logs'/f'{token}.log').open('x',encoding='utf8') as log:
                        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                    active=identity(psutil.Process(proc.pid));atomic_json(session/'active.json',active)
                    entry=dict(id=token,status='RUNNING',fold=job['fold'],seed=job['seed'],head=job['head'],outer=outer,started_epoch=started,pid=proc.pid)
                    state['jobs'].append(entry);state.update(status='RUNNING',active_job=token)
                    while proc.poll() is None:
                        atomic_json(session/'owner_heartbeat.json',dict(epoch=time.time()))
                        device=gpu();state['gpu']=device;memory=psutil.virtual_memory()
                        reason=guard_reason(idle_seconds=activity_age(output,started),elapsed_seconds=time.time()-started,
                            free_bytes=shutil.disk_usage(root).free,stop_requested=(root/'STOP').exists(),wall_limit=12*3600)
                        if guard.poll() is not None:reason='WATCHDOG_DIED'
                        if memory.available<2*1024**3:reason='LOW_HOST_MEMORY'
                        if device['temperature_c']>=86:reason='GPU_TEMPERATURE_GUARD'
                        if device['free_mib']<512:reason='GPU_MEMORY_PRESSURE'
                        if reason:terminate_owned(active);raise RecoveryRequired(reason)
                        if (output/'progress.json').exists():entry['progress']=read(output/'progress.json')
                        state.update(heartbeat_epoch=time.time(),host_available_gib=memory.available/1024**3)
                        atomic_json(root/'queue_state.json',state);time.sleep(10)
                    if proc.returncode:raise RecoveryRequired('Child failed with exit '+str(proc.returncode)+': '+token)
                    files=completion_files(root,m,job,outer)
                    atomic_json(done,receipt(output,token,digest,files));entry.update(status='COMPLETED',ended_epoch=time.time())
                    active=None;atomic_json(session/'active.json',{});atomic_json(root/'queue_state.json',state)
            if m.get('outer_after_all_training'):
                from a3_ultrasound.oof_report import summarize_oof
                results=[read(Path(j['outer_output'])/'summary.json') for j in m['jobs']]
                atomic_json(root/'final_oof.json',summarize_oof(read(root/'config_outer.json'),results,expected_catalog_signature=m['catalog_signature']))
            state['status']='COMPLETED'
        except Exception as exc:
            if active:terminate_owned(active)
            state.update(status='NEEDS_ATTENTION',error_type=type(exc).__name__,error=str(exc))
            if state['jobs'] and state['jobs'][-1]['status']=='RUNNING':state['jobs'][-1]['status']='INTERRUPTED'
        finally:
            atomic_json(session/'stop.json',dict(epoch=time.time()));guard.wait(timeout=20);api(0x80000000)
            state.update(active_job=None,heartbeat_epoch=time.time(),ended_epoch=time.time());atomic_json(root/'queue_state.json',state)
        if state['status']!='COMPLETED':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--snapshot',action='store_true')
    p.add_argument('--watchdog',type=Path);p.add_argument('--job');p.add_argument('--outer',action='store_true');p.add_argument('--owner-session',type=Path)
    args=p.parse_args();m=read(HERE/'manifest.json')
    if args.watchdog:
        s=args.watchdog;watchdog_loop(read(s/'owner.json'),s/'active.json',s/'stop.json',s/'watchdog_event.json',interval=5,heartbeat_file=s/'owner_heartbeat.json');return
    if args.snapshot:snapshot(HERE,m)
    elif args.preflight:preflight(HERE,m)
    elif args.job:run_job(HERE,m,next(j for j in m['jobs'] if j['id']==args.job),args.outer,args.owner_session)
    else:supervise(HERE,m)


if __name__=='__main__':main()
