"""Bounded F1/23/proposed_v15 timing and numeric smoke; never call formal train."""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,random,subprocess,sys,time
import numpy as np
import torch

def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--steps',type=int,choices=(2,200),default=200)
    p.add_argument('--timing-warmup-steps',type=int,default=0)
    p.add_argument('--matched-batch16',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(a.source/'src'))
    if not 0<=a.timing_warmup_steps<a.steps:raise ValueError('Timing warmup must be inside the existing bounded steps')
    from a3_ultrasound.training import ModelConfig,TrainingConfig,build_model,build_optimizer
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.epoch_sampling import use_epoch
    from a3_ultrasound.stream_training import evaluate_stream,catalog_signature
    from a3_ultrasound.v15_protocol import load_physical_artifacts,load_timebase_artifact
    from a3_ultrasound.prefix_perturbation import training_conditional
    from a3_ultrasound.models.losses import head_training_loss
    legacy=a.mode=='v152';speed=None
    if not legacy:
        from a3_ultrasound.speed import SpeedOptions,speed_context,precision_context,configure_backend,configure_model,finish_step,dataset_from_config
        speed=SpeedOptions.for_smoke(a.mode)
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    assert cfg['freeze_training_permitted'] is False and cfg['outer_evaluation_permitted'] is False
    assert cfg['folds'][0]['fold']=='F1' and cfg['training']['prefix_replacement_probability']==.2
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(23);np.random.seed(23);torch.manual_seed(23);torch.cuda.manual_seed_all(23);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory))
    if speed:configure_backend(speed)
    physical,_=load_physical_artifacts(cfg);table=load_timebase_artifact(cfg)
    if legacy:
        train=RFShardDataset(cfg['catalog'],cfg['folds'][0]['train'],training=True,seed=23,normal_ratio=3,physical_config=physical,tail_policy=cfg['input_tail_policy'],window_policy=cfg['input_window_policy'],timebase_config=table,window_jitter=cfg['window_jitter'])
        val=RFShardDataset(cfg['catalog'],cfg['folds'][0]['validation'],physical_config=physical,tail_policy=cfg['input_tail_policy'],window_policy=cfg['input_window_policy'],timebase_config=table)
        val.use_frozen_evaluation_selection(Path(cfg['validation_selection_root'])/'F1.npz')
    else:
        cfg=dict(cfg,speed=speed.to_dict());train=dataset_from_config(cfg,'train');val=dataset_from_config(cfg,'validation')
    use_epoch(train,cfg['epoch_selection_root'],1,seed=23,catalog_signature=catalog_signature(train.catalog))
    assert len(train)==100732 and len(val)==40002
    batchsize=16 if speed is None or speed.batch_size is None else speed.batch_size
    assert a.steps*batchsize<len(train)
    if a.matched_batch16:
        from a3_ultrasound.data.length_batches import matched_smoke_indices
        assert batchsize==16 and speed is not None and speed.tf32 and not speed.fused_adam and not speed.bf16_autocast
        lengths=np.asarray([train._input_length(i) for i in range(len(train))],dtype=np.int64)
        chosen=matched_smoke_indices(lengths,16,a.steps,seed=23,epoch=1)
        population=train.refs[chosen].copy()
        train.refs=population
        np.save(a.output/'smoke_refs.npy',population)
    else:population=train.refs[:a.steps*batchsize].copy()
    population_sha=hashlib.sha256(population[np.lexsort((population[:,1],population[:,0]))].tobytes()).hexdigest()
    if speed and speed.length_buckets and not a.matched_batch16:
        # Same12800-record smoke population as batch64; do not time only the globally shortest boards.
        train.refs=population
    model=build_model(ModelConfig(**cfg['model'],head_type='proposed_v15')).cuda()
    if speed:configure_model(model,speed)
    torch.save(model.state_dict(),a.output/'initial_disposable.pt')
    if a.matched_batch16:
        plan=train.batch_plan(batchsize)
        assert len(plan)==a.steps and all(len(ids)==16 for ids in plan)
        assert np.array_equal(np.sort(np.concatenate(plan)),np.arange(len(train)))
        if speed.length_buckets:assert all(len({train._input_length(int(i)) for i in ids})==1 for ids in plan)
    tc=TrainingConfig(**cfg['training'])
    optimizer,_=build_optimizer(model,tc,**({'fused':speed.fused_adam} if speed else {}))
    before_gpu=gpu();losses=[];ids_hash=hashlib.sha256();torch.cuda.synchronize();start=time.perf_counter();model.train()
    iterator=train.batches(batchsize,'cuda');warm_stamp=None
    try:
        for step in range(a.steps):
            batch,ids=next(iterator);ids_hash.update(np.asarray(train.refs[ids],np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            scope=speed_context(speed,batch,training=True) if speed else contextlib.nullcontext()
            with scope:
                with precision_context(speed) if speed else contextlib.nullcontext():
                    out,attempt=training_conditional(model,batch,probability=.2,seed=23,epoch=1,step=step)
                with torch.autocast('cuda',enabled=False):
                    loss=head_training_loss(out,batch,head_type='proposed_v15',lambda_local=.3,local_aux_mode='clean_upstream')
                if speed and speed.coalesced_finite:
                    loss.total.backward();finish_step(model,loss.total)
                else:
                    if not torch.isfinite(loss.total):raise RuntimeError('Nonfinite loss')
                    loss.total.backward()
                    if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise RuntimeError('Nonfinite gradient')
            optimizer.step();losses.append(loss.total.detach())
            if a.timing_warmup_steps and step+1==a.timing_warmup_steps:
                torch.cuda.synchronize();warm_stamp=time.perf_counter()
            if (step+1)%25==0:save(a.output/'progress.json',dict(phase='smoke_steps',completed=step+1,total=a.steps))
        torch.cuda.synchronize();train_end=time.perf_counter();train_s=train_end-start
    finally:
        if hasattr(iterator,'close'):iterator.close()
    loss_values=torch.stack(losses).cpu().numpy();np.save(a.output/'losses.npy',loss_values)
    torch.save(model.state_dict(),a.output/'after_200_disposable.pt')
    save(a.output/'training_timing.json',dict(steps=a.steps,seconds=train_s,seconds_per_step=train_s/a.steps))
    if a.steps==2:
        save(a.output/'result.json',dict(status='PASS_TWO_STEP_PREFLIGHT_ONLY',steps=2,mode=a.mode,formal_training=False,outer_opened=False))
        print('PASS_TWO_STEP_PREFLIGHT_ONLY',flush=True);return
    alloff=None
    training_diff=None
    if a.baseline and batchsize==16 and not (speed and speed.length_buckets) and not a.matched_batch16:
        reference_loss=np.load(a.baseline/'losses.npy')
        training_diff=dict(loss_max_abs=float(abs(loss_values-reference_loss).max()),loss_bitwise=bool(np.array_equal(loss_values,reference_loss)),different_steps=int(np.count_nonzero(loss_values!=reference_loss)))
        ref_weights=torch.load(a.baseline/'after_200_disposable.pt',map_location='cpu',weights_only=True)
        state={k:v.cpu() for k,v in model.state_dict().items()}
        training_diff.update(final_weights_bitwise=all(torch.equal(v,ref_weights[k]) for k,v in state.items()),
            final_weight_max_abs=max(float((v-ref_weights[k]).abs().max()) for k,v in state.items() if v.is_floating_point()))
    if a.mode=='off' and a.baseline:
        expected=np.load(a.baseline/'losses.npy');assert np.array_equal(loss_values,expected),'Default-off losses changed'
        ref=torch.load(a.baseline/'after_200_disposable.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v.cpu(),ref[k]) for k,v in model.state_dict().items()),'Default-off weights changed'
        alloff=dict(loss_200_bitwise=True,final_weights_bitwise=True)
    save(a.output/'progress.json',dict(phase='full_validation',records=len(val)))
    torch.cuda.synchronize();start=time.perf_counter()
    if speed:
        metrics=evaluate_stream(model,val,batch_size=cfg['evaluation_batch_size'],device='cuda',speed_options=speed)
    else:metrics=evaluate_stream(model,val,batch_size=cfg['evaluation_batch_size'],device='cuda')
    torch.cuda.synchronize();validation_s=time.perf_counter()-start;save(a.output/'validation_metrics.json',metrics)
    save(a.output/'validation_timing.json',dict(records=len(val),seconds=validation_s))
    # Numerics always use EXACT same original FP32200-step weights, not each variant's final training weights.
    if a.baseline:model.load_state_dict(torch.load(a.baseline/'after_200_disposable.pt',map_location='cpu',weights_only=True))
    subset=np.linspace(0,len(val)-1,2000,dtype=np.int64);logits=[];maps=[];masks=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(subset),128):
            batch=val.make_batch(subset[start:start+128],'cuda')
            with speed_context(speed,batch,training=False) if speed else contextlib.nullcontext():
                with precision_context(speed) if speed else contextlib.nullcontext():encoded=model.encode(batch)
                # Enumeration and diagnostic conditional logits are explicitly FP32.
                with torch.autocast('cuda',enabled=False):
                    exact=model.infer_encoded(encoded)
                    # Identical label-independent path in all precision conditions; unknown labels stay unknown.
                    out=model.conditional(encoded,torch.zeros_like(batch.interface_mask,dtype=torch.long))
                    assert out.logits.dtype==exact.log_probabilities.dtype==torch.float32
            logits.append(out.logits.float().cpu().numpy());maps.append(exact.map_states.cpu().numpy());masks.append(batch.interface_mask.cpu().numpy())
    numeric=dict(logits=np.concatenate(logits),MAP=np.concatenate(maps),mask=np.concatenate(masks),indices=subset)
    np.savez_compressed(a.output/'numerical_2000.npz',**numeric)
    diff=None
    if a.baseline:
        with np.load(a.baseline/'numerical_2000.npz') as ref:
            assert np.array_equal(ref['indices'],subset) and np.array_equal(ref['mask'],numeric['mask'])
            diff=dict(logit_max_abs=float(abs(numeric['logits']-ref['logits'])[numeric['mask']].max()),MAP_row_disagreement=float(np.any(numeric['MAP']!=ref['MAP'],axis=1).mean()))
            if a.mode=='off':assert diff['logit_max_abs']==0 and diff['MAP_row_disagreement']==0
    result=dict(status='PASS_BOUNDED_SMOKE',mode=a.mode,steps=a.steps,head='proposed_v15',fold='F1',seed=23,batch_size=batchsize,training_samples=a.steps*batchsize,training_seconds=train_s,seconds_per_step=train_s/a.steps,samples_per_second=a.steps*batchsize/train_s,validation_records=len(val),validation_seconds=validation_s,validation_cache='cold_first_pass_included' if speed and speed.validation_cache else 'disabled',GPU_before=before_gpu,GPU_after=gpu(),numerical_difference_vs_original_FP32=diff,default_off_regression=alloff,training_order_sha256=ids_hash.hexdigest(),speed=None if speed is None else speed.to_dict(),formal_training=False,complete_epoch=False,outer_opened=False,weights_scope='Disposable speed-smoke only; never resume formal training')
    result.update(training_loss_comparison=training_diff,training_population_sha256=population_sha,
        matched_batch16_population=a.matched_batch16,
        timing_warmup_steps=a.timing_warmup_steps,
        steady_seconds_per_step=(train_end-warm_stamp)/(a.steps-a.timing_warmup_steps) if warm_stamp is not None else None,
        numerical_logit_path='fixed all-zero prefix, independent of labels',numerical_weights='original FP32 baseline after200 steps',
        classification=('numerical_opt_in' if speed and (speed.fused_adam or speed.tf32 or speed.bf16_autocast) else
            'training_strategy_opt_in' if speed and (speed.batch_size or speed.length_buckets) else 'bitwise_candidate'),
        backend_state=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    save(a.output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
