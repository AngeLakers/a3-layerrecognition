"""Input-cache benefit at fixed weights, deterministic A+TF32, full validation only."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,statistics,subprocess,sys,time
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-passes',type=int,choices=(1,2),default=2);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.speed import SpeedOptions,dataset_from_config,configure_backend
    from a3_ultrasound.training import ModelConfig,build_model
    from a3_ultrasound.stream_training import evaluate_stream
    opts=SpeedOptions.for_smoke('final_base');configure_backend(opts);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory))
    cfg=json.loads(a.config.read_text(encoding='utf-8'));assert cfg['freeze_training_permitted'] is False and cfg['outer_evaluation_permitted'] is False
    cfg=dict(cfg,speed=opts.to_dict());data=dataset_from_config(cfg,'validation');assert len(data)==40002
    model=build_model(ModelConfig(**cfg['model'],head_type='proposed_v15')).cuda();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));model.eval()
    a.output.mkdir(parents=True,exist_ok=False)
    # Prime actual validation shapes on a separate uncached dataset, not the measured cache.
    prime_cfg=dict(cfg,speed=replace(opts,workers=0,validation_cache=False).to_dict());prime=dataset_from_config(prime_cfg,'validation')
    with torch.no_grad():
        for start in (0,13312,13440,26624,26752,39936):
            batch=prime.make_batch(list(range(start,min(start+128,len(prime)))),'cuda')
            from a3_ultrasound.speed import speed_context
            with speed_context(opts,batch):model.infer_exact(batch)
    torch.cuda.synchronize();assert len(data._record_cache)==0
    snapshot=lambda:subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    results=[];reference=None
    for phase in ['cold_input_cache']+[f'warm_{i+1}' for i in range(a.warm_passes)]:
        before=len(data._record_cache);status=snapshot();torch.cuda.synchronize();start=time.perf_counter()
        metrics=evaluate_stream(model,data,batch_size=128,device='cuda',speed_options=opts)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if reference is None:reference=metrics
        else:assert metrics==reference,'Cached validation metrics changed'
        assert len(data._record_cache)==40002
        (a.output/(phase+'_metrics.json')).write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        results.append(dict(phase=phase,seconds=seconds,cache_before=before,cache_after=len(data._record_cache),GPU_before=status,GPU_after=snapshot()))
        (a.output/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    cold=results[0]['seconds'];warm=statistics.median(r['seconds'] for r in results[1:])
    result=dict(status='PASS_FIXED_WEIGHT_CACHE_BENCHMARK',records=40002,backend='deterministic A+TF32',results=results,
        cold_seconds=cold,warm_median_seconds=warm,seconds_saved=cold-warm,time_reduction_fraction=1-warm/cold,
        throughput_speedup=cold/warm,all_metrics_identical=True,weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),optimizer_updates=0,formal_training=False,outer_opened=False,
        scope='OS/GPU shapes primed; cold means empty RF/meta/energy cache; warm reuses inputs, not encoder features or logits')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
