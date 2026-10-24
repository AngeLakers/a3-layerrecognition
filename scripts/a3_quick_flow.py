"""Two-update, four-head software smoke. No model selection or test scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.training import ModelConfig,TrainingConfig,build_model,build_optimizer

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run(input_dir,output):
    if (input_dir/'RETIRED.json').exists():
        raise ValueError('This input version was retired by the user; use the active replacement.')
    config=json.loads((input_dir/'config.json').read_text(encoding='utf-8'))
    if config.get('status','').startswith('RETIRED'):
        raise ValueError('Retired input configuration cannot start a new run.')
    scope=json.loads((input_dir/'split_scope.json').read_text(encoding='utf-8'))
    smoke=config['smoke'];seed=int(smoke['seed']);device=smoke['device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable.')
    with np.load(input_dir/'smoke_input.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert set(arrays['specimen_ids'])<=set(scope['used_for_learning'])
    assert not set(arrays['specimen_ids'])&set(scope['fold']['test'])
    assert not set(arrays['specimen_ids'])&set(scope['fold']['validation'])
    output.mkdir(parents=True,exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # Small numerical witness; environment reused without installs or rebuild.
    witness=torch.randn(8,8,device=device)
    assert torch.isfinite(witness@witness.T).all()
    save(output/'environment.json',{'python':sys.executable,'torch':str(torch.__version__),
        'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'seeded_matmul_finite':True,'environment_modified':False})
    counts=arrays['interface_mask'].sum(1)
    rng=np.random.default_rng(seed)
    # Exactly the same mixed-K schedules for all four Heads.
    schedule=[]
    for _ in range(smoke['updates_per_head']):
        half=smoke['batch_size']//2
        ids=np.r_[rng.choice(np.flatnonzero(counts==2),half,replace=False),
                  rng.choice(np.flatnonzero(counts==3),smoke['batch_size']-half,replace=False)]
        schedule.append(ids)
    np.save(output/'batch_schedule.npy',np.stack(schedule))
    save(output/'used_sample_ids.json',[[str(x) for x in arrays['sample_ids'][ids]] for ids in schedule])
    names=['x_raw','length_mask','interface_meta','interface_mask','label_mask','y']
    def make_batch(ids):return InterfaceBatch(**{k:torch.from_numpy(arrays[k][ids]).to(device) for k in names})
    reports=[]
    for head in HEAD_TYPES:
        torch.manual_seed(seed);started=time.perf_counter()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        model_options=dict(config.get('model',{}))
        if 'readout' in config:
            readout=config['readout']
            if readout['attention_heads'] != 1:
                raise ValueError('IndexedReadout implements one additive-attention head.')
            model_options['local_prior_sigma_samples']=readout['sigma_us']*1e-6*config['sampling_rate_hz']
            model_options['local_prior_strength']=readout['prior_strength']
        mc=ModelConfig(backbone=smoke['backbone'],head_type=head,**model_options)
        model=build_model(mc).to(device)
        optimizer,_=build_optimizer(model,TrainingConfig(device=device,learning_rate=smoke['learning_rate'],weight_decay=0,seed=seed))
        losses=[];min_support=10**9
        for ids in schedule:
            model.train();b=make_batch(ids);optimizer.zero_grad(set_to_none=True)
            out=model(b,b.y)
            min_support=min(min_support,int(out.local_support.sum(-1)[b.interface_mask].min()))
            loss=head_training_loss(out.conditional,b,head_type=head,lambda_local=.3)
            assert torch.isfinite(loss.total)
            loss.total.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
            optimizer.step();losses.append(float(loss.total.detach()))
        model.eval();b=make_batch(schedule[0]);unlabeled=replace(b,y=None,label_mask=None)
        with torch.inference_mode():
            exact=model.infer_exact(unlabeled)
            assert torch.isfinite(exact.probabilities).all()
            assert torch.equal(exact.path_mask.sum(1),2**b.interface_mask.sum(1))
            assert float(exact.raw_log_normalizer.abs().max())<2e-5
            # Current backbone crops before encoding; padding values must not affect prediction.
            altered=b.x_raw.clone();altered[~b.length_mask]=12345
            second=model.infer_exact(replace(unlabeled,x_raw=altered))
            assert torch.allclose(exact.marginals,second.marginals,atol=1e-6,rtol=1e-6)
        dest=output/head;dest.mkdir()
        save(dest/'model_config.json',asdict(mc))
        torch.save({'state_dict':model.state_dict(),'model_config':asdict(mc),'scope':'software smoke; never reuse as formal training weights'},dest/'checkpoint.pt')
        np.savez_compressed(dest/'flow_predictions.npz',marginals=exact.marginals.cpu().numpy(),
            map_states=exact.map_states.cpu().numpy(),sample_ids=arrays['sample_ids'][schedule[0]])
        record={'head':head,'status':'PASS','updates':len(schedule),'losses':losses,
            'scope':'F1 train-only software smoke; no held-out performance, no model ranking',
            'mixed_K':[2,3],'minimum_valid_local_tokens':min_support,
            'exact_normalization_max_error':float(exact.raw_log_normalizer.abs().max()),
            'padding_invariance_passed':True,'seconds':time.perf_counter()-started,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        save(dest/'result.json',record);reports.append(record);print(json.dumps(record),flush=True)
        del model,optimizer,b,out,loss,exact,second,unlabeled
        if device=='cuda':torch.cuda.empty_cache()
    save(output/'summary.json',{'status':'SOFTWARE_FLOW_PASS','heads':reports,
        'time_window_physical_acceptance':'NOT_ESTABLISHED_BY_THIS_SMOKE',
        'formal_training_started':False,'old_weights_reused':False,
        'normal_waveforms_used_for_time_qa':0,'source_input':str(input_dir)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
