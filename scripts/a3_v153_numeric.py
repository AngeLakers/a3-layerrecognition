"""Same weights, same RF, every binary prefix; validation only, no optimizer."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--mode',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from a3_ultrasound.speed import SpeedOptions,dataset_from_config,configure_backend,configure_model,speed_context,precision_context
    from a3_ultrasound.training import ModelConfig,build_model
    cfg=json.loads(a.config.read_text(encoding='utf-8'));assert cfg['outer_evaluation_permitted'] is False and cfg['freeze_training_permitted'] is False
    opts=SpeedOptions.for_smoke(a.mode);cfg=dict(cfg,speed=opts.to_dict())
    configure_backend(opts);torch.set_num_threads(2);torch.manual_seed(23);torch.cuda.manual_seed_all(23)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory))
    data=dataset_from_config(cfg,'validation');assert len(data)==40002
    model=build_model(ModelConfig(**cfg['model'],head_type='proposed_v15')).cuda();configure_model(model,opts)
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));model.eval()
    indices=np.linspace(0,len(data)-1,2000,dtype=np.int64);logits=[];masks=[];maps=[]
    with torch.no_grad():
        for start in range(0,2000,128):
            batch=data.make_batch(indices[start:start+128],'cuda');counts=batch.interface_mask.sum(1).cpu().numpy()
            with speed_context(opts,batch):
                with precision_context(opts):encoded=model.encode(batch)
                exact=model.infer_encoded(encoded);values=[]
                assert exact.log_probabilities.dtype==torch.float32
                for state in range(8):
                    prefix=np.zeros(batch.interface_mask.shape,dtype=np.int64)
                    for i,k in enumerate(counts):
                        prefix[i,:k]=[(state>>int(k-1-j))&1 for j in range(k)]
                    out=model.conditional(encoded,torch.from_numpy(prefix).cuda());assert out.logits.dtype==torch.float32
                    values.append(out.logits.cpu().numpy())
            logits.append(np.stack(values,axis=1));maps.append(exact.map_states.cpu().numpy())
            masks.append(batch.interface_mask.cpu().numpy()[:,None,:] & (np.arange(8)[None,:,None]<(2**counts)[:,None,None]))
    arrays=dict(logits=np.concatenate(logits),mask=np.concatenate(masks),MAP=np.concatenate(maps),indices=indices)
    a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'allpaths.npz',**arrays)
    result=dict(mode=a.mode,records=2000,path='every binary prefix; padding paths excluded',weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),logits_dtype='float32',enumeration_dtype='float32',optimizer_updates=0,outer_opened=False)
    if a.reference:
        with np.load(a.reference/'allpaths.npz') as ref:
            assert np.array_equal(arrays['indices'],ref['indices']) and np.array_equal(arrays['mask'],ref['mask'])
            result.update(logit_max_abs=float(abs(arrays['logits']-ref['logits'])[arrays['mask']].max()),MAP_row_disagreement=float(np.any(arrays['MAP']!=ref['MAP'],axis=1).mean()))
    else:result.update(logit_max_abs=0.,MAP_row_disagreement=0.)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
