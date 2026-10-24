"""One discarded TRAIN RF update per V1.5 Head, never a performance run."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch


def frontend_digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        if not name.startswith('pcc.'):
            h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.v15_protocol import HEADS,validate_config,load_physical_artifacts
    from a3_ultrasound.freeze_protocol import verify_panels
    from a3_ultrasound.stream_training import catalog_signature
    from a3_ultrasound.training import ModelConfig,TrainingConfig,build_model,build_optimizer,TensorDataset,evaluate
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.epoch_sampling import use_epoch
    from a3_ultrasound.prefix_perturbation import training_conditional
    from a3_ultrasound.models.losses import head_training_loss
    from a3_ultrasound.validation_diagnostics import CorrectionArchive
    cfg=json.loads(a.config.read_text(encoding='utf-8'));physical,_=load_physical_artifacts(cfg)
    for head in HEADS:validate_config(cfg,fold_name='F1',seed=23,head=head,require_permission=False)
    if a.device=='cuda':
        assert '3080 Ti' in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    ds=RFShardDataset(cfg['catalog'],cfg['folds'][0]['train'],training=True,seed=23,
        physical_config=physical,tail_policy='nominal_fixed')
    sig=catalog_signature(ds.catalog);verify_panels(cfg,sig)
    hashes=[use_epoch(ds,cfg['epoch_selection_root'],e,seed=23,catalog_signature=sig) for e in range(1,41)]
    use_epoch(ds,cfg['epoch_selection_root'],1,seed=23,catalog_signature=sig)
    ids=[]
    for i,s in enumerate(ds.shards):
        ix=np.flatnonzero(ds.refs[:,0]==i);positive=s['arrays']['y'][ds.refs[ix,1]].any(1)
        ids.extend([int(ix[positive][0]),int(ix[~positive][0])])
    batch=ds.make_batch(ids,a.device);identities=[ds.identity(i) for i in ids]
    a.output.mkdir(parents=True,exist_ok=False);results=[]
    for head in HEADS:
        torch.manual_seed(23)
        if a.device=='cuda':torch.cuda.manual_seed_all(23)
        model=build_model(ModelConfig(**dict(cfg['model'],head_type=head))).to(a.device)
        tc=TrainingConfig(**dict(cfg['training'],device=a.device))
        optim,_=build_optimizer(model,tc);before=frontend_digest(model)
        model.train();out,pert=training_conditional(model,batch,probability=.2,seed=23,epoch=1,step=0)
        loss=head_training_loss(out,batch,head_type=head,lambda_local=.3,local_aux_mode='clean_upstream')
        loss.total.backward()
        assert torch.isfinite(loss.total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters())
        optim.step();assert frontend_digest(model)!=before
        _,rows=evaluate(model,TensorDataset(batch,tuple(v[0] for v in identities),tuple(v[1] for v in identities)),device=a.device)
        if head=='proposed_v15':
            archive=CorrectionArchive()
            for row in rows:archive.add(dict(row,sample_weight=1))
            archive.save(a.output/'discarded_proposed_diagnostics.npz',epoch=0,checkpoint='discarded_train_witness',commit_hash='candidate',config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest())
        results.append(dict(head=head,initial_frontend_sha256=before,backbone_updated=True,perturbation=pert,
            groupnorm=sum(isinstance(m,torch.nn.GroupNorm) for m in model.backbone.modules()),batchnorm=sum(isinstance(m,torch.nn.BatchNorm1d) for m in model.backbone.modules())))
        del out,loss,optim,model,rows
    assert len(set(r['initial_frontend_sha256'] for r in results))==1
    assert all(r['groupnorm']==12 and r['batchnorm']==0 for r in results)
    chain=[r['perturbation']['attempt_mask_sha256'] for r in results if r['head'] in ('vanilla_chain','proposed_v15')]
    assert len(set(chain))==1
    report=dict(status='PASS',device=a.device,heads=results,train_identities=identities,epoch_indices_sha256=hashes,
        scope='14 TRAIN RF records, one discarded update per Head; no reusable weights; not validation performance',outer_opened=False,validation_RF_read=False)
    (a.output/'witness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',device=a.device,heads=list(HEADS))))


if __name__=='__main__':main()
