"""Frozen ep12 Pre-attention with a train-defined front window, CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-root',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--front-window',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.frozen_root/'deployment/code/src'))
    from a3_ultrasound.training import ModelConfig,build_model
    from a3_ultrasound.data.streaming import RFShardDataset
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    front=read(a.front_window)['front_window_samples'];assert front==[0,180]
    source=a.frozen_root/'analysis_inputs/proposed_ep12'
    for name,sha in read(source/'source_manifest.json')['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
    cp=torch.load(source/'epoch_12.pt',map_location='cpu',weights_only=True)
    cfg=cp['resolved_config'];assert cp['epoch']==12 and not cfg['test_opened']
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model=build_model(ModelConfig(**cfg['model'])).cpu().eval();model.load_state_dict(cp['model_state_dict'])
    data=RFShardDataset(a.catalog,cfg['fold']['validation'])
    data.use_frozen_evaluation_selection(a.frozen_root/'deployment/validation_selection/F1.npz')
    assert len(data)==40002
    a.output.mkdir(parents=True,exist_ok=False);arrays={k:[] for k in ('sample_id','board','k','weight','mass')}
    start=time.perf_counter()
    with torch.inference_mode():
        for batch,indices in data.batches(32,'cpu'):
            e=model.encode(batch);centers=e.backbone.feature_centers
            if centers.ndim==1:centers=centers[None].expand(len(indices),-1)
            for i,idx in enumerate(indices):
                sid,board=data.identity(idx)
                for j in range(int(batch.interface_mask[i].sum())):
                    support=e.pre_support[i,j]
                    f=support&(centers[i]>=0)&(centers[i]<=front[1])
                    up=torch.zeros_like(f)
                    for k in range(j):up|=(centers[i]>=e.window_start[i,k])&(centers[i]<=e.window_end[i,k])
                    up&=support&~f;other=support&~f&~up
                    mass=[float(e.pre_readout.attention[i,j][s].sum()) for s in (f,up,other)]
                    assert abs(sum(mass)-int(support.any()))<2e-6
                    for name,value in [('sample_id',sid),('board',board),('k',j+1),('weight',int(data.evaluation_weights[idx])),('mass',mass)]:arrays[name].append(value)
            if (int(indices[-1])+1)%2560==0 or int(indices[-1])+1==len(data):
                v=dict(records=int(indices[-1])+1,total=len(data),seconds=time.perf_counter()-start)
                (a.output/'progress.json').write_text(json.dumps(v),encoding='utf-8');print(json.dumps(v),flush=True)
    arrays={k:np.asarray(v) for k,v in arrays.items()};np.savez_compressed(a.output/'pre_attention.npz',**arrays)
    med=float(np.median(arrays['mass'][arrays['k']>=2,0]));groups=[]
    for board in ['ALL']+data.entities:
        for k in np.unique(arrays['k']):
            m=(arrays['k']==k)&((arrays['board']==board) if board!='ALL' else True)
            if m.any():groups.append(dict(board=board,k=int(k),count=int(m.sum()),q25_median_q75=np.quantile(arrays['mass'][m],[.25,.5,.75],axis=0).tolist()))
    decision=dict(front_window_samples=front,front_window_audit_sha256=hashlib.sha256(a.front_window.read_bytes()).hexdigest(),front_mass_nonfirst_raw_median=med,threshold=.5,enable_m6=med>.5,pre_exclude_front_end=front[1] if med>.5 else None,groups=groups,scope='fixed validation developmental architecture decision after train-only window definition',outer_opened=False,training_called=False,epoch=12,validation_count=40002)
    (a.output/'M6_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(median=med,enable_m6=med>.5)))


if __name__=='__main__':main()
