"""Train-only full-cohort M5 fitting and descriptive D3 AUCs on CPU."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--frozen-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.quick_input import physical_manifest,transform_trace
    from a3_ultrasound.data.time_prior import derive_time_prior_samples
    from a3_ultrasound.data.physical_features import energy_log_ratios
    from sklearn.metrics import roc_auc_score
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=read(a.development/'diagnostics/front_packet_audit/front_packet_audit.json')
    front_end=audit['front_window_samples'][1]
    cfg=read(a.frozen_root/'analysis_inputs/proposed_ep12/resolved_config.json')
    catalog=read(a.development/'diagnostics/front_window/front_window_check.json')['source_catalog']
    ds=RFShardDataset(catalog,cfg['fold']['train'],training=True,seed=23,tail_policy='nominal_fixed')
    a.output.mkdir(parents=True,exist_ok=False)
    total=0;sum_f=0.;sum_f2=0.;groups=[];all_f=[];all_y=[];all_upper=[]
    started=time.perf_counter()
    def stats(f,y,upper,board):
        result=[]
        for value in (False,True):
            m=upper==value;xx=f[m];yy=y[m]
            result.append(dict(board=board,upstream_any_abnormal=value,count=int(m.sum()),positive_count=int(yy.sum()),negative_count=int((~yy).sum()),auc=float(roc_auc_score(yy,xx)) if len(np.unique(yy))==2 else None,quantiles=np.quantile(xx,[0,.01,.25,.5,.75,.99,1]).tolist() if len(xx) else None))
        return result
    for shard in ds.shards:
        ar=shard['arrays'];ids=np.flatnonzero(ar['eligible']);k=shard['K']
        first=int(ids[0]);offset=int(ar['offsets'][first]);length=int(ar['lengths'][first])
        _,_,info=transform_trace(shard['rf'][offset:offset+length],shard,ds.config)
        stop=info['valid_length'];assert np.all(ar['lengths'][ids]>=stop)
        prior=derive_time_prior_samples(physical_manifest(shard,ds.config),trace_length=length)
        fs=[]
        for j in range(0,len(ids),1024):
            ix=ids[j:j+1024]
            # Match transform_trace's float32 division before float64 energy.
            x=np.asarray(shard['rf'][ar['offsets'][ix,None]+np.arange(stop)],np.float32)/np.float32(ds.config['normalization_divisor'])
            fs.append(energy_log_ratios(x,prior.window_start_samples,prior.window_end_samples,front_end=front_end))
        f=np.concatenate(fs);assert f.shape==(len(ids),k) and np.isfinite(f).all()
        y=np.asarray(ar['y'][ids],bool);upper=(np.cumsum(y,axis=1)-y)>0
        np.savez_compressed(a.output/(shard['entity']+'.npz'),sample_rows=ids,source_rows=ar['source_row'][ids],f=f,y=y,upstream_any=upper)
        total+=f.size;sum_f+=float(f.sum());sum_f2+=float(np.square(f).sum())
        groups+=stats(f.ravel(),y.ravel(),upper.ravel(),shard['entity'])
        all_f.append(f.ravel());all_y.append(y.ravel());all_upper.append(upper.ravel())
        print(json.dumps(dict(board=shard['entity'],records=len(ids),interfaces=f.size,seconds=time.perf_counter()-started),ensure_ascii=True),flush=True)
    mean=sum_f/total;std=float(np.sqrt(sum_f2/total-mean*mean));assert np.isfinite(std) and std>0
    groups+=stats(np.concatenate(all_f),np.concatenate(all_y),np.concatenate(all_upper),'ALL_TRAIN')
    normalization=dict(front_end_samples=front_end,epsilon=1e-8,mean=mean,std=std,variance_ddof=0,interface_count=total,entities=cfg['fold']['train'],scope='all unique complete reliable F1 training interfaces, no B5 repetition',input_tail_policy='nominal_fixed',normalization_divisor=ds.config['normalization_divisor'],catalog_sha256=hashlib.sha256(Path(catalog).read_bytes()).hexdigest())
    (a.output/'normalization.json').write_text(json.dumps(normalization,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=dict(normalization=normalization,groups=groups,cpu=True,validation_read=False,outer_opened=False,training_called=False,auc_interpretation='Raw signed f discriminates true current abnormal; AUC below .5 means reverse ordering, not automatically an invalid feature. No AUC-based tuning.',seconds=time.perf_counter()-started)
    (a.output/'D3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(normalization,ensure_ascii=True))


if __name__=='__main__':
    main()
