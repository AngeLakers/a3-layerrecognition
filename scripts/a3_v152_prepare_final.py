"""Final V1.5.2 paired validation, one global decision, and train-only fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def offset_stats(x):
    q=np.quantile(np.asarray(x),[.25,.5,.75])
    return dict(count=len(x),median=float(q[1]),iqr=float(q[2]-q[0]),within80=float(np.mean(np.abs(x)<=80)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--query-cache',type=Path,required=True)
    p.add_argument('--phase',choices=['validate','fit'],required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.data.board_timebase import adoption_decision
    from a3_ultrasound.data.window_diagnostics import diagnostic_envelope
    from a3_ultrasound.data.quick_input import transform_trace
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.physical_features import energy_log_ratios,packed_energy_bounds
    from a3_ultrasound.v15_protocol import load_timebase_artifact
    cfgpath=a.root/'local_config.json';cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    train=cfg['folds'][0]['train']
    if a.phase=='validate':
        rfpath=a.root/'rf_calibration/rf_only_calibration.json';rf=json.loads(rfpath.read_text(encoding='utf-8'))
        registry={r['board']:r for r in rf['rows']}
        cache=json.loads((a.query_cache/'coverage.json').read_text(encoding='utf-8'))
        assert cache['entities']==train
        check_ds=RFShardDataset(cfg['catalog'],train,training=True,seed=23,tail_policy='nominal_fixed')
        check_shards={r['entity']:r for r in check_ds.shards}
        label_access=[]
        for shard in check_ds.shards:
            for key in ('y','label_mask','eligible'):
                path=check_ds.path.parent/shard['index_dir']/(key+'.npy')
                label_access.append(dict(board=shard['entity'],field=key,path=str(path),sha256=sha(path)))
        records=[];cells=[]
        for r in cache['records']:
            assert r['board'] in train and r['usable']
            ar=check_shards[r['board']]['arrays'];i=r['sample_row']
            assert ar['source_row'][i]==r['source_row']
            assert ar['eligible'][i] and ar['label_mask'][i,r['k']-1] and ar['y'][i,r['k']-1]==1
            b=registry[r['board']];k=r['k'];path=a.query_cache/f"{r['board']}_k{k}_{r['source_row']}.npz"
            with np.load(path,allow_pickle=False) as z:raw=z['RF'].copy()
            assert hashlib.sha256(raw.tobytes()).hexdigest()==r['RF_sha256']
            env=diagnostic_envelope(raw);nominal=b['nominal_tau'][k-1]
            lo=math.ceil(nominal-130);hi=math.floor(nominal+130)
            assert 0<=lo<hi<len(raw)
            peak=lo+int(np.argmax(env[lo:hi+1]));assert env[peak]>1e-12
            records.append(dict(board=r['board'],k=k,source_row=r['source_row'],sample_row=r['sample_row'],
                peak=peak,nominal_tau=nominal,calibrated_tau=b['candidate_tau'][k-1],
                old_offset=peak-nominal,calibrated_offset=peak-b['candidate_tau'][k-1],
                search_edge=peak in (lo,hi),layer_mm=r['layer_mm'],cache_path=str(path),RF_sha256=r['RF_sha256']))
        assert len(records)==384
        for board,k in sorted({(r['board'],r['k']) for r in records}):
            one=[r for r in records if r['board']==board and r['k']==k]
            cells.append(dict(board=board,k=k,records=len(one),old=offset_stats([r['old_offset'] for r in one]),
                calibrated=offset_stats([r['calibrated_offset'] for r in one]),search_edge=sum(r['search_edge'] for r in one),
                shallow_3mm=k==1 and one[0]['layer_mm']==3))
        decision=adoption_decision(records)
        assert len(decision['cells'])==9
        selected=[]
        for r in rf['rows']:
            selected.append(dict(r,selected_tau=r['candidate_tau'] if decision['adopt_calibration'] else r['nominal_tau'],
                selected_bottom=r['t_bottom']['median'] if decision['adopt_calibration'] else r['nominal_bottom'],
                selected_mode='calibrated' if decision['adopt_calibration'] else 'nominal_fallback'))
        table=dict(version='v1.5.2-selected-timebase',rows=selected,global_decision=decision,
            rf_only_calibration_sha256=sha(rfpath),halfwidth_samples=80,gaussian_sigma_samples=40,gaussian_strength=.5)
        save(a.root/'selected_timebase.json',table)
        save(a.root/'validation.json',dict(rows=cells,records=records,decision=decision,
            definition='Same peak from fixed nominal +/-130 source interval; compare both centers; peak identity not ground truth',
            source_scope='Exact original query07 F1 training-defect identities, no replacements',
            source_cache_sha256=sha(a.query_cache/'coverage.json'),F1_train_label_access=label_access,
            heldout_labels_read=False,training_called=False))
        cfg['timebase_calibration_sha256']=sha(a.root/'selected_timebase.json');save(cfgpath,cfg)
        print(json.dumps(decision,ensure_ascii=True),flush=True)
    else:
        table=load_timebase_artifact(cfg)
        ds=RFShardDataset(cfg['catalog'],train,training=True,seed=23,tail_policy='nominal_fixed',
            window_policy=cfg['input_window_policy'],timebase_config=table)
        out=a.root/'physical_statistics';out.mkdir(exist_ok=True)
        total=0;sum_f=0.;sum_f2=0.;rows=[]
        for shard in ds.shards:
            ar=shard['arrays'];ids=np.flatnonzero(ar['eligible']);k=shard['K'];b=ds.timebase_rows[shard['entity']]
            assert np.asarray(ar['label_mask'][ids,:k]).all()
            i=int(ids[0]);off=int(ar['offsets'][i]);length=int(ar['lengths'][i])
            _,meta,info=transform_trace(shard['rf'][off:off+length],shard,ds.config,timebase=b)
            tau=np.asarray(info['nominal_tof_samples'])
            left,right=packed_energy_bounds(meta,info['valid_length'],k)
            needed=max(181,math.floor(float(right[-1]))+1)
            # These actual frozen TRAIN shards have one retained length each.
            # Refuse a future variable-length population rather than fit a
            # first-record packed boundary for a different record length.
            assert np.all(np.minimum(ar['lengths'][ids],info['proposed_stop'])==info['valid_length'])
            assert tau[0]-104>=0
            assert tau[-1]+104<=min(int(np.min(ar['lengths'][ids])),info['proposed_stop'])-1
            pieces=[]
            for start in range(0,len(ids),128):
                ix=ids[start:start+128]
                x=np.asarray(shard['rf'][ar['offsets'][ix,None]+np.arange(needed)],np.float32)/np.float32(ds.config['normalization_divisor'])
                pieces.append(energy_log_ratios(x,left,right,front_end=180,epsilon=1e-8))
            f=np.concatenate(pieces);assert np.isfinite(f).all() and f.shape==(len(ids),k)
            np.savez_compressed(out/(shard['entity']+'.npz'),sample_rows=ids,source_rows=ar['source_row'][ids],f=f)
            total+=f.size;sum_f+=float(f.sum());sum_f2+=float(np.square(f).sum())
            rows.append(dict(board=shard['entity'],records=len(ids),interfaces=f.size,tau=tau.tolist(),half=80,
                cohort_source_rows_sha256=hashlib.sha256(np.asarray(ar['source_row'][ids],dtype='<i8').tobytes()).hexdigest()))
            print(json.dumps(dict(phase='fit',board=shard['entity'],interfaces=f.size)),flush=True)
        assert total==3836015
        mean=sum_f/total;std=float(np.sqrt(sum_f2/total-mean*mean));assert std>0
        norm=dict(front_end_samples=180,epsilon=1e-8,mean=mean,std=std,variance_ddof=0,
            interface_count=total,entities=train,input_tail_policy='nominal_fixed',normalization_divisor=131,
            input_window_policy=cfg['input_window_policy'],timebase_calibration_sha256=cfg['timebase_calibration_sha256'],
            effective_input_config=ds.config,scope='All unique complete reliable F1 training cohort including normals and defects; unjittered fit',
            rows=rows,validation_read=False,test_read=False,training_called=False)
        save(out/'normalization.json',norm);cfg['physical_statistics_sha256']=sha(out/'normalization.json');save(cfgpath,cfg)


if __name__=='__main__':main()
