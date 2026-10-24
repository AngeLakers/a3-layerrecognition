"""Fit one train-QA window and audit all unique reliable training records."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_root/'deployment/code/src'))
    from a3_ultrasound.data.streaming import RFShardDataset
    profiles = a.development/'diagnostics/front_packet_profiles'
    rows = json.loads((profiles/'profiles.json').read_text(encoding='utf-8'))['rows']
    # Common physical upper bound: sample centers strictly before the earliest
    # nominal first-interface TOF on any F1 train plate. No label/val tuning.
    stop = math.ceil(min(r['first_tof'] for r in rows))
    fit = []
    for row in rows:
        with np.load(profiles/(row['board']+'.npz')) as z:
            e = np.cumsum(z['traces'][:, :stop]**2, axis=1)
        end = np.argmax(e >= .99*e[:, -1:], axis=1)
        fit.append(dict(board=row['board'], max_endpoint=int(end.max()), endpoint_q=np.quantile(end,[0,.5,.9,.99,1]).tolist()))
    bound = int(math.ceil(max(r['max_endpoint'] for r in fit)/10)*10)
    assert bound < stop, 'No separate common window supported by QA.'
    cfg = json.loads((a.frozen_root/'analysis_inputs/proposed_ep12/resolved_config.json').read_text(encoding='utf-8'))
    qa = json.loads((a.development/'diagnostics/front_window/front_window_check.json').read_text(encoding='utf-8'))
    ds = RFShardDataset(qa['source_catalog'], cfg['fold']['train'], training=True, seed=23)
    a.output.mkdir(parents=True, exist_ok=False)
    audit=[]
    for shard in ds.shards:
        ar=shard['arrays'];ids=np.flatnonzero(ar['eligible'])
        assert np.all(ar['lengths'][ids]>=stop)
        ratio=[];energy=[];oldratio=[];peaks=[]
        for j in range(0,len(ids),2048):
            ix=ids[j:j+2048]
            x=np.asarray(shard['rf'][ar['offsets'][ix,None]+np.arange(stop)],np.float64)/ds.config['normalization_divisor']
            total=np.sum(x*x,axis=1);front=np.sum(x[:,:bound+1]**2,axis=1)
            assert np.isfinite(total).all() and np.all(total>0)
            ratio.append(front/total);energy.append(front)
            oldratio.append(np.sum(x[:,:71]**2,axis=1)/total)
            peaks.append(np.argmax(np.abs(x),axis=1))
        ratio=np.concatenate(ratio);energy=np.concatenate(energy);oldratio=np.concatenate(oldratio);peaks=np.concatenate(peaks)
        r=dict(board=shard['entity'],count=len(ids),energy_fraction_q=np.quantile(ratio,[0,.001,.01,.05,.5,.95,1]).tolist(),fraction_below_99pct=float(np.mean(ratio<.99)),old_fraction_q=np.quantile(oldratio,[0,.01,.5,.99,1]).tolist(),front_energy_q=np.quantile(energy,[0,.001,.01,.5,.99,1]).tolist(),front_energy_le_epsilon=int(np.sum(energy<=1e-8)),peak_outside_count=int(np.sum(peaks>bound)),short_record_count=0)
        audit.append(r)
        np.savez_compressed(a.output/(shard['entity']+'.npz'),sample_rows=ids,source_rows=ar['source_row'][ids],energy_fraction=ratio,front_energy=energy,early_peak=peaks)
        print(json.dumps(r,ensure_ascii=True),flush=True)
    result=dict(scope='F1 training only; unique complete reliable records; labels not used for window fit objective',front_window_samples=[0,bound],interval='inclusive integer centers',early_reference_stop_exclusive=stop,fit_rule='QA maximum 99pct-energy endpoint before common earliest nominal TOF, rounded upward to 10 samples',fit_rows=fit,audit_rows=audit,training_called=False,outer_opened=False,validation_read=False,catalog_sha256=hashlib.sha256(Path(qa['source_catalog']).read_bytes()).hexdigest(),caveats=['Early-reference energy coverage is not independent acoustic calibration','Front window overlaps fixed Local support on 3mm plates; no local TOF or crop change is included'])
    (a.output/'front_packet_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(front_window_samples=[0,bound],total=sum(r['count'] for r in audit)),ensure_ascii=True))


if __name__ == '__main__':
    main()
