"""Prepare V1.5.2 descriptive RF checks and train-only energy statistics; no tests/training."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(count=0, median=None, iqr=None)
    q = np.quantile(a, [.25, .5, .75])
    return dict(count=len(a), median=float(q[1]), iqr=float(q[2]-q[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--aligned-catalog', type=Path, required=True)
    p.add_argument('--query07-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['fit', 'query', 'backwall', 'all'], default='all')
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.quick_input import transform_trace
    from a3_ultrasound.data.physical_features import energy_log_ratios
    from a3_ultrasound.data.window_diagnostics import diagnostic_envelope, packet_span, span_covered, front_overlap_energy_fraction
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    assert cfg['execution_version'] == 'v1.5.2'
    assert cfg['input_window_policy'] == {'name': 'half_interval_capped', 'cap_samples': 110}
    a.output.mkdir(parents=True, exist_ok=True)
    train = cfg['folds'][0]['train']
    started = time.monotonic()
    # The dataset is constructed for F1 training entities only; held-out labels
    # are never required by any preparation phase.
    ds = None
    if a.phase in ('fit', 'query', 'all'):
        ds = RFShardDataset(cfg['catalog'], train, training=True, seed=23,
            tail_policy='nominal_fixed', window_policy=cfg['input_window_policy'])
        assert ds.entities == train
    if a.phase in ('fit', 'all'):
        folder = a.output/'physical_statistics'; folder.mkdir(exist_ok=True)
        count = 0; sum_f = 0.; sum_f2 = 0.; rows = []; accesses = []
        for shard in ds.shards:
            ar = shard['arrays']; ids = np.flatnonzero(ar['eligible']); k = shard['K']
            assert np.asarray(ar['label_mask'][ids, :k]).all()
            first = int(ids[0]); off = int(ar['offsets'][first]); n = int(ar['lengths'][first])
            _, _, info = transform_trace(shard['rf'][off:off+n], shard, ds.config)
            tau = np.asarray(info['nominal_tof_samples']); half = info['halfwidth_samples']
            needed = max(181, math.floor(float(tau[-1]+half))+1)
            assert np.all(np.asarray(ar['lengths'][ids]) >= needed)
            assert min(info['proposed_stop'], int(np.min(ar['lengths'][ids]))) > tau[-1]+half
            pieces = []
            for j in range(0, len(ids), 128):
                ix = ids[j:j+128]
                x = np.asarray(shard['rf'][ar['offsets'][ix, None]+np.arange(needed)], np.float32)/np.float32(ds.config['normalization_divisor'])
                pieces.append(energy_log_ratios(x, tau-half, tau+half, front_end=180, epsilon=1e-8))
            f = np.concatenate(pieces)
            assert f.shape == (len(ids), k) and np.isfinite(f).all()
            np.savez_compressed(folder/(shard['entity']+'.npz'), sample_rows=ids, source_rows=ar['source_row'][ids], f=f)
            count += f.size; sum_f += float(f.sum()); sum_f2 += float(np.square(f).sum())
            rows.append(dict(board=shard['entity'], records=len(ids), interfaces=f.size, half=half,
                nominal_tau=tau.tolist(), input_prefix_needed=needed,
                source_rows_sha256=hashlib.sha256(np.asarray(ar['source_row'][ids], dtype='<i8').tobytes()).hexdigest()))
            base=ds.path.parent/shard['index_dir']
            for name in ('eligible','label_mask','y'):
                path=base/(name+'.npy');accesses.append(dict(board=shard['entity'],path=str(path),sha256=sha(path),purpose='Train cohort membership only'))
            print(json.dumps(dict(phase='fit',board=shard['entity'],interfaces=f.size,seconds=time.monotonic()-started),ensure_ascii=True),flush=True)
        assert count == 3836015
        mean = sum_f/count; std = float(np.sqrt(sum_f2/count-mean*mean))
        assert np.isfinite(std) and std > 0
        norm = dict(front_end_samples=180, epsilon=1e-8, mean=mean, std=std, variance_ddof=0,
            interface_count=count, entities=train, scope='all unique complete reliable F1 training interfaces, no B5 repetition',
            input_tail_policy='nominal_fixed', normalization_divisor=131,
            catalog_sha256=sha(cfg['catalog']), input_window_policy=cfg['input_window_policy'],
            effective_input_config=ds.config, rows=rows)
        save(folder/'normalization.json', norm)
        save(folder/'access_audit.json',dict(labels=accesses,validation_read=False,test_read=False,training_called=False))
    if a.phase in ('query', 'all'):
        folder=a.output/'query07';folder.mkdir(exist_ok=True)
        records=[];cells=[];artifacts=[]
        for shard in ds.shards:
            ar=shard['arrays'];source=np.asarray(ar['source_row']);order=np.argsort(source);sorted_source=source[order]
            assert len(np.unique(source))==len(source)
            path=a.query07_root/shard['entity']/'defect_timing_qa.json'
            items=json.loads(path.read_text(encoding='utf-8'));artifacts.append(dict(path=str(path),sha256=sha(path)))
            for item in items:
                k=int(item['interface']);one=[]
                for identity in item['sample_ids']:
                    board,source_id=identity.rsplit(':',1);assert board==shard['entity'];source_id=int(source_id)
                    pos=int(np.searchsorted(sorted_source,source_id))
                    r=dict(board=board,k=k,sample_id=identity,source_row=source_id)
                    if pos>=len(source) or sorted_source[pos]!=source_id:
                        r.update(usable=False,reason='source_identity_missing');records.append(r);one.append(r);continue
                    row=int(order[pos]);r['sample_row']=row
                    if not (ar['eligible'][row] and ar['label_mask'][row,k-1] and ar['y'][row,k-1]==1):
                        r.update(usable=False,reason='not_current_reliable_eligible_defect');records.append(r);one.append(r);continue
                    off=int(ar['offsets'][row]);n=int(ar['lengths'][row]);x=np.array(shard['rf'][off:off+n],copy=True)
                    _,_,info=transform_trace(x,shard,ds.config)
                    nominal=info['nominal_tof_samples'][k-1];half=info['halfwidth_samples']
                    span=packet_span(diagnostic_envelope(x),nominal)
                    r.update(usable=True,nominal=nominal,new_half=half,span=span,
                        old_covered=span_covered(span,nominal,70),new_covered=span_covered(span,nominal,half),
                        old_front_overlap=front_overlap_energy_fraction(x,nominal,70),
                        new_front_overlap=front_overlap_energy_fraction(x,nominal,half),
                        layer_mm=shard['layer_mm'],RF_sha256=hashlib.sha256(x.tobytes()).hexdigest())
                    np.savez_compressed(folder/f'{board}_k{k}_{source_id}.npz',RF=x,metadata_json=json.dumps(r,ensure_ascii=False))
                    records.append(r);one.append(r)
                usable=[r for r in one if r['usable']];valid=[r for r in usable if r['span']['valid']]
                denom=len(usable)
                cells.append(dict(board=shard['entity'],k=k,layer_mm=shard['layer_mm'],requested=len(one),usable=denom,
                    invalid=len(one)-denom,valid_candidate=len(valid),
                    old_complete=sum(r['old_covered'] for r in usable),new_complete=sum(r['new_covered'] for r in usable),
                    old_fraction=sum(r['old_covered'] for r in usable)/denom if denom else None,
                    new_fraction=sum(r['new_covered'] for r in usable)/denom if denom else None,
                    search_edge_touched=sum(r['span'].get('search_edge_touched',False) for r in usable),
                    source_censored=sum(r['span'].get('source_censored',False) for r in usable),
                    old_front_overlap=summary([r['old_front_overlap'] for r in usable if r['old_front_overlap'] is not None]),
                    new_front_overlap=summary([r['new_front_overlap'] for r in usable if r['new_front_overlap'] is not None])))
        save(folder/'coverage.json',dict(rows=cells,records=records,source_artifacts=artifacts,
            definition='Same Hilbert-envelope >20% peak span for +/-70 and new windows; fixed +/-130 search. Candidate echo identity unverified.',
            fraction_denominator='All usable mapped reliable eligible defect records, including invalid/zero candidate spans as not covered',
            search_edge_definition='Any above-threshold candidate point touches fixed search edge before full-source extension',
            front_overlap_definition='RF energy in Local intersect [0,180] / RF energy in Local; proxy, not physical separation',
            entities=train,validation_read=False,test_read=False,training_called=False))
        print(json.dumps(dict(phase='query',records=len(records),usable=sum(r['usable'] for r in records)),ensure_ascii=True),flush=True)
    if a.phase in ('backwall', 'all'):
        catalog=json.loads(a.aligned_catalog.read_text(encoding='utf-8'))
        folder=a.output/'backwall';folder.mkdir(exist_ok=True);rows=[];access=[]
        for b in catalog['boards']:
            arrays={}
            for key in ('offsets','lengths','source_row'):
                path=Path(b[key+'_path']);assert sha(path)==b['index_sha256'][key]
                arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
                access.append(dict(board=b['board'],field=key,path=str(path)))
            rfpath=Path(b['rf_path']);before=rfpath.stat()
            assert dict(bytes=before.st_size,mtime_ns=before.st_mtime_ns)==b['rf_source_stat']
            rf=np.load(rfpath,mmap_mode='r',allow_pickle=False);access.append(dict(board=b['board'],field='RF',path=str(rfpath)))
            nrows=b['records'];nominal=(b['K']+1)*2*b['layer_mm']*1e-3*catalog['sampling_rate_hz']/b['nominal_c_mps']
            lo=math.ceil(nominal-130);hi=math.floor(nominal+130)
            lengths=np.asarray(arrays['lengths']);peaks=np.full(nrows,np.nan);edge=np.zeros(nrows,bool)
            complete=(lengths>hi)&(lo>=0)
            for n in np.unique(lengths):
                if n<=lo:continue
                ids=np.flatnonzero(lengths==n);right=min(hi,int(n)-1)
                for j in range(0,len(ids),256):
                    ix=ids[j:j+256]
                    x=np.asarray(rf[arrays['offsets'][ix,None]+np.arange(n)],dtype=np.float64)
                    env=diagnostic_envelope(x);part=env[:,lo:right+1];where=np.argmax(part,axis=1)
                    found=lo+where;amplitude=part[np.arange(len(ix)),where]
                    good=amplitude>1e-12;peaks[ix[good]]=found[good]
                    edge[ix[good]]=(found[good]==lo)|(found[good]==right)
            valid=np.isfinite(peaks)
            np.savez_compressed(folder/(b['board']+'.npz'),peak=peaks,complete=complete,search_edge=edge,source_rows=arrays['source_row'])
            stat=summary(peaks[valid]);full=summary(peaks[valid&complete])
            rows.append(dict(board=b['board'],records=nrows,nominal=nominal,all_available=stat,complete_search=full,
                median_minus_nominal=stat['median']-nominal if stat['count'] else None,
                complete_median_minus_nominal=full['median']-nominal if full['count'] else None,
                partial_search=int(np.sum(~complete)),empty_or_zero=int(np.sum(~valid)),search_edge=int(edge.sum()),
                note='Candidate single-backwall envelope peak, no calibration or label filtering'))
            after=rfpath.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
            print(json.dumps(dict(phase='backwall',board=b['board'],records=nrows,seconds=time.monotonic()-started),ensure_ascii=True),flush=True)
        assert all(v['field'] in ('RF','offsets','lengths','source_row') for v in access)
        save(folder/'summary.json',dict(rows=rows,access_audit=access,labels_read=False,calibration_performed=False,
            search_edge_definition='Candidate peak equals either available edge of the nominal +/-130 source search',
            full_RF_sha256_recomputed=False,source_stat_and_index_unchanged=True,training_called=False))
    save(a.output/f'{a.phase}_completed.json',dict(status='DESCRIPTIVE_PREPARATION_COMPLETE',phase=a.phase,
        seconds=time.monotonic()-started,tests_run=False,training_called=False))


if __name__=='__main__':
    main()
