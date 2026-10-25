"""RF-only per-board anchor census. No dataset, labels or training imported."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ALLOWED={'RF','offsets','lengths','source_row'}


def guarded_load(field,path,access):
    assert field in ALLOWED,'Calibration is forbidden to open labels or other fields.'
    if field!='RF':assert Path(path).name==field+'.npy'
    access.append(dict(field=field,path=str(path)))
    return np.load(path,mmap_mode='r',allow_pickle=False)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def bind_rf_catalog(output,source,resume):
    path=Path(output)/'input_catalog_context.json';digest=sha(source)
    if resume:
        assert path.exists() and json.loads(path.read_text(encoding='utf-8'))['source_catalog_sha256']==digest, 'RF-only resume catalog/timing context changed or was not recorded.'
    else:save(path,dict(source_catalog_sha256=digest))
def stats(x):
    good=np.asarray(x)[np.isfinite(x)]
    if not len(good):return dict(count=0,median=None,iqr=None)
    q=np.quantile(good,[.25,.5,.75]);return dict(count=len(good),median=float(q[1]),iqr=float(q[2]-q[0]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rf-catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--hash-rf',action='store_true')
    a=p.parse_args();assert a.batch_size>0
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.data.board_timebase import envelope_anchor_positions,divided_centers
    catalog=json.loads(a.rf_catalog.read_text(encoding='utf-8'));a.output.mkdir(parents=True,exist_ok=a.resume)
    bind_rf_catalog(a.output,a.rf_catalog,a.resume)
    access=[];cache_access=[];rows=[];started=time.monotonic()
    for b in catalog['boards']:
        arrays={name:guarded_load(name,b[name+'_path'],access) for name in ('offsets','lengths','source_row')}
        for name in arrays:assert sha(b[name+'_path'])==b['index_sha256'][name]
        path=Path(b['rf_path']);before=path.stat()
        assert dict(bytes=before.st_size,mtime_ns=before.st_mtime_ns)==b['rf_source_stat']
        lengths=np.asarray(arrays['lengths'])
        nominal=(b['K']+1)*2*b['layer_mm']*1e-3*catalog['sampling_rate_hz']/b['nominal_c_mps']
        search_center=float(b.get('bottom_search_center_samples',nominal))
        start=np.full(b['records'],np.nan);bottom=start.copy();complete=np.zeros(b['records'],bool);edge=complete.copy()
        cache=a.output/(b['board']+'.npz');reused=a.resume and cache.exists()
        if reused:
            with np.load(cache,allow_pickle=False) as z:
                assert set(z.files)=={'t_start','t_bottom','complete_bottom_search','bottom_search_edge','source_rows'}
                assert np.array_equal(z['source_rows'],arrays['source_row'])
                start=z['t_start'].copy();bottom=z['t_bottom'].copy();complete=z['complete_bottom_search'].copy();edge=z['bottom_search_edge'].copy()
                assert all(v.shape==(b['records'],) for v in (start,bottom,complete,edge))
            cache_access.append(dict(board=b['board'],path=str(cache),sha256=sha(cache),reason='Completed RF-only measurements from interrupted prior attempt; source indices and RF stat unchanged'))
        else:
            rf=guarded_load('RF',path,access)
            for n in np.unique(lengths):
                ids=np.flatnonzero(lengths==n)
                for j in range(0,len(ids),a.batch_size):
                    ix=ids[j:j+a.batch_size];x=np.asarray(rf[arrays['offsets'][ix,None]+np.arange(n)],np.float64)
                    if not np.isfinite(x).all():
                        finite=np.isfinite(x).all(axis=1);ix=ix[finite];x=x[finite]
                    if not len(ix):continue
                    ss,bb,cc,ee=envelope_anchor_positions(x,search_center)
                    start[ix]=ss;bottom[ix]=bb;complete[ix]=cc;edge[ix]=ee
        s=stats(start);t=stats(bottom)
        assert s['count'] and t['count'],'Cannot calibrate a board with no valid anchors.'
        delta,tau=divided_centers(s['median'],t['median'],int(b['K']))
        if not reused:
            np.savez_compressed(cache,t_start=start,t_bottom=bottom,complete_bottom_search=complete,
                bottom_search_edge=edge,source_rows=arrays['source_row'])
        rows.append(dict(board=b['board'],K=b['K'],records=b['records'],t_start=s,t_bottom=t,
            start_failed=int(np.sum(~np.isfinite(start))),bottom_failed=int(np.sum(~np.isfinite(bottom))),
            bottom_incomplete=int(np.sum(~complete)),bottom_edge_peaks=int(edge.sum()),delta=delta,candidate_tau=tau,
            nominal_tau=[nominal*j/(b['K']+1) for j in range(1,b['K']+1)],nominal_bottom=nominal,
            bottom_search_center_samples=search_center,source_RF_sha256=sha(path) if a.hash_rf else None,
            source_RF_stat_unchanged=True,source_index_sha256=b['index_sha256']))
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        print(json.dumps(dict(board=b['board'],t_start=s,t_bottom=t,delta=delta,seconds=time.monotonic()-started),ensure_ascii=True),flush=True)
    assert all(r['field'] in ALLOWED for r in access)
    result=dict(version='v1.5.2-final-rf-anchors',sampling_rate_hz=catalog['sampling_rate_hz'],rows=rows,
        definition='Full-source Hilbert envelope, timing-only first16 baseline, reflect256; front[0,200] and complete predeclared RF coarse bottom +/-130 argmax',
        labels_read=False,access_audit=access,cache_access_audit=cache_access,
        source_catalog_sha256=sha(a.rf_catalog),raw_RF_modified=False)
    save(a.output/'rf_only_calibration.json',result)
    save(a.output/'calibration_sha256.json',dict(path='rf_only_calibration.json',sha256=sha(a.output/'rf_only_calibration.json')))


if __name__=='__main__':main()
