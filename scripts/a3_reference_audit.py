"""RF-only nominal-reference/mask census and reused F1-D waveform diagnostics."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import math
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.data.referenced_input import reference_geometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def qstats(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):
        return dict(count=0,median=None,iqr=None,q01=None,q99=None,within70=None)
    q=np.quantile(x,[.01,.25,.5,.75,.99])
    return dict(count=len(x),median=float(q[2]),iqr=float(q[3]-q[1]),q01=float(q[0]),q99=float(q[4]),within70=float(np.mean(np.abs(x)<=70)))


def run(catalog_path,old_d,output):
    cat=json.loads(catalog_path.read_text(encoding='utf-8'));fs=cat['sampling_rate_hz']
    assert cat['default_mode']=='board' and cat['training_permitted'] is False
    output.mkdir(parents=True,exist_ok=False);(output/'D_waveforms').mkdir()
    registry={b['board']:b for b in cat['boards']}
    access=[]
    allowed={str(Path(b[key]).resolve()) for b in cat['boards'] for key in ('rf_path','offsets_path','lengths_path','source_row_path')}
    def source(path):
        assert str(Path(path).resolve()) in allowed
        access.append(str(Path(path)))
        return np.load(path,mmap_mode='r',allow_pickle=False)
    census=[]
    for b in cat['boards']:
        lengths=source(b['lengths_path']);n=len(lengths)
        new=reference_geometry(K=b['K'],layer_mm=b['layer_mm'],sampling_rate_hz=fs,
            sound_speed_mps=b['nominal_c_mps'],reference_samples=b['board_reference_samples'],source_length=int(lengths.max()))
        old=reference_geometry(K=b['K'],layer_mm=b['layer_mm'],sampling_rate_hz=fs,
            sound_speed_mps=b['nominal_c_mps'],reference_samples=0,source_length=int(lengths.max()))
        def local_counts(g):
            left=np.ceil(g.window_starts).astype(np.int64);right=np.floor(g.window_ends).astype(np.int64)
            available=np.maximum(0,np.minimum(right[None,:],lengths[:,None]-1)-np.maximum(left[None,:],0)+1)
            declared=right-left+1
            return dict(incomplete_per_interface=(available!=declared).sum(axis=0).tolist(),empty_per_interface=(available==0).sum(axis=0).tolist())
        bwleft=new.backwall_search_start;bwright=new.backwall_search_end
        bwcounts=np.maximum(0,np.minimum(bwright,lengths-1)-max(0,bwleft)+1)
        oldstop=np.minimum(lengths,math.ceil(old.backwall_center+.5e-6*fs))
        newstop=math.ceil(new.backwall_center+.5e-6*fs)
        r=dict(board=b['board'],records=n,reference_samples=b['board_reference_samples'],layer_mm=b['layer_mm'],K=b['K'],
            nominal_c_mps=b['nominal_c_mps'],nominal_layer_interval_samples=2*b['layer_mm']*1e-3*fs/b['nominal_c_mps'],
            old_tau=old.interface_centers.tolist(),new_tau=new.interface_centers.tolist(),
            old_local_support=local_counts(old),new_local_support=local_counts(new),
            nominal_backwall=new.backwall_center,backwall_center_outside_count=int(np.sum(new.backwall_center>lengths-1)),
            backwall_window_partial_count=int(np.sum((bwcounts>0)&(bwcounts<new.backwall_declared_count))),
            backwall_window_empty_count=int(np.sum(bwcounts==0)),
            nominal_backwall_plus_guard=newstop,shorter_than_nominal_guard_count=int(np.sum(lengths<newstop)),
            source_length_min=int(lengths.min()),source_length_max=int(lengths.max()),source_length_unique_count=len(np.unique(lengths)),
            old_effective_length_unique_count=len(np.unique(oldstop)),records_with_old_second_crop=int(np.sum(oldstop<lengths)),
            old_removed_samples_total=int(np.sum(lengths-oldstop)),all_records_retained=True)
        census.append(r)
    save(output/'RF_only_census.json',dict(rows=census,total_records=sum(r['records'] for r in census),labels_read=False))
    with np.load(old_d/'validation_samples.npz',allow_pickle=False) as z:
        panel={key:z[key].copy() for key in z.files}
    oldrows=json.loads((old_d/'validation.json').read_text(encoding='utf-8'))['rows']
    train=['TCL3-2','TCL3-3','TCL3-5','TCL3-6','TCL4-3','TCL4-4','薄']
    rows=[]
    for item in oldrows:
        board=item['board'];k=item['k'];assert board in train
        b=registry[board];prefix=f'{board}_k{k}'
        ids=panel[prefix+'_sample_rows'];expected_source=panel[prefix+'_source_rows'];assert len(ids)==50
        raw=source(b['rf_path']);offsets=source(b['offsets_path']);lengths=source(b['lengths_path']);source_rows=source(b['source_row_path'])
        np.testing.assert_array_equal(source_rows[ids],expected_source)
        spacing=2*b['layer_mm']*1e-3*fs/b['nominal_c_mps'];oldtau=k*spacing;newtau=oldtau+b['board_reference_samples']
        middle=(oldtau+newtau)/2;radius=min(.65e-6*fs,.45*spacing)
        roi_left=math.ceil(middle-radius);roi_right=math.floor(middle+radius)
        observed=[];oldenergy=[];newenergy=[];roi_front_ratio=[];truncated=[];flats=[];envflat=[];record_offsets=[];record_lengths=[]
        for i in ids:
            i=int(i);x=np.asarray(raw[int(offsets[i]):int(offsets[i]+lengths[i])],np.float32)
            y=x.astype(float)-np.median(x[:16])
            env=np.abs(hilbert(np.pad(y,(256,256),mode='reflect')))[256:256+len(y)]
            left=max(0,roi_left);right=min(len(y)-1,roi_right);truncated.append(left!=roi_left or right!=roi_right)
            positions=np.arange(left,right+1);power=env[left:right+1]**2;energy=float(power.sum())
            center=float(np.sum(power*positions)/energy) if energy>1e-12 else np.nan
            observed.append(center)
            for tau,target in ((oldtau,oldenergy),(newtau,newenergy)):
                mask=(positions>=tau-70)&(positions<=tau+70)
                target.append(float(power[mask].sum()/energy) if energy>1e-12 else np.nan)
            front=float(np.sum(env[:201]**2))
            roi_front_ratio.append(energy/(front+1e-12))
            record_offsets.append(sum(record_lengths));record_lengths.append(len(x));flats.append(x);envflat.append(env.astype(np.float32))
        observed=np.array(observed);oldoff=observed-oldtau;newoff=observed-newtau
        oldstat=qstats(oldoff);newstat=qstats(newoff)
        r=dict(board=board,k=k,selected=50,old_tau=oldtau,new_tau=newtau,reference_samples=b['board_reference_samples'],
            common_ROI=[roi_left,roi_right],common_ROI_is_echo_identity_verified=False,
            observed_center=qstats(observed),old_offset=oldstat,new_offset=newstat,
            absolute_median_offset_improved=bool(abs(newstat['median'])<abs(oldstat['median'])),
            old_common_ROI_energy_coverage_median=float(np.nanmedian(oldenergy)),
            new_common_ROI_energy_coverage_median=float(np.nanmedian(newenergy)),
            common_ROI_energy_to_front_median=float(np.median(roi_front_ratio)),
            ROI_truncated_count=int(np.sum(truncated)),first_interface=k==1,old_E_pass_claimed=False)
        rows.append(r)
        np.savez_compressed(output/'D_waveforms'/(prefix+'.npz'),sample_rows=ids,source_rows=expected_source,
            raw=np.concatenate(flats),envelope=np.concatenate(envflat),offsets=np.array(record_offsets),lengths=np.array(record_lengths),
            observed_center=observed,old_offset=oldoff,new_offset=newoff,old_coverage=np.array(oldenergy),new_coverage=np.array(newenergy),
            ROI_front_energy_ratio=np.array(roi_front_ratio),ROI_truncated=np.array(truncated),
            old_first_trough=panel[prefix+'_observed'])
        print(json.dumps(r,ensure_ascii=True),flush=True)
    main=[r for r in rows if r['k']>=2]
    summary=dict(rows=rows,scope='Reused F1 training defect development panel, no new labels opened, not independent validation',
        total_records=sum(r['selected'] for r in rows),main_rows=len(main),main_rows_abs_offset_improved=sum(r['absolute_median_offset_improved'] for r in main),
        old_E_pass_claimed=False,nominal_c_or_geometry_tuned=False,training_started=False,
        center_source_catalog_sha256=digest(catalog_path),old_panel_sha256=digest(old_d/'validation_samples.npz'))
    save(output/'D_reference_diagnostic.json',summary)
    save(output/'access_audit.json',dict(status='PASS',opened_source_arrays=access,allowed_source_arrays=sorted(allowed),
        fresh_label_files_opened=0,held_out_label_files_opened=0,calibration_labels_used=False,
        reused_D_identity_file=str(old_d/'validation_samples.npz'),RF_sources_modified=False))
    save(output/'decision.json',dict(status='REFERENCE_CANDIDATE_DIAGNOSED',main_rows=len(main),
         improved=sum(r['absolute_median_offset_improved'] for r in main),training_permitted=False,
         echo_identity_verified=False,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
    print(json.dumps(dict(status='REFERENCE_AUDIT_COMPLETE',main_improved=summary['main_rows_abs_offset_improved'],main_rows=len(main)),ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--old-D',dest='old_d',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.catalog,a.old_d,a.output)
