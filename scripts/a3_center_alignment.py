"""Fit label-free packet centers on all RF and build two reversible RF views."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.fft import set_workers

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.data.center_alignment import packet_centers


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    if not len(x):
        return dict(count=0,median=None,iqr=None,q01=None,q25=None,q75=None,q99=None)
    q=np.quantile(x,[.01,.25,.5,.75,.99])
    return dict(count=len(x),median=float(q[2]),iqr=float(q[3]-q[1]),q01=float(q[0]),
                q25=float(q[1]),q75=float(q[3]),q99=float(q[4]))


def delta_stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):
        return dict(count=0,median=None,q95_abs=None,max_abs=None,fraction_abs_gt5=None)
    return dict(count=len(x),median=float(np.median(x)),q95_abs=float(np.quantile(np.abs(x),.95)),
                max_abs=float(np.abs(x).max()),fraction_abs_gt5=float(np.mean(np.abs(x)>5)))


def run(source,old_root,output):
    spec=json.loads(source.read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False)
    (output/'boards').mkdir();(output/'qa').mkdir()
    allowed={str(Path(b[key]).resolve()) for b in spec['boards']
             for key in ('rf_path','offsets_path','lengths_path','source_row_path')}
    original_load=np.load;access=[]

    def source_load(path,**kwargs):
        name=str(Path(path).resolve())
        assert name in allowed and kwargs.get('mmap_mode')=='r','RF-only source read assertion.'
        access.append(name)
        return original_load(path,**kwargs)

    rows=[];catalog_boards=[]
    for b in spec['boards']:
        p=Path(b['rf_path']);before=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        rf=source_load(p,mmap_mode='r',allow_pickle=False)
        off=source_load(b['offsets_path'],mmap_mode='r',allow_pickle=False)
        lengths=source_load(b['lengths_path'],mmap_mode='r',allow_pickle=False)
        source_rows=source_load(b['source_row_path'],mmap_mode='r',allow_pickle=False)
        assert len(off)==len(lengths)==len(source_rows)==b['records'] and np.all(lengths>=512)
        assert np.all(off>=0) and np.all(off+lengths<=len(rf)) and rf.dtype==np.float32
        n=len(off);center=np.full(n,np.nan,np.float32);peak=center.copy();rawcenter=center.copy()
        boundary=np.zeros(n,bool);failed=np.zeros(n,bool);edge=center.copy()
        for start in range(0,n,2048):
            ids=np.arange(start,min(n,start+2048))
            context=np.asarray(rf[off[ids,None]+np.arange(512)],np.float32)
            with set_workers(4):
                r=packet_centers(context)
            center[ids]=r.envelope_energy;peak[ids]=r.envelope_peak;rawcenter[ids]=r.rf_energy
            boundary[ids]=r.boundary_sensitive;failed[ids]=r.failed;edge[ids]=r.right_edge_energy_fraction
        assert np.isfinite(center).any(),'No usable packet center for this board.'
        board_stats=stats(center);median=board_stats['median'];shift=int(np.rint(256-median))
        assert shift>=0
        finite=np.isfinite(center)
        board_after=center[finite].astype(float)+shift-256
        record_shifts=np.where(finite,np.rint(256-np.nan_to_num(center.astype(np.float64),nan=median)),shift).astype(np.int16)
        record_after=center[finite].astype(float)+record_shifts[finite]-256
        assert np.all(record_shifts>=0) and np.all(np.abs(record_after)<=.50001)
        center_path=output/'boards'/(b['board']+'_centers.npy');np.save(center_path,center)
        np.save(output/'boards'/(b['board']+'_record_shifts.npy'),record_shifts)
        np.savez_compressed(output/'boards'/(b['board']+'_diagnostics.npz'),envelope_peak=peak,
                            rf_energy=rawcenter,failed=failed,boundary_sensitive=boundary,
                            right_edge_energy_fraction=edge)
        qa=np.linspace(0,n-1,256,dtype=np.int64)
        context=np.asarray(rf[off[qa,None]+np.arange(512)],np.float32)
        sensitivity={}
        for gate,pad in ((240,256),(200,128),(200,512)):
            measured=packet_centers(context,gate_end=gate,pad=pad).envelope_energy
            sensitivity[f'gate{gate}_pad{pad}']=delta_stats(measured-center[qa])
        np.savez_compressed(output/'qa'/(b['board']+'.npz'),sample_rows=qa,
                            source_rows=np.asarray(source_rows[qa]),raw_context=context,
                            envelope_center=center[qa],envelope_peak=peak[qa],rf_energy=rawcenter[qa],
                            record_shifts=record_shifts[qa],source_lengths=np.asarray(lengths[qa]))
        with original_load(old_root/(b['board']+'.npz'),allow_pickle=False) as z:
            old_t0=z['t0']
        assert len(old_t0)==n
        old_stats=stats(old_t0[old_t0>=0])
        after=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        assert before==after,'RF source changed during measurement.'
        interval=2*b['layer_mm']*1e-3*spec['sampling_rate_hz']/b['nominal_c_mps']
        nominal_bottom=256+(b['K']+1)*interval
        aligned_lengths=lengths+shift
        row=dict(board=b['board'],records=n,center=board_stats,envelope_peak=stats(peak),rf_energy=stats(rawcenter),
                 old_first_negative=old_stats,board_shift_samples=shift,
                 board_after=stats(board_after),record_after=stats(record_after),
                 board_within10_fraction=float(np.mean(np.abs(board_after)<=10)),
                 board_within35_fraction=float(np.mean(np.abs(board_after)<=35)),
                 record_within1_fraction=float(np.mean(np.abs(record_after)<=1)),
                 failed_count=int(failed.sum()),boundary_sensitive_count=int(boundary.sum()),
                 boundary_sensitive_fraction=float(boundary.mean()),sensitivity=sensitivity,
                 nominal_bottom_outside_record_count=int(np.sum(nominal_bottom>=aligned_lengths)),
                 nominal_bottom_search_right_outside_count=int(np.sum(nominal_bottom+.65e-6*spec['sampling_rate_hz']>=aligned_lengths)),
                 rf_source_stat=before)
        rows.append(row)
        catalog_boards.append(dict(**b,rf_source_stat=before,
            index_sha256={key:digest(b[key+'_path']) for key in ('offsets','lengths','source_row')},
            centers_path=str(center_path.relative_to(output)).replace('\\','/'),centers_sha256=digest(center_path),
            board_reference_samples=median,board_shift_samples=shift,
            nominal_interfaces_relative_samples=[k*interval for k in range(1,b['K']+1)],
            nominal_backwall_relative_samples=(b['K']+1)*interval))
        save(output/'partial_results.json',dict(boards=rows,complete=False))
        print(json.dumps(dict(board=b['board'],records=n,center_median=median,center_iqr=board_stats['iqr'],
              old_valley_iqr=old_stats['iqr'],peak_iqr=row['envelope_peak']['iqr'],
              board_within10=row['board_within10_fraction'],boundary_fraction=row['boundary_sensitive_fraction'],
              gate_sensitivity_q95=sensitivity['gate240_pad256']['q95_abs']),ensure_ascii=True),flush=True)
    medians=[r['center']['median'] for r in rows]
    post=[r['board_after']['median'] for r in rows]
    results=dict(schema='a3-center-alignment-analysis-v1',boards=rows,total_records=sum(b['records'] for b in rows),
                 before_board_median_range_samples=max(medians)-min(medians),
                 after_board_median_range_samples=max(post)-min(post),
                 reference_samples=256,center_definition='Hilbert-envelope energy centroid in inclusive [0,200]',
                 labels_read=False,raw_rf_modified=False,training_started=False,outer_evaluation_opened=False,
                 source_spec_sha256=digest(source),calibration_refitted_sound_speed=False,
                 created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    save(output/'results.json',results)
    save(output/'aligned_catalog.json',dict(schema='a3-center-aligned-rf-view-v1',reference_samples=256,
            sampling_rate_hz=spec['sampling_rate_hz'],center_definition=results['center_definition'],
            modes=['board','record'],default_mode='board',recommendation_status='pending_ARIS_analysis',
            source_spec_sha256=digest(source),boards=catalog_boards,
            preserve_all_source_samples=True,padding_is_measured_data=False,
            nominal_positions_status='Geometry-only coordinates; echo identity/phase and sound-speed not calibrated',
            training_integrated=False))
    save(output/'rf_only_access_audit.json',dict(status='PASS',allowed_source_arrays=sorted(allowed),
            opened_source_arrays=access,forbidden_label_access_count=0,source_array_mode='r',
            old_valley_arrays='RF-only saved timebase arrays; no D validation selections loaded'))
    print(json.dumps(dict(status='CENTER_ALIGNMENT_ASSETS_CREATED',records=results['total_records'],
            boards=len(rows),before_board_range=results['before_board_median_range_samples'],
            after_board_range=results['after_board_median_range_samples']),ensure_ascii=True),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-spec',type=Path,required=True)
    p.add_argument('--old-rf-timing',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source_spec,a.old_rf_timing,a.output)


if __name__=='__main__':
    main()
