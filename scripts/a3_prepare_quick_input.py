"""Build a bounded F1 training-flow input and defect-only timing evidence."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from a3_ultrasound.data.quick_input import physical_manifest,transform_trace
from a3_ultrasound.data.time_prior import derive_time_prior_samples

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def excluded_regions(row,z,review):
    mask=np.zeros(len(z['sample_id']),dtype=bool)
    if row['group']!='TCL3': return mask,[]
    batch=int(row['entity'].split('-')[-1])
    targets=[q for q in review if int(q['batch'])==batch]
    if not targets:return mask,[]
    positions=Path(str(z['source_label_path'].item())).parent/'final_positions.json'
    data=json.loads(positions.read_text(encoding='utf-8'))
    xy=z['global_design_xy_mm']
    used=[]
    for q in targets:
        item=next(s for s in data['slots'] if s['id']==q['slot'])
        local=np.zeros(len(mask),dtype=bool)
        for field in ['nominal_slot','located_slot']:
            slot=item[field];center=np.array([slot['design_x'],slot['design_y']])
            vertices=np.asarray(slot['design_vertices'])
            if vertices.size:
                radius=float(np.linalg.norm(vertices-center,axis=1).max())
            elif slot['shape']=='圆槽':radius=slot['size']/2
            else:radius=slot['size']/np.sqrt(2)
            radius+=float(item['margin_mm'])
            local|=np.linalg.norm(xy-center,axis=1)<=radius
        mask|=local
        used.append({'slot':q['slot'],'records':int(local.sum()),'rule':'union of nominal/located enclosing circles + existing registration margin; entire A-scan excluded from training'})
    return mask,used

def run(config_path,output):
    candidate=json.loads(config_path.read_text(encoding='utf-8'))
    if candidate.get('status','').startswith('RETIRED'):
        raise ValueError('This configuration was retired by the user; select its replacement.')
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(config_path.read_text(encoding='utf-8'));save(output/'config.json',config)
    rows=json.loads(Path(config['specimen_inventory']).read_text(encoding='utf-8'))
    review=list(csv.DictReader(Path(config['review_locations']).open(encoding='utf-8-sig')))
    split=json.loads(Path(config['split_draft']).read_text(encoding='utf-8'))
    fold=next(f for f in split['folds'] if f['fold']==config['smoke']['fold'])
    save(output/'split_scope.json',{'fold':fold,'used_for_learning':fold['train'],'test_loaded_for_learning':False,
        'fold_count_provenance':'Counts inside fold are inherited from the earlier expanded-pool split draft; current clean counts are in preparation.json, not these historical counts.',
        'timing_qa_scope':'all 13 entities, designed defects only, fixed candidate; no parameter optimization'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entity_reports=[];smoke=[];qa=[]
    rng=np.random.default_rng(config['time_qa']['seed'])
    for row in rows:
        entity=row['entity'];dest=output/entity;dest.mkdir()
        with np.load(row['pool'],allow_pickle=False) as source:
            z={k:source[k] for k in source.files}
        region_mask,regions=excluded_regions(row,z,review)
        clean=z['design_complete_mask'] & z['reliable_complete_mask'] & ~z['uncertain_interface_mask'].any(1) & ~region_mask
        assert np.array_equal(z['y_design'][clean],z['reliable_y'][clean])
        ids=np.flatnonzero(clean)
        np.savez_compressed(dest/'training_eligibility.npz',eligible=clean,review_region_excluded=region_mask,
                            source_row=z['source_row'],sample_id=z['sample_id'])
        length=z['valid_length'] if 'valid_length' in z else z['lengths']
        rfpath=Path(row['index']).parent/'rf_lp20_float32.npy'
        rf=np.load(rfpath,mmap_mode='r')
        def trace_at(i):
            off=int(z['offsets'][i]);return np.asarray(rf[off:off+int(length[i])])
        k=int(row['K']);fig,axes=plt.subplots(k,1,figsize=(12,3*k),squeeze=False,constrained_layout=True)
        coverage=[]
        for interface in range(k):
            candidates=ids[z['y_design'][ids,interface]==1]
            # Prefer unshadowed single-positive states without using signal amplitude.
            single=candidates[z['y_design'][candidates].sum(1)==1]
            if len(single):candidates=single
            if not len(candidates):raise RuntimeError(f'No eligible defect QA records: {entity} I{interface+1}')
            chosen=np.sort(rng.choice(candidates,min(len(candidates),config['time_qa']['per_interface']),replace=False))
            traces=[];peaks=[];energies=[];candidate_ids=[];tail_reports=[]
            for i in chosen:
                trace=trace_at(i);x,meta,report=transform_trace(trace,row,config)
                tau=report['nominal_tof_samples'][interface];half=report['halfwidth_samples']
                search=float(config['time_qa']['search_halfwidth_us'])*1e-6*config['sampling_rate_hz']
                spacing=2*row['layer_mm']*.001/config['sound_speed_mps']['AL' if row['group']=='AL' else 'TC4']*config['sampling_rate_hz']
                # Broad candidate search is diagnostic only, never used to change metadata or eligibility.
                search=min(search,.48*spacing)
                lo=max(0,int(np.ceil(tau-search)));hi=min(len(trace),int(np.floor(tau+search))+1)
                centered=trace.astype(float)-float(np.median(trace[:min(16,len(trace))]))
                envelope=np.sqrt(np.maximum(0,gaussian_filter1d(centered**2,4)))
                peak=lo+int(envelope[lo:hi].argmax())
                wl=max(0,int(np.ceil(tau-half)));wh=min(len(trace),int(np.floor(tau+half))+1)
                fraction=float(np.sum(centered[wl:wh]**2)/max(np.sum(centered[lo:hi]**2),1e-20))
                peaks.append(peak);energies.append(fraction);traces.append(trace.copy()/np.float32(131));tail_reports.append(report)
                candidate_ids.append(f'{entity}:{int(z["source_row"][i])}')
            peak_offset=(np.asarray(peaks)-tau)/config['sampling_rate_hz']*1e6
            within=np.abs(peak_offset)<=config['window_total_us']/2
            entry={'entity':entity,'interface':interface+1,'samples':len(chosen),'sample_ids':candidate_ids,
                'pool_indices':chosen.tolist(),'candidate_peak_in_window':int(within.sum()),
                'candidate_peak_offset_us_median':float(np.median(peak_offset)),
                'window_energy_fraction_median':float(np.median(energies)),
                'interpretation':'candidate diagnostic, not echo identity ground truth or timebase calibration',
                'nominal_tof_samples':tau,'tail_reports':tail_reports}
            coverage.append(entry);qa.append(entry)
            ax=axes[interface,0]
            show=traces[0];t=np.arange(len(show))/config['sampling_rate_hz']*1e6
            ax.plot(t,show,lw=.65,color='#254b74',label='existing RF / 131')
            center=tau/config['sampling_rate_hz']*1e6
            half_us=config['window_total_us']/2
            ax.axvspan(center-half_us,center+half_us,color='#e38b2c',alpha=.25,label=f'Local ±{half_us:g} us')
            ax.axvline(center,color='#a75a00',ls='--')
            stop=tail_reports[0]['valid_length']/config['sampling_rate_hz']*1e6
            ax.axvspan(stop,len(show)/config['sampling_rate_hz']*1e6,color='gray',alpha=.2,label='masked tail')
            display={'厚1':'AL-thick1','厚2':'AL-thick2','薄':'AL-thin'}.get(entity,entity)
            ax.set(title=f'{display} I{interface+1}; source row {z["source_row"][chosen[0]]}; zero-origin candidate',xlabel='Time from cropped origin (us)',ylabel='signed RF / 131')
            ax.legend(loc='upper right',fontsize=7)
        fig.savefig(dest/'defect_windows.png',dpi=130);plt.close(fig)
        save(dest/'defect_timing_qa.json',coverage)
        positives=ids[z['y_design'][ids].any(1)];normals=ids[~z['y_design'][ids].any(1)]
        if entity in fold['train']:
            sampled=np.r_[rng.choice(positives,min(len(positives),config['smoke']['per_entity_positive']),replace=False),
                           rng.choice(normals,min(len(normals),config['smoke']['per_entity_normal']),replace=False)]
            for i in sampled:
                x,meta,report=transform_trace(trace_at(i),row,config)
                y=np.zeros(6,dtype=np.int64);y[:k]=z['reliable_y'][i]
                mask=np.arange(6)<k
                smoke.append((x,meta,y,mask,f'{entity}:{int(z["source_row"][i])}',entity))
        entity_reports.append({'entity':entity,'records':len(clean),'eligible_clean':int(clean.sum()),
            'eligible_positive':len(positives),'eligible_normal':len(normals),'review_regions':regions,
            'normal_records_used_for_time_qa':0,'rf_source':str(rfpath),'K':k,
            'manifest':physical_manifest(row,config)})
        del rf,z
    lengths=np.array([len(r[0]) for r in smoke]);padded=np.zeros((len(smoke),int(lengths.max())),dtype=np.float32)
    for i,r in enumerate(smoke):padded[i,:len(r[0])]=r[0]
    np.savez_compressed(output/'smoke_input.npz',x_raw=padded,length_mask=np.arange(padded.shape[1])[None]<lengths[:,None],
        interface_meta=np.stack([r[1] for r in smoke]),y=np.stack([r[2] for r in smoke]),
        interface_mask=np.stack([r[3] for r in smoke]),label_mask=np.stack([r[3] for r in smoke]),
        sample_ids=np.array([r[4] for r in smoke]),specimen_ids=np.array([r[5] for r in smoke]))
    assert set(r[5] for r in smoke)<=set(fold['train'])
    assert len(set(r[4] for r in smoke))==len(smoke)
    save(output/'preparation.json',{'status':'ENGINEERING_INPUT_BUILT_PHYSICAL_ACCEPTANCE_PENDING',
        'entities':entity_reports,'smoke_records':len(smoke),'normal_records_used_for_time_qa':0,
        'candidate_peak_checks':[{k:v for k,v in r.items() if k not in ['sample_ids','pool_indices','tail_reports']} for r in qa],
        'qa_is_not_training_score':True,'formal_training_permitted':False})
    print(json.dumps({'prepared':len(smoke),'entities':len(rows),'output':str(output)},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.config,args.output)
