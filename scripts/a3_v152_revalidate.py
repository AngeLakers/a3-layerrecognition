"""Paired fixed-template validation and one global all-board selection."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def summary(values):
    if not values:return dict(count=0,median=None,iqr=None,within80=None)
    x=np.asarray(values);q=np.quantile(x,[.25,.5,.75]);return dict(count=len(x),median=float(q[1]),iqr=float(q[2]-q[0]),within80=float(np.mean(abs(x)<=80)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fixed-cohort',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.data.board_timebase import adoption_decision
    from a3_ultrasound.data.rf_observation import observed_peak
    from a3_ultrasound.data.window_diagnostics import diagnostic_envelope
    cfg=load(a.root/'local_config.json');catpath=Path(cfg['catalog']);cat=load(catpath)
    rf=load(a.root/'rf_calibration/rf_only_calibration.json');cal={r['board']:r for r in rf['rows']}
    gates_path=a.root/'observation_gates.json';gates=load(gates_path)
    assert gates['labels_read'] is False and gates['fixed_before_D_label_access']
    assert gates['protocol_sha256']==sha(a.root/'PROTOCOL.md')
    observer={(r['board'],g['k']):g for r in gates['rows'] for g in r['gates']}
    cohort=load(a.fixed_cohort);train=cfg['folds'][0]['train'];assert cohort['entities']==train
    fixed=cohort['records'];assert len(fixed)==384 and all(r['board'] in train and r['usable'] for r in fixed)
    from collections import Counter
    assert len({(r['board'],r['k'],r['source_row'],r['sample_row']) for r in fixed})==384
    assert set(Counter((r['board'],r['k']) for r in fixed).values())=={24}
    deep=[r for r in fixed if r['k']>=2];assert len(deep)==216 and len({(r['board'],r['k']) for r in deep})==9
    assert len({(r['board'],r['k']) for r in fixed})==16
    shards={};access=[]
    for name in train:
        e=next(r for r in cat['entities'] if r['entity']==name);base=catpath.parent/e['index_dir']
        ar={key:np.load(base/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('source_row','offsets','lengths','y','label_mask','eligible')}
        shards[name]=(ar,np.load(e['rf_path'],mmap_mode='r',allow_pickle=False))
        access += [dict(board=name,field=key,path=str(base/(key+'.npy')),sha256=sha(base/(key+'.npy'))) for key in ('y','label_mask','eligible')]
    records=[];cache=a.root/'validation_RF';cache.mkdir(exist_ok=False)
    for r in fixed:
        name=r['board'];k=r['k'];i=r['sample_row'];ar,rawflat=shards[name]
        assert ar['source_row'][i]==r['source_row'] and ar['eligible'][i] and ar['label_mask'][i,k-1] and ar['y'][i,k-1]==1
        off=int(ar['offsets'][i]);n=int(ar['lengths'][i]);raw=np.asarray(rawflat[off:off+n]);env=diagnostic_envelope(raw)
        nom=cal[name]['nominal_tau'][k-1];candidate=cal[name]['candidate_tau'][k-1]
        gate=observer.get((name,k));got=observed_peak(env,gate) if gate else dict(valid=False,reason='missing_RF_template_packet')
        lo=int(np.ceil(nom-130));hi=int(np.floor(nom+130));legacy_peak=lo+int(np.argmax(env[lo:hi+1]))
        path=cache/f'{name}_k{k}_{r["source_row"]}.npz';np.savez_compressed(path,RF=raw)
        one=dict(board=name,k=k,source_row=r['source_row'],sample_row=i,nominal_tau=nom,calibrated_tau=candidate,
            valid=got['valid'],reason=got.get('reason'),peak=got.get('peak'),search_edge=got.get('search_edge',False),
            low_amplitude=got.get('amplitude',0.)<3 if got['valid'] else None,amplitude=got.get('amplitude'),
            old_offset=got['peak']-nom if got['valid'] else None,
            calibrated_offset=got['peak']-candidate if got['valid'] else None,
            gate=gate,legacy_nominal_gate_peak=legacy_peak,legacy_old_offset=legacy_peak-nom,
            legacy_calibrated_offset=legacy_peak-candidate,layer_mm=r['layer_mm'],
            cache_path=str(path),RF_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        records.append(one)
    assert [(r['board'],r['k'],r['source_row'],r['sample_row']) for r in records]==[(r['board'],r['k'],r['source_row'],r['sample_row']) for r in fixed]
    good=[r for r in records if r['valid']]
    decision=(adoption_decision(good) if any(r['k']>=2 for r in good) else
        dict(adopt_calibration=False,cells=[],deep_records=0,pooled_absolute_offset_median_old=None,
             pooled_absolute_offset_median_calibrated=None,pooled_improves=False,all_cells_within20=False,
             scope='One decision applied to all13 boards'))
    all_deep=len([r for r in good if r['k']>=2])==216 and len(decision['cells'])==9
    decision.update(all_fixed_deep_records_observed=all_deep,requested_deep_records=216)
    decision['adopt_calibration'] &= all_deep
    rows=[]
    for name,k in sorted({(r['board'],r['k']) for r in records}):
        group=[r for r in records if r['board']==name and r['k']==k];valid=[r for r in group if r['valid']]
        rows.append(dict(board=name,k=k,requested=len(group),records=len(valid),failed=len(group)-len(valid),
            old=summary([r['old_offset'] for r in valid]),calibrated=summary([r['calibrated_offset'] for r in valid]),
            search_edge=sum(r['search_edge'] for r in valid),low_amplitude=sum(bool(r['low_amplitude']) for r in valid),
            shallow_3mm=k==1 and group[0]['layer_mm']==3))
    table=dict(version='v1.5.2-selected-timebase',rows=[dict(r,
        selected_tau=r['candidate_tau'] if decision['adopt_calibration'] else r['nominal_tau'],
        selected_bottom=r['t_bottom']['median'] if decision['adopt_calibration'] else r['nominal_bottom'],
        selected_mode='calibrated' if decision['adopt_calibration'] else 'nominal_fallback') for r in rf['rows']],
        global_decision=decision,rf_only_calibration_sha256=sha(a.root/'rf_calibration/rf_only_calibration.json'),
        observation_gates_sha256=sha(gates_path),halfwidth_samples=80,gaussian_sigma_samples=40,gaussian_strength=.5)
    save(a.root/'selected_timebase.json',table)
    save(a.root/'validation.json',dict(rows=rows,records=records,decision=decision,
        definition='One peak per trace from frozen RF-only template gate; same observed peak compared to nominal and calibrated targets; candidate not physical truth',
        F1_train_label_access=access,heldout_labels_read=False,fixed_cohort_sha256=sha(a.fixed_cohort),
        fixed_observation_gates_sha256=sha(gates_path),training_called=False))
    cfg.update(timebase_calibration_path=str(a.root/'selected_timebase.json'),timebase_calibration_sha256=sha(a.root/'selected_timebase.json'),
        physical_statistics_path=str(a.root/'physical_statistics/normalization.json'),physical_statistics_sha256='PENDING_REFIT',freeze_training_permitted=False)
    save(a.root/'local_config.json',cfg);print(json.dumps(decision,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
