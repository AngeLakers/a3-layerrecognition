"""Freeze a new label catalog and common 60-cell sampling without touching V1/RF."""
from pathlib import Path
import argparse
import csv
import copy
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a3_ultrasound.data.label_revision import admit_design_interiors
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.data.validation_sampling import select_validation_rows
from a3_ultrasound.stream_training import catalog_signature, validate_fold


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-catalog',type=Path,required=True)
    p.add_argument('--base-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conservative-same-outline',action='store_true',help='Historical r2 only; current user-approved release does not use this extra screen.')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    old=json.loads(args.old_catalog.read_text('utf8'));cfg=json.loads(args.base_config.read_text('utf8'))
    catalog=copy.deepcopy(old);rows=[];sources=[]
    for e in catalog['entities']:
        src=args.old_catalog.parent/e['index_dir'];dest=out/'catalog'/e['index_dir'];dest.mkdir(parents=True)
        with np.load(e['source_pool']) as pack:
            source={k:pack[k] for k in ('reliable_y','reliable_label_mask','y_design','design_label_mask','signal_quality_candidate','source_row')}
            label_source=Path(str(pack['source_label_path'].item()))
            position_xy=pack['global_design_xy_mm'] if 'global_design_xy_mm' in pack.files else None
        if sha(e['source_pool'])!=e['pool_sha256']:raise ValueError('Source pool drift.')
        y,mask,admitted=admit_design_interiors(source['reliable_y'],source['reliable_label_mask'],source['y_design'],source['design_label_mask'],source['signal_quality_candidate'])
        protected=0
        if e['group']=='TCL3' and args.conservative_same_outline:
            positions=json.loads((label_source.parent/'final_positions.json').read_text('utf8'))['slots']
            for a in [r for r in positions if r['interface']==1]:
                s=a['nominal_slot']
                matches=[r for r in positions if r['interface']==2 and
                         r['nominal_slot']['shape']==s['shape'] and r['nominal_slot']['size']==s['size'] and
                         r['nominal_slot']['design_x']==s['design_x'] and r['nominal_slot']['design_y']==s['design_y'] and
                         r['nominal_slot']['design_vertices']==s['design_vertices']]
                if not matches:continue
                # Distinct design sites are >=24 mm apart; use the original enclosing shape plus its guard.
                vertices=np.asarray(s['design_vertices']);center=np.array([s['design_x'],s['design_y']])
                radius=np.linalg.norm(vertices-center,axis=1).max() if vertices.size else s['size']/2
                radius+=max(a['margin_mm'],matches[0]['margin_mm'])+max(np.linalg.norm(a['local_shift_design_mm']),np.linalg.norm(matches[0]['local_shift_design_mm']))
                local=np.linalg.norm(position_xy-center,axis=1)<=radius
                conflict=local&mask.all(1)&(y[:,0]!=y[:,1])
                revoke=conflict[:,None]&admitted
                protected+=int(revoke.sum());mask[revoke]=False;y[revoke]=-1;admitted[revoke]=False
                assert not (local&mask.all(1)&(y[:,0]!=y[:,1])).any()
        old_y=np.load(src/'y.npy');old_mask=np.load(src/'label_mask.npy');old_eligible=np.load(src/'eligible.npy')
        eligible=mask.all(1)
        assert np.array_equal(y[old_mask],old_y[old_mask]) and not np.any(old_mask&~mask)
        arrays={key:np.load(src/f'{key}.npy') for key in ('source_row','offsets','lengths','xy','quality')}
        assert np.array_equal(arrays['source_row'],source['source_row'])
        arrays.update(y=y,label_mask=mask,design_y=source['y_design'],design_mask=source['design_label_mask'],eligible=eligible)
        for key,value in arrays.items():np.save(dest/f'{key}.npy',value)
        weights=2**np.arange(e['K']-1,-1,-1)
        joint=np.where(eligible,np.maximum(y,0)@weights,-1)
        old_joint=np.where(old_eligible,np.maximum(old_y,0)@weights,-1)
        row=dict(entity=e['entity'],records=len(y),old_complete=int(old_eligible.sum()),new_complete=int(eligible.sum()),
                 new_complete_records=int((eligible&~old_eligible).sum()),old_known_values_changed=0,
                 admitted_design_positive_interface_values=int(admitted.sum()),
                 same_outline_conflict_admissions_withheld=protected,
                 restored_source_interface_values=int((source['reliable_label_mask']&~old_mask).sum()),
                 old_joint_counts={format(i,f"0{e['K']}b"):int((old_joint==i).sum()) for i in range(2**e['K'])},
                 new_joint_counts={format(i,f"0{e['K']}b"):int((joint==i).sum()) for i in range(2**e['K'])})
        rows.append(row);sources.append(dict(source_pool=e['source_pool'],pool_sha256=e['pool_sha256'],index=e['source_index'],index_sha256=e['index_sha256']))
        label_identity=hashlib.sha256((sha(dest/'y.npy')+sha(dest/'label_mask.npy')+sha(dest/'eligible.npy')).encode()).hexdigest()
        e.update(v1_pool_sha256=e['pool_sha256'],pool_sha256=label_identity,label_release='labels-v2-fixed-green-design-interiors-20261024',
                 train_eligible=int(eligible.sum()),positive=int((eligible&(y==1).any(1)).sum()))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    catalog['input_config']=dict(catalog['input_config'],id='a3-labels-v2-fixed-green-design-interiors-20261024',
        status='V2_FROZEN_PRELAUNCH',training_labels='design interiors admitted at fixed green registration; source quality, exterior buffer and unresolved design masks retained; prior nine position regions restored',
        label_version='V2',v1_catalog_signature=catalog_signature(old))
    catalog['training_label_basis']='V2 complete nominal design-supervised labels; not measured weld quality'
    catalog['evaluation_label_basis']='V2 known interface masks; partial labels only interface metrics; nominal secondary separate'
    save(out/'catalog/catalog.json',catalog);save(out/'label_changes.json',rows);save(out/'source_provenance.json',sources)
    cfg['catalog']=str(out/'catalog/catalog.json');cfg['seeds']=[23,41,59]
    cfg['heads']=['local_independent','vanilla_chain','proposed','local_pre_independent']
    cfg['frozen_selection_root']=str(out/'frozen_training_selections');Path(cfg['frozen_selection_root']).mkdir()
    cfg['validation_selection_root']=str(out/'validation_selection');Path(cfg['validation_selection_root']).mkdir()
    cfg.update(formal_training_permitted=False,outer_evaluation_permitted=False,label_version='V2',
               expected_runs=60,readiness_scope='User authorized V2 rebuild; execution gated on frozen preflight/review.')
    all_test=[];selection_hashes={}
    for fold in cfg['folds']:
        validate_fold(fold,catalog);all_test+=fold['test']
        for seed in cfg['seeds']:
            data=RFShardDataset(cfg['catalog'],fold['train'],training=True,seed=seed,normal_ratio=cfg['normal_to_positive'])
            target=Path(cfg['frozen_selection_root'])/f"{fold['fold']}_s{seed}.npy";np.save(target,data.refs)
            again=RFShardDataset(cfg['catalog'],fold['train'],training=True,seed=seed,normal_ratio=cfg['normal_to_positive'])
            assert np.array_equal(data.refs,again.refs)
            selection_hashes[target.name]=dict(sha256=sha(target),records=len(data),entities=fold['train'])
    assert len(all_test)==len(set(all_test))==len(catalog['entities'])
    registry={e['entity']:e for e in catalog['entities']}
    vm=dict(policy='all reliable positives + spatial mask/quality strata; one uniform sample per bin; integer inverse-probability weights',
            id='validation_compact_labels_v2_20261024',seed=20261023,target_per_entity=13334,catalog_signature=catalog_signature(catalog),folds={})
    for fold in cfg['folds']:
        selections=[];weights=[];refs=[];offset=0;reports=[]
        for s,name in enumerate(fold['validation']):
            e=registry[name];d=out/'catalog'/e['index_dir'];a={key:np.load(d/f'{key}.npy') for key in ('y','label_mask','xy','quality')}
            ids,w=select_validation_rows(a['y'],a['label_mask'],a['xy'],a['quality'],ordinal=e['ordinal'])
            selections.append(ids+offset);weights.append(w);refs.append(np.c_[np.full(len(ids),s,np.int64),ids]);offset+=len(a['y'])
            reports.append(dict(entity=name,population=len(a['y']),selected=len(ids),weight_sum=int(w.sum()),positive=int((a['label_mask']&(a['y']==1)).any(1).sum())))
        dest=Path(cfg['validation_selection_root'])/f"{fold['fold']}.npz"
        np.savez_compressed(dest,indices=np.concatenate(selections),refs=np.concatenate(refs),weights=np.concatenate(weights),entities=np.array(fold['validation']))
        vm['folds'][fold['fold']]=dict(sha256=sha(dest),entities=reports,population=offset,selected=sum(r['selected'] for r in reports))
    save(Path(cfg['validation_selection_root'])/'manifest.json',vm);save(out/'training_selection_manifest.json',selection_hashes)
    save(out/'local_training.json',cfg)
    save(out/'data_validation.json',dict(status='PASS',entities=13,records=sum(r['records'] for r in rows),
         old_known_values_changed=0,green_geometry_changed=False,raw_RF_changed=False,
         catalog_signature=catalog_signature(catalog),fold_entity_disjoint=True,test_entities_each_once=True,
         frozen_training_selections=len(selection_hashes),validation_folds=len(vm['folds']),
         old_complete=sum(r['old_complete'] for r in rows),new_complete=sum(r['new_complete'] for r in rows),
         admitted_positive_interface_values=sum(r['admitted_design_positive_interface_values'] for r in rows)))


if __name__=='__main__':main()
