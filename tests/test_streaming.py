import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.training import ModelConfig,build_model,evaluate,summarize_predictions,synthetic_datasets
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.stream_training import validate_fold,train


def test_stream_counts_equal_dense_metrics_with_partial_labels():
    data=synthetic_datasets(23,6)['train']
    mask=data.batch.label_mask.clone();mask[0,0]=False
    data=replace(data,batch=replace(data.batch,label_mask=mask))
    model=build_model(ModelConfig(backbone='tcn'))
    _,rows=evaluate(model,data,batch_size=2)
    a=PredictionAccumulator()
    for row in rows:a.add(row)
    expected=summarize_predictions(rows);got=a.result()
    for key in ['sample_count','labeled_interface_count','deepest_specimen_macro_f1','joint_nll','joint_map_accuracy','base_error_count','base_wrong_to_exact_right','base_right_to_exact_wrong','raw_logZ_max_abs']:
        assert got[key]==pytest.approx(expected[key])
    for entity in expected['deepest_per_specimen']:
        assert got['deepest_per_specimen'][entity]['confusion_matrix_0_1']==expected['deepest_per_specimen'][entity]['confusion_matrix_0_1']
        assert got['deepest_per_specimen'][entity]['brier']==pytest.approx(expected['deepest_per_specimen'][entity]['brier'])


@pytest.fixture
def toy_catalog(tmp_path):
    config=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_local_ready_v3.json').read_text(encoding='utf-8'))
    entries=[]
    for ordinal,(entity,k) in enumerate([('AL',2),('TC',3)]):
        dest=tmp_path/entity;dest.mkdir();n=12;length=1200
        y=np.zeros((n,k),dtype=np.int8);y[:3,0]=1
        lm=np.ones_like(y,dtype=bool);lm[2,0]=False
        eligible=lm.all(1)
        rf=np.zeros((n,length),np.float32);rf[:,787]=30
        np.save(dest/'rf.npy',rf.ravel())
        arrays={'source_row':np.arange(n)+100,'offsets':np.arange(n)*length,'lengths':np.full(n,length),
                'y':y,'label_mask':lm,'design_y':y,'design_mask':np.ones_like(lm),'eligible':eligible,
                'xy':np.column_stack((np.arange(n),np.arange(n)%3)),'quality':np.zeros(n,dtype=bool)}
        for name,value in arrays.items():np.save(dest/f'{name}.npy',value)
        entries.append({'entity':entity,'ordinal':ordinal,'group':'AL' if k==2 else 'TCL4','K':k,'layer_mm':3,
                        'index_dir':entity,'rf_path':str(dest/'rf.npy')})
    # Unavailable held-out RF must never be opened by a train-only subset.
    entries.append({'entity':'heldout','ordinal':2,'index_dir':'missing','rf_path':'MISSING'})
    path=tmp_path/'catalog.json';path.write_text(json.dumps({'entities':entries,'input_config':config}),encoding='utf-8')
    return path


def test_guarded_best_npz_and_summary_bind_config_and_commit_on_synthetic_data(toy_catalog,tmp_path):
    import hashlib
    cfg=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_stream_v3.json').read_text(encoding='utf-8'))
    cfg.update(catalog=str(toy_catalog),folds=[dict(fold='F1',train=['AL'],validation=['TC'],test=['heldout'])],
        formal_training_permitted=True,outer_evaluation_permitted=False,frozen_selection_root=str(tmp_path),
        save_best_correction_npz=True)
    cfg.pop('validation_selection_root',None)
    cfg['model'].update(backbone='tcn',rho=3.)
    cfg['training'].update(device='cpu',epochs=1,batch_size=4,selection_rule='guarded_joint',normal_fpr_limit=1.)
    data=RFShardDataset(toy_catalog,['AL'],training=True,seed=23)
    np.save(tmp_path/'F1_s23.npy',data.refs)
    path=tmp_path/'run.json';path.write_text(json.dumps(cfg));out=tmp_path/'run'
    result=train(path,out,fold_name='F1',head='proposed',seed=23)
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['config_sha256']==expected and result['commit_hash']
    for kind in ('guarded','fallback'):
        checkpoint=torch.load(out/f'best_{kind}.pt',map_location='cpu',weights_only=True)
        with np.load(out/f'best_{kind}_diagnostics.npz',allow_pickle=False) as pack:
            meta=json.loads(str(pack['metadata_json']))
            assert meta['epoch']==checkpoint['epoch']==1
            assert meta['config_sha256']==expected
            assert meta['commit_hash']==result['commit_hash']
            assert len(pack['g'])==24


def test_interleaved_guarded_fallback_checkpoint_npz_epochs(toy_catalog,tmp_path,monkeypatch):
    import a3_ultrasound.stream_training as runner
    cfg=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_stream_v3.json').read_text(encoding='utf-8'))
    cfg.update(catalog=str(toy_catalog),folds=[dict(fold='F1',train=['AL'],validation=['TC'],test=['heldout'])],
        formal_training_permitted=True,outer_evaluation_permitted=False,frozen_selection_root=str(tmp_path),save_best_correction_npz=True)
    cfg.pop('validation_selection_root',None)
    cfg['model'].update(backbone='tcn',rho=3.)
    cfg['training'].update(device='cpu',epochs=5,batch_size=4,selection_rule='guarded_joint',normal_fpr_limit=.02,min_epochs=15)
    data=RFShardDataset(toy_catalog,['AL'],training=True,seed=23);np.save(tmp_path/'F1_s23.npy',data.refs)
    original=runner.evaluate_stream
    scores={1:(.6,.01),2:(.7,.03),3:(.65,.01),4:(.69,.04),5:(.8,.05)}
    def controlled(*args,**kwargs):
        metrics=original(*args,**kwargs)
        if kwargs.get('phase')=='validation':
            score,fpr=scores[kwargs['epoch']]
            metrics['reliable'].update(positive_joint_map_accuracy=score,normal_joint_false_positive_rate=fpr,joint_nll=.4)
        return metrics
    monkeypatch.setattr(runner,'evaluate_stream',controlled)
    path=tmp_path/'run.json';path.write_text(json.dumps(cfg));out=tmp_path/'run'
    runner.train(path,out,fold_name='F1',head='proposed',seed=23)
    for kind,expected in [('guarded',3),('fallback',5)]:
        cp=torch.load(out/f'best_{kind}.pt',map_location='cpu',weights_only=True)
        with np.load(out/f'best_{kind}_diagnostics.npz',allow_pickle=False) as pack:
            assert json.loads(str(pack['metadata_json']))['epoch']==cp['epoch']==expected
    assert not (out/'screening_stop.json').exists()


def test_lazy_sampling_mixed_k_and_late_tail(toy_catalog):
    a=RFShardDataset(toy_catalog,['AL','TC'],training=True,seed=23,normal_ratio=3)
    b=RFShardDataset(toy_catalog,['AL','TC'],training=True,seed=23,normal_ratio=3)
    np.testing.assert_array_equal(a.refs,b.refs)
    assert len(a)==16
    batch=a.make_batch([0,8])
    assert batch.interface_mask.sum(1).tolist()==[2,3]
    assert batch.x_raw[0,787].item()==pytest.approx(30/131)
    assert (batch.interface_meta[:,:,3]*batch.length_mask.sum(1)[:,None])[batch.interface_mask].tolist()==pytest.approx([70]*5)
    assert not batch.x_raw[~batch.length_mask].any()
    assert all(a.identity(i)[0]!='AL:102' for i in range(len(a)))


def test_evaluation_keeps_natural_counts_and_unknown_mask(toy_catalog):
    a=RFShardDataset(toy_catalog,['AL'])
    assert len(a)==12
    batch=a.make_batch([2]);assert not batch.label_mask[0,0]
    assert a.annotations([2])[0]['quality_flagged'] is True


def test_v14_stage2_uses_actual_epoch_refs_and_never_opens_outer(toy_catalog,tmp_path):
    from a3_ultrasound.data.epoch_sampling import freeze_epochs
    from a3_ultrasound.stream_training import catalog_signature
    cfg=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_stream_v3.json').read_text(encoding='utf-8'))
    fold={'fold':'F1','train':['AL'],'validation':['TC'],'test':['heldout']}
    catalog=json.loads(toy_catalog.read_text());sig=catalog_signature(catalog)
    d=RFShardDataset(toy_catalog,['AL'],training=True,seed=23)
    eroot=tmp_path/'epochs';freeze_epochs(d,eroot,seed=23,epochs=1,catalog_signature=sig,positive_fraction=.2)
    results=[dict(status='COMPLETED_DIAGNOSTIC',epochs=15,rho=rho,shadow=shadow,seed=23,
        final_metrics=dict(correction_saturation_fraction_MAP_path=sat,positive_joint_map_accuracy=.8,
                           normal_joint_false_positive_rate=0.)) for rho,shadow,sat in [(1,False,.99),(3,False,.3),(3,True,.3)]]
    screen=tmp_path/'screen.json';screen.write_text(json.dumps(dict(status='COMPLETED',results=results,
        passed_candidates=['proposed_rho3','proposed_shadow_aux_rho3'],fold=fold,seed=23,catalog_signature=sig)))
    cfg.update(catalog=str(toy_catalog),folds=[fold],seeds=[23],heads=['proposed'],development_stage=2,
        development_training_permitted=True,formal_training_permitted=False,outer_evaluation_permitted=False,
        epoch_selection_root=str(eroot),stage1_result=str(screen),save_epoch_checkpoints=True)
    cfg.pop('validation_selection_root',None)
    cfg['model'].update(architecture_version='a3-v1.4',backbone_norm='group8',query_view='position2',rho=3.)
    cfg['training'].update(device='cpu',epochs=1,batch_size=4,patience=10,min_epochs=15,selection_rule='guarded_joint')
    path=tmp_path/'new.json';path.write_text(json.dumps(cfg))
    out=tmp_path/'stage2';result=train(path,out,fold_name='F1',head='proposed',seed=23)
    assert result['test_opened'] is False and result['formal_result'] is False
    assert len(np.load(out/'training_selection.npy'))==4
    np.testing.assert_array_equal(np.load(out/'training_selection.npy'),np.load(eroot/'epoch_01.npy'))
    resolved=json.loads((out/'resolved_config.json').read_text(encoding='utf-8'))
    assert resolved['selection_counts']=={'AL':4}
    assert (out/'best_fallback.pt').exists()


def test_frozen_validation_loader_rejects_training_and_wrong_refs(toy_catalog,tmp_path):
    from a3_ultrasound.data.validation_sampling import select_validation_rows
    data=RFShardDataset(toy_catalog,['AL'])
    arrays=data.shards[0]['arrays']
    ids,weights=select_validation_rows(arrays['y'],arrays['label_mask'],arrays['xy'],arrays['quality'],target=6)
    path=tmp_path/'selection.npz'
    np.savez(path,indices=ids,weights=weights,refs=data.refs[ids],entities=np.array(['AL']))
    data.use_frozen_evaluation_selection(path)
    assert len(data)==6 and sum(a['sample_weight'] for a in data.annotations(range(len(data))))==12
    with pytest.raises(ValueError,match='once'):data.use_frozen_evaluation_selection(path)
    with pytest.raises(ValueError,match='evaluation dataset'):
        RFShardDataset(toy_catalog,['AL'],training=True).use_frozen_evaluation_selection(path)
    wrong=RFShardDataset(toy_catalog,['AL'])
    np.savez(path,indices=ids,weights=weights,refs=wrong.refs[ids]+1,entities=np.array(['AL']))
    with pytest.raises(ValueError,match='refs'):wrong.use_frozen_evaluation_selection(path)


def test_split_and_formal_gate_fail_before_rf_access(toy_catalog,tmp_path):
    catalog=json.loads(toy_catalog.read_text())
    fold={'fold':'F1','train':['AL'],'validation':['TC'],'test':['heldout']}
    validate_fold(fold,catalog)
    with pytest.raises(ValueError):validate_fold(dict(fold,test=['AL']),catalog)
    cfg={'catalog':str(toy_catalog),'folds':[fold],'seeds':[23],'heads':['proposed'],'formal_training_permitted':False}
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='3080'):
        train(p,tmp_path/'no_output',fold_name='F1',head='proposed',seed=23)
    assert not (tmp_path/'no_output').exists()


@pytest.mark.parametrize('sample_validation',[False,True])
def test_formal_train_validation_and_separate_test_on_synthetic_fixture(toy_catalog,tmp_path,monkeypatch,sample_validation):
    import a3_ultrasound.stream_training as runner
    catalog=json.loads(toy_catalog.read_text())
    catalog['entities'][2]=dict(catalog['entities'][0],entity='heldout',ordinal=2)
    toy_catalog.write_text(json.dumps(catalog))
    cfg=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_stream_v3.json').read_text(encoding='utf-8'))
    cfg.update(catalog=str(toy_catalog),folds=[{'fold':'F1','train':['AL'],'validation':['TC'],'test':['heldout']}],
               formal_training_permitted=True,outer_evaluation_permitted=False,frozen_selection_root=str(tmp_path))
    cfg['training'].update(device='cpu',epochs=1,batch_size=4,patience=1)
    cfg.update(protocol_signature='frozen-synthetic-protocol',label_version='V2',save_epoch_checkpoints=True)
    cfg['model']['backbone']='tcn'
    selection=RFShardDataset(toy_catalog,['AL'],training=True,seed=23)
    np.save(tmp_path/'F1_s23.npy',selection.refs)
    if sample_validation:
        import hashlib
        from a3_ultrasound.data.validation_sampling import select_validation_rows
        validation=RFShardDataset(toy_catalog,['TC']);arrays=validation.shards[0]['arrays']
        ids,w=select_validation_rows(arrays['y'],arrays['label_mask'],arrays['xy'],arrays['quality'],target=6)
        selection_root=tmp_path/'validation_selection';selection_root.mkdir()
        selection_path=selection_root/'F1.npz'
        np.savez(selection_path,indices=ids,weights=w,refs=validation.refs[ids],entities=np.array(['TC']))
        manifest={'catalog_signature':runner.catalog_signature(catalog),'policy':'synthetic fixture','seed':20261023,
                  'folds':{'F1':{'sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest()}}}
        (selection_root/'manifest.json').write_text(json.dumps(manifest))
        cfg['validation_selection_root']=str(selection_root)
    path=tmp_path/'run.json';path.write_text(json.dumps(cfg))
    opened=[];original=runner.RFShardDataset
    def track(path,entities,**kwargs):
        opened.append(list(entities));return original(path,entities,**kwargs)
    monkeypatch.setattr(runner,'RFShardDataset',track)
    result=runner.train(path,tmp_path/'train',fold_name='F1',head='proposed',seed=23)
    assert result['status']=='COMPLETED' and opened==[['AL'],['TC']]
    saved=__import__('torch').load(tmp_path/'train/epoch_01.pt',map_location='cpu',weights_only=True)
    assert saved['epoch']==1 and saved['resolved_config']['protocol_signature']=='frozen-synthetic-protocol'
    assert saved['updates_total']>0 and 'shuffle_rng_state' in saved and 'optimizer_state_dict' in saved
    assert result['metrics']['reliable']['evaluated_sample_count']==(6 if sample_validation else 12)
    assert result['metrics']['reliable']['sample_count']==12
    # The post-training audit restores the full validation population and never
    # opens outer RF or reselects a checkpoint from its full-population score.
    import a3_ultrasound.validation_audit as audit
    monkeypatch.setattr(audit,'RFShardDataset',track)
    opened.clear()
    audited=audit.audit_validation(path,tmp_path/'train',tmp_path/'audit')
    assert opened==[['TC']]
    assert audited['evaluated_records']==12
    assert audited['compact_evaluated_records']==(6 if sample_validation else 12)
    assert not audited['metrics']['reliable']['sampling_weighted']
    assert audited['checkpoint_reselected'] is False and audited['outer_opened'] is False
    assert audited['selected_epoch']==result['best_epoch']
    if not sample_validation:
        assert all(abs(v)<1e-10 for v in audited['full_minus_compact'].values())
    with pytest.raises(ValueError,match='already exists'):
        audit.audit_validation(path,tmp_path/'train',tmp_path/'audit')
    cfg['training']['device']='cuda';path.write_text(json.dumps(cfg))
    queue=tmp_path/'queue.json';queue.write_text(json.dumps({'status':'RUNNING','jobs':[]}))
    with pytest.raises(ValueError,match='queue has stopped'):
        audit.audit_validation(path,tmp_path/'train',tmp_path/'blocked_audit',queue_state=queue)
    assert not (tmp_path/'blocked_audit').exists()
    cfg['training']['device']='cpu';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='closed'):
        runner.test_checkpoint(path,tmp_path/'train/best.pt',tmp_path/'test')
    cfg['outer_evaluation_permitted']=True;path.write_text(json.dumps(cfg))
    cfg['protocol_signature']='different-protocol';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='training protocol'):
        runner.test_checkpoint(path,tmp_path/'train/best.pt',tmp_path/'wrong_protocol_test')
    cfg['protocol_signature']='frozen-synthetic-protocol';path.write_text(json.dumps(cfg))
    opened.clear()
    result=runner.test_checkpoint(path,tmp_path/'train/best.pt',tmp_path/'test')
    assert opened==[['heldout']] and result['reliable']['sample_count']==12
    catalog['input_config']['window_total_us']=.6;toy_catalog.write_text(json.dumps(catalog))
    with pytest.raises(ValueError,match='policy differs'):
        runner.test_checkpoint(path,tmp_path/'train/best.pt',tmp_path/'bad_test')
