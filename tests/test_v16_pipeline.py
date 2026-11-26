"""V1.6 wiring: frozen timebase, M6 off, M3 second channel, fit/D2 runners, metrics."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import a3_ultrasound.v16_protocol as protocol
from a3_ultrasound.data.m3_transform import Decomposer, dictionary, transform
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.data.quick_input import transform_trace
from a3_ultrasound.data.v16_channels import M3Channel
from a3_ultrasound.evaluation import v16_d2
from a3_ultrasound.evaluation.v16_d2 import auc, average_precision, choose_channel, envelope_peak, sample_event
from a3_ultrasound.evaluation.v16_metrics import joint_metrics, pooled_reliability

ROOT=Path(__file__).resolve().parents[1]
POLICY={'name':'board_timebase','halfwidth_samples':80}
JITTER={'enabled':True,'max_shift_samples':24,'seed':23,'distribution':'uniform'}
PARAMS=dict(status='FITTED',alpha=.5,delta=2,f0_cycles_per_sample=.08,sigma_samples=6.)
INPUT=dict(time_reference='cropped_record_origin_is_zero',batch_reference_samples=0,
    sampling_rate_hz=200000000,sound_speed_mps=dict(TC4=6100,AL=6320),window_total_us=.7,
    normalization_divisor=131,tail=dict(policy='nominal_fixed',nominal_backwall_guard_us=.5,
        active_boundary_check_samples=32,quiet_run_samples=32,quiet_floor_raw_units=3,quiet_guard_samples=64))
BOARDS=[(f'TCL3-{i}',2,'TCL3') for i in range(1,7)]+[(f'TCL4-{i}',3,'TCL4') for i in range(1,5)]+[
    ('厚1',2,'AL'),('厚2',2,'AL'),('薄',2,'AL')]
TAU={2:[252.,430.],3:[180.,340.,500.]};BOTTOM={2:608.,3:640.}


def make_catalog(root, boards=BOARDS, n=24, length=850):
    """Synthetic catalog: one RF file and index directory per board."""
    rng=np.random.default_rng(5);entries=[]
    t=np.arange(length)
    for ordinal,(name,k,group) in enumerate(boards):
        base=root/f'b{ordinal}';base.mkdir()
        states=np.array([[int(b) for b in format(i%(2**k),f'0{k}b')] for i in range(n)],np.int8)
        x=np.stack([sum((1+2*s)*np.exp(-.5*((t-c)/6)**2)*np.cos(2*np.pi*.08*(t-c)) for s,c in zip(row,TAU[k]))
                    +.05*rng.standard_normal(length) for row in states]).astype(np.float32)*20
        np.save(base/'RF.npy',x.ravel());lm=np.ones((n,k),bool)
        arrays=dict(offsets=np.arange(n)*length,lengths=np.full(n,length),source_row=np.arange(n)+100,
            y=states,label_mask=lm,eligible=np.ones(n,bool),design_y=states,design_mask=lm,
            xy=np.column_stack([np.arange(n),np.arange(n)]),quality=np.ones(n,bool))
        for key,value in arrays.items():np.save(base/(key+'.npy'),value)
        entries.append(dict(entity=name,ordinal=ordinal,K=k,group=group,layer_mm=3,index_dir=base.name,
                            rf_path=str(base/'RF.npy')))
    path=root/'catalog.json';path.write_text(json.dumps(dict(input_config=INPUT,entities=entries),ensure_ascii=False),encoding='utf-8')
    rows=[dict(board=name,K=k,selected_tau=TAU[k],selected_bottom=BOTTOM[k],nominal_bottom=BOTTOM[k]) for name,k,_ in boards]
    return path,rows


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Template config + synthetic catalog + a stand-in frozen timebase file."""
    catalog,rows=make_catalog(tmp_path)
    raw=json.dumps(dict(version=protocol.TIMEBASE_VERSION,status='FROZEN_V1_6_INPUT',rows=rows),ensure_ascii=False).encode()
    (tmp_path/'timebase.json').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(protocol,'TIMEBASE_SHA256',digest)
    cfg=protocol.template()
    cfg.update(catalog=str(catalog),validation_selection_root=None,timebase_calibration_path=str(tmp_path/'timebase.json'),
               timebase_calibration_sha256=digest,m6_decision_path=str(ROOT/cfg['m6_decision_path']),
               speed=dict(cfg['speed'],workers=0,pin_memory=False,non_blocking=False,cpu_structure=False),
               fold_name='F1',seed=23)
    return cfg,rows,tmp_path


def write_json(path, value):
    raw=(json.dumps(value,ensure_ascii=False)+'\n').encode();Path(path).write_bytes(raw)
    return str(path),hashlib.sha256(raw).hexdigest()


def dual(cfg, tmp_path, fold='F1', channel='residual', manifest_edit=None, decision_edit=None):
    fold_def=next(f for f in cfg['folds'] if f['fold']==fold)
    manifest=dict(status='FITTED',fold=fold,train_entities=fold_def['train'],timebase_sha256=protocol.TIMEBASE_SHA256,
                  family_params={'TC4-K2':PARAMS,'TC4-K3':dict(PARAMS,delta=-3),'Al-K2':dict(status='IDENTITY_FALLBACK')},
                  entity_family={b:('Al-K2' if g=='AL' else f'TC4-K{k}') for b,k,g in BOARDS})
    if manifest_edit:manifest_edit(manifest)
    mpath,msha=write_json(tmp_path/f'{fold}_manifest.json',manifest)
    decision=dict(status='DECIDED',channel=channel,fit_manifest_sha256={fold:msha})
    if decision_edit:decision_edit(decision)
    dpath,dsha=write_json(tmp_path/'decision.json',decision)
    out=deepcopy(cfg);out['model']['input_channels']=2
    out.update(m3_channel=channel,m3_fit_manifest=mpath,m3_fit_manifest_sha256=msha,m3_d2_decision=dpath,m3_d2_decision_sha256=dsha)
    return out


def test_decomposer_is_the_cached_transform():
    x=np.random.default_rng(1).standard_normal((3,700))*10
    op=Decomposer([252.,430.],700,PARAMS)
    for got,want in zip(op(x),transform(x,[252.,430.],PARAMS)):np.testing.assert_array_equal(got,want)
    channel=M3Channel(dict(channel='free_deep',params_by_entity={'A':PARAMS}),{'A':dict(selected_tau=[252.,430.])},['A'])
    np.testing.assert_array_equal(channel('A',x[1]),transform(x[1:2],[252.,430.],PARAMS)[1][0])
    assert channel.operator('A',700) is channel.operator('A',700)
    with pytest.raises(ValueError,match='missing'):M3Channel(dict(channel='residual',params_by_entity={}),{},['A'])


def test_dataset_second_channel_is_sample_aligned(frozen):
    cfg,rows,_=frozen;table=dict(_artifact_sha256='x',rows=rows);catalog=cfg['catalog']
    m3=dict(channel='residual',params_by_entity={'TCL3-2':PARAMS})
    one=RFShardDataset(catalog,['TCL3-2'],training=True,window_policy=POLICY,timebase_config=table,window_jitter=JITTER)
    two=RFShardDataset(catalog,['TCL3-2'],training=True,window_policy=POLICY,timebase_config=table,window_jitter=JITTER,m3_config=m3)
    for d in (one,two):d.set_epoch(4)
    ids=np.arange(len(two));a=one.make_batch(ids);b=two.make_batch(ids)
    assert b.x_raw.shape==(len(ids),2,a.x_raw.shape[-1])
    torch.testing.assert_close(b.x_raw[:,0],a.x_raw,rtol=0,atol=0)
    torch.testing.assert_close(b.interface_meta,a.interface_meta,rtol=0,atol=0)
    assert torch.equal(b.length_mask,a.length_mask)
    shard=two.shards[0];arr=shard['arrays']
    for j,(s,i) in enumerate(two.refs):
        raw=shard['rf'][arr['offsets'][i]:arr['offsets'][i]+arr['lengths'][i]]
        n=int(b.length_mask[j].sum());r,_=transform(np.asarray(raw)[None],TAU[2],PARAMS)
        np.testing.assert_allclose(b.x_raw[j,1,:n].numpy(),r[0,:n]/np.float32(131),rtol=0,atol=1e-7)
        assert float(b.x_raw[j,1,n:].abs().sum())==0
    assert two._spawn_kwargs['m3_config'] is m3


def test_evaluation_cache_keeps_both_channels(frozen):
    cfg,rows,_=frozen;table=dict(_artifact_sha256='x',rows=rows)
    data=RFShardDataset(cfg['catalog'],['TCL4-2'],window_policy=POLICY,timebase_config=table,
                        m3_config=dict(channel='residual',params_by_entity={'TCL4-2':dict(PARAMS,delta=-3)}))
    first=data.make_batch([0,1,2]);again=data.make_batch([0,1,2])
    assert torch.equal(first.x_raw,again.x_raw)
    data._record_cache.clear();data._remember_batch(first,[0,1,2])
    cached=data._record_cache[(0,int(data.refs[0][1]))][0]
    assert cached.shape==(2,int(first.length_mask[0].sum()))


def test_fast_eval_assembly_accepts_two_channels():
    from a3_ultrasound.execution_performance import fast_eval_assembly
    from a3_ultrasound.models.backbone import _encode_by_valid_length
    torch.manual_seed(0);encoder=torch.nn.Conv1d(2,4,3,padding=1).eval()
    x=torch.randn(3,2,40);mask=torch.arange(40)[None]<torch.tensor([[40],[31],[40]])
    with torch.no_grad():
        fast=fast_eval_assembly(_encode_by_valid_length,x,mask,encoder,output_dim=4)
        slow=_encode_by_valid_length(x,mask,encoder,output_dim=4)
    torch.testing.assert_close(fast.features,slow.features)


def test_dataset_from_config_uses_v16_artefacts(frozen):
    from a3_ultrasound.speed import dataset_from_config
    cfg,rows,tmp=frozen
    data=dataset_from_config(cfg,'validation')
    assert data.m3 is None and data.timebase_rows['TCL3-4']['selected_tau']==TAU[2]
    assert data.config['timebase_calibration_sha256']==protocol.TIMEBASE_SHA256
    batch=data.make_batch([0,1]);assert batch.x_raw.ndim==2
    data=dataset_from_config(dual(cfg,tmp),'train')
    assert data.m3.channel=='residual' and data.make_batch([0,1]).x_raw.shape[1]==2
    bad=deepcopy(cfg);bad['model']['pre_exclude_front_end']=180
    with pytest.raises(ValueError,match='M6'):dataset_from_config(bad,'validation')
    bad=deepcopy(cfg);bad['timebase_calibration_sha256']='0'*64
    with pytest.raises(ValueError,match='frozen'):dataset_from_config(bad,'validation')


def test_runtime_record_copies_exact_inputs(frozen, tmp_path):
    cfg,_,tmp=frozen;cfg=dual(cfg,tmp);cfg['speed_decision_path']=str(ROOT/cfg['speed_decision_path'])
    record=protocol.runtime_record(cfg,'F1',tmp_path/'run')
    assert record['m3_channel']=='residual' and record['m3_params_by_entity']['TCL3-2']==PARAMS
    assert (tmp_path/'run'/'m3_fit_manifest.json').read_bytes()==Path(cfg['m3_fit_manifest']).read_bytes()
    assert (tmp_path/'run'/'timebase_calibration.json').read_bytes()==Path(cfg['timebase_calibration_path']).read_bytes()


def test_m3_channel_guards(frozen):
    cfg,_,tmp=frozen
    with pytest.raises(ValueError,match='did not select'):protocol.load_m3_channel(dual(cfg,tmp,channel='residual',
        decision_edit=lambda d:d.update(channel='free_deep')),'F1')
    with pytest.raises(ValueError,match='this fold'):protocol.load_m3_channel(dual(cfg,tmp,
        decision_edit=lambda d:d.update(fit_manifest_sha256={'F1':'0'*64})),'F1')
    with pytest.raises(ValueError,match='TRAIN panel'):protocol.load_m3_channel(dual(cfg,tmp,
        manifest_edit=lambda m:m.update(train_entities=['TCL3-4'])),'F1')
    with pytest.raises(ValueError,match='BOUNDARY'):protocol.load_m3_channel(dual(cfg,tmp,
        manifest_edit=lambda m:m['family_params'].update({'TC4-K2':dict(PARAMS,status='BOUNDARY_LIMITED')})),'F1')
    changed=dual(cfg,tmp);Path(changed['m3_fit_manifest']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):protocol.load_m3_channel(changed,'F1')


def test_validate_config_m3_runtime_keys(frozen):
    cfg,_,tmp=frozen;base=protocol.template()
    base.update(catalog='c',validation_selection_root='v',epoch_selection_root='e',source_commit='s')
    protocol.validate_config(base,fold_name='F1',seed=97,head='vanilla_chain',require_permission=False)
    two=deepcopy(base);two['model']['input_channels']=2;two['m3_channel']='residual'
    with pytest.raises(ValueError,match='require a verified'):protocol.validate_config(two,fold_name='F1',seed=23,head='vanilla_chain',require_permission=False)
    for key in protocol.M3_RUNTIME_KEYS:two[key]='a'*64
    protocol.validate_config(two,fold_name='F1',seed=23,head='vanilla_chain',require_permission=False)
    with pytest.raises(ValueError,match='Unregistered'):protocol.validate_config(two,fold_name='F1',seed=97,head='vanilla_chain',require_permission=False)
    one=deepcopy(base);one['m3_fit_manifest']='x'
    with pytest.raises(ValueError,match='must not reference'):protocol.validate_config(one,fold_name='F1',seed=23,head='vanilla_chain',require_permission=False)
    with pytest.raises(ValueError,match='launch awaits'):protocol.validate_config(base,fold_name='F1',seed=23,head='vanilla_chain')


def planned():
    cfg=protocol.template()
    return [dict(fold=f['fold'],board=b,K=2 if b.startswith('TCL3') else 3) for f in cfg['folds']
            for b in f['validation'] if b.startswith('TCL')]


def d2_rows(gain=.05, d_gain=.04):
    rows=[]
    for e in planned():
        for g in (('K2:0','K2:1') if e['K']==2 else ('K3:00','K3:01','K3:10')):
            up=gain if g in ('K2:1','K3:01','K3:10') else 0.
            rows.append(dict(fold=e['fold'],board=e['board'],group=g,x=.6,r=.6+up,d=.6+min(up,d_gain),fit_normal_support=900))
    return rows


def test_d2_rule_and_insufficiency():
    assert choose_channel(d2_rows(),planned())['channel']=='residual'
    assert choose_channel(d2_rows(d_gain=0.),planned())['channel']=='residual'
    rows=d2_rows();rows[0]['r']=rows[0]['x']-.2   # one K2:0 board strongly worse: median still fine
    assert choose_channel(rows,planned())['status']=='DECIDED'
    rows=d2_rows(gain=.01);assert choose_channel(rows,planned())['channel']=='none'
    rows=[r for r in d2_rows() if not (r['group']=='K3:10' and r['fold']=='F1')]
    result=choose_channel(rows,planned())
    assert result['status']=='DECIDED' and result['events_used']['K3:10']==4
    rows=[r for r in d2_rows() if not (r['group']=='K3:10' and r['fold'] in ('F1','F2','F3'))]
    assert choose_channel(rows,planned())['insufficient_groups']==['K3:10']
    rows=d2_rows();next(r for r in rows if r['group']=='K2:1')['fit_normal_support']=199
    result=choose_channel(rows,planned())
    assert result['status']=='INSUFFICIENT' and result['insufficient_groups']==['K2:1']


def test_scores_auc_ap_and_event_sampling(monkeypatch):
    assert auc(np.array([1.,2.]),np.array([1.,0.]))==pytest.approx(.875)
    assert auc(np.array([1.]),np.array([]))is None
    pos=np.array([.9,.4,.4]);neg=np.array([.8,.4,.1])
    # Brute force: thresholds .9,.8,.4,.1 -> (P,R) (1,1/3),(1/2,1/3),(3/5,1),(1/2,1)
    assert average_precision(pos,neg)==pytest.approx(1/3+(2/3)*.6)
    t=np.arange(900.);burst=3*np.exp(-.5*((t-430)/6)**2)*np.cos(2*np.pi*.08*(t-430))
    assert envelope_peak(burst[None],[900],430.)[0]==pytest.approx(3,rel=.01)
    monkeypatch.setattr(v16_d2,'PER_CLASS_CAP',3)
    y=np.array([[0,0]]*7+[[0,1]]*2+[[1,1]]*5)
    a=v16_d2.sample_event(y,2,9);b=v16_d2.sample_event(y,2,9)
    assert np.array_equal(a,b) and len(a)==3+2+3 and np.all(np.diff(a)>0)
    assert not np.array_equal(a,v16_d2.sample_event(y,3,9))


def test_abnormal_denominators_and_pooled_reliability():
    y=np.array([[0,0],[0,1],[1,0],[1,1]]);w=np.array([2.,1,3,4])
    p=np.array([[.8,.1,.05,.05],[.1,.1,.1,.7],[.1,.6,.2,.1],[.9,.04,.03,.03]])
    m=joint_metrics(y,p,w)
    # abnormal mass 8; row 2 predicts 11 for 01 (overcall), row 3 predicts 01 for 10 (both).
    assert m['attribution_overcall_of_abnormal']['value']==pytest.approx(4/8)
    assert m['attribution_undercall_of_abnormal']['value']==pytest.approx(3/8)
    assert m['attribution_overcall']['value']==pytest.approx(1.)
    y3=np.array([[0,0,0],[0,0,1]]);p3=np.full((2,8),1/8);w3=np.array([1.,1.])
    pooled=pooled_reliability([(y,p,w),(y3,p3,w3)])
    assert pooled['coverage_0_9']['value']==pytest.approx(4/12)
    assert pooled['calibration']['ece']>=0 and pooled['risk_coverage']['all']['denominator']==12


def test_fit_and_d2_runners_touch_only_their_panels(frozen, tmp_path):
    from a3_ultrasound.data.m3_fit import fit_fold
    cfg,rows,_=frozen;timebase=dict(rows=rows);manifests={}
    for fold in cfg['folds']:
        manifest,path=fit_fold(cfg,fold['fold'],timebase,tmp_path/'m3')
        assert manifest['train_entities']==fold['train'] and len(manifest['entity_family'])==13
        assert set(json.loads((tmp_path/'m3'/manifest['side_files']['sampled_identities']['path']).read_text(encoding='utf-8'))) == set(fold['train'])
        # Only 24 records per board here: every family falls back to identity (r=x, d=0).
        assert {p['status'] for p in manifest['family_params'].values()}=={'IDENTITY_FALLBACK'}
        raw=path.read_bytes();manifests[fold['fold']]=(json.loads(raw),hashlib.sha256(raw).hexdigest())
    decision,table,identities=v16_d2.run_d2(cfg,timebase,manifests)
    assert decision['status']=='INSUFFICIENT'
    assert {k.split(':')[1] for k in identities}=={b for f in cfg['folds'] for b in f['validation']}
    assert all(r['x']==r['r'] for r in table if r['x'] is not None)


def test_cosine_learning_rate_schedule():
    from a3_ultrasound.training import TrainingConfig, _validate_training_config, scheduled_learning_rate
    cfg=protocol.template()['training']
    tc=TrainingConfig(**cfg);_validate_training_config(tc)
    lr=[scheduled_learning_rate(tc,u) for u in np.arange(0,40,.25)]
    assert lr[0]==pytest.approx(1e-3/3) and scheduled_learning_rate(tc,2)==pytest.approx(1e-3)
    assert all(b>=a for a,b in zip(lr[:8],lr[1:9])) and all(b<=a for a,b in zip(lr[8:],lr[9:]))
    assert scheduled_learning_rate(tc,40)==pytest.approx(1e-5)
    assert scheduled_learning_rate(TrainingConfig(),7.5)==1e-3   # v1.5 configs: constant
    with pytest.raises(ValueError,match='either'):
        _validate_training_config(TrainingConfig(**dict(cfg,scheduler_step_size=5)))


def test_calibration_export_requires_guarded_v16_runs(tmp_path):
    from a3_ultrasound.evaluation.v16_export import export_rows
    run=tmp_path/'r';run.mkdir()
    fold=dict(fold='F1',validation=['TCL3-4'])
    (run/'resolved_config.json').write_text(json.dumps(dict(fold=fold,seed=23,model=dict(head_type='vanilla_chain'),
        architecture_version='a3-v1.6',diagnostic=False)))
    (run/'summary.json').write_text(json.dumps(dict(status='COMPLETED',guard_violated=False,selected_checkpoint='best_guarded.pt')))
    rows=[dict(sample_id='TCL3-4:1',specimen_id='TCL3-4',y=[0,1],label_mask=[True,True],log_probabilities=[-1.3863]*4,sample_weight=3),
          dict(sample_id='TCL3-4:2',specimen_id='TCL3-4',y=[0,0],label_mask=[True,False],log_probabilities=[-1.3863]*4,sample_weight=1)]
    (run/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=export_rows([run],head='vanilla_chain',seed=23)
    assert len(out)==1 and out[0]['checkpoint_kind']=='guarded' and out[0]['fold']=='F1'
    (run/'summary.json').write_text(json.dumps(dict(status='COMPLETED',guard_violated=True,selected_checkpoint='best_fallback.pt')))
    with pytest.raises(ValueError,match='guarded'):export_rows([run],head='vanilla_chain',seed=23)
