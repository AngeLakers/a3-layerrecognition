from dataclasses import replace
import numpy as np
import pytest
import torch
from a3_ultrasound.evaluation.v16_metrics import joint_metrics, calibration, risk_coverage
from a3_ultrasound.evaluation.v16_temperature import tempered_log_joint, specimen_oof
from a3_ultrasound.data.m3_transform import dictionary, transform, fit_family
from a3_ultrasound.data.v16_channels import attach_channel
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets
from a3_ultrasound.v16_protocol import template,validate_config,matrix


def test_hand_metrics():
    y=np.array([[0,0],[0,1],[1,0],[1,1]])
    p=np.array([[.8,.1,.05,.05],[.1,.7,.1,.1],[.1,.1,.2,.6],[.9,.04,.03,.03]])
    result=joint_metrics(y,p,[2,1,3,4])
    assert result['positive_joint_map_accuracy']['value']==1/8
    assert result['normal_joint_false_positive_rate']['value']==0
    assert sum(v['value'] for v in result['abnormal_three_way'].values())==1
    assert result['hce_0_9']['value']==1
    assert result['coverage_0_9']['value']==.4
    assert result['joint_nll']['value']==pytest.approx(-np.dot([2,1,3,4],np.log([.8,.7,.2,.03]))/10)


def test_ties_and_empty_groups():
    a=calibration(np.array([.6,.6,.9]),np.array([1,0,1]),np.array([1,3,2]))
    b=calibration(np.array([.6,.9,.6]),np.array([0,1,1]),np.array([3,2,1]))
    assert a==b
    assert risk_coverage(np.array([.5,.5]),np.array([0,1]),np.array([1,1]))['augrc']==.25
    assert joint_metrics([[0,0]],[[1,0,0,0]],[1])['positive_joint_map_accuracy']['value'] is None


def test_chain_temperature_is_not_joint_power():
    p=np.array([[.4,.1,.1,.4]])
    q=tempered_log_joint(np.log(p),2)
    np.testing.assert_allclose(np.exp(q),[[1/3,1/6,1/6,1/3]])
    # Unequal prefix masses demonstrate the distinction from joint softmax.
    p=np.array([[.56,.14,.03,.27]])
    q=tempered_log_joint(np.log(p),2)
    assert not np.allclose(np.exp(q),np.sqrt(p)/np.sqrt(p).sum(1,keepdims=True))
    np.testing.assert_allclose(tempered_log_joint(np.log(p),1),np.log(p),atol=1e-14)


def test_lp_temperature_map_and_oof():
    p=np.array([[.72,.08,.18,.02],[.08,.72,.02,.18],[.18,.02,.72,.08],[.02,.18,.08,.72]])
    for t in (.1,.5,2,10):
        assert np.array_equal(tempered_log_joint(np.log(p),t).argmax(1),p.argmax(1))
    output,records=specimen_oof(np.log(p),np.array([0,1,2,3]),np.ones(4),['a','b','a','c'])
    assert records['a']['held_records']==2 and 'a' not in records['a']['fit_specimens']
    assert np.isfinite(output).all()


def test_m3_only_subtracts_normalized_copies():
    params=dict(alpha=.5,delta=2,f0_cycles_per_sample=.08,sigma_samples=6,status='FITTED')
    D,C,deep,lo,hi=dictionary([200,400],600,params)
    j=200-lo;x=(D[:,j]*4)[None]
    r,d=transform(x,[200,400],params)
    expected=x.copy();expected[:,lo:hi]-=4*C[lo:hi,j]
    np.testing.assert_allclose(r,expected,atol=2e-6)
    np.testing.assert_allclose(d,0,atol=2e-6)
    assert np.array_equal(r[:,:lo],x[:,:lo].astype(np.float32))
    with pytest.raises(ValueError,match='BOUNDARY'):dictionary([200,400],600,dict(params,delta=24))


def test_train_only_fit_and_fallback():
    row=dict(x=np.zeros((2,600)),y=np.zeros((2,2)),centers=[200,400],family='Al-K2')
    with pytest.raises(ValueError,match='Non-TRAIN'):fit_family({'held':row},['train'],'Al-K2',lambda_grid=[1])
    p=fit_family({'train':row},['train'],'Al-K2',lambda_grid=[1])
    assert p['status']=='IDENTITY_FALLBACK'
    x=np.ones((2,600));r,d=transform(x,[200,400],p)
    np.testing.assert_array_equal(r,x);np.testing.assert_array_equal(d,0)


def test_dual_channel_shared_geometry():
    batch=synthetic_datasets(23,2)['train'].batch
    n=batch.x_raw.shape[-1]
    out=attach_channel(batch,[np.arange(n+7,dtype=np.float32)]*len(batch.x_raw))
    assert out.x_raw.shape==(len(batch.x_raw),2,n)
    assert out.interface_meta is batch.interface_meta
    assert out.length_mask is batch.length_mask
    torch.testing.assert_close(out.x_raw[:,0],batch.x_raw,rtol=0,atol=0)
    cfg=ModelConfig(architecture_version='a3-v1.6',backbone='tcn',backbone_norm='group8',query_view='position2',head_type='vanilla_chain',input_channels=2)
    model=build_model(cfg);result=model.infer_exact(out)
    assert result is not None


def test_protocol_matrix_and_rejections():
    cfg=template();validate_config(cfg,fold_name='F1',seed=77,head='vanilla_chain',require_permission=False)
    assert len(matrix())==65 and len(matrix('residual'))==95
    with pytest.raises(ValueError):validate_config(cfg,fold_name='F1',seed=77,head='local_independent',require_permission=False)
    with pytest.raises(ValueError):validate_config(cfg,fold_name='F1',seed=23,head='vanilla_chain')
    cfg['training']['learning_rate']=.002
    with pytest.raises(ValueError):validate_config(cfg,fold_name='F1',seed=23,head='vanilla_chain',require_permission=False)


def test_log_map_preserves_near_ties():
    logp=np.log(np.array([[.1,.4,.4+1e-8,.1-1e-8]],dtype=float))
    p=np.exp(logp).astype(np.float32)
    assert p.argmax(1)[0]==1
    result=joint_metrics([[1,0]],p,[1],log_probabilities=logp)
    assert result['positive_joint_map_accuracy']['value']==1


def test_no_dual_l_or_training_bypass(tmp_path):
    import json
    from a3_ultrasound.stream_training import train
    cfg=template();cfg['model']['input_channels']=2;cfg['m3_channel']='residual'
    with pytest.raises(ValueError,match='LP and V'):
        validate_config(cfg,fold_name='F1',seed=23,head='local_independent',require_permission=False)
    cfg=template();path=tmp_path/'cfg.json';path.write_text(json.dumps(cfg),encoding='utf-8')
    with pytest.raises(ValueError,match='launch awaits'):
        train(path,tmp_path/'out',fold_name='F1',head='vanilla_chain',seed=23,diagnostic=True)
    assert not (tmp_path/'out').exists()


def test_mixed_k_single_temperature_and_masks():
    from a3_ultrasound.evaluation.v16_temperature import mixed_specimen_oof
    rows=[]
    for i,k in enumerate((2,3,2,3)):
        rows.append(dict(head='local_pre_independent',seed=23,y=[0]*k,label_mask=[True]*k,
            log_probabilities=[-np.log(2**k)]*(2**k),sample_weight=1,specimen_id=str(i%3)))
    output,records,final=mixed_specimen_oof(rows)
    assert final['temperature']==1 and final['status']=='UNIDENTIFIABLE_FLAT_OBJECTIVE'
    assert len(output)==4 and records['0']['fit_specimens']==['1','2']
    rows[0]['label_mask']=[]
    with pytest.raises(ValueError,match='masks'):mixed_specimen_oof(rows)


def test_fold_sampling_is_deterministic_and_train_only():
    from a3_ultrasound.data.m3_transform import sample_train_records
    row=dict(x=np.arange(6001)[:,None],y=np.zeros((6001,2),int),row_ids=np.arange(6001),ordinal=1,centers=[200,400],family='TC4-K2')
    a=sample_train_records({'a':row},['a'],1);b=sample_train_records({'a':row},['a'],1)
    np.testing.assert_array_equal(a['a']['row_ids'],b['a']['row_ids']);assert len(a['a']['x'])==5000
    with pytest.raises(ValueError):sample_train_records({'a':row},['a','b'],1)


def test_d2_requires_all_groups_and_support():
    from a3_ultrasound.evaluation.v16_d2 import choose_channel
    events=[dict(fold=f['fold'],board=b,K=2 if b.startswith('TCL3-') else 3)
            for f in template()['folds'] for b in f['validation'] if b.startswith(('TCL3-','TCL4-'))]
    rows=[]
    for e in events:
        for group in (('K2:0','K2:1') if e['K']==2 else ('K3:00','K3:01','K3:10')):
            rows.append(dict(fold=e['fold'],board=e['board'],group=group,x=.7,r=.75,d=.74,fit_normal_support=200))
    assert choose_channel(rows,events)['channel']=='residual'
    assert choose_channel(rows[:-1],events)['status']=='INSUFFICIENT'
    rows[0]['fit_normal_support']=199
    assert choose_channel(rows,events)['status']=='INSUFFICIENT'
