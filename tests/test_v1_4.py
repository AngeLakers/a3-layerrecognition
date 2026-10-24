from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from a3_ultrasound.models.backbone import TsaiTimePreservingBackbone
from a3_ultrasound.models.query import InterfaceQueryEncoder
from a3_ultrasound.models.integrated import A3Model
from a3_ultrasound.models.losses import pcc_training_loss
from a3_ultrasound.selection import GuardedSelector
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.training import ModelConfig,build_model,synthetic_datasets,evaluate
from a3_ultrasound.data.epoch_sampling import epoch_refs,freeze_epochs,use_epoch


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_gn_padding_batch_composition_train_eval_and_gradient():
    torch.manual_seed(9)
    net=TsaiTimePreservingBackbone('resnet1d',backbone_norm='group8')
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in net.modules())
    assert all(m.num_groups==8 and m.eps==1e-5 for m in net.modules() if isinstance(m,torch.nn.GroupNorm))
    x=torch.randn(1,73);a=net(x,torch.ones_like(x,dtype=torch.bool)).features
    mixed=torch.randn(3,97);mixed[0,:73]=x[0];mixed[0,73:]=10000
    mask=torch.arange(97)[None,:]<torch.tensor([73,97,73])[:,None]
    b=net(mixed,mask).features[0:1,:73]
    assert (a-b).abs().max()<1e-5
    net.eval();c=net(x,torch.ones_like(x,dtype=torch.bool)).features
    assert torch.equal(a,c)
    a.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters())


def test_query_stop_invariance_and_absolute_prior_windows():
    batch=synthetic_datasets(23,4)['train'].batch
    enc=InterfaceQueryEncoder(8)
    stop=batch.length_mask.sum(1)
    new_stop=stop+13
    meta=batch.interface_meta.clone()
    meta[:,:,[1,3,4]]*=stop[:,None,None]/new_stop[:,None,None]
    b=replace(batch,interface_meta=meta,x_raw=F.pad(batch.x_raw,(0,13)),
              length_mask=torch.arange(batch.x_raw.shape[1]+13)[None,:]<new_stop[:,None])
    assert torch.equal(enc(batch.interface_meta,batch.interface_mask),enc(b.interface_meta,b.interface_mask))
    for a,c in zip(A3Model._sample_windows(batch),A3Model._sample_windows(b)):
        torch.testing.assert_close(a,c,atol=1e-5,rtol=0)
    assert enc.network[0].in_features==2
    assert InterfaceQueryEncoder(8,query_view='legacy3').network[0].in_features==3


def metric(score=.5,fpr=0,nll=.8):
    return dict(positive_joint_map_accuracy=score,normal_joint_false_positive_rate=fpr,joint_nll=nll)


def test_guarded_fallback_tie_and_minimum_epochs():
    s=GuardedSelector()
    assert s.update(1,metric(.9,.02))['fallback_updated']
    assert s.guarded is None
    assert s.update(2,metric(.6,.01))['guarded_updated']
    for e in range(3,16):assert s.update(e,metric(.6,.01))['stale']==0
    assert s.update(16,metric(.601,.01,.7))['stale']==1
    assert s.guarded['epoch']==16
    for e in range(17,25):assert not s.update(e,metric(.6,.01))['stop']
    assert s.update(25,metric(.6,.01))['stop']
    assert s.fallback['epoch']==1
    for key in ('positive_joint_map_accuracy','normal_joint_false_positive_rate','joint_nll'):
        m=metric();m[key]=None
        with pytest.raises(ValueError,match='nonempty'):s.update(26,m)


def test_nll_tie_drift_does_not_reset_progress_and_guard_rejects_high_score():
    s=GuardedSelector(min_epochs=0,patience=10)
    s.update(1,metric(.7,0,1))
    for e,score in enumerate([.699,.698,.697,.698,.699,.700],2):
        assert s.update(e,metric(score,0,1-e*.01))['stale']==e-1
    assert not s.update(8,metric(.99,.01001))['guarded_updated']
    assert s.update(9,metric(.703,0,.9))['stale']==0


def test_shadow_uses_exclusive_prefix_and_own_denominator():
    y=torch.tensor([[0,1,0],[1,0,0]])
    mask=torch.ones_like(y,dtype=torch.bool)
    logits=torch.zeros(2,3);base=torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
    got=pcc_training_loss(logits,base,y,mask,shadow_auxiliary=True)
    terms=F.binary_cross_entropy_with_logits(base,y.float(),reduction='none')
    expected=(terms[0,:2].mean()+terms[1,0])/2
    torch.testing.assert_close(got.local,expected)
    got.local.backward()
    assert base.grad[0,2]==0 and torch.equal(base.grad[1,1:],torch.zeros(2))


def row(y,pred,weight=1,c=None):
    k=len(y)
    return dict(specimen_id='board',y=y,label_mask=[True]*k,map_states=pred,marginals=[.5]*k,
        base_probabilities=[.5]*k,base_logodds=[0.]*k,raw_logZ=0.,log_probabilities=[-math.log(2**k)]*(2**k),
        rho=1. if c is not None else None,correction=c,gate_times_correction=c,sample_weight=weight)


def test_weighted_subset_counts_unknown_exclusion_and_global_quantiles():
    a=PredictionAccumulator()
    a.add(row([0,0],[1,0],3,[0.,1.]));a.add(row([0,0],[0,0],97,[0.,0.]))
    a.add(row([1,0],[1,0],2,[0.,-1.]));a.add(row([1,1],[1,0],1,[0.,.4]))
    unknown=row([1,0],[1,0],200,[0.,.2]);unknown['label_mask']=[False,True];a.add(unknown)
    r=a.result()
    assert r['positive_joint_map_accuracy']==pytest.approx(2/3)
    assert r['normal_joint_false_positive_rate']==.03
    assert r['joint_subsets']['normal']['raw_count']==2
    assert r['correction_quantiles']['median']==.2
    assert r['correction_direction']['upper_correct']['raw_count']==3
    assert r['correction_direction']['upper_wrong']['raw_count']==1


def test_epoch_freezing_head_independence_and_tamper_rejection(tmp_path):
    class Data:
        training=True;entities=['A']
        shards=[dict(ordinal=4,arrays={'eligible':np.ones(100,dtype=bool),
            'y':np.array([[1,0]]*20+[[0,0]]*80),'xy':np.column_stack((np.arange(100),np.arange(100)))})]
    d=Data();a=epoch_refs(d.shards,23,1);b=epoch_refs(d.shards,23,2)
    assert np.array_equal(a,epoch_refs(d.shards,23,1)) and not np.array_equal(a,b)
    assert set(a[a[:,1]<20,1])==set(range(20))
    assert len(epoch_refs(d.shards,23,1,positive_fraction=.2))==16
    root=tmp_path/'frozen';freeze_epochs(d,root,seed=23,epochs=2,catalog_signature='test')
    use_epoch(d,root,1,seed=23,catalog_signature='test');assert np.array_equal(d.refs,a)
    with (root/'epoch_02.npy').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='hash'):use_epoch(d,root,2,seed=23,catalog_signature='test')


@pytest.mark.parametrize('head',['local_independent','local_pre_independent','vanilla_chain','proposed','proposed_shadow_aux'])
def test_all_v14_heads_build_with_same_frontend(head):
    m=build_model(ModelConfig(architecture_version='a3-v1.4',backbone_norm='group8',query_view='position2',head_type=head))
    assert m.query_encoder.network[0].in_features==2
    assert not any(isinstance(x,torch.nn.BatchNorm1d) for x in m.backbone.modules())


def test_cached_head_matches_legacy_frontend_and_fresh_initialization(tmp_path):
    from a3_ultrasound.head_cache import cache_frontend,evaluate_cache,fresh_head,train_cached_head
    from dataclasses import asdict
    cfg=ModelConfig(backbone='tcn');model=build_model(cfg)
    datasets=synthetic_datasets(23,12)
    class Data:
        def __init__(self,data):self.data=data
        def batches(self,size,device):return self.data.batches(size,device)
        def identity(self,i):return self.data.sample_ids[i],self.data.specimen_ids[i]
        def annotations(self,indices):return [{'sample_weight':1} for _ in indices]
    train=cache_frontend(model,Data(datasets['train']),batch_size=4,device='cpu')
    val=cache_frontend(model,Data(datasets['validation']),batch_size=4,device='cpu')
    dense,_=evaluate(model,datasets['validation'],batch_size=4)
    cached=evaluate_cache(model.head,val,batch_size=4)
    assert cached['positive_joint_map_accuracy']==dense['positive_joint_map_accuracy']
    assert cached['joint_nll']==pytest.approx(dense['joint_nll'],abs=1e-7)
    a=fresh_head(asdict(cfg),1,23);b=fresh_head(asdict(cfg),3,23)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    result=train_cached_head(train,val,asdict(cfg),tmp_path/'cache-smoke',rho=3,shadow=True,epochs=2,batch_size=4)
    assert result['status']=='COMPLETED_DIAGNOSTIC' and result['formal_result'] is False
    assert (tmp_path/'cache-smoke/epoch_02.pt').exists()


def test_cache_weights_and_fixed_stage1_endpoint():
    from a3_ultrasound.head_cache import validate_cache_weights,select_stage1_candidates
    c=dict(y=torch.zeros(2,2),weights=torch.tensor([1.,2.]))
    with pytest.raises(ValueError,match='int64'):validate_cache_weights(c)
    c['weights']=torch.tensor([1,2])
    validate_cache_weights(c)
    with pytest.raises(ValueError,match='unit'):validate_cache_weights(c,training=True)
    def result(rho,shadow,score,sat):
        return dict(status='COMPLETED_DIAGNOSTIC',epochs=15,rho=rho,shadow=shadow,seed=23,
            final_metrics=dict(correction_saturation_fraction_MAP_path=sat,positive_joint_map_accuracy=score,normal_joint_false_positive_rate=.005))
    a,b,c=result(1,False,.6,.99),result(3,False,.7,.4),result(3,True,.72,.3)
    assert select_stage1_candidates(a,b,c)==['proposed_rho3','proposed_shadow_aux_rho3']
    a['epochs']=2
    with pytest.raises(ValueError,match='epoch15'):select_stage1_candidates(a,b,c)


def test_stage2_alarm_boundaries():
    from a3_ultrasound.stream_training import stage2_alarms
    score=dict(correction_saturation_fraction_MAP_path=.9,normal_joint_false_positive_rate=.01,
               correction_direction={'upper_correct':{'accuracy':.55}})
    assert stage2_alarms(5,score)==[]
    score['correction_saturation_fraction_MAP_path']=.91
    assert stage2_alarms(2,score)==[] and len(stage2_alarms(3,score))==1
