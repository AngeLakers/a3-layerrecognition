"""V1.5 candidate controls; physical-window-dependent M5/M6 are not frozen."""
import copy
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import torch
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets, evaluate
from a3_ultrasound.models.losses import head_training_loss, pcc_training_loss
from a3_ultrasound.models.pcc import PCCConditionalStep
from a3_ultrasound.prefix_perturbation import replacement_mask, training_conditional
from a3_ultrasound.data.quick_input import transform_trace
from a3_ultrasound.validation_diagnostics import CorrectionArchive


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(head='proposed_v15'):
    torch.manual_seed(17)
    m=build_model(ModelConfig(head_type=head,backbone='tcn',rho=3.))
    return m, synthetic_datasets(31,4)['train']


def test_suppress_initial_bound_all_zero_prefix_and_future_causality():
    m,d=sample();e=m.encode(d.batch)
    zero=torch.zeros_like(d.batch.y);o=m.conditional(e,zero)
    assert (o.correction<=0).all()
    assert o.correction.abs().max().item()==pytest.approx(.142277628,abs=1e-6)
    assert not o.upstream_flag.any()
    assert torch.equal(o.logits,o.base_logodds)
    changed=zero.clone();changed[:,0]=1
    other=m.conditional(e,changed)
    assert torch.equal(other.logits[:,0],o.logits[:,0])
    assert torch.all(other.logits[:,1:] <= other.base_logodds[:,1:])
    assert torch.all(other.upstream_flag[:,1:]==d.batch.interface_mask[:,1:])
    exact=m.infer_encoded(e)
    # The zero path probability equals the independent base zero-path product.
    expected=torch.nn.functional.logsigmoid(-o.base_logodds)*d.batch.interface_mask
    torch.testing.assert_close(exact.raw_log_probabilities[:,0],expected.sum(1),atol=2e-6,rtol=0)
    torch.testing.assert_close(exact.raw_log_normalizer,torch.zeros(4),atol=2e-6,rtol=0)


@pytest.mark.parametrize('head',['proposed','vanilla_chain','proposed_v15'])
def test_zero_probability_preserves_output_gradient_and_global_rng(head):
    m,d=sample(head);a=m(d.batch,d.batch.y).conditional
    state=torch.get_rng_state().clone()
    b,stats=training_conditional(m,d.batch,probability=0,seed=23,epoch=1,step=0)
    assert torch.equal(a.logits,b.logits)
    assert torch.equal(state,torch.get_rng_state())
    assert stats['attempted']==stats['changed']==0
    pa=torch.autograd.grad(a.logits.sum(),m.head.base.network[0].weight,allow_unused=True,retain_graph=True)
    pb=torch.autograd.grad(b.logits.sum(),m.head.base.network[0].weight,allow_unused=True)
    assert (pa[0] is None and pb[0] is None) or torch.equal(pa[0],pb[0])


def test_replacement_mask_is_head_independent_and_has_no_global_rng_effect():
    mask=torch.ones(100,6,dtype=torch.bool);mask[:,4:]=False
    state=torch.get_rng_state().clone()
    one=replacement_mask(mask,probability=.2,seed=23,epoch=2,step=7)
    two=replacement_mask(mask,probability=.2,seed=23,epoch=2,step=7)
    assert torch.equal(one,two) and not one[:,4:].any()
    assert torch.equal(state,torch.get_rng_state())
    assert not torch.equal(one,replacement_mask(mask,probability=.2,seed=23,epoch=2,step=8))


@pytest.mark.parametrize('head',['vanilla_chain','proposed_v15'])
def test_two_head_passes_one_encode_targets_unchanged_and_gradients_live(head):
    m,d=sample(head);y=d.batch.y.clone();calls=[]
    h=m.backbone.register_forward_hook(lambda *a:calls.append(1))
    out,s=training_conditional(m,d.batch,probability=.2,seed=23,epoch=1,step=0);h.remove()
    assert len(calls)==1 and torch.equal(y,d.batch.y)
    loss=head_training_loss(out,d.batch,head_type=head,local_aux_mode='clean_upstream')
    loss.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.backbone.parameters())
    assert 0<=s['changed']<=s['attempted']<=s['eligible']


def test_clean_aux_uses_truth_even_when_mixed_prefix_has_false_alarm():
    m,d=sample();b=replace(d.batch,y=torch.zeros_like(d.batch.y))
    prefix=torch.zeros_like(b.y);prefix[:,0]=1
    out=m.conditional(m.encode(b),prefix)
    assert out.upstream_flag[:,1].all()
    clean=head_training_loss(out,b,head_type='proposed_v15',local_aux_mode='clean_upstream')
    all_loss=head_training_loss(out,b,head_type='proposed_v15',local_aux_mode='all')
    assert torch.equal(clean.local,all_loss.local)
    y=b.y.clone();y[:,0]=1
    dirty=replace(b,y=y)
    dirty_loss=head_training_loss(out,dirty,head_type='proposed_v15',local_aux_mode='clean_upstream')
    expected=torch.nn.functional.binary_cross_entropy_with_logits(out.base_logodds[:,0],y[:,0].float())
    assert torch.equal(dirty_loss.local,expected)


def test_fixed_tail_ignores_later_activity_and_preserves_windows():
    cfg=json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_query07_v2.json').read_text())
    cfg['tail']['policy']='nominal_fixed'
    x=np.zeros(1200,np.float32);x[787:]=100
    a,ma,ra=transform_trace(x,{'K':2,'layer_mm':3,'group':'AL'},cfg)
    b,mb,rb=transform_trace(np.zeros_like(x),{'K':2,'layer_mm':3,'group':'AL'},cfg)
    assert len(a)==len(b)==670 and not ra['tail_extended']
    np.testing.assert_array_equal(ma,mb)
    np.testing.assert_array_equal(a,b)


def test_v15_archive_distinguishes_raw_gc_and_effective_gc(tmp_path):
    m,d=sample();m.eval();_,rows=evaluate(m,d)
    a=CorrectionArchive()
    for row in rows:a.add(row)
    a.save(tmp_path/'test.npz',epoch=1,checkpoint='best.pt',commit_hash='synthetic',config_sha256='synthetic')
    with np.load(tmp_path/'test.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(z['gc'],z['g']*z['c'],atol=1e-7)
        np.testing.assert_allclose(z['effective_gc'],z['gc']*z['upstream_flag'],atol=1e-7)
        np.testing.assert_allclose(z['eta'],z['b']+z['effective_gc'],atol=1e-7)
        assert (z['c']<=0).all()


def test_false_upstream_harm_and_inactive_raw_direction_are_separate():
    from a3_ultrasound.stream_metrics import PredictionAccumulator
    row=dict(sample_id='synthetic:0',specimen_id='synthetic',K=2,y=[0,1],label_mask=[True,True],
        marginals=[.8,.2],map_states=[1,0],probabilities=[.1,.2,.6,.1],
        log_probabilities=np.log([.1,.2,.6,.1]).tolist(),base_logodds=[.3,.2],
        base_probabilities=[.6,.55],raw_logZ=0.,rho=3.,correction=[0.,-1.],
        gate_times_correction=[0.,-.6],upstream_flag=[False,True],sample_weight=5)
    acc=PredictionAccumulator();acc.add(row)
    inactive=dict(row,sample_id='synthetic:1',y=[0,0],map_states=[0,0],sample_weight=7,
        gate_times_correction=[0.,0.],upstream_flag=[False,False])
    acc.add(inactive)
    result=acc.result();risk=result['false_positive_upstream_diagnostic']
    assert risk['weighted_denominator']==risk['weighted_joint_MAP_missed']==5
    assert risk['raw_conditional_harmful_suppression']==1
    assert result['correction_direction']['upper_correct']['weighted_correct']==7
    assert result['active_effective_direction']['upper_correct']['weighted_count']==0
    assert result['active_effective_direction']['upper_wrong']['weighted_count']==5
