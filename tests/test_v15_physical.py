from dataclasses import replace
import numpy as np
import pytest
import torch

from a3_ultrasound.data.physical_features import energy_log_ratios,physical_tokens
from a3_ultrasound.models.indexed_readout import build_indexed_supports
from a3_ultrasound.training import ModelConfig,build_model,synthetic_datasets


def test_energy_integer_windows_tail_invariance_and_batch_equivalence():
    x=np.arange(1,51,dtype=np.float32)/50
    got=energy_log_ratios(x,[10.2,20],[13.8,22],front_end=8)
    expected=np.log((np.array([np.sum(x[11:14].astype(float)**2),np.sum(x[20:23].astype(float)**2)])+1e-8)/(np.sum(x[:9].astype(float)**2)+1e-8))
    np.testing.assert_array_equal(got,expected)
    np.testing.assert_array_equal(got,energy_log_ratios(x[:30],[10.2,20],[13.8,22],front_end=8))
    np.testing.assert_array_equal(np.stack([got,got]),energy_log_ratios(np.stack([x,x]),[10.2,20],[13.8,22],front_end=8))
    assert np.isfinite(energy_log_ratios(np.zeros(30),[10],[15],front_end=8)).all()


def test_physical_context_strictly_upstream_and_padding_zero():
    loc,ctx=physical_tokens([-4,0,-2],mean=-2,std=2)
    np.testing.assert_array_equal(loc[:,0],[-1,1,0,0,0,0])
    np.testing.assert_array_equal(ctx,[[0,0,1],[-1,-1,0],[1,1,0],[0,0,0],[0,0,0],[0,0,0]])
    with pytest.raises(ValueError):physical_tokens([1],mean=0,std=0)
    with pytest.raises(ValueError):energy_log_ratios(np.ones(30),[10],[40],front_end=8)


@pytest.mark.parametrize('head',['local_independent','local_pre_independent','vanilla_chain','proposed_v15'])
@pytest.mark.parametrize('exclude',[None,180])
def test_all_heads_post_ln_physical_dimensions_and_empty_pre(head,exclude):
    torch.set_num_threads(1);torch.manual_seed(3)
    data=synthetic_datasets(31,3)['train'];batch=data.batch
    pl=torch.full((*batch.interface_mask.shape,1),2.75)*batch.interface_mask[...,None]
    pc=torch.full((*batch.interface_mask.shape,3),-1.25)*batch.interface_mask[...,None]
    batch=replace(batch,phys_loc=pl,phys_ctx=pc)
    model=build_model(ModelConfig(backbone='tcn',head_type=head,physical_features=True,pre_exclude_front_end=exclude,rho=3.))
    encoded=model.encode(batch);t=encoded.responsibility_tokens
    assert (t.r_loc.shape[-1],t.r_ctx.shape[-1],t.r_gate.shape[-1])==(33,35,36)
    assert torch.equal(t.r_loc[...,-1:],pl)
    assert torch.equal(t.r_ctx[...,-3:],pc)
    assert torch.equal(t.r_gate[...,-4:],torch.cat([pl,pc],-1))
    out=model.conditional(encoded,batch.y);assert torch.isfinite(out.logits).all()
    model.infer_encoded(encoded)
    out.logits.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.backbone.parameters())
    if exclude is not None:
        assert not encoded.pre_support.any()  # synthetic trace ends before 180
        assert torch.isfinite(encoded.pre_readout.token).all()
    if head=='local_independent':
        changed=replace(batch,phys_ctx=pc*7)
        assert torch.equal(model(batch,batch.y).conditional.logits,model(changed,batch.y).conditional.logits)


def test_m6_disabled_preserves_pre_and_enabled_excludes_inclusive_front():
    centers=torch.arange(12.)
    start=torch.tensor([[3.,10.]])
    end=torch.tensor([[5.,11.]])
    valid=torch.ones(1,12,dtype=torch.bool);mask=torch.ones(1,2,dtype=torch.bool)
    _,old,_=build_indexed_supports(centers,start,end,valid,mask)
    _,new,_=build_indexed_supports(centers,start,end,valid,mask,pre_exclude_front_end=5)
    assert torch.equal(old[0,0],centers<3)
    assert not new[0,0].any()
    assert torch.equal(new[0,1],(centers>5)&(centers<10))


def test_schema_rejects_nonfinite_or_unpaired_physical_features():
    batch=synthetic_datasets(31,2)['train'].batch
    pl=torch.zeros((*batch.interface_mask.shape,1));pc=torch.zeros((*batch.interface_mask.shape,3))
    with pytest.raises(ValueError):replace(batch,phys_loc=pl)
    pl[0,0,0]=float('nan')
    with pytest.raises(ValueError):replace(batch,phys_loc=pl,phys_ctx=pc)
