import numpy as np
import pytest
import torch
from a3_ultrasound.data.length_batches import length_bucket_batches,matched_smoke_indices
from a3_ultrasound.speed import SpeedOptions

def test_full_population_homogeneous_batches_and_tails_retained():
    lengths=np.array([691]*7+[900]*9+[1150]*4)
    plan=length_bucket_batches(lengths,4,seed=23,epoch=1)
    flat=np.concatenate(plan)
    assert np.array_equal(np.sort(flat),np.arange(20)) and len(np.unique(flat))==20
    assert all(len(set(lengths[b]))==1 for b in plan)
    assert sorted(len(b) for b in plan)==[1,3,4,4,4,4]

def test_epoch_seed_changes_order_without_using_head_or_global_RNG():
    lengths=np.tile([691,900,1150],80)
    signatures=[]
    for head in ('local_independent','local_pre_independent','vanilla_chain','proposed_v15'):
        np.random.seed(len(head));torch.manual_seed(len(head));state=torch.get_rng_state()
        plan=length_bucket_batches(lengths,16,seed=23,epoch=1)
        signatures.append(tuple(tuple(b) for b in plan));assert torch.equal(state,torch.get_rng_state())
    assert all(s==signatures[0] for s in signatures)
    other=length_bucket_batches(lengths,16,seed=23,epoch=2)
    assert tuple(tuple(b) for b in other)!=signatures[0]
    batch_lengths=[int(lengths[b[0]]) for b in plan]
    assert batch_lengths!=sorted(batch_lengths)

def test_common_smoke_population_gives_exact_steps_and_same_records():
    lengths=np.array([691]*47+[900]*41+[1150]*39)
    chosen=matched_smoke_indices(lengths,4,10,seed=23,epoch=1)
    assert len(chosen)==40 and len(np.unique(chosen))==40 and np.all(np.diff(chosen)>0)
    plan=length_bucket_batches(lengths[chosen],4,seed=23,epoch=1)
    assert len(plan)==10 and all(len(b)==4 for b in plan)
    assert np.array_equal(np.sort(np.concatenate(plan)),np.arange(40))

def test_final_modes_all_exclude_fused_BF16_and_large_batches():
    for mode in ('final_base','final_bucket','final_benchmark'):
        o=SpeedOptions.for_smoke(mode)
        assert o.batch_size==16 and o.tf32 and not o.fused_adam and not o.bf16_autocast
    assert SpeedOptions.for_smoke('final_bucket').length_buckets
    assert SpeedOptions.for_smoke('final_benchmark').cudnn_benchmark

def test_invalid_epoch_and_insufficient_full_smoke_batches_rejected():
    with pytest.raises(ValueError):length_bucket_batches(np.array([691]),16,seed=23,epoch=0)
    with pytest.raises(ValueError):matched_smoke_indices(np.array([691,900]),16,200,seed=23,epoch=1)
