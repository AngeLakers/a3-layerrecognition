from dataclasses import fields,replace
import pickle
import numpy as np
import pytest
import torch
from test_streaming import toy_catalog
from test_v152_board_timebase import jitter_catalog,POLICY
from a3_ultrasound.speed import (SpeedOptions,speed_context,transfer_batch,validate_cpu_batch,
    finish_step,finite_tensor,dataset_from_config,configure_model,FP32GroupNorm)
from a3_ultrasound.data.streaming import RFShardDataset,_WorkerBatches
from a3_ultrasound.training import ModelConfig,build_model,synthetic_datasets,TensorDataset,evaluate
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.prefix_perturbation import training_conditional

def test_defaults_and_invalid_flags():
    opts=SpeedOptions()
    assert opts.workers==0 and opts.batch_size is None
    assert not any(v for k,v in opts.to_dict().items() if k not in ('prefetch_factor','workers'))
    with pytest.raises(ValueError):SpeedOptions(workers=-1)
    with pytest.raises(ValueError):SpeedOptions(tf32=1)
    with pytest.raises(ValueError):SpeedOptions(batch_size=128)
    combined=SpeedOptions.for_smoke('A_tf32_benchmark')
    assert combined.tf32 and combined.cudnn_benchmark and not combined.bf16_autocast

def test_cpu_structure_rejects_real_invalid_cases():
    batch=synthetic_datasets(23,4)['train'].batch
    validate_cpu_batch(batch,training=True)
    mask=batch.length_mask.clone();mask[0,0]=False
    with pytest.raises(ValueError,match='Noncontiguous'):validate_cpu_batch(replace(batch,length_mask=mask))
    y=batch.y.clone();y[0,0]=-1
    with pytest.raises(ValueError,match='binary'):validate_cpu_batch(replace(batch,y=y),training=True)

def test_fast_path_matches_outputs_gradients_and_does_not_consume_global_RNG():
    torch.manual_seed(23);model=build_model(ModelConfig(backbone='tcn',head_type='proposed_v15'))
    host=synthetic_datasets(23,4)['train'].batch
    out,_=training_conditional(model,host,probability=.2,seed=23,epoch=1,step=0)
    loss=head_training_loss(out,host,head_type='proposed_v15',local_aux_mode='clean_upstream');loss.total.backward()
    expected=loss.total.detach().clone();grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    opts=replace(SpeedOptions.for_smoke('A_no_fused'),workers=0,pin_memory=False,non_blocking=False)
    batch=transfer_batch(host,'cpu',opts,training=True);state=torch.get_rng_state()
    with speed_context(opts,batch,training=True):
        out,_=training_conditional(model,batch,probability=.2,seed=23,epoch=1,step=0)
        loss=head_training_loss(out,batch,head_type='proposed_v15',local_aux_mode='clean_upstream')
        loss.total.backward();finish_step(model,loss.total)
    assert torch.equal(expected,loss.total) and torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(p.grad,grads[k]) for k,p in model.named_parameters() if p.grad is not None)

def test_coalesced_finite_rejects_nan_gradient():
    p=torch.nn.Linear(1,1);p.weight.grad=torch.full_like(p.weight,float('nan'))
    with speed_context(SpeedOptions(coalesced_finite=True)):
        with pytest.raises(RuntimeError,match='Nonfinite'):finish_step(p,torch.tensor(0.))

def test_evaluation_transfer_does_not_authorize_partial_training():
    batch=synthetic_datasets(23,4)['train'].batch
    mask=batch.label_mask.clone();mask[0,0]=False
    opts=SpeedOptions(cpu_structure=True)
    moved=transfer_batch(replace(batch,label_mask=mask),'cpu',opts,training=False)
    with pytest.raises(ValueError,match='Complete training labels'):
        with speed_context(opts,moved,training=True):pass

def test_nonfinite_TOF_rejected_before_GPU_checks_can_be_skipped():
    batch=synthetic_datasets(23,4)['train'].batch
    meta=batch.interface_meta.clone();meta[0,0,1]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite interface metadata'):
        validate_cpu_batch(replace(batch,interface_meta=meta))

def test_validation_cache_reuses_input_with_different_batching_and_unknowns(toy_catalog,monkeypatch):
    import a3_ultrasound.data.streaming as module
    ds=RFShardDataset(toy_catalog,['AL','TC'],tail_policy='nominal_fixed')
    plain=ds.make_batch([0,2,12])
    ds.speed_options=SpeedOptions(validation_cache=True,fast_nominal_fixed=True)
    first=ds.make_batch([0,2,12])
    def forbidden(*a,**kw):raise RuntimeError('RF transform repeated')
    monkeypatch.setattr(module,'transform_trace',forbidden)
    second=ds.make_batch([12,0,2])
    for f in fields(first):
        v=getattr(first,f.name)
        if v is not None:
            assert torch.equal(v,getattr(plain,f.name))
            assert torch.equal(v[[2,0,1]],getattr(second,f.name))

def test_worker_payload_is_small_and_order_preserved(toy_catalog):
    ds=RFShardDataset(toy_catalog,['AL'],training=True,tail_policy='nominal_fixed')
    ds.speed_options=SpeedOptions(workers=1,cpu_structure=True,fast_nominal_fixed=True)
    payload=pickle.dumps(_WorkerBatches(ds));assert len(payload)<20000
    expected=ds.make_batch([0,1,2,3])
    it=ds.batches(4)
    try:got,ids=next(it)
    finally:it.close()
    assert ids.tolist()==[0,1,2,3]
    assert torch.equal(got.x_raw,expected.x_raw) and torch.equal(got.interface_meta,expected.interface_meta)

def test_buckets_keep_same_population_and_only_reorder_training(toy_catalog):
    ds=RFShardDataset(toy_catalog,['AL','TC'],training=True,tail_policy='nominal_fixed')
    ds.speed_options=SpeedOptions(length_buckets=True)
    ds.set_epoch(1);batches=[i for _,i in ds.batches(4)];ids=np.concatenate(batches)
    assert sorted(ids.tolist())==list(range(len(ds)))
    assert all(len({ds._input_length(int(i)) for i in batch})==1 for batch in batches)

def test_factory_blocks_outer_before_any_file_read():
    with pytest.raises(ValueError,match='Outer'):dataset_from_config({'outer_evaluation_permitted':False},'test')

def test_buckets_estimate_board_calibrated_fixed_tail_exactly(jitter_catalog):
    path,table=jitter_catalog
    for bottom in (None,590.164):
        row=dict(table['rows'][0])
        if bottom is not None:row['nominal_bottom']=bottom
        ds=RFShardDataset(path,['A'],training=True,window_policy=POLICY,timebase_config=dict(table,rows=[row]))
        actual=ds.make_batch([0]).x_raw.shape[1]
        assert ds._input_length(0)==actual

def test_bf16_does_not_change_Gaussian_epsilon_or_prior_policy():
    from a3_ultrasound.models.indexed_readout import IndexedReadout
    model=IndexedReadout(1,None,1,prior_strength=.5)
    with torch.no_grad():model.feature_projection.weight.zero_();model.score_projection.weight.zero_()
    features=torch.arange(161,dtype=torch.float32).view(1,161,1)
    support=torch.ones(1,1,161,dtype=torch.bool);mask=torch.ones(1,1,dtype=torch.bool)
    prior=torch.exp(-.5*((torch.arange(161)-80)/40)**2).view(1,1,161);prior/=prior.sum(-1,keepdim=True)
    expected=model(features,None,support,mask,local_prior=prior)
    with speed_context(SpeedOptions(bf16_autocast=True)):
        with torch.autocast('cpu',dtype=torch.bfloat16):got=model(features,None,support,mask,local_prior=prior)
    assert torch.equal(expected.attention,got.attention) and got.attention.dtype==torch.float32

def test_groupnorm_explicit_FP32_keeps_parameters_and_state_keys():
    model=torch.nn.Sequential(torch.nn.GroupNorm(2,4));before=model.state_dict()
    configure_model(model,SpeedOptions(bf16_autocast=True))
    assert isinstance(model[0],FP32GroupNorm)
    assert model(torch.randn(2,4,8).bfloat16()).dtype==torch.float32
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())

def test_evaluation_rows_and_MAP_identical_with_fast_collection():
    torch.manual_seed(23);model=build_model(ModelConfig(backbone='tcn'))
    data=synthetic_datasets(23,4)['validation'];_,expected=evaluate(model,data,batch_size=4)
    opts=SpeedOptions(cpu_structure=True,cpu_correction=True,skip_minibatch_summary=True)
    batch=transfer_batch(data.batch,'cpu',opts)
    small=TensorDataset(batch,data.sample_ids,data.specimen_ids)
    metrics,rows=evaluate(model,small,batch_size=4,speed_options=opts)
    assert metrics=={} and rows==expected

def test_stream_summary_remains_complete_when_leaf_summary_skipped(toy_catalog):
    from a3_ultrasound.stream_training import evaluate_stream
    torch.manual_seed(23);model=build_model(ModelConfig(backbone='tcn'))
    ds=RFShardDataset(toy_catalog,['AL','TC'],tail_policy='nominal_fixed')
    expected=evaluate_stream(model,ds,batch_size=4,device='cpu')
    opts=SpeedOptions(cpu_structure=True,skip_minibatch_summary=True,cpu_correction=True,fast_nominal_fixed=True)
    ds.speed_options=opts;got=evaluate_stream(model,ds,batch_size=4,device='cpu',speed_options=opts)
    assert got==expected and got['reliable']['sample_count']==24

def test_fixed_tail_shortcuts_refuse_active_tail_policy(toy_catalog):
    ds=RFShardDataset(toy_catalog,['AL'])
    ds.speed_options=SpeedOptions(fast_nominal_fixed=True)
    with pytest.raises(ValueError,match='nominal_fixed'):ds.make_batch([0])

def test_exported_source_revision_keeps_uncommitted_status(tmp_path,monkeypatch):
    import json,subprocess
    import a3_ultrasound.training as module
    file=tmp_path/'src/pkg/training.py';file.parent.mkdir(parents=True);file.write_text('')
    monkeypatch.setattr(module,'__file__',str(file))
    def no_git(*a,**k):raise subprocess.CalledProcessError(1,'git')
    monkeypatch.setattr(module.subprocess,'run',no_git)
    p=tmp_path/'SOURCE_REVISION.json';p.write_text(json.dumps(dict(commit='base-commit',dirty=True)))
    assert module._code_version()['dirty'] is True
    p.write_text(json.dumps(dict(commit='base-commit',dirty=False)))
    assert module._code_version()==dict(commit='base-commit',dirty=False,status='frozen SOURCE_REVISION metadata')
