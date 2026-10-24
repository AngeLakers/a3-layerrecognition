import math
import numpy as np
import pytest
import torch

from a3_ultrasound.optimization_pilot import learning_rate, tensor_state_digest, capture_rng, assert_finite_model, promotion_guard, initial_conditions_match
from a3_ultrasound.validation_sampling_audit import cached_metrics, sample_panel, sampling_design, validate_panel_bins
from a3_ultrasound.stream_metrics import PredictionAccumulator


def test_lr_boundaries_and_global_resume_index():
    assert learning_rate('C',1)==.001
    assert learning_rate('L',1)==pytest.approx(.001/3)
    assert learning_rate('LW',1)==pytest.approx(.001/30)
    assert learning_rate('LW',9774)==pytest.approx(.001/3)
    assert learning_rate('LW',9775)==pytest.approx(.001/3)
    assert learning_rate('LW',32580)==pytest.approx(.001/3)
    # A checkpoint's completed update index is all the scheduler needs to resume.
    continuous=[learning_rate('LW',i) for i in range(9772,9777)]
    resumed=[learning_rate('LW',9773+i) for i in range(1,4)]
    assert resumed==continuous[2:]
    for arm,step in [('X',1),('C',0),('LW',True)]:
        with pytest.raises(ValueError):learning_rate(arm,step)


def test_initial_guard_accepts_optimizer_tuple_after_json_roundtrip():
    import json
    parameter=torch.nn.Parameter(torch.zeros(1));optimizer=torch.optim.Adam([parameter],lr=.001)
    current=dict(model_state='same-weight-digest',optimizer_empty_state=not bool(optimizer.state),
        optimizer_groups=[{k:v for k,v in g.items() if k not in ('params','lr','initial_lr')} for g in optimizer.param_groups],
        training_refs='same-selection',first_epoch_order='same-order',first_batch_refs=[[0,1]])
    reference=json.loads(json.dumps(current))
    assert current!=reference
    assert initial_conditions_match(current,reference)
    assert isinstance(optimizer.param_groups[0]['betas'],tuple)
    assert not optimizer.state
    for key in ('model_state','training_refs','first_epoch_order'):
        changed=dict(current);changed[key]='different'
        assert not initial_conditions_match(changed,reference)
    changed=json.loads(json.dumps(current));changed['optimizer_groups'][0]['betas'][1]=.99
    assert not initial_conditions_match(changed,reference)
    changed=json.loads(json.dumps(current));changed['optimizer_empty_state']=False
    assert not initial_conditions_match(changed,reference)


def fixture_cache():
    y=np.array([[0,0,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0]])
    mask=np.array([[1,1,0,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[1,1,1,0,0,0]],bool)
    p=np.array([[.2,.8,0,0,0,0],[.9,.4,0,0,0,0],[.1,.5,0,0,0,0],[.7,.2,.8,0,0,0]],np.float32)
    meta=dict(y=y,mask=mask,K=np.array([2,2,2,3]),entity=np.array([0,0,0,1]),entities=np.array(['a','b']))
    pred=dict(marginals=p,true_nll=np.array([.2,.3,0,.4]),map_correct=np.array([1,0,0,1],bool))
    return meta,pred


def test_cache_reducer_matches_public_weighted_accumulator():
    meta,pred=fixture_cache();w=np.array([9,1,13,1],np.int64)
    expected=PredictionAccumulator()
    for i,k in enumerate(meta['K']):
        y=meta['y'][i,:k].tolist();m=meta['mask'][i,:k].tolist();p=pred['marginals'][i,:k].tolist()
        lp=[-float(pred['true_nll'][i])]*(2**int(k))
        expected.add(dict(specimen_id=meta['entities'][meta['entity'][i]],y=y,label_mask=m,marginals=p,
            map_states=y if pred['map_correct'][i] else [1-v for v in y],log_probabilities=lp,
            base_probabilities=p,base_logodds=[0]*k,raw_logZ=0,rho=None,sample_weight=int(w[i])))
    a=cached_metrics(meta,pred,weights=w);b=expected.result()
    assert a['joint_nll']==pytest.approx(b['joint_nll'])
    assert a['joint_map_accuracy']==pytest.approx(b['joint_map_accuracy'])
    assert a['deepest_specimen_macro_f1']==pytest.approx(b['deepest_specimen_macro_f1'])
    assert a['joint_complete_label_count']==b['joint_complete_label_count']
    for e in ('a','b'):
        assert a['deepest_per_specimen'][e]['confusion_matrix_0_1']==b['deepest_per_specimen'][e]['confusion_matrix_0_1']
        assert a['deepest_per_specimen'][e]['brier']==pytest.approx(b['deepest_per_specimen'][e]['brier'])


def test_sampling_reuses_metadata_preserves_all_known_positive_rows():
    n=101
    meta=dict(y=np.zeros((n,6),np.int64),mask=np.zeros((n,6),bool),K=np.full(n,2),
        entity=np.zeros(n,np.int64),entities=np.array(['a']),xy=np.column_stack((np.arange(n),np.zeros(n))),
        quality=np.ones(n,bool),ordinal=np.full(n,7))
    meta['mask'][:,:2]=True;meta['y'][[3,9],1]=1
    ids,w=sample_panel(meta,target=11,seed=2026102301)
    assert len(ids)==11 and w.sum()==n and np.isin([3,9],ids).all()
    assert np.all(w[np.isin(ids,[3,9])]==1)
    design,report=sampling_design(meta,11)
    validate_panel_bins(meta,ids,w,design,report)
    wrong=w.copy();wrong[np.flatnonzero(w>1)[0]]+=1
    with pytest.raises(RuntimeError,match='Persisted stratum/bin'):
        validate_panel_bins(meta,ids,wrong,design,report)
    ids2,w2=sample_panel(meta,target=11,seed=2026102301)
    assert np.array_equal(ids,ids2) and np.array_equal(w,w2)


def test_epoch_checkpoint_roundtrip_retains_rng_and_model(tmp_path):
    model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
    generator=torch.Generator().manual_seed(23);torch.randperm(17,generator=generator)
    path=tmp_path/'epoch.pt';torch.save(dict(state=model.state_dict(),rng=capture_rng(generator),updates=9774),path)
    saved=torch.load(path,weights_only=True)
    assert tensor_state_digest(saved['state'])==tensor_state_digest(model.state_dict())
    restored=torch.Generator();restored.set_state(saved['rng']['training_order'])
    assert torch.equal(torch.randperm(17,generator=restored),torch.randperm(17,generator=generator))
    assert learning_rate('LW',saved['updates']+1)==pytest.approx(.001/3)
    assert_finite_model(model)
    with torch.no_grad():model[0].weight[0,0]=math.nan
    with pytest.raises(RuntimeError,match='Nonfinite model'):assert_finite_model(model)


def test_promotion_guard_requires_calibration_and_every_board_fnr():
    def result(f1,nll,brier,fnr):
        return {'metrics':dict(deepest_specimen_macro_f1=f1,joint_nll=nll,deepest_brier=brier,
                               deepest_per_specimen={'a':{'false_negative_rate':fnr}})}
    baseline=result(.7,.1,.05,.1)
    assert promotion_guard(result(.706,.099,.049,.109),baseline)['passed']
    assert not promotion_guard(result(.706,.101,.049,.109),baseline)['passed']
    assert not promotion_guard(result(.706,.099,.049,.111),baseline)['passed']
    assert not promotion_guard(result(.704,.099,.049,.1),baseline)['passed']
