"""V1.5.2 checks prepared for execution after the user's preview approval."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from a3_ultrasound.data.quick_input import physical_manifest, transform_trace
from a3_ultrasound.data.time_prior import derive_time_prior_samples
from a3_ultrasound.data.physical_features import energy_log_ratios
from a3_ultrasound.data.window_diagnostics import packet_span, span_covered, front_overlap_energy_fraction
from a3_ultrasound.models.indexed_readout import build_indexed_supports
from a3_ultrasound.v15_protocol import validate_config, load_physical_artifacts


def input_config():
    return dict(time_reference='cropped_record_origin_is_zero', batch_reference_samples=0,
        sampling_rate_hz=200000000, sound_speed_mps=dict(TC4=6100, AL=6320), window_total_us=.7,
        window_halfwidth_policy='half_interval_capped', window_max_halfwidth_samples=110,
        normalization_divisor=131, tail=dict(policy='nominal_fixed',nominal_backwall_guard_us=.5,
            active_boundary_check_samples=32,quiet_run_samples=32,quiet_floor_raw_units=3,quiet_guard_samples=64))


@pytest.mark.parametrize('group,h,k,half',[('TCL3',3,2,98),('TCL4',4,3,110),('AL',3,2,94),('AL',10,2,110)])
def test_geometry_keeps_centers_and_tail_and_updates_energy_windows(group,h,k,half):
    cfg=input_config();row=dict(group=group,layer_mm=h,K=k)
    old=copy.deepcopy(cfg);old.pop('window_halfwidth_policy');old.pop('window_max_halfwidth_samples')
    raw=np.random.default_rng(23).normal(size=2500).astype(np.float32)
    x,meta,info=transform_trace(raw,row,cfg);xo,mo,io=transform_trace(raw,row,old)
    assert x.tobytes()==xo.tobytes() and info['valid_length']==io['valid_length']
    assert meta[:k,:3].tobytes()==mo[:k,:3].tobytes()
    assert info['halfwidth_samples']==half and info['nominal_tof_samples']==io['nominal_tof_samples']
    prior=derive_time_prior_samples(physical_manifest(row,cfg),trace_length=len(raw))
    assert all(a>b for a,b in zip(prior.window_start_samples[1:],prior.window_end_samples[:-1]))
    got=energy_log_ratios(x,prior.window_start_samples,prior.window_end_samples,front_end=180)
    expected=[];xx=x.astype(np.float64)
    for l,r in zip(prior.window_start_samples,prior.window_end_samples):
        expected.append(np.log((np.sum(xx[int(np.ceil(l)):int(np.floor(r))+1]**2)+1e-8)/(np.sum(xx[:181]**2)+1e-8)))
    np.testing.assert_array_equal(got,expected)


def test_sigma72_strength_half_penalty_and_support_boundaries():
    mask,_,prior=build_indexed_supports(feature_centers=torch.tensor([0.,48.,99.]),
        window_start=torch.tensor([[-98.]]),window_end=torch.tensor([[98.]]),
        feature_mask=torch.ones(1,3,dtype=torch.bool),interface_mask=torch.ones(1,1,dtype=torch.bool),
        prior_sigma_samples=72.)
    penalty=.5*(prior[0,0,1].log()-prior[0,0,0].log())
    assert penalty.item()==pytest.approx(-1/9,abs=1e-6)
    assert mask.tolist()==[[[True,True,False]]]


def test_packet_span_does_not_clip_to_either_compared_window():
    e=np.zeros(500);e[190:310]=.3;e[250]=1
    span=packet_span(e,250)
    assert (span['start'],span['end'])==(190,309)
    assert not span_covered(span,250,40) and span_covered(span,250,70)
    e[:]=0;e[100:]=.3;e[250]=1
    span=packet_span(e,250)
    assert span['source_censored'] and not span_covered(span,250,300)


def test_front_overlap_is_a_proxy_with_inclusive_end180():
    x=np.zeros(500);x[180]=2;x[200]=3
    assert front_overlap_energy_fraction(x,200,70)==pytest.approx(4/13)


def test_versioned_config_rejects_old_sigma_and_old_statistics(tmp_path):
    path=Path(__file__).parents[1]/'configs/experiment/v1_5_2_freeze.json'
    cfg=json.loads(path.read_text(encoding='utf-8'))
    validate_config(cfg,fold_name='F1',seed=23,head='proposed_v15',require_permission=False)
    bad=copy.deepcopy(cfg);bad['model']['local_prior_sigma_samples']=36.
    with pytest.raises(ValueError,match='sigma40'):
        validate_config(bad,fold_name='F1',seed=23,head='proposed_v15',require_permission=False)
    for name in ('physical_statistics','m6_decision'):
        p=tmp_path/(name+'.json');p.write_text('{}')
        cfg[name+'_path']=str(p);cfg[name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='different window policy'):
        load_physical_artifacts(cfg)


def test_dataset_rejects_missing_policy_and_different_effective_inputs(tmp_path):
    from a3_ultrasound.data.streaming import RFShardDataset
    path=tmp_path/'catalog.json';cfg=input_config()
    cfg.pop('window_halfwidth_policy');cfg.pop('window_max_halfwidth_samples')
    path.write_text(json.dumps(dict(input_config=cfg,entities=[])))
    policy=dict(name='half_interval_capped',cap_samples=110)
    with pytest.raises(ValueError,match='window policy disagree'):
        RFShardDataset(path,['unused'],physical_config=dict(input_window_policy=policy))
    with pytest.raises(ValueError,match='effective dataset input config disagree'):
        RFShardDataset(path,['unused'],window_policy=policy,
            physical_config=dict(input_window_policy=policy,effective_input_config={}))
