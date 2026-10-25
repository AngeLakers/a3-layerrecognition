import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from a3_ultrasound.data.board_timebase import (
    envelope_anchor_positions, divided_centers, shared_window_shift, selected_prior, adoption_decision)
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.data.quick_input import transform_trace
from a3_ultrasound.data.physical_features import energy_log_ratios,packed_energy_bounds

POLICY={'name':'board_timebase','halfwidth_samples':80}
JITTER={'enabled':True,'max_shift_samples':24,'seed':23,'distribution':'uniform'}


def test_envelope_anchors_share_definition_and_polarity_invariance():
    t=np.arange(900)
    x=3+np.exp(-((t-74)/8)**2)*np.cos(.3*(t-74))+.5*np.exp(-((t-600)/8)**2)*np.cos(.3*(t-600))
    start,bottom,complete,edge=envelope_anchor_positions(x,590.)
    assert start.tolist()==[74.] and bottom.tolist()==[600.]
    assert complete.all() and not edge.any()
    other=envelope_anchor_positions(-x,590.)
    for a,b in zip((start,bottom,complete,edge),other):np.testing.assert_array_equal(a,b)
    start,bottom,complete,_=envelope_anchor_positions(x[:650],590.)
    assert start[0]==74 and np.isnan(bottom[0]) and not complete[0]


def test_rf_reader_rejects_label_access_before_io(monkeypatch):
    p=Path(__file__).parents[1]/'scripts/a3_v152_calibrate_rf.py'
    spec=importlib.util.spec_from_file_location('rf_calibrate',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def forbidden(*args,**kwargs):raise RuntimeError('I/O must not be reached')
    monkeypatch.setattr(module.np,'load',forbidden)
    for field in ('y','label_mask','eligible','design_y'):
        with pytest.raises(AssertionError,match='forbidden'):
            module.guarded_load(field,'MISSING',[])


def test_division_and_prior_ignore_nominal_material_parameters():
    delta,tau=divided_centers(74.,608.,2)
    assert delta==178 and tau==[252.,430.]
    row=dict(selected_tau=tau,selected_bottom=608.)
    for shift in (-24.,0.,24.):
        prior=selected_prior(row,2,850,shift=shift)
        assert prior.nominal_tof_samples==tuple(v+shift for v in tau)
        assert prior.window_left_samples==80 and prior.backwall_tof_samples==608.
    with pytest.raises(ValueError,match='within RF'):
        selected_prior(row,2,510)


def test_global_adoption_uses_median_absolute_and_cell_absolute_median():
    r=[dict(board='A',k=2,old_offset=a,calibrated_offset=b) for a,b in [(-9,-4),(9,4)]]
    d=adoption_decision(r)
    assert d['adopt_calibration'] and d['pooled_absolute_offset_median_old']==9
    assert d['pooled_absolute_offset_median_calibrated']==4
    guard=r+[dict(board='B',k=2,old_offset=10,calibrated_offset=31)]
    assert adoption_decision(guard)['pooled_improves']
    assert not adoption_decision(guard)['adopt_calibration']
    guard[-1]['calibrated_offset']=30
    assert adoption_decision(guard)['adopt_calibration']
    assert not adoption_decision([dict(board='A',k=2,old_offset=5,calibrated_offset=-5)])['pooled_improves']


@pytest.fixture
def jitter_catalog(tmp_path):
    config=dict(time_reference='cropped_record_origin_is_zero',batch_reference_samples=0,
        sampling_rate_hz=200000000,sound_speed_mps=dict(TC4=6100,AL=6320),window_total_us=.7,
        normalization_divisor=131,tail=dict(policy='nominal_fixed',nominal_backwall_guard_us=.5,
            active_boundary_check_samples=32,quiet_run_samples=32,quiet_floor_raw_units=3,quiet_guard_samples=64))
    n=4;length=850;t=np.arange(length);x=np.stack([np.sin(t*.15)*(1+i/4) for i in range(n)]).astype(np.float32)
    np.save(tmp_path/'RF.npy',x.ravel())
    y=np.array([[0,0],[1,0],[0,1],[1,1]],np.int8);lm=np.ones((n,2),bool)
    arrays=dict(offsets=np.arange(n)*length,lengths=np.full(n,length),source_row=np.arange(n)+10,
        y=y,label_mask=lm,eligible=np.ones(n,bool),design_y=y,design_mask=lm,
        xy=np.column_stack([np.arange(n),np.arange(n)]),quality=np.ones(n,bool))
    for key,value in arrays.items():np.save(tmp_path/(key+'.npy'),value)
    entry=dict(entity='A',ordinal=4,K=2,group='TCL3',layer_mm=3,index_dir='.',rf_path=str(tmp_path/'RF.npy'))
    path=tmp_path/'catalog.json';path.write_text(json.dumps(dict(input_config=config,entities=[entry])))
    table=dict(_artifact_sha256='fixture',rows=[dict(board='A',selected_tau=[252.,430.],selected_bottom=608.)])
    return path,table


def test_training_shared_jitter_changes_all_windows_and_energy_but_not_rf(jitter_catalog):
    path,table=jitter_catalog;sequences=[]
    for head in ('local_independent','local_pre_independent','vanilla_chain','proposed_v15'):
        ds=RFShardDataset(path,['A'],training=True,window_policy=POLICY,timebase_config=table,window_jitter=JITTER)
        ds.set_epoch(3)
        ds.physical_config=dict(front_end_samples=180,epsilon=1e-8,mean=0.,std=1.)
        batch=ds.make_batch(np.arange(len(ds)));seq=[]
        for i,(s,row) in enumerate(ds.refs):
            ar=ds.shards[s]['arrays'];expected=shared_window_shift(23,3,4,int(ar['source_row'][row]))
            centers=batch.interface_meta[i,:2,1].numpy()*int(batch.length_mask[i].sum())
            np.testing.assert_allclose(centers,np.array([252.,430.])+expected,atol=1e-4,rtol=0)
            seq.append(expected)
            raw=ds.shards[s]['rf'][ar['offsets'][row]:ar['offsets'][row]+850]
            plain,_,_=transform_trace(raw,ds.shards[s],ds.config,timebase=table['rows'][0])
            np.testing.assert_array_equal(batch.x_raw[i].numpy(),plain)
            f=energy_log_ratios(plain,np.array([252.,430.])+expected-80,np.array([252.,430.])+expected+80,front_end=180)
            np.testing.assert_allclose(batch.phys_loc[i,:2,0].numpy(),f,rtol=0,atol=1e-6)
        sequences.append(seq)
        # Call order does not advance a hidden augmentation RNG.
        again=ds.make_batch([0]);np.testing.assert_array_equal(again.interface_meta[0].numpy(),batch.interface_meta[0].numpy())
    assert all(v==sequences[0] for v in sequences)
    assert np.all(np.abs(sequences[0])<=24)


def test_evaluation_never_jitters_even_if_strategy_is_present(jitter_catalog):
    path,table=jitter_catalog
    ds=RFShardDataset(path,['A'],training=False,window_policy=POLICY,timebase_config=table,window_jitter=JITTER)
    ds.set_epoch(99);batch=ds.make_batch([0,1])
    centers=batch.interface_meta[:,:2,1].numpy()*batch.length_mask.sum(1).numpy()[:,None]
    np.testing.assert_allclose(centers,np.array([[252.,430.],[252.,430.]]),rtol=0,atol=1e-4)


def test_calibration_and_jitter_preserve_v151_retained_rf(jitter_catalog):
    path,table=jitter_catalog
    rawcfg=json.loads(path.read_text())['input_config'];row=dict(K=2,group='TCL3',layer_mm=3)
    raw=np.arange(850,dtype=np.float32)
    old=transform_trace(raw,row,rawcfg)
    cfg=dict(rawcfg,window_halfwidth_policy='board_timebase')
    timebase=dict(table['rows'][0],nominal_bottom=590.1639344262295)
    for shift in (-24.,24.):
        new=transform_trace(raw,row,cfg,timebase=timebase,window_shift=shift)
        assert new[0].tobytes()==old[0].tobytes() and new[2]['valid_length']==old[2]['valid_length']


@pytest.mark.parametrize('shift',[-21.999999743896655,-12.999967613724793,-12.000031344744123,-24.,0.,24.])
def test_energy_exactly_matches_model_integer_support_after_float32_packing(shift):
    from a3_ultrasound.models.integrated import A3PCCModel
    from a3_ultrasound.models.indexed_readout import build_indexed_supports
    from a3_ultrasound.schema import InterfaceBatch
    import torch
    n=691;k=2;centers=np.array([255.,435.])+shift
    meta=np.zeros((6,5),np.float32);meta[:k,0]=[.5,1.];meta[:k,1]=centers/n
    meta[:k,2]=[1/3,2/3];meta[:k,3:5]=80/n
    x=np.zeros(n,np.float32);x[153]=2;x[154]=3;x[333]=5;x[334]=7;x[493]=11
    im=torch.tensor([[True,True,False,False,False,False]])
    batch=InterfaceBatch(x_raw=torch.from_numpy(x[None]),length_mask=torch.ones(1,n,dtype=torch.bool),
        interface_meta=torch.from_numpy(meta[None]),interface_mask=im)
    starts,ends,tau=A3PCCModel._sample_windows(batch)
    mask,pre,_=build_indexed_supports(torch.arange(n,dtype=torch.float32),starts,ends,batch.length_mask,im,
        prior_center=tau,prior_sigma_samples=40.)
    left,right=packed_energy_bounds(meta,n,k)
    np.testing.assert_array_equal(left,starts[0,:k].numpy());np.testing.assert_array_equal(right,ends[0,:k].numpy())
    got=energy_log_ratios(x,left,right,front_end=180)
    expected=np.array([np.log((np.sum(x[mask[0,j].numpy()].astype(float)**2)+1e-8)/(np.sum(x[:181].astype(float)**2)+1e-8)) for j in range(k)])
    np.testing.assert_array_equal(got,expected)
    assert not torch.any(mask&pre)
