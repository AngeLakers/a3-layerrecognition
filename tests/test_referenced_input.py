import numpy as np
import pytest

from a3_ultrasound.data.referenced_input import reference_geometry,available_window_mask,prepare_referenced_trace
from a3_ultrasound.data.quick_input import transform_trace


def geom(**updates):
    kwargs=dict(K=2,layer_mm=3.,sampling_rate_hz=200000000.,sound_speed_mps=6100.,
                reference_samples=86.,source_length=800)
    kwargs.update(updates);return reference_geometry(**kwargs)


def test_reference_added_once_and_storage_translation_equivalent():
    raw=geom();aligned=geom(storage_shift=170)
    np.testing.assert_allclose(aligned.interface_centers,raw.interface_centers+170)
    assert aligned.reference==256 and raw.reference==86
    np.testing.assert_array_equal(raw.local_available_counts,aligned.local_available_counts)
    assert raw.backwall_available_count==aligned.backwall_available_count


def test_bottom_truncation_is_availability_not_record_failure():
    g=geom(source_length=700)
    assert g.local_complete.all()
    assert 0<g.backwall_available_count<g.backwall_declared_count
    assert not geom(source_length=549).local_complete.all()


def test_source_mask_is_distinct_from_interface_or_label_masks():
    positions=np.arange(8);source=np.array([0,0,1,1,1,1,0,0],bool)
    m=available_window_mask(positions,source,1,6)
    np.testing.assert_array_equal(m,source)
    with pytest.raises(ValueError):
        available_window_mask(positions,np.ones(8),0,7)


def test_fractional_window_endpoint_matches_model_contract():
    g=geom(sound_speed_mps=6000.,reference_samples=86.5,source_length=557)
    assert g.local_complete.all()
    assert not g.local_model_support_complete.all()
    cfg=config();cfg['sound_speed_mps']['TC4']=6000.
    with pytest.raises(ValueError,match='Target Local support'):
        prepare_referenced_trace(np.ones(557,np.float32),dict(K=2,layer_mm=3,group='TCL3'),cfg,
                                 board_reference_samples=86.5)
    _,_,info=prepare_referenced_trace(np.ones(700,np.float32),dict(K=2,layer_mm=3,group='TCL3'),config(),
                                    board_reference_samples=86.)
    assert info['backwall_source_mask'].sum()==info['backwall_available_count']
    assert info['backwall_source_mask'].size==info['backwall_declared_count']
    assert not info['backwall_source_mask'].all()


def config():
    return dict(time_reference='cropped_record_origin_is_zero',batch_reference_samples=0,
        window_total_us=.7,sampling_rate_hz=200000000,sound_speed_mps={'TC4':6100,'AL':6320},normalization_divisor=131,
        tail=dict(nominal_backwall_guard_us=.5,active_boundary_check_samples=32,quiet_run_samples=32,
                  quiet_floor_raw_units=3,quiet_guard_samples=64,policy='nominal_fixed'))


def test_off_path_is_bitwise_identical_and_on_retains_source():
    row=dict(K=2,layer_mm=3,group='TCL3')
    raw=np.random.default_rng(23).normal(size=850).astype(np.float32)
    old=transform_trace(raw,row,config());off=prepare_referenced_trace(raw,row,config())
    assert old[0].tobytes()==off[0].tobytes() and old[1].tobytes()==off[1].tobytes() and old[2]==off[2]
    x,meta,info=prepare_referenced_trace(raw,row,config(),board_reference_samples=86)
    assert len(x)==len(raw) and x.tobytes()==(raw/np.float32(131)).tobytes()
    np.testing.assert_allclose(meta[:2,1]*len(x),geom(source_length=850).interface_centers,atol=5e-5)
    assert info['physical_normalization_refit_required'] and info['source_mask'].all()


def test_position2_query_stays_same_while_prior_moves_and_masks_padding():
    import torch
    from a3_ultrasound.models.query import InterfaceQueryEncoder
    from a3_ultrasound.models.indexed_readout import build_indexed_supports
    torch.manual_seed(23)
    old=torch.zeros((1,6,5),dtype=torch.float32)
    old[0,:2,0]=torch.tensor([.5,1.]);old[0,:2,2]=torch.tensor([1/3,2/3])
    old[0,:2,1]=torch.tensor([200.,400.])/800
    old[0,:2,3:5]=70/800
    new=old.clone();new[0,:2,1]+=86/800
    interface=torch.tensor([[True,True,False,False,False,False]])
    query=InterfaceQueryEncoder(8,query_view='position2')
    assert torch.equal(query(old,interface),query(new,interface))
    centers=torch.arange(800,dtype=torch.float32)
    available=(centers<600)[None]
    def supports(meta):
        tau=meta[:,:,1]*800;half=meta[:,:,3]*800
        return build_indexed_supports(centers,tau-half,tau+half,available,interface,
                                      prior_center=tau,prior_sigma_samples=36)
    oldlocal,_,oldprior=supports(old);newlocal,_,newprior=supports(new)
    assert oldprior[0,0].argmax()==200 and newprior[0,0].argmax()==286
    assert not torch.equal(oldlocal,newlocal)
    assert not newlocal[:,:,600:].any() and not newprior[:,:,600:].any()
