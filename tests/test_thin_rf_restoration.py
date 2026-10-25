"""Regression tests for explicit per-board input-prefix restoration."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from a3_ultrasound.data.quick_input import transform_trace


@pytest.fixture
def setup_trace():
    cfg=dict(time_reference='cropped_record_origin_is_zero',batch_reference_samples=0,
        sampling_rate_hz=200000000,sound_speed_mps=dict(TC4=6100,AL=6320),window_total_us=.7,
        normalization_divisor=131,window_halfwidth_policy='board_timebase',
        tail=dict(policy='nominal_fixed',nominal_backwall_guard_us=.5,
            active_boundary_check_samples=32,quiet_run_samples=32,quiet_floor_raw_units=3,quiet_guard_samples=64))
    tb=dict(selected_tau=[225+2/3,372+1/3],selected_bottom=519.,nominal_bottom=3*2*.003*200e6/6320)
    raw=np.zeros(1343,np.float32);raw[783:790]=[2,4,9,-8,-6,-2,1]
    row=dict(entity='thin',group='AL',layer_mm=3,K=2)
    return raw,row,cfg,tb


@pytest.mark.parametrize('prefix',[900,1314])
def test_fixed_prefix_recovers_late_packet_without_moving_RF_or_centers(setup_trace,prefix):
    raw,row,cfg,tb=setup_trace
    before=transform_trace(raw,row,cfg,timebase=tb)
    assert len(before[0])==670
    cfg=copy.deepcopy(cfg);cfg['tail']['fixed_prefix_samples_by_entity']={'thin':prefix}
    for shift in (-24.,0.,24.):
        x,meta,info=transform_trace(raw,row,cfg,timebase=tb,window_shift=shift)
        assert len(x)==prefix and np.array_equal(x,raw[:prefix]/np.float32(131))
        assert np.array_equal(x[:670],before[0]) and np.any(x[783:790])
        np.testing.assert_allclose(meta[:2,1]*prefix,np.array(tb['selected_tau'])+shift,atol=1e-4,rtol=0)
        assert info['proposed_stop']==prefix and info['tail_policy']=='nominal_fixed'


def test_override_does_not_change_other_board_RF_meta_or_info(setup_trace):
    raw,row,cfg,tb=setup_trace;row=dict(row,entity='other')
    old=transform_trace(raw,row,cfg,timebase=tb)
    cfg=copy.deepcopy(cfg);cfg['tail']['fixed_prefix_samples_by_entity']={'thin':1314}
    new=transform_trace(raw,row,cfg,timebase=tb)
    assert new[0].tobytes()==old[0].tobytes() and new[1].tobytes()==old[1].tobytes() and new[2]==old[2]


@pytest.mark.parametrize('value',[True,0,-1,1314.0,'1314'])
def test_override_rejects_invalid_lengths(setup_trace,value):
    raw,row,cfg,tb=setup_trace;cfg['tail']['fixed_prefix_samples_by_entity']={'thin':value}
    with pytest.raises(ValueError,match='positive integer'):
        transform_trace(raw,row,cfg,timebase=tb)


def test_override_refuses_short_source_instead_of_silent_padding(setup_trace):
    raw,row,cfg,tb=setup_trace;cfg['tail']['fixed_prefix_samples_by_entity']={'thin':1314}
    with pytest.raises(ValueError,match='complete source RF'):
        transform_trace(raw[:1000],row,cfg,timebase=tb)


def test_RF_restore_reader_forbids_label_access_before_IO(monkeypatch):
    path=Path(__file__).parents[1]/'scripts/a3_v152_restore_thin_rf.py'
    spec=importlib.util.spec_from_file_location('restore_thin',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def forbidden(*args,**kwargs):raise RuntimeError('I/O must not be reached')
    monkeypatch.setattr(m.np,'load',forbidden)
    for field in ('y','label_mask','eligible','design_y','design_mask','quality'):
        with pytest.raises(AssertionError,match='forbidden'):
            m.guarded_load(field,'MISSING',[])
