import json
from pathlib import Path
import numpy as np
import pytest
from a3_ultrasound.data.quick_input import physical_manifest, transform_trace

CONFIG = json.loads((Path(__file__).parents[1]/'configs/experiment/a3_13board_query07_v2.json').read_text(encoding='utf-8'))

def test_thin_al_zero_reference_and_new_effective_length():
    trace=np.zeros(1200,dtype=np.float32);trace[190]=131
    x,meta,report=transform_trace(trace,{'K':2,'layer_mm':3,'group':'AL'},CONFIG)
    assert report['nominal_tof_samples']==pytest.approx([189.87341772,379.74683544])
    assert report['halfwidth_samples']==pytest.approx(70)
    assert len(x)==670 and x[190]==1
    assert meta[0,1]*len(x)==pytest.approx(189.87341772,abs=2e-5)
    assert not meta[2:].any()

def test_active_tail_is_extended_without_touching_preserved_values():
    trace=np.zeros(1200,dtype=np.float32);trace[660:700]=20
    x,_,report=transform_trace(trace,{'K':2,'layer_mm':3,'group':'AL'},CONFIG)
    assert report['active_boundary_extended'] and len(x)==764
    np.testing.assert_array_equal(x,trace[:764]/np.float32(131))

def test_unknown_thickness_and_nonfinite_signal_rejected():
    with pytest.raises((ValueError,TypeError)):
        physical_manifest({'K':2,'layer_mm':None,'group':'TCL3'},CONFIG)
    with pytest.raises(ValueError):
        transform_trace(np.array([np.nan]),{'K':2,'layer_mm':3,'group':'AL'},CONFIG)


def test_quiet_gap_does_not_discard_late_packet():
    config=dict(CONFIG,tail=dict(CONFIG['tail'],policy='last_active_tail'))
    trace=np.zeros(1200,dtype=np.float32);trace[787]=30
    x,meta,report=transform_trace(trace,{'K':2,'layer_mm':3,'group':'AL'},config)
    assert len(x)==852 and x[787]==pytest.approx(30/131)
    assert report['tail_extended'] and not report['active_boundary_extended']
    assert meta[1,1]*len(x)==pytest.approx(379.74683544,abs=3e-5)


def test_active_tail_end_keeps_all_existing_samples():
    config=dict(CONFIG,tail=dict(CONFIG['tail'],policy='last_active_tail'))
    trace=np.zeros(1200,dtype=np.float32);trace[-1]=4
    x,_,_=transform_trace(trace,{'K':2,'layer_mm':3,'group':'AL'},config)
    assert len(x)==len(trace)
