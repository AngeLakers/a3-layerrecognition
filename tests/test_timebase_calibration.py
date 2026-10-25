import numpy as np
import pytest
import importlib.util
from pathlib import Path

from a3_ultrasound.data.timebase_calibration import first_negative_trough, resolve_saturation, complete_negative_trough


def test_first_segment_not_later_deeper_echo():
    x = np.array([[0, -4, -6, -4, 0, -10, -9, 0]], dtype=np.float32)
    r = first_negative_trough(x, 11)
    assert r.positions.tolist() == [13]
    assert r.crossings.tolist() == [12]
    assert not r.failed[0] and not r.plateau[0]


def test_saturation_requires_three_equal_global_minima():
    x = np.array([[0, -4, -8, -8, -8, -4, 0],
                  [0, -4, -8, -8, -7, -4, 0]], dtype=np.float32)
    r = first_negative_trough(x)
    pos, sat = resolve_saturation(r, [-8, -8])
    assert pos.tolist() == [1, 2] and sat.tolist() == [True, False]
    pos, sat = resolve_saturation(r, [-9, -8])
    assert pos.tolist() == [2, 2] and not sat.any()


def test_strict_threshold_and_failed():
    r = first_negative_trough(np.array([[10, -3, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    assert r.failed.all() and (r.positions == -1).all()


def test_recovery_at_equal_threshold_ends_segment():
    r = first_negative_trough(np.array([[10, -4, -3, -9, 0]], dtype=np.float32))
    assert r.positions[0] == 1


def test_unclosed_and_nonfinite():
    r = first_negative_trough(np.array([[0, -4, -6]], dtype=np.float32))
    assert r.unclosed[0] and r.positions[0] == 2
    with pytest.raises(ValueError):
        first_negative_trough(np.array([[np.nan]]))


def test_half_cycle_continues_beyond_search_window():
    x=np.array([0, -4, -6, -9, -8, 0, -20, 0],np.float32)
    r=complete_negative_trough(x,0,2)
    assert r == dict(position=3,crossing=1,saturated=False,unclosed=False)


def test_complete_segment_saturation_and_record_end():
    x=np.array([0,-4,-8,-8,-8],np.float32)
    r=complete_negative_trough(x,0,2)
    assert r == dict(position=1,crossing=1,saturated=True,unclosed=True)


def test_calibration_cannot_open_labels_or_eligibility(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/a3_v152_timebase_diagnose.py'
    spec=importlib.util.spec_from_file_location('v152_diagnosis',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=np.zeros(1200,np.float32)
    raw[10:13]=[-4,-8,-4];raw[599:602]=[-4,-7,-4]
    values=dict(rf=raw,offsets=np.array([0],np.int64),lengths=np.array([1200],np.int64),source_row=np.array([7],np.int64))
    for name,value in values.items():
        np.save(tmp_path/(name+'.npy'),value)
    for name in ('y','label_mask','eligible','quality','design_y','design_mask'):
        (tmp_path/(name+'.npy')).write_bytes(b'Forbidden label file: must not be opened by calibration')
    board=dict(board='synthetic',K=2,layer_mm=3,nominal_c_mps=6100,records=1,
               rf_path=str(tmp_path/'rf.npy'),**{key+'_path':str(tmp_path/(key+'.npy')) for key in ('offsets','lengths','source_row')})
    output=tmp_path/'out';output.mkdir()
    fit=module.fit_boards(dict(sampling_rate_hz=200000000,boards=[board]),output)
    assert fit['boards'][0]['t0']['median']==11
    assert fit['boards'][0]['backwall']['count']==1
    import json
    audit=json.loads((output/'rf_only_access_audit.json').read_text(encoding='utf-8'))
    assert len(audit['opened_paths'])==4 and audit['forbidden_label_access_count']==0
