import copy
import hashlib
import json
from pathlib import Path

import pytest

from a3_ultrasound.speed import SpeedOptions
from a3_ultrasound.v15_protocol import HEADS, copy_runtime_artifacts, validate_config
from a3_ultrasound.v153_protocol import DECISION_SHA256, verify_speed_decision


ROOT = Path(__file__).parents[1]


def candidate():
    cfg = json.loads((ROOT / 'configs/experiment/v1_5_3_freeze.json').read_text(encoding='utf-8'))
    for name in ('physical_statistics', 'm6_decision', 'timebase_calibration', 'speed_decision'):
        cfg[name + '_path'] = str(ROOT / cfg[name + '_path'])
    return cfg


def validate(cfg, **kwargs):
    validate_config(cfg, fold_name='F1', seed=23, head='proposed_v15', **kwargs)


def test_frozen_defaults_and_stopped_permission():
    cfg = candidate()
    for head in HEADS:
        validate_config(cfg, fold_name='F1', seed=23, head=head, require_permission=False)
    assert SpeedOptions(**cfg['speed']) == SpeedOptions.for_smoke('final_base')
    with pytest.raises(ValueError, match='prerequisites'):
        validate(cfg)
    assert cfg['outer_evaluation_permitted'] is False
    assert cfg['formal_training_permitted'] is False
    assert cfg['speed_backend_policy']['deterministic_algorithms'] is True
    decision = verify_speed_decision(cfg)
    assert decision['selected_speed'] == cfg['speed']


@pytest.mark.parametrize('key', list(SpeedOptions().to_dict()))
def test_reject_each_speed_switch_drift(key):
    cfg = candidate()
    value = cfg['speed'][key]
    cfg['speed'][key] = not value if type(value) is bool else value + 1
    with pytest.raises(ValueError, match='locked'):
        validate(cfg, require_permission=False)


@pytest.mark.parametrize('section,key,value', [
    ('speed_backend_policy', 'dtype', 'bfloat16'),
    ('speed_backend_policy', 'tf32', False),
    ('speed_backend_policy', 'cudnn_benchmark', True),
    ('speed_backend_policy', 'cudnn_deterministic', False),
    ('speed_backend_policy', 'deterministic_algorithms', False),
    ('speed_batching_policy', 'name', 'length_buckets'),
    ('speed_batching_policy', 'seed', 24),
    ('speed_batching_policy', 'drop_last', True),
    ('speed_batching_policy', 'head_independent', False),
    ('training', 'batch_size', 64),
])
def test_reject_backend_and_batching_drift(section, key, value):
    cfg = candidate()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate(cfg, require_permission=False)


def test_decision_artifact_binding_and_runtime_copy(tmp_path):
    cfg = candidate()
    raw = Path(cfg['speed_decision_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DECISION_SHA256 == cfg['speed_decision_sha256']
    copy_runtime_artifacts(cfg, tmp_path / 'run')
    assert (tmp_path / 'run/speed_decision.json').read_bytes() == raw
    changed = tmp_path / 'speed_decision.json'
    changed.write_bytes(raw + b' ')
    cfg['speed_decision_path'] = str(changed)
    with pytest.raises(ValueError, match='changed'):
        verify_speed_decision(cfg)
    cfg['speed_decision_sha256'] = hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='locked'):
        validate(cfg, require_permission=False)


def test_v152_template_retains_legacy_validator():
    cfg = json.loads((ROOT / 'configs/experiment/v1_5_2_freeze.json').read_text(encoding='utf-8'))
    validate(cfg, require_permission=False)
    assert 'speed' not in cfg
    assert SpeedOptions() != SpeedOptions.for_smoke('final_base')
