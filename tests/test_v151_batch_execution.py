import copy
import json
from pathlib import Path

import pytest
import torch

from a3_ultrasound.models.backbone import TsaiTimePreservingBackbone
from a3_ultrasound.training import ModelConfig, build_model
from a3_ultrasound.v15_protocol import validate_config
from a3_ultrasound.freeze_protocol import validate_freeze_config


def test_length_grouped_gn_preserves_values_gradients_and_padding():
    torch.manual_seed(23)
    canonical = TsaiTimePreservingBackbone('resnet1d', backbone_norm='group8')
    batched = copy.deepcopy(canonical)
    batched.canonical_per_trace = False
    lengths = torch.tensor([48, 64, 48, 64])
    mask = torch.arange(64)[None, :] < lengths[:, None]
    x = torch.randn(4, 64)
    x[~mask] = 10000.
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    calls = []
    hook = batched.trunk.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    left, right = canonical(a, mask), batched(b, mask)
    hook.remove()
    assert calls == [2, 2]
    torch.testing.assert_close(left.features, right.features, atol=1e-4, rtol=0.)
    assert torch.equal(left.feature_mask, right.feature_mask)
    assert torch.equal(left.feature_centers, right.feature_centers)
    assert not right.features[~mask].count_nonzero()
    left.features.square().mean().backward()
    right.features.square().mean().backward()
    assert not b.grad[~mask].count_nonzero()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=0.)
    for p, q in zip(canonical.parameters(), batched.parameters()):
        assert torch.isfinite(q.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-4, rtol=0.)
    # A real record keeps its result when its length peer or padded tail changes.
    single = batched(x[:1, :48], torch.ones(1, 48, dtype=torch.bool))
    torch.testing.assert_close(single.features[0], right.features[0, :48], atol=1e-4, rtol=0.)


def test_default_and_explicit_canonical_keep_legacy_call_shape():
    torch.manual_seed(23)
    model = build_model(ModelConfig(backbone_norm='group8'))
    assert model.backbone.canonical_per_trace is True
    batched = build_model(ModelConfig(backbone_norm='group8', canonical_per_trace=False))
    assert batched.backbone.canonical_per_trace is False
    with pytest.raises(ValueError, match='bool'):
        TsaiTimePreservingBackbone('resnet1d', canonical_per_trace='false')


def test_v151_execution_cannot_silently_change_old_frozen_protocols():
    root = Path(__file__).parents[1] / 'configs/experiment'
    old = json.loads((root/'v1_4_freeze.json').read_text(encoding='utf-8'))
    old['model']['canonical_per_trace'] = False
    with pytest.raises(ValueError):
        validate_freeze_config(old, fold_name='F1', seed=23, head='proposed', require_permission=False)
    v15 = json.loads((root/'v1_5_freeze.json').read_text(encoding='utf-8'))
    v15['model']['canonical_per_trace'] = False
    with pytest.raises(ValueError, match='execution'):
        validate_config(v15, fold_name='F1', seed=23, head='proposed_v15', require_permission=False)
    v15['execution_version'] = 'v1.5.1'
    validate_config(v15, fold_name='F1', seed=23, head='proposed_v15', require_permission=False)
    v15['model']['canonical_per_trace'] = True
    with pytest.raises(ValueError, match='execution'):
        validate_config(v15, fold_name='F1', seed=23, head='proposed_v15', require_permission=False)
