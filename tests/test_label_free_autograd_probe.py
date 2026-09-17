from __future__ import annotations

import pytest
import torch

from a3_ultrasound.diagnostics import (
    build_label_free_autograd_probe,
    summarize_parameter_gradients,
)
from a3_ultrasound.models import A3PCCModel, LengthGroupedTCNBackbone
from a3_ultrasound.schema import InterfaceBatch


def _model() -> A3PCCModel:
    torch.manual_seed(31)
    return A3PCCModel(
        backbone=LengthGroupedTCNBackbone(
            channels=(4, 4),
            kernel_size=3,
            dilations=(1, 2),
            use_norm=None,
        ),
        query_dim=5,
        attention_dim=6,
        token_dim=7,
        prefix_dim=4,
        pcc_hidden_dim=8,
    )


def _batch() -> InterfaceBatch:
    x_raw = torch.randn((2, 24), dtype=torch.float32, requires_grad=True)
    length_mask = torch.tensor(
        [[True] * 20 + [False] * 4, [True] * 24],
        dtype=torch.bool,
    )
    interface_meta = torch.tensor(
        [
            [
                [0.5, 6 / 20, 3 / 9, 2 / 20, 2 / 20],
                [1.0, 14 / 20, 6 / 9, 2 / 20, 2 / 20],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1 / 3, 5 / 24, 0.25, 1 / 24, 1 / 24],
                [2 / 3, 12 / 24, 0.50, 2 / 24, 2 / 24],
                [1.0, 19 / 24, 0.75, 1 / 24, 1 / 24],
            ],
        ],
        dtype=torch.float32,
    )
    interface_mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )
    return InterfaceBatch(
        x_raw=x_raw,
        length_mask=length_mask,
        interface_meta=interface_meta,
        interface_mask=interface_mask,
    )


def test_label_free_probe_backpropagates_without_updating_model() -> None:
    model = _model().eval()
    batch = _batch()
    prefix_states = torch.zeros_like(batch.interface_mask, dtype=torch.long)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    output = model(batch, prefix_states)
    probe = build_label_free_autograd_probe(output, batch.interface_mask)
    probe.scalar.backward()

    assert torch.isfinite(probe.scalar)
    assert batch.x_raw.grad is not None
    assert torch.isfinite(batch.x_raw.grad).all()
    assert torch.count_nonzero(batch.x_raw.grad[batch.length_mask]) > 0
    assert torch.count_nonzero(batch.x_raw.grad[~batch.length_mask]) == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0.0, rtol=0.0)


def test_label_free_probe_touches_every_registered_branch() -> None:
    model = _model().eval()
    batch = _batch()
    output = model(
        batch,
        torch.zeros_like(batch.interface_mask, dtype=torch.long),
    )
    build_label_free_autograd_probe(output, batch.interface_mask).scalar.backward()
    summary = summarize_parameter_gradients(
        model,
        {
            "backbone": ("backbone.",),
            "adapter": ("backbone_adapter.",),
            "query": ("query_encoder.",),
            "local": ("local_readout.",),
            "pre": ("pre_readout.",),
            "tokens": ("responsibility_projector.",),
            "base": ("pcc.base.",),
            "prefix": ("pcc.prefix_encoder.",),
            "correction": ("pcc.correction.",),
            "gate": ("pcc.gate.",),
        },
    )

    assert all(group["parameter_tensors"] > 0 for group in summary.values())
    assert all(group["all_gradients_finite"] for group in summary.values())
    assert all(group["nonzero_gradient_present"] for group in summary.values())


def test_label_free_probe_rejects_empty_or_invalid_interface_mask() -> None:
    model = _model().eval()
    batch = _batch()
    output = model(
        batch,
        torch.zeros_like(batch.interface_mask, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="at least one valid interface"):
        build_label_free_autograd_probe(
            output,
            torch.zeros_like(batch.interface_mask),
        )
    with pytest.raises(ValueError, match="must be bool"):
        build_label_free_autograd_probe(
            output,
            batch.interface_mask.to(dtype=torch.int64),
        )
