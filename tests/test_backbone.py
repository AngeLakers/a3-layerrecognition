from __future__ import annotations

import pytest
import torch

from a3_ultrasound.models.backbone import LengthGroupedTCNBackbone


def _backbone() -> LengthGroupedTCNBackbone:
    torch.manual_seed(31)
    return LengthGroupedTCNBackbone(
        channels=(4, 8),
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
    )


def test_backbone_preserves_time_axis_and_exact_centers() -> None:
    backbone = _backbone()
    x_raw = torch.randn(2, 32)
    length_mask = torch.ones(2, 32, dtype=torch.bool)

    output = backbone(x_raw, length_mask)

    assert output.features.shape == (2, 32, 8)
    assert output.feature_mask is length_mask
    torch.testing.assert_close(output.feature_centers, torch.arange(32.0))


def test_variable_length_grouping_blocks_padded_tail_leakage() -> None:
    backbone = _backbone().eval()
    short = torch.randn(19)
    long = torch.randn(32)
    padded = torch.zeros(2, 32)
    padded[0, :19] = short
    padded[1] = long
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[0, :19] = True
    mask[1] = True

    batched = backbone(padded, mask)
    isolated = backbone(
        short.unsqueeze(0),
        torch.ones(1, 19, dtype=torch.bool),
    )

    torch.testing.assert_close(
        batched.features[0, :19],
        isolated.features[0],
    )
    assert torch.count_nonzero(batched.features[0, 19:]) == 0


def test_invalid_padded_tail_has_zero_gradient() -> None:
    backbone = _backbone()
    x_raw = torch.randn(2, 32, requires_grad=True)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[0, 19:] = False

    output = backbone(x_raw, mask)
    output.features.square().mean().backward()

    assert x_raw.grad is not None
    assert torch.count_nonzero(x_raw.grad[0, 19:]) == 0
    assert torch.isfinite(x_raw.grad).all()


def test_non_prefix_or_empty_length_mask_fails_fast() -> None:
    backbone = _backbone()
    x_raw = torch.randn(2, 8)
    hole_mask = torch.ones(2, 8, dtype=torch.bool)
    hole_mask[0, 3] = False
    empty_mask = torch.ones(2, 8, dtype=torch.bool)
    empty_mask[0] = False

    with pytest.raises(ValueError, match="contiguous valid prefix"):
        backbone(x_raw, hole_mask)
    with pytest.raises(ValueError, match="at least one valid sample"):
        backbone(x_raw, empty_mask)


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_even_or_nonpositive_kernel_is_rejected(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        LengthGroupedTCNBackbone(kernel_size=kernel_size)
