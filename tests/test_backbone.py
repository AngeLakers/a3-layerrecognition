from __future__ import annotations

import pytest
import torch

from a3_ultrasound.models.backbone import (
    CommonTemporalBackboneAdapter,
    LengthGroupedTCNBackbone,
    TsaiTimePreservingBackbone,
)


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
    torch.testing.assert_close(output.feature_mask, length_mask)
    torch.testing.assert_close(
        output.feature_centers,
        torch.arange(32.0).expand(2, -1),
    )
    assert backbone.training_role == "smoke_only"


def test_common_adapter_changes_only_channels_and_preserves_geometry() -> None:
    output = _backbone()(torch.randn(2, 32), torch.ones(2, 32, dtype=torch.bool))
    adapted = CommonTemporalBackboneAdapter(8, 64)(output)

    assert adapted.features.shape == (2, 32, 64)
    assert adapted.feature_centers is output.feature_centers
    assert adapted.feature_mask is output.feature_mask
    assert adapted.effective_stride == output.effective_stride == 1.0


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


@pytest.mark.parametrize("name", ["resnet1d", "inceptiontime", "fcn1d"])
def test_tsai_formal_backbones_expose_prepool_time_preserving_features(name: str) -> None:
    torch.manual_seed(41)
    backbone = TsaiTimePreservingBackbone(name).eval()
    x_raw = torch.randn(2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)

    with torch.no_grad():
        output = backbone(x_raw, mask)

    assert output.features.shape == (2, 64, backbone.output_dim)
    torch.testing.assert_close(output.feature_centers, torch.arange(64.0).expand(2, -1))
    assert torch.equal(output.feature_mask, mask)
    assert backbone.training_role == "formal_candidate"
    forbidden = {"GAP1d", "AdaptiveAvgPool1d"}
    assert forbidden.isdisjoint({type(module).__name__ for module in backbone.trunk.modules()})


@pytest.mark.parametrize(
    ("name", "feature_stage", "expected_dim"),
    [
        ("resnet1d", "stage1", 64),
        ("resnet1d", "stage2", 128),
        ("fcn1d", "stage1", 128),
        ("fcn1d", "stage2", 256),
    ],
)
def test_tsai_stage_selection_preserves_time_and_reports_selected_dim(
    name: str,
    feature_stage: str,
    expected_dim: int,
) -> None:
    backbone = TsaiTimePreservingBackbone(name, feature_stage=feature_stage).eval()
    x_raw = torch.randn((2, 31), dtype=torch.float32)
    length_mask = torch.ones_like(x_raw, dtype=torch.bool)

    with torch.no_grad():
        output = backbone(x_raw, length_mask)

    assert backbone.feature_stage == feature_stage
    assert backbone.output_dim == expected_dim
    assert output.features.shape == (2, 31, expected_dim)
    torch.testing.assert_close(output.feature_centers[0], torch.arange(31.0))


def test_inceptiontime_rejects_unavailable_intermediate_stage() -> None:
    with pytest.raises(ValueError, match="only final_pre_pool"):
        TsaiTimePreservingBackbone("inceptiontime", feature_stage="stage1")


def test_tsai_backbone_blocks_padding_leakage_and_invalid_tail_gradient() -> None:
    torch.manual_seed(43)
    backbone = TsaiTimePreservingBackbone("resnet1d").eval()
    short = torch.randn(48)
    long = torch.randn(64)
    padded = torch.zeros(2, 64, requires_grad=True)
    with torch.no_grad():
        padded[0, :48] = short
        padded[1] = long
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :48] = True
    mask[1] = True

    batched = backbone(padded, mask)
    isolated = backbone(short.unsqueeze(0), torch.ones(1, 48, dtype=torch.bool))
    torch.testing.assert_close(batched.features[0, :48], isolated.features[0])
    assert torch.count_nonzero(batched.features[0, 48:]) == 0

    batched.features.square().mean().backward()
    assert padded.grad is not None
    assert torch.count_nonzero(padded.grad[0, 48:]) == 0
