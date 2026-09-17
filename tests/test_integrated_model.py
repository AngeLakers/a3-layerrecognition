from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from a3_ultrasound.data import InterfaceSample, collate_interface_samples
from a3_ultrasound.models import A3PCCModel, LengthGroupedTCNBackbone
from a3_ultrasound.schema import InterfaceBatch


def _model() -> A3PCCModel:
    torch.manual_seed(23)
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


def _mixed_k_batch() -> tuple[InterfaceBatch, torch.Tensor]:
    x_raw = torch.randn((2, 24), dtype=torch.float32)
    length_mask = torch.tensor(
        [
            [True] * 20 + [False] * 4,
            [True] * 24,
        ]
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
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    prefix_states = torch.tensor(
        [
            [0, 1, -1],
            [1, 0, 1],
        ],
        dtype=torch.long,
    )
    return (
        InterfaceBatch(
            x_raw=x_raw,
            length_mask=length_mask,
            interface_meta=interface_meta,
            interface_mask=interface_mask,
        ),
        prefix_states,
    )


def test_integrated_model_mixed_k_forward_shapes_and_masks() -> None:
    model = _model()
    batch, prefix_states = _mixed_k_batch()
    output = model(batch, prefix_states)

    assert output.backbone.features.shape == (2, 24, 64)
    assert output.query.shape == (2, 3, 5)
    assert output.local_support.shape == output.pre_support.shape == (2, 3, 24)
    assert output.local_prior.shape == (2, 3, 24)
    assert output.local_token_diagnostics.counts.shape == (2, 3)
    assert output.local_token_diagnostics.minimum.item() == 3
    assert output.local_token_diagnostics.maximum.item() == 5
    assert output.local_token_diagnostics.fraction_le_2.item() == 0.0
    assert output.local_readout.token.shape == output.pre_readout.token.shape == (2, 3, 64)
    assert output.responsibility_tokens.r_loc.shape == (2, 3, 7)
    assert output.responsibility_tokens.r_ctx.shape == (2, 3, 7)
    assert output.responsibility_tokens.r_gate.shape == (2, 3, 7)
    assert output.conditional.logits.shape == (2, 3)

    invalid_slot = (0, 2)
    assert torch.count_nonzero(output.query[invalid_slot]) == 0
    assert torch.count_nonzero(output.local_support[invalid_slot]) == 0
    assert torch.count_nonzero(output.pre_support[invalid_slot]) == 0
    assert torch.count_nonzero(output.local_readout.attention[invalid_slot]) == 0
    assert torch.count_nonzero(output.pre_readout.attention[invalid_slot]) == 0
    assert torch.count_nonzero(output.responsibility_tokens.r_loc[invalid_slot]) == 0
    assert output.conditional.logits[invalid_slot].item() == 0.0

    valid = batch.interface_mask
    torch.testing.assert_close(
        output.local_readout.attention.sum(dim=-1)[valid],
        torch.ones_like(output.local_readout.attention.sum(dim=-1)[valid]),
    )
    torch.testing.assert_close(
        output.pre_readout.attention.sum(dim=-1)[valid],
        torch.ones_like(output.pre_readout.attention.sum(dim=-1)[valid]),
    )
    assert torch.isfinite(output.conditional.logits[valid]).all()


def test_integrated_model_reconstructs_each_trace_window_from_normalized_prior() -> None:
    model = _model()
    batch, prefix_states = _mixed_k_batch()
    output = model(batch, prefix_states)

    torch.testing.assert_close(
        output.window_start,
        torch.tensor([[4.0, 12.0, 0.0], [4.0, 10.0, 18.0]]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.window_end,
        torch.tensor([[8.0, 16.0, 0.0], [6.0, 14.0, 20.0]]),
        atol=1e-5,
        rtol=0.0,
    )
    assert output.local_support[0, 0].nonzero().squeeze(1).tolist() == [4, 5, 6, 7, 8]
    assert output.local_support[1, 2].nonzero().squeeze(1).tolist() == [18, 19, 20]


def test_integrated_model_rejects_nonprefix_interfaces_and_invalid_truth_prefix() -> None:
    model = _model()
    batch, prefix_states = _mixed_k_batch()
    hole_mask = batch.interface_mask.clone()
    hole_mask[1] = torch.tensor([True, False, True])
    hole_batch = InterfaceBatch(
        x_raw=batch.x_raw,
        length_mask=batch.length_mask,
        interface_meta=batch.interface_meta,
        interface_mask=hole_mask,
    )
    with pytest.raises(ValueError, match="contiguous ordered prefix"):
        model(hole_batch, prefix_states)

    invalid_states = prefix_states.clone()
    invalid_states[1, 1] = 2
    with pytest.raises(ValueError, match="binary"):
        model(batch, invalid_states)


def test_collate_to_model_exact_inference_pipeline_is_normalized() -> None:
    manifest = {
        "schema_version": "a3-acquisition-metadata-v2",
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "acquisition_profile_id": "ap_pipeline_test_v1",
            "window_profile": {
                "profile_id": "wp_pipeline_test_v1",
                "window_left_samples": 2.0,
                "window_left_s": None,
                "window_right_samples": 2.0,
                "window_right_s": None,
            },
        },
        "raw_layout": {
            "trace_length_mode": "variable_within_batch",
            "trace_samples": None,
        },
        "time_reference": {
            "batch_reference_samples": 0.0,
            "batch_reference_s": None,
        },
        "structure": {
            "K": 2,
            "total_thickness_mm": 9.0,
            "sound_speed_mps": 150_000.0,
            "interfaces": [
                {"interface_order": 1, "nominal_depth_mm": 3.0},
                {"interface_order": 2, "nominal_depth_mm": 6.0},
            ],
        },
    }
    batch = collate_interface_samples(
        [
            InterfaceSample(torch.randn(20)),
            InterfaceSample(torch.randn(24)),
        ],
        manifest,
    )
    output = _model().infer_exact(batch)

    assert output.path_mask.sum(dim=1).tolist() == [4, 4]
    for batch_index in range(2):
        valid = output.path_mask[batch_index]
        torch.testing.assert_close(
            output.probabilities[batch_index, valid].sum(),
            torch.tensor(1.0),
            atol=1e-6,
            rtol=0.0,
        )


def test_integrated_head_accepts_k6_without_k_dependent_weights() -> None:
    model = _model()
    x_raw = torch.randn((1, 48), dtype=torch.float32)
    length_mask = torch.ones((1, 48), dtype=torch.bool)
    interface_mask = torch.ones((1, 6), dtype=torch.bool)
    interface_meta = torch.zeros((1, 6, 5), dtype=torch.float32)
    for index in range(6):
        tau = 5 + index * 7
        interface_meta[0, index] = torch.tensor(
            [(index + 1) / 6, tau / 48, (index + 1) / 7, 1 / 48, 1 / 48]
        )
    batch = InterfaceBatch(
        x_raw=x_raw,
        length_mask=length_mask,
        interface_meta=interface_meta,
        interface_mask=interface_mask,
    )
    parameter_shapes = {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }
    exact = model.infer_exact(batch)

    assert exact.path_mask.sum().item() == 64
    assert parameter_shapes == {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }


def test_a3_backend_capacity_is_independent_of_backbone_output_width() -> None:
    narrow = A3PCCModel(
        backbone=LengthGroupedTCNBackbone(
            channels=(4,),
            kernel_size=3,
            dilations=(1,),
            use_norm=None,
        ),
        query_dim=5,
        attention_dim=6,
        token_dim=7,
        prefix_dim=4,
        pcc_hidden_dim=8,
    )
    wide = A3PCCModel(
        backbone=LengthGroupedTCNBackbone(
            channels=(9,),
            kernel_size=3,
            dilations=(1,),
            use_norm=None,
        ),
        query_dim=5,
        attention_dim=6,
        token_dim=7,
        prefix_dim=4,
        pcc_hidden_dim=8,
    )

    def _backend_shapes(model: A3PCCModel) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
            if not name.startswith(("backbone.", "backbone_adapter."))
        }

    assert _backend_shapes(narrow) == _backend_shapes(wide)


def test_truth_prefix_forward_matches_integrated_exact_enumeration() -> None:
    model = _model()
    final_correction = model.pcc.correction[-1]
    assert isinstance(final_correction, torch.nn.Linear)
    with torch.no_grad():
        final_correction.weight.normal_(mean=0.0, std=0.2)
        final_correction.bias.fill_(0.1)

    batch, _ = _mixed_k_batch()
    exact = model.infer_exact(batch)
    valid_paths = exact.path_states[0, exact.path_mask[0], :2]
    direct_log_scores = []
    for path in valid_paths:
        prefix_states = torch.zeros_like(batch.interface_mask, dtype=torch.long)
        prefix_states[0, :2] = path
        conditional = model(batch, prefix_states).conditional.logits[0, :2]
        path_values = path.to(dtype=conditional.dtype)
        direct_log_scores.append(
            (
                path_values * F.logsigmoid(conditional)
                + (1.0 - path_values) * F.logsigmoid(-conditional)
            ).sum()
        )
    direct_log_scores_tensor = torch.stack(direct_log_scores)
    direct_log_probabilities = (
        direct_log_scores_tensor
        - torch.logsumexp(direct_log_scores_tensor, dim=0)
    )
    torch.testing.assert_close(
        exact.log_probabilities[0, exact.path_mask[0]],
        direct_log_probabilities,
        atol=1e-6,
        rtol=1e-6,
    )
