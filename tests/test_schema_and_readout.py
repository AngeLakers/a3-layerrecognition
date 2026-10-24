from __future__ import annotations

import pytest
import torch

from a3_ultrasound.models.indexed_readout import (
    IndexedReadout,
    build_indexed_supports,
    conv1d_feature_centers,
)
from a3_ultrasound.models.query import InterfaceQueryEncoder
from a3_ultrasound.schema import (
    INTERFACE_META_DIM,
    INTERFACE_META_FIELDS,
    INTERFACE_META_SCHEMA_VERSION,
    PRIOR_META_FIELDS,
    PRIOR_META_INDICES,
    QUERY_META_DIM,
    QUERY_META_FIELDS,
    QUERY_META_INDICES,
    InterfaceBatch,
)


def test_interface_batch_accepts_only_frozen_network_schema() -> None:
    batch = InterfaceBatch(
        x_raw=torch.zeros((2, 12), dtype=torch.float32),
        length_mask=torch.ones((2, 12), dtype=torch.bool),
        interface_meta=torch.zeros((2, 3, 5), dtype=torch.float32),
        interface_mask=torch.tensor([[True, True, False], [True, False, False]]),
        label_mask=torch.tensor([[True, True, False], [True, False, False]]),
        y=torch.zeros((2, 3), dtype=torch.long),
    )
    assert INTERFACE_META_SCHEMA_VERSION == "a3-interface-meta-v3"
    assert INTERFACE_META_DIM == 5
    assert INTERFACE_META_FIELDS == (
        "relative_order",
        "normalized_tof",
        "normalized_depth",
        "normalized_left_tolerance",
        "normalized_right_tolerance",
    )
    assert QUERY_META_DIM == 2
    assert QUERY_META_FIELDS == (INTERFACE_META_FIELDS[0], INTERFACE_META_FIELDS[2])
    assert QUERY_META_INDICES == (0, 2)
    assert PRIOR_META_FIELDS == (
        "normalized_tof",
        "normalized_left_tolerance",
        "normalized_right_tolerance",
    )
    assert PRIOR_META_INDICES == (1, 3, 4)
    assert batch.valid_interface_counts.tolist() == [2, 1]


def test_interface_batch_rejects_non_float32_raw_rf() -> None:
    with pytest.raises(TypeError, match="float32"):
        InterfaceBatch(
            x_raw=torch.zeros((1, 4), dtype=torch.float64),
            length_mask=torch.ones((1, 4), dtype=torch.bool),
            interface_meta=torch.zeros((1, 1, 5), dtype=torch.float32),
            interface_mask=torch.ones((1, 1), dtype=torch.bool),
        )


def test_interface_batch_rejects_superseded_six_field_query_schema() -> None:
    with pytest.raises(ValueError, match="five-field"):
        InterfaceBatch(
            x_raw=torch.zeros((1, 4), dtype=torch.float32),
            length_mask=torch.ones((1, 4), dtype=torch.bool),
            interface_meta=torch.zeros((1, 1, 6), dtype=torch.float32),
            interface_mask=torch.ones((1, 1), dtype=torch.bool),
        )


def test_query_encoder_masks_invalid_slots_without_absolute_id() -> None:
    torch.manual_seed(1)
    encoder = InterfaceQueryEncoder(query_dim=5)
    metadata = torch.randn((2, 3, 5), dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    query = encoder(metadata, mask)
    assert query.shape == (2, 3, 5)
    assert torch.count_nonzero(query[~mask]) == 0
    assert all("embedding" not in name for name, _ in encoder.named_parameters())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in encoder.modules())


def test_query_ignores_prior_tolerances_but_uses_identity_position_fields() -> None:
    torch.manual_seed(7)
    encoder = InterfaceQueryEncoder(query_dim=4)
    metadata = torch.randn((1, 2, 5), dtype=torch.float32)
    mask = torch.tensor([[True, True]])
    baseline = encoder(metadata, mask)

    changed_tolerances = metadata.clone()
    changed_tolerances[..., 3:] += 100.0
    torch.testing.assert_close(encoder(changed_tolerances, mask), baseline)

    changed_query_field = metadata.clone()
    changed_query_field[..., 2] += 1.0
    assert not torch.allclose(encoder(changed_query_field, mask), baseline)


def test_query_encoder_is_invariant_to_padding_kmax() -> None:
    torch.manual_seed(3)
    encoder = InterfaceQueryEncoder(query_dim=4)
    valid_metadata = torch.randn((1, 2, 5), dtype=torch.float32)
    compact_mask = torch.tensor([[True, True]])
    padded_metadata = torch.cat(
        (valid_metadata, torch.randn((1, 2, 5), dtype=torch.float32)),
        dim=1,
    )
    padded_mask = torch.tensor([[True, True, False, False]])
    compact_query = encoder(valid_metadata, compact_mask)
    padded_query = encoder(padded_metadata, padded_mask)
    torch.testing.assert_close(padded_query[:, :2], compact_query)
    assert torch.count_nonzero(padded_query[:, 2:]) == 0


def test_feature_centers_use_exact_conv_geometry() -> None:
    centers = conv1d_feature_centers(10, kernel_size=3, stride=2, padding=1)
    torch.testing.assert_close(centers, torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]))


def test_feature_centers_cover_valid_and_asymmetric_padding_geometry() -> None:
    valid = conv1d_feature_centers(8, kernel_size=3, stride=2, padding=0)
    asymmetric = conv1d_feature_centers(
        8,
        kernel_size=3,
        stride=2,
        padding=(2, 0),
    )

    torch.testing.assert_close(valid, torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(asymmetric, torch.tensor([-1.0, 1.0, 3.0, 5.0]))
    with pytest.raises(ValueError, match="padding"):
        conv1d_feature_centers(8, kernel_size=3, stride=2, padding=-1)


def test_indexed_supports_respect_feature_padding_and_interface_mask() -> None:
    centers = torch.arange(10, dtype=torch.float32)
    starts = torch.tensor([[3.0, 7.0]])
    ends = torch.tensor([[6.0, 8.0]])
    feature_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, False]])
    interface_mask = torch.tensor([[True, False]])
    local, pre, prior = build_indexed_supports(centers, starts, ends, feature_mask, interface_mask)
    assert local[0, 0].nonzero().squeeze(1).tolist() == [3, 4, 5, 6]
    assert pre[0, 0].nonzero().squeeze(1).tolist() == [0, 1, 2]
    assert not local[0, 1].any()
    assert not pre[0, 1].any()
    torch.testing.assert_close(prior[0, 0].sum(), torch.tensor(1.0))


def test_asymmetric_hard_window_can_keep_soft_prior_centered_on_nominal_tof() -> None:
    centers = torch.arange(10, dtype=torch.float32)
    starts = torch.tensor([[1.0]])
    ends = torch.tensor([[7.0]])
    feature_mask = torch.ones((1, 10), dtype=torch.bool)
    interface_mask = torch.ones((1, 1), dtype=torch.bool)
    _, _, prior = build_indexed_supports(
        centers,
        starts,
        ends,
        feature_mask,
        interface_mask,
        prior_center=torch.tensor([[3.0]]),
    )
    assert prior[0, 0].argmax().item() == 3


def test_local_readout_zeroes_attention_outside_support_and_normalizes_inside() -> None:
    torch.manual_seed(2)
    branch = IndexedReadout(feature_dim=4, query_dim=3, attention_dim=5)
    features = torch.randn((1, 5, 4))
    query = torch.randn((1, 2, 3))
    support = torch.tensor([[[False, True, True, False, False], [True, False, False, False, True]]])
    mask = torch.tensor([[True, True]])
    prior = support.to(dtype=torch.float32) / 2.0
    output = branch(features, query, support, mask, local_prior=prior)
    assert torch.count_nonzero(output.attention[~support]) == 0
    torch.testing.assert_close(output.attention.sum(dim=-1), torch.ones((1, 2)))


def test_empty_local_support_fails_fast_but_empty_pre_returns_learned_context() -> None:
    local_branch = IndexedReadout(feature_dim=2, query_dim=2, attention_dim=3)
    pre_branch = IndexedReadout(feature_dim=2, query_dim=None, attention_dim=3)
    features = torch.zeros((1, 3, 2))
    query = torch.zeros((1, 1, 2))
    mask = torch.ones((1, 1), dtype=torch.bool)
    empty = torch.zeros((1, 1, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="Local support"):
        local_branch(features, query, empty, mask)
    with torch.no_grad():
        pre_branch.empty_context.copy_(torch.tensor([2.0, -1.0]))
    output = pre_branch(features, None, empty, mask, allow_empty_context=True)
    torch.testing.assert_close(output.token[0, 0], torch.tensor([2.0, -1.0]))
    assert torch.count_nonzero(output.attention) == 0


def test_pre_readout_is_query_free_while_local_readout_remains_query_conditioned() -> None:
    torch.manual_seed(11)
    features = torch.randn((1, 5, 3))
    support = torch.ones((1, 1, 5), dtype=torch.bool)
    mask = torch.ones((1, 1), dtype=torch.bool)

    pre_branch = IndexedReadout(feature_dim=3, query_dim=None, attention_dim=4)
    pre_output = pre_branch(features, None, support, mask, allow_empty_context=True)
    with pytest.raises(ValueError, match="query=None"):
        pre_branch(features, torch.randn((1, 1, 2)), support, mask, allow_empty_context=True)

    local_branch = IndexedReadout(feature_dim=3, query_dim=2, attention_dim=4)
    query = torch.randn((1, 1, 2), requires_grad=True)
    local_output = local_branch(features, query, support, mask)
    gradient = torch.autograd.grad(local_output.token.sum(), query)[0]
    assert torch.count_nonzero(gradient) > 0
    assert pre_output.token.shape == local_output.token.shape
