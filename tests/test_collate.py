from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from a3_ultrasound.data.collate import (
    InterfaceSample,
    NetworkApiDerivationError,
    collate_interface_samples,
    derive_interface_tensors,
)
from a3_ultrasound.models.query import InterfaceQueryEncoder


def _manifest(*, k: int = 2, trace_length_mode: str = "variable_within_batch") -> dict:
    interfaces = [
        {
            "interface_order": 1,
            "nominal_depth_mm": 3.0,
        },
        {
            "interface_order": 2,
            "nominal_depth_mm": 6.0,
        },
    ]
    total_thickness_mm = 9.0
    if k == 3:
        interfaces.append(
            {
                "interface_order": 3,
                "nominal_depth_mm": 9.0,
            }
        )
        total_thickness_mm = 12.0
    return {
        "schema_version": "a3-acquisition-metadata-v2",
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "acquisition_profile_id": "ap_test_v1",
            "window_profile": {
                "profile_id": "wp_test_v1",
                "window_left_samples": 1.0,
                "window_left_s": None,
                "window_right_samples": 1.0,
                "window_right_s": None,
            },
        },
        "raw_layout": {
            "trace_length_mode": trace_length_mode,
            "trace_samples": 16 if trace_length_mode == "fixed_within_batch" else None,
        },
        "time_reference": {
            "batch_reference_samples": 0.0,
            "batch_reference_s": None,
        },
        "structure": {
            "K": k,
            "total_thickness_mm": total_thickness_mm,
            "sound_speed_mps": 300_000.0,
            "interfaces": interfaces,
        },
    }


def test_collate_derives_v3_metadata_and_variable_length_masks() -> None:
    batch = collate_interface_samples(
        [
            InterfaceSample(torch.arange(12), y=[0, 1], label_mask=[True, False]),
            InterfaceSample(torch.arange(16), y=[1, 0]),
        ],
        _manifest(),
    )

    assert batch.x_raw.shape == (2, 16)
    assert batch.x_raw.dtype == torch.float32
    assert batch.length_mask.tolist() == [
        [True] * 12 + [False] * 4,
        [True] * 16,
    ]
    assert batch.interface_mask.tolist() == [
        [True, True, False],
        [True, True, False],
    ]
    torch.testing.assert_close(
        batch.interface_meta[0, 0],
        torch.tensor([0.5, 4 / 12, 3 / 9, 1 / 12, 1 / 12]),
    )
    torch.testing.assert_close(
        batch.interface_meta[1, 1],
        torch.tensor([1.0, 8 / 16, 6 / 9, 1 / 16, 1 / 16]),
    )
    assert torch.count_nonzero(batch.interface_meta[:, 2]) == 0
    assert batch.y is not None
    assert batch.label_mask is not None
    assert batch.y.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert batch.label_mask.tolist() == [
        [True, False, False],
        [True, True, False],
    ]


def test_zmax_is_each_batch_registered_total_thickness() -> None:
    thin_manifest = _manifest(k=2)
    thin_manifest["structure"]["total_thickness_mm"] = 6.0
    thin_manifest["structure"]["interfaces"][0]["nominal_depth_mm"] = 2.0
    thin_manifest["structure"]["interfaces"][1]["nominal_depth_mm"] = 4.0
    thin_manifest["structure"]["sound_speed_mps"] = 200_000.0
    thin_interface, _ = derive_interface_tensors(
        thin_manifest,
        trace_length=16,
        kmax=3,
    )
    two_interface, _ = derive_interface_tensors(_manifest(k=2), trace_length=16, kmax=3)
    three_interface, _ = derive_interface_tensors(_manifest(k=3), trace_length=16, kmax=3)

    torch.testing.assert_close(thin_interface[:2, 2], torch.tensor([2 / 6, 4 / 6]))
    torch.testing.assert_close(two_interface[:2, 2], torch.tensor([3 / 9, 6 / 9]))
    torch.testing.assert_close(
        three_interface[:3, 2],
        torch.tensor([3 / 12, 6 / 12, 9 / 12]),
    )


def test_delta_changes_only_prior_fields_and_not_query() -> None:
    manifest = _manifest()
    baseline, mask = derive_interface_tensors(manifest, trace_length=16)
    changed = deepcopy(manifest)
    changed["acquisition"]["window_profile"]["window_left_samples"] = 1.5
    changed["acquisition"]["window_profile"]["window_right_samples"] = 1.5
    changed_meta, _ = derive_interface_tensors(changed, trace_length=16)

    torch.testing.assert_close(changed_meta[..., :3], baseline[..., :3])
    assert not torch.equal(changed_meta[..., 3:], baseline[..., 3:])

    torch.manual_seed(9)
    query = InterfaceQueryEncoder(query_dim=4)
    torch.testing.assert_close(
        query(baseline.unsqueeze(0), mask.unsqueeze(0)),
        query(changed_meta.unsqueeze(0), mask.unsqueeze(0)),
    )


def test_valid_metadata_is_invariant_to_padding_kmax() -> None:
    compact, compact_mask = derive_interface_tensors(
        _manifest(),
        trace_length=16,
        kmax=2,
    )
    padded, padded_mask = derive_interface_tensors(
        _manifest(),
        trace_length=16,
        kmax=3,
    )
    torch.testing.assert_close(padded[:2], compact)
    assert compact_mask.tolist() == [True, True]
    assert padded_mask.tolist() == [True, True, False]
    assert torch.count_nonzero(padded[2]) == 0


def test_batch_reference_dual_representation_and_window_bounds_fail_fast() -> None:
    manifest = _manifest()
    manifest["time_reference"]["batch_reference_samples"] = None
    manifest["time_reference"]["batch_reference_s"] = 1 / 200_000_000
    metadata, _ = derive_interface_tensors(manifest, trace_length=16)
    torch.testing.assert_close(metadata[0, 1], torch.tensor(5 / 16))

    manifest["time_reference"]["batch_reference_samples"] = 2.0
    with pytest.raises(NetworkApiDerivationError, match="disagree"):
        derive_interface_tensors(manifest, trace_length=16)

    manifest["time_reference"]["batch_reference_samples"] = 1.0
    manifest["time_reference"]["batch_reference_s"] = None
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 5.0
    with pytest.raises(NetworkApiDerivationError, match="front-surface reference"):
        derive_interface_tensors(manifest, trace_length=16)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["structure"].__setitem__("total_thickness_mm", None),
            "total_thickness_mm",
        ),
        (
            lambda manifest: manifest["structure"].__setitem__("sound_speed_mps", None),
            "sound_speed_mps",
        ),
        (
            lambda manifest: manifest["acquisition"]["window_profile"].__setitem__(
                "window_left_samples", None
            ),
            "window_left_samples",
        ),
    ],
)
def test_missing_derivation_values_fail_fast(mutation, message: str) -> None:
    manifest = _manifest()
    mutation(manifest)
    with pytest.raises(NetworkApiDerivationError, match=message):
        derive_interface_tensors(manifest, trace_length=16)


def test_fixed_length_manifest_rejects_trace_length_mismatch() -> None:
    with pytest.raises(NetworkApiDerivationError, match="trace_samples=16"):
        collate_interface_samples(
            [InterfaceSample(torch.zeros(15))],
            _manifest(trace_length_mode="fixed_within_batch"),
        )


def test_collate_rejects_per_interface_time_override_and_float_labels() -> None:
    manifest = _manifest()
    manifest["structure"]["interfaces"][0]["nominal_tof_samples"] = 4.0
    with pytest.raises(NetworkApiDerivationError, match="forbidden per-interface"):
        derive_interface_tensors(manifest, trace_length=16)

    with pytest.raises(NetworkApiDerivationError, match="integer dtype"):
        collate_interface_samples(
            [InterfaceSample(torch.zeros(16), y=[0.0, 1.0])],
            _manifest(),
        )
