from __future__ import annotations

from copy import deepcopy

import pytest

from a3_ultrasound.data.time_prior import (
    TimePriorDerivationError,
    derive_time_prior_samples,
)


def _current_batch_like_manifest() -> dict:
    return {
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "acquisition_profile_id": "ap_10mhz_200mhz_48db_v1",
            "window_profile": {
                "profile_id": "wp_shared_96_samples_v1",
                "window_left_samples": 96.0,
                "window_left_s": None,
                "window_right_samples": 96.0,
                "window_right_s": None,
            },
        },
        "time_reference": {
            "batch_reference_samples": 34.5,
            "batch_reference_s": None,
        },
        "structure": {
            "K": 2,
            "total_thickness_mm": 9.0,
            "sound_speed_mps": 5_700.0,
            "layers": [
                {"layer_order": 1, "thickness_mm": 3.0},
                {"layer_order": 2, "thickness_mm": 3.0},
                {"layer_order": 3, "thickness_mm": 3.0},
            ],
            "interfaces": [
                {"interface_order": 1, "nominal_depth_mm": 3.0},
                {"interface_order": 2, "nominal_depth_mm": 6.0},
            ],
        },
    }


def test_current_batch_nominal_tofs_and_shared_window_are_derived() -> None:
    prior = derive_time_prior_samples(
        _current_batch_like_manifest(),
        trace_length=2300,
    )

    assert prior.nominal_tof_samples == pytest.approx((245.0263158, 455.5526316))
    assert prior.window_start_samples == pytest.approx((149.0263158, 359.5526316))
    assert prior.window_end_samples == pytest.approx((341.0263158, 551.5526316))
    assert prior.backwall_tof_samples == pytest.approx(666.0789474)


def test_one_batch_reference_translates_all_interfaces_equally() -> None:
    manifest = _current_batch_like_manifest()
    baseline = derive_time_prior_samples(manifest, trace_length=2300)
    shifted = deepcopy(manifest)
    shifted["time_reference"]["batch_reference_samples"] = 41.75
    changed = derive_time_prior_samples(shifted, trace_length=2300)

    shifts = [
        changed_tau - baseline_tau
        for baseline_tau, changed_tau in zip(
            baseline.nominal_tof_samples,
            changed.nominal_tof_samples,
        )
    ]
    assert shifts == pytest.approx((7.25, 7.25))
    assert (
        changed.nominal_tof_samples[1] - changed.nominal_tof_samples[0]
    ) == pytest.approx(
        baseline.nominal_tof_samples[1] - baseline.nominal_tof_samples[0]
    )


def test_layered_sound_speed_uses_cumulative_path_time() -> None:
    manifest = _current_batch_like_manifest()
    manifest["structure"]["sound_speed_mps"] = None
    speeds = (5_000.0, 6_000.0, 7_000.0)
    for layer, speed in zip(manifest["structure"]["layers"], speeds):
        layer["sound_speed_mps"] = speed
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 20.0
    manifest["acquisition"]["window_profile"]["window_right_samples"] = 20.0

    prior = derive_time_prior_samples(manifest, trace_length=2300)
    expected_first = 34.5 + 2 * 200_000_000 * (0.003 / 5_000.0)
    expected_second = expected_first + 2 * 200_000_000 * (0.003 / 6_000.0)
    assert prior.nominal_tof_samples == pytest.approx((expected_first, expected_second))


def test_shared_window_overlap_and_dual_unit_conflict_fail_fast() -> None:
    manifest = _current_batch_like_manifest()
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 106.0
    manifest["acquisition"]["window_profile"]["window_right_samples"] = 106.0
    with pytest.raises(TimePriorDerivationError, match="overlap"):
        derive_time_prior_samples(manifest, trace_length=2300)

    manifest = _current_batch_like_manifest()
    manifest["time_reference"]["batch_reference_s"] = 35.0 / 200_000_000
    with pytest.raises(TimePriorDerivationError, match="disagree"):
        derive_time_prior_samples(manifest, trace_length=2300)
