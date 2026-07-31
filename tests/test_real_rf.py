from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from a3_ultrasound.data.real_rf import (
    FIVE_POINT_SPATIAL_STENCIL,
    RealRFLoadError,
    build_label_free_real_rf_batch,
    select_five_point_spatial_stencil,
)


def _manifest() -> dict:
    return {
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "window_profile": {
                "window_left_samples": 1.0,
                "window_left_s": None,
                "window_right_samples": 1.0,
                "window_right_s": None,
            },
        },
        "raw_layout": {
            "raw_dtype": "signed_integer_text",
            "raw_unit": "level",
            "trace_length_mode": "fixed_within_batch",
            "trace_samples": 16,
            "file_pattern": "x{ix}y{iy}.txt",
        },
        "scan_geometry": {
            "layout": "regular_grid",
            "grid_shape": [3, 3],
        },
        "time_reference": {
            "batch_reference_samples": 0.0,
            "batch_reference_s": None,
        },
        "structure": {
            "K": 2,
            "total_thickness_mm": 9.0,
            "sound_speed_mps": 300_000.0,
            "interfaces": [
                {"interface_order": 1, "nominal_depth_mm": 3.0},
                {"interface_order": 2, "nominal_depth_mm": 6.0},
            ],
        },
        "preprocessing": {
            "amplitude_normalization": {
                "method": "fixed_batch_reference_division",
                "divisor_level": 2.0,
                "raw_unit": "level",
            },
        },
    }


def _write_trace(root: Path, x: int, y: int, values: list[int]) -> None:
    (root / f"x{x}y{y}.txt").write_text(
        " ".join(str(value) for value in values),
        encoding="utf-8",
    )


def test_five_point_stencil_is_spatially_distributed_and_deterministic() -> None:
    selected = select_five_point_spatial_stencil(_manifest())
    assert selected == ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    assert len(set(selected)) == 5


def test_real_rf_batch_uses_one_fixed_divisor_and_has_no_labels(tmp_path: Path) -> None:
    manifest = _manifest()
    selected = select_five_point_spatial_stencil(manifest)
    for index, (x, y) in enumerate(selected):
        _write_trace(tmp_path, x, y, list(range(-8 + index, 8 + index)))

    result = build_label_free_real_rf_batch(tmp_path, manifest)
    batch = result.batch

    assert result.selection_policy == FIVE_POINT_SPATIAL_STENCIL
    assert result.trace_count == 5
    assert result.trace_lengths == (16,) * 5
    assert result.normalization_divisor == 2.0
    assert result.raw_min == -8
    assert result.raw_max == 11
    assert batch.x_raw.shape == (5, 16)
    assert batch.x_raw.dtype == torch.float32
    torch.testing.assert_close(
        batch.x_raw[0],
        torch.arange(-8, 8, dtype=torch.float32) / 2.0,
    )
    assert batch.length_mask.all()
    assert batch.interface_mask.tolist() == [[True, True, False]] * 5
    assert batch.y is None
    assert batch.label_mask is None


def test_real_rf_loader_rejects_noninteger_and_fixed_length_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_trace(tmp_path, 0, 0, list(range(16)))
    (tmp_path / "x0y0.txt").write_text(
        " ".join(["0"] * 15 + ["1.5"]),
        encoding="utf-8",
    )
    with pytest.raises(RealRFLoadError, match="non-integer"):
        build_label_free_real_rf_batch(
            tmp_path,
            manifest,
            coordinates=[(0, 0)],
        )

    _write_trace(tmp_path, 0, 0, list(range(15)))
    with pytest.raises(ValueError, match="trace_samples=16"):
        build_label_free_real_rf_batch(
            tmp_path,
            manifest,
            coordinates=[(0, 0)],
        )


def test_real_rf_loader_fails_on_per_trace_or_invalid_normalization(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_trace(tmp_path, 0, 0, list(range(16)))

    per_trace = deepcopy(manifest)
    per_trace["preprocessing"]["amplitude_normalization"]["method"] = "per_trace_peak"
    with pytest.raises(RealRFLoadError, match="fixed_batch_reference_division"):
        build_label_free_real_rf_batch(
            tmp_path,
            per_trace,
            coordinates=[(0, 0)],
        )

    invalid_divisor = deepcopy(manifest)
    invalid_divisor["preprocessing"]["amplitude_normalization"]["divisor_level"] = 0
    with pytest.raises(RealRFLoadError, match="finite and positive"):
        build_label_free_real_rf_batch(
            tmp_path,
            invalid_divisor,
            coordinates=[(0, 0)],
        )


def test_real_rf_loader_rejects_path_escape_and_duplicate_coordinates(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_trace(tmp_path, 0, 0, list(range(16)))

    escaped = deepcopy(manifest)
    escaped["raw_layout"]["file_pattern"] = "../x{ix}y{iy}.txt"
    with pytest.raises(RealRFLoadError, match="escapes raw_root"):
        build_label_free_real_rf_batch(
            tmp_path,
            escaped,
            coordinates=[(0, 0)],
        )

    with pytest.raises(RealRFLoadError, match="unique"):
        build_label_free_real_rf_batch(
            tmp_path,
            manifest,
            coordinates=[(0, 0), (0, 0)],
        )
