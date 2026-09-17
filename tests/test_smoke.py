from __future__ import annotations

import copy
from pathlib import Path

import pytest

from a3_ultrasound.data.smoke import scan_label_free


FIXED_TIME = "2026-07-24T00:00:00+00:00"


def _manifest() -> dict:
    return {
        "schema_version": "a3-acquisition-metadata-v2",
        "identity": {
            "dataset_id": "ds_test",
            "specimen_id": "sp_test",
            "acquisition_batch_id": "batch_test",
            "scan_id": "scan_test",
        },
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "acquisition_averaging_enabled": False,
            "gate_threshold_level": 100,
            "values_above_gate_threshold_allowed": True,
        },
        "raw_layout": {
            "export_format": "one_whitespace_separated_signed_integer_trace_per_txt",
            "raw_dtype": "signed_integer_text",
            "raw_unit": "level",
            "trace_length_mode": "fixed_within_batch",
            "trace_samples": 32,
            "file_pattern": "x{ix}y{iy}.txt",
        },
        "scan_geometry": {
            "layout": "regular_grid",
            "grid_shape": [2, 2],
        },
        "time_reference": {
            "reference_event": "water_specimen_front_surface_echo_packet",
            "front_surface_material": "aluminum_alloy",
        },
        "structure": {
            "K": 2,
            "interfaces": [{}, {}],
        },
        "provenance": {
            "source_ref": "opaque/raw",
            "source_checksum": None,
        },
        "labels": {
            "schema": "good_abnormal_v1",
            "embedded_defect_manifest_ref": "private-a",
        },
    }


def _schema() -> dict:
    return {
        "schema_version": "a3-smoke-summary-v1",
        "input_contract": {
            "metadata_schema": "a3-acquisition-metadata-v2",
        },
    }


def _config() -> dict:
    return {
        "scanner_version": "test-smoke-v1",
        "trace_name_regex": r"^x(?P<x>\d+)y(?P<y>\d+)\.txt$",
        "example_limit": 20,
        "progress_every": 0,
        "front_detector": {
            "name": "front_local_rms_snr_v1",
            "search_start_sample": 0,
            "search_stop_sample_exclusive": 16,
            "moving_rms_samples": 4,
            "tail_noise_samples": 8,
            "detection_snr_db": 6.0,
            "width_fraction_of_peak": 0.5,
        },
        "plateau_descriptor": {
            "minimum_absolute_level": 100,
            "minimum_run_samples": 4,
            "interpretation": "descriptive_not_clipping",
        },
        "thresholds": {
            "allowed_missing_trace_fraction": None,
            "allowed_invalid_length_fraction": None,
            "minimum_reference_detection_rate": None,
            "minimum_snr_proxy_db": None,
            "maximum_constant_trace_fraction": None,
            "maximum_plateau_fraction": None,
        },
    }


def _write_trace(path: Path, values: list[float]) -> None:
    path.write_text(" ".join(str(value) for value in values), encoding="utf-8")


def _valid_grid(root: Path) -> None:
    weak = [0] * 32
    strong = [0] * 32
    strong[4:8] = [20, -20, 20, -20]
    plateau = [0] * 32
    plateau[4:8] = [120, 120, 120, 120]
    _write_trace(root / "x0y0.txt", strong)
    _write_trace(root / "x0y1.txt", weak)
    _write_trace(root / "x1y0.txt", plateau)
    _write_trace(root / "x1y1.txt", strong)


def test_valid_scan_is_label_free_and_privacy_safe(tmp_path: Path) -> None:
    _valid_grid(tmp_path)
    manifest_a = _manifest()
    manifest_b = copy.deepcopy(manifest_a)
    manifest_b["labels"] = {
        "schema": "changed",
        "embedded_defect_manifest_ref": "different-private-value",
    }
    manifest_c = copy.deepcopy(manifest_a)
    manifest_c.pop("labels")

    summary_a = scan_label_free(
        tmp_path,
        manifest_a,
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )
    summary_b = scan_label_free(
        tmp_path,
        manifest_b,
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )
    summary_c = scan_label_free(
        tmp_path,
        manifest_c,
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )

    assert summary_a == summary_b == summary_c
    assert summary_a["integrity"]["status"] == "PASS"
    assert summary_a["numeric"]["status"] == "PASS"
    assert summary_a["contract"]["label_fields_were_not_read"] is True
    assert summary_a["contract"]["absolute_paths_absent_from_summary"] is True
    assert str(tmp_path) not in str(summary_a)
    assert summary_a["outcome"]["overall_status"] == "INCOMPLETE"


def test_metadata_schema_version_must_match_smoke_input_contract(
    tmp_path: Path,
) -> None:
    _valid_grid(tmp_path)
    manifest = _manifest()
    manifest["schema_version"] = "a3-acquisition-metadata-v1"

    with pytest.raises(ValueError, match="must match smoke input contract"):
        scan_label_free(
            tmp_path,
            manifest,
            _schema(),
            _config(),
            generated_at=FIXED_TIME,
        )


def test_regular_grid_stride_samples_content_but_inventories_full_grid(
    tmp_path: Path,
) -> None:
    _valid_grid(tmp_path)
    config = _config()
    config["sampling"] = {
        "mode": "regular_grid_stride",
        "start_x": 0,
        "start_y": 0,
        "stride_x": 2,
        "stride_y": 2,
        "selection_basis": "test",
    }

    summary = scan_label_free(
        tmp_path,
        _manifest(),
        _schema(),
        config,
        generated_at=FIXED_TIME,
    )

    assert summary["audit_scope"]["registered_grid_trace_count"] == 4
    assert summary["audit_scope"]["selected_coordinate_count"] == 1
    assert summary["audit_scope"]["selected_fraction"] == pytest.approx(0.25)
    assert summary["integrity"]["observed_trace_count"] == 4
    assert summary["integrity"]["readable_trace_count"] == 1
    assert summary["integrity"]["status"] == "WARN"
    scope_check = next(
        check
        for check in summary["outcome"]["checks"]
        if check["check_id"] == "integrity.content_audit_scope"
    )
    assert scope_check["status"] == "WARN"


def test_gate_exceedance_and_plateau_are_descriptive_not_clipping(
    tmp_path: Path,
) -> None:
    _valid_grid(tmp_path)
    summary = scan_label_free(
        tmp_path,
        _manifest(),
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )

    assert summary["amplitude"]["traces_exceeding_gate_threshold_count"] == 1
    assert summary["numeric"]["plateau_fraction"] == pytest.approx(0.25)
    assert summary["amplitude"]["clipping_code"] is None
    assert summary["amplitude"]["clipping_fraction"] is None
    clipping_check = next(
        check
        for check in summary["outcome"]["checks"]
        if check["check_id"] == "amplitude.clipping_code"
    )
    assert clipping_check["status"] == "NOT_EVALUATED"


def test_detector_failures_are_explicit_and_not_imputed(tmp_path: Path) -> None:
    _valid_grid(tmp_path)
    summary = scan_label_free(
        tmp_path,
        _manifest(),
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )

    assert summary["temporal"]["detected_trace_count"] == 3
    assert summary["temporal"]["detection_failure_count"] == 1
    assert summary["temporal"]["failure_reason_counts"] == {
        "front_snr_below_6_db": 1
    }
    assert len(summary["temporal"]["failure_examples"]) == 1


def test_missing_invalid_length_and_nonfinite_are_locatable(
    tmp_path: Path,
) -> None:
    strong = [0] * 32
    strong[4:8] = [20, -20, 20, -20]
    _write_trace(tmp_path / "x0y0.txt", strong)
    _write_trace(tmp_path / "x0y1.txt", [1] * 7)
    nonfinite = strong.copy()
    nonfinite[10] = float("nan")
    _write_trace(tmp_path / "x1y0.txt", nonfinite)

    summary = scan_label_free(
        tmp_path,
        _manifest(),
        _schema(),
        _config(),
        generated_at=FIXED_TIME,
    )

    assert summary["integrity"]["missing_trace_count"] == 1
    assert summary["integrity"]["missing_examples"] == ["x1y1.txt"]
    assert summary["integrity"]["invalid_length_count"] == 1
    assert summary["numeric"]["nonfinite_count"] == 1
    assert summary["temporal"]["failure_reason_counts"] == {
        "invalid_trace_length": 1,
        "nonfinite_trace": 1,
    }
    assert summary["outcome"]["overall_status"] == "FAIL"
