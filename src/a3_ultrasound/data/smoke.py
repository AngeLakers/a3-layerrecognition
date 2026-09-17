"""Deterministic label-free quality checks for one registered A3 RF scan.

The scanner intentionally operates before denoising and never consumes label
fields.  It reports acquisition integrity, raw numeric properties, and the
detectability of the strong water-specimen front-surface echo packet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import load_yaml_mapping, validate_manifest


ProgressCallback = Callable[[int, int], None]
CHECK_STATUSES = ("PASS", "WARN", "FAIL", "NOT_EVALUATED")

_LABEL_FREE_PLACEHOLDER = {
    "schema": "good_abnormal_v1",
    "source_type": "embedded_defect_design",
    "embedded_defect_manifest_ref": None,
    "good_region_ref": None,
    "uncertain_policy": "mask",
    "uncertain_boundary_ref": None,
}
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def load_label_free_manifest(
    manifest_path: str | Path,
    metadata_schema_path: str | Path,
) -> dict[str, Any]:
    """Load and validate a manifest without consuming its label fields.

    The metadata validator expects the ``labels`` group to exist.  For this
    label-free stage the original group is removed before validation and a
    fixed schema-compatible placeholder is inserted.  The returned mapping
    does not contain a ``labels`` key.
    """

    manifest = load_yaml_mapping(manifest_path)
    metadata_schema = load_yaml_mapping(metadata_schema_path)
    label_free_view = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "labels"
    }
    label_free_view["labels"] = copy.deepcopy(_LABEL_FREE_PLACEHOLDER)
    validate_manifest(
        label_free_view,
        metadata_schema,
        stage="before_label_free_smoke",
    )
    label_free_view.pop("labels", None)
    return label_free_view


def scan_label_free(
    raw_root: str | Path,
    manifest: Mapping[str, Any],
    smoke_schema: Mapping[str, Any],
    scanner_config: Mapping[str, Any],
    *,
    generated_at: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Scan a raw TXT grid and return a JSON-serializable quality summary."""

    input_contract = _require_mapping(smoke_schema, "input_contract")
    expected_metadata_schema = input_contract.get("metadata_schema")
    if not isinstance(expected_metadata_schema, str) or not expected_metadata_schema:
        raise ValueError("input_contract.metadata_schema must be a non-empty string")
    actual_metadata_schema = manifest.get("schema_version")
    if actual_metadata_schema != expected_metadata_schema:
        raise ValueError(
            "manifest schema_version must match smoke input contract "
            f"({expected_metadata_schema!r}, got {actual_metadata_schema!r})"
        )

    root = Path(raw_root)
    if not root.is_dir():
        raise ValueError("raw_root must be an existing directory")

    identity = _require_mapping(manifest, "identity")
    acquisition = _require_mapping(manifest, "acquisition")
    raw_layout = _require_mapping(manifest, "raw_layout")
    geometry = _require_mapping(manifest, "scan_geometry")
    time_reference = _require_mapping(manifest, "time_reference")
    structure = _require_mapping(manifest, "structure")
    provenance = _require_mapping(manifest, "provenance")
    front_config = _require_mapping(scanner_config, "front_detector")
    plateau_config = _require_mapping(scanner_config, "plateau_descriptor")
    thresholds = _require_mapping(scanner_config, "thresholds")
    sampling_config = scanner_config.get("sampling", {"mode": "full"})
    if not isinstance(sampling_config, Mapping):
        raise ValueError("sampling must be a mapping")

    sampling_rate_hz = int(acquisition["sampling_rate_hz"])
    trace_length_mode = str(raw_layout["trace_length_mode"])
    expected_trace_samples = raw_layout.get("trace_samples")
    gate_threshold = float(acquisition["gate_threshold_level"])
    grid_shape = geometry.get("grid_shape")
    if (
        not isinstance(grid_shape, Sequence)
        or isinstance(grid_shape, (str, bytes))
        or len(grid_shape) != 2
    ):
        raise ValueError("scan_geometry.grid_shape must contain two dimensions")
    grid_x, grid_y = (int(grid_shape[0]), int(grid_shape[1]))
    expected_trace_count = grid_x * grid_y

    trace_regex = re.compile(str(scanner_config["trace_name_regex"]))
    example_limit = int(scanner_config.get("example_limit", 20))
    progress_every = int(scanner_config.get("progress_every", 0))

    all_txt_paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    coordinate_paths: dict[tuple[int, int], Path] = {}
    duplicate_trace_count = 0
    bad_filename_count = 0
    for path in all_txt_paths:
        match = trace_regex.fullmatch(path.name)
        if match is None:
            bad_filename_count += 1
            continue
        try:
            coordinate = (int(match.group("x")), int(match.group("y")))
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "trace_name_regex must define integer x and y groups"
            ) from exc
        if not (0 <= coordinate[0] < grid_x and 0 <= coordinate[1] < grid_y):
            bad_filename_count += 1
            continue
        if coordinate in coordinate_paths:
            duplicate_trace_count += 1
            if path.name < coordinate_paths[coordinate].name:
                coordinate_paths[coordinate] = path
            continue
        coordinate_paths[coordinate] = path

    missing_trace_count = expected_trace_count - len(coordinate_paths)
    missing_examples = _missing_examples(
        coordinate_paths,
        grid_x,
        grid_y,
        str(raw_layout["file_pattern"]),
        example_limit,
    )

    readable_trace_count = 0
    unreadable_trace_count = 0
    unreadable_examples: list[str] = []
    trace_lengths: list[int] = []
    invalid_length_count = 0
    nonfinite_count = 0
    empty_trace_count = 0
    constant_trace_count = 0
    noninteger_value_count = 0
    zero_count = 0
    parsed_sample_count = 0
    finite_min: float | None = None
    finite_max: float | None = None
    peak_absolute_level: float | None = None
    plateau_trace_count = 0
    threshold_exceeding_trace_count = 0
    raw_rms_values: list[float] = []
    noise_rms_values: list[float] = []
    snr_values: list[float] = []
    detected_positions: list[float] = []
    detected_widths: list[float] = []
    detected_confidences: list[float] = []
    detected_front_peaks: list[float] = []
    detected_front_energies: list[float] = []
    failure_reason_counts: Counter[str] = Counter()
    failure_examples: list[str] = []

    selected_coordinates, sampling_details = _select_coordinates(
        grid_x,
        grid_y,
        sampling_config,
    )
    selected_missing_count = sum(
        coordinate not in coordinate_paths for coordinate in selected_coordinates
    )
    ordered_paths = [
        coordinate_paths[coordinate]
        for coordinate in selected_coordinates
        if coordinate in coordinate_paths
    ]
    total_to_scan = len(ordered_paths)
    for index, path in enumerate(ordered_paths, start=1):
        relative_identifier = path.name
        try:
            text = path.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                values = np.fromstring(text, dtype=np.float64, sep=" ")
        except (OSError, UnicodeError, ValueError, DeprecationWarning):
            unreadable_trace_count += 1
            _append_example(
                unreadable_examples,
                relative_identifier,
                example_limit,
            )
            _record_detection_failure(
                "unreadable_trace",
                relative_identifier,
                failure_reason_counts,
                failure_examples,
                example_limit,
            )
            if progress and progress_every > 0 and index % progress_every == 0:
                progress(index, total_to_scan)
            continue

        readable_trace_count += 1
        length = int(values.size)
        trace_lengths.append(length)
        parsed_sample_count += length
        if length == 0:
            empty_trace_count += 1
            invalid_length_count += 1
            _record_detection_failure(
                "empty_trace",
                relative_identifier,
                failure_reason_counts,
                failure_examples,
                example_limit,
            )
            if progress and progress_every > 0 and index % progress_every == 0:
                progress(index, total_to_scan)
            continue

        if (
            trace_length_mode == "fixed_within_batch"
            and expected_trace_samples is not None
            and length != int(expected_trace_samples)
        ):
            invalid_length_count += 1
            _record_detection_failure(
                "invalid_trace_length",
                relative_identifier,
                failure_reason_counts,
                failure_examples,
                example_limit,
            )
            if progress and progress_every > 0 and index % progress_every == 0:
                progress(index, total_to_scan)
            continue

        finite_mask = np.isfinite(values)
        trace_nonfinite_count = int(length - np.count_nonzero(finite_mask))
        nonfinite_count += trace_nonfinite_count
        zero_count += int(np.count_nonzero(values == 0))
        if trace_nonfinite_count:
            finite_values = values[finite_mask]
            if finite_values.size:
                current_min = float(np.min(finite_values))
                current_max = float(np.max(finite_values))
                current_peak = float(np.max(np.abs(finite_values)))
                finite_min = (
                    current_min if finite_min is None else min(finite_min, current_min)
                )
                finite_max = (
                    current_max if finite_max is None else max(finite_max, current_max)
                )
                peak_absolute_level = (
                    current_peak
                    if peak_absolute_level is None
                    else max(peak_absolute_level, current_peak)
                )
            _record_detection_failure(
                "nonfinite_trace",
                relative_identifier,
                failure_reason_counts,
                failure_examples,
                example_limit,
            )
            if progress and progress_every > 0 and index % progress_every == 0:
                progress(index, total_to_scan)
            continue

        current_min = float(np.min(values))
        current_max = float(np.max(values))
        current_peak = float(np.max(np.abs(values)))
        finite_min = current_min if finite_min is None else min(finite_min, current_min)
        finite_max = current_max if finite_max is None else max(finite_max, current_max)
        peak_absolute_level = (
            current_peak
            if peak_absolute_level is None
            else max(peak_absolute_level, current_peak)
        )
        noninteger_value_count += int(
            np.count_nonzero(values != np.rint(values))
        )
        if current_min == current_max:
            constant_trace_count += 1
        if current_peak > gate_threshold:
            threshold_exceeding_trace_count += 1
        if _has_repeated_high_level_plateau(values, plateau_config):
            plateau_trace_count += 1
        raw_rms_values.append(float(np.sqrt(np.mean(np.square(values)))))

        detection = _detect_front_echo(values, front_config)
        if detection["noise_rms"] is not None:
            noise_rms_values.append(float(detection["noise_rms"]))
        if detection["snr_db"] is not None:
            snr_values.append(float(detection["snr_db"]))
        if detection["detected"]:
            detected_positions.append(float(detection["position_samples"]))
            detected_widths.append(float(detection["width_samples"]))
            detected_confidences.append(float(detection["snr_db"]))
            detected_front_peaks.append(float(detection["front_peak_absolute_level"]))
            detected_front_energies.append(float(detection["front_energy_level_squared"]))
        else:
            _record_detection_failure(
                str(detection["failure_reason"]),
                relative_identifier,
                failure_reason_counts,
                failure_examples,
                example_limit,
            )

        if progress and progress_every > 0 and index % progress_every == 0:
            progress(index, total_to_scan)

    if progress and total_to_scan and (
        progress_every <= 0 or total_to_scan % progress_every != 0
    ):
        progress(total_to_scan, total_to_scan)

    detected_trace_count = len(detected_positions)
    detection_failure_count = sum(failure_reason_counts.values())
    detection_denominator = detected_trace_count + detection_failure_count
    detection_rate = _safe_fraction(
        detected_trace_count,
        detection_denominator,
    )
    constant_fraction = _safe_fraction(
        constant_trace_count,
        readable_trace_count,
    )
    zero_fraction = _safe_fraction(zero_count, parsed_sample_count)
    plateau_fraction = _safe_fraction(
        plateau_trace_count,
        readable_trace_count,
    )
    threshold_exceeding_fraction = _safe_fraction(
        threshold_exceeding_trace_count,
        readable_trace_count,
    )

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "integrity.expected_grid_coverage",
        "integrity",
        "PASS"
        if missing_trace_count == 0
        and duplicate_trace_count == 0
        and bad_filename_count == 0
        else "FAIL",
        {
            "missing": missing_trace_count,
            "duplicates": duplicate_trace_count,
            "bad_filenames": bad_filename_count,
        },
        {"all": 0},
        "Regular-grid coordinates must be present exactly once.",
    )
    _add_check(
        checks,
        "integrity.trace_readability",
        "integrity",
        "PASS" if unreadable_trace_count == 0 else "FAIL",
        unreadable_trace_count,
        0,
        "Every selected coordinate-resolved TXT trace must be readable numeric text.",
    )
    _add_check(
        checks,
        "integrity.fixed_trace_length",
        "integrity",
        "PASS" if invalid_length_count == 0 else "FAIL",
        invalid_length_count,
        0,
        "Selected fixed-length traces must match the registered sample count.",
    )
    _threshold_check(
        checks,
        "integrity.allowed_missing_trace_fraction",
        "integrity",
        _safe_fraction(missing_trace_count, expected_trace_count),
        thresholds.get("allowed_missing_trace_fraction"),
        "maximum",
    )
    _threshold_check(
        checks,
        "integrity.allowed_invalid_length_fraction",
        "integrity",
        _safe_fraction(invalid_length_count, max(readable_trace_count, 1)),
        thresholds.get("allowed_invalid_length_fraction"),
        "maximum",
    )
    _add_check(
        checks,
        "integrity.content_audit_scope",
        "integrity",
        "PASS" if sampling_details["mode"] == "full" else "WARN",
        {
            "selected_coordinate_count": len(selected_coordinates),
            "registered_grid_trace_count": expected_trace_count,
        },
        (
            "all coordinate-resolved traces"
            if sampling_details["mode"] == "full"
            else "spatially distributed spot check"
        ),
        (
            "Content-derived metrics are full-scan."
            if sampling_details["mode"] == "full"
            else "Content-derived metrics are sample-based and must not be presented as full validation."
        ),
    )
    _add_check(
        checks,
        "numeric.nonfinite_count",
        "numeric",
        "PASS" if nonfinite_count == 0 else "FAIL",
        nonfinite_count,
        0,
        "Raw traces must not contain NaN or infinite values.",
    )
    _add_check(
        checks,
        "numeric.empty_trace_count",
        "numeric",
        "PASS" if empty_trace_count == 0 else "FAIL",
        empty_trace_count,
        0,
        "Raw trace files must contain at least one numeric sample.",
    )
    _add_check(
        checks,
        "numeric.signed_integer_text",
        "numeric",
        "PASS" if noninteger_value_count == 0 else "FAIL",
        noninteger_value_count,
        0,
        "Parsed finite samples must preserve the signed-integer export contract.",
    )
    _threshold_check(
        checks,
        "numeric.maximum_constant_trace_fraction",
        "numeric",
        constant_fraction,
        thresholds.get("maximum_constant_trace_fraction"),
        "maximum",
    )
    _threshold_check(
        checks,
        "numeric.maximum_plateau_fraction",
        "numeric",
        plateau_fraction,
        thresholds.get("maximum_plateau_fraction"),
        "maximum",
    )
    _add_check(
        checks,
        "temporal.detector_configuration",
        "temporal",
        "PASS",
        str(front_config["name"]),
        "explicit label-free detector",
        "Reference-event failures are recorded and never imputed.",
    )
    _threshold_check(
        checks,
        "temporal.minimum_reference_detection_rate",
        "temporal",
        detection_rate,
        thresholds.get("minimum_reference_detection_rate"),
        "minimum",
    )
    _threshold_check(
        checks,
        "temporal.minimum_snr_proxy_db",
        "temporal",
        _median_or_none(snr_values),
        thresholds.get("minimum_snr_proxy_db"),
        "minimum",
    )
    _add_check(
        checks,
        "amplitude.gate_threshold_role",
        "amplitude",
        "PASS",
        {
            "gate_threshold_level": gate_threshold,
            "traces_above_threshold": threshold_exceeding_trace_count,
        },
        "descriptive_only_not_clipping",
        "Values above 100 level are allowed and do not imply clipping.",
    )
    _add_check(
        checks,
        "amplitude.clipping_code",
        "amplitude",
        "NOT_EVALUATED",
        None,
        "digitizer_or_export_clipping_code",
        "Clipping remains unevaluated because the calibrated clipping code is unknown.",
    )

    metadata_schema_valid = True
    required_non_null_fields_present = all(
        value is not None
        for value in (
            identity.get("dataset_id"),
            identity.get("specimen_id"),
            identity.get("acquisition_batch_id"),
            identity.get("scan_id"),
            acquisition.get("sampling_rate_hz"),
            raw_layout.get("export_format"),
            raw_layout.get("raw_dtype"),
            raw_layout.get("trace_length_mode"),
            raw_layout.get("file_pattern"),
            geometry.get("layout"),
            time_reference.get("front_surface_material"),
            provenance.get("source_ref"),
        )
    )
    sampling_rate_matches = sampling_rate_hz == 200_000_000
    raw_unit_matches = raw_layout.get("raw_unit") == "level"
    no_acquisition_averaging = (
        acquisition.get("acquisition_averaging_enabled") is False
    )
    k_value = structure.get("K")
    k_supported = k_value is None or int(k_value) in (2, 3)
    interfaces = structure.get("interfaces")
    interface_count_matches = (
        k_value is None
        or isinstance(interfaces, Sequence)
        and not isinstance(interfaces, (str, bytes))
        and len(interfaces) == int(k_value)
    )
    variable_length_contract_valid = (
        trace_length_mode == "variable_within_batch"
        or expected_trace_samples is not None
    )
    geometry_contract_valid = (
        geometry.get("layout") == "regular_grid"
        and grid_x > 0
        and grid_y > 0
    )
    fixed_contract_values = (
        metadata_schema_valid,
        required_non_null_fields_present,
        sampling_rate_matches,
        raw_unit_matches,
        no_acquisition_averaging,
        k_supported,
        interface_count_matches,
        variable_length_contract_valid,
        geometry_contract_valid,
    )
    _add_check(
        checks,
        "contract.fixed_manifest_contract",
        "contract",
        "PASS" if all(fixed_contract_values) else "FAIL",
        list(fixed_contract_values),
        [True] * len(fixed_contract_values),
        "Registered acquisition facts must match the frozen M0 contract.",
    )
    _add_check(
        checks,
        "contract.label_free",
        "contract",
        "PASS",
        True,
        True,
        "The scanner does not access label fields.",
    )
    _add_check(
        checks,
        "contract.raw_before_denoising",
        "contract",
        "PASS",
        False,
        False,
        "No denoising or per-trace normalization is applied.",
    )

    integrity_status = (
        "FAIL"
        if any(
            value > 0
            for value in (
                missing_trace_count,
                duplicate_trace_count,
                bad_filename_count,
                unreadable_trace_count,
                invalid_length_count,
            )
        )
        else (
            "WARN"
            if sampling_details["mode"] != "full"
            or selected_missing_count > 0
            else "PASS"
        )
    )
    numeric_status = (
        "FAIL"
        if nonfinite_count > 0
        or empty_trace_count > 0
        or noninteger_value_count > 0
        else "PASS"
    )
    temporal_status = _section_status(checks, "temporal")
    amplitude_status = _section_status(checks, "amplitude")
    contract_status = (
        "PASS" if all(fixed_contract_values) else "FAIL"
    )

    sample_interval_ns = 1e9 / sampling_rate_hz
    position_samples_summary = _five_number_summary(detected_positions)
    position_us_summary = _scale_summary(
        position_samples_summary,
        1e6 / sampling_rate_hz,
    )
    trace_length_summary = _trace_length_summary(trace_lengths)
    smoke_schema_version = str(smoke_schema.get("schema_version"))
    generated_time = generated_at or datetime.now(timezone.utc).isoformat()

    summary: dict[str, Any] = {
        "summary_identity": {
            "dataset_id": str(identity["dataset_id"]),
            "specimen_id": str(identity["specimen_id"]),
            "acquisition_batch_id": str(identity["acquisition_batch_id"]),
            "scan_id": str(identity["scan_id"]),
            "metadata_schema_version": str(manifest["schema_version"]),
            "smoke_schema_version": smoke_schema_version,
            "scanner_version": str(scanner_config["scanner_version"]),
            "scanner_config_checksum": _canonical_checksum(scanner_config),
            "source_checksum": provenance.get("source_checksum"),
            "generated_at": generated_time,
        },
        "audit_scope": {
            "mode": sampling_details["mode"],
            "filename_inventory_scope": "full_registered_grid",
            "content_metric_scope": (
                "all_coordinate_resolved_traces"
                if sampling_details["mode"] == "full"
                else "selected_coordinate_traces"
            ),
            "registered_grid_trace_count": expected_trace_count,
            "selected_coordinate_count": len(selected_coordinates),
            "selected_fraction": _safe_fraction(
                len(selected_coordinates),
                expected_trace_count,
            ),
            "selection": sampling_details,
            "limitation": (
                "none_full_content_scan"
                if sampling_details["mode"] == "full"
                else "content_statistics_are_spatial_spot_check_not_full_content_validation"
            ),
        },
        "integrity": {
            "expected_trace_count": expected_trace_count,
            "observed_trace_count": len(all_txt_paths),
            "readable_trace_count": readable_trace_count,
            "missing_trace_count": missing_trace_count,
            "duplicate_trace_count": duplicate_trace_count,
            "bad_filename_count": bad_filename_count,
            "unreadable_trace_count": unreadable_trace_count,
            "trace_length_mode": trace_length_mode,
            "trace_length_samples": trace_length_summary,
            "invalid_length_count": invalid_length_count,
            "coordinate_duplicate_count": duplicate_trace_count,
            "missing_examples": missing_examples,
            "unreadable_examples": unreadable_examples,
            "status": integrity_status,
        },
        "numeric": {
            "observed_dtype": (
                "signed_integer_text"
                if noninteger_value_count == 0
                else "numeric_text_noninteger_observed"
            ),
            "raw_unit": str(raw_layout["raw_unit"]),
            "finite_min": finite_min,
            "finite_max": finite_max,
            "peak_absolute_level": peak_absolute_level,
            "nonfinite_count": nonfinite_count,
            "empty_trace_count": empty_trace_count,
            "constant_trace_count": constant_trace_count,
            "constant_trace_fraction": constant_fraction,
            "zero_fraction": zero_fraction,
            "plateau_fraction": plateau_fraction,
            "status": numeric_status,
        },
        "temporal": {
            "sampling_rate_hz": sampling_rate_hz,
            "sample_interval_ns": sample_interval_ns,
            "reference_event": str(time_reference["reference_event"]),
            "reference_rule": "strong_rf_echo_packet_at_front_of_cropped_time_series",
            "front_surface_material": str(time_reference["front_surface_material"]),
            "front_search_window": {
                "start_sample": int(front_config["search_start_sample"]),
                "stop_sample_exclusive": int(
                    front_config["search_stop_sample_exclusive"]
                ),
                "start_us": int(front_config["search_start_sample"])
                * 1e6
                / sampling_rate_hz,
                "stop_us_exclusive": int(
                    front_config["search_stop_sample_exclusive"]
                )
                * 1e6
                / sampling_rate_hz,
            },
            "detector_name": str(front_config["name"]),
            "detector_config_checksum": _canonical_checksum(front_config),
            "detected_trace_count": detected_trace_count,
            "detection_failure_count": detection_failure_count,
            "detection_rate": detection_rate,
            "reference_position_samples": position_samples_summary,
            "reference_position_us": position_us_summary,
            "echo_packet_width_samples": _distribution_summary(detected_widths),
            "detector_confidence": {
                "metric": "front_peak_moving_rms_over_tail_noise_rms_db",
                "detection_threshold_db": float(front_config["detection_snr_db"]),
                "distribution": _distribution_summary(detected_confidences),
            },
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "failure_examples": failure_examples,
            "status": temporal_status,
        },
        "amplitude": {
            "raw_unit": str(raw_layout["raw_unit"]),
            "gate_threshold_level": gate_threshold,
            "gate_threshold_role": "acquisition_detection_threshold",
            "values_above_gate_threshold_allowed": bool(
                acquisition["values_above_gate_threshold_allowed"]
            ),
            "traces_exceeding_gate_threshold_count": (
                threshold_exceeding_trace_count
            ),
            "traces_exceeding_gate_threshold_fraction": (
                threshold_exceeding_fraction
            ),
            "raw_rms_level": _required_rms_summary(raw_rms_values),
            "front_echo_peak_absolute_level": _distribution_summary(
                detected_front_peaks
            ),
            "front_echo_energy_level_squared": _distribution_summary(
                detected_front_energies
            ),
            "noise_rms_level": {
                "window": "last_n_samples_after_trace_median_removal",
                "tail_noise_samples": int(front_config["tail_noise_samples"]),
                "distribution": _distribution_summary(noise_rms_values),
            },
            "snr_proxy_db": {
                "definition": "20log10(front_peak_moving_rms/tail_noise_rms)",
                "distribution": _distribution_summary(snr_values),
            },
            "clipping_code": None,
            "clipping_fraction": None,
            "status": amplitude_status,
        },
        "contract": {
            "metadata_schema_valid": metadata_schema_valid,
            "required_non_null_fields_present": required_non_null_fields_present,
            "sampling_rate_matches_contract": sampling_rate_matches,
            "raw_unit_matches_contract": raw_unit_matches,
            "no_acquisition_averaging": no_acquisition_averaging,
            "K_is_supported_when_present": k_supported,
            "interface_count_matches_K_when_present": interface_count_matches,
            "variable_trace_length_contract_valid": variable_length_contract_valid,
            "scan_geometry_contract_valid": geometry_contract_valid,
            "label_fields_were_not_read": True,
            "absolute_paths_absent_from_summary": True,
            "denoising_applied": False,
            "status": contract_status,
        },
    }

    paths_absent = not _contains_absolute_path(summary)
    summary["contract"]["absolute_paths_absent_from_summary"] = paths_absent
    _add_check(
        checks,
        "contract.privacy",
        "contract",
        "PASS" if paths_absent else "FAIL",
        paths_absent,
        True,
        "Generated summaries must not contain absolute filesystem paths.",
    )
    if not paths_absent:
        summary["contract"]["status"] = "FAIL"

    summary["outcome"] = _build_outcome(checks)
    return _to_builtin(summary)


def _detect_front_echo(
    values: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(config["search_start_sample"])
    stop = int(config["search_stop_sample_exclusive"])
    moving_window = int(config["moving_rms_samples"])
    tail_samples = int(config["tail_noise_samples"])
    threshold_db = float(config["detection_snr_db"])
    width_fraction = float(config["width_fraction_of_peak"])
    if (
        start < 0
        or stop <= start
        or moving_window <= 0
        or tail_samples <= 0
        or values.size < max(stop, tail_samples)
        or stop - start < moving_window
    ):
        return _failed_detection("trace_too_short")

    baseline = float(np.median(values[-tail_samples:]))
    centered = values - baseline
    tail = centered[-tail_samples:]
    noise_rms = float(np.sqrt(np.mean(np.square(tail))))
    front = centered[start:stop]
    squared = np.square(front)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    window_energy = cumulative[moving_window:] - cumulative[:-moving_window]
    moving_rms = np.sqrt(window_energy / moving_window)
    if moving_rms.size == 0:
        return _failed_detection("trace_too_short", noise_rms=noise_rms)

    peak_index = int(np.argmax(moving_rms))
    peak_rms = float(moving_rms[peak_index])
    epsilon = 1e-12
    snr_db = float(
        20.0 * math.log10((peak_rms + epsilon) / (noise_rms + epsilon))
    )
    if snr_db < threshold_db:
        return _failed_detection(
            f"front_snr_below_{_compact_number(threshold_db)}_db",
            noise_rms=noise_rms,
            snr_db=snr_db,
        )

    above = moving_rms >= width_fraction * peak_rms
    left = peak_index
    right = peak_index
    while left > 0 and bool(above[left - 1]):
        left -= 1
    while right + 1 < above.size and bool(above[right + 1]):
        right += 1
    packet_start = start + left
    packet_stop = start + right + moving_window
    packet = centered[packet_start:packet_stop]
    return {
        "detected": True,
        "failure_reason": None,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "position_samples": start + peak_index + (moving_window - 1) / 2.0,
        "width_samples": float(packet_stop - packet_start),
        "front_peak_absolute_level": float(np.max(np.abs(packet))),
        "front_energy_level_squared": float(np.sum(np.square(packet))),
    }


def _failed_detection(
    reason: str,
    *,
    noise_rms: float | None = None,
    snr_db: float | None = None,
) -> dict[str, Any]:
    return {
        "detected": False,
        "failure_reason": reason,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "position_samples": None,
        "width_samples": None,
        "front_peak_absolute_level": None,
        "front_energy_level_squared": None,
    }


def _has_repeated_high_level_plateau(
    values: np.ndarray,
    config: Mapping[str, Any],
) -> bool:
    minimum_level = float(config["minimum_absolute_level"])
    minimum_run = int(config["minimum_run_samples"])
    if minimum_run <= 1:
        return bool(np.any(np.abs(values) >= minimum_level))
    if values.size < minimum_run:
        return False
    equal_high_pairs = (
        (values[1:] == values[:-1])
        & (np.abs(values[1:]) >= minimum_level)
        & (np.abs(values[:-1]) >= minimum_level)
    )
    required_pairs = minimum_run - 1
    if required_pairs == 1:
        return bool(np.any(equal_high_pairs))
    run_sum = np.convolve(
        equal_high_pairs.astype(np.int8),
        np.ones(required_pairs, dtype=np.int8),
        mode="valid",
    )
    return bool(np.any(run_sum == required_pairs))


def _record_detection_failure(
    reason: str,
    identifier: str,
    counts: Counter[str],
    examples: list[str],
    example_limit: int,
) -> None:
    counts[reason] += 1
    if len(examples) < example_limit:
        examples.append(f"{identifier}:{reason}")


def _missing_examples(
    coordinates: Mapping[tuple[int, int], Path],
    grid_x: int,
    grid_y: int,
    file_pattern: str,
    limit: int,
) -> list[str]:
    examples: list[str] = []
    for x_index in range(grid_x):
        for y_index in range(grid_y):
            if (x_index, y_index) not in coordinates:
                examples.append(
                    file_pattern.format(ix=x_index, iy=y_index)
                )
                if len(examples) >= limit:
                    return examples
    return examples


def _select_coordinates(
    grid_x: int,
    grid_y: int,
    config: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    mode = str(config.get("mode", "full"))
    if mode == "full":
        coordinates = [
            (x_index, y_index)
            for x_index in range(grid_x)
            for y_index in range(grid_y)
        ]
        return coordinates, {"mode": "full"}
    if mode != "regular_grid_stride":
        raise ValueError(f"unsupported sampling mode: {mode}")

    start_x = int(config.get("start_x", 0))
    start_y = int(config.get("start_y", 0))
    stride_x = int(config.get("stride_x", 1))
    stride_y = int(config.get("stride_y", 1))
    if (
        start_x < 0
        or start_y < 0
        or start_x >= grid_x
        or start_y >= grid_y
        or stride_x <= 0
        or stride_y <= 0
    ):
        raise ValueError("regular_grid_stride sampling parameters are invalid")
    coordinates = [
        (x_index, y_index)
        for x_index in range(start_x, grid_x, stride_x)
        for y_index in range(start_y, grid_y, stride_y)
    ]
    return coordinates, {
        "mode": mode,
        "start_x": start_x,
        "start_y": start_y,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "selection_basis": str(
            config.get("selection_basis", "deterministic_regular_grid_stride")
        ),
    }


def _trace_length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "minimum": 0,
            "maximum": 0,
            "median": 0.0,
            "p01": 0.0,
            "p99": 0.0,
            "unique_length_count": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
        "median": float(np.median(array)),
        "p01": float(np.percentile(array, 1)),
        "p99": float(np.percentile(array, 99)),
        "unique_length_count": int(np.unique(array).size),
    }


def _five_number_summary(
    values: Sequence[float],
) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "p01": float(np.percentile(array, 1)),
        "p99": float(np.percentile(array, 99)),
    }


def _distribution_summary(
    values: Sequence[float],
) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _required_rms_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p01": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p01": float(np.percentile(array, 1)),
        "p99": float(np.percentile(array, 99)),
    }


def _scale_summary(
    summary: Mapping[str, float] | None,
    scale: float,
) -> dict[str, float] | None:
    if summary is None:
        return None
    return {key: float(value * scale) for key, value in summary.items()}


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _to_builtin(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    section: str,
    status: str,
    observed: Any,
    expected: Any,
    message: str,
) -> None:
    if status not in CHECK_STATUSES:
        raise ValueError(f"unknown check status: {status}")
    checks.append(
        {
            "check_id": check_id,
            "section": section,
            "status": status,
            "observed": _to_builtin(observed),
            "expected": _to_builtin(expected),
            "message": message,
        }
    )


def _threshold_check(
    checks: list[dict[str, Any]],
    check_id: str,
    section: str,
    observed: float | None,
    threshold: Any,
    direction: str,
) -> None:
    if threshold is None:
        _add_check(
            checks,
            check_id,
            section,
            "NOT_EVALUATED",
            observed,
            None,
            "The batch-specific acceptance threshold is not frozen.",
        )
        return
    numeric_threshold = float(threshold)
    if observed is None:
        status = "FAIL"
    elif direction == "minimum":
        status = "PASS" if observed >= numeric_threshold else "FAIL"
    elif direction == "maximum":
        status = "PASS" if observed <= numeric_threshold else "FAIL"
    else:
        raise ValueError(f"unknown threshold direction: {direction}")
    _add_check(
        checks,
        check_id,
        section,
        status,
        observed,
        {direction: numeric_threshold},
        "Observed value was compared with the frozen batch threshold.",
    )


def _section_status(
    checks: Sequence[Mapping[str, Any]],
    section: str,
) -> str:
    statuses = {
        str(check["status"])
        for check in checks
        if check["section"] == section
    }
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_EVALUATED" in statuses:
        return "NOT_EVALUATED"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _build_outcome(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(check["status"]) for check in checks)
    failures = [
        str(check["message"])
        for check in checks
        if check["status"] == "FAIL"
    ]
    warnings_list = [
        str(check["message"])
        for check in checks
        if check["status"] == "WARN"
    ]
    incomplete = [
        str(check["message"])
        for check in checks
        if check["status"] == "NOT_EVALUATED"
    ]
    if failures:
        overall_status = "FAIL"
    elif incomplete:
        overall_status = "INCOMPLETE"
    elif warnings_list:
        overall_status = "WARN"
    else:
        overall_status = "PASS"
    return {
        "overall_status": overall_status,
        "check_counts": {
            "pass": status_counts["PASS"],
            "warn": status_counts["WARN"],
            "fail": status_counts["FAIL"],
            "not_evaluated": status_counts["NOT_EVALUATED"],
        },
        "checks": list(checks),
        "failure_reasons": failures,
        "warning_reasons": warnings_list,
        "incomplete_reasons": incomplete,
    }


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    return bool(_WINDOWS_ABSOLUTE_RE.match(value)) or value.startswith("/")


def _append_example(values: list[str], value: str, limit: int) -> None:
    if len(values) < limit:
        values.append(value)


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _compact_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "_")


def _progress_printer(completed: int, total: int) -> None:
    print(f"label-free smoke progress: {completed}/{total}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic label-free A3 raw-data quality checks."
    )
    parser.add_argument("--root", required=True, help="Local raw TXT directory.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata-schema", required=True)
    parser.add_argument("--smoke-schema", required=True)
    parser.add_argument("--scanner-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    manifest = load_label_free_manifest(args.manifest, args.metadata_schema)
    smoke_schema = load_yaml_mapping(args.smoke_schema)
    scanner_config = load_yaml_mapping(args.scanner_config)
    summary = scan_label_free(
        args.root,
        manifest,
        smoke_schema,
        scanner_config,
        progress=_progress_printer,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_status": summary["outcome"]["overall_status"],
                "observed_trace_count": summary["integrity"]["observed_trace_count"],
                "detection_rate": summary["temporal"]["detection_rate"],
                "output_name": output_path.name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 2 if summary["outcome"]["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
