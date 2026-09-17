"""Build a no-RF acceptance report for the A3 M0-02 metadata contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from a3_ultrasound.data.metadata import load_yaml_mapping, validate_manifest


EXPECTED_SCHEMA_VERSION = "a3-acquisition-metadata-v2"
EXPECTED_UNITS = {
    "field_groups.acquisition.sampling_rate_hz.unit": "Hz",
    "field_groups.acquisition.adc_bits.unit": "bit",
    "field_groups.acquisition.gain_db.unit": "dB",
    "field_groups.acquisition.probe_center_frequency_hz.unit": "Hz",
    "field_groups.acquisition.excitation.object_schema.center_frequency_mhz.unit": "MHz",
    "field_groups.acquisition.excitation.object_schema.num_cycles.unit": "cycle",
    "field_groups.acquisition.window_profile.object_schema.window_left_samples.unit": "sample",
    "field_groups.acquisition.window_profile.object_schema.window_left_s.unit": "s",
    "field_groups.acquisition.window_profile.object_schema.window_right_samples.unit": "sample",
    "field_groups.acquisition.window_profile.object_schema.window_right_s.unit": "s",
    "field_groups.acquisition.gate_threshold_level.unit": "level",
    "field_groups.raw_layout.raw_unit.const": "level",
    "field_groups.raw_layout.trace_samples.unit": "sample",
    "field_groups.scan_geometry.scan_step_x_mm.unit": "mm",
    "field_groups.scan_geometry.scan_step_y_mm.unit": "mm",
    "field_groups.time_reference.pretrigger_samples.unit": "sample",
    "field_groups.time_reference.batch_reference_samples.unit": "sample",
    "field_groups.time_reference.batch_reference_s.unit": "s",
    "field_groups.structure.layers.item_schema.thickness_mm.unit": "mm",
    "field_groups.structure.layers.item_schema.sound_speed_mps.unit": "m/s",
    "field_groups.structure.total_thickness_mm.unit": "mm",
    "field_groups.structure.sound_speed_mps.unit": "m/s",
    "field_groups.structure.interfaces.item_schema.nominal_depth_mm.unit": "mm",
}


def _get_dotted(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    metadata_schema_path: Path,
    manifest_path: Path,
    smoke_schema_path: Path,
) -> dict[str, Any]:
    metadata_schema = load_yaml_mapping(metadata_schema_path)
    manifest = load_yaml_mapping(manifest_path)
    smoke_schema = load_yaml_mapping(smoke_schema_path)

    stage_results: dict[str, str] = {}
    for stage in ("before_label_free_smoke", "before_network_api_derivation"):
        validate_manifest(manifest, metadata_schema, stage=stage)
        stage_results[stage] = "PASS"

    unit_results = {
        path: {
            "expected": expected,
            "actual": _get_dotted(metadata_schema, path),
        }
        for path, expected in EXPECTED_UNITS.items()
    }
    units_complete = all(
        result["actual"] == result["expected"] for result in unit_results.values()
    )
    null_policy = metadata_schema.get("null_policy")
    smoke_input_contract = smoke_schema.get("input_contract")
    checks = {
        "schema_version_is_current": (
            metadata_schema.get("schema_version") == EXPECTED_SCHEMA_VERSION
        ),
        "manifest_schema_matches_metadata_schema": (
            manifest.get("schema_version") == metadata_schema.get("schema_version")
        ),
        "smoke_input_schema_matches_metadata_schema": (
            isinstance(smoke_input_contract, Mapping)
            and smoke_input_contract.get("metadata_schema")
            == metadata_schema.get("schema_version")
        ),
        "unknown_values_must_be_null": (
            isinstance(null_policy, Mapping)
            and null_policy.get("representation") == "yaml_null"
            and null_policy.get("rule") == "unknown_values_must_be_null"
            and null_policy.get("forbid_guessed_values") is True
            and null_policy.get("forbid_placeholder_numbers") is True
        ),
        "critical_units_are_explicit": units_complete,
        "label_free_stage_valid": stage_results["before_label_free_smoke"] == "PASS",
        "network_api_stage_valid": (
            stage_results["before_network_api_derivation"] == "PASS"
        ),
        "raw_rf_not_read": True,
        "labels_not_required_for_acceptance": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "a3-m0-02-metadata-contract-acceptance-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M0-02",
        "status": status,
        "scope": "schema_manifest_and_downstream_version_contract_without_raw_rf",
        "checks": checks,
        "stage_validation": stage_results,
        "unit_contract": unit_results,
        "artifacts": {
            "metadata_schema": metadata_schema_path.name,
            "manifest": manifest_path.name,
            "smoke_schema": smoke_schema_path.name,
        },
        "artifact_sha256": {
            metadata_schema_path.name: _sha256(metadata_schema_path),
            manifest_path.name: _sha256(manifest_path),
            smoke_schema_path.name: _sha256(smoke_schema_path),
        },
        "interpretation_boundary": (
            "PASS proves the metadata field, unit, null, stage, and downstream "
            "schema-version contracts only. It reads no raw RF or labels and "
            "does not provide model-performance or physical-calibration evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-schema", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--smoke-schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report = build_report(
        args.metadata_schema,
        args.manifest,
        args.smoke_schema,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "status": report["status"],
                "raw_rf_not_read": report["checks"]["raw_rf_not_read"],
                "output_name": args.output.name,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
