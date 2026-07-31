from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from a3_ultrasound.data.metadata import (
    ManifestLoadError,
    ManifestValidationError,
    load_and_validate_manifest,
    load_yaml_mapping,
    validate_manifest,
)


def _schema() -> dict:
    nullable_string = {"type": "string", "required": False, "nullable": True}
    return {
        "schema_version": "a3-acquisition-metadata-v2",
        "privacy_policy": {
            "forbid_absolute_paths": True,
            "forbid_secrets": True,
        },
        "project_contract": {
            "sampling_rate_hz": {"expected_value": 200_000_000},
        },
        "field_groups": {
            "identity": {
                "dataset_id": {"type": "string", "required": True, "rule": "opaque_id"},
                "specimen_id": {"type": "string", "required": True, "rule": "opaque_id"},
                "acquisition_batch_id": {
                    "type": "string",
                    "required": True,
                    "rule": "opaque_id",
                },
                "scan_id": {"type": "string", "required": True, "rule": "opaque_id"},
                "scan_role": {
                    "type": "string",
                    "required": True,
                    "allowed": ["train_candidate", "reference", "qc_only", "unlabeled"],
                },
            },
            "acquisition": {
                "sampling_rate_hz": {
                    "type": "integer",
                    "required": True,
                    "nullable_until_batch_registration": True,
                },
                "acquisition_profile_id": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "rule": "opaque_id",
                },
                "window_profile": {
                    "type": "object",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "object_schema": {
                        "profile_id": {
                            "type": "string",
                            "required": True,
                            "rule": "opaque_id",
                        },
                        "window_left_samples": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                        "window_left_s": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                        "window_right_samples": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                        "window_right_s": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                    },
                },
                "probe_code": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "rule": "opaque_code",
                },
                "excitation": {
                    "type": "object",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "object_schema": {
                        "waveform": {
                            "type": "string",
                            "required": True,
                            "nullable": True,
                            "allowed": ["sine_tone_burst", "other"],
                        },
                        "center_frequency_mhz": {
                            "type": "number",
                            "required": True,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                        "num_cycles": {
                            "type": "number",
                            "required": True,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                        "window_type": {
                            "type": "string",
                            "required": True,
                            "nullable": True,
                            "allowed": ["rectangular", "hann", "other"],
                        },
                        "status": {
                            "type": "string",
                            "required": True,
                            "allowed": [
                                "confirmed_by_record",
                                "reported_by_user",
                                "user_estimate_pending_confirmation",
                            ],
                        },
                    },
                },
            },
            "raw_layout": {
                "export_format": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                },
                "raw_dtype": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                },
                "trace_length_mode": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "allowed": ["fixed_within_batch", "variable_within_batch"],
                },
                "trace_samples": {
                    "type": "integer",
                    "required": False,
                    "nullable": True,
                    "minimum": 1,
                },
                "trace_lengths_ref": {
                    **nullable_string,
                    "rule": "relative_or_opaque_reference",
                },
                "file_pattern": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                },
            },
            "scan_geometry": {
                "layout": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "allowed": ["regular_grid", "irregular_points"],
                },
                "grid_shape": {
                    "type": "array",
                    "required": False,
                    "nullable": True,
                    "item_type": "integer",
                    "length": 2,
                },
                "scan_step_x_mm": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum_exclusive": 0,
                },
                "scan_step_y_mm": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum_exclusive": 0,
                },
                "scan_coordinates_ref": {
                    **nullable_string,
                    "rule": "relative_or_opaque_reference",
                },
            },
            "time_reference": {
                "front_surface_material": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                },
                "batch_reference_samples": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum": 0,
                },
                "batch_reference_s": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum": 0,
                },
            },
            "structure": {
                "K": {
                    "type": "integer",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "allowed": [2, 3],
                },
                "K_source": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "allowed": ["declared_by_user", "derived_from_layer_stack"],
                },
                "layers": {
                    "type": "array",
                    "required": False,
                    "nullable": True,
                    "item_schema": {
                        "layer_order": {"type": "integer", "required": True, "minimum": 1},
                        "material": {"type": "string", "required": True},
                        "thickness_mm": {
                            "type": "number",
                            "required": True,
                            "minimum_exclusive": 0,
                        },
                        "sound_speed_mps": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                            "minimum_exclusive": 0,
                        },
                    },
                },
                "total_thickness_mm": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum_exclusive": 0,
                },
                "sound_speed_mps": {
                    "type": "number",
                    "required": False,
                    "nullable": True,
                    "minimum_exclusive": 0,
                },
                "interfaces": {
                    "type": "array",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "length_must_equal_field": "K",
                    "item_schema": {
                        "interface_order": {
                            "type": "integer",
                            "required": True,
                            "minimum": 1,
                        },
                        "nominal_depth_mm": {
                            "type": "number",
                            "required": False,
                            "nullable": True,
                        },
                    },
                },
            },
            "labels": {
                "embedded_defect_manifest_ref": {
                    **nullable_string,
                    "rule": "relative_or_opaque_reference",
                },
                "good_region_ref": {
                    **nullable_string,
                    "rule": "relative_or_opaque_reference",
                },
                "uncertain_boundary_ref": {
                    **nullable_string,
                    "rule": "relative_or_opaque_reference",
                },
            },
            "preprocessing": {
                "preserve_phase": {"type": "boolean", "required": True, "const": True},
                "preserve_delay": {"type": "boolean", "required": True, "const": True},
                "preserve_relative_amplitude": {
                    "type": "boolean",
                    "required": True,
                    "const": True,
                },
            },
            "provenance": {
                "source_ref": {
                    "type": "string",
                    "required": True,
                    "nullable_until_batch_registration": True,
                    "rule": "relative_or_opaque_reference",
                },
                "created_at": {
                    "type": "string",
                    "required": True,
                    "format": "iso8601",
                },
                "created_by": {
                    "type": "string",
                    "required": True,
                    "rule": "non_personal_process_identifier",
                },
            },
        },
        "stage_requirements": {
            "schema_only": {"allow_null_batch_values": True},
            "before_label_free_smoke": {
                "require_non_null": [
                    "identity.dataset_id",
                    "identity.specimen_id",
                    "identity.acquisition_batch_id",
                    "identity.scan_id",
                    "acquisition.sampling_rate_hz",
                    "raw_layout.export_format",
                    "raw_layout.raw_dtype",
                    "raw_layout.trace_length_mode",
                    "raw_layout.file_pattern",
                    "scan_geometry.layout",
                    "time_reference.front_surface_material",
                    "provenance.source_ref",
                ]
            },
            "before_network_api_derivation": {
                "require_non_null": [
                    "structure.K",
                    "structure.K_source",
                    "structure.total_thickness_mm",
                    "structure.interfaces",
                ]
            },
            "before_supervised_training": {
                "require_non_null": [
                    "labels.embedded_defect_manifest_ref",
                    "labels.good_region_ref",
                    "labels.uncertain_boundary_ref",
                ]
            },
        },
    }


def _manifest() -> dict:
    return {
        "schema_version": "a3-acquisition-metadata-v2",
        "identity": {
            "dataset_id": "ds_a3_001",
            "specimen_id": "sp_a3_001",
            "acquisition_batch_id": "batch_a3_001",
            "scan_id": "scan_a3_001",
            "scan_role": "unlabeled",
        },
        "acquisition": {
            "sampling_rate_hz": 200_000_000,
            "acquisition_profile_id": "ap_test_v1",
            "window_profile": {
                "profile_id": "wp_test_v1",
                "window_left_samples": None,
                "window_left_s": None,
                "window_right_samples": None,
                "window_right_s": None,
            },
            "probe_code": None,
            "excitation": {
                "waveform": "sine_tone_burst",
                "center_frequency_mhz": 10.0,
                "num_cycles": 5,
                "window_type": "hann",
                "status": "user_estimate_pending_confirmation",
            },
        },
        "raw_layout": {
            "export_format": "signed_integer_text",
            "raw_dtype": "signed_integer_text",
            "trace_length_mode": "fixed_within_batch",
            "trace_samples": 2300,
            "trace_lengths_ref": None,
            "file_pattern": "x{ix}y{iy}.txt",
        },
        "scan_geometry": {
            "layout": "regular_grid",
            "grid_shape": [550, 550],
            "scan_step_x_mm": 0.2,
            "scan_step_y_mm": 0.2,
            "scan_coordinates_ref": None,
        },
        "time_reference": {
            "front_surface_material": "aluminum_alloy",
            "batch_reference_samples": None,
            "batch_reference_s": None,
        },
        "structure": {
            "K": 2,
            "K_source": "derived_from_layer_stack",
            "layers": [
                {"layer_order": 1, "material": "aluminum_alloy", "thickness_mm": 3.0},
                {"layer_order": 2, "material": "aluminum_alloy", "thickness_mm": 3.0},
                {"layer_order": 3, "material": "aluminum_alloy", "thickness_mm": 3.0},
            ],
            "total_thickness_mm": 9.0,
            "sound_speed_mps": None,
            "interfaces": [
                {
                    "interface_order": 1,
                    "nominal_depth_mm": 3.0,
                },
                {
                    "interface_order": 2,
                    "nominal_depth_mm": 6.0,
                },
            ],
        },
        "labels": {
            "embedded_defect_manifest_ref": None,
            "good_region_ref": None,
            "uncertain_boundary_ref": None,
        },
        "preprocessing": {
            "preserve_phase": True,
            "preserve_delay": True,
            "preserve_relative_amplitude": True,
        },
        "provenance": {
            "source_ref": "data/raw/a3/batch_001",
            "created_at": "2026-07-24",
            "created_by": "contract_test",
        },
    }


def test_load_yaml_mapping_and_validate_registered_manifest(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    schema_path.write_text(yaml.safe_dump(_schema(), sort_keys=False), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")

    loaded = load_and_validate_manifest(
        manifest_path,
        schema_path,
        stage="before_label_free_smoke",
    )

    assert loaded["structure"]["K"] == 2
    assert loaded["acquisition"]["probe_code"] is None


def test_load_yaml_mapping_rejects_non_mapping_root(tmp_path: Path) -> None:
    source = tmp_path / "list.yaml"
    source.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="root must be a mapping"):
        load_yaml_mapping(source)


def test_nested_excitation_fields_are_validated() -> None:
    manifest = _manifest()
    validate_manifest(manifest, _schema(), stage="before_label_free_smoke")

    manifest["acquisition"]["excitation"]["window_type"] = "unregistered_window"
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_label_free_smoke")

    assert any(
        issue.path == "acquisition.excitation.window_type"
        and "must be one of" in issue.message
        for issue in exc_info.value.issues
    )


def test_null_batch_value_is_stage_gated() -> None:
    manifest = _manifest()
    manifest["acquisition"]["sampling_rate_hz"] = None

    validate_manifest(manifest, _schema(), stage="schema_only")
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_label_free_smoke")

    assert any(
        issue.path == "acquisition.sampling_rate_hz"
        and "must be non-null" in issue.message
        for issue in exc_info.value.issues
    )


def test_validator_rejects_absolute_source_and_inconsistent_layer_stack() -> None:
    manifest = _manifest()
    manifest["provenance"]["source_ref"] = r"C:\private\raw"
    manifest["structure"]["K"] = 3

    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_label_free_smoke")

    issues = {(issue.path, issue.message) for issue in exc_info.value.issues}
    assert any(path == "provenance.source_ref" and "relative" in message for path, message in issues)
    assert any(path == "structure.interfaces" and "length must equal" in message for path, message in issues)
    assert any(path == "structure.K" and "len(layers) - 1" in message for path, message in issues)


def test_network_stage_requires_batch_reference_speed_and_shared_window() -> None:
    manifest = _manifest()
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_network_api_derivation")
    assert any("batch_reference" in issue.message for issue in exc_info.value.issues)
    assert any("sound speed" in issue.message for issue in exc_info.value.issues)
    assert any("window_left" in issue.message for issue in exc_info.value.issues)

    manifest["time_reference"]["batch_reference_samples"] = 10.0
    manifest["structure"]["sound_speed_mps"] = 5_700.0
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 50.0
    manifest["acquisition"]["window_profile"]["window_right_samples"] = 50.0
    validate_manifest(manifest, _schema(), stage="before_network_api_derivation")

    manifest["structure"]["interfaces"][0]["nominal_tof_samples"] = 600.0
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_network_api_derivation")
    assert any("override is forbidden" in issue.message for issue in exc_info.value.issues)


def test_network_stage_requires_per_batch_total_thickness_as_zmax() -> None:
    manifest = _manifest()
    manifest["structure"]["total_thickness_mm"] = None
    manifest["time_reference"]["batch_reference_samples"] = 10.0
    manifest["structure"]["sound_speed_mps"] = 5_700.0
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 50.0
    manifest["acquisition"]["window_profile"]["window_right_samples"] = 50.0

    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_network_api_derivation")
    assert any(
        issue.path == "structure.total_thickness_mm"
        and "must be non-null" in issue.message
        for issue in exc_info.value.issues
    )


def test_supervised_stage_requires_label_references() -> None:
    manifest = _manifest()
    manifest["time_reference"]["batch_reference_samples"] = 10.0
    manifest["structure"]["sound_speed_mps"] = 5_700.0
    manifest["acquisition"]["window_profile"]["window_left_samples"] = 50.0
    manifest["acquisition"]["window_profile"]["window_right_samples"] = 50.0

    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest, _schema(), stage="before_supervised_training")
    assert {
        issue.path
        for issue in exc_info.value.issues
        if "must be non-null" in issue.message
    } == {
        "labels.embedded_defect_manifest_ref",
        "labels.good_region_ref",
        "labels.uncertain_boundary_ref",
    }

    manifest["labels"] = {
        "embedded_defect_manifest_ref": "local/defect_map.yaml",
        "good_region_ref": "local/good_regions.yaml",
        "uncertain_boundary_ref": "local/uncertain_mask.yaml",
    }
    validate_manifest(manifest, _schema(), stage="before_supervised_training")
