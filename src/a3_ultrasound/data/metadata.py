"""Load and validate A3 acquisition manifests.

The validator consumes the project-owned ``metadata_schema.yaml`` format. It
does not read RF traces or derive network tensors; those are later milestones.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


STAGE_ORDER = (
    "schema_only",
    "before_label_free_smoke",
    "before_network_api_derivation",
    "before_supervised_training",
)

_AUXILIARY_TOP_LEVEL_KEYS = {
    "manifest_version",
    "registration_status",
    "usage_gate",
    "notes",
}
_OPAQUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SENSITIVE_KEY_RE = re.compile(r"(?:api[_-]?key|password|passwd|token|secret)", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic manifest validation failure."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ManifestLoadError(ValueError):
    """Raised when a schema or manifest cannot be loaded as a YAML mapping."""


class ManifestValidationError(ValueError):
    """Raised when one or more manifest contract checks fail."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Manifest validation failed with {len(self.issues)} issue(s):\n{details}")


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML file and require a mapping at its root."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestLoadError(f"Cannot read YAML file: {source}") from exc
    except yaml.YAMLError as exc:
        raise ManifestLoadError(f"Invalid YAML syntax: {source}") from exc

    if not isinstance(payload, Mapping):
        raise ManifestLoadError(f"YAML root must be a mapping: {source}")
    return dict(payload)


def load_and_validate_manifest(
    manifest_path: str | Path,
    schema_path: str | Path,
    *,
    stage: str = "schema_only",
) -> dict[str, Any]:
    """Load a manifest and schema, validate them, and return the manifest."""

    manifest = load_yaml_mapping(manifest_path)
    schema = load_yaml_mapping(schema_path)
    validate_manifest(manifest, schema, stage=stage)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    stage: str = "schema_only",
) -> None:
    """Validate one acquisition manifest against the custom A3 schema."""

    issues: list[ValidationIssue] = []
    if stage not in STAGE_ORDER:
        issues.append(ValidationIssue("stage", f"unknown validation stage {stage!r}"))
        raise ManifestValidationError(issues)

    expected_schema_version = schema.get("schema_version")
    if manifest.get("schema_version") != expected_schema_version:
        issues.append(
            ValidationIssue(
                "schema_version",
                f"expected {expected_schema_version!r}, got {manifest.get('schema_version')!r}",
            )
        )

    field_groups = schema.get("field_groups")
    if not isinstance(field_groups, Mapping):
        issues.append(ValidationIssue("schema.field_groups", "must be a mapping"))
    else:
        allowed_top_level = {"schema_version", *field_groups.keys(), *_AUXILIARY_TOP_LEVEL_KEYS}
        for key in manifest.keys() - allowed_top_level:
            issues.append(ValidationIssue(str(key), "unknown top-level manifest field"))

        for group_name, group_schema in field_groups.items():
            group_value = manifest.get(group_name)
            if not isinstance(group_schema, Mapping):
                issues.append(ValidationIssue(f"schema.field_groups.{group_name}", "must be a mapping"))
                continue
            if not isinstance(group_value, Mapping):
                issues.append(ValidationIssue(str(group_name), "required group must be a mapping"))
                continue
            _validate_object(
                group_value,
                group_schema,
                path=str(group_name),
                issues=issues,
            )

    _validate_stage_requirements(manifest, schema, stage, issues)
    _validate_project_contract(manifest, schema, issues)
    _validate_layout_contract(manifest, issues)
    _validate_structure_contract(manifest, issues)
    _validate_privacy_contract(manifest, schema, issues)

    if issues:
        raise ManifestValidationError(issues)


def _validate_object(
    value: Mapping[str, Any],
    object_schema: Mapping[str, Any],
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    schema_fields = set(object_schema)
    for key in value.keys() - schema_fields:
        issues.append(ValidationIssue(f"{path}.{key}", "unknown field"))

    for field_name, field_schema in object_schema.items():
        field_path = f"{path}.{field_name}"
        if not isinstance(field_schema, Mapping):
            issues.append(ValidationIssue(field_path, "field schema must be a mapping"))
            continue

        if field_name not in value:
            if field_schema.get("required") is True:
                issues.append(ValidationIssue(field_path, "required field is missing"))
            continue

        field_value = value[field_name]
        if field_value is None:
            nullable = (
                field_schema.get("nullable") is True
                or field_schema.get("nullable_until_batch_registration") is True
            )
            if not nullable:
                issues.append(ValidationIssue(field_path, "null is not allowed"))
            continue

        _validate_field_value(field_value, field_schema, field_path, issues)

    for field_name, field_schema in object_schema.items():
        if not isinstance(field_schema, Mapping):
            continue
        peer_name = field_schema.get("length_must_equal_field")
        field_value = value.get(field_name)
        if peer_name is None or not isinstance(field_value, list):
            continue
        peer_value = value.get(peer_name)
        if isinstance(peer_value, int) and len(field_value) != peer_value:
            issues.append(
                ValidationIssue(
                    f"{path}.{field_name}",
                    f"length must equal {path}.{peer_name} ({peer_value})",
                )
            )


def _validate_field_value(
    value: Any,
    field_schema: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    expected_type = field_schema.get("type")
    if expected_type is not None and not _matches_type(value, str(expected_type)):
        issues.append(ValidationIssue(path, f"expected type {expected_type}"))
        return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            issues.append(ValidationIssue(path, "numeric value must be finite"))
            return
        minimum = field_schema.get("minimum")
        if minimum is not None and value < minimum:
            issues.append(ValidationIssue(path, f"must be >= {minimum}"))
        minimum_exclusive = field_schema.get("minimum_exclusive")
        if minimum_exclusive is not None and value <= minimum_exclusive:
            issues.append(ValidationIssue(path, f"must be > {minimum_exclusive}"))

    if "const" in field_schema and value != field_schema["const"]:
        issues.append(ValidationIssue(path, f"must equal {field_schema['const']!r}"))

    allowed = field_schema.get("allowed")
    if allowed is not None and value not in allowed:
        issues.append(ValidationIssue(path, f"must be one of {list(allowed)!r}"))

    allowed_when_filled = field_schema.get("allowed_when_filled")
    if allowed_when_filled is not None and value not in allowed_when_filled:
        issues.append(ValidationIssue(path, f"must be one of {list(allowed_when_filled)!r}"))

    if isinstance(value, list):
        expected_length = field_schema.get("length")
        if expected_length is not None and len(value) != expected_length:
            issues.append(ValidationIssue(path, f"must contain exactly {expected_length} items"))

        item_type = field_schema.get("item_type")
        if item_type is not None:
            for index, item in enumerate(value):
                if not _matches_type(item, str(item_type)):
                    issues.append(ValidationIssue(f"{path}[{index}]", f"expected type {item_type}"))

        item_schema = field_schema.get("item_schema")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    issues.append(ValidationIssue(f"{path}[{index}]", "expected an object"))
                    continue
                _validate_object(item, item_schema, path=f"{path}[{index}]", issues=issues)

        item_rule = field_schema.get("item_rule")
        if item_rule is not None:
            for index, item in enumerate(value):
                _validate_rule(item, str(item_rule), f"{path}[{index}]", issues)

    if isinstance(value, Mapping):
        object_schema = field_schema.get("object_schema")
        if isinstance(object_schema, Mapping):
            _validate_object(value, object_schema, path=path, issues=issues)

    field_format = field_schema.get("format")
    if field_format == "iso8601" and not _is_iso8601(value):
        issues.append(ValidationIssue(path, "must be an ISO-8601 date or datetime"))

    rule = field_schema.get("rule")
    if rule is not None:
        _validate_rule(value, str(rule), path, issues)


def _validate_rule(value: Any, rule: str, path: str, issues: list[ValidationIssue]) -> None:
    if rule in {"opaque_id", "opaque_code", "non_personal_process_identifier"}:
        if not isinstance(value, str) or _OPAQUE_RE.fullmatch(value) is None:
            issues.append(ValidationIssue(path, f"must satisfy {rule}"))
    elif rule == "relative_or_opaque_reference":
        if not isinstance(value, str) or not value or _is_absolute_reference(value):
            issues.append(ValidationIssue(path, "must be a relative or opaque reference"))


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, Mapping)
    return False


def _validate_stage_requirements(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    stage: str,
    issues: list[ValidationIssue],
) -> None:
    requirements = schema.get("stage_requirements")
    if not isinstance(requirements, Mapping):
        issues.append(ValidationIssue("schema.stage_requirements", "must be a mapping"))
        return

    stage_index = STAGE_ORDER.index(stage)
    for active_stage in STAGE_ORDER[1 : stage_index + 1]:
        stage_contract = requirements.get(active_stage, {})
        if not isinstance(stage_contract, Mapping):
            issues.append(ValidationIssue(f"schema.stage_requirements.{active_stage}", "must be a mapping"))
            continue
        for dotted_path in stage_contract.get("require_non_null", []):
            if _get_dotted(manifest, dotted_path) is None:
                issues.append(ValidationIssue(dotted_path, f"must be non-null at stage {stage}"))

    if stage_index >= STAGE_ORDER.index("before_network_api_derivation"):
        acquisition = manifest.get("acquisition")
        if isinstance(acquisition, Mapping):
            if acquisition.get("acquisition_profile_id") is None:
                issues.append(
                    ValidationIssue(
                        "acquisition.acquisition_profile_id",
                        "required for network API",
                    )
                )
            window_profile = acquisition.get("window_profile")
            if not isinstance(window_profile, Mapping):
                issues.append(
                    ValidationIssue(
                        "acquisition.window_profile",
                        "shared acquisition-profile window is required for network API",
                    )
                )
            else:
                if window_profile.get("profile_id") is None:
                    issues.append(
                        ValidationIssue(
                            "acquisition.window_profile.profile_id",
                            "required for network API",
                        )
                    )
                for sample_key, second_key in (
                    ("window_left_samples", "window_left_s"),
                    ("window_right_samples", "window_right_s"),
                ):
                    if (
                        window_profile.get(sample_key) is None
                        and window_profile.get(second_key) is None
                    ):
                        issues.append(
                            ValidationIssue(
                                "acquisition.window_profile",
                                f"one of {sample_key}/{second_key} is required",
                            )
                        )

        time_reference = manifest.get("time_reference")
        if isinstance(time_reference, Mapping) and (
            time_reference.get("batch_reference_samples") is None
            and time_reference.get("batch_reference_s") is None
        ):
            issues.append(
                ValidationIssue(
                    "time_reference",
                    "one batch_reference_samples/batch_reference_s value is required",
                )
            )

        structure = manifest.get("structure")
        if isinstance(structure, Mapping):
            layers = structure.get("layers")
            layer_speeds = (
                [
                    layer.get("sound_speed_mps")
                    for layer in layers
                    if isinstance(layer, Mapping)
                ]
                if isinstance(layers, list)
                else []
            )
            if structure.get("sound_speed_mps") is None and (
                not layer_speeds or any(speed is None for speed in layer_speeds)
            ):
                issues.append(
                    ValidationIssue(
                        "structure.sound_speed_mps",
                        "one batch sound speed or a sound speed for every layer is required",
                    )
                )

        interfaces = _get_dotted(manifest, "structure.interfaces")
        if isinstance(interfaces, list):
            for index, interface in enumerate(interfaces):
                if not isinstance(interface, Mapping):
                    continue
                prefix = f"structure.interfaces[{index}]"
                if interface.get("nominal_depth_mm") is None:
                    issues.append(ValidationIssue(f"{prefix}.nominal_depth_mm", "required for network API"))
                for forbidden_name in (
                    "nominal_tof_samples",
                    "nominal_tof_s",
                    "window_left_samples",
                    "window_right_samples",
                ):
                    if forbidden_name in interface:
                        issues.append(
                            ValidationIssue(
                                f"{prefix}.{forbidden_name}",
                                "per-interface TOF/window override is forbidden",
                            )
                        )


def _validate_project_contract(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    contract = schema.get("project_contract")
    if not isinstance(contract, Mapping):
        issues.append(ValidationIssue("schema.project_contract", "must be a mapping"))
        return

    expected_sampling_rate = _get_dotted(contract, "sampling_rate_hz.expected_value")
    actual_sampling_rate = _get_dotted(manifest, "acquisition.sampling_rate_hz")
    if expected_sampling_rate is not None and actual_sampling_rate not in {None, expected_sampling_rate}:
        issues.append(
            ValidationIssue(
                "acquisition.sampling_rate_hz",
                f"must equal project value {expected_sampling_rate}",
            )
        )


def _validate_layout_contract(
    manifest: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    raw_layout = manifest.get("raw_layout")
    if isinstance(raw_layout, Mapping):
        mode = raw_layout.get("trace_length_mode")
        if mode == "fixed_within_batch" and raw_layout.get("trace_samples") is None:
            issues.append(ValidationIssue("raw_layout.trace_samples", "required for fixed-length traces"))
        if mode == "variable_within_batch" and raw_layout.get("trace_lengths_ref") is None:
            issues.append(
                ValidationIssue("raw_layout.trace_lengths_ref", "required for variable-length traces")
            )

    geometry = manifest.get("scan_geometry")
    if isinstance(geometry, Mapping):
        layout = geometry.get("layout")
        if layout == "regular_grid":
            for field_name in ("grid_shape", "scan_step_x_mm", "scan_step_y_mm"):
                if geometry.get(field_name) is None:
                    issues.append(
                        ValidationIssue(f"scan_geometry.{field_name}", "required for regular_grid")
                    )
        if layout == "irregular_points" and geometry.get("scan_coordinates_ref") is None:
            issues.append(
                ValidationIssue(
                    "scan_geometry.scan_coordinates_ref",
                    "required for irregular_points",
                )
            )


def _validate_structure_contract(
    manifest: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    structure = manifest.get("structure")
    if not isinstance(structure, Mapping):
        return

    interface_count = structure.get("K")
    interfaces = structure.get("interfaces")
    if isinstance(interfaces, list):
        expected_orders = list(range(1, len(interfaces) + 1))
        orders = [
            interface.get("interface_order") if isinstance(interface, Mapping) else None
            for interface in interfaces
        ]
        if orders != expected_orders:
            issues.append(
                ValidationIssue("structure.interfaces", f"interface_order must be {expected_orders}")
            )

    layers = structure.get("layers")
    if not isinstance(layers, list) or not layers:
        return

    expected_layer_orders = list(range(1, len(layers) + 1))
    layer_orders = [layer.get("layer_order") if isinstance(layer, Mapping) else None for layer in layers]
    if layer_orders != expected_layer_orders:
        issues.append(ValidationIssue("structure.layers", f"layer_order must be {expected_layer_orders}"))

    thicknesses = [
        layer.get("thickness_mm") for layer in layers if isinstance(layer, Mapping)
    ]
    if len(thicknesses) != len(layers) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) for value in thicknesses
    ):
        return

    total_thickness = structure.get("total_thickness_mm")
    thickness_sum = float(sum(thicknesses))
    if isinstance(total_thickness, (int, float)) and not math.isclose(
        float(total_thickness), thickness_sum, rel_tol=1e-9, abs_tol=1e-9
    ):
        issues.append(
            ValidationIssue(
                "structure.total_thickness_mm",
                f"must equal sum of layer thicknesses ({thickness_sum})",
            )
        )

    if structure.get("K_source") == "derived_from_layer_stack":
        derived_k = len(layers) - 1
        if interface_count != derived_k:
            issues.append(
                ValidationIssue("structure.K", f"must equal len(layers) - 1 ({derived_k})")
            )
        if isinstance(interfaces, list) and len(interfaces) == derived_k:
            cumulative_depths: list[float] = []
            depth = 0.0
            for thickness in thicknesses[:-1]:
                depth += float(thickness)
                cumulative_depths.append(depth)
            for index, (interface, expected_depth) in enumerate(zip(interfaces, cumulative_depths)):
                if not isinstance(interface, Mapping):
                    continue
                actual_depth = interface.get("nominal_depth_mm")
                if actual_depth is not None and not math.isclose(
                    float(actual_depth), expected_depth, rel_tol=1e-9, abs_tol=1e-9
                ):
                    issues.append(
                        ValidationIssue(
                            f"structure.interfaces[{index}].nominal_depth_mm",
                            f"must equal cumulative layer depth {expected_depth}",
                        )
                    )


def _validate_privacy_contract(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    policy = schema.get("privacy_policy")
    if not isinstance(policy, Mapping):
        issues.append(ValidationIssue("schema.privacy_policy", "must be a mapping"))
        return

    for path, value in _walk_values(manifest):
        if policy.get("forbid_absolute_paths") is True and isinstance(value, str):
            if _is_absolute_reference(value):
                issues.append(ValidationIssue(path, "absolute paths are forbidden"))
        if policy.get("forbid_secrets") is True:
            key = path.rsplit(".", maxsplit=1)[-1]
            if _SENSITIVE_KEY_RE.search(key) and value not in {None, ""}:
                issues.append(ValidationIssue(path, "secret-like fields must not be versioned"))


def _walk_values(value: Any, path: str = "") -> Sequence[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.extend(_walk_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_walk_values(child, f"{path}[{index}]"))
    else:
        result.append((path, value))
    return result


def _is_absolute_reference(value: str) -> bool:
    normalized = value.strip()
    return (
        bool(_WINDOWS_ABSOLUTE_RE.match(normalized))
        or normalized.startswith(("/", "\\\\", "file://"))
    )


def _is_iso8601(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False


def _get_dotted(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for deterministic contract checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGE_ORDER, default="schema_only")
    args = parser.parse_args(argv)

    try:
        load_and_validate_manifest(args.manifest, args.schema, stage=args.stage)
    except (ManifestLoadError, ManifestValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"VALID stage={args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
