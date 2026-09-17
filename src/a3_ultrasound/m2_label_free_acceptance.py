"""Deterministic acceptance for the non-supervised portion of A3 M2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from a3_ultrasound.data import (
    build_label_free_real_rf_batch,
    load_label_free_network_manifest,
)


class M2LabelFreeAcceptanceError(ValueError):
    """Raised when an M2 label-free evidence artifact is inconsistent."""


ARTIFACT_SPECS: Mapping[str, Mapping[str, Any]] = {
    "m2_01": {
        "schema_version": "a3-m2-01-real-rf-batch-contract-v1",
        "run_id": "M2-01",
        "evidence_class": "sampled_real_rf_label_free_batch_construction_only",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "performs_network_forward": False,
            "performs_training": False,
            "performs_performance_evaluation": False,
            "content_audit_is_full_scan": False,
        },
        "classification": "label_free_real_rf_engineering",
        "source_keys": {
            "real_rf.py", "collate.py", "build_m2_01_real_rf_batch.py",
            "current_acquisition_manifest.yaml", "metadata_schema.yaml",
        },
    },
    "m2_02": {
        "schema_version": "a3-m2-02-real-rf-forward-smoke-v1",
        "run_id": "M2-02",
        "evidence_class": "sampled_real_rf_label_free_forward_only",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "performs_network_forward": True,
            "performs_exact_enumeration": True,
            "performs_loss": False,
            "performs_backward": False,
            "performs_optimizer_step": False,
            "performs_training": False,
            "performs_model_selection": False,
            "performs_performance_evaluation": False,
        },
        "classification": "label_free_real_rf_engineering",
        "source_keys": {
            "real_rf.py", "backbone.py", "integrated.py",
            "m2_real_rf_forward_smoke.py", "current_acquisition_manifest.yaml",
            "metadata_schema.yaml",
        },
    },
    "m2_03": {
        "schema_version": "a3-m2-03-attention-support-diagnostic-v1",
        "run_id": "M2-03",
        "evidence_class": "sampled_real_rf_label_free_attention_support_diagnostic",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "serializes_raw_rf": False,
            "performs_network_forward": True,
            "performs_loss": False,
            "performs_backward": False,
            "performs_optimizer_step": False,
            "performs_training": False,
            "performs_model_selection": False,
            "performs_performance_evaluation": False,
        },
        "classification": "label_free_real_rf_engineering",
        "source_keys": {
            "m2_real_rf_forward_smoke.py", "m2_attention_support_diagnostic.py",
            "current_acquisition_manifest.yaml", "metadata_schema.yaml",
        },
    },
    "m2_04": {
        "schema_version": "a3-m2-04-gpu-profile-v1",
        "run_id": "M2-04",
        "evidence_class": "synthetic_runtime_profile_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
            "profiles_exact_enumeration": False,
        },
        "classification": "synthetic_runtime",
        "source_keys": {
            "integrated.py", "backbone.py", "losses.py",
            "m2_real_rf_forward_smoke.py", "m2_04_gpu_profile.py",
        },
    },
    "m2_05": {
        "schema_version": "a3-m2-05-safe-batch-search-v1",
        "run_id": "M2-05",
        "evidence_class": "synthetic_runtime_budget_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
        },
        "classification": "synthetic_runtime",
        "source_keys": {
            "integrated.py", "backbone.py", "losses.py",
            "m2_04_gpu_profile.py", "m2_05_safe_batch_search.py",
        },
    },
    "m2_06": {
        "schema_version": "a3-m2-06-runtime-acceptance-v1",
        "run_id": "M2-06",
        "evidence_class": "deterministic_label_free_runtime_acceptance_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "requires_label_schema": False,
            "requires_sensitive_mapping": False,
            "requires_split_manifest": False,
            "uses_synthetic_labels_in_source_profiles": True,
            "performs_model_selection": False,
        },
        "classification": "deterministic_runtime_acceptance",
        "source_keys": {"runtime_contract.py", "build_m2_06_runtime_acceptance.py"},
    },
    "m2_07": {
        "schema_version": "a3-m2-07-synthetic-tiny-overfit-v1",
        "run_id": "M2-07",
        "evidence_class": "synthetic_learning_path_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
            "runs_exact_enumeration": False,
        },
        "classification": "synthetic_learning_path",
        "source_keys": {
            "integrated.py", "losses.py", "runtime_contract.py",
            "m2_real_rf_forward_smoke.py", "m2_04_gpu_profile.py",
            "m2_07_synthetic_tiny_overfit.py",
        },
    },
    "m2_lf_01": {
        "schema_version": "a3-m2-label-free-real-rf-autograd-v2",
        "run_id": "M2-LF-01",
        "evidence_class": "sampled_real_rf_label_free_single_backward_diagnostic",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "reads_independent_actual_label_schema": False,
            "metadata_schema_contains_static_label_enum_for_manifest_validation": True,
            "reads_sensitive_mapping": False,
            "reads_split_manifest": False,
            "uses_target": False,
            "performs_network_forward": True,
            "performs_backward": True,
            "performs_optimizer_step": False,
            "performs_parameter_update": False,
            "performs_training": False,
            "performs_model_selection": False,
            "performs_performance_evaluation": False,
        },
        "classification": "label_free_real_rf_engineering",
        "source_keys": {
            "diagnostics.py", "real_rf.py", "backbone.py", "integrated.py",
            "m2_real_rf_forward_smoke.py",
            "m2_label_free_real_rf_backward_diagnostic.py",
            "current_acquisition_manifest.yaml", "metadata_schema.yaml",
        },
    },
}

AUDIT_INPUT_REF_KEYS = {
    "runtime_config",
    "m2_03_arrays",
    "m2_03_figure",
    "m2_04_log",
    "m2_05_log",
    "m2_06_log",
    "m2_06_failed_timing_log",
    "m2_07_log",
    "m2_lf_log",
    "pytest_junit",
    "pytest_log",
}

ACCEPTANCE_SOURCE_KEYS = {
    "m2_label_free_acceptance.py",
    "build_m2_label_free_acceptance.py",
    "test_m2_label_free_acceptance.py",
    "test_m2_cli_entrypoints.py",
    "test_real_rf.py",
    "package____init__.py",
    "package__data____init__.py",
    "package__data__metadata.py",
    "package__data__smoke.py",
    "package__data__splits.py",
    "package__data__time_prior.py",
    "package__inference____init__.py",
    "package__inference__enumerator.py",
    "package__inference__orders.py",
    "package__metrics____init__.py",
    "package__metrics__classification.py",
    "package__models____init__.py",
    "package__models__capacity.py",
    "package__models__indexed_readout.py",
    "package__models__pcc.py",
    "package__models__query.py",
    "package__models__responsibility_tokens.py",
    "package__review____init__.py",
    "package__review__sanitizer.py",
    "package__schema.py",
    "test__test_acceptance_contracts.py",
    "test__test_backbone.py",
    "test__test_collate.py",
    "test__test_external_review_sanitizer.py",
    "test__test_integrated_model.py",
    "test__test_label_free_autograd_probe.py",
    "test__test_losses.py",
    "test__test_metadata.py",
    "test__test_pcc_enumerator.py",
    "test__test_responsibility_tokens.py",
    "test__test_runtime_contract.py",
    "test__test_schema_and_readout.py",
    "test__test_smoke.py",
    "test__test_time_prior.py",
    "script__build_m0_02_metadata_acceptance.py",
    "script__build_m1_acceptance.py",
    "script__compute_capacity_match.py",
    "script__gpu_smoke.py",
    "script__m1_integrated_gpu_smoke.py",
    "project__pyproject.toml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M2LabelFreeAcceptanceError(f"{name} must be a mapping.")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise M2LabelFreeAcceptanceError(f"Cannot load config: {path.name}.") from exc
    return _mapping(value, "config")


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M2LabelFreeAcceptanceError(f"Cannot load {name}: {path.name}.") from exc
    return _mapping(value, name)


def _resolve_relative(base: Path, reference: Any, name: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise M2LabelFreeAcceptanceError(f"{name} must be a nonempty relative path.")
    candidate = Path(reference)
    if candidate.is_absolute():
        raise M2LabelFreeAcceptanceError(f"{name} must be relative.")
    path = (base / candidate).resolve()
    if not path.is_file():
        raise M2LabelFreeAcceptanceError(f"{name} does not resolve to a file.")
    return path


def _all_checks_pass(value: Any, name: str) -> None:
    checks = _mapping(value, name)
    if not checks or any(item is not True for item in checks.values()):
        raise M2LabelFreeAcceptanceError(f"Every {name} check must be true.")


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M2LabelFreeAcceptanceError(f"{name} must be numeric.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise M2LabelFreeAcceptanceError(f"{name} must be finite.")
    return resolved


def _sha256_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise M2LabelFreeAcceptanceError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        current = _mapping(current, ".".join(keys)).get(key)
    return current


def _contains_machine_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return re.search(r"(?:^|\s)[A-Za-z]:[\\/]", value) is not None
    if isinstance(value, Mapping):
        return any(_contains_machine_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_machine_absolute_path(item) for item in value)
    return False


def _input_fingerprint(value: Any, name: str) -> Mapping[str, Any]:
    fingerprint = _mapping(value, name)
    expected_keys = {
        "algorithm",
        "trace_count",
        "raw_trace_content_sha256",
        "normalized_batch_sha256",
        "contains_filenames_or_coordinates",
        "aggregate_sha256",
    }
    if set(fingerprint) != expected_keys:
        raise M2LabelFreeAcceptanceError(f"{name} fields do not match the contract.")
    raw_hashes = fingerprint.get("raw_trace_content_sha256")
    if (
        fingerprint.get("algorithm") != "sha256"
        or fingerprint.get("trace_count") != 5
        or not isinstance(raw_hashes, list)
        or len(raw_hashes) != 5
        or fingerprint.get("contains_filenames_or_coordinates") is not False
    ):
        raise M2LabelFreeAcceptanceError(f"{name} is not a five-trace private fingerprint.")
    for index, digest in enumerate(raw_hashes):
        _sha256_string(digest, f"{name}.raw_trace_content_sha256[{index}]")
    _sha256_string(fingerprint.get("normalized_batch_sha256"), f"{name}.normalized_batch_sha256")
    body = {key: fingerprint[key] for key in expected_keys if key != "aggregate_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_aggregate = hashlib.sha256(canonical).hexdigest()
    if fingerprint.get("aggregate_sha256") != expected_aggregate:
        raise M2LabelFreeAcceptanceError(f"{name}.aggregate_sha256 does not bind its fields.")
    return fingerprint


def _validate_frontend_contract(value: Any, name: str, entrypoint: str) -> None:
    frontend = _mapping(value, name)
    if _contains_machine_absolute_path(frontend):
        raise M2LabelFreeAcceptanceError(f"{name} contains an absolute path.")
    expected_keys = {
        "entrypoint",
        "required_parameters",
        "pycharm",
        "codex_cli",
        "contains_machine_absolute_paths",
        "codex_chat_state_required",
    }
    if set(frontend) != expected_keys or frontend.get("entrypoint") != entrypoint:
        raise M2LabelFreeAcceptanceError(f"{name} fields do not match the portable contract.")
    pycharm = _mapping(frontend.get("pycharm"), f"{name}.pycharm")
    codex = _mapping(frontend.get("codex_cli"), f"{name}.codex_cli")
    if dict(pycharm) != {
        "project": "a3-layerrecognition",
        "interpreter": "aislee-a3-layercongition",
        "same_cli_contract_available": True,
        "execution_recorded_by_this_run": False,
    }:
        raise M2LabelFreeAcceptanceError(f"{name}.pycharm overstates or changes the contract.")
    if dict(codex) != {
        "same_cli_contract_available": True,
        "execution_recorded_by_this_run": True,
    }:
        raise M2LabelFreeAcceptanceError(f"{name}.codex_cli execution record changed.")
    if frontend.get("contains_machine_absolute_paths") is not False:
        raise M2LabelFreeAcceptanceError(f"{name} must declare no machine absolute paths.")
    if frontend.get("codex_chat_state_required") is not False:
        raise M2LabelFreeAcceptanceError(f"{name} must not require Codex chat state.")


def _validate_common(name: str, payload: Mapping[str, Any]) -> None:
    spec = ARTIFACT_SPECS[name]
    for key in ("schema_version", "run_id", "evidence_class"):
        if payload.get(key) != spec[key]:
            raise M2LabelFreeAcceptanceError(f"{name}.{key} does not match the contract.")
    if payload.get("status") != "PASS":
        raise M2LabelFreeAcceptanceError(f"{name} is not a PASS artifact.")
    scope = _mapping(payload.get("scope"), f"{name}.scope")
    expected_scope = _mapping(spec["scope"], f"{name}.scope contract")
    if dict(scope) != dict(expected_scope):
        raise M2LabelFreeAcceptanceError(f"{name}.scope must match the exact scope contract.")
    _all_checks_pass(payload.get("checks"), f"{name}.checks")


def _validate_task_specific(
    name: str,
    payload: Mapping[str, Any],
    artifact_path: Path,
) -> None:
    if name == "m2_01":
        _input_fingerprint(payload.get("input_fingerprint"), "M2-01.input_fingerprint")
        if _nested(payload, "selection", "trace_count") != 5:
            raise M2LabelFreeAcceptanceError("M2-01 must contain exactly five traces.")
        tensor = _mapping(payload.get("tensor_contract"), "M2-01.tensor_contract")
        if _nested(tensor, "x_raw", "shape") != [5, 2300]:
            raise M2LabelFreeAcceptanceError("M2-01 x_raw shape changed.")
        if _nested(tensor, "interface_meta", "shape") != [5, 3, 5]:
            raise M2LabelFreeAcceptanceError("M2-01 interface_meta shape changed.")
        if tensor.get("y") is not None or tensor.get("label_mask") is not None:
            raise M2LabelFreeAcceptanceError("M2-01 must remain label-free.")
    elif name == "m2_02":
        _input_fingerprint(payload.get("input_fingerprint"), "M2-02.input_fingerprint")
        if _nested(payload, "resolved_config", "batch_size") != 5:
            raise M2LabelFreeAcceptanceError("M2-02 batch size must remain five.")
        if _nested(payload, "registered_contract", "labels_absent") is not True:
            raise M2LabelFreeAcceptanceError("M2-02 labels_absent must be true.")
        if _nested(payload, "support_audit", "valid_local_support_count") != 10:
            raise M2LabelFreeAcceptanceError("M2-02 must audit ten valid interfaces.")
        if _finite(
            _nested(payload, "output_audit", "exact_probability_max_abs_error"),
            "M2-02 exact probability error",
        ) >= 1e-5:
            raise M2LabelFreeAcceptanceError("M2-02 exact probability error is too high.")
    elif name == "m2_03":
        _input_fingerprint(payload.get("input_fingerprint"), "M2-03.input_fingerprint")
        if _nested(payload, "global_audit", "valid_interface_rows") != 10:
            raise M2LabelFreeAcceptanceError("M2-03 must audit ten interface rows.")
        artifacts = _mapping(payload.get("artifacts"), "M2-03.artifacts")
        if artifacts.get("arrays_contain_raw_rf") is not False:
            raise M2LabelFreeAcceptanceError("M2-03 arrays must not contain raw RF.")
        if artifacts.get("arrays_contain_scan_coordinates") is not False:
            raise M2LabelFreeAcceptanceError("M2-03 arrays must not contain coordinates.")
        for ref_key, hash_key in (("arrays", "arrays_sha256"), ("figure", "figure_sha256")):
            output = _resolve_relative(artifact_path.parent, artifacts.get(ref_key), f"M2-03 {ref_key}")
            if _sha256(output) != artifacts.get(hash_key):
                raise M2LabelFreeAcceptanceError(f"M2-03 {ref_key} hash mismatch.")
    elif name == "m2_06":
        boundary = _mapping(payload.get("integrity_boundary"), "M2-06.integrity_boundary")
        if boundary.get("real_label_gate_evaluated") is not False:
            raise M2LabelFreeAcceptanceError("M2-06 must not evaluate the real-label gate.")
        if boundary.get("numeric_search_space_frozen") is not False:
            raise M2LabelFreeAcceptanceError("M2-06 must not freeze numeric search space.")
    elif name == "m2_07":
        config = _mapping(payload.get("resolved_config"), "M2-07.resolved_config")
        data = _mapping(payload.get("data_contract"), "M2-07.data_contract")
        training = _mapping(payload.get("training"), "M2-07.training")
        if config.get("micro_batch_size") != 128 or config.get("amp_enabled") is not False:
            raise M2LabelFreeAcceptanceError("M2-07 runtime differs from the freeze.")
        if data.get("valid_interface_count") != 40:
            raise M2LabelFreeAcceptanceError("M2-07 valid synthetic interface count changed.")
        if data.get("positive_valid_labels") != 20 or data.get("negative_valid_labels") != 20:
            raise M2LabelFreeAcceptanceError("M2-07 synthetic classes are not balanced.")
        if _finite(training.get("loss_ratio"), "M2-07 loss ratio") > _finite(
            config.get("loss_ratio_threshold"), "M2-07 loss threshold"
        ):
            raise M2LabelFreeAcceptanceError("M2-07 loss ratio misses its threshold.")
        _validate_frontend_contract(
            payload.get("execution_frontend_contract"),
            "M2-07.execution_frontend_contract",
            "scripts/m2_07_synthetic_tiny_overfit.py",
        )
        artifact = _mapping(payload.get("artifact_contract"), "M2-07.artifact_contract")
        if artifact.get("companion_log_generated_by_runner") is not True:
            raise M2LabelFreeAcceptanceError("M2-07 companion log is not runner-generated.")
        log_path = _resolve_relative(
            artifact_path.parent,
            artifact.get("companion_log_name"),
            "M2-07 companion log",
        )
        if _sha256(log_path) != artifact.get("companion_log_sha256"):
            raise M2LabelFreeAcceptanceError("M2-07 companion log hash mismatch.")
    elif name == "m2_lf_01":
        _input_fingerprint(payload.get("input_fingerprint"), "M2-LF-01.input_fingerprint")
        probe = _mapping(payload.get("probe_contract"), "M2-LF-01.probe_contract")
        if any(
            probe.get(key) is not False
            for key in ("is_training_loss", "is_model_selection_metric", "has_physical_interpretation")
        ):
            raise M2LabelFreeAcceptanceError("M2-LF-01 probe boundary changed.")
        gradients = _mapping(payload.get("gradient_audit"), "M2-LF-01.gradient_audit")
        if gradients.get("model_state_sha256_before") != gradients.get("model_state_sha256_after"):
            raise M2LabelFreeAcceptanceError("M2-LF-01 changed model parameters.")
        groups = _mapping(gradients.get("parameter_groups"), "M2-LF-01.parameter_groups")
        if len(groups) != 9 or any(
            _mapping(group, "M2-LF-01 parameter group").get("nonzero_gradient_present") is not True
            or _mapping(group, "M2-LF-01 parameter group").get("all_gradients_finite") is not True
            for group in groups.values()
        ):
            raise M2LabelFreeAcceptanceError("M2-LF-01 parameter-group gradients are incomplete.")
        environment = _mapping(payload.get("environment"), "M2-LF-01.environment")
        if environment.get("registered_environment_matches") is not True:
            raise M2LabelFreeAcceptanceError("M2-LF-01 did not use the registered environment.")
        if environment.get("pythonpath_environment_variable_present") is not False:
            raise M2LabelFreeAcceptanceError("M2-LF-01 must not depend on PYTHONPATH.")
        _validate_frontend_contract(
            payload.get("execution_frontend_contract"),
            "M2-LF-01.execution_frontend_contract",
            "scripts/m2_label_free_real_rf_backward_diagnostic.py",
        )
        artifact = _mapping(payload.get("artifact_contract"), "M2-LF-01.artifact_contract")
        if artifact.get("companion_log_generated_by_runner") is not True:
            raise M2LabelFreeAcceptanceError("M2-LF-01 companion log is not runner-generated.")
        log_path = _resolve_relative(
            artifact_path.parent,
            artifact.get("companion_log_name"),
            "M2-LF-01 companion log",
        )
        if _sha256(log_path) != artifact.get("companion_log_sha256"):
            raise M2LabelFreeAcceptanceError("M2-LF-01 companion log hash mismatch.")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "a3-m2-label-free-engineering-acceptance-config-v2":
        raise M2LabelFreeAcceptanceError("Unsupported acceptance config schema.")
    if config.get("run_id") != "M2-LF-GATE":
        raise M2LabelFreeAcceptanceError("Acceptance config run_id must be M2-LF-GATE.")
    if config.get("status") != "engineering_subgate_only_not_formal_m2_gate":
        raise M2LabelFreeAcceptanceError("Acceptance must remain an engineering subgate.")
    boundary = _mapping(config.get("boundary"), "boundary")
    required = {
        "actual_label_tasks_excluded": True,
        "m2_08_status": "EXCLUDED_BY_CURRENT_USER_SCOPE",
        "m2_09_status": "EXCLUDED_BY_CURRENT_USER_SCOPE",
        "numeric_search_space_frozen": False,
        "formal_m2_10_status": "GATED",
        "formal_m2_gate_status": "GATED",
        "authorizes_supervised_training": False,
        "supports_performance_or_mechanism_claims": False,
    }
    if dict(boundary) != required:
        raise M2LabelFreeAcceptanceError("boundary must match the exact nonformal contract.")
    pytest_contract = _mapping(config.get("pytest_contract"), "pytest_contract")
    if set(pytest_contract) != {
        "tests",
        "failures",
        "errors",
        "skipped",
        "junit_sha256",
        "log_sha256",
    }:
        raise M2LabelFreeAcceptanceError("pytest_contract fields are incomplete.")
    expected_tests = pytest_contract.get("tests")
    if not isinstance(expected_tests, int) or isinstance(expected_tests, bool) or expected_tests <= 0:
        raise M2LabelFreeAcceptanceError("pytest_contract.tests must be a positive integer.")
    for key in ("failures", "errors", "skipped"):
        if pytest_contract.get(key) != 0:
            raise M2LabelFreeAcceptanceError(f"pytest_contract.{key} must be zero.")
    _sha256_string(pytest_contract.get("junit_sha256"), "pytest_contract.junit_sha256")
    _sha256_string(pytest_contract.get("log_sha256"), "pytest_contract.log_sha256")


def _resolve_sources(
    config: Mapping[str, Any],
    config_path: Path,
    implementation_root: Path,
) -> dict[str, Path]:
    bindings = _mapping(config.get("source_bindings"), "source_bindings")
    required_source_keys = {
        key for spec in ARTIFACT_SPECS.values() for key in spec["source_keys"]
    } | ACCEPTANCE_SOURCE_KEYS
    if set(bindings) != required_source_keys:
        raise M2LabelFreeAcceptanceError(
            "source_bindings must cover the exact evidence and acceptance dependency set."
        )
    resolved: dict[str, Path] = {}
    for key, raw_binding in bindings.items():
        binding = _mapping(raw_binding, f"source_bindings.{key}")
        root = binding.get("root")
        if root == "implementation":
            base = implementation_root
        elif root == "config":
            base = config_path.parent
        else:
            raise M2LabelFreeAcceptanceError(f"source_bindings.{key}.root is invalid.")
        resolved[key] = _resolve_relative(base, binding.get("path"), f"source {key}")
    return resolved


def _resolve_audit_inputs(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    references = _mapping(config.get("audit_input_refs"), "audit_input_refs")
    if set(references) != AUDIT_INPUT_REF_KEYS:
        raise M2LabelFreeAcceptanceError("audit_input_refs must cover the exact side-evidence set.")
    return {
        key: _resolve_relative(config_path.parent, reference, f"audit_input_refs.{key}")
        for key, reference in references.items()
    }


def _current_real_rf_fingerprint(sources: Mapping[str, Path]) -> Mapping[str, Any]:
    manifest_path = sources["current_acquisition_manifest.yaml"]
    metadata_schema_path = sources["metadata_schema.yaml"]
    manifest = load_label_free_network_manifest(manifest_path, metadata_schema_path)
    provenance = _mapping(manifest.get("provenance"), "manifest.provenance")
    source_ref = provenance.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise M2LabelFreeAcceptanceError("manifest provenance.source_ref must be relative.")
    relative = Path(source_ref)
    if relative.is_absolute():
        raise M2LabelFreeAcceptanceError("manifest provenance.source_ref must be relative.")
    try:
        workspace_root = manifest_path.parents[3]
    except IndexError as exc:
        raise M2LabelFreeAcceptanceError("manifest path is outside the expected workspace layout.") from exc
    raw_root = (workspace_root / relative).resolve()
    if not raw_root.is_relative_to(workspace_root) or not raw_root.is_dir():
        raise M2LabelFreeAcceptanceError("manifest provenance.source_ref is not an available raw root.")
    loaded = build_label_free_real_rf_batch(raw_root, manifest)
    return _input_fingerprint(loaded.input_fingerprint(), "current real-RF fingerprint")


def _parse_single_json_log(path: Path, run_id: str) -> Mapping[str, Any]:
    nonempty = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(nonempty) != 1:
        raise M2LabelFreeAcceptanceError(f"{run_id} log must contain one JSON summary line.")
    try:
        payload = json.loads(nonempty[0])
    except json.JSONDecodeError as exc:
        raise M2LabelFreeAcceptanceError(f"{run_id} log is not valid JSON.") from exc
    summary = _mapping(payload, f"{run_id} log")
    if summary.get("run_id") != run_id or summary.get("status") != "PASS":
        raise M2LabelFreeAcceptanceError(f"{run_id} log identity/status mismatch.")
    return summary


def build_m2_label_free_acceptance(
    config_path: Path,
    implementation_root: Path,
) -> dict[str, Any]:
    """Validate all current non-supervised M2 evidence and its audit trace."""

    config_path = config_path.resolve()
    implementation_root = implementation_root.resolve()
    if not implementation_root.is_dir():
        raise M2LabelFreeAcceptanceError("implementation_root must be a directory.")
    config = _load_yaml(config_path)
    _validate_config(config)
    evidence_refs = _mapping(config.get("evidence_refs"), "evidence_refs")
    if set(evidence_refs) != set(ARTIFACT_SPECS):
        raise M2LabelFreeAcceptanceError("evidence_refs must cover exactly the M2 label-free set.")
    sources = _resolve_sources(config, config_path, implementation_root)
    audit_inputs = _resolve_audit_inputs(config, config_path)

    artifacts: dict[str, Mapping[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    verified_source_hashes: dict[str, str] = {}
    task_summary: dict[str, Any] = {}
    for name, reference in evidence_refs.items():
        path = _resolve_relative(config_path.parent, reference, f"evidence_refs.{name}")
        payload = _load_json(path, name)
        _validate_common(name, payload)
        _validate_task_specific(name, payload, path)
        source_hashes = _mapping(payload.get("source_sha256"), f"{name}.source_sha256")
        if not source_hashes:
            raise M2LabelFreeAcceptanceError(f"{name}.source_sha256 must not be empty.")
        if set(source_hashes) != set(ARTIFACT_SPECS[name]["source_keys"]):
            raise M2LabelFreeAcceptanceError(
                f"{name}.source_sha256 does not cover the required source dependency set."
            )
        for key, expected in source_hashes.items():
            source = sources.get(key)
            if source is None:
                raise M2LabelFreeAcceptanceError(f"No source binding for {key}.")
            actual = _sha256(source)
            if actual != expected:
                raise M2LabelFreeAcceptanceError(f"{name} source hash mismatch for {key}.")
            verified_source_hashes[key] = actual
        artifacts[name] = payload
        artifact_paths[name] = path
        artifact_hashes[path.name] = _sha256(path)
        task_summary[name] = {
            "run_id": payload["run_id"],
            "classification": ARTIFACT_SPECS[name]["classification"],
            "status": "PASS",
        }

    current_fingerprint = _current_real_rf_fingerprint(sources)
    for name in ("m2_01", "m2_02", "m2_03", "m2_lf_01"):
        if artifacts[name].get("input_fingerprint") != current_fingerprint:
            raise M2LabelFreeAcceptanceError(
                f"{name} does not bind the current five-trace real-RF content."
            )

    runtime_path = audit_inputs["runtime_config"]
    m2_06_inputs = _mapping(artifacts["m2_06"].get("input_sha256"), "M2-06.input_sha256")
    expected_inputs = {
        runtime_path.name: _sha256(runtime_path),
        artifact_paths["m2_04"].name: artifact_hashes[artifact_paths["m2_04"].name],
        artifact_paths["m2_05"].name: artifact_hashes[artifact_paths["m2_05"].name],
    }
    if dict(m2_06_inputs) != expected_inputs:
        raise M2LabelFreeAcceptanceError("M2-06 input hashes do not bind current runtime evidence.")
    m2_07_runtime = _mapping(
        artifacts["m2_07"].get("runtime_contract"),
        "M2-07.runtime_contract",
    )
    if m2_07_runtime.get("config_name") != runtime_path.name:
        raise M2LabelFreeAcceptanceError("M2-07 runtime config name mismatch.")
    if m2_07_runtime.get("config_sha256") != _sha256(runtime_path):
        raise M2LabelFreeAcceptanceError("M2-07 runtime config hash mismatch.")
    if m2_07_runtime.get("resolved_runtime") != artifacts["m2_06"].get(
        "resolved_runtime"
    ):
        raise M2LabelFreeAcceptanceError("M2-07 resolved runtime differs from M2-06.")
    if m2_07_runtime.get("runtime_acceptance_input_sha256") != dict(m2_06_inputs):
        raise M2LabelFreeAcceptanceError("M2-07 does not bind the current M2-06 inputs.")

    m2_03_side = _mapping(artifacts["m2_03"].get("artifacts"), "M2-03.artifacts")
    for key, field in (("m2_03_arrays", "arrays"), ("m2_03_figure", "figure")):
        embedded = _resolve_relative(
            artifact_paths["m2_03"].parent,
            m2_03_side.get(field),
            f"M2-03 {field}",
        )
        if embedded != audit_inputs[key]:
            raise M2LabelFreeAcceptanceError(f"audit_input_refs.{key} differs from M2-03.")

    for name, key in (("m2_07", "m2_07_log"), ("m2_lf_01", "m2_lf_log")):
        contract = _mapping(artifacts[name].get("artifact_contract"), f"{name}.artifact_contract")
        embedded = _resolve_relative(
            artifact_paths[name].parent,
            contract.get("companion_log_name"),
            f"{name} companion log",
        )
        if embedded != audit_inputs[key]:
            raise M2LabelFreeAcceptanceError(f"audit_input_refs.{key} differs from {name}.")

    m2_04_log = _parse_single_json_log(audit_inputs["m2_04_log"], "M2-04")
    if m2_04_log.get("K2_peak_allocated_mib") != _nested(
        artifacts["m2_04"], "profiles", "K2", "metrics", "peak_allocated_mib"
    ) or m2_04_log.get("K3_peak_allocated_mib") != _nested(
        artifacts["m2_04"], "profiles", "K3", "metrics", "peak_allocated_mib"
    ):
        raise M2LabelFreeAcceptanceError("M2-04 log numbers differ from its artifact.")
    m2_05_log = _parse_single_json_log(audit_inputs["m2_05_log"], "M2-05")
    if m2_05_log.get("shared_safe_batch_candidate") != _nested(
        artifacts["m2_05"], "selection", "shared_safe_batch_candidate"
    ):
        raise M2LabelFreeAcceptanceError("M2-05 log numbers differ from its artifact.")
    m2_06_log = _parse_single_json_log(audit_inputs["m2_06_log"], "M2-06")
    if m2_06_log.get("frozen_batch") != _nested(
        artifacts["m2_06"], "resolved_runtime", "micro_batch_size"
    ):
        raise M2LabelFreeAcceptanceError("M2-06 log numbers differ from its artifact.")
    failed_timing_log = audit_inputs["m2_06_failed_timing_log"].read_text(encoding="utf-8")
    if (
        "RuntimeContractError" not in failed_timing_log
        or "exact timing gate" not in failed_timing_log
        or "no acceptance JSON was written" not in failed_timing_log
    ):
        raise M2LabelFreeAcceptanceError("M2-06 failed-timing evidence is incomplete.")

    junit_path = audit_inputs["pytest_junit"]
    pytest_log_path = audit_inputs["pytest_log"]
    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise M2LabelFreeAcceptanceError("Cannot parse pytest JUnit artifact.") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise M2LabelFreeAcceptanceError("Pytest JUnit contains no testsuite.")
    pytest_summary = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    pytest_contract = _mapping(config.get("pytest_contract"), "pytest_contract")
    expected_pytest_summary = {
        key: int(pytest_contract[key]) for key in ("tests", "failures", "errors", "skipped")
    }
    if pytest_summary != expected_pytest_summary:
        raise M2LabelFreeAcceptanceError("Pytest JUnit does not match the exact test contract.")
    if _sha256(junit_path) != pytest_contract.get("junit_sha256"):
        raise M2LabelFreeAcceptanceError("Pytest JUnit hash does not match the frozen contract.")
    if _sha256(pytest_log_path) != pytest_contract.get("log_sha256"):
        raise M2LabelFreeAcceptanceError("Pytest log hash does not match the frozen contract.")
    log_lines = [
        line.strip()
        for line in pytest_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary_match = re.fullmatch(
        r"(?P<tests>\d+) passed in (?P<seconds>\d+(?:\.\d+)?)s"
        r"(?: \(\d+:\d+:\d+\))?",
        log_lines[-1] if log_lines else "",
    )
    if summary_match is None or int(summary_match.group("tests")) != pytest_summary["tests"]:
        raise M2LabelFreeAcceptanceError("Pytest log summary does not match JUnit.")

    audit_input_hashes = {
        **{f"artifact:{name}": artifact_hashes[path.name] for name, path in artifact_paths.items()},
        **{f"side:{name}": _sha256(path) for name, path in audit_inputs.items()},
        **{f"source:{name}": _sha256(path) for name, path in sources.items()},
        f"config:{config_path.name}": _sha256(config_path),
        **{
            f"raw_trace:{index:02d}": digest
            for index, digest in enumerate(current_fingerprint["raw_trace_content_sha256"])
        },
    }

    audit_path = _resolve_relative(
        config_path.parent,
        config.get("integrity_audit_ref"),
        "integrity_audit_ref",
    )
    audit = _load_json(audit_path, "experiment audit")
    if audit.get("audit_skill") != "experiment-audit":
        raise M2LabelFreeAcceptanceError("Integrity audit skill identity is missing.")
    if audit.get("review_independence") != "same-family":
        raise M2LabelFreeAcceptanceError("Integrity audit independence must be same-family.")
    if audit.get("acceptance_status") != "provisional":
        raise M2LabelFreeAcceptanceError("Integrity audit must remain provisional.")
    verdict = str(audit.get("overall_verdict", "")).upper()
    if verdict not in {"PASS", "WARN"}:
        raise M2LabelFreeAcceptanceError("Integrity audit has a blocking verdict.")
    audited_hashes = _mapping(audit.get("audited_input_sha256"), "audit input hashes")
    if dict(audited_hashes) != audit_input_hashes:
        raise M2LabelFreeAcceptanceError("Integrity audit does not bind the exact full input set.")

    checks = {
        "all_eight_label_free_or_synthetic_m2_artifacts_pass": True,
        "all_top_level_checks_pass": True,
        "all_recorded_source_hashes_match_current_sources": True,
        "m2_01_to_03_real_rf_scope_is_label_free": True,
        "m2_04_to_06_runtime_scope_is_synthetic_or_deterministic": True,
        "m2_07_is_synthetic_only": True,
        "m2_lf_01_has_no_target_optimizer_or_parameter_update": True,
        "m2_03_side_artifact_hashes_match": True,
        "current_five_trace_real_rf_content_is_hash_bound": True,
        "all_real_rf_artifacts_share_the_current_input_fingerprint": True,
        "m2_06_binds_current_runtime_config_and_m2_04_05": True,
        "m2_07_binds_current_runtime_config": True,
        "all_side_logs_and_failure_evidence_are_bound": True,
        "pycharm_codex_frontend_contract_is_portable": True,
        "full_pytest_junit_and_log_match_exact_zero_skip_contract": True,
        "fresh_audit_binds_artifacts_sides_sources_config_and_raw_inputs": True,
        "provisional_integrity_audit_is_bound_and_not_fail": True,
        "actual_label_tasks_remain_excluded": True,
        "formal_m2_10_and_m2_gate_remain_gated": True,
    }
    return {
        "schema_version": "a3-m2-label-free-engineering-acceptance-v2",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-LF-GATE",
        "status": "PASS",
        "evidence_class": "deterministic_label_free_engineering_subgate_only",
        "scope": {
            "reads_actual_labels": False,
            "requires_label_schema": False,
            "requires_sensitive_mapping": False,
            "requires_split_manifest": False,
            "performs_training": False,
            "performs_model_selection": False,
            "is_formal_m2_gate": False,
        },
        "task_summary": task_summary,
        "checks": checks,
        "artifact_sha256": artifact_hashes,
        "audit_input_sha256": audit_input_hashes,
        "verified_source_sha256": verified_source_hashes,
        "pytest": {
            **pytest_summary,
            "junit_name": junit_path.name,
            "junit_sha256": _sha256(junit_path),
            "log_name": pytest_log_path.name,
            "log_sha256": _sha256(pytest_log_path),
        },
        "integrity_audit": {
            "overall_verdict": verdict,
            "reason_code": audit.get("reason_code"),
            "review_independence": "same-family",
            "acceptance_status": "provisional",
            "artifact_sha256": _sha256(audit_path),
        },
        "boundary": config["boundary"],
        "interpretation_boundary": (
            "PASS is a deterministic engineering subgate for the current M2 work that "
            "does not use actual labels. It binds label-free real-RF batch/forward/"
            "support/autograd evidence and synthetic runtime/learning-path evidence to "
            "current sources plus a provisional integrity audit. It is not formal "
            "M2-GATE, does not freeze label-dependent numeric choices, and does not "
            "authorize supervised training or support performance/mechanism claims."
        ),
    }


__all__ = [
    "ARTIFACT_SPECS",
    "M2LabelFreeAcceptanceError",
    "build_m2_label_free_acceptance",
]
