"""Deterministic acceptance checks for the label-free M2 runtime freeze."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


class RuntimeContractError(ValueError):
    """Raised when the runtime freeze or its evidence is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(f"{name} must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeContractError(f"{name} must be a positive integer.")
    return value


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeContractError(f"{name} must be a finite number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeContractError(f"{name} must be a finite number.")
    return resolved


def _all_boolean_checks_pass(checks: Any, name: str) -> None:
    resolved = _mapping(checks, name)
    if not resolved or any(value is not True for value in resolved.values()):
        raise RuntimeContractError(f"Every {name} check must be true.")


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeContractError(f"Cannot load runtime config: {path.name}.") from exc
    return _mapping(payload, "runtime config")


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"Cannot load {name}: {path.name}.") from exc
    return _mapping(payload, name)


def _resolve_evidence(config_path: Path, reference: Any, name: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeContractError(f"{name} must be a nonempty relative path.")
    candidate = Path(reference)
    if candidate.is_absolute():
        raise RuntimeContractError(f"{name} must remain a relative path.")
    resolved = (config_path.parent / candidate).resolve()
    if not resolved.is_file():
        raise RuntimeContractError(f"{name} does not resolve to an existing file.")
    return resolved


def _require_label_free_scope(payload: Mapping[str, Any], name: str) -> None:
    scope = _mapping(payload.get("scope"), f"{name}.scope")
    required = {
        "reads_real_rf": False,
        "reads_real_labels": False,
        "uses_synthetic_labels": True,
        "performs_optimizer_steps": True,
        "saves_checkpoint": False,
        "performs_model_selection": False,
    }
    for key, expected in required.items():
        if scope.get(key) is not expected:
            raise RuntimeContractError(f"{name}.scope.{key} must be {expected}.")


def _find_profile(
    rows: Any,
    *,
    batch_size: int,
    interface_count: int,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise RuntimeContractError(f"{name} must be a list.")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("batch_size") == batch_size
        and row.get("interface_count") == interface_count
    ]
    if len(matches) != 1:
        raise RuntimeContractError(
            f"{name} must contain exactly one batch={batch_size}, K={interface_count} row."
        )
    return matches[0]


def _close(actual: Any, expected: Any, name: str) -> None:
    if not math.isclose(
        _finite_number(actual, f"{name}.actual"),
        _finite_number(expected, f"{name}.expected"),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeContractError(f"{name} does not match the evidence artifact.")


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "a3-runtime-freeze-v1":
        raise RuntimeContractError("runtime config schema_version is not supported.")
    if config.get("run_id") != "M2-06":
        raise RuntimeContractError("runtime config run_id must be M2-06.")
    if config.get("status") != "frozen_development_runtime_without_supervision":
        raise RuntimeContractError("runtime config must remain supervision-free.")

    scope = _mapping(config.get("scope"), "scope")
    if scope.get("evidence_class") != "synthetic_runtime_budget_only":
        raise RuntimeContractError("scope.evidence_class must remain synthetic-only.")
    if scope.get("supported_interface_counts") != [2, 3]:
        raise RuntimeContractError("scope must cover exactly K=2/3.")
    if set(scope.get("applies_to", [])) != {"pcc", "matched_joint"}:
        raise RuntimeContractError("runtime freeze must apply to PCC and matched joint.")

    runtime = _mapping(config.get("runtime"), "runtime")
    micro_batch = _positive_int(runtime.get("micro_batch_size"), "micro_batch_size")
    accumulation = _positive_int(
        runtime.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
    )
    effective_batch = _positive_int(
        runtime.get("effective_batch_size"), "effective_batch_size"
    )
    if effective_batch != micro_batch * accumulation:
        raise RuntimeContractError(
            "effective_batch_size must equal micro_batch_size times accumulation."
        )
    expected_runtime = {
        "precision": "float32",
        "amp_enabled": False,
        "amp_dtype": None,
        "gradient_scaler_enabled": False,
        "torch_compile_enabled": False,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise RuntimeContractError(f"runtime.{key} must be {expected!r}.")

    frontends = _mapping(config.get("execution_frontends"), "execution_frontends")
    if frontends.get("required_supported") != [
        "pycharm_run_debug",
        "codex_command_line",
    ]:
        raise RuntimeContractError("PyCharm and Codex must both be required frontends.")
    pycharm = _mapping(frontends.get("pycharm"), "execution_frontends.pycharm")
    if pycharm.get("project") != "a3-layerrecognition":
        raise RuntimeContractError("PyCharm project must be a3-layerrecognition.")
    if pycharm.get("interpreter") != "aislee-a3-layercongition":
        raise RuntimeContractError("PyCharm interpreter must use the registered Conda env.")
    if pycharm.get("entrypoint_rule") != "importable_module_or_cli_script":
        raise RuntimeContractError("PyCharm entrypoints must be importable or CLI-runnable.")
    if frontends.get("parity_invariants") != [
        "same_versioned_config",
        "same_seed",
        "same_data_manifest",
        "same_output_contract",
    ]:
        raise RuntimeContractError("Execution frontend parity invariants changed.")
    if frontends.get("forbid_codex_chat_state_as_runtime_dependency") is not True:
        raise RuntimeContractError("Codex chat state must not be a runtime dependency.")

    fairness = _mapping(config.get("fairness"), "fairness")
    fairness_flags = (
        "same_micro_batch_size_for_pcc_and_matched_joint",
        "same_effective_batch_size_for_pcc_and_matched_joint",
        "same_precision_for_pcc_and_matched_joint",
        "same_gradient_accumulation_for_pcc_and_matched_joint",
    )
    if any(fairness.get(flag) is not True for flag in fairness_flags):
        raise RuntimeContractError("PCC/matched-joint runtime fairness flags must be true.")

    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "precision": runtime["precision"],
        "amp_enabled": runtime["amp_enabled"],
    }


def build_runtime_acceptance(config_path: Path) -> dict[str, Any]:
    """Validate M2-06 against M2-04/05 and return a machine-readable record."""

    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    resolved_runtime = _validate_config(config)
    evidence = _mapping(config.get("evidence"), "evidence")
    profile_path = _resolve_evidence(
        config_path, evidence.get("batch1_profile_ref"), "batch1_profile_ref"
    )
    search_path = _resolve_evidence(
        config_path, evidence.get("safe_batch_search_ref"), "safe_batch_search_ref"
    )
    profile = _load_json(profile_path, "M2-04 profile")
    search = _load_json(search_path, "M2-05 search")

    if (
        profile.get("schema_version") != "a3-m2-04-gpu-profile-v1"
        or profile.get("run_id") != "M2-04"
        or profile.get("status") != "PASS"
    ):
        raise RuntimeContractError("M2-04 profile is not an accepted PASS artifact.")
    if (
        search.get("schema_version") != "a3-m2-05-safe-batch-search-v1"
        or search.get("run_id") != "M2-05"
        or search.get("status") != "PASS"
    ):
        raise RuntimeContractError("M2-05 search is not an accepted PASS artifact.")
    _require_label_free_scope(profile, "M2-04")
    _require_label_free_scope(search, "M2-05")
    _all_boolean_checks_pass(profile.get("checks"), "M2-04 top-level")
    _all_boolean_checks_pass(search.get("checks"), "M2-05 top-level")

    profiles = _mapping(profile.get("profiles"), "M2-04.profiles")
    for interface_count in (2, 3):
        row = _mapping(profiles.get(f"K{interface_count}"), f"M2-04.K{interface_count}")
        if row.get("interface_count") != interface_count or row.get("batch_size") != 1:
            raise RuntimeContractError("M2-04 must contain separate batch=1 K=2/3 rows.")
        if row.get("precision") != "float32" or row.get("amp_enabled") is not False:
            raise RuntimeContractError("M2-04 rows must use float32 without AMP.")
        _all_boolean_checks_pass(row.get("checks"), f"M2-04.K{interface_count}")

    selection = _mapping(search.get("selection"), "M2-05.selection")
    selected_batch = _positive_int(
        selection.get("shared_safe_batch_candidate"), "shared_safe_batch_candidate"
    )
    if selected_batch != evidence.get("observed_max_safe_batch_candidate"):
        raise RuntimeContractError("Frozen safe batch candidate differs from M2-05.")
    boundary_batch = _positive_int(
        selection.get("next_unsafe_or_oom_K3_batch"), "next boundary batch"
    )
    if boundary_batch != evidence.get("next_profiled_batch"):
        raise RuntimeContractError("Frozen boundary batch differs from M2-05.")

    matrix = _mapping(search.get("profile_matrix"), "M2-05.profile_matrix")
    k3_rows = matrix.get("K3_doubling_search")
    k2_rows = matrix.get("K2_selected_batch_validation")
    selected_k3 = _find_profile(
        k3_rows,
        batch_size=selected_batch,
        interface_count=3,
        name="M2-05 K3 search",
    )
    selected_k2 = _find_profile(
        k2_rows,
        batch_size=selected_batch,
        interface_count=2,
        name="M2-05 K2 validation",
    )
    minimum_free = _finite_number(
        evidence.get("minimum_required_free_fraction_at_peak"),
        "minimum_required_free_fraction_at_peak",
    )
    search_config = _mapping(search.get("resolved_config"), "M2-05.resolved_config")
    _close(
        search_config.get("minimum_free_fraction_at_peak"),
        minimum_free,
        "minimum free fraction",
    )
    if evidence.get("step_time_is_acceptance_gate") is not False:
        raise RuntimeContractError(
            "Shared-GPU step time must remain reference-only, not an acceptance gate."
        )
    reference_step_cuda_ms = _finite_number(
        evidence.get("k3_batch_128_reference_step_cuda_ms"),
        "k3_batch_128_reference_step_cuda_ms",
    )
    if reference_step_cuda_ms <= 0.0:
        raise RuntimeContractError("Reference step time must be positive.")
    for name, row in (("K2", selected_k2), ("K3", selected_k3)):
        if row.get("status") != "PASS" or row.get("safe_under_15_percent_rule") is not True:
            raise RuntimeContractError(f"Selected M2-05 {name} row is not safe.")
        _all_boolean_checks_pass(row.get("checks"), f"M2-05 selected {name}")
        reserve = _finite_number(
            _mapping(row.get("metrics"), f"M2-05 selected {name}.metrics").get(
                "estimated_device_free_fraction_at_peak"
            ),
            f"M2-05 selected {name} reserve",
        )
        if reserve < minimum_free:
            raise RuntimeContractError(f"Selected M2-05 {name} reserve is below minimum.")

    boundary = _find_profile(
        k3_rows,
        batch_size=boundary_batch,
        interface_count=3,
        name="M2-05 K3 search",
    )
    if boundary.get("status") == "OOM":
        boundary_status = "OOM"
    elif (
        boundary.get("status") == "PASS"
        and boundary.get("safe_under_15_percent_rule") is False
    ):
        boundary_status = "PASS_BUT_UNSAFE_UNDER_15_PERCENT_RULE"
    else:
        raise RuntimeContractError("M2-05 boundary is neither OOM nor unsafe.")
    if boundary_status != evidence.get("next_profiled_batch_status"):
        raise RuntimeContractError("Frozen boundary status differs from M2-05.")

    frozen_k3 = _find_profile(
        k3_rows,
        batch_size=resolved_runtime["micro_batch_size"],
        interface_count=3,
        name="M2-05 K3 search",
    )
    if frozen_k3.get("status") != "PASS" or frozen_k3.get("safe_under_15_percent_rule") is not True:
        raise RuntimeContractError("Frozen K3 micro batch is not a safe PASS row.")
    frozen_metrics = _mapping(frozen_k3.get("metrics"), "frozen K3 metrics")
    _close(
        frozen_metrics.get("peak_allocated_mib"),
        evidence.get("k3_batch_128_peak_allocated_mib"),
        "frozen K3 peak allocated",
    )
    observed_step_cuda_ms = _finite_number(
        frozen_metrics.get("step_cuda_ms"),
        "frozen K3 observed step time",
    )
    if observed_step_cuda_ms <= 0.0:
        raise RuntimeContractError("Observed K3 step time must be positive.")
    _close(
        frozen_metrics.get("estimated_device_free_fraction_at_peak"),
        evidence.get("k3_batch_128_estimated_free_fraction_at_peak"),
        "frozen K3 free fraction",
    )

    checks = {
        "runtime_schema_and_scope_valid": True,
        "effective_batch_arithmetic_valid": True,
        "float32_amp_and_compile_freeze_valid": True,
        "pycharm_codex_frontend_parity_valid": True,
        "pcc_matched_joint_fairness_flags_valid": True,
        "m2_04_batch1_k2_k3_profiles_pass": True,
        "m2_05_selected_k2_k3_batch_safe": True,
        "m2_05_boundary_matches_freeze": True,
        "frozen_batch_has_required_reserve": True,
        "step_time_recorded_but_not_used_as_acceptance_gate": True,
        "no_real_rf_or_real_labels_read": True,
    }
    profile_metrics = _mapping(
        _mapping(profiles["K3"], "M2-04.K3").get("metrics"), "M2-04.K3.metrics"
    )
    return {
        "schema_version": "a3-m2-06-runtime-acceptance-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-06",
        "status": "PASS",
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
        "resolved_runtime": resolved_runtime,
        "evidence_summary": {
            "m2_04_k3_batch1_peak_allocated_mib": profile_metrics.get(
                "peak_allocated_mib"
            ),
            "m2_04_k3_batch1_step_cuda_ms_median": profile_metrics.get(
                "step_cuda_ms_median"
            ),
            "m2_05_shared_safe_batch_candidate": selected_batch,
            "m2_05_boundary_batch": boundary_batch,
            "m2_05_boundary_status": boundary_status,
            "frozen_k3_peak_allocated_mib": frozen_metrics.get(
                "peak_allocated_mib"
            ),
            "frozen_k3_observed_step_cuda_ms": observed_step_cuda_ms,
            "initial_freeze_reference_step_cuda_ms": reference_step_cuda_ms,
            "step_time_is_acceptance_gate": False,
            "frozen_k3_estimated_free_fraction_at_peak": frozen_metrics.get(
                "estimated_device_free_fraction_at_peak"
            ),
            "minimum_required_free_fraction_at_peak": minimum_free,
        },
        "execution_frontends": config["execution_frontends"],
        "checks": checks,
        "input_sha256": {
            config_path.name: _sha256(config_path),
            profile_path.name: _sha256(profile_path),
            search_path.name: _sha256(search_path),
        },
        "integrity_boundary": {
            "independent_experiment_audit_performed": False,
            "deterministic_contract_acceptance_only": True,
            "real_label_gate_evaluated": False,
            "numeric_search_space_frozen": False,
        },
        "interpretation_boundary": (
            "PASS proves that the label-free M2-06 runtime freeze is internally "
            "consistent with the M2-04/05 synthetic GPU evidence and that PyCharm "
            "and Codex are required to share one explicit runtime contract. It does "
            "not gate on volatile shared-GPU step time or freeze label-dependent "
            "hyperparameters, authorize supervised "
            "training, or support RF performance or mechanism claims."
        ),
    }


__all__ = ["RuntimeContractError", "build_runtime_acceptance"]
