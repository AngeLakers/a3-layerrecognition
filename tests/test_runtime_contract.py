from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from a3_ultrasound.runtime_contract import (
    RuntimeContractError,
    build_runtime_acceptance,
)


def _profile() -> dict:
    row_checks = {"finite": True, "gradient": True}
    return {
        "schema_version": "a3-m2-04-gpu-profile-v1",
        "run_id": "M2-04",
        "status": "PASS",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
        },
        "checks": {"K2": True, "K3": True},
        "profiles": {
            "K2": {
                "interface_count": 2,
                "batch_size": 1,
                "precision": "float32",
                "amp_enabled": False,
                "metrics": {"peak_allocated_mib": 27.4, "step_cuda_ms_median": 31.0},
                "checks": dict(row_checks),
            },
            "K3": {
                "interface_count": 3,
                "batch_size": 1,
                "precision": "float32",
                "amp_enabled": False,
                "metrics": {"peak_allocated_mib": 27.6, "step_cuda_ms_median": 32.0},
                "checks": dict(row_checks),
            },
        },
    }


def _search() -> dict:
    def row(batch: int, k: int, *, safe: bool, reserve: float) -> dict:
        return {
            "batch_size": batch,
            "interface_count": k,
            "status": "PASS",
            "safe_under_15_percent_rule": safe,
            "metrics": {
                "peak_allocated_mib": 1167.984375 if batch == 128 else 2318.0,
                "step_cuda_ms": 208.89804077148438 if batch == 128 else 600.0,
                "estimated_device_free_fraction_at_peak": reserve,
            },
            "checks": {"finite": True, "gradient": True},
        }

    return {
        "schema_version": "a3-m2-05-safe-batch-search-v1",
        "run_id": "M2-05",
        "status": "PASS",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
        },
        "checks": {"boundary": True, "selected": True},
        "resolved_config": {"minimum_free_fraction_at_peak": 0.15},
        "selection": {
            "shared_safe_batch_candidate": 256,
            "next_unsafe_or_oom_K3_batch": 512,
            "next_unsafe_or_oom_K3_status": "PASS",
        },
        "profile_matrix": {
            "K3_doubling_search": [
                row(128, 3, safe=True, reserve=0.5580164002862595),
                row(256, 3, safe=True, reserve=0.2549),
                row(512, 3, safe=False, reserve=0.0),
            ],
            "K2_selected_batch_validation": [
                row(256, 2, safe=True, reserve=0.2549)
            ],
        },
    }


def _config() -> dict:
    return {
        "schema_version": "a3-runtime-freeze-v1",
        "run_id": "M2-06",
        "status": "frozen_development_runtime_without_supervision",
        "scope": {
            "evidence_class": "synthetic_runtime_budget_only",
            "supported_interface_counts": [2, 3],
            "applies_to": ["pcc", "matched_joint"],
        },
        "evidence": {
            "batch1_profile_ref": "m2_04.json",
            "safe_batch_search_ref": "m2_05.json",
            "observed_max_safe_batch_candidate": 256,
            "next_profiled_batch": 512,
            "next_profiled_batch_status": "PASS_BUT_UNSAFE_UNDER_15_PERCENT_RULE",
            "k3_batch_128_peak_allocated_mib": 1167.984375,
            "k3_batch_128_reference_step_cuda_ms": 208.89804077148438,
            "step_time_is_acceptance_gate": False,
            "k3_batch_128_estimated_free_fraction_at_peak": 0.5580164002862595,
            "minimum_required_free_fraction_at_peak": 0.15,
        },
        "runtime": {
            "micro_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 128,
            "precision": "float32",
            "amp_enabled": False,
            "amp_dtype": None,
            "gradient_scaler_enabled": False,
            "torch_compile_enabled": False,
        },
        "execution_frontends": {
            "required_supported": ["pycharm_run_debug", "codex_command_line"],
            "pycharm": {
                "project": "a3-layerrecognition",
                "interpreter": "aislee-a3-layercongition",
                "entrypoint_rule": "importable_module_or_cli_script",
            },
            "codex": {"role": "implementation_validation_optional_launch_and_monitoring"},
            "parity_invariants": [
                "same_versioned_config",
                "same_seed",
                "same_data_manifest",
                "same_output_contract",
            ],
            "forbid_codex_chat_state_as_runtime_dependency": True,
        },
        "fairness": {
            "same_micro_batch_size_for_pcc_and_matched_joint": True,
            "same_effective_batch_size_for_pcc_and_matched_joint": True,
            "same_precision_for_pcc_and_matched_joint": True,
            "same_gradient_accumulation_for_pcc_and_matched_joint": True,
        },
    }


def _write_fixture(tmp_path: Path, config: dict, profile: dict, search: dict) -> Path:
    (tmp_path / "m2_04.json").write_text(json.dumps(profile), encoding="utf-8")
    (tmp_path / "m2_05.json").write_text(json.dumps(search), encoding="utf-8")
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_runtime_acceptance_passes_without_real_rf_or_label_inputs(tmp_path: Path) -> None:
    result = build_runtime_acceptance(
        _write_fixture(tmp_path, _config(), _profile(), _search())
    )
    assert result["status"] == "PASS"
    assert result["resolved_runtime"]["micro_batch_size"] == 128
    assert result["scope"]["reads_real_labels"] is False
    assert result["scope"]["requires_sensitive_mapping"] is False
    assert result["integrity_boundary"]["numeric_search_space_frozen"] is False
    assert all(result["checks"].values())


def test_runtime_acceptance_rejects_effective_batch_mismatch(tmp_path: Path) -> None:
    config = _config()
    config["runtime"]["effective_batch_size"] = 256
    path = _write_fixture(tmp_path, config, _profile(), _search())
    with pytest.raises(RuntimeContractError, match="effective_batch_size"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_missing_pycharm_frontend(tmp_path: Path) -> None:
    config = _config()
    config["execution_frontends"]["required_supported"] = ["codex_command_line"]
    path = _write_fixture(tmp_path, config, _profile(), _search())
    with pytest.raises(RuntimeContractError, match="PyCharm and Codex"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_chat_state_dependency(tmp_path: Path) -> None:
    config = _config()
    config["execution_frontends"]["forbid_codex_chat_state_as_runtime_dependency"] = False
    path = _write_fixture(tmp_path, config, _profile(), _search())
    with pytest.raises(RuntimeContractError, match="chat state"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_failed_profile_check(tmp_path: Path) -> None:
    profile = _profile()
    profile["profiles"]["K3"]["checks"]["gradient"] = False
    path = _write_fixture(tmp_path, _config(), profile, _search())
    with pytest.raises(RuntimeContractError, match="Every M2-04.K3"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_selected_batch_drift(tmp_path: Path) -> None:
    search = _search()
    search["selection"]["shared_safe_batch_candidate"] = 128
    path = _write_fixture(tmp_path, _config(), _profile(), search)
    with pytest.raises(RuntimeContractError, match="safe batch candidate differs"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_boundary_status_drift(tmp_path: Path) -> None:
    config = _config()
    config["evidence"]["next_profiled_batch_status"] = "OOM"
    path = _write_fixture(tmp_path, config, _profile(), _search())
    with pytest.raises(RuntimeContractError, match="boundary status differs"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_rejects_frozen_metric_drift(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["evidence"]["k3_batch_128_peak_allocated_mib"] = 999.0
    path = _write_fixture(tmp_path, config, _profile(), _search())
    with pytest.raises(RuntimeContractError, match="peak allocated"):
        build_runtime_acceptance(path)


def test_runtime_acceptance_allows_shared_gpu_step_time_drift(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["evidence"]["k3_batch_128_reference_step_cuda_ms"] = 999.0
    result = build_runtime_acceptance(
        _write_fixture(tmp_path, config, _profile(), _search())
    )
    assert result["status"] == "PASS"
    assert result["evidence_summary"]["frozen_k3_observed_step_cuda_ms"] == pytest.approx(
        208.89804077148438
    )
    assert result["evidence_summary"]["initial_freeze_reference_step_cuda_ms"] == 999.0
    assert result["evidence_summary"]["step_time_is_acceptance_gate"] is False


def test_runtime_acceptance_rejects_step_time_as_gate(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["evidence"]["step_time_is_acceptance_gate"] = True
    with pytest.raises(RuntimeContractError, match="reference-only"):
        build_runtime_acceptance(
            _write_fixture(tmp_path, config, _profile(), _search())
        )
