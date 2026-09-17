from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import a3_ultrasound.m2_label_free_acceptance as acceptance_module
from a3_ultrasound.m2_label_free_acceptance import (
    ACCEPTANCE_SOURCE_KEYS,
    ARTIFACT_SPECS,
    M2LabelFreeAcceptanceError,
    build_m2_label_free_acceptance,
)


RAW_HASHES = [hashlib.sha256(f"trace-{index}".encode()).hexdigest() for index in range(5)]
FINGERPRINT_BODY = {
    "algorithm": "sha256",
    "trace_count": 5,
    "raw_trace_content_sha256": RAW_HASHES,
    "normalized_batch_sha256": hashlib.sha256(b"normalized-batch").hexdigest(),
    "contains_filenames_or_coordinates": False,
}
FINGERPRINT = {
    **FINGERPRINT_BODY,
    "aggregate_sha256": hashlib.sha256(
        json.dumps(FINGERPRINT_BODY, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
}


@pytest.fixture(autouse=True)
def _stub_current_real_rf_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acceptance_module,
        "_current_real_rf_fingerprint",
        lambda _sources: copy.deepcopy(FINGERPRINT),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_payload(name: str, source_hash: str) -> dict:
    spec = ARTIFACT_SPECS[name]
    return {
        "schema_version": spec["schema_version"],
        "run_id": spec["run_id"],
        "status": "PASS",
        "evidence_class": spec["evidence_class"],
        "scope": dict(spec["scope"]),
        "checks": {"fixture_check": True},
        "source_sha256": {
            key: source_hash for key in spec["source_keys"]
        },
    }


def _payloads(source_hash: str) -> dict[str, dict]:
    rows = {name: _base_payload(name, source_hash) for name in ARTIFACT_SPECS}
    rows["m2_01"].update(
        {
            "input_fingerprint": copy.deepcopy(FINGERPRINT),
            "selection": {"trace_count": 5},
            "tensor_contract": {
                "x_raw": {"shape": [5, 2300]},
                "interface_meta": {"shape": [5, 3, 5]},
                "y": None,
                "label_mask": None,
            },
        }
    )
    rows["m2_02"].update(
        {
            "input_fingerprint": copy.deepcopy(FINGERPRINT),
            "resolved_config": {"batch_size": 5},
            "registered_contract": {"labels_absent": True},
            "support_audit": {"valid_local_support_count": 10},
            "output_audit": {"exact_probability_max_abs_error": 1e-7},
        }
    )
    rows["m2_03"].update(
        {
            "input_fingerprint": copy.deepcopy(FINGERPRINT),
            "global_audit": {"valid_interface_rows": 10},
            "artifacts": {
                "arrays": "attention.npz",
                "figure": "attention.png",
                "arrays_contain_raw_rf": False,
                "arrays_contain_scan_coordinates": False,
            },
        }
    )
    rows["m2_04"]["profiles"] = {
        "K2": {"metrics": {"peak_allocated_mib": 27.0}},
        "K3": {"metrics": {"peak_allocated_mib": 28.0}},
    }
    rows["m2_05"]["selection"] = {"shared_safe_batch_candidate": 256}
    rows["m2_06"]["integrity_boundary"] = {
        "real_label_gate_evaluated": False,
        "numeric_search_space_frozen": False,
    }
    rows["m2_07"].update(
        {
            "resolved_config": {
                "micro_batch_size": 128,
                "amp_enabled": False,
                "loss_ratio_threshold": 0.25,
            },
            "data_contract": {
                "valid_interface_count": 40,
                "positive_valid_labels": 20,
                "negative_valid_labels": 20,
            },
            "training": {"loss_ratio": 0.1},
            "execution_frontend_contract": {
                "entrypoint": "scripts/m2_07_synthetic_tiny_overfit.py",
                "required_parameters": [
                    "--runtime-config <RUNTIME_CONFIG>",
                    "--output <RESULT_JSON>",
                    "--log <RUN_LOG>",
                ],
                "pycharm": {
                    "project": "a3-layerrecognition",
                    "interpreter": "aislee-a3-layercongition",
                    "same_cli_contract_available": True,
                    "execution_recorded_by_this_run": False,
                },
                "codex_cli": {
                    "same_cli_contract_available": True,
                    "execution_recorded_by_this_run": True,
                },
                "contains_machine_absolute_paths": False,
                "codex_chat_state_required": False,
            },
            "runtime_contract": {
                "config_name": "runtime.yaml",
                "config_sha256": "filled_later",
                "resolved_runtime": {
                    "micro_batch_size": 128,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 128,
                    "precision": "float32",
                    "amp_enabled": False,
                },
                "runtime_acceptance_input_sha256": "filled_later",
            },
            "artifact_contract": {
                "companion_log_name": "m2_07.log",
                "companion_log_generated_by_runner": True,
            },
        }
    )
    rows["m2_lf_01"].update(
        {
            "input_fingerprint": copy.deepcopy(FINGERPRINT),
            "probe_contract": {
                "is_training_loss": False,
                "is_model_selection_metric": False,
                "has_physical_interpretation": False,
            },
            "gradient_audit": {
                "model_state_sha256_before": "same",
                "model_state_sha256_after": "same",
                "parameter_groups": {
                    f"group_{index}": {
                        "nonzero_gradient_present": True,
                        "all_gradients_finite": True,
                    }
                    for index in range(9)
                },
            },
            "environment": {
                "registered_environment_matches": True,
                "pythonpath_environment_variable_present": False,
            },
            "execution_frontend_contract": {
                "entrypoint": "scripts/m2_label_free_real_rf_backward_diagnostic.py",
                "required_parameters": [
                    "--raw-root <RAW_ROOT>",
                    "--manifest <ACQUISITION_MANIFEST>",
                    "--metadata-schema <METADATA_SCHEMA>",
                    "--output <RESULT_JSON>",
                    "--log <RUN_LOG>",
                ],
                "pycharm": {
                    "project": "a3-layerrecognition",
                    "interpreter": "aislee-a3-layercongition",
                    "same_cli_contract_available": True,
                    "execution_recorded_by_this_run": False,
                },
                "codex_cli": {
                    "same_cli_contract_available": True,
                    "execution_recorded_by_this_run": True,
                },
                "contains_machine_absolute_paths": False,
                "codex_chat_state_required": False,
            },
            "artifact_contract": {
                "companion_log_name": "m2_lf.log",
                "companion_log_generated_by_runner": True,
            },
        }
    )
    return rows


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    shared = implementation / "shared.py"
    shared.write_text("VALUE = 1\n", encoding="utf-8")
    source_hash = _sha256(shared)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "runtime.yaml").write_text("runtime: fixture\n", encoding="utf-8")
    (config_dir / "attention.npz").write_bytes(b"arrays")
    (config_dir / "attention.png").write_bytes(b"figure")
    (config_dir / "m2_lf.log").write_text("PASS\n", encoding="utf-8")
    (config_dir / "m2_07.log").write_text("PASS\n", encoding="utf-8")
    (config_dir / "m2_04.log").write_text(
        json.dumps(
            {
                "run_id": "M2-04",
                "status": "PASS",
                "K2_peak_allocated_mib": 27.0,
                "K3_peak_allocated_mib": 28.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "m2_05.log").write_text(
        json.dumps(
            {"run_id": "M2-05", "status": "PASS", "shared_safe_batch_candidate": 256}
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "m2_06.log").write_text(
        json.dumps({"run_id": "M2-06", "status": "PASS", "frozen_batch": 128})
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "m2_06_failed.log").write_text(
        "RuntimeContractError: exact timing gate rejected drift; "
        "no acceptance JSON was written\n",
        encoding="utf-8",
    )
    (config_dir / "pytest.xml").write_text(
        '<testsuites><testsuite tests="8" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    (config_dir / "pytest.log").write_text("8 passed in 1.00s\n", encoding="utf-8")
    payloads = _payloads(source_hash)
    payloads["m2_07"]["runtime_contract"]["config_sha256"] = _sha256(
        config_dir / "runtime.yaml"
    )
    artifacts: dict[str, Path] = {}
    for name, payload in payloads.items():
        if name == "m2_03":
            payload["artifacts"]["arrays_sha256"] = _sha256(config_dir / "attention.npz")
            payload["artifacts"]["figure_sha256"] = _sha256(config_dir / "attention.png")
        if name == "m2_lf_01":
            payload["artifact_contract"]["companion_log_sha256"] = _sha256(
                config_dir / "m2_lf.log"
            )
        if name == "m2_07":
            payload["artifact_contract"]["companion_log_sha256"] = _sha256(
                config_dir / "m2_07.log"
            )
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[name] = path
    payloads["m2_06"]["input_sha256"] = {
        "runtime.yaml": _sha256(config_dir / "runtime.yaml"),
        "m2_04.json": _sha256(artifacts["m2_04"]),
        "m2_05.json": _sha256(artifacts["m2_05"]),
    }
    payloads["m2_06"]["resolved_runtime"] = copy.deepcopy(
        payloads["m2_07"]["runtime_contract"]["resolved_runtime"]
    )
    payloads["m2_07"]["runtime_contract"]["runtime_acceptance_input_sha256"] = (
        copy.deepcopy(payloads["m2_06"]["input_sha256"])
    )
    artifacts["m2_07"].write_text(json.dumps(payloads["m2_07"]), encoding="utf-8")
    artifacts["m2_06"].write_text(json.dumps(payloads["m2_06"]), encoding="utf-8")
    audit_input_refs = {
        "runtime_config": "runtime.yaml",
        "m2_03_arrays": "attention.npz",
        "m2_03_figure": "attention.png",
        "m2_04_log": "m2_04.log",
        "m2_05_log": "m2_05.log",
        "m2_06_log": "m2_06.log",
        "m2_06_failed_timing_log": "m2_06_failed.log",
        "m2_07_log": "m2_07.log",
        "m2_lf_log": "m2_lf.log",
        "pytest_junit": "pytest.xml",
        "pytest_log": "pytest.log",
    }
    source_keys = {
        key for spec in ARTIFACT_SPECS.values() for key in spec["source_keys"]
    } | ACCEPTANCE_SOURCE_KEYS
    config = {
        "schema_version": "a3-m2-label-free-engineering-acceptance-config-v2",
        "run_id": "M2-LF-GATE",
        "status": "engineering_subgate_only_not_formal_m2_gate",
        "evidence_refs": {name: path.name for name, path in artifacts.items()},
        "audit_input_refs": audit_input_refs,
        "pytest_contract": {
            "tests": 8,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "junit_sha256": _sha256(config_dir / "pytest.xml"),
            "log_sha256": _sha256(config_dir / "pytest.log"),
        },
        "integrity_audit_ref": "audit.json",
        "source_bindings": {
            key: {"root": "implementation", "path": "shared.py"}
            for key in source_keys
        },
        "boundary": {
            "actual_label_tasks_excluded": True,
            "m2_08_status": "EXCLUDED_BY_CURRENT_USER_SCOPE",
            "m2_09_status": "EXCLUDED_BY_CURRENT_USER_SCOPE",
            "numeric_search_space_frozen": False,
            "formal_m2_10_status": "GATED",
            "formal_m2_gate_status": "GATED",
            "authorizes_supervised_training": False,
            "supports_performance_or_mechanism_claims": False,
        },
    }
    config_path = config_dir / "acceptance.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    audit_input_hashes = {
        **{f"artifact:{name}": _sha256(path) for name, path in artifacts.items()},
        **{
            f"side:{name}": _sha256(config_dir / reference)
            for name, reference in audit_input_refs.items()
        },
        **{f"source:{name}": source_hash for name in source_keys},
        "config:acceptance.yaml": _sha256(config_path),
        **{f"raw_trace:{index:02d}": digest for index, digest in enumerate(RAW_HASHES)},
    }
    audit = {
        "audit_skill": "experiment-audit",
        "overall_verdict": "WARN",
        "reason_code": "scope_limited_engineering_only",
        "review_independence": "same-family",
        "acceptance_status": "provisional",
        "audited_input_sha256": audit_input_hashes,
    }
    (config_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return config_path, implementation, payloads


def _rewrite_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_m2_label_free_acceptance_passes_as_nonformal_subgate(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    result = build_m2_label_free_acceptance(config_path, implementation)
    assert result["status"] == "PASS"
    assert result["scope"]["reads_actual_labels"] is False
    assert result["scope"]["is_formal_m2_gate"] is False
    assert result["boundary"]["formal_m2_gate_status"] == "GATED"
    assert result["integrity_audit"]["overall_verdict"] == "WARN"
    assert all(result["checks"].values())


def test_rejects_actual_label_scope_drift(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_02"]["scope"]["reads_registered_labels"] = True
    _rewrite_json(config_path.parent / "m2_02.json", payloads["m2_02"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="exact scope contract"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_current_source_hash_drift(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    (implementation / "shared.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(M2LabelFreeAcceptanceError, match="source hash mismatch"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_missing_transitive_source_binding(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["source_bindings"].pop("package__schema.py")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(M2LabelFreeAcceptanceError, match="exact evidence and acceptance"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_absolute_frontend_path(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_07"]["execution_frontend_contract"]["parameters"] = (
        "--output C:/machine/result.json"
    )
    _rewrite_json(config_path.parent / "m2_07.json", payloads["m2_07"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="absolute path"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_formal_gate_promotion(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["boundary"]["formal_m2_gate_status"] = "PASS"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(M2LabelFreeAcceptanceError, match="exact nonformal contract"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_integrity_audit_fail(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    audit_path = config_path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["overall_verdict"] = "FAIL"
    _rewrite_json(audit_path, audit)
    with pytest.raises(M2LabelFreeAcceptanceError, match="blocking verdict"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_runtime_input_hash_drift(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_06"]["input_sha256"]["runtime.yaml"] = "0" * 64
    _rewrite_json(config_path.parent / "m2_06.json", payloads["m2_06"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="input hashes"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_m2_lf_parameter_update(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_lf_01"]["gradient_audit"]["model_state_sha256_after"] = "changed"
    _rewrite_json(config_path.parent / "m2_lf_01.json", payloads["m2_lf_01"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="changed model parameters"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_real_rf_input_fingerprint_drift(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_03"]["input_fingerprint"]["normalized_batch_sha256"] = "0" * 64
    body = {
        key: value
        for key, value in payloads["m2_03"]["input_fingerprint"].items()
        if key != "aggregate_sha256"
    }
    payloads["m2_03"]["input_fingerprint"]["aggregate_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _rewrite_json(config_path.parent / "m2_03.json", payloads["m2_03"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="current five-trace"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_pytest_count_or_skip_drift(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    (config_path.parent / "pytest.xml").write_text(
        '<testsuites><testsuite tests="8" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(M2LabelFreeAcceptanceError, match="exact test contract"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_pytest_log_drift(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    (config_path.parent / "pytest.log").write_text("7 passed in 1.00s\n", encoding="utf-8")
    with pytest.raises(M2LabelFreeAcceptanceError, match="log hash"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_unrecorded_pycharm_execution_claim(tmp_path: Path) -> None:
    config_path, implementation, payloads = _write_fixture(tmp_path)
    payloads["m2_07"]["execution_frontend_contract"]["pycharm"][
        "execution_recorded_by_this_run"
    ] = True
    _rewrite_json(config_path.parent / "m2_07.json", payloads["m2_07"])
    with pytest.raises(M2LabelFreeAcceptanceError, match="overstates"):
        build_m2_label_free_acceptance(config_path, implementation)


def test_rejects_incomplete_fresh_audit_binding(tmp_path: Path) -> None:
    config_path, implementation, _ = _write_fixture(tmp_path)
    audit_path = config_path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["audited_input_sha256"].pop("side:pytest_log")
    _rewrite_json(audit_path, audit)
    with pytest.raises(M2LabelFreeAcceptanceError, match="exact full input set"):
        build_m2_label_free_acceptance(config_path, implementation)
