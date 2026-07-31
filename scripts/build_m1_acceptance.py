"""Build deterministic M1-10 and M1-GATE acceptance records from artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


STRUCTURAL_ITEMS: tuple[dict[str, Any], ...] = (
    {
        "item": 1,
        "requirement": "Valid exact-enumeration path probabilities sum to one.",
        "tests": [
            "test_exact_enumerator_normalizes_and_matches_bruteforce_marginals",
            "test_collate_to_model_exact_inference_pipeline_is_normalized",
        ],
    },
    {
        "item": 2,
        "requirement": "Marginals match brute-force valid-path summation.",
        "tests": ["test_exact_enumerator_normalizes_and_matches_bruteforce_marginals"],
    },
    {
        "item": 3,
        "requirement": "K=1 degenerates to a plain Bernoulli model.",
        "tests": ["test_k1_degenerates_to_a_plain_bernoulli_and_map_matches_argmax"],
    },
    {
        "item": 4,
        "requirement": "Padding values do not alter valid path distributions.",
        "tests": ["test_padding_values_do_not_change_valid_path_distribution"],
    },
    {
        "item": 5,
        "requirement": "Invalid slots are masked across Query, tokens, loss, and enumeration.",
        "tests": [
            "test_query_encoder_masks_invalid_slots_without_absolute_id",
            "test_mixed_k_masks_exclude_padding_from_paths_marginals_and_probabilities",
            "test_loss_rejects_padding_supervision_and_dirty_labels",
            "test_delta_changes_only_prior_fields_and_not_query",
            "test_valid_metadata_is_invariant_to_padding_kmax",
        ],
    },
    {
        "item": 6,
        "requirement": "Log-space probabilities match direct small-K Bernoulli products.",
        "tests": [
            "test_log_space_probabilities_match_direct_bernoulli_product",
            "test_exact_enumerator_is_explicitly_normalized_for_extreme_logits",
        ],
    },
    {
        "item": 7,
        "requirement": "MAP state is the valid-path probability argmax.",
        "tests": ["test_k1_degenerates_to_a_plain_bernoulli_and_map_matches_argmax"],
    },
    {
        "item": 8,
        "requirement": "Mixed K=2/3 batches exclude padded interfaces.",
        "tests": [
            "test_mixed_k_masks_exclude_padding_from_paths_marginals_and_probabilities",
            "test_integrated_model_mixed_k_forward_shapes_and_masks",
        ],
    },
    {
        "item": 9,
        "requirement": "Truth-prefix conditionals match integrated exact enumeration.",
        "tests": ["test_truth_prefix_forward_matches_integrated_exact_enumeration"],
    },
    {
        "item": 10,
        "requirement": "Correction has no local-evidence direct path and Pre is query-free.",
        "tests": [
            "test_correction_has_no_direct_autograd_path_from_local_representation",
            "test_correction_functionally_reads_only_context_and_prefix_path",
            "test_pre_readout_is_query_free_while_local_readout_remains_query_conditioned",
        ],
    },
    {
        "item": 11,
        "requirement": "Feature centers, support masks, and padding boundaries are exact.",
        "tests": [
            "test_feature_centers_use_exact_conv_geometry",
            "test_indexed_supports_respect_feature_padding_and_interface_mask",
            "test_integrated_model_reconstructs_each_trace_window_from_normalized_prior",
        ],
    },
    {
        "item": 12,
        "requirement": "RTX 4060 single-batch integrated forward/backward passes within 8 GB.",
        "tests": [],
        "artifact_key": "m1_13",
    },
    {
        "item": 13,
        "requirement": "Attention is zero outside support and normalized inside support.",
        "tests": [
            "test_local_readout_zeroes_attention_outside_support_and_normalizes_inside"
        ],
        "artifact_key": "m1_13",
    },
    {
        "item": 14,
        "requirement": "Empty local support fails; empty Pre returns shared context.",
        "tests": [
            "test_empty_local_support_fails_fast_but_empty_pre_returns_learned_context"
        ],
    },
    {
        "item": 15,
        "requirement": "Readouts/projectors are shared across slots with independent token projections.",
        "tests": [
            "test_pre_readout_is_query_free_while_local_readout_remains_query_conditioned",
            "test_responsibility_projector_uses_independent_source_projections_and_norms",
            "test_responsibility_projector_reapplies_interface_mask_after_layer_norm",
        ],
    },
    {
        "item": 16,
        "requirement": "K=2 has two orders and K=3 has all six permutations.",
        "tests": ["test_lexicographic_paths_and_order_controls_are_complete"],
    },
    {
        "item": 17,
        "requirement": "Single-class specimen macro-F1 records present classes and zero_division=0.",
        "tests": [
            "test_single_class_specimen_macro_f1_uses_present_classes_and_zero_division_zero"
        ],
    },
    {
        "item": 18,
        "requirement": "Synthetic Train/Validation/Calibration/Test fixture is specimen-disjoint.",
        "tests": [
            "test_specimen_split_fixture_is_disjoint_and_statistics_are_train_only"
        ],
    },
    {
        "item": 19,
        "requirement": "PCC/joint local and loss fairness plus <=10% capacity contract are frozen.",
        "tests": [
            "test_per_scan_ki_normalization_has_k2_k3_scale_parity",
            "test_matched_joint_width_is_label_free_and_within_ten_percent",
            "test_matched_joint_spec_freezes_fairness_and_seed_independence",
        ],
        "artifact_key": "m1_11_12",
    },
    {
        "item": 20,
        "requirement": "Shared initialization is seed-matched but parameter objects remain independent.",
        "tests": ["test_matched_joint_spec_freezes_fairness_and_seed_independence"],
        "artifact_key": "m1_11_12",
    },
    {
        "item": 21,
        "requirement": "Prefix/Gate invariants and Train-only shortcut-statistics guard pass.",
        "tests": [
            "test_prefix_mlp_is_ordered_masked_and_has_exact_zero_empty_prefix",
            "test_prefix_mlp_distinguishes_real_good_state_from_padding_and_rejects_k_above_three",
            "test_gate_is_path_independent_while_correction_can_depend_on_prefix",
            "test_specimen_split_fixture_is_disjoint_and_statistics_are_train_only",
        ],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _passed_tests(junit_path: Path) -> tuple[set[str], dict[str, int]]:
    root = ET.parse(junit_path).getroot()
    cases = root.findall(".//testcase")
    passed: set[str] = set()
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        name = case.attrib["name"]
        if case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            passed.add(name)
    return passed, counts


def _test_is_passed(test_name: str, passed: set[str]) -> bool:
    return any(name == test_name or name.startswith(f"{test_name}[") for name in passed)


def build(
    *,
    junit_path: Path,
    artifacts: dict[str, Path],
    structural_output: Path,
    gate_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    passed_tests, counts = _passed_tests(junit_path)
    if counts["failures"] or counts["errors"] or counts["skipped"]:
        raise RuntimeError(f"JUnit suite is not a clean full pass: {counts}")

    artifact_payloads = {key: _json(path) for key, path in artifacts.items()}
    accepted_statuses = {"PASS", "COMPLETE"}
    for key, payload in artifact_payloads.items():
        if payload.get("status") not in accepted_statuses:
            raise RuntimeError(f"Artifact {key} is not accepted: {payload.get('status')}")

    item_records = []
    for item in STRUCTURAL_ITEMS:
        missing_tests = [
            name for name in item["tests"] if not _test_is_passed(name, passed_tests)
        ]
        artifact_key = item.get("artifact_key")
        artifact_passed = (
            artifact_key is None
            or artifact_payloads[artifact_key]["status"] in accepted_statuses
        )
        status = "PASS" if not missing_tests and artifact_passed else "FAIL"
        item_records.append(
            {
                "item": item["item"],
                "status": status,
                "requirement": item["requirement"],
                "tests": item["tests"],
                "artifact": (
                    artifacts[artifact_key].name if artifact_key is not None else None
                ),
                "missing_tests": missing_tests,
            }
        )
    if any(record["status"] != "PASS" for record in item_records):
        raise RuntimeError("One or more M1 structural acceptance items failed.")

    generated_at = datetime.now().astimezone().isoformat()
    structural = {
        "schema_version": "a3-m1-10-structural-acceptance-v1",
        "generated_at": generated_at,
        "run_id": "M1-10",
        "status": "PASS",
        "evidence_class": "deterministic_software_acceptance_only",
        "pytest": counts,
        "items": item_records,
        "input_sha256": {
            "junit": _sha256(junit_path),
            **{path.name: _sha256(path) for path in artifacts.values()},
        },
        "boundary": {
            "synthetic_split_fixture_only": True,
            "real_specimen_manifest_validated": False,
            "matched_joint_training_implemented": False,
            "performance_claims_released": False,
        },
    }
    structural_output.parent.mkdir(parents=True, exist_ok=True)
    structural_output.write_text(
        json.dumps(structural, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_evidence = {
        "M1-01..03": artifacts["m1_01_03"].name,
        "M1-04": artifacts["m1_04"].name,
        "M1-05..06": artifacts["m1_05_06"].name,
        "M1-07": artifacts["m1_07"].name,
        "M1-08": "JUnit:test_empty_local_support_fails_fast_but_empty_pre_returns_learned_context",
        "M1-09": artifacts["m1_09"].name,
        "M1-10": structural_output.name,
        "M1-11..12": artifacts["m1_11_12"].name,
        "M1-13": artifacts["m1_13"].name,
    }
    gate = {
        "schema_version": "a3-m1-software-gate-v1",
        "generated_at": generated_at,
        "run_id": "M1-GATE",
        "status": "PASS",
        "evidence_class": "software_structure_and_synthetic_smoke_only",
        "run_evidence": run_evidence,
        "all_m1_tasks_01_through_13_accepted": True,
        "releases": {
            "M2_01_label_free_batch_construction": True,
            "real_rf_forward": False,
            "supervised_training": False,
            "outer_evaluation": False,
            "performance_or_mechanism_claims": False,
        },
        "remaining_blocks": [
            "M0-SUP label/specimen/split truth closure",
            "real RF loader and label-free forward acceptance",
            "matched-joint M4 implementation and runtime fairness",
            "specimen-safe training and evaluation",
        ],
        "integrity_boundary": (
            "This deterministic gate proves M1 software contracts and one "
            "synthetic CUDA batch only. It is not an independent experiment "
            "integrity audit and cannot support performance claims."
        ),
        "input_sha256": {
            structural_output.name: _sha256(structural_output),
            **{path.name: _sha256(path) for path in artifacts.values()},
        },
    }
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return structural, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--structural-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    names = {
        "m1_01_03": "m1_01_03_responsibility_acceptance.json",
        "m1_04": "m1_04_integrated_model_acceptance.json",
        "m1_05_06": "m0_15_m1_05_06_acceptance.json",
        "m1_07": "m1_07_consistency_acceptance.json",
        "m1_09": "m1_09_scale_sensitivity.json",
        "m1_11_12": "m1_11_12_capacity_match.json",
        "m1_13": "m1_13_integrated_gpu_smoke.json",
    }
    artifacts = {key: args.results_dir / name for key, name in names.items()}
    structural, gate = build(
        junit_path=args.junit,
        artifacts=artifacts,
        structural_output=args.structural_output,
        gate_output=args.gate_output,
    )
    print(
        json.dumps(
            {
                "m1_10": structural["status"],
                "accepted_items": len(structural["items"]),
                "pytest_tests": structural["pytest"]["tests"],
                "m1_gate": gate["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
