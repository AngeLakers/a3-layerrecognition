"""Compute the label-free M1-12 PCC/matched-joint head capacity match."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from a3_ultrasound.models.capacity import select_matched_joint_hidden_width


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository: Path) -> dict[str, Any]:
    common = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    commit = subprocess.run(
        [*common, "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [*common, "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    entries = (
        [line for line in status.stdout.splitlines() if line.strip()]
        if status.returncode == 0
        else []
    )
    return {
        "state": "tracked" if commit.returncode == 0 else "unborn_repository",
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(entries) if status.returncode == 0 else None,
        "status_entry_count": len(entries) if status.returncode == 0 else None,
    }


def compute(spec_path: Path) -> dict[str, Any]:
    repository = spec_path.resolve().parents[2]
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "a3-matched-joint-spec-v1":
        raise ValueError("Unsupported matched-joint spec schema.")
    capacity = spec["capacity_match"]
    dimensions = capacity["reference_dimensions"]
    search = capacity["joint_hidden_width_search"]
    if search["reads_labels_or_validation_metrics"] is not False:
        raise ValueError("Capacity search must be label- and metric-free.")
    supported_k = tuple(spec["state_space"]["supported_K"])
    match = select_matched_joint_hidden_width(
        feature_dim=dimensions["feature_dim"],
        query_dim=dimensions["query_dim"],
        token_dim=dimensions["token_dim"],
        prefix_dim=dimensions["prefix_dim"],
        pcc_hidden_dim=dimensions["pcc_hidden_dim"],
        fusion_dim=dimensions["joint_fusion_dim"],
        minimum=search["minimum"],
        maximum=search["maximum"],
        tolerance_relative=capacity["tolerance_relative"],
        max_interfaces=dimensions["max_interfaces"],
        supported_k=supported_k,
    )
    if not match.within_tolerance:
        raise RuntimeError("No selected matched-joint width satisfies the tolerance.")

    script_path = Path(__file__).resolve()
    capacity_source = (
        repository / "src" / "a3_ultrasound" / "models" / "capacity.py"
    )
    return {
        "schema_version": "a3-m1-11-12-capacity-match-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_ids": ["M1-11", "M1-12"],
        "status": "PASS",
        "evidence_class": "structural_parameter_count_only",
        "input_record": {
            "reads_real_rf": False,
            "reads_labels": False,
            "reads_validation_metrics": False,
            "performs_training": False,
            "spec_schema": spec["schema_version"],
        },
        "reference_dimensions": dimensions,
        "selection_rule": {
            "candidate_minimum": search["minimum"],
            "candidate_maximum": search["maximum"],
            "candidate_count": search["maximum"] - search["minimum"] + 1,
            "objective": search["objective"],
            "tie_break": search["tie_break"],
            "shared_width_for_K2_and_K3": search["shared_for_K2_and_K3"],
        },
        "parameter_scope": capacity["comparison_scope"],
        "counts": {
            "pcc": match.pcc.to_dict(),
            "matched_joint": match.matched_joint.to_dict(),
            "selected_joint_hidden_dim": match.joint_hidden_dim,
            "absolute_difference": match.absolute_difference,
            "relative_difference": match.relative_difference,
            "relative_difference_percent": 100.0 * match.relative_difference,
            "tolerance_relative": match.tolerance_relative,
            "within_tolerance": match.within_tolerance,
        },
        "formula_contract": {
            "pcc": (
                "projector + local_base + prefix_embedding + prefix_mlp "
                "+ correction_mlp + gate_mlp"
            ),
            "matched_joint": (
                "projector + local_base + fusion_linear "
                "+ K2_flat_mlp + K3_flat_mlp"
            ),
        },
        "fairness_contract": spec["initialization_and_optimization"],
        "source_repository": _git_state(repository),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": "cpu",
        },
        "source_sha256": {
            "matched_joint_spec.yaml": _sha256(spec_path),
            "capacity.py": _sha256(capacity_source),
            "compute_capacity_match.py": _sha256(script_path),
        },
        "interpretation_boundary": (
            "PASS freezes a label-free structural parameter-count match only. "
            "It does not implement or train the M4 matched-joint baseline, "
            "measure runtime fairness, or provide performance evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/model/matched_joint_spec.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "joint_hidden_dim": result["counts"]["selected_joint_hidden_dim"],
                "relative_difference_percent": result["counts"][
                    "relative_difference_percent"
                ],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
