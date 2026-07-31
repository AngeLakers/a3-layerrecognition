"""Build the M2-01 label-free real-RF batch contract without model forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from a3_ultrasound.data import (
    build_label_free_real_rf_batch,
    load_label_free_network_manifest,
)
from a3_ultrasound.schema import INTERFACE_META_FIELDS, INTERFACE_META_SCHEMA_VERSION


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


def _is_prefix_mask(mask: torch.Tensor) -> bool:
    counts = mask.sum(dim=1)
    expected = (
        torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    return bool(torch.equal(mask, expected))


def run(
    *,
    raw_root: Path,
    manifest_path: Path,
    metadata_schema_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_label_free_network_manifest(
        manifest_path,
        metadata_schema_path,
    )
    result = build_label_free_real_rf_batch(raw_root, manifest)
    batch = result.batch

    valid_lengths = batch.length_mask.sum(dim=1)
    scale = valid_lengths.to(dtype=batch.interface_meta.dtype).unsqueeze(1)
    nominal_tof = batch.interface_meta[..., 1] * scale
    window_left = batch.interface_meta[..., 3] * scale
    window_right = batch.interface_meta[..., 4] * scale
    window_start = nominal_tof - window_left
    window_end = nominal_tof + window_right
    valid = batch.interface_mask

    raw_layout = manifest["raw_layout"]
    structure = manifest["structure"]
    acquisition = manifest["acquisition"]
    normalization = manifest["preprocessing"]["amplitude_normalization"]
    expected_trace_samples = raw_layout.get("trace_samples")
    checks = {
        "network_manifest_validated_at_before_network_api_derivation": True,
        "selected_trace_count_is_five": result.trace_count == 5,
        "selected_traces_match_registered_length": (
            expected_trace_samples is not None
            and all(
                length == int(expected_trace_samples)
                for length in result.trace_lengths
            )
        ),
        "x_raw_float32_and_finite": (
            batch.x_raw.dtype == torch.float32
            and bool(torch.isfinite(batch.x_raw).all())
        ),
        "length_mask_bool_prefix": (
            batch.length_mask.dtype == torch.bool
            and _is_prefix_mask(batch.length_mask)
        ),
        "interface_meta_v3_float32_and_finite": (
            batch.interface_meta.dtype == torch.float32
            and batch.interface_meta.shape[-1] == 5
            and bool(torch.isfinite(batch.interface_meta).all())
        ),
        "interface_mask_bool_prefix": (
            batch.interface_mask.dtype == torch.bool
            and _is_prefix_mask(batch.interface_mask)
        ),
        "labels_absent": batch.y is None and batch.label_mask is None,
        "one_fixed_batch_divisor": (
            normalization["method"] == "fixed_batch_reference_division"
            and result.normalization_divisor == float(normalization["divisor_level"])
        ),
        "valid_windows_inside_trace": bool(
            torch.all(window_start[valid] >= 0)
            and torch.all(
                window_end[valid]
                <= (scale.expand_as(window_end)[valid] - 1.0)
            )
        ),
        "no_network_forward_or_training": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-01 batch contract checks failed: {checks}")

    valid_meta = batch.interface_meta[0, : int(batch.interface_mask[0].sum())]
    valid_tof = nominal_tof[0, batch.interface_mask[0]]
    valid_starts = window_start[0, batch.interface_mask[0]]
    valid_ends = window_end[0, batch.interface_mask[0]]
    x = batch.x_raw
    repository = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": "a3-m2-01-real-rf-batch-contract-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-01",
        "status": "PASS",
        "evidence_class": "sampled_real_rf_label_free_batch_construction_only",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "performs_network_forward": False,
            "performs_training": False,
            "performs_performance_evaluation": False,
            "content_audit_is_full_scan": False,
        },
        "selection": {
            "policy": result.selection_policy,
            "trace_count": result.trace_count,
            "registered_grid_shape": manifest["scan_geometry"]["grid_shape"],
            "coordinates_enter_network_batch": False,
            "coordinates_written_to_artifact": False,
            "enumerates_full_raw_directory": False,
        },
        "registered_contract": {
            "dataset_id": manifest["identity"]["dataset_id"],
            "acquisition_batch_id": manifest["identity"]["acquisition_batch_id"],
            "manifest_version": manifest.get("manifest_version"),
            "acquisition_profile_id": acquisition["acquisition_profile_id"],
            "window_profile_id": acquisition["window_profile"]["profile_id"],
            "sampling_rate_hz": acquisition["sampling_rate_hz"],
            "raw_unit": raw_layout["raw_unit"],
            "trace_samples": expected_trace_samples,
            "K": structure["K"],
            "zmax_mm": structure["total_thickness_mm"],
            "normalization_method": normalization["method"],
            "normalization_divisor_level": result.normalization_divisor,
            "denoising_applied": False,
            "source_ref": manifest["provenance"]["source_ref"],
        },
        "tensor_contract": {
            "interface_meta_schema": INTERFACE_META_SCHEMA_VERSION,
            "interface_meta_fields": list(INTERFACE_META_FIELDS),
            "x_raw": {
                "shape": list(batch.x_raw.shape),
                "dtype": str(batch.x_raw.dtype),
                "amplitude_state": "raw_level_divided_by_one_fixed_batch_reference",
            },
            "length_mask": {
                "shape": list(batch.length_mask.shape),
                "dtype": str(batch.length_mask.dtype),
                "valid_counts": valid_lengths.tolist(),
            },
            "interface_meta": {
                "shape": list(batch.interface_meta.shape),
                "dtype": str(batch.interface_meta.dtype),
                "first_trace_valid_rows": valid_meta.tolist(),
            },
            "interface_mask": {
                "shape": list(batch.interface_mask.shape),
                "dtype": str(batch.interface_mask.dtype),
                "valid_counts": batch.interface_mask.sum(dim=1).tolist(),
            },
            "label_mask": None,
            "y": None,
        },
        "time_support_audit": {
            "nominal_tof_samples": valid_tof.tolist(),
            "window_start_samples": valid_starts.tolist(),
            "window_end_samples": valid_ends.tolist(),
            "shared_window_left_samples": float(window_left[0, 0]),
            "shared_window_right_samples": float(window_right[0, 0]),
        },
        "sampled_numeric_audit": {
            "raw_integer_min": result.raw_min,
            "raw_integer_max": result.raw_max,
            "normalized_min": float(x.min()),
            "normalized_max": float(x.max()),
            "normalized_mean": float(x.mean()),
            "normalized_rms": float(torch.sqrt(torch.mean(x.square()))),
            "all_values_finite": bool(torch.isfinite(x).all()),
        },
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "compute_ledger_present": False,
        },
        "source_repository": _git_state(repository),
        "source_sha256": {
            "real_rf.py": _sha256(
                repository / "src" / "a3_ultrasound" / "data" / "real_rf.py"
            ),
            "collate.py": _sha256(
                repository / "src" / "a3_ultrasound" / "data" / "collate.py"
            ),
            "build_m2_01_real_rf_batch.py": _sha256(Path(__file__).resolve()),
            manifest_path.name: _sha256(manifest_path),
            metadata_schema_path.name: _sha256(metadata_schema_path),
        },
        "interpretation_boundary": (
            "PASS proves strict parsing, one fixed batch-level amplitude divisor, "
            "v3 metadata derivation, and label-free tensor shape/dtype/mask contracts "
            "for five spatially distributed real traces. It does not prove full-scan "
            "integrity, real-RF model forward, training stability, performance, or mechanism."
        ),
    }
    numeric_values = payload["sampled_numeric_audit"].values()
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in numeric_values
    ):
        raise RuntimeError("M2-01 numeric audit contains a non-finite value.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        metadata_schema_path=args.metadata_schema,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "status": payload["status"],
                "trace_count": payload["selection"]["trace_count"],
                "x_raw_shape": payload["tensor_contract"]["x_raw"]["shape"],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
