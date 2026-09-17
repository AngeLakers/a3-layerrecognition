"""Run the M2-02 label-free real-RF PCC forward-only smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from a3_ultrasound.data import (
    build_label_free_real_rf_batch,
    load_label_free_network_manifest,
)
from a3_ultrasound.models import A3PCCModel, LengthGroupedTCNBackbone
from a3_ultrasound.schema import InterfaceBatch


SEED = 3114


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


def _build_model(device: torch.device) -> A3PCCModel:
    """Use the frozen M1 reference architecture without loading a checkpoint."""

    return A3PCCModel(
        backbone=LengthGroupedTCNBackbone(
            channels=(16, 32, 32),
            kernel_size=5,
            dilations=(1, 2, 4),
            dropout=0.0,
            use_norm="weight_norm",
        ),
        query_dim=32,
        attention_dim=64,
        token_dim=32,
        prefix_dim=32,
        pcc_hidden_dim=64,
        max_interfaces=6,
        rho=1.0,
    ).to(device)


def _to_device(batch: InterfaceBatch, device: torch.device) -> InterfaceBatch:
    return InterfaceBatch(
        x_raw=batch.x_raw.to(device),
        length_mask=batch.length_mask.to(device),
        interface_meta=batch.interface_meta.to(device),
        interface_mask=batch.interface_mask.to(device),
    )


def _max_abs(values: torch.Tensor) -> float:
    return float(values.abs().max()) if values.numel() else 0.0


def _attention_error(
    attention: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, float, float]:
    """Return normalization, outside-support, and invalid-slot errors."""

    has_support = support.any(dim=-1)
    normalized = valid & has_support
    sums = attention.sum(dim=-1)
    normalization_error = _max_abs(sums[normalized] - 1.0)
    outside_support_error = _max_abs(attention[~support])
    invalid_slot_error = _max_abs(attention[~valid])
    return normalization_error, outside_support_error, invalid_slot_error


def run(
    *,
    raw_root: Path,
    manifest_path: Path,
    metadata_schema_path: Path,
    output_path: Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run inference only; labels, gradients, loss, and optimizer are absent."""

    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    manifest = load_label_free_network_manifest(manifest_path, metadata_schema_path)
    loaded = build_label_free_real_rf_batch(raw_root, manifest)
    if loaded.batch.y is not None or loaded.batch.label_mask is not None:
        raise RuntimeError("M2-02 must receive a label-free real-RF batch.")
    batch = _to_device(loaded.batch, device)
    model = _build_model(device).eval()
    prefix_states = torch.zeros_like(batch.interface_mask, dtype=torch.long)

    with torch.inference_mode():
        inference_mode_enabled = not torch.is_grad_enabled()
        output = model(batch, prefix_states)
        exact = model.infer_exact(batch)

    valid = batch.interface_mask
    invalid = ~valid
    local_norm_error, local_outside_error, local_invalid_error = _attention_error(
        output.local_readout.attention,
        output.local_support,
        valid,
    )
    pre_norm_error, pre_outside_error, pre_invalid_error = _attention_error(
        output.pre_readout.attention,
        output.pre_support,
        valid,
    )
    exact_probability_error = _max_abs(
        (exact.probabilities * exact.path_mask).sum(dim=1) - 1.0
    )
    invalid_token_max = max(
        _max_abs(getattr(output.responsibility_tokens, name)[invalid])
        for name in ("r_loc", "r_ctx", "r_gate")
    )
    invalid_logit_max = _max_abs(output.conditional.logits[invalid])
    invalid_marginal_max = _max_abs(exact.marginals[invalid])
    invalid_map_state_ok = bool(torch.all(exact.map_states[invalid] == -1))
    valid_local_support = output.local_support[valid]
    valid_pre_support = output.pre_support[valid]
    feature_mask = batch.length_mask.unsqueeze(1)

    checks = {
        "label_free_batch": batch.y is None and batch.label_mask is None,
        "all_network_outputs_finite": all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                output.query,
                output.conditional.logits,
                output.conditional.base_logodds,
                output.conditional.correction,
                output.conditional.gate,
                exact.probabilities,
                exact.marginals,
            )
        ),
        "valid_logits_finite": bool(torch.isfinite(output.conditional.logits[valid]).all()),
        "local_support_nonempty_for_valid_interfaces": bool(valid_local_support.any(dim=-1).all()),
        "supports_within_valid_trace": not bool(
            (output.local_support & ~feature_mask).any()
            or (output.pre_support & ~feature_mask).any()
        ),
        "invalid_slots_have_no_support": not bool(
            output.local_support[invalid].any() or output.pre_support[invalid].any()
        ),
        "local_attention_normalized": local_norm_error < 1e-5,
        "pre_attention_normalized_where_supported": pre_norm_error < 1e-5,
        "local_attention_within_support": local_outside_error == 0.0,
        "pre_attention_within_support": pre_outside_error == 0.0,
        "invalid_attention_zero": local_invalid_error == 0.0 and pre_invalid_error == 0.0,
        "invalid_responsibility_tokens_zero": invalid_token_max == 0.0,
        "invalid_logits_zero": invalid_logit_max == 0.0,
        "exact_path_probabilities_normalized": exact_probability_error < 1e-5,
        "invalid_marginals_zero": invalid_marginal_max == 0.0,
        "invalid_map_states_are_sentinel": invalid_map_state_ok,
        "no_gradient_tracking": inference_mode_enabled,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-02 real-RF forward checks failed: {checks}")

    repository = Path(__file__).resolve().parents[1]
    valid_pre_count = int(valid_pre_support.any(dim=-1).sum())
    payload = {
        "schema_version": "a3-m2-02-real-rf-forward-smoke-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-02",
        "status": "PASS",
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
        "resolved_config": {
            "seed": SEED,
            "device": str(device),
            "model_weight_state": "seeded_random_initialization_no_checkpoint",
            "batch_size": int(batch.x_raw.shape[0]),
            "trace_length": int(batch.x_raw.shape[1]),
            "valid_lengths": batch.length_mask.sum(dim=1).tolist(),
            "valid_interface_counts": batch.interface_mask.sum(dim=1).tolist(),
            "backbone_channels": [16, 32, 32],
            "kernel_size": 5,
            "dilations": [1, 2, 4],
            "query_dim": 32,
            "attention_dim": 64,
            "token_dim": 32,
            "prefix_dim": 32,
            "pcc_hidden_dim": 64,
            "rho": 1.0,
        },
        "registered_contract": {
            "dataset_id": manifest["identity"]["dataset_id"],
            "acquisition_batch_id": manifest["identity"]["acquisition_batch_id"],
            "manifest_version": manifest.get("manifest_version"),
            "sampling_rate_hz": manifest["acquisition"]["sampling_rate_hz"],
            "raw_unit": manifest["raw_layout"]["raw_unit"],
            "K": manifest["structure"]["K"],
            "labels_absent": True,
        },
        "input_fingerprint": loaded.input_fingerprint(),
        "support_audit": {
            "valid_local_support_count": int(valid_local_support.any(dim=-1).sum()),
            "valid_pre_support_count": valid_pre_count,
            "local_attention_max_abs_normalization_error": local_norm_error,
            "pre_attention_max_abs_normalization_error": pre_norm_error,
            "local_attention_max_abs_outside_support": local_outside_error,
            "pre_attention_max_abs_outside_support": pre_outside_error,
            "invalid_attention_max_abs": max(local_invalid_error, pre_invalid_error),
        },
        "output_audit": {
            "valid_logit_min": float(output.conditional.logits[valid].min()),
            "valid_logit_max": float(output.conditional.logits[valid].max()),
            "exact_probability_max_abs_error": exact_probability_error,
            "invalid_token_max_abs": invalid_token_max,
            "invalid_logit_max_abs": invalid_logit_max,
            "invalid_marginal_max_abs": invalid_marginal_max,
            "invalid_map_state_sentinel": -1,
        },
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_type": device.type,
            "compute_ledger_present": False,
        },
        "source_repository": _git_state(repository),
        "source_sha256": {
            "real_rf.py": _sha256(repository / "src" / "a3_ultrasound" / "data" / "real_rf.py"),
            "backbone.py": _sha256(repository / "src" / "a3_ultrasound" / "models" / "backbone.py"),
            "integrated.py": _sha256(repository / "src" / "a3_ultrasound" / "models" / "integrated.py"),
            "m2_real_rf_forward_smoke.py": _sha256(Path(__file__).resolve()),
            manifest_path.name: _sha256(manifest_path),
            metadata_schema_path.name: _sha256(metadata_schema_path),
        },
        "interpretation_boundary": (
            "PASS proves that five selected label-free real RF traces can pass through "
            "the seeded PCC reference architecture and exact enumerator with finite "
            "outputs, support/mask isolation, and normalized attention. It does not "
            "prove model performance, training stability, baseline fairness, or the "
            "propagation-order mechanism."
        ),
    }
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        metadata_schema_path=args.metadata_schema,
        output_path=args.output,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "device": result["resolved_config"]["device"],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
