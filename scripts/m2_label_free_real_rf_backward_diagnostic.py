"""Run a target-free backward diagnostic on the M2-02 real-RF batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
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
from a3_ultrasound.diagnostics import (
    build_label_free_autograd_probe,
    summarize_parameter_gradients,
)
from a3_ultrasound.schema import InterfaceBatch
from m2_real_rf_forward_smoke import _build_model, _git_state, _sha256


SEED = 3114
MIB = 1024**2
GRADIENT_GROUP_PREFIXES = {
    "backbone": ("backbone.",),
    "backbone_adapter": ("backbone_adapter.",),
    "query_encoder": ("query_encoder.",),
    "local_readout": ("local_readout.",),
    "pre_readout": ("pre_readout.",),
    "responsibility_projector": ("responsibility_projector.",),
    "pcc_base": ("pcc.base.",),
    "pcc_prefix": ("pcc.prefix_encoder.",),
    "pcc_correction": ("pcc.correction.",),
    "pcc_gate": ("pcc.gate.",),
}


def _nvidia_smi_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False, "returncode": completed.returncode}
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        return {"available": False, "field_count": len(fields)}
    used_mib = float(fields[2])
    return {
        "available": True,
        "gpu_index": int(fields[0]),
        "device": fields[1],
        "memory_used_mib": used_mib,
        "memory_total_mib": float(fields[3]),
        "utilization_percent": float(fields[4]),
        "free_gpu_threshold_mib": 500.0,
        "meets_free_gpu_threshold": used_mib < 500.0,
    }


def _to_device_with_input_gradient(
    batch: InterfaceBatch,
    device: torch.device,
) -> InterfaceBatch:
    x_raw = batch.x_raw.to(device).detach().requires_grad_(True)
    return InterfaceBatch(
        x_raw=x_raw,
        length_mask=batch.length_mask.to(device),
        interface_meta=batch.interface_meta.to(device),
        interface_mask=batch.interface_mask.to(device),
    )


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _max_abs(values: torch.Tensor) -> float:
    return float(values.detach().abs().max()) if values.numel() else 0.0


def _attention_normalization_error(
    attention: torch.Tensor,
    support: torch.Tensor,
    interface_mask: torch.Tensor,
) -> float:
    active = interface_mask & support.any(dim=-1)
    return _max_abs(attention.sum(dim=-1)[active] - 1.0)


def run(
    *,
    raw_root: Path,
    manifest_path: Path,
    metadata_schema_path: Path,
    output_path: Path,
    log_path: Path,
    compute_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Run one backward pass with no targets, optimizer, or model update."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered A3 environment.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    preflight = _nvidia_smi_snapshot()

    manifest = load_label_free_network_manifest(manifest_path, metadata_schema_path)
    loaded = build_label_free_real_rf_batch(raw_root, manifest)
    if loaded.batch.y is not None or loaded.batch.label_mask is not None:
        raise RuntimeError("M2-LF-01 must receive a label-free real-RF batch.")
    batch = _to_device_with_input_gradient(loaded.batch, device)
    prefix_states = torch.zeros_like(batch.interface_mask, dtype=torch.long)
    model = _build_model(device).eval()
    model.zero_grad(set_to_none=True)
    state_before = _model_state_sha256(model)

    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    output = model(batch, prefix_states)
    probe = build_label_free_autograd_probe(output, batch.interface_mask)
    probe.scalar.backward()
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))

    state_after = _model_state_sha256(model)
    if batch.x_raw.grad is None:
        raise RuntimeError("The real-RF input gradient is missing.")
    gradient_groups = summarize_parameter_gradients(model, GRADIENT_GROUP_PREFIXES)
    valid_trace = batch.length_mask
    invalid_trace = ~valid_trace
    input_gradient_valid_max = _max_abs(batch.x_raw.grad[valid_trace])
    input_gradient_invalid_max = _max_abs(batch.x_raw.grad[invalid_trace])
    valid = batch.interface_mask
    invalid = ~valid
    local_attention_error = _attention_normalization_error(
        output.local_readout.attention,
        output.local_support,
        valid,
    )
    pre_attention_error = _attention_normalization_error(
        output.pre_readout.attention,
        output.pre_support,
        valid,
    )
    invalid_token_max = max(
        _max_abs(getattr(output.responsibility_tokens, name)[invalid])
        for name in ("r_loc", "r_ctx", "r_gate")
    )
    invalid_logit_max = _max_abs(output.conditional.logits[invalid])
    components = {
        name: float(value.detach()) for name, value in probe.components.items()
    }
    checks = {
        "label_free_batch": batch.y is None and batch.label_mask is None,
        "probe_scalar_finite": math.isfinite(float(probe.scalar.detach())),
        "probe_components_finite": all(math.isfinite(value) for value in components.values()),
        "raw_input_gradient_present": batch.x_raw.grad is not None,
        "raw_input_gradient_finite": bool(torch.isfinite(batch.x_raw.grad).all()),
        "raw_input_gradient_nonzero_on_valid_trace": input_gradient_valid_max > 0.0,
        "raw_input_gradient_zero_on_padding": input_gradient_invalid_max == 0.0,
        "all_required_parameter_groups_present": all(
            group["parameter_tensors"] > 0 for group in gradient_groups.values()
        ),
        "all_required_parameter_groups_have_finite_gradients": all(
            group["all_gradients_finite"] for group in gradient_groups.values()
        ),
        "all_required_parameter_groups_have_nonzero_gradient": all(
            group["nonzero_gradient_present"] for group in gradient_groups.values()
        ),
        "model_state_unchanged": state_before == state_after,
        "valid_logits_finite": bool(torch.isfinite(output.conditional.logits[valid]).all()),
        "local_attention_normalized": local_attention_error < 1e-5,
        "pre_attention_normalized_where_supported": pre_attention_error < 1e-5,
        "invalid_responsibility_tokens_zero": invalid_token_max == 0.0,
        "invalid_logits_zero": invalid_logit_max == 0.0,
        "cuda_time_positive": cuda_ms > 0.0,
        "wall_time_positive": wall_ms > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-LF-01 checks failed: {checks}")

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    compute_ledger_present = bool(
        compute_ledger_path is not None and compute_ledger_path.is_file()
    )
    environment_name = Path(sys.prefix).name
    payload = {
        "schema_version": "a3-m2-label-free-real-rf-autograd-v2",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-LF-01",
        "status": "PASS",
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
        "probe_contract": {
            "name": "deterministic_target_free_autograd_connectivity_scalar",
            "weights": dict(probe.weights),
            "components": components,
            "scalar": float(probe.scalar.detach()),
            "is_training_loss": False,
            "is_model_selection_metric": False,
            "has_physical_interpretation": False,
        },
        "resolved_config": {
            "seed": SEED,
            "device": str(device),
            "precision": "float32",
            "amp_enabled": False,
            "model_mode": "eval_with_autograd_enabled",
            "model_weight_state": "seeded_random_initialization_no_checkpoint",
            "prefix_state": "all_zero_deterministic_reference_not_ground_truth",
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
            "K": manifest["structure"]["K"],
            "labels_absent": True,
        },
        "input_fingerprint": loaded.input_fingerprint(),
        "gradient_audit": {
            "input_gradient_valid_max_abs": input_gradient_valid_max,
            "input_gradient_invalid_max_abs": input_gradient_invalid_max,
            "parameter_groups": gradient_groups,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
        },
        "support_audit": {
            "local_attention_max_abs_normalization_error": local_attention_error,
            "pre_attention_max_abs_normalization_error": pre_attention_error,
            "invalid_token_max_abs": invalid_token_max,
            "invalid_logit_max_abs": invalid_logit_max,
        },
        "runtime": {
            "cuda_forward_backward_ms": cuda_ms,
            "wall_forward_backward_ms": wall_ms,
            "baseline_allocated_mib": baseline_allocated / MIB,
            "peak_allocated_mib": peak_allocated / MIB,
            "peak_allocated_delta_mib": max(0, peak_allocated - baseline_allocated) / MIB,
            "baseline_reserved_mib": baseline_reserved / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            "device_free_before_mib": free_before / MIB,
            "device_free_after_mib": free_after / MIB,
            "device_total_memory_mib": total_memory / MIB,
            "preflight": preflight,
            "timing_is_isolated_benchmark": False,
        },
        "checks": checks,
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
            "result_name": output_path.name,
            "companion_log_name": log_path.name,
            "companion_log_generated_by_runner": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_type": device.type,
            "registered_conda_environment": "aislee-a3-layercongition",
            "resolved_environment_name": environment_name,
            "registered_environment_matches": environment_name
            == "aislee-a3-layercongition",
            "pythonpath_environment_variable_present": bool(os.environ.get("PYTHONPATH")),
            "compute_ledger_present": compute_ledger_present,
        },
        "source_repository": _git_state(REPOSITORY_ROOT),
        "source_sha256": {
            "diagnostics.py": _sha256(SOURCE_ROOT / "a3_ultrasound" / "diagnostics.py"),
            "real_rf.py": _sha256(SOURCE_ROOT / "a3_ultrasound" / "data" / "real_rf.py"),
            "backbone.py": _sha256(SOURCE_ROOT / "a3_ultrasound" / "models" / "backbone.py"),
            "integrated.py": _sha256(SOURCE_ROOT / "a3_ultrasound" / "models" / "integrated.py"),
            "m2_real_rf_forward_smoke.py": _sha256(
                REPOSITORY_ROOT / "scripts" / "m2_real_rf_forward_smoke.py"
            ),
            "m2_label_free_real_rf_backward_diagnostic.py": _sha256(Path(__file__).resolve()),
            manifest_path.name: _sha256(manifest_path),
            metadata_schema_path.name: _sha256(metadata_schema_path),
        },
        "interpretation_boundary": (
            "PASS proves only that one deterministic target-free scalar can backpropagate "
            "through every registered PCC branch on the same five sampled real-RF traces "
            "used by M2-02, with finite gradients and no parameter update. It is not a "
            "training loss, learned result, performance metric, baseline comparison, "
            "specimen-level claim, or propagation-mechanism evidence."
        ),
    }
    if not payload["environment"]["registered_environment_matches"]:
        raise RuntimeError("The runner is not using the registered A3 Conda environment.")
    summary = json.dumps(
        {
            "run_id": payload["run_id"],
            "status": payload["status"],
            "device": payload["resolved_config"]["device"],
            "output": output_path.name,
        },
        ensure_ascii=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(summary + "\n", encoding="utf-8")
    payload["artifact_contract"]["companion_log_sha256"] = _sha256(log_path)
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
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--compute-ledger", type=Path)
    args = parser.parse_args()
    result = run(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        metadata_schema_path=args.metadata_schema,
        output_path=args.output,
        log_path=args.log,
        compute_ledger_path=args.compute_ledger,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "device": result["resolved_config"]["device"],
                "output": args.output.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
