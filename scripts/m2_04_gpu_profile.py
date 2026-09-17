"""Profile batch=1 K=2/3 PCC forward/backward steps on the registered GPU."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
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

from a3_ultrasound.models import pcc_training_loss
from a3_ultrasound.schema import InterfaceBatch
from m2_real_rf_forward_smoke import _build_model, _git_state, _sha256


SEED = 3115
MIB = 1024**2
TRACE_LENGTH = 2300
WARMUP_STEPS = 2
MEASURED_STEPS = 10


def _mib(value: int) -> float:
    return value / MIB


def _max_abs(values: torch.Tensor) -> float:
    return float(values.detach().abs().max()) if values.numel() else 0.0


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
        return {
            "available": False,
            "returncode": completed.returncode,
        }
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        return {
            "available": False,
            "returncode": completed.returncode,
            "field_count": len(fields),
        }
    return {
        "available": True,
        "gpu_index": int(fields[0]),
        "device": fields[1],
        "memory_used_mib": float(fields[2]),
        "memory_total_mib": float(fields[3]),
        "utilization_percent": float(fields[4]),
        "free_gpu_threshold_mib": 500.0,
        "meets_free_gpu_threshold": float(fields[2]) < 500.0,
    }


def _make_batch(interface_count: int, device: torch.device) -> tuple[InterfaceBatch, torch.Tensor]:
    if interface_count not in {2, 3}:
        raise ValueError("M2-04 profiles only K=2 and K=3.")

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + interface_count)
    x_raw = torch.randn(
        (1, TRACE_LENGTH),
        generator=generator,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    length_mask = torch.ones(
        (1, TRACE_LENGTH),
        dtype=torch.bool,
        device=device,
    )
    interface_mask = torch.zeros((1, 3), dtype=torch.bool, device=device)
    interface_mask[:, :interface_count] = True
    interface_meta = torch.zeros((1, 3, 5), dtype=torch.float32, device=device)
    nominal_tofs = (245.026, 455.553, 666.079)
    for interface_index in range(interface_count):
        interface_meta[0, interface_index] = torch.tensor(
            [
                (interface_index + 1) / interface_count,
                nominal_tofs[interface_index] / TRACE_LENGTH,
                (interface_index + 1) / (interface_count + 1),
                88.0 / TRACE_LENGTH,
                104.0 / TRACE_LENGTH,
            ],
            dtype=torch.float32,
            device=device,
        )
    labels = torch.tensor([[0, 1, 0]], dtype=torch.int64, device=device)
    return (
        InterfaceBatch(
            x_raw=x_raw,
            length_mask=length_mask,
            interface_meta=interface_meta,
            interface_mask=interface_mask,
        ),
        labels,
    )


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: InterfaceBatch,
    labels: torch.Tensor,
) -> tuple[Any, Any]:
    optimizer.zero_grad(set_to_none=True)
    batch.x_raw.grad = None
    output = model(batch, labels)
    loss = pcc_training_loss(
        output.conditional.logits,
        output.conditional.base_logodds,
        labels,
        batch.interface_mask,
        lambda_local=0.3,
    )
    loss.total.backward()
    optimizer.step()
    return output, loss


def _profile_interface_count(
    interface_count: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = _build_model(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, labels = _make_batch(interface_count, device)

    for _ in range(WARMUP_STEPS):
        _run_step(model, optimizer, batch, labels)
    torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    batch.x_raw.grad = None
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    cuda_times_ms: list[float] = []
    wall_times_ms: list[float] = []
    final_output = None
    final_loss = None
    for _ in range(MEASURED_STEPS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record()
        final_output, final_loss = _run_step(model, optimizer, batch, labels)
        end_event.record()
        torch.cuda.synchronize(device)
        cuda_times_ms.append(float(start_event.elapsed_time(end_event)))
        wall_times_ms.append((time.perf_counter() - wall_started) * 1000.0)

    if final_output is None or final_loss is None:
        raise RuntimeError("M2-04 measured no profile steps.")
    if batch.x_raw.grad is None:
        raise RuntimeError("M2-04 synthetic input gradient is missing.")

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    valid = batch.interface_mask
    invalid = ~valid
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    invalid_token_max = max(
        _max_abs(getattr(final_output.responsibility_tokens, name)[invalid])
        for name in ("r_loc", "r_ctx", "r_gate")
    )
    invalid_logit_max = _max_abs(final_output.conditional.logits[invalid])
    local_attention_error = _max_abs(
        final_output.local_readout.attention.sum(dim=-1)[valid] - 1.0
    )
    pre_has_support = final_output.pre_support.any(dim=-1)
    pre_valid = valid & pre_has_support
    pre_attention_error = _max_abs(
        final_output.pre_readout.attention.sum(dim=-1)[pre_valid] - 1.0
    )
    checks = {
        "loss_finite": math.isfinite(float(final_loss.total.detach())),
        "valid_logits_finite": bool(
            torch.isfinite(final_output.conditional.logits[valid]).all()
        ),
        "parameter_gradients_present": bool(parameter_gradients),
        "parameter_gradients_finite": all(
            bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients
        ),
        "raw_input_gradient_finite": bool(torch.isfinite(batch.x_raw.grad).all()),
        "invalid_responsibility_tokens_zero": invalid_token_max == 0.0,
        "invalid_logits_zero": invalid_logit_max == 0.0,
        "local_attention_normalized": local_attention_error < 1e-5,
        "pre_attention_normalized_where_supported": pre_attention_error < 1e-5,
        "cuda_times_positive": all(value > 0.0 for value in cuda_times_ms),
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-04 K={interface_count} checks failed: {checks}")

    return {
        "interface_count": interface_count,
        "batch_size": 1,
        "trace_length": TRACE_LENGTH,
        "precision": "float32",
        "amp_enabled": False,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "metrics": {
            "last_total_loss": float(final_loss.total.detach()),
            "last_chain_loss": float(final_loss.chain.detach()),
            "last_local_loss": float(final_loss.local.detach()),
            "step_cuda_ms_median": statistics.median(cuda_times_ms),
            "step_cuda_ms_mean": statistics.fmean(cuda_times_ms),
            "step_cuda_ms_min": min(cuda_times_ms),
            "step_cuda_ms_max": max(cuda_times_ms),
            "step_wall_ms_median": statistics.median(wall_times_ms),
            "baseline_allocated_mib": _mib(baseline_allocated),
            "peak_allocated_mib": _mib(peak_allocated),
            "peak_allocated_delta_mib": _mib(
                max(0, peak_allocated - baseline_allocated)
            ),
            "baseline_reserved_mib": _mib(baseline_reserved),
            "peak_reserved_mib": _mib(peak_reserved),
            "device_free_before_mib": _mib(free_before),
            "device_free_after_mib": _mib(free_after),
            "device_total_memory_mib": _mib(total_memory),
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_tensors_with_grad": len(parameter_gradients),
            "local_attention_max_abs_error": local_attention_error,
            "pre_attention_max_abs_error": pre_attention_error,
            "invalid_token_max_abs": invalid_token_max,
            "invalid_logit_max_abs": invalid_logit_max,
        },
        "checks": checks,
    }


def run(output_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered A3 environment.")
    preflight = _nvidia_smi_snapshot()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    profiles: dict[str, Any] = {}
    for interface_count in (2, 3):
        profiles[f"K{interface_count}"] = _profile_interface_count(
            interface_count,
            device,
        )
        torch.cuda.empty_cache()

    all_checks_pass = all(
        all(profile["checks"].values()) for profile in profiles.values()
    )
    repository = Path(__file__).resolve().parents[1]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "a3-m2-04-gpu-profile-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-04",
        "status": "PASS" if all_checks_pass else "FAIL",
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
        "resolved_config": {
            "seed": SEED,
            "device": "cuda:0",
            "interface_counts_profiled_separately": [2, 3],
            "batch_size_per_profile": 1,
            "trace_length": TRACE_LENGTH,
            "backbone_channels": [16, 32, 32],
            "kernel_size": 5,
            "dilations": [1, 2, 4],
            "query_dim": 32,
            "attention_dim": 64,
            "token_dim": 32,
            "prefix_dim": 32,
            "pcc_hidden_dim": 64,
            "rho": 1.0,
            "lambda_local": 0.3,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": properties.name,
            "device_total_memory_mib": _mib(properties.total_memory),
            "compute_ledger_present": False,
        },
        "preflight": preflight,
        "profiles": profiles,
        "checks": {
            "K2_profile_pass": all(profiles["K2"]["checks"].values()),
            "K3_profile_pass": all(profiles["K3"]["checks"].values()),
            "both_profiles_batch_size_one": all(
                profile["batch_size"] == 1 for profile in profiles.values()
            ),
            "both_profiles_float32_without_amp": all(
                profile["precision"] == "float32"
                and profile["amp_enabled"] is False
                for profile in profiles.values()
            ),
        },
        "source_repository": _git_state(repository),
        "source_sha256": {
            "integrated.py": _sha256(
                repository / "src" / "a3_ultrasound" / "models" / "integrated.py"
            ),
            "backbone.py": _sha256(
                repository / "src" / "a3_ultrasound" / "models" / "backbone.py"
            ),
            "losses.py": _sha256(
                repository / "src" / "a3_ultrasound" / "models" / "losses.py"
            ),
            "m2_real_rf_forward_smoke.py": _sha256(
                repository / "scripts" / "m2_real_rf_forward_smoke.py"
            ),
            "m2_04_gpu_profile.py": _sha256(Path(__file__).resolve()),
        },
        "interpretation_boundary": (
            "PASS records batch=1 K=2/K=3 float32 synthetic runtime and memory "
            "for the reference PCC implementation. It does not freeze a safe "
            "batch size, AMP, gradient accumulation, RF performance, training "
            "stability, model selection, or propagation-mechanism evidence."
        ),
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"M2-04 top-level checks failed: {result['checks']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "K2_peak_allocated_mib": result["profiles"]["K2"]["metrics"][
                    "peak_allocated_mib"
                ],
                "K3_peak_allocated_mib": result["profiles"]["K3"]["metrics"][
                    "peak_allocated_mib"
                ],
                "K2_step_cuda_ms_median": result["profiles"]["K2"]["metrics"][
                    "step_cuda_ms_median"
                ],
                "K3_step_cuda_ms_median": result["profiles"]["K3"]["metrics"][
                    "step_cuda_ms_median"
                ],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
