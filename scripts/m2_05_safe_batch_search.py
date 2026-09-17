"""Search a conservative float32 PCC batch under the 8 GB GPU budget."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
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
from m2_04_gpu_profile import _mib, _nvidia_smi_snapshot
from m2_real_rf_forward_smoke import _build_model, _git_state, _sha256


SEED = 3116
TRACE_LENGTH = 2300
MIN_FREE_FRACTION_AT_PEAK = 0.15
MAX_BATCH_SIZE = 2048


def _make_batch(
    batch_size: int,
    interface_count: int,
    device: torch.device,
) -> tuple[InterfaceBatch, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + batch_size * 10 + interface_count)
    x_raw = torch.randn(
        (batch_size, TRACE_LENGTH),
        generator=generator,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    length_mask = torch.ones(
        (batch_size, TRACE_LENGTH),
        dtype=torch.bool,
        device=device,
    )
    interface_mask = torch.zeros(
        (batch_size, 3),
        dtype=torch.bool,
        device=device,
    )
    interface_mask[:, :interface_count] = True
    interface_meta = torch.zeros(
        (batch_size, 3, 5),
        dtype=torch.float32,
        device=device,
    )
    nominal_tofs = (245.026, 455.553, 666.079)
    one_trace_meta = torch.zeros((3, 5), dtype=torch.float32, device=device)
    for interface_index in range(interface_count):
        one_trace_meta[interface_index] = torch.tensor(
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
    interface_meta[:] = one_trace_meta
    labels = torch.zeros(
        (batch_size, 3),
        dtype=torch.int64,
        device=device,
    )
    labels[:, 1] = 1
    if interface_count == 3:
        labels[1::2, 2] = 1
    return (
        InterfaceBatch(
            x_raw=x_raw,
            length_mask=length_mask,
            interface_meta=interface_meta,
            interface_mask=interface_mask,
        ),
        labels,
    )


def _step(
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


def _profile_once(
    batch_size: int,
    interface_count: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = _build_model(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, labels = _make_batch(batch_size, interface_count, device)

    _step(model, optimizer, batch, labels)
    torch.cuda.synchronize(device)
    optimizer.zero_grad(set_to_none=True)
    batch.x_raw.grad = None
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    output, loss = _step(model, optimizer, batch, labels)
    end_event.record()
    torch.cuda.synchronize(device)
    step_cuda_ms = float(start_event.elapsed_time(end_event))
    step_wall_ms = (time.perf_counter() - wall_started) * 1000.0

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_delta = max(0, peak_allocated - baseline_allocated)
    estimated_free_at_peak = max(0, free_before - peak_delta)
    estimated_free_fraction_at_peak = estimated_free_at_peak / total_memory
    free_after, _ = torch.cuda.mem_get_info(device)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    checks = {
        "loss_finite": math.isfinite(float(loss.total.detach())),
        "valid_logits_finite": bool(
            torch.isfinite(output.conditional.logits[batch.interface_mask]).all()
        ),
        "parameter_gradients_present": bool(gradients),
        "parameter_gradients_finite": all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "raw_input_gradient_finite": (
            batch.x_raw.grad is not None
            and bool(torch.isfinite(batch.x_raw.grad).all())
        ),
        "step_time_positive": step_cuda_ms > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"M2-05 batch={batch_size}, K={interface_count} checks failed: {checks}"
        )

    return {
        "batch_size": batch_size,
        "interface_count": interface_count,
        "status": "PASS",
        "safe_under_15_percent_rule": (
            estimated_free_fraction_at_peak >= MIN_FREE_FRACTION_AT_PEAK
        ),
        "metrics": {
            "total_loss": float(loss.total.detach()),
            "step_cuda_ms": step_cuda_ms,
            "step_wall_ms": step_wall_ms,
            "baseline_allocated_mib": _mib(baseline_allocated),
            "peak_allocated_mib": _mib(peak_allocated),
            "peak_allocated_delta_mib": _mib(peak_delta),
            "baseline_reserved_mib": _mib(baseline_reserved),
            "peak_reserved_mib": _mib(peak_reserved),
            "device_free_before_mib": _mib(free_before),
            "estimated_device_free_at_peak_mib": _mib(estimated_free_at_peak),
            "estimated_device_free_fraction_at_peak": (
                estimated_free_fraction_at_peak
            ),
            "device_free_after_mib": _mib(free_after),
            "device_total_memory_mib": _mib(total_memory),
        },
        "checks": checks,
    }


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _try_profile(
    batch_size: int,
    interface_count: int,
    device: torch.device,
) -> dict[str, Any]:
    try:
        return _profile_once(batch_size, interface_count, device)
    except torch.OutOfMemoryError as exc:
        return {
            "batch_size": batch_size,
            "interface_count": interface_count,
            "status": "OOM",
            "safe_under_15_percent_rule": False,
            "error_type": type(exc).__name__,
        }
    finally:
        _cleanup_cuda()


def run(output_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered A3 environment.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _cleanup_cuda()
    preflight = _nvidia_smi_snapshot()

    k3_search: list[dict[str, Any]] = []
    safe_k3_batches: list[int] = []
    batch_size = 1
    boundary_observed = False
    while batch_size <= MAX_BATCH_SIZE:
        row = _try_profile(batch_size, 3, device)
        k3_search.append(row)
        if row["status"] == "PASS" and row["safe_under_15_percent_rule"]:
            safe_k3_batches.append(batch_size)
            batch_size *= 2
            continue
        boundary_observed = True
        break

    if not safe_k3_batches:
        raise RuntimeError("M2-05 found no K=3 batch satisfying the 15% rule.")
    if not boundary_observed:
        raise RuntimeError(
            "M2-05 reached MAX_BATCH_SIZE without observing an unsafe/OOM boundary."
        )

    k2_validation: list[dict[str, Any]] = []
    selected_batch = safe_k3_batches[-1]
    while selected_batch >= 1:
        row = _try_profile(selected_batch, 2, device)
        k2_validation.append(row)
        if row["status"] == "PASS" and row["safe_under_15_percent_rule"]:
            break
        selected_batch //= 2
    if selected_batch < 1:
        raise RuntimeError("M2-05 found no shared K=2/3 safe batch.")

    selected_k3_row = next(
        row
        for row in k3_search
        if row["batch_size"] == selected_batch and row["status"] == "PASS"
    )
    selected_k2_row = next(
        row
        for row in k2_validation
        if row["batch_size"] == selected_batch and row["status"] == "PASS"
    )
    checks = {
        "unsafe_or_oom_boundary_observed": boundary_observed,
        "shared_safe_batch_found": selected_batch >= 1,
        "selected_K3_reserve_at_least_15_percent": (
            selected_k3_row["metrics"]["estimated_device_free_fraction_at_peak"]
            >= MIN_FREE_FRACTION_AT_PEAK
        ),
        "selected_K2_reserve_at_least_15_percent": (
            selected_k2_row["metrics"]["estimated_device_free_fraction_at_peak"]
            >= MIN_FREE_FRACTION_AT_PEAK
        ),
        "selected_batch_has_finite_K2_K3_steps": all(
            all(row["checks"].values()) for row in (selected_k2_row, selected_k3_row)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-05 checks failed: {checks}")

    repository = Path(__file__).resolve().parents[1]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "a3-m2-05-safe-batch-search-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-05",
        "status": "PASS",
        "evidence_class": "synthetic_runtime_budget_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
        },
        "resolved_config": {
            "seed": SEED,
            "trace_length": TRACE_LENGTH,
            "precision": "float32",
            "amp_enabled": False,
            "batch_search": "K3 doubling then K2 validation at shared candidate",
            "minimum_free_fraction_at_peak": MIN_FREE_FRACTION_AT_PEAK,
            "maximum_batch_size_considered": MAX_BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "lambda_local": 0.3,
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
        "profile_matrix": {
            "K3_doubling_search": k3_search,
            "K2_selected_batch_validation": k2_validation,
        },
        "selection": {
            "shared_safe_batch_candidate": selected_batch,
            "selected_K3_estimated_free_fraction_at_peak": selected_k3_row[
                "metrics"
            ]["estimated_device_free_fraction_at_peak"],
            "selected_K2_estimated_free_fraction_at_peak": selected_k2_row[
                "metrics"
            ]["estimated_device_free_fraction_at_peak"],
            "next_unsafe_or_oom_K3_batch": k3_search[-1]["batch_size"],
            "next_unsafe_or_oom_K3_status": k3_search[-1]["status"],
        },
        "checks": checks,
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
            "m2_04_gpu_profile.py": _sha256(
                repository / "scripts" / "m2_04_gpu_profile.py"
            ),
            "m2_05_safe_batch_search.py": _sha256(Path(__file__).resolve()),
        },
        "interpretation_boundary": (
            "PASS identifies a conservative float32 synthetic batch candidate "
            "under the observed GPU state. It is not yet the M2-06 runtime "
            "freeze, does not cover AMP or matched-joint parity, and provides "
            "no RF performance, training-stability, or mechanism evidence."
        ),
    }
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
                "shared_safe_batch_candidate": result["selection"][
                    "shared_safe_batch_candidate"
                ],
                "next_K3_batch": result["selection"][
                    "next_unsafe_or_oom_K3_batch"
                ],
                "next_K3_status": result["selection"][
                    "next_unsafe_or_oom_K3_status"
                ],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
