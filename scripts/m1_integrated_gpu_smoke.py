"""M1-13 integrated synthetic CUDA forward/backward acceptance smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from a3_ultrasound.models import (
    A3PCCModel,
    LengthGroupedTCNBackbone,
    pcc_training_loss,
)
from a3_ultrasound.schema import InterfaceBatch


SEED = 3113
MIB = 1024**2


def _mib(value: int) -> float:
    return value / MIB


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


def _make_batch(device: torch.device) -> tuple[InterfaceBatch, torch.Tensor]:
    trace_length = 2300
    valid_lengths = torch.tensor([2300, 1700, 2300, 1700], device=device)
    sample_index = torch.arange(trace_length, device=device).unsqueeze(0)
    length_mask = sample_index < valid_lengths.unsqueeze(1)
    x_raw = torch.randn((4, trace_length), device=device, requires_grad=True)
    with torch.no_grad():
        x_raw[~length_mask] = 0.0

    interface_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
        device=device,
    )
    interface_meta = torch.zeros((4, 3, 5), dtype=torch.float32, device=device)
    nominal_tofs = torch.tensor([245.026, 455.553, 666.079], device=device)
    half_window = 96.0
    for batch_index, valid_length in enumerate(valid_lengths.tolist()):
        k_i = int(interface_mask[batch_index].sum().item())
        for interface_index in range(k_i):
            interface_meta[batch_index, interface_index] = torch.tensor(
                [
                    (interface_index + 1) / k_i,
                    nominal_tofs[interface_index].item() / valid_length,
                    (interface_index + 1) / (k_i + 1),
                    half_window / valid_length,
                    half_window / valid_length,
                ],
                device=device,
            )
    labels = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=torch.int64,
        device=device,
    )
    return (
        InterfaceBatch(
            x_raw=x_raw,
            length_mask=length_mask,
            interface_meta=interface_meta,
            interface_mask=interface_mask,
        ),
        labels,
    )


def _build_model(device: torch.device) -> A3PCCModel:
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


def run(output_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered A3 environment.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = _build_model(device).train()
    batch, labels = _make_batch(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with torch.no_grad():
        warmup = model(batch, labels)
        if not torch.isfinite(warmup.conditional.logits[batch.interface_mask]).all():
            raise RuntimeError("Warm-up produced non-finite valid logits.")
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
    output = model(batch, labels)
    loss = pcc_training_loss(
        output.conditional.logits,
        output.conditional.base_logodds,
        labels,
        batch.interface_mask,
        lambda_local=0.3,
    )
    loss.total.backward()
    end_event.record()
    torch.cuda.synchronize(device)
    forward_backward_ms = start_event.elapsed_time(end_event)
    optimizer.step()
    torch.cuda.synchronize(device)
    wall_time_s = time.perf_counter() - wall_started

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    valid = batch.interface_mask
    invalid = ~valid
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    x_gradient = batch.x_raw.grad
    if x_gradient is None:
        raise RuntimeError("Raw synthetic input gradient is missing.")

    with torch.no_grad():
        exact = model.infer_exact(batch)
        probability_sums = (exact.probabilities * exact.path_mask).sum(dim=1)
        probability_error = float((probability_sums - 1.0).abs().max())
        local_attention_error = float(
            (
                output.local_readout.attention.sum(dim=-1)[valid]
                - torch.ones_like(output.local_readout.attention.sum(dim=-1)[valid])
            )
            .abs()
            .max()
        )
        pre_attention_error = float(
            (
                output.pre_readout.attention.sum(dim=-1)[valid]
                - torch.ones_like(output.pre_readout.attention.sum(dim=-1)[valid])
            )
            .abs()
            .max()
        )
        invalid_token_max = max(
            float(getattr(output.responsibility_tokens, name)[invalid].abs().max())
            for name in ("r_loc", "r_ctx", "r_gate")
        )
        invalid_logit_max = float(output.conditional.logits[invalid].abs().max())
        invalid_tail_gradient_max = float(x_gradient[~batch.length_mask].abs().max())

    checks = {
        "loss_finite": math.isfinite(loss.total.detach().item()),
        "valid_logits_finite": bool(
            torch.isfinite(output.conditional.logits[valid]).all()
        ),
        "parameter_gradients_present": bool(parameter_gradients),
        "parameter_gradients_finite": all(
            bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients
        ),
        "raw_input_gradient_finite": bool(torch.isfinite(x_gradient).all()),
        "invalid_tail_gradient_zero": invalid_tail_gradient_max == 0.0,
        "invalid_responsibility_tokens_zero": invalid_token_max == 0.0,
        "invalid_logits_zero": invalid_logit_max == 0.0,
        "exact_path_probabilities_normalized": probability_error < 1e-5,
        "local_attention_normalized": local_attention_error < 1e-5,
        "pre_attention_normalized": pre_attention_error < 1e-5,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M1-13 GPU smoke checks failed: {checks}")

    repository = Path(__file__).resolve().parents[1]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "a3-m1-13-integrated-gpu-smoke-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M1-13",
        "status": "PASS",
        "evidence_class": "synthetic_integrated_software_only",
        "scope": {
            "reads_real_rf": False,
            "reads_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_step": True,
            "performs_real_training": False,
        },
        "resolved_config": {
            "seed": SEED,
            "batch_size": 4,
            "trace_length": 2300,
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
            "lambda_local": 0.3,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": properties.name,
            "device_total_memory_mib": _mib(properties.total_memory),
        },
        "metrics": {
            "total_loss": loss.total.detach().item(),
            "chain_loss": loss.chain.detach().item(),
            "local_loss": loss.local.detach().item(),
            "forward_backward_cuda_ms": forward_backward_ms,
            "optimizer_wall_time_s": wall_time_s,
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_tensors_with_grad": len(parameter_gradients),
            "baseline_allocated_mib": _mib(baseline_allocated),
            "peak_allocated_mib": _mib(peak_allocated),
            "peak_allocated_delta_mib": _mib(
                max(0, peak_allocated - baseline_allocated)
            ),
            "baseline_reserved_mib": _mib(baseline_reserved),
            "peak_reserved_mib": _mib(peak_reserved),
            "device_free_before_mib": _mib(free_before),
            "device_free_after_mib": _mib(free_after),
            "exact_probability_max_abs_error": probability_error,
            "local_attention_max_abs_error": local_attention_error,
            "pre_attention_max_abs_error": pre_attention_error,
            "invalid_token_max_abs": invalid_token_max,
            "invalid_logit_max_abs": invalid_logit_max,
            "invalid_tail_gradient_max_abs": invalid_tail_gradient_max,
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
            "m1_integrated_gpu_smoke.py": _sha256(Path(__file__).resolve()),
        },
        "interpretation_boundary": (
            "PASS proves one seeded synthetic integrated CUDA forward/backward "
            "software path within the 8 GB device. It does not prove RF "
            "performance, training stability, capacity/runtime fairness, or mechanism."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "peak_allocated_mib": result["metrics"]["peak_allocated_mib"],
                "forward_backward_cuda_ms": result["metrics"][
                    "forward_backward_cuda_ms"
                ],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
