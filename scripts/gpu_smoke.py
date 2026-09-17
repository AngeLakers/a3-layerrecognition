"""Single-batch CUDA forward/backward smoke test for the A3 prototype API."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import psutil
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from a3_ultrasound.inference.enumerator import enumerate_pcc  # noqa: E402
from a3_ultrasound.schema import INTERFACE_META_DIM  # noqa: E402
from a3_ultrasound.models.indexed_readout import (  # noqa: E402
    IndexedReadout,
    build_indexed_supports,
)
from a3_ultrasound.models.pcc import PCCConditionalStep  # noqa: E402
from a3_ultrasound.models.query import InterfaceQueryEncoder  # noqa: E402
from a3_ultrasound.models.responsibility_tokens import (  # noqa: E402
    ResponsibilityTokenProjector,
)


MIB = 1024**2


def mib(value: int) -> float:
    return round(value / MIB, 3)


def make_batch(
    device: torch.device,
    *,
    batch_size: int,
    feature_count: int,
    feature_dim: int,
    kmax: int,
) -> dict[str, Tensor]:
    features = torch.randn(
        batch_size,
        feature_count,
        feature_dim,
        device=device,
        requires_grad=True,
    )
    feature_centers = torch.arange(feature_count, device=device, dtype=torch.float32)
    feature_mask = torch.ones(batch_size, feature_count, device=device, dtype=torch.bool)

    interface_mask = torch.ones(batch_size, kmax, device=device, dtype=torch.bool)
    interface_mask[::2, -1] = False
    base_starts = torch.tensor([64.0, 208.0, 352.0], device=device)
    jitter = torch.arange(batch_size, device=device, dtype=torch.float32).remainder(5).unsqueeze(1)
    window_start = base_starts.unsqueeze(0) + jitter
    window_end = window_start + 48.0

    valid_counts = interface_mask.sum(dim=1, keepdim=True).clamp_min(1)
    relative_order = (
        torch.arange(1, kmax + 1, device=device, dtype=torch.float32).unsqueeze(0)
        / valid_counts
    )
    normalized_center = (window_start + window_end) / (2.0 * feature_count)
    normalized_half_width = (window_end - window_start) / (2.0 * feature_count)

    interface_meta = torch.zeros(
        batch_size,
        kmax,
        INTERFACE_META_DIM,
        device=device,
    )
    interface_meta[..., 0] = relative_order
    interface_meta[..., 1] = normalized_center
    interface_meta[..., 2] = relative_order
    interface_meta[..., 3] = normalized_half_width
    interface_meta[..., 4] = normalized_half_width
    interface_meta = interface_meta * interface_mask.unsqueeze(-1)
    targets = torch.randint(0, 2, (batch_size, kmax), device=device)
    targets = targets * interface_mask.to(dtype=targets.dtype)

    return {
        "features": features,
        "feature_centers": feature_centers,
        "feature_mask": feature_mask,
        "interface_mask": interface_mask,
        "window_start": window_start,
        "window_end": window_end,
        "interface_meta": interface_meta,
        "targets": targets,
    }


def forward_batch(
    modules: dict[str, nn.Module],
    batch: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    query = modules["query"](batch["interface_meta"], batch["interface_mask"])
    local_mask, pre_mask, local_prior = build_indexed_supports(
        batch["feature_centers"],
        batch["window_start"],
        batch["window_end"],
        batch["feature_mask"],
        batch["interface_mask"],
    )
    local = modules["local_readout"](
        batch["features"],
        query,
        local_mask,
        batch["interface_mask"],
        local_prior=local_prior,
    )
    pre = modules["pre_readout"](
        batch["features"],
        None,
        pre_mask,
        batch["interface_mask"],
        allow_empty_context=True,
    )
    tokens = modules["tokens"](
        local.token,
        pre.token,
        query,
        batch["interface_mask"],
    )
    output = modules["pcc"](
        tokens.r_loc,
        tokens.r_ctx,
        tokens.r_gate,
        query,
        batch["targets"],
        batch["interface_mask"],
    )
    mask = batch["interface_mask"].to(dtype=output.logits.dtype)
    element_loss = F.binary_cross_entropy_with_logits(
        output.logits,
        batch["targets"].to(dtype=output.logits.dtype),
        reduction="none",
    )
    loss = (element_loss * mask).sum() / mask.sum()
    return loss, {
        "query": query,
        "local_token": local.token,
        "pre_token": pre.token,
        "r_loc": tokens.r_loc,
        "r_ctx": tokens.r_ctx,
        "r_gate": tokens.r_gate,
        "local_attention": local.attention,
        "pre_attention": pre.attention,
        "logits": output.logits,
    }


def build_modules(
    device: torch.device,
    *,
    feature_dim: int,
    query_dim: int,
) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {
        "query": InterfaceQueryEncoder(query_dim=query_dim, hidden_dim=64),
        "local_readout": IndexedReadout(
            feature_dim=feature_dim,
            query_dim=query_dim,
            attention_dim=64,
        ),
        "pre_readout": IndexedReadout(
            feature_dim=feature_dim,
            query_dim=None,
            attention_dim=64,
        ),
        "tokens": ResponsibilityTokenProjector(
            local_dim=feature_dim,
            context_dim=feature_dim,
            query_dim=query_dim,
            token_dim=query_dim,
        ),
        "pcc": PCCConditionalStep(
            local_dim=query_dim,
            context_dim=query_dim,
            gate_dim=query_dim,
            query_dim=query_dim,
            prefix_dim=32,
            hidden_dim=64,
            max_interfaces=6,
            rho=1.0,
        ),
    }
    for module in modules.values():
        module.to(device).train()
    return modules


def run_smoke(*, batch_size: int, output_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment.")

    torch.manual_seed(20260722)
    torch.cuda.manual_seed_all(20260722)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    feature_count = 512
    feature_dim = 64
    query_dim = 32
    kmax = 3
    batch = make_batch(
        device,
        batch_size=batch_size,
        feature_count=feature_count,
        feature_dim=feature_dim,
        kmax=kmax,
    )
    modules = build_modules(device, feature_dim=feature_dim, query_dim=query_dim)
    parameters = [parameter for module in modules.values() for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)

    # Initialize lazy CUDA kernels without creating training gradients.
    with torch.no_grad():
        warmup_loss, _ = forward_batch(modules, batch)
    torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    batch["features"].grad = None
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()
    loss, tensors = forward_batch(modules, batch)
    loss.backward()
    end_event.record()
    torch.cuda.synchronize(device)
    forward_backward_ms = start_event.elapsed_time(end_event)
    optimizer.step()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    feature_gradient_finite = batch["features"].grad is not None and bool(
        torch.isfinite(batch["features"].grad).all()
    )

    with torch.no_grad():
        _, inference_tensors = forward_batch(modules, batch)
        exact = enumerate_pcc(
            modules["pcc"],
            inference_tensors["r_loc"],
            inference_tensors["r_ctx"],
            inference_tensors["r_gate"],
            inference_tensors["query"],
            batch["interface_mask"],
        )
        probability_sums = (exact.probabilities * exact.path_mask).sum(dim=1)
        probability_error = (probability_sums - 1.0).abs().max().item()
        valid = batch["interface_mask"]
        local_attention_error = (
            tensors["local_attention"].sum(dim=-1)[valid] - 1.0
        ).abs().max().item()
        pre_attention_error = (
            tensors["pre_attention"].sum(dim=-1)[valid] - 1.0
        ).abs().max().item()

    properties = torch.cuda.get_device_properties(device)
    process_rss = psutil.Process().memory_info().rss
    checks = {
        "loss_finite": math.isfinite(loss.item()),
        "parameter_gradients_present": len(gradients) > 0,
        "parameter_gradients_finite": gradient_finite,
        "backbone_boundary_gradient_finite": feature_gradient_finite,
        "exact_path_probabilities_normalized": probability_error < 1e-5,
        "local_attention_normalized": local_attention_error < 1e-5,
        "pre_attention_normalized": pre_attention_error < 1e-5,
    }
    if not all(checks.values()):
        raise RuntimeError(f"GPU smoke checks failed: {checks}")

    result: dict[str, object] = {
        "status": "passed",
        "scope": "synthetic single-batch API/environment smoke; no real training",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": properties.name,
            "device_total_memory_mib": mib(properties.total_memory),
        },
        "batch": {
            "batch_size": batch_size,
            "feature_shape": list(batch["features"].shape),
            "interface_mask_shape": list(batch["interface_mask"].shape),
            "valid_interface_counts": batch["interface_mask"].sum(dim=1).tolist(),
        },
        "metrics": {
            "warmup_loss": warmup_loss.item(),
            "measured_loss": loss.item(),
            "forward_backward_cuda_ms": round(forward_backward_ms, 3),
            "forward_backward_optimizer_wall_ms": round(wall_ms, 3),
            "baseline_allocated_mib": mib(baseline_allocated),
            "peak_allocated_mib": mib(peak_allocated),
            "peak_allocated_delta_mib": mib(peak_allocated - baseline_allocated),
            "after_allocated_mib": mib(after_allocated),
            "baseline_reserved_mib": mib(baseline_reserved),
            "peak_reserved_mib": mib(peak_reserved),
            "after_reserved_mib": mib(after_reserved),
            "process_rss_mib": mib(process_rss),
            "parameter_tensors_with_grad": len(gradients),
            "exact_probability_max_abs_error": probability_error,
            "local_attention_max_abs_error": local_attention_error,
            "pre_attention_max_abs_error": pre_attention_error,
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "profile_output" / "gpu_smoke.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    result = run_smoke(batch_size=args.batch_size, output_path=args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
