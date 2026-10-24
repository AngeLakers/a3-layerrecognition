"""Synthetic formal-backbone/four-Head CPU or CUDA smoke; no research gate promotion."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch

from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, build_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback.")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    device = torch.device(args.device)
    x = torch.randn(2, 2300, device=device)
    length_mask = torch.arange(2300, device=device)[None, :] < torch.tensor([[2000], [2300]], device=device)
    mask = torch.tensor([[True, True, False], [True, True, True]], device=device)
    metadata = torch.zeros(2, 3, 5, device=device)
    for i, length in enumerate((2000, 2300)):
        count = int(mask[i].sum())
        for j in range(count):
            metadata[i, j] = torch.tensor([(j + 1) / count, (400 + j * 550) / length,
                                          (j + 1) / (count + 1), 50 / length, 50 / length], device=device)
    y = torch.tensor([[0, 1, -1], [1, 0, 1]], device=device)
    batch = InterfaceBatch(x, length_mask, metadata, mask, mask.clone(), y)
    rows = []
    for backbone in ("resnet1d", "inceptiontime", "fcn1d"):
        for head in HEAD_TYPES:
            torch.manual_seed(43)
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            config = ModelConfig(backbone=backbone, head_type=head)
            model = build_model(config).to(device)
            optimizer, counts = build_optimizer(model, TrainingConfig(device=args.device))
            model.train()
            output = model(batch, y).conditional
            loss = head_training_loss(output, batch, head_type=head)
            loss.total.backward()
            if not torch.isfinite(loss.total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient/loss for {backbone}/{head}.")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                exact = model.infer_exact(batch)
            torch.testing.assert_close(exact.raw_log_normalizer, torch.zeros(2, device=device), atol=2e-5, rtol=0)
            assert exact.path_mask.sum(1).tolist() == [4, 8]
            if device.type == "cuda":
                torch.cuda.synchronize()
            rows.append({"backbone": backbone, "head": head, "device": args.device, "status": "passed",
                         "parameters": sum(counts.values()), "head_parameters": counts["pcc"],
                         "raw_logZ_max_abs": float(exact.raw_log_normalizer.abs().max()),
                         "loss": float(loss.total.detach()), "wall_time_seconds": time.perf_counter() - started,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None})
            print(f"{backbone}/{head}: passed", flush=True)
            del model, optimizer, output, loss, exact
            gc.collect()
    result = {"architecture_version": "a3-v1.3", "data_kind": "synthetic", "seed": 43,
              "batch_size": 2, "trace_lengths": [2000, 2300], "K": [2, 3], "float_dtype": "float32", "amp": False,
              "torch_version": str(torch.__version__), "device": args.device,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "runs": rows, "performance_claims_released": False, "real_training_gate_changed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
