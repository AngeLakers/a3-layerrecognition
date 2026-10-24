"""Toy-only offline preprocessing -> four Heads -> exact evaluation -> reload."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import yaml

from a3_ultrasound.data.preprocessing import preprocess_npz
from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.training import run_from_config, synthetic_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(23)
    profile = {"schema_version": "a3-rf-preprocessing-v1", "preprocessing_id": "toy-fixed131-lp20-pad256-v1",
               "acquisition": {"probe_mhz": 10, "gain_db": 18, "batch_id": "synthetic-only"},
               "processing": {"sample_rate_hz": 200_000_000, "normalization_divisor": 131, "padlen": 256}}
    profile_path = args.output / "preprocessing.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    for split, dataset in synthetic_datasets(23, 4).items():
        arrays = {f.name: v.numpy().copy() for f in fields(InterfaceBatch)
                  if (v := getattr(dataset.batch, f.name)) is not None}
        lengths = np.array([1500, 1646, 1500, 1646])
        arrays["x_raw"] = rng.normal(0, 3, (4, 1646))
        arrays["length_mask"] = np.arange(1646)[None, :] < lengths[:, None]
        for i in range(4):
            k = int(arrays["interface_mask"][i].sum())
            for j in range(k):
                tau = 350 + j * 400
                wave = 60 * np.sin(np.arange(61) * np.pi / 10) * np.exp(-(np.arange(61) - 30)**2 / 150)
                arrays["x_raw"][i, tau - 30:tau + 31] += (2 * arrays["y"][i, j] - 1) * wave
                arrays["interface_meta"][i, j] = [(j + 1) / k, tau / lengths[i],
                                                  (j + 1) / (k + 1), 50 / lengths[i], 50 / lengths[i]]
        arrays["sample_ids"] = np.array(dataset.sample_ids)
        arrays["specimen_ids"] = np.array(dataset.specimen_ids)
        source = args.output / f"{split}_raw.npz"
        np.savez_compressed(source, **arrays)
        preprocess_npz(source, args.output / f"{split}.npz", profile_path)
    manifest = {"schema_version": "a3-tensor-dataset-v1", "data_kind": "synthetic",
                "preprocessing_id": profile["preprocessing_id"], "split_id": "toy-disjoint-v1",
                "splits": {name: f"{name}.npz" for name in ("train", "validation", "test")}}
    (args.output / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    config = {"schema_version": "a3-run-v1.3", "model": {"backbone": "fcn1d"},
              "training": {"seed": 23, "epochs": 1, "batch_size": 2, "patience": 1,
                           "device": args.device, "real_training_permitted": False},
              "data": {"source": "tensor_manifest", "manifest": "manifest.yaml"}}
    path = args.output / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    results = []
    for head in HEAD_TYPES:
        start = time.perf_counter()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        trained = run_from_config(path, args.output / head, head_type=head)
        restored = run_from_config(path, args.output / f"{head}_reload", head_type=head,
                                   checkpoint_path=args.output / head / "best.pt")
        original_predictions = (args.output / head / "test_predictions.jsonl").read_bytes()
        reloaded_predictions = (args.output / f"{head}_reload" / "test_predictions.jsonl").read_bytes()
        if original_predictions != reloaded_predictions:
            raise AssertionError("Checkpoint reload changed exact predictions.")
        results.append({"head": head, "training": trained["status"], "reload": restored["status"],
                        "exact_predictions_identical": True, "wall_seconds": time.perf_counter() - start,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None})
        print(f"{head}: preprocessing/train/exact/reload passed", flush=True)
    summary = {"data_kind": "synthetic", "device": args.device, "torch_version": str(torch.__version__),
               "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
               "runs": results, "real_training_gate_changed": False, "performance_claims_released": False}
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
