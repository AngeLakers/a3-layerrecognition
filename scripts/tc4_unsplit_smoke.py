"""Authorized small real-RF flow check, with no train/validation/test split."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, build_optimizer
from tc4_label_pool import sha, write_json


def main(root, device_name, output=None):
    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    output = output if output is not None else root / "engineering_smoke"
    output.mkdir(exist_ok=False)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(2)
    torch.manual_seed(profile["seed"])
    np_rng = np.random.default_rng(profile["seed"])
    pairs = np.asarray(json.loads((root / "smoke_selection.json").read_text("utf-8")))
    pairs = pairs[np_rng.permutation(len(pairs))]
    divisor = np.float32(profile["normalization_divisor"])
    assert np.isfinite(divisor) and divisor > 0
    signal = Path(profile["signal_root"])
    indexes = {b: dict(np.load(signal / f"batch{b}" / "index.npz")) for b in range(1, 5)}
    labels = {b: dict(np.load(root / f"batch{b}" / "labels.npz")) for b in range(1, 5)}
    rf = {b: np.load(signal / f"batch{b}" / "rf_lp20_float32.npy", mmap_mode="r")
          for b in range(1, 5)}
    width = profile["smoke_window_halfwidth_samples"]
    all_window_checks = []
    for b, idx in indexes.items():
        centers = np.asarray(profile["smoke_original_window_centers"][str(b)])
        assert np.all(centers[None, :] - width >= idx["crop_start"][:, None])
        assert np.all(centers[None, :] + width < idx["crop_end_original_exclusive"][:, None])
        all_window_checks.append({"batch": b, "records": len(idx["source_row"]),
                                  "all_three_windows_inside_trace": True})
    lengths = np.array([indexes[b]["lengths"][i] for b, i in pairs])
    x = np.zeros((len(pairs), int(lengths.max())), dtype=np.float32)
    length_mask = np.arange(x.shape[1])[None] < lengths[:, None]
    meta = np.zeros((len(pairs), 3, 5), dtype=np.float32)
    y = np.empty((len(pairs), 3), dtype=np.int64)
    starts, source_rows = [], []
    for j, (b, i) in enumerate(pairs):
        idx, lab = indexes[b], labels[b]
        assert lab["complete_label_mask"][i]
        assert lab["source_row"][i] == idx["source_row"][i]
        start, length = int(idx["crop_start"][i]), int(lengths[j])
        offset = int(idx["offsets"][i])
        trace = rf[b][offset:offset + length]
        x[j, :length] = trace / divisor
        assert np.array_equal(x[j, :length], trace / divisor)
        assert np.all(np.signbit(x[j, :length]) == np.signbit(trace))
        centers = np.asarray(profile["smoke_original_window_centers"][str(b)]) - start
        meta[j, :, 0] = np.arange(1, 4) / 3
        meta[j, :, 1] = centers / length
        meta[j, :, 2] = np.arange(1, 4) / 4
        meta[j, :, 3:5] = width / length
        y[j] = lab["y"][i]
        starts.append(start)
        source_rows.append(int(idx["source_row"][i]))
    assert np.all(x[~length_mask] == 0) and np.isfinite(x).all()
    mask = np.ones_like(y, dtype=bool)
    arrays = dict(x_raw=x, length_mask=length_mask, interface_meta=meta,
                  interface_mask=mask, label_mask=mask.copy(), y=y)
    np.savez_compressed(output / "unsplit_flow_input.npz", **arrays,
                        batch_id=pairs[:, 0], sample_id=pairs[:, 1],
                        source_row=source_rows, crop_start=starts)
    device = torch.device(device_name)

    def batch_at(ids):
        return InterfaceBatch(**{k: torch.from_numpy(v[ids]).to(device)
                                 for k, v in arrays.items()})

    # Draw diagnostics only from data already selected by geometry.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), layout="constrained")
    for b, ax in enumerate(axes, 1):
        j = int(np.flatnonzero((pairs[:, 0] == b) & y.any(1))[0])
        n = int(lengths[j])
        ax.plot(np.arange(n) + starts[j], x[j, :n], lw=.7)
        for k, center in enumerate(profile["smoke_original_window_centers"][str(b)]):
            ax.axvspan(center - width, center + width, alpha=.13)
            ax.text(center, .85, f"I{k+1}", transform=ax.get_xaxis_transform(), ha="center")
        ax.set(title=f"Specimen {b}, sample {pairs[j,1]}, labels {y[j].tolist()}",
               ylabel="RF / 131")
    axes[-1].set_xlabel("Original sample index (200 MHz)")
    fig.savefig(output / "window_examples.png", dpi=150)
    plt.close(fig)
    rows = []
    # Same selected data and order across all Heads. No scoring/early stopping.
    for head in HEAD_TYPES:
        torch.manual_seed(profile["seed"])
        started = time.perf_counter()
        config = ModelConfig(backbone="fcn1d", head_type=head)
        model = build_model(config).to(device)
        optimizer, counts = build_optimizer(model, TrainingConfig(device=device_name))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses, token_counts = [], []
        model.train()
        for pos in range(0, len(pairs), 8):
            batch = batch_at(slice(pos, min(pos + 8, len(pairs))))
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, batch.y)
            support = result.local_support.sum(-1)
            assert (support > 2).all()
            token_counts.extend(support.detach().cpu().reshape(-1).tolist())
            loss = head_training_loss(result.conditional, batch, head_type=head)
            assert torch.isfinite(loss.total)
            loss.total.backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(float(loss.total.detach()))
        changed = [name for name, p in model.named_parameters()
                   if not torch.equal(before[name], p)]
        assert any(n.startswith("backbone.") for n in changed)
        assert any(n.startswith("pcc.") for n in changed)
        del before
        model.eval()
        audit_batch = batch_at(slice(0, min(32, len(pairs))))
        with torch.no_grad():
            exact = model.infer_exact(audit_batch)
            blind = model.infer_exact(replace(audit_batch, y=None, label_mask=None))
        assert torch.isfinite(exact.probabilities).all()
        torch.testing.assert_close(exact.probabilities, blind.probabilities, atol=0, rtol=0)
        torch.testing.assert_close(exact.raw_log_normalizer,
                                   torch.zeros(len(audit_batch.x_raw), device=device),
                                   atol=2e-5, rtol=0)
        assert (exact.path_mask.sum(1) == 8).all()
        checkpoint = output / f"{head}_engineering_only.pt"
        torch.save({"status": "engineering_only_do_not_use_as_pretrained_model",
                    "model_config": asdict(config), "state_dict": model.state_dict(),
                    "source_tensor_sha256": sha(output / "unsplit_flow_input.npz")}, checkpoint)
        restored = build_model(config).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        restored.load_state_dict(payload["state_dict"])
        restored.eval()
        with torch.no_grad():
            restored_exact = restored.infer_exact(audit_batch)
        torch.testing.assert_close(exact.probabilities, restored_exact.probabilities, atol=0, rtol=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rows.append(dict(head=head, backbone="fcn1d", status="passed",
                         updates=len(losses), loss_first=losses[0], loss_last=losses[-1],
                         finite_loss_gradients=True, backbone_and_head_updated=True,
                         changed_parameter_tensors=len(changed), parameters=sum(counts.values()),
                         local_token_count_min=min(token_counts), local_token_count_max=max(token_counts),
                         log_normalizer_max_abs=float(exact.raw_log_normalizer.abs().max()),
                         label_blind_exact_unchanged=True, checkpoint_reload_exact=True,
                         seconds=time.perf_counter() - started,
                         peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
                         if device.type == "cuda" else None))
        print(f"{head}: {len(losses)} updates, exact inference and reload passed", flush=True)
        del model, optimizer, restored, payload, result, loss, exact, blind, restored_exact
    result = dict(status="passed", data_kind="real_RF_drawing_derived_labels",
                  scope="small_unsplit_engineering_flow_only", formal_split_created=False,
                  accuracy_or_generalization_claim=False, seed=profile["seed"],
                  samples=len(pairs), interface_labels=int(y.size), device=device_name,
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  torch_version=str(torch.__version__), python=sys.executable,
                  fixed_divisor=float(divisor), filter_reapplied=False,
                  windows_scope="diagnostic centers for smoke, not final TOF calibration",
                  all_pool_window_checks=all_window_checks, runs=rows,
                  input_sha256=sha(output / "unsplit_flow_input.npz"),
                  script_sha256=sha(__file__),
                  source_code_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): sha(p)
                                      for p in (Path(__file__).resolve().parents[1] / "src" / "a3_ultrasound").rglob("*.py")})
    write_json(output / "summary.json", result)
    print("UNSPLIT ENGINEERING FLOW PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, help="New run directory; never overwrites an existing run.")
    args = parser.parse_args()
    main(args.labels, args.device, args.output)
