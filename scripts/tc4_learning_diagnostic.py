"""Fixed-cohort learning diagnostic. Resubstitution only; no held-out claims."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, build_optimizer
from tc4_label_pool import sha, write_json


def metrics(y, marginals, map_states, probabilities):
    pred = marginals >= .5
    states = y @ np.array([4, 2, 1])
    mapped = map_states @ np.array([4, 2, 1])
    joint_cm = np.zeros((8, 8), dtype=int)
    np.add.at(joint_cm, (states, mapped), 1)
    per = []
    for k in range(3):
        cm = np.zeros((2, 2), dtype=int)
        np.add.at(cm, (y[:, k], pred[:, k].astype(int)), 1)
        tn, fp, fn, tp = cm.ravel()
        per.append(dict(interface=k + 1, confusion=cm.tolist(),
                        positive_support=int(tp + fn),
                        recall=float(tp / (tp + fn)) if tp + fn else None,
                        precision=float(tp / (tp + fp)) if tp + fp else None,
                        f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
                        specificity=float(tn / (tn + fp)) if tn + fp else None))
    p = np.clip(marginals, 1e-7, 1 - 1e-7)
    return dict(samples=len(y), joint_nll=float(-np.log(np.clip(
        probabilities[np.arange(len(y)), states], 1e-30, 1)).mean()),
        marginal_bce=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        exact_match=float((states == mapped).mean()),
        interface_accuracy=float((pred == y).mean()),
        per_interface=per, joint_confusion=joint_cm.tolist())


def build_input(config, output):
    root = Path(config["label_root"])
    profile = json.loads((root / "profile.json").read_text("utf-8"))
    pairs = np.asarray(json.loads((root / "diagnostic_selection.json").read_text("utf-8")))
    pairs = pairs[np.random.default_rng(config["seed"]).permutation(len(pairs))]
    records, checks = [], []
    width = profile["smoke_window_halfwidth_samples"]
    for b in range(1, 5):
        source = Path(profile["signal_root"]) / f"batch{b}"
        idx = dict(np.load(source / "index.npz"))
        labels = dict(np.load(root / f"batch{b}" / "labels.npz"))
        manifest = json.loads((root / f"batch{b}" / "manifest.json").read_text("utf-8"))
        assert sha(root / f"batch{b}" / "labels.npz") == manifest["labels_sha256"]
        assert sha(source / "index.npz") == manifest["index_sha256"]
        rf = np.load(source / "rf_lp20_float32.npy", mmap_mode="r")
        centers = np.asarray(profile["smoke_original_window_centers"][str(b)])
        assert np.all(centers[None, :] - width >= idx["crop_start"][:, None])
        assert np.all(centers[None, :] + width < idx["crop_end_original_exclusive"][:, None])
        checks.append(dict(batch=b, records=len(idx["source_row"]), all_windows_inside=True))
        for j in np.flatnonzero(pairs[:, 0] == b):
            i = int(pairs[j, 1])
            assert labels["complete_label_mask"][i]
            assert labels["source_row"][i] == idx["source_row"][i]
            n, offset, start = map(int, (idx["lengths"][i], idx["offsets"][i], idx["crop_start"][i]))
            trace = rf[offset:offset + n] / np.float32(profile["normalization_divisor"])
            meta = np.zeros((3, 5), dtype=np.float32)
            meta[:, 0], meta[:, 2] = np.arange(1, 4) / 3, np.arange(1, 4) / 4
            meta[:, 1], meta[:, 3:5] = (centers - start) / n, width / n
            records.append((int(j), trace.copy(), meta, labels["y"][i].astype(np.int64),
                            int(idx["source_row"][i]), start,
                            labels["boundary_admitted_mask"][i], labels["small_defect_mask"][i]))
        del rf, idx, labels
    records.sort(key=lambda r: r[0])
    lengths = np.array([len(r[1]) for r in records])
    x = np.zeros((len(records), lengths.max()), dtype=np.float32)
    for j, r in enumerate(records):
        x[j, :len(r[1])] = r[1]
    y = np.stack([r[3] for r in records])
    arrays = dict(x_raw=x, length_mask=np.arange(x.shape[1])[None] < lengths[:, None],
                  interface_meta=np.stack([r[2] for r in records]),
                  interface_mask=np.ones_like(y, dtype=bool),
                  label_mask=np.ones_like(y, dtype=bool), y=y)
    metadata = dict(batch_id=pairs[:, 0], sample_id=pairs[:, 1],
                    source_row=np.array([r[4] for r in records]),
                    crop_start=np.array([r[5] for r in records]),
                    boundary_admitted_mask=np.stack([r[6] for r in records]),
                    small_defect_mask=np.stack([r[7] for r in records]))
    assert np.isfinite(x).all() and np.all(x[~arrays["length_mask"]] == 0)
    assert len(set(map(tuple, pairs))) == len(pairs) == 512
    assert (y.any(1)).sum() == 256
    np.savez_compressed(output / "diagnostic_input.npz", **arrays, **metadata)
    write_json(output / "input_validation.json", dict(
        status="passed", unique_records=len(records), complete_labels=int(y.size),
        all_pool_window_checks=checks, normal_records=256, abnormal_records=256,
        new_boundary_records=int(metadata["boundary_admitted_mask"].any(1).sum()),
        filter_reapplied=False, divisor=profile["normalization_divisor"],
        windows_scope="Inherited engineering diagnostic windows, not final physical TOF prior"))
    return arrays, metadata


def run(config_path, output):
    config = json.loads(config_path.read_text("utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    shutil.copy2(__file__, output / Path(__file__).name)
    arrays, metadata = build_input(config, output)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable.")
    # Seeded dispatch witness without changing or rebuilding the existing env.
    torch.manual_seed(seed)
    witness = torch.randn(8, 8, device=device)
    assert torch.isfinite(witness @ witness.T).all()
    write_json(output / "environment.json", dict(python=sys.executable,
        torch=str(torch.__version__), numpy=np.__version__,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        seeded_cuda_matmul_passed=True, environment_modified=False,
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])): sha(p)
                 for p in (Path(__file__).resolve().parents[1] / "src").rglob("*.py")}))
    n, y = len(arrays["y"]), arrays["y"]
    schedules = np.stack([np.random.default_rng(seed + e).permutation(n)
                          for e in range(1, config["epochs"] + 1)])
    np.save(output / "epoch_orders.npy", schedules)
    size = config["batch_size"]

    def batch_at(ids):
        return InterfaceBatch(**{k: torch.from_numpy(v[ids]).to(device)
                                 for k, v in arrays.items()})

    def evaluate(model):
        model.eval()
        collected = {k: [] for k in ("marginals", "map_states", "probabilities")}
        max_log_z = 0.
        with torch.inference_mode():
            for start in range(0, n, size):
                b = replace(batch_at(slice(start, start + size)), y=None, label_mask=None)
                exact = model.infer_exact(b)
                assert torch.isfinite(exact.probabilities).all()
                assert (exact.path_mask.sum(1) == 8).all()
                max_log_z = max(max_log_z, float(exact.raw_log_normalizer.abs().max()))
                for key in collected:
                    collected[key].append(getattr(exact, key).cpu().numpy())
        assert max_log_z < 2e-5
        preds = {k: np.concatenate(v) for k, v in collected.items()}
        result = metrics(y, **preds)
        result["max_abs_log_normalizer"] = max_log_z
        result["per_batch"] = {str(b): metrics(y[metadata["batch_id"] == b],
            **{k: v[metadata["batch_id"] == b] for k, v in preds.items()}) for b in range(1, 5)}
        for name, select in (
            ("new_boundary_records", metadata["boundary_admitted_mask"].any(1)),
            ("small_defect_records", metadata["small_defect_mask"].any(1))):
            result[name] = metrics(y[select], **{k: v[select] for k, v in preds.items()})
        return result, preds

    runs = []
    for head in HEAD_TYPES:
        torch.manual_seed(seed)
        started = time.perf_counter()
        dest = output / head
        dest.mkdir()
        model_config = ModelConfig(backbone="fcn1d", head_type=head)
        model = build_model(model_config).to(device)
        optimizer, _ = build_optimizer(model, TrainingConfig(
            device=config["device"], learning_rate=config["learning_rate"],
            weight_decay=0., seed=seed))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        history = []
        initial, preds = evaluate(model)
        history.append(dict(epoch=0, **initial))
        np.savez_compressed(dest / "predictions_initial.npz", **preds)
        steps = []
        write_json(dest / "history.json", history)
        for epoch, order in enumerate(schedules, 1):
            model.train()
            for pos in range(0, n, size):
                batch = batch_at(order[pos:pos + size])
                optimizer.zero_grad(set_to_none=True)
                result = model(batch, batch.y)
                assert (result.local_support.sum(-1) > 2).all()
                loss = head_training_loss(result.conditional, batch, head_type=head)
                assert torch.isfinite(loss.total)
                loss.total.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all()
                           for p in model.parameters())
                optimizer.step()
                steps.append(dict(step=len(steps) + 1, epoch=epoch,
                                  total=float(loss.total.detach()),
                                  primary=float(loss.chain.detach()),
                                  auxiliary=float(loss.local.detach())))
            if epoch in config["evaluation_epochs"]:
                score, preds = evaluate(model)
                history.append(dict(epoch=epoch, **score))
                write_json(dest / "history.json", history)
                print(f"{head} epoch {epoch}: same-cohort joint NLL={score['joint_nll']:.4f}, "
                      f"exact match={score['exact_match']:.3f}", flush=True)
            write_json(dest / "steps.json", steps)
        changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
        assert any(n.startswith("backbone.") for n in changed)
        assert any(n.startswith("pcc.") for n in changed)
        np.savez_compressed(dest / "predictions_final.npz", **preds)
        model.eval()
        audit = batch_at(slice(0, 16))
        with torch.inference_mode():
            a = model.infer_exact(audit).probabilities
            b = model.infer_exact(replace(audit, y=None, label_mask=None)).probabilities
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        checkpoint = dest / "diagnostic_only.pt"
        torch.save(dict(status="resubstitution_diagnostic_only_not_for_pretraining",
                        model_config=asdict(model_config), state_dict=model.state_dict(),
                        input_sha256=sha(output / "diagnostic_input.npz")), checkpoint)
        restored = build_model(model_config).to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device,
                                             weights_only=True)["state_dict"])
        restored.eval()
        with torch.inference_mode():
            c = restored.infer_exact(replace(audit, y=None, label_mask=None)).probabilities
        torch.testing.assert_close(a, c, atol=0, rtol=0)
        row = dict(head=head, status="passed", updates=len(steps),
                   initial=initial, final=history[-1], seconds=time.perf_counter() - started,
                   finite_losses_gradients=True, backbone_and_head_updated=True,
                   label_blind_inference=True, checkpoint_reload_exact=True)
        runs.append(row)
        write_json(dest / "summary.json", row)
        del model, optimizer, restored, before, result, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = dict(status="passed", scope="same_cohort_resubstitution_learning_diagnostic",
        formal_split_created=False, held_out_evaluation=False, generalization_claim=False,
        samples=n, epochs=config["epochs"], seed=seed, runs=runs,
        input_sha256=sha(output / "diagnostic_input.npz"), config_sha256=sha(config_path))
    write_json(output / "summary.json", summary)
    plot(output, runs)
    print("LEARNING DIAGNOSTIC COMPLETED; SAME COHORT, NO HELD-OUT CLAIM", flush=True)


def plot(output, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for row in runs:
        history = json.loads((output / row["head"] / "history.json").read_text("utf-8"))
        x = [h["epoch"] for h in history]
        for ax, key in zip(axes, ("joint_nll", "marginal_bce", "exact_match")):
            ax.plot(x, [h[key] for h in history], marker="o", label=row["head"])
            ax.set(xlabel="Epoch", ylabel=key)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed 512-sample cohort: training and scoring use the SAME samples")
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), layout="constrained")
    for r, run in enumerate(runs):
        for k, item in enumerate(run["final"]["per_interface"]):
            cm = np.asarray(item["confusion"])
            ax = axes[r, k]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=512)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Drawing label",
                   title=f"{run['head']}\nI{k+1} recall={item['recall']:.3f}")
    fig.suptitle("Final same-cohort interface confusion matrices | NOT held-out performance")
    fig.savefig(output / "interface_confusion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
