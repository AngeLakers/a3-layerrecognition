"""Versioned drawing-derived labels and unsplit sampling pools for TC4 V3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def signed_distance(points, slot):
    """Positive inside; distances in front drawing mm, before the fitted affine."""
    delta = np.abs(points - [slot["front_x"], slot["front_y"]])
    radius = slot["size"] / 2
    if not slot["front_vertices"]:
        return radius - np.linalg.norm(delta, axis=1)
    # K3 contains axis-aligned squares only. Reject other designs/shapes.
    vertices = np.asarray(slot["front_vertices"])
    assert len(vertices) == 4
    assert np.allclose(np.ptp(vertices, axis=0), slot["size"])
    q = delta - radius
    return -(np.linalg.norm(np.maximum(q, 0), axis=1)
             + np.minimum(np.max(q, axis=1), 0))


def classify(points, slots, margin, board_inset):
    distances = np.stack([
        np.max(np.stack([signed_distance(points, s) for s in slots
                         if s["interface"] == k]), axis=0)
        for k in (1, 2, 3)
    ], axis=1)
    reference = ((points >= board_inset) & (points <= 100 - board_inset)).all(1)
    nominal = (distances >= 0).astype(np.int8)
    mask = (np.abs(distances) >= margin) & reference[:, None]
    y = np.where(mask, nominal, -1).astype(np.int8)
    joint = np.where(mask.all(1), nominal @ np.array([4, 2, 1]), -1).astype(np.int8)
    return distances, nominal, mask, y, joint


def spatial_draw(ids, groups, n, rng):
    """Round-robin shuffled spatial groups, without replacement."""
    pools = [rng.permutation(ids[groups[ids] == g]).tolist()
             for g in np.unique(groups[ids])]
    rng.shuffle(pools)
    chosen = []
    while pools and len(chosen) < n:
        following = []
        for pool in pools:
            if len(chosen) == n:
                break
            chosen.append(pool.pop())
            if pool:
                following.append(pool)
        pools = following
    return np.asarray(chosen, dtype=np.int64)


def epoch_indices(labels, seed, normal_to_abnormal=1.0):
    """Per-specimen example: keep all reliable abnormalities, rotate normal tiles.

    This is an unsplit demonstration. Production callers must first restrict
    the pool to their future training specimens.
    """
    rng = np.random.default_rng(seed)
    joint, tile = labels["joint_state"], labels["spatial_tile"]
    abnormal = np.flatnonzero(joint > 0)
    normal = np.flatnonzero(joint == 0)
    count = min(len(normal), int(np.ceil(len(abnormal) * normal_to_abnormal)))
    if not len(abnormal):
        raise ValueError("No reliable abnormal sample; do not silently sample normals only.")
    selected = np.r_[abnormal, spatial_draw(normal, tile, count, rng)]
    return rng.permutation(selected)


def build(profile_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "profile.json", profile)
    geom_path = Path(profile["geometry"])
    assert sha(geom_path) == profile["geometry_sha256"]
    slots = [s for s in json.loads(geom_path.read_text(encoding="utf-8"))["slots"]
             if s["design"] == "K3"]
    assert len(slots) == 15 and len({s["id"] for s in slots}) == 15
    source = Path(profile["signal_root"])
    registration = Path(profile["registration_root"])
    summary = {"status": "drawing_labels_unsplit", "formal_split": "deferred_by_user",
               "profile_sha256": sha(profile_path), "script_sha256": sha(__file__),
               "geometry_sha256": sha(geom_path), "batches": []}
    smoke_selection = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout="constrained")
    cmap = ListedColormap(["#b9bec6", "#f4f4f4", "#1b9e77", "#d95f02",
                           "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"])
    norm = BoundaryNorm(np.arange(-1.5, 8.5), cmap.N)
    for b in range(1, 5):
        dest = output / f"batch{b}"
        dest.mkdir()
        index_path = source / f"batch{b}" / "index.npz"
        index = dict(np.load(index_path))
        reg_path = registration / f"batch{b}_registration.json"
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        assert reg["geometry_sha256"] == sha(geom_path)
        affine = np.asarray(reg["coefficients"])
        front = (index["xy_local_mm"] - affine[2]) @ np.linalg.inv(affine[:2])
        assert np.allclose(np.c_[front, np.ones(len(front))] @ affine,
                           index["xy_local_mm"], atol=1e-10)
        # Same margin algorithm across batches; an engineering guard, not a
        # certified manufacturing or registration uncertainty bound.
        scale_min = np.linalg.svd(affine[:2], compute_uv=False).min()
        margin = float(np.ceil((reg["fit_residual_max_mm"] / scale_min
                                + profile["margin_extra_mm"]) / 0.05) * 0.05)
        distance, nominal, mask, y, joint = classify(
            front, slots, margin, profile["board_inset_mm"])
        complete = mask.all(1)
        tiles_xy = np.clip((front // 10).astype(int), 0, 9)
        tile = (tiles_xy[:, 1] * 10 + tiles_xy[:, 0]).astype(np.int16)
        centers = np.unique([[s["front_x"], s["front_y"]] for s in slots], axis=0)
        region = np.argmin(np.linalg.norm(front[:, None, :] - centers, axis=2),
                           axis=1).astype(np.int16)
        labels = dict(source_row=index["source_row"], sample_id=np.arange(len(front)),
                      parent_sample_id=index["parent_sample_id"], xy_front_mm=front,
                      xy_local_mm=index["xy_local_mm"], nominal_y=nominal, y=y,
                      label_mask=mask, signed_distance_mm=distance.astype(np.float32),
                      complete_label_mask=complete, joint_state=joint,
                      spatial_tile=tile, nearest_design_region=region)
        assert np.all(y[~mask] == -1)
        assert np.all((y[mask] == 0) | (y[mask] == 1))
        assert np.all(joint[~complete] == -1)
        assert len(np.unique(index["source_row"])) == len(front)
        np.savez_compressed(dest / "labels.npz", **labels)
        counts = {f"{k:03b}": int((joint == k).sum()) for k in range(8)}
        nominal_counts = {f"{k:03b}": int((nominal @ [4, 2, 1] == k).sum())
                          for k in range(8)}
        sensitivity = {}
        strict_margin = float(np.ceil((reg["leave_one_feature_out_max_mm"] / scale_min
                                       + profile["margin_extra_mm"]) / 0.05) * 0.05)
        for delta in sorted(set([0.25, 0.5, 0.75, 1., margin, strict_margin])):
            m = (np.abs(distance) >= delta) & (
                ((front >= profile["board_inset_mm"]) &
                 (front <= 100 - profile["board_inset_mm"])).all(1)[:, None])
            states = np.where(m.all(1), nominal @ [4, 2, 1], -1)
            sensitivity[f"{delta:.2f}"] = {f"{k:03b}": int((states == k).sum())
                                          for k in range(8)}
        sampled = epoch_indices(labels, profile["seed"] + b)
        next_epoch = epoch_indices(labels, profile["seed"] + b + 100)
        assert np.array_equal(sampled, epoch_indices(labels, profile["seed"] + b))
        assert len(np.unique(sampled)) == len(sampled)
        assert np.all(complete[sampled])
        assert set(np.flatnonzero(joint > 0)) <= set(sampled)
        assert not np.array_equal(sampled, next_epoch)
        np.savez_compressed(dest / "unsplit_sampling_example.npz",
                            sample_id=sampled, next_seed_sample_id=next_epoch)
        rng = np.random.default_rng(profile["seed"] + b)
        for k in range(8):
            ids = np.flatnonzero(joint == k)
            if not len(ids):
                continue
            groups = tile if k == 0 else region
            chosen = spatial_draw(ids, groups, min(len(ids), profile["smoke_per_state"]),
                                  rng)
            smoke_selection.extend((b, int(i)) for i in chosen)
        row = dict(batch=b, entity_id=f"TC4-4L-{b:02}", records=len(front),
                   margin_mm=margin, margin_basis="ceil_0.05(max_fit_residual/min_affine_scale + 0.15)",
                   strict_loo_diagnostic_margin_mm=strict_margin,
                   complete_labels=int(complete.sum()), masked_records=int((~complete).sum()),
                   state_counts=counts, nominal_state_counts=nominal_counts,
                   sensitivity=sensitivity, index_sha256=sha(index_path),
                   registration_sha256=sha(reg_path), labels_sha256=sha(dest / "labels.npz"),
                   example_samples=len(sampled),
                   example_normal=int((joint[sampled] == 0).sum()),
                   example_abnormal=int((joint[sampled] > 0).sum()))
        summary["batches"].append(row)
        write_json(dest / "manifest.json", row)
        ax = axes.flat[b - 1]
        # Plot sampled grid for visibility; no spatial interpolation of RF/labels.
        ax.scatter(front[::3, 0], front[::3, 1], c=joint[::3], s=1,
                   cmap=cmap, norm=norm, rasterized=True)
        ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal",
               title=f"Specimen {b} | margin {margin:.2f} mm",
               xlabel="Drawing front X / mm", ylabel="Drawing front Y / mm")
    fig.suptitle("Drawing-derived joint labels: gray = masked; white = 000")
    fig.savefig(output / "joint_label_maps.png", dpi=180)
    plt.close(fig)
    summary["total_records"] = sum(r["records"] for r in summary["batches"])
    summary["state_counts"] = {s: sum(r["state_counts"][s] for r in summary["batches"])
                               for s in summary["batches"][0]["state_counts"]}
    summary["complete_labels"] = sum(summary["state_counts"].values())
    summary["masked_records"] = summary["total_records"] - summary["complete_labels"]
    summary["normal_fraction_of_complete"] = summary["state_counts"]["000"] / summary["complete_labels"]
    summary["smoke_records"] = len(smoke_selection)
    write_json(output / "summary.json", summary)
    write_json(output / "smoke_selection.json", smoke_selection)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    values = list(summary["state_counts"].values())
    ax.bar(summary["state_counts"].keys(), values)
    ax.set(yscale="log", ylabel="Reliable A-scans (log scale)",
           xlabel="I1 I2 I3: 0 = no designed defect, 1 = designed defect")
    for i, n in enumerate(values):
        ax.text(i, n * 1.12, f"{n:,}", ha="center")
    fig.savefig(output / "state_counts.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.profile, args.output)
