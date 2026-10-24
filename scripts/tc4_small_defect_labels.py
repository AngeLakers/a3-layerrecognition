"""Versioned 2 mm interior-only label relaxation; never edit frozen labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tc4_label_pool import classify, epoch_indices, sha, spatial_draw, write_json


def interior(points, slot):
    """Independent analytic membership, including the nominal boundary."""
    delta = np.abs(points - [slot["front_x"], slot["front_y"]])
    radius = slot["size"] / 2
    if slot["front_vertices"]:
        return (delta <= radius).all(1)
    return (delta**2).sum(1) <= radius**2


def relax(points, slots, old_mask, board_inset):
    reference = ((points >= board_inset) & (points <= 100 - board_inset)).all(1)
    small = np.zeros_like(old_mask)
    for slot in slots:
        if slot["size"] == 2:
            small[:, slot["interface"] - 1] |= interior(points, slot)
    small &= reference[:, None]
    admitted = small & ~old_mask
    return old_mask | small, admitted, small


def build(profile_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    profile = json.loads(profile_path.read_text("utf-8"))
    parent = Path(profile["parent_label_root"])
    parent_summary = json.loads((parent / "summary.json").read_text("utf-8"))
    geometry = Path(profile["geometry"])
    assert sha(geometry) == profile["geometry_sha256"]
    slots = [s for s in json.loads(geometry.read_text("utf-8"))["slots"]
             if s["design"] == "K3"]
    small_slots = [s for s in slots if s["size"] == 2]
    assert len(small_slots) == 3
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "profile.json", profile)
    provenance = output / "provenance"
    provenance.mkdir()
    for source in (geometry, profile_path, Path(__file__),
                   Path(__file__).with_name("tc4_label_pool.py")):
        shutil.copy2(source, provenance / source.name)
    rows, selection = [], []
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), layout="constrained")
    for parent_row in parent_summary["batches"]:
        b = parent_row["batch"]
        old_path = parent / f"batch{b}" / "labels.npz"
        assert sha(old_path) == parent_row["labels_sha256"]
        old = dict(np.load(old_path))
        points = old["xy_front_mm"]
        _, nominal, old_recomputed, _, _ = classify(
            points, slots, parent_row["margin_mm"], profile["board_inset_mm"])
        np.testing.assert_array_equal(old_recomputed, old["label_mask"])
        np.testing.assert_array_equal(nominal, old["nominal_y"])
        mask, admitted, small = relax(points, slots, old["label_mask"],
                                      profile["board_inset_mm"])
        assert np.all(nominal[admitted] == 1)
        new = dict(old)
        new.update(label_mask=mask, y=np.where(mask, nominal, -1).astype(np.int8),
                   complete_label_mask=mask.all(1),
                   joint_state=np.where(mask.all(1), nominal @ [4, 2, 1], -1).astype(np.int8),
                   boundary_admitted_mask=admitted, small_defect_mask=small)
        np.testing.assert_array_equal(new["y"][old["label_mask"]],
                                      old["y"][old["label_mask"]])
        np.testing.assert_array_equal(new["y"][~small], old["y"][~small])
        assert np.array_equal(new["joint_state"] == 0, old["joint_state"] == 0)
        index_path = Path(profile["signal_root"]) / f"batch{b}" / "index.npz"
        idx = dict(np.load(index_path))
        np.testing.assert_array_equal(idx["source_row"], new["source_row"])
        np.testing.assert_array_equal(idx["parent_sample_id"], new["parent_sample_id"])
        sel = np.load(index_path.parent / "parent_selection.npz")
        np.testing.assert_array_equal(new["source_row"],
                                      sel["parent_source_row"][sel["selected"]])
        dest = output / f"batch{b}"
        dest.mkdir()
        np.savez_compressed(dest / "labels.npz", **new)
        sampled = epoch_indices(new, profile["seed"] + b)
        np.savez_compressed(dest / "unsplit_sampling_example.npz", sample_id=sampled)
        assert len(np.unique(sampled)) == len(sampled)
        assert (new["joint_state"][sampled] > 0).sum() == (new["joint_state"][sampled] == 0).sum()
        counts = {f"{k:03b}": int((new["joint_state"] == k).sum()) for k in range(8)}
        per_slot = {}
        rng = np.random.default_rng(profile["seed"] + b)
        chosen = set()
        for j, slot in enumerate(small_slots):
            k = slot["interface"] - 1
            inside = interior(points, slot)
            usable = inside & new["complete_label_mask"]
            added = usable & ~old["complete_label_mask"]
            ids = np.flatnonzero(usable)
            # Ensure small defects are represented without manufacturing duplicates.
            new_ids = np.flatnonzero(added)
            take_new = rng.choice(new_ids, min(4, len(new_ids)), replace=False)
            chosen.update(take_new.tolist())
            remaining = np.asarray([i for i in ids if i not in chosen], dtype=np.int64)
            chosen.update(rng.choice(remaining, min(8 - len(take_new), len(remaining)),
                                     replace=False).tolist())
            per_slot[slot["id"]] = {
                "nominal_records": int(inside.sum()),
                "old_complete_positive": int((inside & old["complete_label_mask"]).sum()),
                "new_complete_positive": int(usable.sum()),
                "new_interface_positive": int((inside & mask[:, k]).sum()),
            }
            ax = axes[b - 1, j]
            near = (np.abs(points - [slot["front_x"], slot["front_y"]]) <= 2.0).all(1)
            ax.scatter(points[near, 0], points[near, 1], s=7, c="#c6c9cf")
            keep = near & inside & old["complete_label_mask"]
            ax.scatter(points[keep, 0], points[keep, 1], s=13, c="#1865ab")
            ax.scatter(points[added, 0], points[added, 1], s=13, c="#e06525")
            center = (slot["front_x"], slot["front_y"])
            patch = (Rectangle((center[0] - 1, center[1] - 1), 2, 2, fill=False)
                     if slot["front_vertices"] else Circle(center, 1, fill=False))
            ax.add_patch(patch)
            ax.set(xlim=(center[0] - 2, center[0] + 2),
                   ylim=(center[1] - 2, center[1] + 2), aspect="equal",
                   title=f"Batch {b} / {slot['id']}\n"
                         f"{per_slot[slot['id']]['old_complete_positive']} -> {int(usable.sum())}",
                   xlabel="Front X / mm", ylabel="Front Y / mm")
            if b == 1 and k == 0:
                ax.text(*center, "Excluded region\nunchanged", ha="center", va="center")
        abnormal = np.flatnonzero(new["joint_state"] > 0)
        available = np.asarray([i for i in abnormal if i not in chosen], dtype=np.int64)
        chosen.update(spatial_draw(available, new["joint_state"],
                                   64 - len(chosen), rng).tolist())
        assert len(chosen) == 64
        normals = spatial_draw(np.flatnonzero(new["joint_state"] == 0),
                               new["spatial_tile"], 64, rng)
        ids = np.r_[np.array(sorted(chosen)), normals]
        assert len(ids) == len(np.unique(ids)) == 128
        selection.extend((b, int(i)) for i in ids)
        row = dict(batch=b, records=len(points), state_counts=counts,
                   old_complete=int(old["complete_label_mask"].sum()),
                   complete_labels=int(new["complete_label_mask"].sum()),
                   admitted_interface_labels=int(admitted.sum()),
                   newly_complete_records=int((new["complete_label_mask"]
                                                & ~old["complete_label_mask"]).sum()),
                   masked_records=int((~new["complete_label_mask"]).sum()),
                   per_small_slot=per_slot, old_labels_sha256=sha(old_path),
                   labels_sha256=sha(dest / "labels.npz"), index_sha256=sha(index_path),
                   other_labels_unchanged=True, source_exclusions_preserved=True)
        write_json(dest / "manifest.json", row)
        rows.append(row)
        print(f"Batch {b}: {row['newly_complete_records']} newly complete records", flush=True)
    fig.suptitle("2 mm nominal interiors only | blue: old complete, orange: added complete\n"
                 "Gray: other points; black: nominal outline. No spatial resampling.")
    fig.savefig(output / "small_defect_boundary_comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(status="passed", version=profile["id"], batches=rows,
                   total_records=sum(r["records"] for r in rows),
                   complete_labels=sum(r["complete_labels"] for r in rows),
                   newly_complete_records=sum(r["newly_complete_records"] for r in rows),
                   admitted_interface_labels=sum(r["admitted_interface_labels"] for r in rows),
                   state_counts={f"{k:03b}": sum(r["state_counts"][f"{k:03b}"] for r in rows)
                                 for k in range(8)},
                   formal_split_created=False, source_waveforms_modified=False,
                   label_basis="User-accepted nominal design interiors, not measured boundary certainty",
                   profile_sha256=sha(profile_path), script_sha256=sha(__file__))
    write_json(output / "summary.json", summary)
    write_json(output / "diagnostic_selection.json", selection)
    write_json(output / "label_validation.json", dict(
        status="passed", records_checked=summary["total_records"],
        old_masks_recomputed_exactly=True, independent_analytic_small_membership=True,
        only_unknown_to_positive_in_small_interiors=True, normals_unchanged=True,
        source_exclusions_preserved=True, selected_unique_complete_records=len(selection)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.profile, args.output)
