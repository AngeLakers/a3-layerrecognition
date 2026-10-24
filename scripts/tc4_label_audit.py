"""Independent geometry-membership and current source-index audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from tc4_label_pool import epoch_indices, signed_distance, sha, write_json


def main(root):
    profile = json.loads((root / "profile.json").read_text("utf-8"))
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    slots = [s for s in json.loads(Path(profile["geometry"]).read_text("utf-8"))["slots"]
             if s["design"] == "K3"]
    # Explicit analytic edge cases, including circle vs square corner.
    circle = dict(front_x=0, front_y=0, size=2, front_vertices=[])
    square = dict(circle, front_vertices=[[-1, -1], [-1, 1], [1, 1], [1, -1]])
    points = np.array([[0, 0], [1, 0], [2, 0], [1, 1], [.75, .75]])
    np.testing.assert_allclose(signed_distance(points, circle),
                               1 - np.sqrt(np.sum(points**2, axis=1)))
    np.testing.assert_allclose(signed_distance(points, square), [1, 0, -1, 0, .25])
    assert signed_distance(points[-1:], circle)[0] < 0
    checked, rows = 0, []
    for row in summary["batches"]:
        b, delta = row["batch"], row["margin_mm"]
        labels_path = root / f"batch{b}" / "labels.npz"
        lab = dict(np.load(labels_path))
        path = Path(profile["signal_root"]) / f"batch{b}" / "index.npz"
        idx = dict(np.load(path))
        assert sha(path) == row["index_sha256"]
        assert sha(labels_path) == row["labels_sha256"]
        assert np.array_equal(lab["source_row"], idx["source_row"])
        assert np.array_equal(lab["xy_local_mm"], idx["xy_local_mm"])
        # Check prior V3 exclusions survive one-to-one in the label sidecar.
        sel = np.load(path.parent / "parent_selection.npz")
        assert set(lab["source_row"]) == set(sel["parent_source_row"][sel["selected"]])
        pts = lab["xy_front_mm"]
        reference = ((pts >= profile["board_inset_mm"]) &
                     (pts <= 100 - profile["board_inset_mm"])).all(1)
        direct_nominal = np.zeros((len(pts), 3), dtype=bool)
        direct_positive = np.zeros_like(direct_nominal)
        near = np.zeros_like(direct_nominal)
        per_slot = {}
        for s in slots:
            k = s["interface"] - 1
            dx = np.abs(pts[:, 0] - s["front_x"])
            dy = np.abs(pts[:, 1] - s["front_y"])
            radius = s["size"] / 2
            if not s["front_vertices"]:
                r2 = dx**2 + dy**2
                inside = r2 <= radius**2
                shrunk = r2 <= max(0, radius - delta)**2
                expanded = r2 < (radius + delta)**2
            else:
                inside = (dx <= radius) & (dy <= radius)
                shrunk = (dx <= radius - delta) & (dy <= radius - delta)
                expanded = (np.maximum(dx - radius, 0)**2
                            + np.maximum(dy - radius, 0)**2) < delta**2
            direct_nominal[:, k] |= inside
            direct_positive[:, k] |= shrunk
            near[:, k] |= expanded
            per_slot[s["id"]] = dict(nominal_records=int(inside.sum()),
                                    complete_positive_records=int(
                                        (shrunk & lab["complete_label_mask"]).sum()))
        direct_mask = reference[:, None] & (direct_positive | ~near)
        assert np.array_equal(direct_nominal, lab["nominal_y"])
        assert np.array_equal(direct_mask, lab["label_mask"])
        assert np.array_equal(np.where(direct_mask, direct_nominal, -1), lab["y"])
        ids = epoch_indices(lab, profile["seed"] + b)
        recorded = np.load(root / f"batch{b}" / "unsplit_sampling_example.npz")["sample_id"]
        assert np.array_equal(ids, recorded)
        normal_ids = ids[lab["joint_state"][ids] == 0]
        abnormal_ids = ids[lab["joint_state"][ids] > 0]
        assert len(normal_ids) == len(abnormal_ids)
        assert len(np.unique(ids)) == len(ids)
        assert np.all(lab["complete_label_mask"][ids])
        rows.append(dict(batch=b, records=len(pts), per_slot=per_slot,
                         normal_tiles_sampled=len(np.unique(lab["spatial_tile"][normal_ids])),
                         no_record_loss=True, source_exclusions_preserved=True,
                         independent_geometry_membership_exact=True))
        checked += len(pts)
    # Separate reader verifies saved smoke RF and labels from the parent.
    smoke = np.load(root / "engineering_smoke" / "unsplit_flow_input.npz")
    for b in range(1, 5):
        source = Path(profile["signal_root"]) / f"batch{b}"
        idx = np.load(source / "index.npz")
        rf = np.load(source / "rf_lp20_float32.npy", mmap_mode="r")
        labels = np.load(root / f"batch{b}" / "labels.npz")
        for j in np.flatnonzero(smoke["batch_id"] == b):
            i = int(smoke["sample_id"][j])
            length, offset = int(idx["lengths"][i]), int(idx["offsets"][i])
            assert np.array_equal(smoke["x_raw"][j, :length],
                                  rf[offset:offset+length] / np.float32(profile["normalization_divisor"]))
            assert np.array_equal(smoke["y"][j], labels["y"][i])
            assert smoke["source_row"][j] == idx["source_row"][i]
            assert smoke["crop_start"][j] == idx["crop_start"][i]
            assert smoke["length_mask"][j].sum() == length
    write_json(root / "label_validation.json",
               dict(status="passed", records_checked=checked, batches=rows,
                    analytic_circle_square_cases_passed=True,
                    saved_smoke_records_rechecked=len(smoke["y"]),
                    formal_split_created=False, original_rf_modified=False,
                    script_sha256=sha(__file__)))
    print(f"PASS: {checked} label rows, geometry/masks/source identity; {len(smoke['y'])} smoke RF rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    main(parser.parse_args().labels)
