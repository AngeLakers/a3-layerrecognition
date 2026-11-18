"""Freeze specimen-local V1.7.1 B/C row exclusions without reading RF."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from a3_ultrasound.data.validation_sampling import select_validation_rows


class InsufficientSupport(ValueError):
    """The registered paired experiment has no usable population for this fold or seed."""


FIELDS = ("source_row", "y", "label_mask", "eligible", "xy", "quality")


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")


def read_index(catalog_path, entry):
    root = Path(catalog_path).parent / entry["index_dir"]
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FIELDS}
    return arrays


def complete_target(entry, arrays, target):
    if entry["K"] != 3:
        return np.zeros(len(arrays["source_row"]), dtype=bool)
    if arrays["y"].shape != (len(arrays["source_row"]), 3):
        raise ValueError("K3 catalog and label width disagree.")
    return arrays["label_mask"].all(1) & (arrays["y"] == np.array(list(map(int, target)))).all(1)


def _grid(xy):
    if len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError("Training spatial coordinates must be finite and nonempty.")
    start = xy.min(0)
    span = np.maximum(xy.max(0) - start, 1e-12)
    cells = np.minimum(7, ((xy - start) / span * 8).astype(np.int64))
    return cells[:, 0] * 8 + cells[:, 1]


def _stratified_remove(candidates, count, codes, source, rng):
    if count > len(candidates):
        raise ValueError("Non-target abnormal capacity is below the requested removal.")
    groups = [candidates[codes[candidates] == code] for code in np.unique(codes[candidates])]
    if not count:
        return np.empty(0, dtype=np.int64)
    sizes = np.asarray([len(group) for group in groups])
    quota, remainder = np.divmod(sizes * count, len(candidates))
    missing = count - int(quota.sum())
    quota[np.argsort(-remainder, kind="stable")[:missing]] += 1
    parts = []
    for group, size in zip(groups, quota):
        ordered = group[np.argsort(source[group], kind="stable")]
        parts.append(rng.choice(ordered, size=int(size), replace=False))
    return np.sort(np.concatenate(parts))


def make_pair(entry, arrays, target, seed):
    source = arrays["source_row"]
    labels = arrays["y"]
    eligible = arrays["eligible"].astype(bool, copy=False)
    if not (arrays["label_mask"][eligible].all() and np.isin(labels[eligible], (0, 1)).all()):
        raise ValueError(f"TRAIN eligibility includes incomplete labels on {entry['entity']}.")
    positive = eligible & (labels == 1).any(1)
    normal_count = int((eligible & ~positive).sum())
    held = complete_target(entry, arrays, target) & eligible
    removed_b = np.flatnonzero(held)
    candidates = np.flatnonzero(positive & ~held)
    remaining = int(positive.sum()) - len(removed_b)
    if remaining == 0 or normal_count < 3 * remaining or len(candidates) < len(removed_b):
        raise InsufficientSupport(f"Insufficient same-board B/C training capacity on {entry['entity']}.")
    removed_c = np.empty(0, dtype=np.int64)
    spill = 0
    if len(removed_b):
        grid = _grid(arrays["xy"])
        codes = labels @ np.asarray([4, 2, 1], dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, entry["ordinal"], 171]))
        parts = []
        for block, count in Counter(grid[removed_b]).items():
            pool = candidates[grid[candidates] == block]
            taken = min(count, len(pool))
            parts.append(_stratified_remove(pool, taken, codes, source, rng))
            spill += count - taken
        local = np.concatenate(parts)
        unused = np.setdiff1d(candidates, local)
        residual = _stratified_remove(unused, spill, codes, source, rng)
        removed_c = np.sort(np.concatenate((local, residual)))
    if len(removed_b) != len(removed_c):
        raise RuntimeError("B/C board removal counts differ.")
    def identifiers(rows):
        return [[int(i), int(source[i])] for i in rows]
    def profile(removed):
        keep = eligible.copy()
        keep[removed] = False
        y = labels[keep]
        width = y.shape[1]
        codes = y @ (1 << np.arange(width - 1, -1, -1))
        state = {format(i, f"0{width}b"): int(n) for i, n in enumerate(np.bincount(codes, minlength=2**width)) if n}
        marginal = np.stack([(y == bit).sum(0) for bit in (0, 1)], axis=1).tolist()
        clean = []
        for layer in range(labels.shape[1]):
            known_upstream_clean = (y[:, :layer] == 0).all(1)
            clean.append([int((known_upstream_clean & (y[:, layer] == bit)).sum()) for bit in (0, 1)])
        return {"states": dict(sorted(state.items())), "interface_support": marginal,
                "clean_upstream_interface_support": clean}
    return identifiers(removed_b), identifiers(removed_c), {
        "board": entry["entity"], "removed_each_arm": len(removed_b),
        "C_same_board_other_block": spill, "epoch_positives": remaining,
        "epoch_normals": 3 * remaining,
        "before": profile(np.empty(0,dtype=np.int64)),
        "B": profile(removed_b), "C": profile(removed_c),
    }


def prepare_roles(catalog_path, catalog, fold, target, seed, output):
    entries = {row["entity"]: row for row in catalog["entities"]}
    arm_b, arm_c, validation = {}, {}, {}
    train_report, validation_report = [], []
    refs, sample_indices, weights = [], [], []
    offset = 0
    for board in fold["train"]:
        entry = entries[board]
        b, c, row = make_pair(entry, read_index(catalog_path, entry), target, seed)
        arm_b[board], arm_c[board] = b, c
        train_report.append(row)
    train_k3 = [row for row in train_report if entries[row["board"]]["K"] == 3]
    support = np.asarray([row["B"]["interface_support"] for row in train_k3]).sum(axis=0)
    if not train_k3 or (support == 0).any() or not sum(row["C"]["states"].get(target, 0) for row in train_k3):
        raise InsufficientSupport(f"{fold['fold']} lacks K3 support after holding out {target}.")
    k3_normal = k3_positive = 0
    for ordinal, board in enumerate(fold["validation"]):
        entry = entries[board]
        arrays = read_index(catalog_path, entry)
        target_mask = complete_target(entry, arrays, target)
        kept = np.flatnonzero(~target_mask)
        rejected = np.flatnonzero(target_mask)
        validation[board] = [[int(i), int(arrays["source_row"][i])] for i in rejected]
        ids, w = select_validation_rows(*(arrays[k][kept] for k in ("y", "label_mask", "xy", "quality")),
                                        ordinal=entry["ordinal"])
        if int(w.sum()) != len(kept):
            raise RuntimeError("Validation IPW does not represent the filtered board.")
        refs.append(np.column_stack((np.full(len(ids), ordinal), kept[ids])))
        sample_indices.append(offset + ids)
        weights.append(w)
        offset += len(kept)
        eligible = arrays["eligible"][kept]
        abnormal = (arrays["y"][kept] == 1).any(1)
        if entry["K"] == 3:
            k3_normal += int((eligible & ~abnormal).sum())
            k3_positive += int((eligible & abnormal).sum())
        validation_report.append({"board": board, "removed_targets": len(rejected), "population": len(kept),
                                  "selected": len(ids), "weight": int(w.sum())})
    if not k3_normal or not k3_positive:
        raise InsufficientSupport(f"{fold['fold']} lacks complete normal or abnormal K3 validation support.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for role, members in (("B", arm_b), ("C", arm_c), ("validation", validation)):
        write_json(output / f"{role}.json", {"fold": fold["fold"], "role": role, "target": target,
                                            "excluded_rows": members})
    panel = output / f"{fold['fold']}.npz"
    np.savez_compressed(panel, indices=np.concatenate(sample_indices), weights=np.concatenate(weights),
                        refs=np.concatenate(refs), entities=np.asarray(fold["validation"]))
    from a3_ultrasound.stream_training import catalog_signature
    write_json(output / "manifest.json", {
        "catalog_signature": catalog_signature(catalog), "policy": "V171 detargeted spatial IPW",
        "seed": 20261023, "folds": {fold["fold"]: {"sha256": file_sha(panel)}},
        "validation_view_sha256": file_sha(output / "validation.json"),
    })
    write_json(output / "support.json", {"fold": fold["fold"], "seed": seed, "target": target,
                                        "train": train_report, "validation": validation_report})
