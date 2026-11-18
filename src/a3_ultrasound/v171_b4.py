"""Summarize authorized paired B4 outer predictions in raw-MAP space."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from a3_ultrasound.data.v171_cohort import file_sha, write_json
from a3_ultrasound.v171_b3 import _deep_counts, read_panel
from a3_ultrasound.v171_protocol import GLOBAL, LP, PROTOCOL, cell_identity, load_config, matrix, validate_config

GROUPS = ((LP, "B"), (GLOBAL, "B"), (LP, "C"), (GLOBAL, "C"))
LABELS = ("LP_B", "G1D_B", "LP_C", "G1D_C")


def raw_board(panel, board, target):
    rows = panel.specimen == board
    complete = rows & panel.complete & (panel.k == 3)
    truth = panel.truth
    prediction = panel.raw_map
    valid = rows & ~panel.technical
    selected = complete & (truth == target)
    other = complete & (truth != target)
    seen = other & (truth != 0)
    normal = complete & (truth == 0)
    proposed = valid & complete & (prediction == target)
    matching = valid & (prediction == truth)
    counts = {}

    def endpoint(name, numerator, denominator):
        counts[name] = [int(numerator.sum()), int(denominator.sum())]

    if (rows & (panel.k == 3)).any():
        endpoint("target_exact", matching & selected, selected)
        endpoint("target_fpr", proposed & other, other)
        endpoint("target_precision", proposed & selected, proposed)
        endpoint("seen_abnormal_exact", matching & seen, seen)
        endpoint("normal_fpr_K3", valid & normal & (prediction != 0), normal)
    endpoint("normal_fpr_all_K", valid & rows & panel.complete & (truth == 0) & (prediction != 0),
             rows & panel.complete & (truth == 0))
    destination = {format(i, "03b"): int((selected & valid & (prediction == i)).sum()) for i in range(8)}
    destination["technical_invalid"] = int((selected & panel.technical).sum())
    if sum(destination.values()) != int(selected.sum()):
        raise ValueError("The target MAP destinations do not conserve the full target population.")
    deep = _deep_counts(panel, valid, rows)
    for line in deep:
        subset = f"deep_{line['interface']}_upstream_{line['upstream']}"
        counts[subset] = [line["auto_normal"], line["truth_defect"]]
        counts[subset + "_scorable"] = [line["auto_normal"], line["truth_defect"] - line["technical_review"]]
    return {"endpoints": counts, "target_map_destinations": destination, "deep": deep,
            "technical_invalid": int((rows & panel.technical).sum())}


def _common(board_results, metric, group_indices):
    seeds = defaultdict(dict)
    missing = []
    relevant = {board for (board, _), rows in board_results.items()
                if any(metric in line["endpoints"] for line in rows.values())}
    for (board, seed), rows in sorted(board_results.items()):
        if board not in relevant:
            continue
        required = [rows.get(GROUPS[i], {}).get("endpoints", {}).get(metric, [0, 0]) for i in group_indices]
        if any(n == 0 for _, n in required):
            missing.append({"board": board, "seed": seed})
        else:
            seeds[board][str(seed)] = [a/n for a, n in required]
    return {board: {"seeds": values, "values": np.mean(list(values.values()), axis=0).tolist()}
            for board, values in seeds.items()}, missing


def paired(board_results):
    """Use the common seeds of each contrast and weight held-out specimens equally."""
    design = {
        "LP_B_minus_G1D_B": ((0, 1), (1, -1)),
        "LP_B_minus_LP_C": ((0, 2), (1, -1)),
        "G1D_B_minus_G1D_C": ((1, 3), (1, -1)),
        "difference_in_differences": ((0, 1, 2, 3), (1, -1, -1, 1)),
    }
    metrics = sorted({m for rows in board_results.values() for result in rows.values() for m in result["endpoints"]})
    summary = {}
    for metric in metrics:
        contrasts = {}
        for name, (indices, coefficients) in design.items():
            boards, missing = _common(board_results, metric, indices)
            line = {"specimens": boards, "missing": missing, "estimate": None}
            if boards:
                values = np.asarray([row["values"] for row in boards.values()])
                diffs = values @ coefficients
                rng = np.random.default_rng(0)
                samples = rng.integers(0, len(boards), size=(10000, len(boards)))
                line.update(estimate=float(diffs.mean()),
                            means=dict(zip((LABELS[i] for i in indices), values.mean(0).tolist())),
                            bootstrap_95=np.quantile(diffs[samples].mean(1), [.025, .975]).tolist(),
                            per_specimen=dict(zip(boards, diffs.tolist())))
            contrasts[name] = line
        summary[metric] = contrasts
    return summary


def summarize(matrix_file, outer_root, authorization):
    manifest = json.loads(matrix_file.read_bytes())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unknown B4 matrix.")
    target = manifest["target"]
    rows = [row for row in manifest["cells"] if row["block"] == "B4"]
    if len(rows) != 32 or {row["run_id"] for row in rows} != {c.run_id for c in matrix() if c.block == "B4"}:
        raise ValueError("B4 matrix does not enumerate the registered 32 cells.")
    boards = defaultdict(dict)
    missing = []
    details = {}
    for row in rows:
        name = row["run_id"]
        location = outer_root / name
        if row["status"] == "NOT_ESTIMABLE" or not location.exists():
            missing.append({"run_id": name, "reason": row.get("reason", "no outer evaluation")})
            continue
        config = matrix_file.parent / row["config"]
        cfg = load_config(config)
        cell = cell_identity(cfg)
        if cell["run_id"] != name or cell["target"] != target or file_sha(config) != row["sha256"]:
            raise ValueError("B4 matrix configuration identity changed.")
        validate_config(cfg, fold_name=cell["fold"], seed=cell["seed"], head=cell["head"],
                        phase="outer", config_path=config, authorization=authorization)
        summary = json.loads((location / "summary.json").read_bytes())
        prediction = location / "predictions.jsonl"
        if (summary.get("status") != "EVALUATED" or summary.get("phase") != "outer"
                or summary["cell"] != cell or summary["config_sha256"] != row["sha256"]
                or summary["predictions_sha256"] != file_sha(prediction)):
            raise ValueError(f"B4 outer predictions do not belong to the authorized cell {name}.")
        fold = next(f for f in cfg["folds"] if f["fold"] == cell["fold"])
        panel = read_panel(prediction, cfg["catalog"], fold["test"])
        details[name] = {}
        for board in fold["test"]:
            result = raw_board(panel, board, int(target, 2))
            details[name][board] = result
            boards[(board, cell["seed"])][(cell["head"], cell["arm"])] = result
    return {"target": target, "matrix_sha256": file_sha(matrix_file), "missing_cells": missing,
            "group_order": LABELS, "cells": details, "paired": paired(boards)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matrix", "outer-root", "authorization", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.matrix, args.outer_root, args.authorization)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
