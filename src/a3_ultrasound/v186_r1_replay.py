"""Replay C3 with archived first-interface reflection ratios on development predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from a3_ultrasound.v186 import digest, registry
from a3_ultrasound.v186_replay import read_fold, score, select, summarize, paired


def percentile_ranks(values):
    """Return equal-count midranks without the validation sampling weights."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return ((counts.cumsum() - 0.5 * counts) / values.size)[inverse]


def archived_ratios(path, board, rows):
    with np.load(path, allow_pickle=False) as data:
        ids, risk = data["source_row"], data["R1"]
        if ids.ndim != 1 or risk.shape != ids.shape or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid R1 source rows for {board}.")
        order = np.argsort(ids)
        requested = []
        for row in rows:
            prefix, source_id = row["sample_id"].rsplit(":", 1)
            if prefix != board:
                raise ValueError(f"Wrong R1 board for {row['sample_id']}.")
            requested.append(int(source_id))
        indices = np.searchsorted(ids[order], requested)
        if (indices >= len(ids)).any() or not np.array_equal(ids[order[indices]], requested):
            raise ValueError(f"Missing R1 source rows for {board}.")
        result = risk[order[indices]]
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError(f"Missing or invalid R1 ratios for {board}.")
        return result


def budgeted_points(rows, priority, budget):
    """Use the same weighted longest-prefix budget rule as C3."""
    order = sorted(range(len(rows)), key=lambda i: (-priority[i], rows[i]["sample_id"]))
    limit = sum(row["sample_weight"] for row in rows) * budget
    used = 0
    points = []
    for i in order:
        weight = rows[i]["sample_weight"]
        if used + weight > limit:
            break
        used += weight
        points.append(i)
    return points


def evaluate_actions(rows, choices, *, uniform=False):
    counts = Counter()
    for i, row in enumerate(rows):
        prediction = row["map_states"]
        weight = row["sample_weight"]
        if i not in choices:
            layers = (None,)
        elif uniform:
            layers = range(len(prediction))
        else:
            layers = (choices[i],)
        mass = weight / len(layers)
        for layer in layers:
            action = prediction.copy()
            if layer is not None:
                action[layer] = -1
            score(counts, row, action, mass, layer is not None)
            counts["all_errors"] += mass * sum(
                valid and result >= 0 and result != target
                for result, target, valid in zip(action, row["y"], row["label_mask"])
            )
    return counts


def policies(rows, reflection_risk=None, budget=0.10):
    uncertain = select(rows, budget)
    deepest = {i: len(rows[i]["base_logodds"]) - 1 for i in uncertain}
    result = {
        "uniform": evaluate_actions(rows, uncertain, uniform=True),
        "deepest": evaluate_actions(rows, deepest),
        "uncertain": evaluate_actions(rows, uncertain),
    }
    if reflection_risk is None:
        return result
    margin = -np.abs(np.array([row["base_logodds"] for row in rows]))
    depth_count = margin.shape[1]
    model_rank = percentile_ranks(margin.ravel()).reshape(margin.shape)
    physical_rank = percentile_ranks(reflection_risk)
    # The first-interface proxy says nothing about choosing between deeper slots.
    physical_by_layer = physical_rank[:, None] * (np.arange(depth_count) > 0)
    fused = (model_rank + physical_by_layer) / 2
    choice = fused.argmax(axis=1)
    result["fused_same_points"] = evaluate_actions(rows, {i: int(choice[i]) for i in uncertain})
    proposed = budgeted_points(rows, fused.max(axis=1), budget)
    result["fused_reselected"] = evaluate_actions(rows, {i: int(choice[i]) for i in proposed})
    physical_points = budgeted_points(rows, physical_rank, budget)
    result["physical_reselected"] = evaluate_actions(
        rows, {i: len(rows[i]["base_logodds"]) - 1 for i in physical_points})

    return result


def replay(predictions, risk_dir=None):
    folds = {}
    sources = {}
    manifest = {}
    if risk_dir is not None:
        rows = json.loads((risk_dir / "DELIVERY_MANIFEST.json").read_bytes())
        manifest = {row["path"]: row["sha256"] for row in rows}
    for fold in registry():
        boards = read_fold(predictions / fold / "predictions.jsonl", fold, frozen=True)
        folds[fold] = {}
        for board, rows in boards.items():
            risk = None
            if risk_dir is not None:
                path = risk_dir / f"{board}_all_features.npz"
                if board not in sources:
                    sources[board] = digest(path)
                    if sources[board] != manifest.get(path.name):
                        raise ValueError(f"Archived reflection-risk hash differs for {board}.")
                risk = archived_ratios(path, board, rows)
            folds[fold][board] = policies(rows, risk)
    result = summarize(folds)
    comparisons = ("deepest", "fused_same_points", "fused_reselected", "physical_reselected")
    result["paired_errors_per_100_review"] = {
        arm: paired(result, arm, "uncertain", "wrong_per_100_review")
        for arm in comparisons if arm in result["macro"]
    }
    # Whole-panel error capture remains comparable when the two policies pick different points.
    all_panel = {}
    for fold, boards in folds.items():
        all_panel[fold] = {
            board: {arm: 100 * c["all_errors"] / c["points"] for arm, c in arms.items()}
            for board, arms in boards.items()
        }
    result["all_panel_wrong_per_100_points"] = all_panel
    result["risk_sha256"] = sources
    result["score_rank"] = "equal_count_within_fold_board" if risk_dir is not None else None
    result["prediction_sha256"] = {fold: pins["predictions"] for fold, pins in registry().items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archived-risk", type=Path)
    args = parser.parse_args()
    result = replay(args.predictions, args.archived_risk)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
