"""Paired development replay for the v1.8.6 report policies and Stage 1 controls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np
from pathlib import Path
from statistics import mean


from a3_ultrasound.v186 import EXPORTS, PROTOCOL, registry, validate_configuration


def read_fold(path, fold, *, frozen=False, prediction_sha=None):
    info = registry()[fold]
    boards = defaultdict(list)
    digest = hashlib.sha256()
    cohort = []
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            board = row["specimen_id"]
            cohort.append((row["sample_id"], board, row["y"], row["label_mask"], row["sample_weight"]))
            boards[board].append(row)
    actual = hashlib.sha256(json.dumps(sorted(cohort), separators=(",", ":")).encode()).hexdigest()
    if actual != info["cohort"] or (frozen and digest.hexdigest() != info["predictions"]):
        raise ValueError(f"{fold} development cohort or frozen predictions differ")
    if prediction_sha is not None and digest.hexdigest() != prediction_sha:
        raise ValueError(f"{fold} predictions differ from training receipt")
    return boards


def select(rows, budget=.10):
    ranked = []
    for i, row in enumerate(rows):
        logits = row["base_logodds"]
        if not all(map(math.isfinite, logits)):
            raise ValueError("Nonfinite logit")
        k = min(range(len(logits)), key=lambda k: abs(logits[k]))
        ranked.append((abs(logits[k]), row["sample_id"], i, k))
    limit = sum(r["sample_weight"] for r in rows) * budget
    used = 0
    chosen = {}
    for _, _, i, k in sorted(ranked):
        w = rows[i]["sample_weight"]
        if used + w > limit:
            break
        used += w
        chosen[i] = k
    return chosen


def score(counts, row, action, weight, reviewed):
    y, mask = row["y"], row["label_mask"]
    counts["points"] += weight
    if reviewed:
        counts["review"] += weight
        counts["held"] += weight * action.count(-1)
        counts["review_layers"] += weight * len(y)
        counts["review_known"] += weight * sum(mask)
        counts["review_errors"] += weight * sum(
            m and a >= 0 and a != t for a, m, t in zip(action, mask, y))
    if all(mask):
        if any(y):
            counts["abnormal"] += weight
            counts["exact"] += weight * (action == y)
        else:
            counts["normal"] += weight
            counts["fp"] += weight * (1 in action)
    for k in range(1, len(y)):
        if mask[k] and y[k] == 1:
            counts["deep"] += weight
            name = {1: "detected", 0: "miss", -1: "pending"}[action[k]]
            counts[name] += weight
            if k == 2:
                counts["third"] += weight
                counts["third_detected"] += weight * (action[k] == 1)


RATIOS = {"abnormal_exact": ("exact", "abnormal"), "normal_fpr": ("fp", "normal"),
          "deep_detected": ("detected", "deep"), "deep_missed": ("miss", "deep"),
          "deep_pending": ("pending", "deep"), "third_detected": ("third_detected", "third"),
          "review_rate": ("review", "points"), "held_per_review": ("held", "review"),
          "known_layer_fraction_review": ("review_known", "review_layers"),
          "wrong_per_100_held": ("review_errors", "held"),
          "wrong_per_100_review": ("review_errors", "review")}


def rates(counts):
    return {k: (counts.get(n, 0) / counts[d] * (100 if k.startswith("wrong_per_100_") else 1)
                if counts.get(d) else None) for k, (n, d) in RATIOS.items()}


def policies(rows):
    chosen = select(rows)
    arms = {key: Counter() for key in ("whole", "uncertain", "uniform")}
    for i, row in enumerate(rows):
        action = row["map_states"]
        w = row["sample_weight"]
        if i not in chosen:
            for c in arms.values():
                score(c, row, action, w, False)
            continue
        score(arms["whole"], row, [-1] * len(action), w, True)
        for k in range(len(action)):
            held = action.copy()
            held[k] = -1
            score(arms["uniform"], row, held, w / len(action), True)
            if k == chosen[i]:
                score(arms["uncertain"], row, held, w, True)
    return arms


def macro(rows):
    return {k: mean(v) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in RATIOS}


def summarize(folds):
    by_board = defaultdict(lambda: defaultdict(list))
    for boards in folds.values():
        for board, arms in boards.items():
            for arm, counts in arms.items():
                by_board[board][arm].append(rates(counts))
    per_board = {b: {arm: macro(rows) for arm, rows in arms.items()} for b, arms in by_board.items()}
    arms = sorted({a for b in per_board.values() for a in b})
    return {"fold_board_counts": folds, "per_board": per_board,
            "macro": {a: macro([b[a] for b in per_board.values() if a in b]) for a in arms}}


def paired(result, left, right, metric):
    by_board = defaultdict(list)
    matched = []
    for fold, boards in sorted(result["fold_board_counts"].items()):
        for board, arms in sorted(boards.items()):
            if left not in arms or right not in arms:
                continue
            a, b = rates(arms[left])[metric], rates(arms[right])[metric]
            if a is not None and b is not None:
                by_board[board].append(a - b)
                matched.append([fold, board])
    differences = {board: mean(v) for board, v in sorted(by_board.items())}
    if not differences:
        return {"fold_boards": [], "board_deltas": {}, "mean": None, "interval95": None}
    values = np.array(list(differences.values()))
    rng = np.random.default_rng(23)
    samples = rng.choice(values, (5000, len(values))).mean(axis=1)
    return {"fold_boards": matched, "board_deltas": differences, "mean": float(values.mean()),
            "interval95": np.quantile(samples, [.025, .975]).tolist()}


def c3(root):
    folds = {}
    for f in registry():
        boards = read_fold(root / f / "predictions.jsonl", f, frozen=True)
        folds[f] = {b: policies(rows) for b, rows in boards.items()}
    result = summarize(folds)
    result["paired_selection"] = paired(result, "uncertain", "uniform", "wrong_per_100_review")
    result["prediction_sha256"] = {f: item["predictions"] for f, item in registry().items()}
    return result

def map_from_delta(row, delta):
    # New log P(state) differs by state dot delta and a pointwise constant.
    # Use the original enumerator's state scores, including its tie behavior.
    k = len(delta)
    old = row["log_probabilities"]
    if len(old) != 2 ** k:
        raise ValueError("Unexpected state lattice")
    best = max(range(len(old)), key=lambda idx: old[idx] + sum(
        delta[j] for j in range(k) if (idx >> (k - j - 1)) & 1))
    return [(best >> (k - j - 1)) & 1 for j in range(k)]


def feature_replay(rows, baseline_probabilities, seed=23):
    groups = defaultdict(list)
    shifts = [[0.] * len(r["map_states"]) for r in rows]
    coverage = Counter()
    for i, r in enumerate(rows):
        k = len(shifts[i])
        p = r["stage1_donor_projection"]
        final = r["stage1_logits"]
        if (not all(math.isfinite(v) for v in (*p, *final)) or len(p) != k or p[0] != 0 or
                map_from_delta(r, shifts[i]) != r["map_states"]):
            raise ValueError("Identity replay does not reproduce the joint MAP")
        for state, value in enumerate(r["log_probabilities"]):
            expected = sum(final[j] for j in range(k) if state & (1 << (k-j-1)))
            if not math.isclose(value-r["log_probabilities"][0], expected, rel_tol=2e-4, abs_tol=2e-4):
                raise ValueError("Stored state distribution disagrees with the exported final logits")
        for b, v, z in zip(r["stage1_score_logits"], p, final, strict=True):
            if not math.isclose(b+v, z, abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError("Stage 1 projection does not reconstruct final logits")
        for j in range(1, k):
            probability = baseline_probabilities[r["sample_id"]][j-1]
            groups[(k, j, min(9, int(10 * probability)))].append(i)
    for (_, j, _), indices in groups.items():
        def order(i):
            return hashlib.sha256(f"{seed}/{rows[i]['sample_id']}".encode()).digest()
        ids = sorted(indices, key=order)
        for i, other in zip(ids, ids[1:] + ids[:1]):
            r = rows[i]
            delta = rows[other]["stage1_donor_projection"][j] - r["stage1_donor_projection"][j]
            shifts[i][j] = delta
            prefix = "positive_" if r["label_mask"][j] and r["y"][j] == 1 else None
            for label in ("all_", prefix):
                if label is None:
                    continue
                for event, active in (("targets", True), ("swappable", i != other), ("changed", delta != 0)):
                    if active:
                        coverage[label + event + "_n"] += 1
                        coverage[label + event + "_weight"] += r["sample_weight"]
    counts = Counter()
    for row, delta in zip(rows, shifts):
        score(counts, row, map_from_delta(row, delta), row["sample_weight"], False)
    return counts, coverage


def native(rows):
    counts = Counter()
    for row in rows:
        score(counts, row, row["map_states"], row["sample_weight"], False)
    return counts


CHECKPOINT_FILES = {k: files[0] for k, files in EXPORTS.items()}


def c1(root, baseline, checkpoint):
    folds, coverage, missing = {}, {}, []
    for fold in registry():
        sources = {}
        for arm in ("score", "feature"):
            job = root / f"{fold}_{arm}"
            if not (job / "run" / "summary.json").is_file():
                missing.append({"fold": fold, "arm": arm, "reason": "no_terminal_summary"})
                continue
            cfg_bytes = (job / "config.json").read_bytes()
            cfg = json.loads(cfg_bytes)
            receipt = json.loads((job / "run" / "summary.json").read_bytes())
            validate_configuration(cfg, fold, arm)
            if (receipt["protocol"] != PROTOCOL or cfg["source_commit"] != receipt["commit_hash"]
                    or hashlib.sha256(cfg_bytes).hexdigest() != receipt["config_sha256"]
                    or receipt["test_opened"] is not False):
                raise ValueError(f"Unexpected run for {fold}/{arm}")
            if receipt["status"] == "NO_ADMISSIBLE_CHECKPOINT" and checkpoint == "best":
                missing.append({"fold": fold, "arm": arm, "reason": "no_admissible_checkpoint"})
                continue
            if receipt["status"] not in ("COMPLETED", "NO_ADMISSIBLE_CHECKPOINT"):
                raise ValueError(f"Unfinished run for {fold}/{arm}")
            entry = receipt["stage1_exports"][checkpoint]
            filename, weights = EXPORTS[checkpoint]
            epoch = {"best": receipt["best_epoch"], "fixed15": 15, "last": receipt["epochs"]}[checkpoint]
            if ((entry["predictions"], entry["checkpoint"], entry["epoch"]) != (filename, weights, epoch)
                    or not entry["predictions_sha256"] or len(entry["checkpoint_sha256"]) != 64
                    or (checkpoint == "best" and
                        (receipt["selected_checkpoint"] != weights or receipt["guard_violated"] is not False))):
                raise ValueError("Mismatched Stage 1 checkpoint receipt")
            sources[arm] = read_fold(job / "run" / filename, fold, prediction_sha=entry["predictions_sha256"])
        anchors = read_fold(baseline / fold / CHECKPOINT_FILES["best"], fold, frozen=True)
        if checkpoint == "best":
            sources["baseline"] = anchors
        folds[fold], coverage[fold] = {}, {}
        for b in anchors:
            arms = {a: native(data[b]) for a, data in sources.items()}
            if "feature" in sources:
                arms["permuted"], coverage[fold][b] = feature_replay(
                    sources["feature"][b], {r["sample_id"]: r["base_probabilities"] for r in anchors[b]})
            folds[fold][b] = arms
    result = summarize(folds)
    controls = ["score", "permuted"] + (["baseline"] if checkpoint == "best" else [])
    result["paired_feature_minus_control"] = {
        arm: {m: paired(result, "feature", arm, m) for m in ("abnormal_exact", "normal_fpr", "deep_missed")}
        for arm in controls}
    return dict(result, checkpoint=checkpoint, missing_guarded=missing,
                full_primary_comparison=checkpoint == "best" and not missing, permutation_support=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", choices=("c1", "c3"))
    parser.add_argument("baseline", type=Path, help="Returned directory containing F1..F5/predictions.jsonl")
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINT_FILES), default="best")
    args = parser.parse_args()
    if args.comparison == "c1" and not args.jobs:
        parser.error("C1 requires --jobs")
    if args.comparison == "c3" and (args.jobs or args.checkpoint != "best"):
        parser.error("C3 uses the frozen best checkpoints only")
    result = c3(args.baseline) if args.comparison == "c3" else c1(args.jobs, args.baseline, args.checkpoint)
    with args.output.open("x", encoding="utf8") as out:
        json.dump(result, out, ensure_ascii=False, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
