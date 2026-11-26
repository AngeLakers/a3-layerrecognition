"""Board-level accounting for one arm over one or more development folds (JSONL)."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def events(row):
    truth, action, known = row["truth"], row["report"], row["known"]
    complete = all(known)
    counts = Counter(points=1, review_points=int(-1 in action), all_pending=int(all(a == -1 for a in action)))
    if row.get("calibration_guard_met") is False:
        counts["calibration_ineligible_points"] += 1
    fn = fp = 0
    for m, (y, a, reliable) in enumerate(zip(truth, action, known)):
        if not reliable:
            continue
        fn += int(y == 1 and a == 0)
        fp += int(y == 0 and a == 1)
        groups = ["shallow"] if m == 0 else ["deep"]
        if m == len(action)-1:
            groups.append("deepest")
        for group in groups:
            counts[group+"_positive"] += int(y == 1)
            counts[group+"_negative"] += int(y == 0)
            counts[group+"_fp"] += int(y == 0 and a == 1)
            counts[group+"_detected"] += int(y == 1 and a == 1)
            counts[group+"_missed"] += int(y == 1 and a == 0)
            counts[group+"_pending"] += int(y == 1 and a == -1)
        if row.get("eligible",row["active"])[m]:
            counts["triggered"] += 1
            counts["trigger_correct"] += int(a == y)
            counts["trigger_wrong"] += int(a in (0, 1) and a != y)
            counts["trigger_pending"] += int(a == -1)
    counts.update(fn=fn, fp=fp)
    if complete:
        counts["complete"] += 1
        counts["complete_correct"] += int(action == truth)
        if "logits" in row:
            counts["scored_states"] += 1
            for label, temperature in (("raw", 1.), ("cal", row["temperature"])):
                nll = 0.
                squared_mass = 1.
                for z, y in zip(row["logits"], truth):
                    signed = z/temperature * (1 if y else -1)
                    nll += max(0., -signed) + math.log1p(math.exp(-abs(signed)))
                    q = math.exp(-abs(signed))
                    squared_mass *= (1+q*q)/((1+q)**2)
                counts[label+"_state_nll_sum"] += nll
                counts[label+"_state_brier_sum"] += max(0., 1 - 2*math.exp(-nll) + squared_mass)
        normal = not any(truth)
        counts.update(normal=int(normal), normal_fp=int(normal and 1 in action),
                      abnormal=int(not normal), exchange=int(fn == fp and fn > 0),
                      mixed=int(fn > 0 and fp > 0), pure_fn=int(fn > 0 and fp == 0),
                      pure_fp=int(fp > 0 and fn == 0))
    return counts


RATES = {"raw_state_nll": ("raw_state_nll_sum","scored_states"), "cal_state_nll": ("cal_state_nll_sum","scored_states"),
         "raw_state_brier": ("raw_state_brier_sum","scored_states"), "cal_state_brier": ("cal_state_brier_sum","scored_states"),
         "complete_accuracy": ("complete_correct","complete"), "normal_fpr": ("normal_fp", "normal"), "review_rate": ("review_points", "points"),
         "all_pending_rate": ("all_pending", "points"), "exchange_rate": ("exchange", "abnormal"),
         "shallow_detection": ("shallow_detected", "shallow_positive"),
         "shallow_extra": ("shallow_fp", "shallow_negative"),
         "deep_detection": ("deep_detected", "deep_positive"), "deepest_detection": ("deepest_detected", "deepest_positive"),
         "deep_missed": ("deep_missed", "deep_positive"), "deep_pending": ("deep_pending", "deep_positive")}


def aggregate(files):
    board_folds = []
    by_board = defaultdict(lambda: defaultdict(list))
    for file in files:
        raw, weighted = defaultdict(Counter), defaultdict(Counter)
        with file.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                board, weight = row["board"], row["weight"]
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("Invalid sampling weight")
                count = events(row)
                raw[board].update(count)
                weighted[board].update({k: v*weight for k, v in count.items()})
        for board, count in weighted.items():
            rates = {}
            for metric, (top, bottom) in RATES.items():
                rates[metric] = count[top]/count[bottom] if count[bottom] else None
                if rates[metric] is not None:
                    by_board[board][metric].append(rates[metric])
            board_folds.append(dict(file=str(file),board=board,raw=raw[board],weighted=count,rates=rates))
    boards = {b:{m:sum(v)/len(v) for m,v in values.items()} for b,values in by_board.items()}
    metrics = {}
    for m in RATES:
        values = [b[m] for b in boards.values() if m in b]
        metrics[m] = dict(mean=sum(values)/len(values) if values else None,support_boards=len(values))
    return dict(board_folds=board_folds,boards=boards,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="One held-out JSONL per fold for the same arm and policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.files)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2)


if __name__ == "__main__":
    main()

