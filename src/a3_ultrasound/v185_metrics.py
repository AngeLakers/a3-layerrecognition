"""Board-level accounting for one arm over one or more development folds (JSONL)."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def events(row, reference=None):
    truth, action, known = row["truth"], row["report"], row["known"]
    complete = all(known)
    counts = Counter(points=1, review_points=int(-1 in action), all_pending=int(all(a == -1 for a in action)))
    if row.get("calibration_guard_met") is False:
        counts["calibration_ineligible_points"] += 1
    if row.get("review_status", "").startswith("skipped"):
        counts["review_skipped_points"] += 1
    paired = reference is not None and row.get("review_status") == reference.get("review_status") == "fitted"
    if paired:
        counts["paired_points"] = 1
        counts["paired_normals"] = int(complete and not any(truth))
        for arm, actions in (("features", action), ("scores", reference["report"])):
            counts[arm+"_review"] = int(-1 in actions)
            counts[arm+"_normal_fp"] = int(bool(counts["paired_normals"]) and 1 in actions)
    fn = fp = 0
    for m, (y, a, reliable) in enumerate(zip(truth, action, known)):
        if not reliable:
            continue
        fn += int(y == 1 and a == 0)
        fp += int(y == 0 and a == 1)
        groups = ["shallow" if m == 0 else "deep", f"layer_{m+1}"]
        if m == len(action)-1:
            groups.append("deepest")
        for group in groups:
            counts[group+"_known"] += 1
            counts[group+"_committed"] += int(a != -1)
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
            if paired:
                counts["paired_triggers"] += 1
                counts["net_additional_correct"] += int(a == y) - int(reference["report"][m] == y)
    counts.update(fn=fn, fp=fp)
    if complete:
        counts["complete"] += 1
        counts["complete_correct"] += int(action == truth)
        if "logits" in row:
            counts["scored_states"] += 1
            stages = [("raw", row["logits"], 1.), ("cal", row["logits"], row["temperature"])]
            if "final_logits" in row:
                counts["final_scored_states"] += 1
                stages.append(("final_cal", row["final_logits"], row["temperature"]))
            for label, logits, temperature in stages:
                nll = 0.
                squared_mass = 1.
                for z, y in zip(logits, truth):
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
                      pure_fp=int(fp > 0 and fn == 0), abnormal_pure_fp=int(not normal and fp > 0 and fn == 0))
    return counts


RATES = {"raw_state_nll": ("raw_state_nll_sum","scored_states"), "cal_state_nll": ("cal_state_nll_sum","scored_states"),
         "raw_state_brier": ("raw_state_brier_sum","scored_states"), "cal_state_brier": ("cal_state_brier_sum","scored_states"),
         "complete_accuracy": ("complete_correct","complete"), "normal_fpr": ("normal_fp", "normal"), "review_rate": ("review_points", "points"),
         "all_pending_rate": ("all_pending", "points"), "exchange_rate": ("exchange", "abnormal"),
         "shallow_detection": ("shallow_detected", "shallow_positive"),
         "shallow_extra": ("shallow_fp", "shallow_negative"),
         "deep_detection": ("deep_detected", "deep_positive"), "deepest_detection": ("deepest_detected", "deepest_positive"),
         "deep_missed": ("deep_missed", "deep_positive"), "deep_pending": ("deep_pending", "deep_positive")}
RATES.update({
    "final_cal_state_nll": ("final_cal_state_nll_sum", "final_scored_states"),
    "final_cal_state_brier": ("final_cal_state_brier_sum", "final_scored_states"),
    "trigger_correct_rate": ("trigger_correct", "triggered"),
    "trigger_wrong_rate": ("trigger_wrong", "triggered"),
    "trigger_pending_rate": ("trigger_pending", "triggered"),
    "net_gain_vs_reference": ("net_additional_correct", "paired_triggers"),
    "mixed_rate": ("mixed", "abnormal"), "pure_fn_rate": ("pure_fn", "abnormal"),
    "pure_fp_rate": ("abnormal_pure_fp", "abnormal"),
})
for group in ("shallow", "deep", "deepest", *(f"layer_{i}" for i in range(1, 7))):
    RATES[group+"_coverage"] = (group+"_committed", group+"_known")


def paired_rows(file, reference):
    with file.open(encoding="utf-8") as current:
        if reference is None:
            for line in current:
                yield json.loads(line), None
            return
        with reference.open(encoding="utf-8") as baseline:
            for line, other in zip(current, baseline, strict=True):
                row, ref = json.loads(line), json.loads(other)
                if (row["review_head"] != "features" or ref["review_head"] != "scores"
                        or row["mode"] != "cost" or ref["mode"] != "cost"
                        or row["shuffled"] or ref["shuffled"]):
                    raise ValueError("C1 requires unshuffled features versus scores in cost mode")
                keys = ("sample_id", "board", "checkpoint", "budget", "temperature", "point_fee", "weight",
                        "truth", "known", "initial", "eligible", "logits")
                if any(row[k] != ref[k] for k in keys):
                    raise ValueError("Paired reports differ in identity, model, targets or routing")
                yield row, ref


def aggregate(files, reference_files=None):
    board_folds = []
    by_board = defaultdict(lambda: defaultdict(list))
    budget_values = set()
    calibration_failed = False
    rates_to_compute = dict(RATES)
    if reference_files is not None:
        for arm in ("features", "scores"):
            rates_to_compute[arm+"_review_rate"] = (arm+"_review", "paired_points")
            rates_to_compute[arm+"_normal_fpr"] = (arm+"_normal_fp", "paired_normals")
    for file, reference in zip(files, reference_files if reference_files is not None else [None]*len(files), strict=True):
        raw, weighted = defaultdict(Counter), defaultdict(Counter)
        for row, ref in paired_rows(file, reference):
            board, weight = row["board"], row["weight"]
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Invalid sampling weight")
            budget_values.add(row.get("budget"))
            calibration_failed |= row.get("calibration_guard_met") is False
            count = events(row, ref)
            raw[board].update(count)
            weighted[board].update({k: v*weight for k, v in count.items()})
        for board, count in weighted.items():
            rates = {}
            for metric, (top, bottom) in rates_to_compute.items():
                rates[metric] = count[top]/count[bottom] if count[bottom] else None
                if rates[metric] is not None:
                    by_board[board][metric].append(rates[metric])
            board_folds.append(dict(file=str(file),board=board,raw=raw[board],weighted=count,rates=rates))
    boards = {b:{m:sum(v)/len(v) for m,v in values.items()} for b,values in by_board.items()}
    metrics = {}
    for m in rates_to_compute:
        values = [b[m] for b in boards.values() if m in b]
        metrics[m] = dict(mean=sum(values)/len(values) if values else None,support_boards=len(values))
    if len(budget_values)>1:
        raise ValueError("Reports use different review budgets")
    budget = next(iter(budget_values), None)
    fpr, review = metrics["normal_fpr"]["mean"], metrics["review_rate"]["mean"]
    operating = dict(normal_fpr=fpr, review_rate=review, review_budget=budget,
                     fpr_within_limit=None if fpr is None else fpr <= .02+1e-9,
                     within_budget=None if budget is None or review is None else review <= budget+1e-9,
                     any_initial_calibration_failure=calibration_failed)
    result = dict(board_folds=board_folds,boards=boards,metrics=metrics,observed_final=operating)
    if reference_files is not None:
        rates = {arm: {m: metrics[arm+"_"+m]["mean"] for m in ("review_rate", "normal_fpr")}
                 for arm in ("features", "scores")}
        supported = budget is not None and all(v is not None for a in rates.values() for v in a.values())
        delta = rates["features"]["review_rate"] - rates["scores"]["review_rate"] if supported else None
        within = supported and all(a["normal_fpr"] <= .02+1e-9 and a["review_rate"] <= budget+1e-9 for a in rates.values())
        if not supported or not metrics["net_gain_vs_reference"]["support_boards"]:
            status = "insufficient_support"
        elif not within or calibration_failed:
            status = "outside_constraints"
        elif abs(delta) > 1e-9:
            status = "different_achieved_budgets"
        else:
            status = "matched_observed_budget"
        result["paired_observed"] = dict(**rates, review_rate_difference=delta, status=status,
                                         net_gain_descriptive=metrics["net_gain_vs_reference"]["mean"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="One held-out JSONL per fold for the same arm and policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", help="Same-fold aligned reports for the paired C1 comparison")
    args = parser.parse_args()
    result = aggregate(args.files, args.reference)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2)


if __name__ == "__main__":
    main()

