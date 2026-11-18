"""Offline, whole-record MAP-confidence review using preserved joint posteriors."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from a3_ultrasound.data.v171_cohort import file_sha, write_json
from a3_ultrasound.v171_protocol import FOLDS, GLOBAL, LP, PROTOCOL, SEEDS

TECHNICAL_REASONS = {"invalid_rf", "nonfinite_posterior", "inference_failure"}


@dataclass
class Panel:
    specimen: np.ndarray
    k: np.ndarray
    y: np.ndarray
    known: np.ndarray
    complete: np.ndarray
    technical: np.ndarray
    log_joint: np.ndarray
    raw_map: np.ndarray

    @cached_property
    def abnormal(self):
        return self.complete & (self.y == 1).any(axis=1)

    @cached_property
    def truth(self):
        return (np.where(self.known, self.y, 0) * np.array([4, 2, 1])).sum(1) >> (3 - self.k)


def _catalog_rows(catalog_path, boards):
    catalog = json.loads(Path(catalog_path).read_bytes())
    entries = {entry["entity"]: entry for entry in catalog["entities"]}
    for board in boards:
        item = entries[board]
        root = Path(catalog_path).parent / item["index_dir"]
        arrays = {k: np.load(root / f"{k}.npy", allow_pickle=False, mmap_mode="r")
                  for k in ("source_row", "y", "label_mask", "design_y", "eligible")}
        yield board, item["K"], arrays


def read_panel(path, catalog_path, boards):
    """Join by original board/source identities and verify the complete expected population."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return _read_stream(path, catalog_path, boards)
    with np.load(path, allow_pickle=False) as pack:
        names = pack["specimen"].astype(str)
        n = len(names)
        k = pack["K"].astype(np.int64)
        log = np.full((n, 8), np.nan)
        values = pack["joint_log_posterior"]
        if values.ndim != 2 or values.shape[0] != n or values.shape[1] not in (4, 8):
            raise ValueError("Invalid stored posterior table.")
        log[:, :values.shape[1]] = values
        raw = pack["joint_map"].astype(np.int64)
        technical = pack["technical_invalid"].astype(bool) if "technical_invalid" in pack else np.zeros(n, bool)
        if technical.any():
            reasons = pack["technical_reason"].astype(str)
            if not np.isin(reasons[technical], sorted(TECHNICAL_REASONS)).all():
                raise ValueError("Every technical-invalid row needs a registered input or inference cause.")
        y = np.full((n, 3), -1, dtype=np.int8)
        known = np.zeros((n, 3), dtype=bool)
        complete = np.zeros(n, dtype=bool)
        if set(names) != set(boards):
            raise ValueError("Prediction specimens differ from the requested fold.")
        for board, count, data in _catalog_rows(catalog_path, boards):
            positions = np.flatnonzero(names == board)
            idx = pack["row_id"][positions].astype(np.int64)
            if not np.array_equal(np.sort(idx), np.arange(len(data["source_row"]))):
                raise ValueError(f"Missing, duplicate or unexpected original rows on {board}.")
            if (not np.array_equal(pack["source_row"][positions], data["source_row"][idx])
                    or not np.all(k[positions] == count)
                    or not np.array_equal(pack["y"][positions, :count], data["y"][idx])
                    or not np.array_equal(pack["label_mask"][positions, :count], data["label_mask"][idx])
                    or not np.array_equal(pack["eligible"][positions], data["eligible"][idx])):
                raise ValueError(f"Prediction and label catalog identities disagree on {board}.")
            y[positions, :count] = data["y"][idx]
            known[positions, :count] = (data["label_mask"][idx] & (data["design_y"][idx] >= 0)
                                        & np.isin(data["y"][idx], (0, 1)))
            complete[positions] = data["eligible"][idx] & known[positions, :count].all(1)
    return _validate(Panel(names, k, y, known, complete, technical, log, raw))


def _read_stream(path, catalog_path, boards):
    sources = list(_catalog_rows(catalog_path, boards))
    lengths = [len(data["source_row"]) for _, _, data in sources]
    n = sum(lengths)
    names = np.repeat(np.asarray(boards), lengths)
    y = np.full((n, 3), -1, dtype=np.int8)
    known = np.zeros((n, 3), dtype=bool)
    complete, technical, seen = (np.zeros(n, dtype=bool) for _ in range(3))
    log = np.full((n, 8), np.nan)
    raw = np.full(n, -1, dtype=np.int64)
    k = np.repeat([count for _, count, _ in sources], lengths)
    address, index_by_board = {}, {}
    offset = 0
    for board, count, data in sources:
        end = offset + len(data["source_row"])
        address[board] = {int(row): offset + i for i, row in enumerate(data["source_row"])}
        index_by_board[board] = offset, data
        y[offset:end, :count] = data["y"]
        known[offset:end, :count] = data["label_mask"] & (data["design_y"] >= 0) & np.isin(data["y"], (0, 1))
        complete[offset:end] = data["eligible"] & known[offset:end, :count].all(1)
        offset = end
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            board, sep, source = row["sample_id"].rpartition(":")
            i = address.get(board, {}).get(int(source)) if sep else None
            if i is None or seen[i] or board != row["specimen_id"] or row["K"] != k[i]:
                raise ValueError("Duplicate or mismatched evaluation row identity.")
            seen[i] = True
            start, index = index_by_board[board]
            local = i - start
            if (not np.array_equal(row["y"], y[i, :k[i]])
                    or not np.array_equal(row["label_mask"], index["label_mask"][local])
                    or not np.array_equal(row["nominal_y"], index["design_y"][local])):
                raise ValueError("Evaluation label or eligibility does not match the catalog.")
            if row.get("technical_invalid"):
                if row.get("technical_reason") not in TECHNICAL_REASONS:
                    raise ValueError("A technical-invalid row needs a registered input or inference cause.")
                technical[i] = True
            if row.get("log_probabilities") is not None:
                log[i, :2**k[i]] = row["log_probabilities"]
                bits = np.asarray(row["map_states"])
                raw[i] = int(bits @ (1 << np.arange(len(bits)-1, -1, -1)))
    if not seen.all():
        raise ValueError("Evaluation output omits records in the frozen population.")
    return _validate(Panel(names, k, y, known, complete, technical, log, raw))


def _validate(panel):
    if not np.isin(panel.k, (2, 3)).all():
        raise ValueError("Only K2/K3 joint state tables are supported.")
    for k in (2, 3):
        selected = np.flatnonzero(panel.k == k)
        if not len(selected):
            continue
        values = panel.log_joint[selected, :2**k]
        finite = np.isfinite(values).all(1)
        if (not np.array_equal(finite, ~panel.technical[selected])
                or not np.allclose(logsumexp(values[finite], axis=1), 0, atol=1e-5, rtol=0)
                or not np.array_equal(values[finite].argmax(1), panel.raw_map[selected[finite]])):
            raise ValueError("Legal-state posterior, raw MAP, or documented technical status disagree.")
        legal = values[finite]
        base = legal[:, 0]
        increments = legal[:, 1 << np.arange(k)] - base[:, None]
        state_bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
        if not np.allclose(legal, base[:, None] + increments @ state_bits.T, atol=4e-5, rtol=0):
            raise ValueError("V1.7.1 MAP confidence requires an independent LP/GLOBAL-style posterior.")
    return panel


def temperature(panel):
    """Fit on full eligible validation at unit weight, with no score-based filtering."""
    if not panel.complete.any():
        raise ValueError("No complete validation labels for temperature fitting.")
    groups = []
    for k in (2, 3):
        rows = (panel.k == k) & panel.complete
        p = panel.log_joint[rows, :2**k]
        if not np.isfinite(p).all():
            raise ValueError("A complete validation record has no legal posterior for temperature fitting.")
        groups.append((p, panel.truth[rows]))
    def objective(log_t):
        t = np.exp(log_t)
        total = 0.0
        for log_p, truth in groups:
            scaled = log_p / t
            total += float((logsumexp(scaled, axis=1) - scaled[np.arange(len(scaled)), truth]).sum())
        return total / int(panel.complete.sum())
    lower, upper = np.log(.1), np.log(10)
    opt = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    if not opt.success:
        raise RuntimeError("Temperature optimization failed.")
    best = min((objective(x), x) for x in (lower, 0.0, upper, float(opt.x)))
    return float(np.exp(best[1]))


def score(panel, t):
    result = np.full(len(panel.k), np.nan)
    for k in (2, 3):
        sel = (panel.k == k) & ~panel.technical
        values = panel.log_joint[sel, :2**k]
        chosen = values.argmax(1)
        rows = np.arange(len(chosen))
        shifted = (values - values[rows, chosen, None]) / t
        shifted[rows, chosen] = -np.inf
        result[sel] = -np.log1p(np.exp(shifted).sum(1))
    return result


def calibration_metrics(panel, t):
    """Joint-state proper scores and MAP reliability on complete, scorable records."""
    metrics = []
    available = panel.complete & ~panel.technical
    for k in (2, 3):
        selected = np.flatnonzero(available & (panel.k == k))
        if not len(selected):
            metrics.append({"K": k, "records": 0})
            continue
        values = panel.log_joint[selected, :2**k] / t
        log_q = values - logsumexp(values, axis=1, keepdims=True)
        probabilities = np.exp(log_q)
        columns = np.arange(len(selected))
        truth = panel.truth[selected]
        confidence = probabilities[columns, panel.raw_map[selected]]
        correct = panel.raw_map[selected] == truth
        bins = []
        ece = 0.0
        assigned = np.minimum(14, (confidence * 15).astype(int))
        for i in np.unique(assigned):
            keep = assigned == i
            n = int(keep.sum())
            mean, accuracy = float(confidence[keep].mean()), float(correct[keep].mean())
            ece += abs(mean - accuracy) * n / len(selected)
            bins.append({"lower": float(i/15), "upper": float((i+1)/15), "records": n,
                         "mean_confidence": mean, "accuracy": accuracy})
        brier = np.square(probabilities).sum(1) + 1 - 2 * probabilities[columns, truth]
        metrics.append({"K": k, "records": len(selected), "joint_nll": float(-log_q[columns, truth].mean()),
                        "joint_brier": float(np.maximum(brier, 0).mean()), "map_ece": ece, "bins": bins})
    return metrics


def fit_threshold(panel, scores):
    n = int(panel.abnormal.sum())
    tech = int((panel.abnormal & panel.technical).sum())
    if not n:
        return {"mode": "NA", "validation_abnormal": 0, "attained": None}
    values, counts = np.unique(scores[panel.abnormal & ~panel.technical], return_counts=True)
    rejected = tech + np.concatenate(([0], np.cumsum(counts)))
    # Smallest work on ties. Scores equal to threshold are accepted as a group.
    index = int(np.abs(2 * rejected - n).argmin())
    mode = "accept_all" if index == 0 else "reject_all" if index == len(values) else "threshold"
    return {"mode": mode, "log_threshold": float(values[index]) if mode == "threshold" else None,
            "validation_abnormal": n, "technical": tech, "attained": float(rejected[index]) / n}


def accept(panel, scores, rule):
    mode = rule["mode"]
    if mode == "accept_all":
        return ~panel.technical
    if mode == "reject_all":
        return np.zeros(len(scores), bool)
    if mode != "threshold":
        raise ValueError("No identifiable validation threshold.")
    return ~panel.technical & (scores >= rule["log_threshold"])


def _ratio(x, n):
    return x / n if n else None


def _counts(panel, accepted, rows):
    tech = panel.technical & rows
    low = rows & ~panel.technical & ~accepted
    accept_rows = rows & accepted
    pc = panel.complete & rows
    abnormal = panel.abnormal & rows
    normal = pc & ~abnormal
    correct = panel.raw_map == panel.truth
    ac = int((pc & accept_rows & correct).sum())
    ae = int((pc & accept_rows & ~correct).sum())
    rc = int((pc & low & correct).sum())
    re = int((pc & low & ~correct).sum())
    tc = int((pc & tech).sum())
    if ac + ae + rc + re + tc != pc.sum():
        raise RuntimeError("Complete-state count does not conserve the population.")
    scorable_abn = abnormal & ~panel.technical
    matched = int((abnormal & accept_rows).sum())
    random_wrong = int((scorable_abn & ~correct).sum()) * (matched / int(scorable_abn.sum()) if scorable_abn.any() else 0)
    n_abn, n_normal = int(abnormal.sum()), int(normal.sum())
    return {
        "rows": int(rows.sum()), "complete": int(pc.sum()), "complete_abnormal": n_abn, "complete_normal": n_normal,
        "accepted_correct": ac, "accepted_wrong": ae, "review_original_correct": rc,
        "review_original_wrong": re, "technical_complete": tc,
        "review_total": int((tech | low).sum()), "review_technical": int(tech.sum()), "review_confidence": int(low.sum()),
        "review_abnormal": _ratio(n_abn - matched, n_abn),
        "review_abnormal_technical": _ratio(int((tech & abnormal).sum()), n_abn),
        "review_abnormal_confidence": _ratio(int((low & abnormal).sum()), n_abn),
        "review_normal": _ratio(int((normal & ~accepted).sum()), n_normal),
        "review_normal_technical": _ratio(int((normal & tech).sum()), n_normal),
        "review_normal_confidence": _ratio(int((normal & low).sum()), n_normal),
        "wrong_auto_abnormal": _ratio(int((abnormal & accept_rows & ~correct).sum()), n_abn),
        "wrong_given_accepted": _ratio(ae, ac + ae),
        "wrong_given_accepted_abnormal": _ratio(int((abnormal & accept_rows & ~correct).sum()), matched),
        "correct_abnormal_referred": _ratio(int((abnormal & low & correct).sum()), n_abn),
        "false_alarm_normal": _ratio(int((normal & accept_rows & (panel.raw_map != 0)).sum()), n_normal),
        "random_matched_expected_wrong_auto_abnormal": _ratio(random_wrong, n_abn),
    }


def _deep_counts(panel, accepted, rows):
    tables = []
    for layer in (1, 2):
        mask = rows & (panel.k > layer) & panel.known[:, layer] & (panel.y[:, layer] == 1)
        if not mask.any():
            continue
        pred = ((np.maximum(panel.raw_map, 0) >> np.maximum(panel.k - layer - 1, 0)) & 1) != 0
        prefix_known = panel.known[:, :layer].all(1)
        prefix_num = np.where(prefix_known, panel.y[:, :layer] @ (1 << np.arange(layer-1, -1, -1)), -1)
        for value in (-2, *sorted(np.unique(prefix_num[mask]))):
            part = mask if value == -2 else mask & (prefix_num == value)
            size = int(part.sum())
            tech = part & panel.technical
            auto = part & accepted
            low = part & ~panel.technical & ~accepted
            missed = int((auto & ~pred).sum())
            bad = int((auto & pred).sum())
            invalid = int(tech.sum())
            review = int(low.sum())
            if missed + bad + invalid + review != size:
                raise RuntimeError("Deep true-positive denominator is inconsistent.")
            scorable = part & ~panel.technical
            raw_missed = int((scorable & ~pred).sum())
            prefix = "ALL" if value == -2 else "UNKNOWN" if value == -1 else format(int(value), f"0{layer}b")
            tables.append(dict(interface=layer+1, upstream=prefix, truth_defect=size, auto_normal=missed,
                               auto_defect=bad, confidence_review=review, technical_review=invalid,
                               auto_miss_rate=_ratio(missed,size), raw_miss_original_denominator=_ratio(raw_missed,size),
                               raw_miss_scorable=_ratio(raw_missed, int(scorable.sum()))))
    return tables


def report(panel, scores, rule):
    accepted = accept(panel, scores, rule)
    tables, strata, deep, curves = [], [], [], []
    for board in np.unique(panel.specimen):
        rows = panel.specimen == board
        tables.append({"board": str(board), **_counts(panel, accepted, rows)})
        deep.extend({"board": str(board), **item} for item in _deep_counts(panel, accepted, rows))
        for k in (2, 3):
            group = rows & (panel.k == k)
            if group.any():
                strata.append({"board": str(board), "K": k, "state":"ALL", **_counts(panel, accepted, group)})
            for code in np.unique(panel.truth[group & panel.complete]):
                s = group & panel.complete & (panel.truth == code)
                strata.append({"board": str(board), "K": k, "state":format(int(code), f"0{k}b"), **_counts(panel, accepted, s)})
        available = np.flatnonzero(rows & panel.complete & ~panel.technical)
        ordered = available[np.argsort(-scores[available], kind="stable")]
        wrong = (panel.raw_map[ordered] != panel.truth[ordered]).astype(np.int64)
        cuts = np.r_[0, np.flatnonzero(np.diff(scores[ordered]) != 0) + 1, len(ordered)]
        cuts = np.unique(cuts)
        cum_error = np.r_[0, np.cumsum(wrong)]
        picks = np.unique(np.linspace(0, len(cuts)-1, min(201, len(cuts)), dtype=np.int64))
        original = int((rows & panel.complete).sum())
        for index in picks:
            keep = int(cuts[index])
            errors = int(cum_error[keep])
            curves.append({"board": str(board), "accepted": keep, "accepted_wrong": errors,
                           "coverage_scorable":_ratio(keep,len(ordered)), "coverage_original":_ratio(keep,original),
                           "conditional_risk":_ratio(errors,keep), "wrong_auto_original":_ratio(errors,original)})
    return {"threshold":rule,"specimens":tables,"deep":deep,"true_state_strata":strata,"curves":curves}



def read_full_curves(panel, scores, output):
    """Persist every reachable joint threshold, with per-specimen denominators."""
    available = panel.complete & ~panel.technical
    thresholds = np.r_[np.inf, np.unique(scores[available])[::-1], -np.inf]
    boards = np.unique(panel.specimen)
    columns = ("accepted", "accepted_wrong", "accepted_abnormal", "accepted_wrong_abnormal",
               "accepted_normal", "accepted_false_alarm")
    counts = np.zeros((len(boards), len(columns), len(thresholds)), dtype=np.int32)
    populations = []
    for i, board in enumerate(boards):
        rows = panel.specimen == board
        idx = np.flatnonzero(rows & available)
        ordered = idx[np.argsort(-scores[idx], kind="stable")]
        accepted = np.searchsorted(-scores[ordered], -thresholds, side="right")
        wrong = panel.raw_map[ordered] != panel.truth[ordered]
        abnormal = panel.abnormal[ordered]
        normal = ~abnormal
        indicators = (np.ones(len(ordered), bool), wrong, abnormal, abnormal & wrong,
                      normal, normal & (panel.raw_map[ordered] != 0))
        for j, indicator in enumerate(indicators):
            counts[i, j] = np.r_[0, np.cumsum(indicator, dtype=np.int32)][accepted]
        populations.append([int((rows & mask).sum()) for mask in (
            panel.complete, panel.abnormal, panel.complete & ~panel.abnormal,
            panel.complete & panel.technical, panel.abnormal & panel.technical,
            panel.complete & ~panel.abnormal & panel.technical,
        )])
    with Path(output).open("xb") as handle:
        np.savez_compressed(handle, boards=boards, log_threshold=thresholds, count_columns=columns,
                            counts=counts, population=populations,
                            population_columns=("complete", "abnormal", "normal", "technical",
                                                "technical_abnormal", "technical_normal"))
    return {"file": Path(output).name, "sha256": file_sha(output), "thresholds": len(thresholds)}


def _load_json(path):
    return json.loads(Path(path).read_bytes())


def _archive_run_id(fold, seed, head):
    return f"{fold}_s{seed}_{head}" + ("_none" if head == LP else "")


def compare_frozen(paths):
    """Check whether the primary LP/G1D outer operating points have comparable workload."""
    reports = [_load_json(path) for path in paths]
    identities = {(r["fold"], r["seed"], r["head"]): r for r in reports}
    expected = {(fold, seed, head) for fold in FOLDS for seed in SEEDS for head in (LP, GLOBAL)}
    if (len(identities) != len(reports) or set(identities) - expected
            or any(r["run_id"] != _archive_run_id(r["fold"], r["seed"], r["head"]) for r in reports)
            or any("outer" not in r and r["threshold"]["mode"] != "NA" for r in reports)
            or len({r["catalog_sha256"] for r in reports}) > 1):
        raise ValueError("B3 comparison requires unique compatible ResNet outer reports.")
    missing = sorted(expected - set(identities))
    if missing or any("outer" not in r for r in reports):
        return {"status": "NA", "reason": "Incomplete LP/G1D locked-threshold evidence",
                "missing": missing, "unattainable": sorted(k for k, r in identities.items() if "outer" not in r)}
    pooled = defaultdict(lambda: np.zeros((2, 3)))
    for fold in FOLDS:
        for seed in SEEDS:
            rows = [{row["board"]: row for row in identities[fold, seed, h]["outer"]["specimens"]}
                    for h in (LP, GLOBAL)]
            if rows[0].keys() != rows[1].keys():
                raise ValueError("B3 compared models did not evaluate the same specimens.")
            for board in rows[0]:
                lp, g1d = [r[board] for r in rows]
                n = lp["complete_abnormal"]
                if n != g1d["complete_abnormal"]:
                    raise ValueError("B3 complete-abnormal denominators differ within a pair.")
                if n:
                    for i, line in enumerate((lp, g1d)):
                        pooled[board, seed][i] += [n, line["review_abnormal"]*n,
                                                   line["wrong_auto_abnormal"]*n]
    if not pooled:
        return {"status": "NA", "reason": "No common complete abnormal specimens"}
    boards = {}
    for board in sorted({name for name, _ in pooled}):
        rates = [values[:, 1:] / values[:, :1] for (name, _), values in pooled.items() if name == board]
        boards[board] = np.mean(rates, axis=0).tolist()
    overall = np.mean(list(boards.values()), axis=0)
    gap = float(abs(overall[0, 0] - overall[1, 0]))
    comparable = gap <= .005 + 1e-12
    return {"status": "COMPARABLE" if comparable else "NA", "gap_abnormal_review": gap,
            "group_order": ["LP", "G1D"], "metric_order": ["T_A", "wrong_auto_abnormal"],
            "specimens": boards, "means": overall.tolist(),
            "same_budget_wrong_auto_abnormal_LP_minus_G1D": float(overall[0, 1] - overall[1, 1]) if comparable else None}


def verify_source(path, *, phase, fold, head, seed, run_id, validation=None, authorization=None):
    """Check model and phase identities before reading the predictions as a population."""
    if phase not in ("validation", "outer"):
        raise ValueError("Unknown B3 prediction phase.")
    path = Path(path)
    if phase == "outer":
        if authorization is None:
            raise PermissionError("B3 requires authorization for the saved outer predictions.")
        receipt = _load_json(authorization)
        grants = receipt.get("cells", {})
        if (receipt.get("protocol") != PROTOCOL or receipt.get("phase") != "b3-outer"
                or receipt.get("authorized") is not True or run_id not in grants):
            raise PermissionError("B3 authorization does not cover this cell.")
        if grants[run_id] != file_sha(path):
            raise PermissionError("B3 authorization does not cover this prediction digest.")
    digest = file_sha(path)
    if path.suffix == ".jsonl":
        summary = _load_json(path.parent / "summary.json")
        cell = summary["cell"]
        expected_phase = "calibration-validation" if phase == "validation" else "outer"
        if (summary.get("status") != "EVALUATED" or summary.get("phase") != expected_phase
                or (cell["fold"], cell["head"], cell["seed"], cell["run_id"]) != (fold, head, seed, run_id)
                or cell.get("arm") is not None or summary.get("predictions_sha256") != digest):
            raise ValueError("B3 stream result is not the expected raw whole-population cell.")
        if phase == "outer":
            fitted = _load_json(Path(validation).parent / "summary.json")
            for key in ("config_sha256", "guarded_weights_sha256", "epoch", "cell"):
                if summary[key] != fitted[key]:
                    raise ValueError("B3 validation and outer used different guarded checkpoints.")
        return digest
    expected_run = _archive_run_id(fold, seed, head)
    if path.name != "raw.npz" or run_id != expected_run or path.parent.name != run_id:
        raise ValueError("Unknown archived B3 raw cell.")
    alias = "LP" if head == LP else "G1D"
    with np.load(path, allow_pickle=False) as pack:
        if (str(pack["fold"]), str(pack["head"]).upper(), int(pack["seed"])) != (fold, alias, seed):
            raise ValueError("B3 archived raw tool exported a different model cell.")
        files = set(pack.files)
        if phase == "validation" and ("epoch" not in files or "cell" in files):
            raise ValueError("B3 validation requires the full archived validation export.")
        if phase == "outer" and ("cell" not in files or str(pack["cell"]) != run_id):
            raise ValueError("B3 outer requires the matching archived outer export.")
    if phase == "validation":
        sidecar = path.parent / ("FULL_SUBSET_GATE.json" if head == LP else "WITNESS.json")
        if (not sidecar.is_file() or _load_json(sidecar).get("passed") is not True
                or not (path.parent / "CALIBRATION.json").is_file()):
            raise ValueError("Archived validation evidence is incomplete.")
    else:
        completed = _load_json(path.parent / "COMPLETED.json")
        manifest = _load_json(path.parent / "MANIFEST.json")
        inputs = _load_json(path.parent / "INPUT_MANIFEST.json")["before"]
        cal_sha = file_sha(Path(validation).parent / "CALIBRATION.json")
        used_calibration = [h for name, h in inputs.items() if name.endswith("/CALIBRATION.json")]
        if (completed.get("outer") is not True or completed["cell"] != run_id
                or manifest.get("raw.npz") != digest or cal_sha not in used_calibration):
            raise ValueError("Archived outer receipt does not bind this raw export and validation calibration.")
    return digest


def main():
    if sys.argv[1:2] == ["compare"]:
        parser = argparse.ArgumentParser(description="Compare validation-locked ResNet outer reports.")
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("reports", nargs="+", type=Path)
        args = parser.parse_args(sys.argv[2:])
        write_json(args.output, compare_frozen(args.reports))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "catalog", "validation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--head", choices=(LP, GLOBAL), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outer-predictions", type=Path)
    parser.add_argument("--outer-authorization", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = _load_json(args.reference)
    fold = next(row for row in reference["folds"] if row["fold"] == args.fold)
    from a3_ultrasound.stream_training import catalog_signature
    if catalog_signature(_load_json(args.catalog)) != reference["catalog_signature"]:
        raise ValueError("B3 catalog differs from the fixed40 reference.")
    identity = {"fold": args.fold, "head": args.head, "seed": args.seed, "run_id": args.run_id}
    val_sha = verify_source(args.validation, phase="validation", **identity)
    val = read_panel(args.validation, args.catalog, fold["validation"])
    t = temperature(val)
    scores = score(val,t)
    rule = fit_threshold(val,scores)
    result = {**identity, "temperature":t, "threshold":rule, "validation_sha256":val_sha,
              "reference_sha256":file_sha(args.reference),"catalog_sha256":file_sha(args.catalog),
              "temperature_validation_rows":int(val.complete.sum())}
    def store_phase(name, panel, values):
        curves_path = args.output.with_name(f"{args.output.stem}.{name}.curves.npz")
        result[f"{name}_curves"] = read_full_curves(panel, values, curves_path)
        result[f"{name}_calibration"] = {"raw": calibration_metrics(panel, 1), "temperature": calibration_metrics(panel, t)}
        result[f"{name}_zero_confidence_review"] = report(panel, values, {"mode":"accept_all"})["specimens"]
        if rule["mode"] != "NA":
            result[name] = report(panel, values, rule)
    if args.outer_predictions:
        result["outer_predictions_sha256"] = verify_source(
            args.outer_predictions, phase="outer", validation=args.validation,
            authorization=args.outer_authorization, **identity)
    store_phase("validation", val, scores)
    if args.outer_predictions:
        test = read_panel(args.outer_predictions, args.catalog, fold["test"])
        store_phase("outer", test, score(test,t))
    write_json(args.output,result)


if __name__ == "__main__":
    main()
