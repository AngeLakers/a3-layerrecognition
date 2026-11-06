"""IPW joint metrics. Complete labels only; binary state order is MSB first."""
from __future__ import annotations
import numpy as np


def _inputs(y, probabilities, weights):
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (y.ndim != 2 or y.shape[1] not in (2, 3) or not len(y)
            or not np.isin(y, [0, 1]).all() or p.shape != (len(y), 2**y.shape[1])
            or w.shape != (len(y),) or not np.isfinite(w).all() or (w <= 0).any()
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-6)):
        raise ValueError('Complete K2/K3 labels, normalized posterior and positive IPW required.')
    return y.astype(np.int64), p, w


def ratio(event, eligible, weights):
    event, eligible = np.asarray(event, bool), np.asarray(eligible, bool)
    numerator = float(weights[event & eligible].sum())
    denominator = float(weights[eligible].sum())
    return dict(value=numerator / denominator if denominator else None,
                weighted_numerator=numerator, weighted_denominator=denominator,
                raw_numerator=int((event & eligible).sum()), raw_denominator=int(eligible.sum()))


def calibration(confidence, correct, weights, bins=15):
    """Split confidence tie groups proportionally across equal IPW-mass bins.

    This makes ECE invariant to input order within a tie. Fractional mass is
    bin bookkeeping, not additional observations or effective sample size.
    """
    c, z, w = map(np.asarray, (confidence, correct, weights))
    total = float(w.sum())
    edges = np.linspace(0, total, bins + 1)
    mass = np.zeros(bins); success = mass.copy(); score = mass.copy()
    start = 0.
    order=np.argsort(c,kind="stable"); c,z,w=c[order],z[order],w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(c))+1]
    widths=np.add.reduceat(w,starts); successes=np.add.reduceat(w*z,starts)
    for value,width,success_count in zip(c[starts],widths,successes):
        width=float(width); fraction=float(success_count/width)
        overlap = np.maximum(0, np.minimum(edges[1:], start + width) - np.maximum(edges[:-1], start))
        mass += overlap; success += overlap * fraction; score += overlap * value
        start += width
    records = [dict(weight=float(m), accuracy=float(a/m), confidence=float(s/m))
               for m, a, s in zip(mass, success, score) if m > 0]
    return dict(ece=float(np.abs(success-score).sum()/total), bins=records,
                convention='15 equal IPW mass bins; proportional confidence-tie splitting')


def risk_coverage(confidence, error, weights):
    """Threshold endpoints and tie-randomized linear integration of generalized risk."""
    c, e, w = map(np.asarray, (confidence, error, weights))
    total = float(w.sum())
    if not total:
        return dict(augrc=None, coverage=[], risk=[], generalized_risk=[])
    coverage = [0.]; generalized = [0.]; risk = [None]; accepted = wrong = 0.
    order=np.argsort(-c,kind="stable");c,e,w=c[order],e[order],w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(c))+1]
    for width,mistakes in zip(np.add.reduceat(w,starts),np.add.reduceat(w*e,starts)):
        accepted += float(width); wrong += float(mistakes)
        coverage.append(accepted/total); generalized.append(wrong/total); risk.append(wrong/accepted)
    area = sum((b-a)*(u+v)/2 for a,b,u,v in zip(coverage[:-1],coverage[1:],generalized[:-1],generalized[1:]))
    return dict(augrc=float(area), coverage=coverage, risk=risk, generalized_risk=generalized,
                denominator=total, tie_convention='accept confidence ties together; linear randomized tie integration')


def joint_metrics(y, probabilities, weights, *, log_probabilities=None):
    y, p, w = _inputs(y, probabilities, weights)
    k = y.shape[1]; powers = 2 ** np.arange(k-1, -1, -1)
    truth = y @ powers
    # Exact inference selects in log space; float32 exp may collapse near ties.
    prediction = (p if log_probabilities is None else np.asarray(log_probabilities)).argmax(1)
    pred_bits = ((prediction[:, None] & powers) != 0).astype(int)
    correct = prediction == truth; positive = truth != 0; normal = ~positive
    all_rows = np.ones(len(y), bool); detected = prediction != 0
    attribution = positive & detected & ~correct
    confidence = p.max(1); high = confidence >= .9
    if log_probabilities is None:
        with np.errstate(divide='ignore'):
            logp = np.log(p)
    else:
        logp = np.asarray(log_probabilities, float)
        if logp.shape != p.shape or np.isnan(logp).any() or np.isposinf(logp).any() or not np.allclose(np.exp(logp), p, atol=2e-6, rtol=0):
            raise ValueError('Log posteriors do not match probabilities.')
    nll = -logp[np.arange(len(y)), truth]
    brier = (p*p).sum(1) - 2*p[np.arange(len(y)), truth] + 1
    def mean(values):
        numerator = float(np.dot(w, values))
        return dict(value=numerator/float(w.sum()), weighted_numerator=numerator, weighted_denominator=float(w.sum()))
    result = dict(
        positive_joint_map_accuracy=ratio(correct, positive, w),
        normal_joint_false_positive_rate=ratio(detected, normal, w),
        abnormal_three_way=dict(correct=ratio(correct, positive, w),
            detected_wrong=ratio(attribution, positive, w), missed=ratio(~detected, positive, w)),
        detection_recall=ratio(detected, positive, w),
        attribution_overcall=ratio(((pred_bits==1)&(y==0)).any(1), attribution, w),
        attribution_undercall=ratio(((pred_bits==0)&(y==1)).any(1), attribution, w),
        interface_map_recall=[ratio(pred_bits[:,j]==1, y[:,j]==1, w) for j in range(k)],
        joint_nll=mean(nll), joint_brier=mean(brier),
        calibration=calibration(confidence, correct, w),
        hce_0_9=ratio(~correct, high, w), coverage_0_9=ratio(high, all_rows, w),
        risk_coverage={name:risk_coverage(confidence[mask], (~correct)[mask], w[mask])
                       for name,mask in [('all', all_rows),('abnormal',positive),('normal',normal)]})
    result['deepest_map_recall'] = result['interface_map_recall'][-1]
    result['true_upper_groups'] = {}
    for upper in range(2**(k-1)):
        mask = (truth >> 1) == upper
        den = float(w[mask].sum())
        result['true_upper_groups'][f'K{k}:{upper:0{k-1}b}'] = dict(
            deepest_recall=ratio(pred_bits[:,-1]==1, mask & (y[:,-1]==1), w),
            deepest_map_error=ratio(pred_bits[:,-1]!=y[:,-1], mask, w),
            joint_error=ratio(~correct, mask, w),
            joint_nll=float(np.dot(w[mask],nll[mask])/den) if den else None)
    return result
