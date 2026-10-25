"""RF-only first-negative-half-cycle timing; this module has no label or I/O API."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TroughBatch:
    positions: np.ndarray
    crossings: np.ndarray
    minima: np.ndarray
    plateau: np.ndarray
    failed: np.ndarray
    unclosed: np.ndarray


def first_negative_trough(windows: np.ndarray, origins=0) -> TroughBatch:
    """Return first < -0.3*A segment's trough and potential clipping plateau.

    Caller must compare minima with the complete RF record minimum before
    applying the saturated fallback. No equality tolerance is introduced.
    """
    x = np.asarray(windows)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite batch of RF windows.')
    origin = np.broadcast_to(np.asarray(origins, dtype=np.int64), (len(x),))
    threshold = -.3 * np.max(np.abs(x.astype(np.float64)), axis=1)
    below = x < threshold[:, None]
    failed = ~below.any(axis=1)
    start = below.argmax(axis=1)
    cols = np.arange(x.shape[1])[None, :]
    recovered = (cols > start[:, None]) & ~below
    unclosed = ~recovered.any(axis=1) & ~failed
    end = np.where(recovered.any(axis=1), recovered.argmax(axis=1), x.shape[1])
    inside = (cols >= start[:, None]) & (cols < end[:, None])
    segment = np.where(inside, x, np.inf)
    trough = segment.argmin(axis=1)
    minima = segment[np.arange(len(x)), trough]
    eq = inside & (x == minima[:, None])
    plateau = ((eq[:, :-2] & eq[:, 1:-1] & eq[:, 2:]).any(axis=1)
               if x.shape[1] >= 3 else np.zeros(len(x), dtype=bool))
    return TroughBatch(np.where(failed, -1, trough + origin),
                       np.where(failed, -1, start + origin), minima,
                       plateau & ~failed, failed, unclosed)


def resolve_saturation(found: TroughBatch, full_record_minima: np.ndarray):
    """Use the crossing only for a >=3-point plateau at the record minimum."""
    saturated = found.plateau & (found.minima == np.asarray(full_record_minima))
    return np.where(saturated, found.crossings, found.positions), saturated


def complete_negative_trough(trace, left, right):
    """Cross inside [left,right], then follow that half-cycle until recovery.

    The threshold is fixed from the search interval even when the segment
    continues outside it. Record end is the only remaining truncation.
    """
    raw = np.asarray(trace)
    window = raw[left:right+1]
    threshold = -.3 * float(np.max(np.abs(window.astype(np.float64))))
    crossings = np.flatnonzero(window < threshold)
    if not len(crossings):
        return dict(position=-1, crossing=-1, saturated=False, unclosed=False)
    crossing = left + int(crossings[0])
    recovered = np.flatnonzero(raw[crossing:] >= threshold)
    end = crossing + int(recovered[0]) if len(recovered) else len(raw)
    segment = raw[crossing:end]
    minimum = segment.min()
    eq = segment == minimum
    plateau = bool(len(eq) >= 3 and np.any(eq[:-2] & eq[1:-1] & eq[2:]))
    saturated = bool(plateau and minimum == raw.min())
    return dict(position=crossing if saturated else crossing+int(segment.argmin()),
                crossing=crossing, saturated=saturated, unclosed=not len(recovered))


def timing_stats(values):
    x = np.asarray(values)
    x = x[np.isfinite(x) & (x >= 0)]
    if not len(x):
        return dict(count=0, median=None, q25=None, q75=None, iqr=None, q01=None, q99=None)
    q = np.quantile(x, [.01, .25, .5, .75, .99])
    return dict(count=len(x), median=float(q[2]), q25=float(q[1]),
                q75=float(q[3]), iqr=float(q[3]-q[1]), q01=float(q[0]), q99=float(q[4]))
