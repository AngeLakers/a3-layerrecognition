"""Descriptive envelope positions, with explicit source and search censoring."""
from __future__ import annotations
import math
import numpy as np
from scipy.signal import hilbert


def diagnostic_envelope(raw):
    """Timing-only baseline removal; never use this as the model RF input."""
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[-1] < 16 or not np.isfinite(x).all():
        raise ValueError('Expected finite RF with at least 16 samples.')
    x = x - np.median(x[..., :16], axis=-1, keepdims=True)
    pads = [(0, 0)] * x.ndim
    pads[-1] = (256, 256)
    return np.abs(hilbert(np.pad(x, pads, mode='reflect'), axis=-1)[..., 256:-256])


def packet_span(envelope, nominal, *, search_half=130, fraction=.2):
    """Span all above-threshold samples in the fixed search, extending its edges.

    This is a candidate span, not a physical echo-identity classifier. Multiple
    packets or surface tail in the search contribute to the same span.
    A live threshold hit at either source endpoint is conservatively censored:
    the unrecorded continuation cannot establish the full packet extent.
    """
    e = np.asarray(envelope, dtype=np.float64)
    if (e.ndim != 1 or not len(e) or not np.isfinite(e).all()
            or not math.isfinite(nominal) or search_half <= 0 or not 0 < fraction < 1):
        raise ValueError('Invalid envelope/search definition.')
    left = max(0, math.ceil(nominal-search_half))
    right = min(len(e)-1, math.floor(nominal+search_half))
    if left > right:
        return dict(valid=False, reason='empty_source_search')
    peak = left+int(np.argmax(e[left:right+1]))
    if e[peak] <= 1e-12:
        return dict(valid=False, reason='zero_envelope')
    hits = np.flatnonzero(e[left:right+1] > fraction*e[peak])+left
    start, end = int(hits[0]), int(hits[-1])
    search_edge = start == left or end == right
    while start > 0 and e[start-1] > fraction*e[peak]:
        start -= 1
    while end < len(e)-1 and e[end+1] > fraction*e[peak]:
        end += 1
    return dict(valid=True, peak=peak, start=start, end=end,
                search_left=left, search_right=right, search_edge_touched=search_edge,
                source_censored=start == 0 or end == len(e)-1,
                threshold=float(fraction*e[peak]))


def span_covered(span, nominal, half):
    return bool(span['valid'] and not span['source_censored']
                and span['start'] >= nominal-half and span['end'] <= nominal+half)


def front_overlap_energy_fraction(raw, nominal, half, *, front_end=180):
    """Fixed front-zone proxy, not a separation of overlapping physical echoes."""
    x = np.asarray(raw, dtype=np.float64)
    left, right = math.ceil(nominal-half), math.floor(nominal+half)
    if x.ndim != 1 or left < 0 or right >= len(x) or left > right:
        raise ValueError('Local window outside RF.')
    denominator = float(np.sum(x[left:right+1]**2))
    numerator = float(np.sum(x[left:min(right, front_end)+1]**2)) if left <= front_end else 0.
    return numerator/denominator if denominator > 0 else None
