"""Fixed-window RF energy ratios; normalization is fitted on training only."""
from __future__ import annotations

import math
import numpy as np


def packed_energy_bounds(interface_meta, valid_length, k):
    """Decode exactly the float32 bounds used by A3PCCModel._sample_windows.

    Using pre-pack float64 centers can select another integer endpoint after
    continuous jitter. Keep each multiply/subtract in float32 before energy
    accumulation converts the decoded endpoints to float64.
    """
    meta=np.asarray(interface_meta,dtype=np.float32)
    if meta.ndim!=2 or meta.shape[1]!=5 or not 1<=k<=len(meta) or valid_length<1:
        raise ValueError('Packed metadata, positive length and valid interface count required.')
    scale=np.float32(valid_length)
    tau=meta[:k,1]*scale;left=meta[:k,3]*scale;right=meta[:k,4]*scale
    return tau-left,tau+right


def energy_log_ratios(trace, window_start, window_end, *, front_end, epsilon=1e-8):
    """Sum x^2 at inclusive integer sample centers, without altering RF."""
    x = np.asarray(trace, dtype=np.float64)
    starts = np.asarray(window_start, dtype=np.float64)
    ends = np.asarray(window_end, dtype=np.float64)
    if (x.ndim not in (1,2) or not x.size or not np.isfinite(x).all()
            or starts.ndim != 1 or starts.shape != ends.shape or not len(starts)
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()):
        raise ValueError('Finite RF and matching nonempty window vectors required.')
    if type(front_end) is not int or not 0 <= front_end < x.shape[-1]:
        raise ValueError('Front window must lie within RF.')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite energy epsilon required.')
    left = np.ceil(starts).astype(np.int64)
    right = np.floor(ends).astype(np.int64)
    if np.any(left < 0) or np.any(right >= x.shape[-1]) or np.any(left > right):
        raise ValueError('Local windows must contain samples inside RF.')
    # Direct sums avoid subtracting large cumulative front energy when local
    # energy is small. The same function is used by fitting and make_batch.
    front = np.sum(x[...,:front_end+1]**2,axis=-1)
    local = np.stack([np.sum(x[...,l:r+1]**2,axis=-1) for l,r in zip(left,right)],axis=-1)
    ratios = np.log((local+epsilon)/(front[...,None]+epsilon))
    if not np.isfinite(ratios).all():
        raise ValueError('Nonfinite physical energy ratio.')
    return ratios


def physical_tokens(ratios, *, mean, std, max_interfaces=6):
    """Append normalized local f and strictly upstream f context plus is_first."""
    f = np.asarray(ratios, dtype=np.float64)
    if (f.ndim != 1 or not 1 <= len(f) <= max_interfaces or not np.isfinite(f).all()
            or not math.isfinite(mean) or not math.isfinite(std) or std <= 0):
        raise ValueError('Finite ratios and training mean/positive std required.')
    z = (f-mean)/std
    loc = np.zeros((max_interfaces,1),dtype=np.float32)
    ctx = np.zeros((max_interfaces,3),dtype=np.float32)
    loc[:len(z),0] = z
    ctx[0,2] = 1
    for k in range(1,len(z)):
        ctx[k,:2] = [z[k-1],np.max(z[:k])]
    if not np.isfinite(loc).all() or not np.isfinite(ctx).all():
        raise ValueError('Nonfinite normalized physical token.')
    return loc,ctx
