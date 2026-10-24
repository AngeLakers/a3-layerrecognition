"""Frozen spatially stratified validation sampling with integer inverse weights."""
from __future__ import annotations
import numpy as np


def select_validation_rows(y, label_mask, xy, quality, *, target=13334, seed=20261023, ordinal=0):
    """Keep every reliably positive row; sample one row per spatial bin.

    Negative/unknown strata have a fixed label-mask pattern and quality flag.
    Within each stratum, bins partition the spatially sorted rows. A uniform
    draw within a bin has inclusion probability 1 / bin_size, giving an exact
    integer inverse probability weight. No model output informs selection.
    """
    y=np.asarray(y);mask=np.asarray(label_mask,dtype=bool);xy=np.asarray(xy);quality=np.asarray(quality,dtype=bool)
    if y.ndim!=2 or mask.shape!=y.shape or xy.shape!=(len(y),2) or quality.shape!=(len(y),):
        raise ValueError('Validation metadata shape mismatch.')
    if not len(y) or type(target) is not int or target<1:raise ValueError('Positive target and nonempty population required.')
    if not np.isin(y[mask],(0,1)).all() or not np.isfinite(xy).all():raise ValueError('Invalid reliable labels or coordinates.')
    positive=(mask&(y==1)).any(1)
    kept=np.flatnonzero(positive);negative=np.flatnonzero(~positive)
    if len(y)<=target:return np.arange(len(y),dtype=np.int64),np.ones(len(y),dtype=np.int64)
    codes=(mask.astype(np.int64)*(1<<np.arange(y.shape[1]))).sum(1)*2+quality
    groups=[negative[codes[negative]==code] for code in np.unique(codes[negative])]
    if not groups:return kept,np.ones(len(kept),dtype=np.int64)
    sizes=np.array([len(group) for group in groups],dtype=np.int64)
    # At least one representative in every nonempty mask/quality stratum.
    budget=min(len(negative),max(len(groups),target-len(kept)))
    quotas=np.ones(len(groups),dtype=np.int64)
    remaining=budget-len(groups)
    capacities=sizes-1
    if remaining:
        ideal=remaining*capacities/capacities.sum()
        extra=np.floor(ideal).astype(np.int64);quotas+=extra
        left=remaining-int(extra.sum())
        if left:quotas[np.argsort(-(ideal-extra),kind='stable')[:left]]+=1
    rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal]))
    rows=[kept];weights=[np.ones(len(kept),dtype=np.int64)]
    for group,n in zip(groups,quotas):
        group=group[np.lexsort((xy[group,0],xy[group,1]))]
        edges=np.linspace(0,len(group),int(n)+1,dtype=np.int64)
        chosen=rng.integers(edges[:-1],edges[1:])
        rows.append(group[chosen]);weights.append(np.diff(edges))
    rows=np.concatenate(rows);weights=np.concatenate(weights);order=np.argsort(rows,kind='stable')
    rows=rows[order];weights=weights[order]
    if len(np.unique(rows))!=len(rows) or weights.sum()!=len(y) or not np.isin(kept,rows).all():
        raise RuntimeError('Validation sampling invariant failed.')
    return rows,weights
