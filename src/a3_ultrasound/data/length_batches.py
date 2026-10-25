"""Seeded homogeneous-length batches without changing the supplied population."""
from __future__ import annotations
import numpy as np

def length_bucket_batches(lengths,batch_size,*,seed,epoch):
    """Shuffle within length buckets, retain tails, then shuffle all batches."""
    lengths=np.asarray(lengths)
    if lengths.ndim!=1 or not len(lengths) or lengths.dtype.kind not in 'iu' or np.any(lengths<1):
        raise ValueError('Positive integer effective lengths are required')
    if type(batch_size) is not int or batch_size<1 or type(seed) is not int or seed<0 or type(epoch) is not int or epoch<1:
        raise ValueError('Expected positive batch_size/epoch and nonnegative integer seed')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,0xA3153]))
    batches=[]
    for length in np.unique(lengths):
        indices=rng.permutation(np.flatnonzero(lengths==length))
        batches.extend(indices[start:start+batch_size] for start in range(0,len(indices),batch_size))
    return [batches[int(i)] for i in rng.permutation(len(batches))]

def matched_smoke_indices(lengths,batch_size,steps,*,seed,epoch):
    """One shared smoke population with full homogeneous batches in both arms.

    This selects a bounded smoke subset; it never rewrites the full frozen epoch.
    Full-epoch tail retention is independently verified on length_bucket_batches.
    """
    if type(steps) is not int or steps<1:raise ValueError('Positive smoke steps required')
    full=[b for b in length_bucket_batches(lengths,batch_size,seed=seed,epoch=epoch) if len(b)==batch_size]
    if len(full)<steps:raise ValueError('Insufficient complete batches for matched smoke')
    return np.sort(np.concatenate(full[:steps]))
