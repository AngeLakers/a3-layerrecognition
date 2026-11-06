"""Attach a precomputed M3 channel after the shared RF crop; no independent jitter."""
from dataclasses import replace
import numpy as np
import torch


def attach_channel(batch, channel_traces):
    """Raw-unit full traces in batch order; same retained prefix and /131 as RF.

    Window jitter is already represented by the single interface_meta tensor.
    Cache provenance and fold membership must be checked by the caller.
    """
    if batch.x_raw.ndim != 2 or batch.phys_loc is not None or len(channel_traces)!=len(batch.x_raw):
        raise ValueError('Single-channel nonphysical batch and one M3 trace per row required.')
    lengths=batch.length_mask.sum(1).cpu().tolist()
    second=torch.zeros_like(batch.x_raw)
    for i,(trace,n) in enumerate(zip(channel_traces,lengths)):
        value=np.asarray(trace,dtype=np.float32)
        if value.ndim!=1 or len(value)<n or not np.isfinite(value[:n]).all():
            raise ValueError('M3 channel must contain the full finite retained RF prefix.')
        second[i,:n]=torch.as_tensor(value[:n].copy(),device=second.device)/131.
    return replace(batch,x_raw=torch.stack((batch.x_raw,second),dim=1))
