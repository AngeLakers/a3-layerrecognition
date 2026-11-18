"""Fold-fixed M3 second input channel, computed inside the shared RF pipeline.

The second channel is derived from the same raw record as x with the fold's
TRAIN-fitted scalars and the board's frozen v1.6 centres. It never reads labels
and has no jitter of its own: window jitter only moves interface_meta, so the
two channels stay sample-aligned and share one length mask.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from a3_ultrasound.data.m3_transform import Decomposer

_ACCEPTED=('FITTED','POOLED_FIT','IDENTITY_FALLBACK')


class M3Channel:
    def __init__(self, config, timebase_rows, entities):
        channel=config.get('channel')
        if channel not in ('residual','free_deep'):
            raise ValueError('M3 channel must be residual or free_deep.')
        params=config.get('params_by_entity',{})
        missing=[e for e in entities if e not in params or e not in timebase_rows]
        if missing:raise ValueError('M3 parameters or frozen centres missing for: '+', '.join(missing))
        for entity in entities:
            if params[entity].get('status') not in _ACCEPTED:
                raise ValueError('Unusable M3 parameter status for '+entity)
        self.channel=channel
        self.params={e:params[e] for e in entities}
        self.centers={e:[float(t) for t in timebase_rows[e]['selected_tau']] for e in entities}
        self._operators={}

    def operator(self, entity, length):
        key=(entity,int(length))
        if key not in self._operators:
            self._operators[key]=Decomposer(self.centers[entity],int(length),self.params[entity])
        return self._operators[key]

    def __call__(self, entity, raw):
        """raw: one complete stored record in raw units; returns the same length."""
        raw=np.asarray(raw,dtype=float)
        if raw.ndim!=1:raise ValueError('One raw RF record required.')
        residual,free=self.operator(entity,len(raw))(raw[None])
        return (residual if self.channel=='residual' else free)[0]


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
