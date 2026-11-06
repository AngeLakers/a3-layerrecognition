"""Fold-fixed M3 second input channel, computed inside the shared RF pipeline.

The second channel is derived from the same raw record as x with the fold's
TRAIN-fitted scalars and the board's frozen v1.6 centres. It never reads labels
and has no jitter of its own: window jitter only moves interface_meta, so the
two channels stay sample-aligned and share one length mask.
"""
from __future__ import annotations

import numpy as np

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
