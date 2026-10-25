"""Candidate board-reference geometry, without changing the frozen input path."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from a3_ultrasound.data.quick_input import transform_trace


@dataclass(frozen=True)
class ReferencedGeometry:
    reference: float
    interface_centers: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    local_available_counts: np.ndarray
    local_declared_counts: np.ndarray
    local_complete: np.ndarray
    local_model_support_complete: np.ndarray
    backwall_center: float
    backwall_search_start: int
    backwall_search_end: int
    backwall_available_count: int
    backwall_declared_count: int


def reference_geometry(*,K,layer_mm,sampling_rate_hz,sound_speed_mps,reference_samples,
                       source_length,halfwidth_samples=70.,storage_shift=0,backwall_search_us=.65):
    if (type(K) is not int or not 1<=K<=6 or type(source_length) is not int or source_length<1
            or type(storage_shift) is not int or storage_shift<0
            or not all(math.isfinite(float(v)) for v in (layer_mm,sampling_rate_hz,sound_speed_mps,reference_samples,halfwidth_samples,backwall_search_us))
            or min(layer_mm,sampling_rate_hz,sound_speed_mps,halfwidth_samples,backwall_search_us)<=0
            or reference_samples<0):
        raise ValueError('Finite geometry, nonnegative reference and positive source support required.')
    interval=2*layer_mm*1e-3*sampling_rate_hz/sound_speed_mps
    ref=float(reference_samples)+storage_shift
    centers=ref+np.arange(1,K+1)*interval
    starts=centers-halfwidth_samples;ends=centers+halfwidth_samples
    left=np.ceil(starts).astype(np.int64);right=np.floor(ends).astype(np.int64)
    declared=right-left+1
    available=np.maximum(0,np.minimum(right,storage_shift+source_length-1)-np.maximum(left,storage_shift)+1)
    bw=ref+(K+1)*interval
    radius=backwall_search_us*1e-6*sampling_rate_hz
    bwleft=math.ceil(bw-radius);bwright=math.floor(bw+radius)
    bwcount=max(0,min(bwright,storage_shift+source_length-1)-max(bwleft,storage_shift)+1)
    # The model checks continuous window endpoints, stricter than integer counts.
    model_complete=(starts>=storage_shift)&(ends<=storage_shift+source_length-1)
    return ReferencedGeometry(ref,centers,starts,ends,available,declared,(declared>0)&(available==declared),model_complete,
                              bw,bwleft,bwright,bwcount,bwright-bwleft+1)


def available_window_mask(samples,source_mask,left,right):
    positions=np.asarray(samples)
    mask=np.asarray(source_mask)
    if mask.dtype!=np.bool_ or mask.shape!=positions.shape:
        raise ValueError('Source availability mask must match sample coordinates.')
    return mask&(positions>=left)&(positions<=right)


def prepare_referenced_trace(trace,row,config,*,board_reference_samples=None):
    """Off is the exact old function; on is an unintegrated diagnostic candidate.

    Keep raw-coordinate samples so V1.5.1's valid-prefix batched trunk does not
    receive a leading-invalid mask. Nominal centers carry the board offset.
    No physical-feature normalization artifact is silently reused.
    """
    if board_reference_samples is None:
        return transform_trace(trace,row,config)
    raw=np.asarray(trace,dtype=np.float32)
    if raw.ndim!=1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected nonempty finite original-coordinate RF.')
    material='AL' if row['group']=='AL' else 'TC4'
    half=config['window_total_us']*1e-6*config['sampling_rate_hz']/2
    g=reference_geometry(K=int(row['K']),layer_mm=float(row['layer_mm']),
        sampling_rate_hz=float(config['sampling_rate_hz']),sound_speed_mps=float(config['sound_speed_mps'][material]),
        reference_samples=float(board_reference_samples),source_length=len(raw),halfwidth_samples=half)
    if not g.local_complete.all() or not g.local_model_support_complete.all():
        raise ValueError('Target Local support is incomplete; mask/recovery policy must be resolved before model use.')
    divisor=float(config['normalization_divisor'])
    if not math.isfinite(divisor) or divisor<=0:
        raise ValueError('Positive fixed divisor required.')
    x=raw.copy()/np.float32(divisor)
    meta=np.zeros((6,5),np.float32);k=int(row['K'])
    meta[:k,0]=np.arange(1,k+1)/k;meta[:k,1]=g.interface_centers/len(x)
    meta[:k,2]=np.arange(1,k+1)/(k+1);meta[:k,3:5]=half/len(x)
    bw_samples=np.arange(g.backwall_search_start,g.backwall_search_end+1)
    bw_mask=(bw_samples>=0)&(bw_samples<len(raw))
    info=dict(original_length=len(raw),valid_length=len(x),nominal_tof_samples=g.interface_centers.tolist(),
        nominal_backwall_samples=g.backwall_center,halfwidth_samples=half,divisor=divisor,
        reference_samples=g.reference,input_tail_policy='retain_source_diagnostic',source_mask=np.ones(len(x),bool),
        local_support_complete=g.local_complete.tolist(),
        local_model_support_complete=g.local_model_support_complete.tolist(),
        backwall_sample_index=bw_samples,backwall_source_mask=bw_mask,
        backwall_available_count=g.backwall_available_count,backwall_declared_count=g.backwall_declared_count,
        physical_normalization_refit_required=True,training_integrated=False)
    return x,meta,info
