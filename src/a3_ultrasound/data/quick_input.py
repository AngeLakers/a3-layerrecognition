"""Explicit zero-origin experiment input policy, without changing stored RF."""
from __future__ import annotations

import math
import numpy as np

from a3_ultrasound.data.time_prior import derive_time_prior_samples


def physical_manifest(row: dict, config: dict) -> dict:
    """The approved reference is zero AFTER cropping; never subtract crop_start twice."""
    if config['time_reference'] != 'cropped_record_origin_is_zero':
        raise ValueError('This policy only implements the explicit cropped zero reference.')
    k = int(row['K'])
    h = float(row['layer_mm'])
    material = 'AL' if row['group'] == 'AL' else 'TC4'
    half = float(config['window_total_us']) * 1e-6 / 2
    return {'acquisition': {'sampling_rate_hz': config['sampling_rate_hz'],
            'window_profile': {'window_left_s': half, 'window_right_s': half}},
            'time_reference': {'batch_reference_samples': config['batch_reference_samples']},
            'structure': {'K': k, 'total_thickness_mm': (k+1)*h,
                'sound_speed_mps': config['sound_speed_mps'][material],
                'interfaces': [{'interface_order':i, 'nominal_depth_mm':i*h} for i in range(1,k+1)]}}


def transform_trace(trace: np.ndarray, row: dict, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the versioned tail-retention rule, then one fixed scaling.

    V3 scans all later activity; the historical boundary-only rule remains
    reproducible for old configs. Neither policy identifies a physical echo.
    Existing filtering may already mix later samples into the retained prefix.
    """
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 1 or not len(trace) or not np.isfinite(trace).all():
        raise ValueError('Expected nonempty finite RF vector.')
    prior = derive_time_prior_samples(physical_manifest(row, config), trace_length=len(trace))
    fs = float(config['sampling_rate_hz'])
    tail = config['tail']
    proposed = math.ceil(prior.backwall_tof_samples + tail['nominal_backwall_guard_us']*1e-6*fs)
    stop = min(len(trace), proposed)
    check = int(tail['active_boundary_check_samples'])
    quiet_run = int(tail['quiet_run_samples'])
    baseline = float(np.median(trace[:min(16,len(trace))]))
    threshold = float(tail['quiet_floor_raw_units'])
    deviation = np.abs(trace.astype(float)-baseline)
    active = stop < len(trace) and bool(np.any(deviation[max(0,stop-check):stop] > threshold))
    policy=tail.get('policy','active_boundary_first_quiet')
    if policy=='nominal_fixed':
        pass
    elif policy=='last_active_tail':
        # A quiet gap at the nominal cutoff cannot establish that later packets
        # are absent. Scan the entire remainder before discarding any tail.
        later=np.flatnonzero(deviation[stop:] > threshold)
        if len(later):
            stop=min(len(trace),stop+int(later[-1])+1+int(tail['quiet_guard_samples']))
        elif active:
            last=np.flatnonzero(deviation[:stop] > threshold)
            stop=min(len(trace),max(stop,int(last[-1])+1+int(tail['quiet_guard_samples'])))
    elif policy!='active_boundary_first_quiet':
        raise ValueError('Unknown tail policy.')
    elif active:
        quiet = deviation[stop:] <= threshold
        counts = np.convolve(quiet.astype(int),np.ones(quiet_run,dtype=int),'valid') if len(quiet)>=quiet_run else np.array([])
        hits = np.flatnonzero(counts==quiet_run)
        stop = min(len(trace),stop+int(hits[0])+max(quiet_run,int(tail['quiet_guard_samples']))) if len(hits) else len(trace)
    if stop <= max(prior.window_end_samples):
        raise ValueError('Tail policy would truncate a Local window.')
    divisor = float(config['normalization_divisor'])
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError('Fixed divisor must be finite and positive.')
    # No filtering, realignment, interpolation or amplitude-dependent normalization.
    x = trace[:stop].copy()/np.float32(divisor)
    k = int(row['K'])
    meta = np.zeros((6,5),dtype=np.float32)
    meta[:k,0] = np.arange(1,k+1)/k
    meta[:k,1] = np.asarray(prior.nominal_tof_samples)/stop
    meta[:k,2] = np.arange(1,k+1)/(k+1)
    meta[:k,3] = prior.window_left_samples/stop
    meta[:k,4] = prior.window_right_samples/stop
    return x,meta,{'original_length':len(trace),'valid_length':stop,'proposed_stop':proposed,
        'active_boundary_extended':bool(active and stop>min(len(trace),proposed)),
        'tail_extended':bool(stop>min(len(trace),proposed)),
        'tail_policy':policy,
        'nominal_tof_samples':list(prior.nominal_tof_samples),
        'nominal_backwall_samples':prior.backwall_tof_samples,
        'halfwidth_samples':prior.window_left_samples,'divisor':divisor}
