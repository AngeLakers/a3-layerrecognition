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
    policy = config.get('window_halfwidth_policy', 'fixed_us')
    if policy == 'half_interval_capped':
        fs = float(config['sampling_rate_hz'])
        interval = 2 * h * 1e-3 * fs / config['sound_speed_mps'][material]
        cap = config['window_max_halfwidth_samples']
        if type(cap) is not int or cap <= 0 or config['batch_reference_samples'] != 0:
            raise ValueError('Capped nominal windows require a positive integer cap and zero reference.')
        half_samples = min(cap, math.floor(interval / 2))
        if half_samples < 1:
            raise ValueError('Layer interval is too small for a Local window.')
        profile = {'window_left_samples': half_samples, 'window_right_samples': half_samples}
    elif policy == 'fixed_us':
        half = float(config['window_total_us']) * 1e-6 / 2
        profile = {'window_left_s': half, 'window_right_s': half}
    else:
        raise ValueError('Unknown Local window policy.')
    return {'acquisition': {'sampling_rate_hz': config['sampling_rate_hz'],
            'window_profile': profile},
            'time_reference': {'batch_reference_samples': config['batch_reference_samples']},
            'structure': {'K': k, 'total_thickness_mm': (k+1)*h,
                'sound_speed_mps': config['sound_speed_mps'][material],
                'interfaces': [{'interface_order':i, 'nominal_depth_mm':i*h} for i in range(1,k+1)]}}


def transform_trace(trace: np.ndarray, row: dict, config: dict, *, timebase=None,
                    window_shift=0., skip_nominal_diagnostics=False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the versioned tail-retention rule, then one fixed scaling.

    V3 scans all later activity; the historical boundary-only rule remains
    reproducible for old configs. Neither policy identifies a physical echo.
    Existing filtering may already mix later samples into the retained prefix.
    """
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 1 or not len(trace) or not np.isfinite(trace).all():
        raise ValueError('Expected nonempty finite RF vector.')
    if timebase is None:
        if window_shift != 0:
            raise ValueError('Window jitter requires an explicit board timebase.')
        prior = derive_time_prior_samples(physical_manifest(row, config), trace_length=len(trace))
    else:
        from a3_ultrasound.data.board_timebase import selected_prior
        if config.get('window_halfwidth_policy') != 'board_timebase':
            raise ValueError('Board centers require the explicit timebase window policy.')
        prior = selected_prior(timebase,int(row['K']),len(trace),shift=window_shift)
    fs = float(config['sampling_rate_hz'])
    tail = config['tail']
    policy=tail.get('policy','active_boundary_first_quiet')
    tail_backwall = (float(timebase.get('nominal_bottom',prior.backwall_tof_samples))
                     if timebase is not None else prior.backwall_tof_samples)
    proposed = math.ceil(tail_backwall + tail['nominal_backwall_guard_us']*1e-6*fs)
    prefixes=tail.get('fixed_prefix_samples_by_entity',{})
    if (not isinstance(prefixes,dict) or any(not isinstance(key,str) or not key
            or type(value) is not int or value<1 for key,value in prefixes.items())):
        raise ValueError('Fixed prefix overrides require entity names and positive integer lengths.')
    fixed_prefix=prefixes.get(row.get('entity'))
    if fixed_prefix is not None:
        if policy!='nominal_fixed' or fixed_prefix>len(trace):
            raise ValueError('Fixed prefix requires nominal_fixed policy and complete source RF.')
        proposed=fixed_prefix
    stop = min(len(trace), proposed)
    check = int(tail['active_boundary_check_samples'])
    quiet_run = int(tail['quiet_run_samples'])
    baseline = 0. if policy=='nominal_fixed' and skip_nominal_diagnostics else float(np.median(trace[:min(16,len(trace))]))
    threshold = float(tail['quiet_floor_raw_units'])
    deviation = None if policy=='nominal_fixed' and skip_nominal_diagnostics else np.abs(trace.astype(float)-baseline)
    active = deviation is not None and stop < len(trace) and bool(np.any(deviation[max(0,stop-check):stop] > threshold))
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
    if timebase is not None and max(prior.window_end_samples)>stop-1:
        raise ValueError('Selected continuous Local endpoint exceeds retained input.')
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
        'nominal_backwall_samples':tail_backwall,
        'halfwidth_samples':prior.window_left_samples,'divisor':divisor}
