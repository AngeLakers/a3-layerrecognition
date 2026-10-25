"""RF-only envelope anchors and label-independent shared window jitter."""
from __future__ import annotations
import math
import numpy as np
from a3_ultrasound.data.window_diagnostics import diagnostic_envelope


def envelope_anchor_positions(raw, nominal_backwall, *, front_end=200, search_half=130):
    """Use exactly the same full-RF analytic envelope for both anchor peaks."""
    x=np.asarray(raw)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]<front_end+1 or not np.isfinite(x).all():
        raise ValueError('Finite full RF with complete front gate required.')
    env=diagnostic_envelope(x)
    front=env[:,:front_end+1];start=np.argmax(front,axis=1).astype(float)
    start[front.max(axis=1)<=1e-12]=np.nan
    lo=math.ceil(nominal_backwall-search_half);hi=math.floor(nominal_backwall+search_half)
    bottom=np.full(len(x),np.nan);edge=np.zeros(len(x),bool)
    complete=0<=lo<=hi<x.shape[1]
    if complete:
        part=env[:,lo:hi+1];where=np.argmax(part,axis=1)
        good=part.max(axis=1)>1e-12;bottom[good]=lo+where[good]
        edge[good]=(where[good]==0)|(where[good]==part.shape[1]-1)
    return start,bottom,np.full(len(x),complete,bool),edge


def shared_window_shift(seed, epoch, ordinal, source_row, radius=24.):
    """Stateless continuous uniform shift: identical across Heads and call order."""
    if min(seed,epoch,ordinal,source_row)<0 or not math.isfinite(radius) or radius<0:
        raise ValueError('Nonnegative identity/epoch and finite radius required.')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,ordinal,source_row,0xA3152]))
    return float(rng.uniform(-radius,radius))


def divided_centers(start_median,bottom_median,k):
    if (type(k) is not int or k<1 or not math.isfinite(start_median)
            or not math.isfinite(bottom_median) or bottom_median<=start_median):
        raise ValueError('Ordered finite anchors and positive interface count required.')
    interval=(bottom_median-start_median)/(k+1)
    return interval,[start_median+j*interval for j in range(1,k+1)]


def selected_prior(timebase,k,trace_length,*,shift=0.):
    from a3_ultrasound.data.time_prior import DerivedTimePrior
    centers=np.asarray(timebase['selected_tau'],dtype=float)+shift
    bottom=float(timebase['selected_bottom']);half=80.
    if (centers.shape!=(k,) or not np.isfinite(centers).all() or not np.isfinite(shift) or not math.isfinite(bottom)
            or np.any(np.diff(centers)<=0) or centers[0]-half<0
            or centers[-1]+half>trace_length-1 or centers[-1]+half>=bottom):
        raise ValueError('Selected board windows must lie within RF and before its backwall.')
    return DerivedTimePrior(tuple(centers),half,half,tuple(centers-half),tuple(centers+half),0.,bottom)


def adoption_decision(records):
    """User-fixed pooled absolute-offset gate plus per-cell 20-sample guard."""
    deep=[r for r in records if r['k']>=2]
    if not deep:raise ValueError('Deep-interface validation observations are required.')
    old=np.array([r['old_offset'] for r in deep],float)
    new=np.array([r['calibrated_offset'] for r in deep],float)
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('All paired validation offsets must be finite.')
    cells=[]
    for board,k in sorted({(r['board'],r['k']) for r in deep}):
        one=[r for r in deep if r['board']==board and r['k']==k]
        a=float(np.median([r['old_offset'] for r in one]));b=float(np.median([r['calibrated_offset'] for r in one]))
        cells.append(dict(board=board,k=k,old_median=a,calibrated_median=b,
            worsening=abs(b)-abs(a),within20=abs(b)-abs(a)<=20.))
    a=float(np.median(np.abs(old)));b=float(np.median(np.abs(new)))
    pooled=b<a;guard=all(r['within20'] for r in cells)
    return dict(adopt_calibration=pooled and guard,pooled_absolute_offset_median_old=a,
        pooled_absolute_offset_median_calibrated=b,pooled_improves=pooled,all_cells_within20=guard,
        deep_records=len(deep),cells=cells,scope='One decision applied to all13 boards')
