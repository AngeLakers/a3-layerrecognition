"""Fixed RF-template observation gates, separate from the compared centers."""
import math
import numpy as np
from scipy.signal import find_peaks


def template_observation_gates(q99,start,bottom,k,*,half=80,front_end=200,
                               backwall_guard=80,min_separation=80,prominence=3.):
    """Associate observed template packets with ordered interfaces.

    Equal-transit fractions are an association assumption, not physical truth.
    Absent template packets stay absent. Neither comparison's target centers
    nor labels enter this function.
    """
    e=np.asarray(q99,dtype=float)
    if e.ndim!=1 or len(e)<201 or not np.isfinite(e).all() or k<1 or bottom<=start:
        raise ValueError('Finite full RF template and ordered anchors required.')
    delta=(bottom-start)/(k+1);peaks,_=find_peaks(e,distance=min_separation,prominence=prominence)
    selected={}
    for p in peaks:
        if p<=front_end or p>=bottom-backwall_guard:continue
        index=math.floor((p-start)/delta+.5)
        if not 1<=index<=k or abs(p-(start+index*delta))>.4*delta:continue
        if index not in selected or e[p]>e[selected[index]]:selected[index]=int(p)
    out=[]
    for index,p in sorted(selected.items()):
        lo=max(front_end+1,math.ceil(p-half));hi=min(len(e)-1,math.floor(bottom-backwall_guard),math.floor(p+half))
        earlier=[v for j,v in selected.items() if j<index];later=[v for j,v in selected.items() if j>index]
        if earlier:lo=max(lo,math.floor((max(earlier)+p)/2)+1)
        if later:hi=min(hi,math.floor((min(later)+p)/2))
        if lo<=p<=hi:out.append(dict(k=index,template_peak=p,left=lo,right=hi,template_amplitude=float(e[p])))
    return dict(gates=out,missing=[i for i in range(1,k+1) if i not in selected],
        association='Nearest equal-transit interface fraction, tolerance0.4*interval; RF-only template; not echo ground truth')


def observed_peak(envelope,gate):
    e=np.asarray(envelope,dtype=float);lo,hi=gate['left'],gate['right']
    if e.ndim!=1 or not 0<=lo<=hi<len(e) or not np.isfinite(e[lo:hi+1]).all():
        return dict(valid=False,reason='missing_or_nonfinite_source_gate')
    p=lo+int(np.argmax(e[lo:hi+1]));amp=float(e[p])
    if amp<=1e-12:return dict(valid=False,reason='zero_envelope')
    return dict(valid=True,peak=p,search_edge=p in (lo,hi),amplitude=amp)
