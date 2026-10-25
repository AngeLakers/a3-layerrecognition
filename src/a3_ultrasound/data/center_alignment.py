"""Label-free surface-packet centers and reversible integer-shift RF views."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class PacketCenters:
    envelope_energy: np.ndarray
    envelope_peak: np.ndarray
    rf_energy: np.ndarray
    failed: np.ndarray
    boundary_sensitive: np.ndarray
    right_edge_energy_fraction: np.ndarray


def packet_centers(context, *, gate_end=200, pad=256):
    """Measure centers in the front gate; envelope never replaces source RF."""
    raw=np.asarray(context,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=512 or not 16<=gate_end<512 or pad<1:
        raise ValueError('Expected N x 512 RF context and a valid front gate/pad.')
    finite=np.isfinite(raw).all(axis=1)
    clean=np.where(np.isfinite(raw),raw,0)
    x=clean-np.median(clean[:,:16],axis=1,keepdims=True)
    padded=np.pad(x,((0,0),(pad,pad)),mode='reflect')
    env=np.abs(hilbert(padded,axis=1)[:,pad:pad+512])[:,:gate_end+1]
    power=env*env;total=power.sum(axis=1)
    failed=~finite | (total<=1e-12)
    denom=np.where(failed,1,total)
    times=np.arange(gate_end+1,dtype=np.float64)
    center=(power*times).sum(axis=1)/denom
    peak=env.argmax(axis=1).astype(np.float64)
    rawpower=x[:,:gate_end+1]**2;rawtotal=rawpower.sum(axis=1)
    rawcenter=(rawpower*times).sum(axis=1)/np.where(rawtotal>1e-12,rawtotal,1)
    edge=power[:,-16:].sum(axis=1)/denom
    boundary=(peak<8)|(peak>gate_end-8)|(edge>.1)
    return PacketCenters(np.where(failed,np.nan,center),np.where(failed,np.nan,peak),
                         np.where(failed|(rawtotal<=1e-12),np.nan,rawcenter),
                         failed,boundary & ~failed,np.where(failed,np.nan,edge))


def integer_shift(raw, shift, *, reference=256):
    """Preserve every source sample bit-for-bit and mark leading pad invalid."""
    x=np.asarray(raw)
    if x.ndim!=1 or not len(x) or isinstance(shift,bool) or int(shift)!=shift or shift<0:
        raise ValueError('Expected nonempty RF and a nonnegative integer shift.')
    shift=int(shift)
    aligned=np.zeros(len(x)+shift,dtype=x.dtype)
    aligned[shift:]=x
    mask=np.zeros(len(aligned),dtype=bool);mask[shift:]=True
    return aligned,mask,np.arange(len(aligned),dtype=np.int64)-int(reference)


class AlignedRFView:
    """Load only RF, boundary/identity indices and RF-only alignment sidecars.

    A caller must propagate source_mask: leading zero pad is not a measurement.
    No training dataset, label catalog or classifier is imported here.
    """
    def __init__(self,catalog,*,mode='board'):
        self.path=Path(catalog)
        self.catalog=json.loads(self.path.read_text(encoding='utf-8'))
        if self.catalog.get('schema')!='a3-center-aligned-rf-view-v1' or mode not in ('board','record'):
            raise ValueError('Unknown RF alignment schema/mode.')
        self.mode=mode;self.boards={b['board']:b for b in self.catalog['boards']};self.cache={}
        self.reference=int(self.catalog['reference_samples'])

    def _open(self,board):
        if board in self.cache:
            return self.cache[board]
        b=self.boards[board];p=Path(b['rf_path']);stat=p.stat()
        if dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)!=b['rf_source_stat']:
            raise ValueError('RF source metadata changed; inspect before reuse.')
        arrays={key:np.load(b[key+'_path'],mmap_mode='r',allow_pickle=False)
                for key in ('offsets','lengths','source_row')}
        for key in ('offsets','lengths','source_row'):
            if hashlib.sha256(Path(b[key+'_path']).read_bytes()).hexdigest()!=b['index_sha256'][key]:
                raise ValueError('RF source index changed.')
        centers_path=self.path.parent/b['centers_path']
        if hashlib.sha256(centers_path.read_bytes()).hexdigest()!=b['centers_sha256']:
            raise ValueError('RF alignment sidecar changed.')
        centers=np.load(centers_path,mmap_mode='r',allow_pickle=False)
        if len(centers)!=len(arrays['offsets']) or len(centers)!=b['records']:
            raise ValueError('Alignment/source record mismatch.')
        arrays.update(rf=np.load(p,mmap_mode='r',allow_pickle=False),centers=centers)
        self.cache[board]=arrays
        return arrays

    def read(self,board,row):
        b=self.boards[board];a=self._open(board)
        if isinstance(row,bool) or int(row)!=row or not 0<=row<b['records']:
            raise IndexError('RF record outside registered board.')
        row=int(row);start=int(a['offsets'][row]);length=int(a['lengths'][row])
        center=float(a['centers'][row]);fallback=not np.isfinite(center)
        shift=(int(b['board_shift_samples']) if self.mode=='board' or fallback
               else int(np.rint(self.reference-center)))
        raw=a['rf'][start:start+length]
        aligned,mask,coordinate=integer_shift(raw,shift,reference=self.reference)
        return dict(x=aligned,source_mask=mask,relative_sample=coordinate,
                    board=board,sample_row=row,source_row=int(a['source_row'][row]),
                    source_length=length,source_start=shift,shift_samples=shift,
                    center_failed=fallback,record_mode_used_board_fallback=bool(self.mode=='record' and fallback),
                    nominal_interfaces_relative_samples=b['nominal_interfaces_relative_samples'],
                    nominal_backwall_relative_samples=b['nominal_backwall_relative_samples'])
