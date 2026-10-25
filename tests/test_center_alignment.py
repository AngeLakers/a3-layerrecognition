import numpy as np
import pytest
import hashlib
import json

from a3_ultrasound.data.center_alignment import packet_centers,integer_shift,AlignedRFView


def packet(center=90):
    t=np.arange(512,dtype=float)
    return np.exp(-.5*((t-center)/16)**2)*np.cos(2*np.pi*.04*(t-center))


def test_packet_center_tracks_shift_and_ignores_gain_polarity():
    x=packet();r=packet_centers(np.stack([x,-3*x,packet(110)]))
    assert abs(r.envelope_energy[0]-90)<.5
    assert abs(r.envelope_energy[0]-r.envelope_energy[1])<1e-10
    assert abs(r.envelope_energy[2]-r.envelope_energy[0]-20)<.5
    assert not r.failed.any()


def test_packet_center_baseline_and_invalid_inputs():
    x=packet();r=packet_centers(np.stack([x,x+100,np.zeros(512)]))
    assert abs(r.envelope_energy[0]-r.envelope_energy[1])<1e-8
    assert r.failed[2] and np.isnan(r.envelope_energy[2])
    bad=x.copy();bad[20]=np.nan
    assert packet_centers(bad[None]).failed[0]
    with pytest.raises(ValueError):
        packet_centers(np.zeros((1,200)))


@pytest.mark.parametrize('shift',[0,5,256])
def test_integer_shift_preserves_all_samples_and_support(shift):
    x=np.array([4,-7,1,0,2],np.float32)
    y,mask,time=integer_shift(x,shift)
    assert y.dtype==x.dtype and y[mask].tobytes()==x.tobytes()
    assert mask.sum()==len(x) and not mask[:shift].any()
    assert time[shift]==shift-256 and len(y)==len(x)+shift


def test_integer_shift_rejects_truncation_or_interpolation():
    for shift in (-1,.5,True):
        with pytest.raises(ValueError):
            integer_shift(np.ones(4),shift)


@pytest.mark.parametrize('mode',['board','record'])
def test_view_preserves_source_and_explicit_failed_fallback_without_labels(tmp_path,mode):
    raw=np.arange(13,dtype=np.float32)
    data=dict(rf=raw,offsets=np.array([0,7],np.int64),lengths=np.array([7,6],np.int64),
              source_row=np.array([31,32],np.int64),centers=np.array([90.500008,np.nan],np.float32))
    for name,value in data.items():
        np.save(tmp_path/(name+'.npy'),value)
    for name in ('y','label_mask','eligible','quality'):
        (tmp_path/(name+'.npy')).write_bytes(b'Label access must not occur')
    p=tmp_path/'rf.npy'
    b=dict(board='test',records=2,rf_path=str(p),rf_source_stat=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns),
        **{k+'_path':str(tmp_path/(k+'.npy')) for k in ('offsets','lengths','source_row')},
        index_sha256={k:hashlib.sha256((tmp_path/(k+'.npy')).read_bytes()).hexdigest() for k in ('offsets','lengths','source_row')},
        centers_path='centers.npy',centers_sha256=hashlib.sha256((tmp_path/'centers.npy').read_bytes()).hexdigest(),
        board_shift_samples=165,nominal_interfaces_relative_samples=[200,400],nominal_backwall_relative_samples=600)
    catalog=tmp_path/'aligned_catalog.json'
    catalog.write_text(json.dumps(dict(schema='a3-center-aligned-rf-view-v1',reference_samples=256,boards=[b])),encoding='utf-8')
    view=AlignedRFView(catalog,mode=mode)
    for row in range(2):
        r=view.read('test',row);expected=raw[:7] if row==0 else raw[7:]
        assert r['x'][r['source_mask']].tobytes()==expected.tobytes()
        assert r['source_row']==31+row
        assert r['record_mode_used_board_fallback'] is (mode=='record' and row==1)
        assert r['shift_samples']==165
    with pytest.raises(IndexError):
        view.read('test',2)
    np.save(tmp_path/'tampered.npy',np.zeros(2,np.float32))
    b['centers_path']='tampered.npy'
    catalog.write_text(json.dumps(dict(schema='a3-center-aligned-rf-view-v1',reference_samples=256,boards=[b])),encoding='utf-8')
    with pytest.raises(ValueError,match='sidecar changed'):
        AlignedRFView(catalog,mode=mode).read('test',0)
