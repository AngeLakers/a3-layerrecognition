from copy import deepcopy
import numpy as np
import pytest
from a3_ultrasound.data.validation_sampling import select_validation_rows
from a3_ultrasound.stream_metrics import PredictionAccumulator


def test_spatial_selection_preserves_positives_masks_quality_and_population():
    n=3000;y=np.zeros((n,3),dtype=np.int8);mask=np.ones_like(y,dtype=bool)
    y[::127,2]=1;mask[::3,1]=False;mask[::11]=False
    y[::11,0]=1 # Unknown entries must not silently become supervised positives.
    xy=np.column_stack((np.arange(n)%60,np.arange(n)//60));quality=np.arange(n)%7!=0
    ids,w=select_validation_rows(y,mask,xy,quality,target=300,ordinal=4)
    ids2,w2=select_validation_rows(y,mask,xy,quality,target=300,ordinal=4)
    np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(w,w2)
    assert len(ids)==300 and len(np.unique(ids))==300 and w.sum()==n
    positive=(mask&(y==1)).any(1)
    assert set(np.flatnonzero(positive))<=set(ids)
    assert np.all(w[positive[ids]]==1)
    np.testing.assert_array_equal((mask[ids]*w[:,None]).sum(0),mask.sum(0))
    codes=(mask*(1<<np.arange(3))).sum(1)*2+quality
    for code in np.unique(codes):assert w[codes[ids]==code].sum()==(codes==code).sum()


def test_small_populations_and_positive_heavy_budget():
    y=np.zeros((10,2),dtype=np.int64);m=np.ones_like(y,dtype=bool);xy=np.column_stack((np.arange(10),np.zeros(10)))
    ids,w=select_validation_rows(y,m,xy,np.ones(10),target=20)
    assert ids.tolist()==list(range(10)) and w.tolist()==[1]*10
    y[:9,0]=1
    ids,w=select_validation_rows(y,m,xy,np.ones(10),target=2)
    assert ids.tolist()==list(range(10)) and w.sum()==10


def row(y,p,weight=1):
    joint=[(1-p[0])*(1-p[1]),(1-p[0])*p[1],p[0]*(1-p[1]),p[0]*p[1]]
    return dict(y=y,label_mask=[True,True],marginals=p,map_states=[int(x>=.5) for x in p],
        log_probabilities=np.log(joint).tolist(),base_probabilities=p,base_logodds=[0.,0.],
        rho=None,raw_logZ=0.,specimen_id='board',sample_weight=weight)


def test_weighted_metrics_equal_expanded_population_and_count_actual_evaluations():
    rows=[row([0,0],[.1,.2],11),row([0,0],[.6,.7],3),row([1,0],[.9,.2]),row([0,1],[.1,.8])]
    weighted=PredictionAccumulator();expanded=PredictionAccumulator()
    for r in rows:
        weighted.add(r)
        for _ in range(r['sample_weight']):expanded.add(dict(r,sample_weight=1))
    a=weighted.result();b=expanded.result()
    for key in ('sample_count','labeled_interface_count','deepest_specimen_macro_f1','joint_nll','joint_map_accuracy','base_error_count'):
        assert a[key]==pytest.approx(b[key])
    assert a['deepest_per_specimen']['board']['confusion_matrix_0_1']==b['deepest_per_specimen']['board']['confusion_matrix_0_1']
    assert a['evaluated_sample_count']==4 and a['sample_count']==16 and a['sampling_weighted']
    for invalid in (0,-1,1.5,True):
        with pytest.raises(ValueError):weighted.add(dict(rows[0],sample_weight=invalid))


def test_unknown_labels_do_not_contribute_even_when_weighted():
    r=row([0,1],[.1,.8],10);r['label_mask']=[True,False]
    a=PredictionAccumulator();a.add(r);result=a.result()
    assert result['labeled_interface_count']==10 and result['joint_complete_label_count']==0
    assert result['deepest_per_specimen']=={} and '2' not in result['per_interface']
