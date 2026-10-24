import json
import math
from pathlib import Path

import numpy as np
import pytest

from a3_ultrasound.freeze_protocol import validate_freeze_config
from a3_ultrasound.selection import GuardedSelector
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.validation_diagnostics import CorrectionArchive


def row(y, pred, weight=1, mask=None):
    n=len(y);p=[.9 if v else .1 for v in pred]
    return dict(sample_id='sample',specimen_id='specimen',y=y,map_states=pred,
        label_mask=mask if mask is not None else [True]*n,marginals=p,
        base_probabilities=p,base_logodds=[0.]*n,raw_logZ=0.,rho=None,
        log_probabilities=[-math.log(2**n)]*(2**n),sample_weight=weight,
        gate=None,correction=None,gate_times_correction=None)


def test_missing_any_separates_extra_defects_and_preserves_raw_ipw():
    acc=PredictionAccumulator()
    for r in [row([1,1,0],[1,1,1],2),row([1,1,0],[1,0,0],3),
              row([0,1,1],[0,1,1],5),row([1,0,0],[1,1,0],7),
              row([0,0,0],[1,0,0],11)]:acc.add(r)
    m=acc.result()
    assert m['multiple_defect_missing_any_rate']==pytest.approx(.3)
    assert m['multiple_defect_joint_error_rate']==pytest.approx(.5)
    assert m['single_defect_to_multiple_rate']==1
    assert m['downstream_abnormal_recall_given_upstream_abnormal']==pytest.approx(.7)
    assert m['defect_diagnostics']['multiple_missing_any']==dict(
        weighted_numerator=3,weighted_denominator=10,raw_numerator=1,raw_denominator=3)
    assert m['defect_diagnostics']['downstream_recall_upstream_abnormal']['raw_denominator']==3


def test_partial_joint_excluded_but_known_upstream_abnormal_is_definite():
    acc=PredictionAccumulator();acc.add(row([1,1,1],[1,0,1],13,[True,False,True]))
    m=acc.result()
    assert m['multiple_defect_missing_any_rate'] is None
    assert m['multiple_defect_joint_error_rate'] is None
    assert m['downstream_abnormal_recall_given_upstream_abnormal']==1
    assert m['defect_diagnostics']['downstream_recall_upstream_abnormal']['weighted_denominator']==13
    acc=PredictionAccumulator();acc.add(row([0,0,1],[0,0,1],1,[False,True,True]))
    assert acc.result()['downstream_abnormal_recall_given_upstream_abnormal'] is None


def test_raw_archive_keeps_unknown_truth_and_selected_epoch_without_pickle(tmp_path):
    r=row([1,0,1],[1,0,1],13,[True,False,True])
    r.update(gate=[0.,.25,.5],correction=[0.,1.25,-2.],gate_times_correction=[0.,.3125,-1.])
    archive=CorrectionArchive();archive.add(r)
    path=tmp_path/'best_guarded_diagnostics.npz'
    archive.save(path,epoch=17,checkpoint='best_guarded.pt',commit_hash='a'*40,config_sha256='b'*64)
    with np.load(path,allow_pickle=False) as p:
        assert p['g'].tolist()==[.25,.5] and p['gc'].tolist()==[.3125,-1.]
        assert p['interface_index'].tolist()==[1,2] and p['current_truth'].tolist()==[-1,1]
        assert p['upstream_truth'].tolist()==[[1,-1,-1],[1,-1,-1]]
        assert p['upstream_interface_mask'].tolist()==[[True,False,False],[True,True,False]]
        assert p['upstream_any_abnormal'].tolist()==[1,1]
        assert json.loads(str(p['metadata_json']))['epoch']==17
        assert all(p[k].dtype.kind!='O' for k in p.files)


def template():
    return json.loads((Path(__file__).parents[1]/'configs/experiment/v1_4_freeze.json').read_text(encoding='utf-8'))


def test_freeze_closed_before_catalog_or_training(tmp_path):
    from a3_ultrasound.stream_training import train
    cfg=template();cfg['catalog']='MISSING'
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='explicitly restart'):
        train(path,tmp_path/'not_started',fold_name='F1',seed=23,head='proposed')
    assert not (tmp_path/'not_started').exists()
    validate_freeze_config(cfg,fold_name='F1',seed=23,head='proposed',require_permission=False)


@pytest.mark.parametrize('section,key,value',[
    ('training','normal_fpr_limit',.01),('training','epochs',10),
    ('model','rho',1.),(None,'heads',['proposed','proposed_shadow_aux']),
    (None,'automatic_screening',True),(None,'outer_evaluation_permitted',True),
    (None,'development_stage',2)])
def test_final_protocol_rejects_old_thresholds_and_variants(section,key,value):
    cfg=template();(cfg if section is None else cfg[section])[key]=value
    with pytest.raises(ValueError):validate_freeze_config(cfg,fold_name='F1',seed=23,head='proposed',require_permission=False)


def test_two_percent_guard_and_nll_tie_do_not_use_diagnostic_metrics():
    s=GuardedSelector(normal_fpr_limit=.02)
    def m(score,fpr,nll=.4):
        return dict(positive_joint_map_accuracy=score,normal_joint_false_positive_rate=fpr,joint_nll=nll,
                    multiple_defect_missing_any_rate=1.,downstream_abnormal_recall_given_upstream_abnormal=0.)
    assert s.update(1,m(.6,.02))['guarded_updated']
    assert not s.update(2,m(.8,.020001))['guarded_updated']
    assert s.fallback['epoch']==2 and s.guarded['epoch']==1
    assert s.update(3,m(.601,.019,.3))['guarded_updated']
    assert s.guarded['epoch']==3


def test_B8_earliest_no_progress_stop_is_epoch25_without_metric_screening():
    s=GuardedSelector(min_epochs=15,patience=10,min_delta=.002,normal_fpr_limit=.02)
    m=dict(positive_joint_map_accuracy=.6,normal_joint_false_positive_rate=.02,joint_nll=.4,
        correction_saturation_fraction_MAP_path=1.,correction_direction={'upper_correct':{'accuracy':0.}})
    for epoch in range(1,16):
        decision=s.update(epoch,m)
        assert not decision['stop'] and decision['stale']==0
    for epoch in range(16,25):
        assert not s.update(epoch,m)['stop']
    assert s.update(25,m)['stop'] and s.stale==10
