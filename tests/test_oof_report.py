import copy
import pytest
from a3_ultrasound.oof_report import summarize_oof


def fixture_results():
    cfg={'folds':[{'fold':'F1','test':['a','b','c']},{'fold':'F2','test':['d','e','f']}],
         'seeds':[23,41,59,83,101],'heads':['proposed','vanilla_chain'],
         'statistics':{'primary_pair':['proposed','vanilla_chain'],'bootstrap_seed':20261022,
          'bootstrap_replicates':10000,'paired_positive_seeds_required':4,
          'calibration_guard_max_mean_delta_joint_nll':0.,'calibration_guard_max_mean_delta_deepest_brier':0.}}
    rows=[]
    for f in cfg['folds']:
        for seed in cfg['seeds']:
            for head in cfg['heads']:
                gain=.1 if head=='proposed' else 0
                per={e:{'macro_f1':.5+gain,'brier':.2-gain} for e in f['test']}
                rows.append({'status':'EVALUATED','fold':f['fold'],'seed':seed,'head':head,'catalog_signature':'fixed',
                             'metrics':{'reliable':{'deepest_per_specimen':per,'joint_complete_label_count':10,
                                                    'joint_nll':1-gain}}})
    return cfg,rows


def test_seed_entity_pairing_and_prespecified_bootstrap():
    cfg,rows=fixture_results();result=summarize_oof(cfg,rows,expected_catalog_signature='fixed')
    assert result['paired_difference_mean']==pytest.approx(.1)
    assert result['entity_cluster_bootstrap_95ci']==pytest.approx([.1,.1])
    assert result['numerical_H2_criteria_met'] and not result['claim_released']
    assert summarize_oof(cfg,list(reversed(rows)),expected_catalog_signature='fixed')==result


def test_incomplete_or_duplicate_or_mixed_version_rejected():
    cfg,rows=fixture_results()
    with pytest.raises(ValueError,match='Incomplete'):summarize_oof(cfg,rows[:-1],expected_catalog_signature='fixed')
    with pytest.raises(ValueError,match='duplicate'):summarize_oof(cfg,rows+[rows[0]],expected_catalog_signature='fixed')
    bad=copy.deepcopy(rows);bad[0]['catalog_signature']='changed'
    with pytest.raises(ValueError,match='different data'):summarize_oof(cfg,bad,expected_catalog_signature='fixed')
    with pytest.raises(ValueError,match='active catalog'):summarize_oof(cfg,rows,expected_catalog_signature='new')


def test_calibration_degradation_prevents_numerical_H2_pass():
    cfg,rows=fixture_results()
    for r in rows:
        if r['head']=='proposed':r['metrics']['reliable']['joint_nll']=2.
    result=summarize_oof(cfg,rows,expected_catalog_signature='fixed')
    assert result['f1_criterion_met'] and not result['numerical_H2_criteria_met']
