import copy
import hashlib
import json
from pathlib import Path
import pytest

from a3_ultrasound.v15_protocol import HEADS,validate_config,load_physical_artifacts,copy_runtime_artifacts


def candidate():
    c=json.loads((Path(__file__).parents[1]/'configs/experiment/v1_4_freeze.json').read_text(encoding='utf-8'))
    c.update(protocol='V1.5-freeze',development_stage='v1.5-freeze',heads=list(HEADS),input_tail_policy='nominal_fixed')
    c['model'].update(architecture_version='a3-v1.5',physical_features=True,pre_exclude_front_end=None)
    c['training'].update(local_aux_mode='clean_upstream',prefix_replacement_probability=.2)
    return c


def test_v15_closed_until_prerequisites_and_outer_always_closed(tmp_path):
    from a3_ultrasound.stream_training import train,test_checkpoint
    c=candidate();c['catalog']='MISSING'
    for head in HEADS:validate_config(c,fold_name='F1',seed=23,head=head,require_permission=False)
    path=tmp_path/'run.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='prerequisites'):
        train(path,tmp_path/'out',fold_name='F1',seed=23,head='proposed_v15')
    c['outer_evaluation_permitted']=True;path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='closed'):
        test_checkpoint(path,'MISSING',tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('section,key,value',[
    ('model','architecture_version','a3-v1.4'),('model','physical_features',False),
    ('model','pre_exclude_front_end',70),('training','local_aux_mode','all'),
    ('training','prefix_replacement_probability',0.),('training','normal_fpr_limit',.01),
    (None,'heads',['proposed_v15']), (None,'input_tail_policy','last_active_tail')])
def test_protocol_rejects_unreviewed_mix(section,key,value):
    c=candidate();(c if section is None else c[section])[key]=value
    with pytest.raises(ValueError):validate_config(c,fold_name='F1',seed=23,head='proposed_v15',require_permission=False)


@pytest.mark.parametrize('median',[.5,.500001])
def test_norm_and_m6_replay_binding(tmp_path,median):
    c=candidate();c['model']['pre_exclude_front_end']=180 if median>.5 else None
    stats=dict(front_end_samples=180,epsilon=1e-8,mean=-6.4,std=1.8,entities=c['folds'][0]['train'],variance_ddof=0,interface_count=3836015,input_tail_policy='nominal_fixed')
    decision=dict(front_window_samples=[0,180],front_mass_nonfirst_raw_median=median,threshold=.5,epoch=12,validation_count=40002,outer_opened=False,enable_m6=median>.5,pre_exclude_front_end=c['model']['pre_exclude_front_end'])
    for name,value in [('physical_statistics',stats),('m6_decision',decision)]:
        path=tmp_path/(name+'.json')
        path.write_bytes(json.dumps(value,indent=2).replace('\n','\r\n').encode('utf-8'))
        c[name+'_path']=str(path);c[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_physical_artifacts(c)==(stats,decision)
    copy_runtime_artifacts(c,tmp_path/'run')
    assert (tmp_path/'run/physical_normalization.json').read_bytes()==Path(c['physical_statistics_path']).read_bytes()
    assert (tmp_path/'run/m6_decision.json').read_bytes()==Path(c['m6_decision_path']).read_bytes()
    wrong=copy.deepcopy(c);wrong['model']['pre_exclude_front_end']=None if median>.5 else 180
    with pytest.raises(ValueError,match='M6'):load_physical_artifacts(wrong)
    Path(c['physical_statistics_path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):load_physical_artifacts(c)
