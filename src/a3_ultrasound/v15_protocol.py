"""Independent V1.5 development freeze; V1.4 retains its own validator."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL='V1.5-freeze'
HEADS=('local_independent','local_pre_independent','vanilla_chain','proposed_v15')


def validate_config(cfg,*,fold_name,seed,head,require_permission=True):
    t,m=cfg['training'],cfg['model']
    execution = cfg.get('execution_version', 'v1.5-freeze')
    if (execution not in ('v1.5-freeze', 'v1.5.1', 'v1.5.2')
            or m.get('canonical_per_trace', True) is not (execution == 'v1.5-freeze')):
        raise ValueError('Backbone execution mode differs from the declared V1.5 version.')
    if execution == 'v1.5.2':
        if (cfg.get('input_window_policy') != {'name': 'board_timebase', 'halfwidth_samples': 80}
                or m.get('local_prior_sigma_samples') != 40.0
                or m.get('local_prior_strength') != 0.5
                or cfg.get('window_jitter') != {'enabled':True,'max_shift_samples':24,'seed':23,'distribution':'uniform'}):
            raise ValueError('V1.5.2 requires board windows80, sigma40/strength0.5 and shared jitter24.')
    elif cfg.get('input_window_policy') is not None:
        raise ValueError('Earlier V1.5 versions cannot use the V1.5.2 window policy.')
    fixed=dict(epochs=40,min_epochs=15,patience=10,min_delta=.002,
        normal_fpr_limit=.02,selection_rule='guarded_joint',batch_size=16,
        learning_rate=.001,weight_decay=0.,lambda_local=.3,scheduler_step_size=None,
        disable_local_auxiliary=False,positive_class_weight=None,
        outer_evaluation_permitted=False,device='cuda',real_training_permitted=True,
        local_aux_mode='clean_upstream',prefix_replacement_probability=.2)
    if (cfg.get('protocol')!=PROTOCOL or cfg.get('heads')!=list(HEADS)
            or cfg.get('development_stage')!='v1.5-freeze'
            or cfg.get('seeds')!=[23] or seed!=23 or fold_name!='F1' or head not in HEADS
            or len(cfg.get('folds',[]))!=1 or cfg['folds'][0]['fold']!='F1'
            or cfg.get('outer_evaluation_permitted') is not False
            or cfg.get('formal_training_permitted') is not False
            or cfg.get('automatic_screening') is not False
            or cfg.get('normal_to_positive')!=3 or cfg.get('label_version')!='V2'
            or cfg.get('evaluation_batch_size')!=128
            or cfg.get('save_epoch_checkpoints') is not True
            or cfg.get('save_best_correction_npz') is not True
            or cfg.get('input_tail_policy')!='nominal_fixed'
            or m.get('architecture_version')!='a3-v1.5'
            or m.get('backbone')!='resnet1d' or m.get('backbone_norm')!='group8'
            or m.get('query_view')!='position2' or m.get('rho')!=3.
            or m.get('physical_features') is not True
            or m.get('pre_exclude_front_end') not in (None,180)
            or any(t.get(k)!=v for k,v in fixed.items())):
        raise ValueError('Configuration differs from V1.5-freeze.')
    if any('checkpoint' in k.lower() and k!='save_epoch_checkpoints' for k in cfg):
        raise ValueError('V1.5 always starts from fresh weights.')
    if require_permission and cfg.get('freeze_training_permitted') is not True:
        raise ValueError('V1.5 prerequisites/launch permission are not complete.')


def load_physical_artifacts(cfg):
    """Read immutable train-fit statistics and the predetermined M6 decision."""
    values={}
    for name in ('physical_statistics','m6_decision'):
        path=Path(cfg[name+'_path'])
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg[name+'_sha256']:
            raise ValueError(name+' artifact changed.')
        values[name]=json.loads(raw.decode('utf-8'))
    stats=values['physical_statistics'];decision=values['m6_decision']
    if stats.get('input_window_policy') != cfg.get('input_window_policy'):
        raise ValueError('Physical statistics were fitted with a different window policy.')
    if (cfg.get('execution_version')=='v1.5.2'
            and stats.get('timebase_calibration_sha256')!=cfg.get('timebase_calibration_sha256')):
        raise ValueError('Physical statistics were fitted with a different timebase artifact.')
    if (stats.get('front_end_samples')!=180 or stats.get('epsilon')!=1e-8
            or stats.get('entities')!=cfg['folds'][0]['train']
            or stats.get('variance_ddof')!=0 or stats.get('interface_count')!=3836015
            or stats.get('input_tail_policy')!='nominal_fixed'
            or not math.isfinite(stats.get('mean',float('nan')))
            or not math.isfinite(stats.get('std',float('nan'))) or stats['std']<=0):
        raise ValueError('Invalid F1 train-only physical normalization.')
    median=decision.get('front_mass_nonfirst_raw_median')
    if (not isinstance(median,(int,float)) or not math.isfinite(median)
            or not 0<=median<=1 or decision.get('threshold')!=.5
            or decision.get('front_window_samples')!=[0,180]
            or decision.get('epoch')!=12 or decision.get('validation_count')!=40002
            or decision.get('outer_opened') is not False
            or decision.get('enable_m6')!=(median>.5)
            or decision.get('pre_exclude_front_end')!=(180 if median>.5 else None)
            or cfg['model'].get('pre_exclude_front_end')!=decision['pre_exclude_front_end']):
        raise ValueError('M6 setting does not follow the fixed attention decision.')
    return stats,decision


def load_timebase_artifact(cfg):
    if cfg.get('execution_version')!='v1.5.2':return None
    raw=Path(cfg['timebase_calibration_path']).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=cfg['timebase_calibration_sha256']:
        raise ValueError('Board timebase calibration artifact changed.')
    value=json.loads(raw.decode('utf-8'))
    if value.get('version')!='v1.5.2-selected-timebase' or len(value.get('rows',[]))!=13:
        raise ValueError('Invalid board timebase calibration table.')
    value['_artifact_sha256']=digest
    return value


def copy_runtime_artifacts(cfg,output):
    """Retain exact input bytes; runtime atomic JSON uses different newlines."""
    from a3_ultrasound.runtime_supervisor import atomic_write
    for key,name in (('physical_statistics','physical_normalization.json'),('m6_decision','m6_decision.json')):
        raw=Path(cfg[key+'_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg[key+'_sha256']:
            raise ValueError(key+' artifact changed.')
        atomic_write(Path(output)/name,lambda handle:handle.write(raw))
    if cfg.get('execution_version')=='v1.5.2':
        raw=Path(cfg['timebase_calibration_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg['timebase_calibration_sha256']:
            raise ValueError('Timebase artifact changed.')
        atomic_write(Path(output)/'timebase_calibration.json',lambda handle:handle.write(raw))
