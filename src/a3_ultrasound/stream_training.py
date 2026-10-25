"""Sharded train/validation runner with separately invoked outer evaluation."""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
import hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.training import ModelConfig,TrainingConfig,TensorDataset,build_model,build_optimizer,evaluate,_validate_training_config,_code_version
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.prefix_perturbation import training_conditional
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.selection import GuardedSelector
from a3_ultrasound.data.epoch_sampling import use_epoch
from a3_ultrasound.validation_diagnostics import CorrectionArchive
from a3_ultrasound.freeze_protocol import PROTOCOL as FREEZE_PROTOCOL,validate_freeze_config,verify_panels,commit_identity
from a3_ultrasound.v15_protocol import PROTOCOL as V15_PROTOCOL,validate_config as validate_v15_config,load_physical_artifacts,copy_runtime_artifacts
from a3_ultrasound.v15_protocol import load_timebase_artifact


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def progress(path, **fields):
    if path is not None:
        temp=path.with_suffix('.tmp')
        save(temp,dict(heartbeat_epoch=time.time(),**fields));temp.replace(path)


def catalog_signature(catalog):
    # Data locations can change on the target host; scientific inputs cannot.
    keys=('entity','K','layer_mm','group','pool_sha256','index_sha256','records')
    data={'input_config':catalog['input_config'],
          'entities':[{k:r.get(k) for k in keys} for r in catalog['entities']]}
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def evaluate_stream(model,data,*,batch_size,device,prediction_path=None,progress_path=None,phase='evaluation',epoch=None,correction_archive=None):
    primary=PredictionAccumulator();nominal=PredictionAccumulator();quality=PredictionAccumulator()
    per_specimen=defaultdict(PredictionAccumulator);per_material=defaultdict(PredictionAccumulator)
    materials={s['entity']:('AL' if s['group']=='AL' else 'TC4') for s in data.shards}
    handle=prediction_path.open('w',encoding='utf-8') if prediction_path else None
    try:
        seen=0
        for batch,indices in data.batches(batch_size,device):
            names=[data.identity(i) for i in indices]
            small=TensorDataset(batch,tuple(n[0] for n in names),tuple(n[1] for n in names))
            _,rows=evaluate(model,small,batch_size=len(indices),device=device)
            for row,annotation in zip(rows,data.annotations(indices)):
                row.update(annotation);primary.add(row)
                per_specimen[row['specimen_id']].add(row)
                per_material[materials[row['specimen_id']]].add(row)
                if correction_archive is not None:correction_archive.add(row)
                nominal.add(dict(row,y=row['nominal_y'],label_mask=row['nominal_label_mask']))
                if row['quality_flagged']:quality.add(row)
                if handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            seen+=len(indices)
            if seen==len(data) or seen%(batch_size*100)==0:
                progress(progress_path,phase=phase,epoch=epoch,records=seen,total_records=len(data))
    finally:
        if handle:handle.close()
    return {'reliable':primary.result(),'nominal_secondary':nominal.result(),'quality_flagged':quality.result(),
            'per_specimen':{k:v.result() for k,v in per_specimen.items()},
            'per_material':{k:v.result() for k,v in per_material.items()}}


def validate_fold(fold,catalog):
    groups=[set(fold[k]) for k in ('train','validation','test')]
    if any(not g for g in groups) or any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Entity split overlap or empty split.')
    all_entities={e['entity'] for e in catalog['entities']}
    if set.union(*groups)!=all_entities:raise ValueError('Split must cover catalog exactly.')


def train(config_path,output,*,fold_name,head,seed,diagnostic=False):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    v15=cfg.get('protocol')==V15_PROTOCOL
    frozen=cfg.get('protocol') in (FREEZE_PROTOCOL,V15_PROTOCOL)
    code_version=_code_version()
    config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if frozen:
        if diagnostic:raise ValueError('Frozen real runs cannot use a train-only diagnostic override.')
        (validate_v15_config if v15 else validate_freeze_config)(cfg,fold_name=fold_name,seed=seed,head=head)
        commit_identity(cfg,code_version)
    catalog_path=Path(cfg['catalog']);catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    fold=next(f for f in cfg['folds'] if f['fold']==fold_name);validate_fold(fold,catalog)
    if seed not in cfg['seeds'] or head not in cfg['heads']:raise ValueError('Unregistered seed or Head.')
    if diagnostic and fold_name not in cfg['diagnostic']['folds']:
        raise ValueError('This fold is not registered for local diagnostic runs.')
    development=not frozen and cfg.get('development_stage')==2
    if development:
        if (not cfg.get('development_training_permitted') or cfg.get('formal_training_permitted') or
            cfg.get('outer_evaluation_permitted') or fold_name!='F1' or seed!=23 or
            head not in ('proposed','proposed_shadow_aux') or cfg['training']['epochs']>10):
            raise ValueError('Stage2 is restricted to reviewed F1/seed23 Proposed diagnostics, <=10 epochs, outer closed.')
        if (cfg['training'].get('selection_rule')!='guarded_joint' or
            cfg['model'].get('architecture_version')!='a3-v1.4' or not cfg.get('epoch_selection_root')):
            raise ValueError('Stage2 requires the complete V1.4 protocol and frozen epoch indices.')
        screening=json.loads(Path(cfg['stage1_result']).read_text(encoding='utf-8'))
        from a3_ultrasound.head_cache import select_stage1_candidates
        passed=select_stage1_candidates(*screening['results'])
        candidate=f"{head}_rho{int(cfg['model']['rho'])}"
        if (screening.get('status')!='COMPLETED' or candidate not in passed or passed!=screening.get('passed_candidates') or
            screening.get('fold')!=fold or screening.get('seed')!=seed or screening.get('catalog_signature')!=catalog_signature(catalog)):
            raise ValueError('Stage2 candidate did not pass stage1.')
    if not diagnostic and not development and not frozen and not cfg['formal_training_permitted']:
        raise ValueError('Formal training awaits final 3080 Ti authorization/preflight.')
    tvals=dict(cfg['training'],seed=seed)
    if diagnostic:tvals.update(cfg['diagnostic']['training'])
    tc=TrainingConfig(**tvals);_validate_training_config(tc)
    evaluation_batch_size=cfg.get('evaluation_batch_size',tc.batch_size)
    if type(evaluation_batch_size) is not int or evaluation_batch_size<1:
        raise ValueError('evaluation_batch_size must be a positive integer.')
    if tc.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback.')
    model_config=ModelConfig(**dict(cfg['model'],head_type=head))
    physical, m6 = load_physical_artifacts(cfg) if v15 else (None,None)
    timebase = load_timebase_artifact(cfg) if v15 else None
    if model_config.physical_features and not v15:
        raise ValueError('Physical stream training requires the V1.5 normalization contract.')
    if frozen:verify_panels(cfg,catalog_signature(catalog))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    training=RFShardDataset(catalog_path,fold['train'],training=True,seed=seed,
        normal_ratio=cfg['normal_to_positive'],
        diagnostic_per_class=cfg['diagnostic']['per_entity_per_class'] if diagnostic else None,
        physical_config=physical,tail_policy=cfg.get('input_tail_policy'),
        window_policy=cfg.get('input_window_policy'),timebase_config=timebase,
        window_jitter=cfg.get('window_jitter'))
    if not diagnostic and not cfg.get('epoch_selection_root'):
        selection=Path(cfg['frozen_selection_root'])/f'{fold_name}_s{seed}.npy'
        if not np.array_equal(training.refs,np.load(selection,allow_pickle=False)):
            raise ValueError('Training selection differs from frozen specimen/seed indices.')
    # Diagnostic never opens outer validation or test RF. Formal uses validation only.
    validation=None if diagnostic else RFShardDataset(catalog_path,fold['validation'],
        physical_config=physical,tail_policy=cfg.get('input_tail_policy'),
        window_policy=cfg.get('input_window_policy'),timebase_config=timebase)
    validation_sampling=None
    if validation is not None and cfg.get('validation_selection_root'):
        selection_root=Path(cfg['validation_selection_root'])
        selection_manifest=json.loads((selection_root/'manifest.json').read_text(encoding='utf-8'))
        selection_path=selection_root/f'{fold_name}.npz'
        selection_sha=hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if selection_manifest['catalog_signature']!=catalog_signature(catalog) or selection_manifest['folds'][fold_name]['sha256']!=selection_sha:
            raise ValueError('Frozen validation selection or catalog signature mismatch.')
        population=len(validation);validation.use_frozen_evaluation_selection(selection_path)
        if frozen and len(validation)!=40002:raise ValueError('Frozen validation must retain 40,002 input rows.')
        validation_sampling=dict(policy=selection_manifest['policy'],selection_sha256=selection_sha,
            evaluated_records=len(validation),population_records=population,seed=selection_manifest['seed'])
    if cfg.get('epoch_selection_root'):
        use_epoch(training,cfg['epoch_selection_root'],1,seed=seed,catalog_signature=catalog_signature(catalog))
        if frozen and len(training)!=100732:raise ValueError('Frozen F1 must use all positives and 100,732 training rows per epoch.')
    model=build_model(model_config).to(tc.device);optimizer,counts=build_optimizer(model,tc)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    np.save(output/'training_selection.npy',training.refs)
    resolved={'model':asdict(model_config),'training':asdict(tc),'fold':fold,'input_config':training.config,
        'catalog':str(catalog_path),'catalog_signature':catalog_signature(catalog),
        'selection_entities':training.entities,'selection_counts':training.selection_summary(),
        'diagnostic':diagnostic,'test_opened':False,'parameter_counts':counts,
        'evaluation_batch_size':evaluation_batch_size,
        'validation_sampling':validation_sampling,
        'selection_rule':'diagnostic uses only train panel; formal validation specimen macro-F1, tie lower complete joint NLL',
        'gpu':torch.cuda.get_device_name() if tc.device=='cuda' else None,'seed':seed,
        'learning_scope':'in-sample engineering diagnostic; no cross-fold hyperparameter selection' if diagnostic else 'outer train/validation only'}
    if cfg.get('protocol_signature'):
        resolved.update(protocol_signature=cfg['protocol_signature'],label_version=cfg.get('label_version'),
                        prediction_contract='joint-path-v2')
    resolved['selection_rule']=tc.selection_rule
    resolved['architecture_version']=model_config.architecture_version
    resolved['epoch_selection_root']=cfg.get('epoch_selection_root')
    resolved['training_selection_file_scope']='epoch1 only; per-epoch refs/order in epoch_selection_root' if cfg.get('epoch_selection_root') else 'fixed cohort'
    resolved['development_stage']=cfg.get('development_stage')
    if v15:
        resolved.update(physical_normalization=physical,m6_decision=m6,
            physical_statistics_sha256=cfg['physical_statistics_sha256'],m6_decision_sha256=cfg['m6_decision_sha256'],
            physical_interpretation='Engineering energy ratio; front and first Local overlap on 3mm plates; not calibrated reflectivity or causal isolation.')
        copy_runtime_artifacts(cfg,output)
        if timebase is not None:
            resolved.update(timebase_calibration_sha256=cfg['timebase_calibration_sha256'],
                window_jitter=cfg['window_jitter'],
                window_jitter_identity='seed,epoch,catalog ordinal,source_row; common to all Heads; evaluation shift0')
    resolved.update(code_version=code_version,commit_hash=code_version.get('commit'),config_sha256=config_sha256,
                    protocol=cfg.get('protocol'),automatic_screening=False if frozen else development,old_weights_reused=False)
    if development:
        resolved['budget_exception']='Stage2 max10 is shorter than min15; no selector early stop, only exploratory alarms. Not convergence evidence.'
    save(output/'resolved_config.json',resolved)
    history=[];best=(-float('inf'),-float('inf'));stale=0;updates=0;alarm_reasons=[]
    generator=torch.Generator().manual_seed(seed)
    selector=GuardedSelector(tc.min_epochs,tc.patience,tc.min_delta,tc.normal_fpr_limit) if tc.selection_rule=='guarded_joint' else None
    if selector and diagnostic:raise ValueError('Guarded selection requires fixed held-out validation.')
    try:
        before=evaluate_stream(model,training,batch_size=tc.batch_size,device=tc.device) if diagnostic else None
        if before:save(output/'before.json',before)
        for epoch in range(1,tc.epochs+1):
            epoch_sha=None
            if cfg.get('epoch_selection_root'):
                epoch_sha=use_epoch(training,cfg['epoch_selection_root'],epoch,seed=seed,catalog_signature=catalog_signature(catalog))
            model.train();total=0.;seen=0
            perturbation=dict(attempted=0,changed=0,eligible=0)
            perturbation_hash=hashlib.sha256()
            progress(output/'progress.json',phase='training',epoch=epoch,records=0,total_records=len(training),updates_total=updates)
            for step,(batch,ids) in enumerate(training.batches(tc.batch_size,tc.device,generator=None if cfg.get('epoch_selection_root') else generator)):
                optimizer.zero_grad(set_to_none=True)
                out,attempt=training_conditional(model,batch,probability=tc.prefix_replacement_probability,
                    seed=seed,epoch=epoch,step=step)
                for name in perturbation:perturbation[name]+=attempt[name]
                if attempt['attempt_mask_sha256']:perturbation_hash.update(attempt['attempt_mask_sha256'].encode('ascii'))
                loss=head_training_loss(out,batch,head_type=head,lambda_local=tc.lambda_local,
                    positive_class_weight=tc.positive_class_weight,disable_local_auxiliary=tc.disable_local_auxiliary,
                    local_aux_mode=tc.local_aux_mode)
                if not torch.isfinite(loss.total):raise RuntimeError('Nonfinite loss.')
                loss.total.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient.')
                optimizer.step();total+=float(loss.total.detach())*len(ids);seen+=len(ids);updates+=1
                if updates%100==0 or seen==len(training):
                    progress(output/'progress.json',phase='training',epoch=epoch,records=seen,total_records=len(training),updates_total=updates)
            record={'epoch':epoch,'train_loss':total/seen,'records':seen,'updates_total':updates}
            if tc.prefix_replacement_probability:
                record['prefix_perturbation']=dict(perturbation,attempt_sequence_sha256=perturbation_hash.hexdigest())
            if validation is not None:
                correction_archive=CorrectionArchive() if head in ('proposed','proposed_v15') and cfg.get('save_best_correction_npz') else None
                progress(output/'progress.json',phase='validation',epoch=epoch,records=0,total_records=len(validation))
                metrics=evaluate_stream(model,validation,batch_size=evaluation_batch_size,device=tc.device,
                    progress_path=output/'progress.json',phase='validation',epoch=epoch,correction_archive=correction_archive)
                score=metrics['reliable'];key=(score['deepest_specimen_macro_f1'],-score['joint_nll'] if score['joint_nll'] is not None else -float('inf'))
                if key[0] is None:raise RuntimeError('Validation has no reliable deepest labels.')
                record['validation']=metrics
                if selector:record['selection']=selector.update(epoch,score)
            else:key=(float(epoch),0.)
            record['epoch_selection_sha256']=epoch_sha
            if selector:
                payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'resolved_config':resolved,'epoch':epoch,'selection_state':selector.state_dict(),
                    'diagnostic_weights_never_reuse':diagnostic}
                if record['selection']['guarded_updated']:torch.save(payload,output/'best_guarded.pt')
                if record['selection']['fallback_updated']:torch.save(payload,output/'best_fallback.pt')
                if correction_archive is not None:
                    for kind in ('guarded','fallback'):
                        if record['selection'][kind+'_updated']:
                            correction_archive.save(output/f'best_{kind}_diagnostics.npz',epoch=epoch,
                                checkpoint=f'best_{kind}.pt',commit_hash=code_version.get('commit'),config_sha256=config_sha256)
            history.append(record);save(output/'history.json',history)
            if selector is None and key>best:
                best=key;stale=0
                torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'resolved_config':resolved,'epoch':epoch,'diagnostic_weights_never_reuse':diagnostic},output/'best.pt')
            else:stale+=1
            if cfg.get('save_epoch_checkpoints',False):
                np_state=np.random.get_state()
                torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'resolved_config':resolved,'epoch':epoch,'updates_total':updates,
                    'epoch_selection_sha256':epoch_sha,'selection_state':selector.state_dict() if selector else None,
                    'torch_rng_state':torch.get_rng_state(),'shuffle_rng_state':generator.get_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'python_rng_state':random.getstate(),
                    'numpy_rng_state':{'generator':np_state[0],'keys':np_state[1].tolist(),
                        'position':np_state[2],'has_gauss':np_state[3],'cached_gaussian':np_state[4]}},
                    output/f'epoch_{epoch:02d}.pt')
            print(json.dumps({'fold':fold_name,'head':head,**{k:v for k,v in record.items() if k!='validation'}},ensure_ascii=True),flush=True)
            if development:
                alarm_reasons=stage2_alarms(epoch,score)
                if alarm_reasons:
                    save(output/'screening_stop.json',dict(epoch=epoch,reasons=alarm_reasons))
                    break
            if selector:
                if record['selection']['stop']:break
            elif not diagnostic and stale>=tc.patience:break
        selected_name=('best_guarded.pt' if selector.guarded else 'best_fallback.pt') if selector else 'best.pt'
        checkpoint=torch.load(output/selected_name,map_location=tc.device,weights_only=True);model.load_state_dict(checkpoint['model_state_dict'])
        target=training if diagnostic else validation
        final=evaluate_stream(model,target,batch_size=evaluation_batch_size,device=tc.device,prediction_path=output/'predictions.jsonl',
            progress_path=output/'progress.json',phase='final_validation',epoch=checkpoint['epoch'])
        result={'status':'COMPLETED','diagnostic':diagnostic,'updates':updates,'epochs':len(history),'best_epoch':checkpoint['epoch'],
            'seconds':time.perf_counter()-started,'selection_counts':training.selection_summary(),'metrics':final,
            'test_opened':False,'performance_claims_released':False,'old_weights_reused':False}
        result.update(selected_checkpoint=selected_name,guard_violated=selector.guarded is None if selector else None,
                      result_role='diagnostic_fallback_only' if selector and selector.guarded is None else 'guarded_or_legacy')
        result.update(commit_hash=code_version.get('commit'),config_sha256=config_sha256,code_version=code_version,
            protocol=cfg.get('protocol'),best_guarded=selector.guarded if selector else None,
            best_fallback=selector.fallback if selector else None,
            stop_reason='stage2_screening' if alarm_reasons else 'early_stopping_B8' if selector and len(history)<tc.epochs else 'legacy_early_stopping' if len(history)<tc.epochs else 'max_epochs',automatic_screening=False if frozen else development)
        if frozen:
            result.update(frozen_run=True,automatic_next_stage=False)
        if development:
            result.update(status='SCREENED_OUT' if alarm_reasons else 'COMPLETED_DEVELOPMENT_DIAGNOSTIC',
                          formal_result=False,screening_reasons=alarm_reasons,development_stage=2)
        save(output/'summary.json',result)
        progress(output/'progress.json',phase='completed',epochs=len(history),updates_total=updates)
        return result
    except Exception as exc:
        save(output/'failure.json',{'type':type(exc).__name__,'message':str(exc)});raise


def stage2_alarms(epoch,score):
    reasons=[]
    saturation=score.get('correction_saturation_fraction_MAP_path')
    direction=score.get('correction_direction',{}).get('upper_correct',{}).get('accuracy')
    fpr=score.get('normal_joint_false_positive_rate')
    if epoch>=3 and (saturation is None or saturation>.9):reasons.append('correction_saturation_above_90_or_undefined')
    if epoch>=5 and (direction is None or direction<.55):reasons.append('upper_correct_direction_below_55_or_undefined')
    if epoch>=5 and (fpr is None or fpr>.01):reasons.append('normal_false_positive_above_1_or_undefined')
    return reasons


def test_checkpoint(config_path,checkpoint_path,output):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg.get('protocol') in (FREEZE_PROTOCOL,V15_PROTOCOL):raise ValueError('Outer evaluation is closed for frozen development runs.')
    if not cfg.get('outer_evaluation_permitted',False):raise ValueError('Outer evaluation remains closed.')
    c=torch.load(checkpoint_path,map_location=cfg['training']['device'],weights_only=True);resolved=c['resolved_config']
    if resolved.get('development_stage') is not None:
        raise ValueError('Development diagnostic checkpoint cannot enter outer evaluation.')
    if c.get('selection_state') is not None and c['selection_state']['guarded'] is None:
        raise ValueError('Unguarded fallback cannot enter outer evaluation.')
    if resolved['diagnostic']:raise ValueError('Diagnostic checkpoint cannot enter outer evaluation.')
    fold=next(f for f in cfg['folds'] if f['fold']==resolved['fold']['fold'])
    if fold!=resolved['fold']:raise ValueError('Checkpoint fold differs from active frozen fold.')
    catalog=json.loads(Path(cfg['catalog']).read_text(encoding='utf-8'));validate_fold(fold,catalog)
    if resolved.get('catalog_signature')!=catalog_signature(catalog):
        raise ValueError('Checkpoint data/input policy differs from active catalog.')
    if cfg.get('protocol_signature') and resolved.get('protocol_signature')!=cfg['protocol_signature']:
        raise ValueError('Checkpoint training protocol differs from frozen phase.')
    if resolved['seed'] not in cfg['seeds'] or resolved['model']['head_type'] not in cfg['heads']:
        raise ValueError('Checkpoint seed or Head is not registered.')
    data=RFShardDataset(cfg['catalog'],fold['test'])
    if set(data.entities)&set(resolved['selection_entities']):raise ValueError('Test entity leakage.')
    model=build_model(ModelConfig(**resolved['model'])).to(cfg['training']['device']);model.load_state_dict(c['model_state_dict'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    metrics=evaluate_stream(model,data,batch_size=resolved.get('evaluation_batch_size',cfg['training']['batch_size']),
        device=cfg['training']['device'],prediction_path=output/'test_predictions.jsonl')
    save(output/'summary.json',{'status':'EVALUATED','fold':fold['fold'],
        'seed':resolved['seed'],'head':resolved['model']['head_type'],'catalog_signature':resolved['catalog_signature'],
        'protocol_signature':resolved.get('protocol_signature'),'label_version':resolved.get('label_version'),
        'metrics':metrics,'checkpoint':str(checkpoint_path),'performance_claims_released':False})
    return metrics
