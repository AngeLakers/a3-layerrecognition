"""Full validation audit of an already selected checkpoint; never selects a model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.stream_training import catalog_signature, evaluate_stream, save, validate_fold
from a3_ultrasound.training import ModelConfig, build_model


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_validation(config_path, run_directory, output, *, queue_state=None):
    """Evaluate original validation rows with unit weights; outer RF stays unopened."""
    cfg = _read(config_path)
    run = Path(run_directory)
    output = Path(output)
    if output.exists():
        raise ValueError('Audit output already exists; preserve prior evidence.')
    summary = _read(run/'summary.json')
    resolved = _read(run/'resolved_config.json')
    if (summary.get('status') != 'COMPLETED' or summary.get('diagnostic') is not False
            or summary.get('test_opened') is not False or resolved.get('diagnostic') is not False
            or resolved.get('test_opened') is not False):
        raise ValueError('Audit requires a completed formal train/validation run.')
    checkpoint_path = run/'best.pt'
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    device = cfg['training']['device']
    if device != 'cpu':
        if queue_state is None:
            raise ValueError('GPU audit requires a stopped training queue witness.')
        queue = _read(queue_state)
        if queue['status'] not in ('COMPLETED', 'STOPPED_AT_BOUNDARY') or any(
                job['status'] == 'RUNNING' for job in queue['jobs']):
            raise ValueError('GPU audit must wait until the training queue has stopped.')
        jobs = [job for job in queue['jobs'] if Path(job['output']).resolve() == run.resolve()]
        if len(jobs) != 1 or jobs[0]['status'] != 'COMPLETED' or jobs[0].get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError('Completed queue checkpoint witness mismatch.')
    # Deserialize on CPU after completion/resource checks, before opening any RF.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['resolved_config'] != resolved or checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError('Checkpoint identity differs from completed run artifacts.')
    fold = next(f for f in cfg['folds'] if f['fold'] == resolved['fold']['fold'])
    catalog = _read(cfg['catalog'])
    validate_fold(fold, catalog)
    if fold != resolved['fold'] or catalog_signature(catalog) != resolved['catalog_signature']:
        raise ValueError('Audit fold or scientific input signature mismatch.')
    head = resolved['model']['head_type']
    if resolved['seed'] not in cfg['seeds'] or head not in cfg['heads']:
        raise ValueError('Unregistered checkpoint seed or Head.')
    if asdict(ModelConfig(**dict(cfg['model'], head_type=head))) != resolved['model']:
        raise ValueError('Checkpoint model differs from active configuration.')
    data = RFShardDataset(cfg['catalog'], fold['validation'])
    if data.evaluation_weights is not None or set(data.entities) & (set(fold['train']) | set(fold['test'])):
        raise ValueError('Audit must use full independent validation with unit weights.')
    sampling = resolved.get('validation_sampling')
    if sampling and len(data) != sampling['population_records']:
        raise ValueError('Full validation population differs from frozen run.')
    model = build_model(ModelConfig(**resolved['model'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        metrics = evaluate_stream(model, data,
            batch_size=resolved.get('evaluation_batch_size', cfg['training']['batch_size']),
            device=device, progress_path=output/'progress.json', phase='full_validation_audit',
            epoch=checkpoint['epoch'])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_sha:
            raise RuntimeError('Source checkpoint changed during the audit.')
        full = metrics['reliable']
        compact = summary['metrics']['reliable']
        if full['sampling_weighted'] or full['evaluated_sample_count'] != len(data):
            raise RuntimeError('Full validation count/weight invariant failed.')
        keys = ('deepest_specimen_macro_f1', 'joint_nll', 'joint_map_accuracy')
        delta = {key: full[key]-compact[key] if full[key] is not None and compact[key] is not None else None
                 for key in keys}
        per_entity = {}
        for entity, values in full['deepest_per_specimen'].items():
            sampled = compact['deepest_per_specimen'][entity]
            per_entity[entity] = {key: values[key]-sampled[key] for key in ('macro_f1','brier','marginal_nll')}
        result = dict(status='AUDITED', checkpoint_sha256=checkpoint_sha,
            checkpoint=str(checkpoint_path), selected_epoch=checkpoint['epoch'],
            fold=fold['fold'], seed=resolved['seed'], head=head,
            catalog_signature=resolved['catalog_signature'],
            validation_entities=data.entities, evaluated_records=len(data),
            compact_evaluated_records=compact['evaluated_sample_count'],
            full_minus_compact=delta, deepest_per_entity_full_minus_compact=per_entity,
            metrics=metrics, seconds=time.perf_counter()-started,
            checkpoint_reselected=False, outer_opened=False, performance_claims_released=False,
            scope='selected-checkpoint sampling error audit; no checkpoint ranking stability claim')
        save(output/'summary.json', result)
        return result
    except Exception as exc:
        save(output/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
