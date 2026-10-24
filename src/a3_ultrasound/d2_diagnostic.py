"""V1.4-D2: two fresh end-to-end development runs with a fixed ten-epoch budget."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from a3_ultrasound.data.epoch_sampling import use_epoch
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.runtime_supervisor import atomic_json, file_digest
from a3_ultrasound.stream_training import catalog_signature, evaluate_stream, progress, validate_fold
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, build_optimizer


PROTOCOL = 'V1.4-D2'
HEADS = ('proposed', 'vanilla_chain')


def validate_config(cfg, *, require_permission=True):
    """Keep the independent diagnostic from opening D1, other runs, or outer RF."""
    t, m = cfg['training'], cfg['model']
    expected = dict(seed=23, epochs=10, batch_size=16, learning_rate=.001, weight_decay=0.,
                    lambda_local=.3, positive_class_weight=None, disable_local_auxiliary=False,
                    scheduler_step_size=None, device='cuda', min_delta=.002,
                    selection_rule='diagnostic_joint', normal_fpr_limit=None,
                    outer_evaluation_permitted=False, real_training_permitted=True)
    if (cfg.get('protocol') != PROTOCOL or cfg.get('development_stage') != 'D2'
            or cfg.get('seeds') != [23] or cfg.get('heads') != list(HEADS)
            or cfg.get('label_version') != 'V2' or len(cfg.get('folds', [])) != 1
            or cfg['folds'][0]['fold'] != 'F1' or cfg.get('normal_to_positive') != 3
            or cfg.get('evaluation_batch_size') != 128
            or cfg.get('formal_training_permitted') is not False
            or cfg.get('outer_evaluation_permitted') is not False
            or any(t.get(k) != v for k, v in expected.items())
            or m.get('architecture_version') != 'a3-v1.4'
            or m.get('backbone') != 'resnet1d' or m.get('backbone_norm') != 'group8'
            or m.get('query_view') != 'position2' or m.get('rho') != 3.):
        raise ValueError('Configuration is outside the fixed V1.4-D2 scope.')
    if any(k in section for section in (cfg, t, m) for k in ('amp', 'fp16', 'bf16', 'dtype', 'optimizer')):
        raise ValueError('D2 fixes FP32 AdamW; dtype/AMP/optimizer overrides are not accepted.')
    if any('checkpoint' in k.lower() and k != 'save_epoch_checkpoints' for k in cfg):
        raise ValueError('D2 never accepts initialization checkpoints.')
    if require_permission and not cfg.get('development_training_permitted'):
        raise ValueError('D2 launch permission is closed pending protocol review.')


class DiagnosticSelector:
    """The D1 fallback comparison alone, without FPR admissibility or early stop."""
    def __init__(self, min_delta=.002):
        self.min_delta = min_delta
        self.best = None

    def update(self, epoch, score):
        values = [score.get(k) for k in ('positive_joint_map_accuracy', 'joint_nll',
                                         'normal_joint_false_positive_rate')]
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError('D2 requires finite metrics and nonempty positive/normal validation subsets.')
        candidate = dict(epoch=epoch, score=values[0], nll=values[1], normal_fpr=values[2])
        # Same inclusive tolerance as the frozen D1 fallback, with no FPR branch.
        updated = self.best is None or candidate['score'] > self.best['score'] + self.min_delta or (
            abs(candidate['score'] - self.best['score']) <= self.min_delta + 1e-12
            and candidate['nll'] < self.best['nll'])
        if updated:
            self.best = candidate
        return dict(best_updated=updated, stop=False, normal_fpr_guard_enabled=False)

    def state_dict(self):
        return dict(best=self.best, min_delta=self.min_delta, normal_fpr_guard_enabled=False)


def summarize_rows(rows):
    """Raw counts, IPW counts, and exact medians from one epoch's persisted rows."""
    names = ('single_to_multiple', 'multiple_to_single', 'other')
    errors = {scope: {k: 0 for k in names} for scope in ('raw', 'ipw', 'abnormal_raw', 'abnormal_ipw')}
    groups = {key: [] for key in ('all_nonfirst', 'wrong_nonfirst')}
    direction = {key: [0, 0, 0, 0] for key in ('upper_correct', 'upper_wrong')}
    n = 0
    for row in rows:
        n += 1
        y, mask, pred = row['y'], row['label_mask'], row['map_states']
        if y is None:
            continue
        w = row.get('sample_weight', 1)
        if type(w) is not int or w < 1:
            raise ValueError('Positive integer IPW required.')
        if all(mask) and pred != y:
            truth_n, pred_n = sum(y), sum(pred)
            key = ('single_to_multiple' if truth_n == 1 and pred_n > 1 else
                   'multiple_to_single' if truth_n > 1 and pred_n == 1 else 'other')
            errors['raw'][key] += 1
            errors['ipw'][key] += w
            if any(y):
                errors['abnormal_raw'][key] += 1
                errors['abnormal_ipw'][key] += w
        if row.get('correction') is None:
            continue
        for k in range(1, len(y)):
            if not mask[k]:
                continue
            c, g, gc = (float(row[name][k]) for name in ('correction', 'gate', 'gate_times_correction'))
            if not all(math.isfinite(v) for v in (c, g, gc)):
                raise ValueError('Nonfinite correction diagnostic.')
            item = (abs(c) >= .95 * row['rho'], g, gc, w)
            groups['all_nonfirst'].append(item)
            if pred[k] != y[k]:
                groups['wrong_nonfirst'].append(item)
            if all(mask[:k]):
                key = 'upper_correct' if pred[:k] == y[:k] else 'upper_wrong'
                right = (c > 0 and y[k] == 1) or (c < 0 and y[k] == 0)
                a = direction[key]
                a[0] += 1; a[1] += right; a[2] += w; a[3] += w * right
    diagnostics = {}
    for key, items in groups.items():
        a = np.asarray(items, dtype=float)
        diagnostics[key] = dict(raw_count=len(items), ipw_count=int(a[:, 3].sum()) if len(items) else 0,
            c_saturation_raw=float(a[:, 0].mean()) if len(items) else None,
            c_saturation_ipw=float(np.average(a[:, 0], weights=a[:, 3])) if len(items) else None,
            gate_median=float(np.median(a[:, 1])) if len(items) else None,
            gate_times_c_median=float(np.median(a[:, 2])) if len(items) else None)
    return dict(evaluated_rows=n, errors=errors, proposed=diagnostics if groups['all_nonfirst'] else None,
        correction_direction={k: dict(raw_count=a[0], raw_correct=a[1],
            raw_accuracy=a[1]/a[0] if a[0] else None, ipw_count=a[2], ipw_correct=a[3],
            ipw_accuracy=a[3]/a[2] if a[2] else None) for k, a in direction.items()},
        scope='evaluated_rows counts every input row, including unscored rows; complete joint errors; abnormal buckets exclude normal truths; correction on reliable non-first inferred-MAP positions; raw exact medians; |c|>=.95rho; sign(c), zero incorrect')


def datasets(cfg):
    catalog_path = Path(cfg['catalog'])
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    fold = cfg['folds'][0]
    validate_fold(fold, catalog)
    sig = catalog_signature(catalog)
    if sig != cfg['catalog_signature']:
        raise ValueError('Catalog scientific identity changed.')
    training = RFShardDataset(catalog_path, fold['train'], training=True, seed=23, normal_ratio=3)
    validation = RFShardDataset(catalog_path, fold['validation'])
    root = Path(cfg['validation_selection_root'])
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    path = root/'F1.npz'
    if manifest['catalog_signature'] != sig or file_digest(path) != cfg['validation_sha256'] or file_digest(path) != manifest['folds']['F1']['sha256']:
        raise ValueError('Frozen validation changed.')
    validation.use_frozen_evaluation_selection(path)
    if len(validation) != 40002:
        raise ValueError('D2 must evaluate exactly 40,002 frozen rows.')
    ep = Path(cfg['epoch_selection_root'])
    if file_digest(ep/'manifest.json') != cfg['epoch_manifest_sha256']:
        raise ValueError('Frozen epoch manifest changed.')
    manifest = json.loads((ep/'manifest.json').read_text(encoding='utf-8'))
    if manifest['epochs'] != 10 or manifest['positive_fraction'] != .2 or manifest['normal_ratio'] != 3:
        raise ValueError('D2 requires the original fixed 20% positive subset and 3x normals.')
    for epoch in range(1, 11):
        use_epoch(training, ep, epoch, seed=23, catalog_signature=sig)
        if len(training) != 20132:
            raise ValueError('Unexpected frozen training count.')
    return training, validation


def frontend_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        # A3 registers its replaceable Head under the historical state_dict key pcc.
        if not name.startswith('pcc.'):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_one(cfg, output, head):
    validate_config(cfg)
    if head not in HEADS:
        raise ValueError('D2 only runs Proposed rho3 and Vanilla.')
    training, validation = datasets(cfg)
    random.seed(23); np.random.seed(23); torch.manual_seed(23); torch.cuda.manual_seed_all(23)
    mc = ModelConfig(**dict(cfg['model'], head_type=head))
    # D2 validates this config itself and never invokes the D1 guarded selector.
    tc = TrainingConfig(**cfg['training'])
    model = build_model(mc).to(tc.device)
    optimizer, counts = build_optimizer(model, tc)
    if not isinstance(optimizer, torch.optim.AdamW) or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('D2 requires the FP32 model and AdamW optimizer.')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    resolved = dict(cfg, head=head, model=asdict(mc), parameter_counts=counts,
        initial_frontend_sha256=frontend_digest(model), old_weights_reused=False, test_opened=False,
        formal_result=False, automatic_next_stage=False, gpu=torch.cuda.get_device_name(0))
    atomic_json(output/'resolved_config.json', resolved)
    selector = DiagnosticSelector(); history = []; updates = 0; started = time.perf_counter()
    try:
        for epoch in range(1, 11):
            sha = use_epoch(training, cfg['epoch_selection_root'], epoch, seed=23,
                            catalog_signature=cfg['catalog_signature'])
            model.train(); total = 0.; seen = 0
            progress(output/'progress.json', phase='training', epoch=epoch, records=0, total_records=len(training))
            for batch, ids in training.batches(tc.batch_size, tc.device):
                optimizer.zero_grad(set_to_none=True)
                cond = model(batch, batch.y).conditional
                loss = head_training_loss(cond, batch, head_type=head, lambda_local=tc.lambda_local,
                    positive_class_weight=None, disable_local_auxiliary=False)
                if not torch.isfinite(loss.total):
                    raise RuntimeError('Nonfinite loss.')
                loss.total.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite gradient.')
                optimizer.step(); total += float(loss.total.detach()) * len(ids); seen += len(ids); updates += 1
                if updates % 100 == 0 or seen == len(training):
                    progress(output/'progress.json', phase='training', epoch=epoch, records=seen,
                             total_records=len(training), updates_total=updates)
            predictions = output/f'epoch_{epoch:02d}_predictions.jsonl'
            metrics = evaluate_stream(model, validation, batch_size=cfg['evaluation_batch_size'], device=tc.device,
                prediction_path=predictions, progress_path=output/'progress.json', epoch=epoch)
            with predictions.open(encoding='utf-8') as handle:
                diagnostics = summarize_rows(json.loads(line) for line in handle)
            decision = selector.update(epoch, metrics['reliable'])
            record = dict(epoch=epoch, train_loss=total/seen, records=seen, updates_total=updates,
                epoch_selection_sha256=sha, validation=metrics, diagnostics=diagnostics, selection=decision,
                predictions_sha256=file_digest(predictions))
            history.append(record)
            ns = np.random.get_state()
            payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                resolved_config=resolved, epoch=epoch, updates_total=updates, epoch_selection_sha256=sha,
                selection_state=selector.state_dict(), torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(), python_rng_state=random.getstate(),
                numpy_rng_state=dict(generator=ns[0], keys=ns[1].tolist(), position=ns[2],
                                     has_gauss=ns[3], cached_gaussian=ns[4]),
                diagnostic_weights_never_reuse=True)
            torch.save(payload, output/f'epoch_{epoch:02d}.pt')
            atomic_json(output/'history.json', history)
            progress(output/'progress.json', phase='epoch_completed', epoch=epoch, records=40002, updates_total=updates)
            print(json.dumps(dict(head=head, epoch=epoch, best_epoch=selector.best['epoch'],
                positive_accuracy=metrics['reliable']['positive_joint_map_accuracy'],
                normal_fpr=metrics['reliable']['normal_joint_false_positive_rate'])), flush=True)
        result = dict(status='COMPLETED_DEVELOPMENT_DIAGNOSTIC', protocol=PROTOCOL, head=head, epochs=10,
            best_epoch=selector.best['epoch'], best=selector.best, endpoint=history[-1],
            seconds=time.perf_counter()-started, updates=updates, formal_result=False, test_opened=False,
            old_weights_reused=False, automatic_next_stage=False, normal_fpr_guard_enabled=False)
        atomic_json(output/'summary.json', result)
        return result
    except BaseException as exc:
        atomic_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), automatic_retry=False))
        raise


def main(config_path, output):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(cfg)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    results = [train_one(cfg, output/head, head) for head in HEADS]
    resolved = [json.loads((output/h/'resolved_config.json').read_text(encoding='utf-8')) for h in HEADS]
    if resolved[0]['initial_frontend_sha256'] != resolved[1]['initial_frontend_sha256']:
        raise ValueError('Fresh shared frontend initializations differ.')
    histories = [json.loads((output/h/'history.json').read_text(encoding='utf-8')) for h in HEADS]
    if [x['epoch_selection_sha256'] for x in histories[0]] != [x['epoch_selection_sha256'] for x in histories[1]]:
        raise ValueError('The two runs did not use identical epoch order.')
    atomic_json(output/'d2_result.json', dict(status='COMPLETED', protocol=PROTOCOL, results=results,
        frontend_initialization_identical=True, epoch_indices_identical=True, formal_result=False,
        outer_opened=False, automatic_next_stage=False))
