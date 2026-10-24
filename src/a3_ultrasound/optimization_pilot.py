"""Fixed-update C/L/LW development pilot, separate from the formal runner."""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import time

import numpy as np
import torch

from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.stream_training import catalog_signature, evaluate_stream, progress, save, validate_fold
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, build_optimizer, _validate_training_config
from a3_ultrasound.validation_sampling_audit import infer_cache
from a3_ultrasound.runtime_supervisor import all_finite


def learning_rate(arm, update):
    """One-based next optimizer update; depends on global count, so resumes agree."""
    if arm not in ('C', 'L', 'LW') or type(update) is not int or update < 1:
        raise ValueError('Registered arm and positive one-based update required.')
    if arm == 'C':return .001
    target = .001/3
    return target*(.1+.9*min(update-1, 9773)/9773) if arm == 'LW' else target


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initial_conditions_match(current, reference):
    """Compare the persisted contract; JSON serializes optimizer tuples as lists."""
    return json.loads(json.dumps(current, allow_nan=False)) == reference


def capture_rng(generator):
    ns = np.random.get_state()
    return dict(python=random.getstate(), numpy=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                training_order=generator.get_state())


def bn_counters(model):
    return {name:int(module.num_batches_tracked.item()) for name,module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.num_batches_tracked is not None}


def assert_finite_model(model):
    if all_finite(model.state_dict().values()):return
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError('Nonfinite model state: '+name)


def train_pilot(config_path, output, *, arm):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if not cfg.get('development_batch_permitted') or cfg.get('outer_evaluation_permitted'):
        raise ValueError('Requires authorized development batch with outer closed.')
    fold = next(f for f in cfg['folds'] if f['fold']=='F1')
    catalog = json.loads(Path(cfg['catalog']).read_text(encoding='utf-8')); validate_fold(fold, catalog)
    tc = TrainingConfig(**dict(cfg['training'], seed=23, epochs=10, learning_rate=learning_rate(arm,1)))
    _validate_training_config(tc)
    if tc.batch_size != 16 or tc.scheduler_step_size is not None:
        raise ValueError('This pilot fixes batch16 and uses only the explicit update schedule.')
    random.seed(23); np.random.seed(23); torch.manual_seed(23)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(23)
    torch.set_num_threads(2)
    training = RFShardDataset(cfg['catalog'], fold['train'], training=True, seed=23,
                              normal_ratio=cfg['normal_to_positive'])
    frozen = np.load(Path(cfg['frozen_selection_root'])/'F1_s23.npy', allow_pickle=False)
    if not np.array_equal(training.refs, frozen) or len(training)!=52120 or math.ceil(len(training)/16)!=3258:
        raise ValueError('Frozen training selection or 3258 updates/epoch mismatch.')
    validation = RFShardDataset(cfg['catalog'], fold['validation'])
    compact = RFShardDataset(cfg['catalog'], fold['validation'])
    compact.use_frozen_evaluation_selection(Path(cfg['validation_selection_root'])/'F1.npz')
    mc = ModelConfig(**dict(cfg['model'], head_type='vanilla_chain'))
    model = build_model(mc).to(tc.device); optimizer, counts = build_optimizer(model,tc)
    generator = torch.Generator().manual_seed(23)
    order_probe = torch.Generator(); order_probe.set_state(generator.get_state())
    first_order = torch.randperm(len(training), generator=order_probe).numpy()
    initial = dict(model_state=tensor_state_digest(model.state_dict()),
                   optimizer_empty_state=not bool(optimizer.state),
                   optimizer_groups=[{k:v for k,v in g.items() if k not in ('params','lr','initial_lr')} for g in optimizer.param_groups],
                   training_refs=hashlib.sha256(training.refs.tobytes()).hexdigest(),
                   first_epoch_order=hashlib.sha256(first_order.tobytes()).hexdigest(),
                   first_batch_refs=training.refs[first_order[:16]].tolist())
    if arm != 'C':
        reference=json.loads(Path(cfg['pilot_initial_reference']).read_text(encoding='utf-8'))['initial']
        if not initial_conditions_match(initial,reference):
            raise RuntimeError('Initial weights, optimizer moments/settings or training order differ across arms.')
    # LR differs by design; the optimizer's empty moments and other settings must match.
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    resolved = dict(arm=arm, seed=23, fold=fold, model=asdict(mc), training=asdict(tc),
        catalog_signature=catalog_signature(catalog), catalog=cfg['catalog'], parameter_counts=counts,
        initial=initial, early_stopping_disabled=True, checkpoint_policy='every_epoch_fixed_update',
        expected_updates=32580, warmup_updates=9774 if arm=='LW' else 0,
        lr_policy='before optimizer update, one-based global counter',
        bn_policy='original training updates; stored eval statistics; no recalibration',
        endpoint_policy='reload epoch_10.pt into a fresh model; full validation; no best checkpoint',
        compact_policy='descriptive only; does not control stopping or selection',
        scope='F1/seed23 development pilot; no significance or H2 acceptance', test_opened=False)
    save(output/'resolved_config.json',resolved)
    np.save(output/'training_selection.npy',training.refs)
    history=[]; updates=0; started=time.perf_counter()
    try:
        with (output/'updates.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file); writer.writerow(['update','epoch','lr','loss','gradient_l2'])
            for epoch in range(1,11):
                model.train(); total=0.; seen=0; epoch_norms=[]
                order_before=generator.get_state().clone()
                probe=torch.Generator(); probe.set_state(order_before)
                order_sha=hashlib.sha256(torch.randperm(len(training),generator=probe).numpy().tobytes()).hexdigest()
                for batch, ids in training.batches(16,tc.device,generator=generator):
                    lr=learning_rate(arm,updates+1)
                    for group in optimizer.param_groups:group['lr']=lr
                    optimizer.zero_grad(set_to_none=True); out=model(batch,batch.y)
                    loss=head_training_loss(out.conditional,batch,head_type='vanilla_chain',
                        lambda_local=tc.lambda_local,positive_class_weight=tc.positive_class_weight,
                        disable_local_auxiliary=tc.disable_local_auxiliary)
                    if not torch.isfinite(loss.total):raise RuntimeError('Nonfinite loss.')
                    loss.total.backward()
                    gradients=[p.grad.detach() for p in model.parameters() if p.grad is not None]
                    if not all_finite(gradients):raise RuntimeError('Nonfinite gradient.')
                    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm.')
                    optimizer.step(); assert_finite_model(model)
                    updates+=1; value=float(loss.total.detach()); grad=float(norm)
                    total+=value*len(ids); seen+=len(ids); epoch_norms.append(grad)
                    writer.writerow([updates,epoch,lr,value,grad])
                    if updates%100==0 or seen==len(training):
                        file.flush(); progress(output/'progress.json',phase='pilot_training',arm=arm,epoch=epoch,
                            records=seen,total_records=len(training),updates_total=updates,lr=lr)
                if updates!=epoch*3258:raise RuntimeError('Fixed update count changed.')
                checkpoint=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    resolved_config=resolved,epoch=epoch,updates_total=updates,lr=lr,
                    next_lr=learning_rate(arm,updates+1),rng_state=capture_rng(generator),
                    epoch_order_rng_before=order_before,epoch_order_sha256=order_sha,bn_counters=bn_counters(model))
                torch.save(checkpoint,output/f'epoch_{epoch:02d}.pt')
                metrics=evaluate_stream(model,compact,batch_size=cfg['evaluation_batch_size'],device=tc.device,
                    progress_path=output/'progress.json',phase='descriptive_compact_validation',epoch=epoch)
                record=dict(epoch=epoch,updates_total=updates,train_loss=total/seen,
                    epoch_order_sha256=order_sha,
                    lr_first=learning_rate(arm,(epoch-1)*3258+1),lr_last=lr,
                    gradient_l2_mean=float(np.mean(epoch_norms)),gradient_l2_max=max(epoch_norms),
                    bn_counters=bn_counters(model),compact_validation=metrics)
                history.append(record); save(output/'history.json',history)
        if updates!=32580:raise RuntimeError('Incomplete fixed endpoint.')
        endpoint=torch.load(output/'epoch_10.pt',map_location=tc.device,weights_only=True)
        if endpoint['updates_total']!=32580:raise RuntimeError('Wrong endpoint checkpoint.')
        fresh=build_model(mc).to(tc.device); fresh.load_state_dict(endpoint['model_state_dict'])
        del model,optimizer
        full=infer_cache(fresh,validation,output/'endpoint_validation',
                         batch_size=cfg['evaluation_batch_size'],device=tc.device)
        result=dict(status='COMPLETED',arm=arm,updates=updates,epochs=10,endpoint_epoch=10,
            initial=initial,metrics=full,seconds=time.perf_counter()-started,outer_opened=False,
            performance_claims_released=False,best_checkpoint_used=False)
        save(output/'summary.json',result); progress(output/'progress.json',phase='completed',updates_total=updates)
        return result
    except Exception as exc:
        save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc))); raise


def promotion_guard(candidate, baseline):
    a=candidate['metrics']; b=baseline['metrics']
    fnr_deltas={e:a['deepest_per_specimen'][e]['false_negative_rate']-b['deepest_per_specimen'][e]['false_negative_rate']
                for e in b['deepest_per_specimen']}
    differences=dict(f1=a['deepest_specimen_macro_f1']-b['deepest_specimen_macro_f1'],
                     joint_nll=a['joint_nll']-b['joint_nll'],brier=a['deepest_brier']-b['deepest_brier'],fnr=fnr_deltas)
    passed=differences['f1']>=.005 and differences['joint_nll']<=0 and differences['brier']<=0 and max(fnr_deltas.values())<=.01
    return dict(passed=bool(passed),deltas=differences,scope='development selection guard, no statistical conclusion')
