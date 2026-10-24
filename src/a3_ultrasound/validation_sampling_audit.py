"""One inference per fixed model; metadata-only repeated validation sampling."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.data.validation_sampling import select_validation_rows
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.stream_training import catalog_signature, progress, save, validate_fold
from a3_ultrasound.training import ModelConfig, TensorDataset, build_model, evaluate


def metadata(data):
    """Row positions are the full-validation dataset order, never sampled order."""
    n = len(data)
    arrays = dict(y=np.zeros((n, 6), np.int64), mask=np.zeros((n, 6), bool),
                  xy=np.zeros((n, 2)), quality=np.zeros(n, bool),
                  source_row=np.zeros(n, np.int64), ordinal=np.zeros(n, np.int64),
                  K=np.zeros(n, np.int64), entity=data.refs[:, 0].copy(),
                  refs=data.refs.copy(), entities=np.array(data.entities))
    for s, shard in enumerate(data.shards):
        positions = np.flatnonzero(data.refs[:, 0] == s)
        rows = data.refs[positions, 1]
        a = shard['arrays']; k = shard['K']
        arrays['y'][positions, :k] = a['y'][rows]
        arrays['mask'][positions, :k] = a['label_mask'][rows]
        for key in ('xy', 'quality', 'source_row'):
            arrays[key][positions] = a[key][rows]
        arrays['K'][positions] = k
        arrays['ordinal'][positions] = shard['ordinal']
    return arrays


def binary_metrics(y, p, weights):
    p = np.clip(p.astype(np.float64), 1e-7, 1-1e-7)
    # Classification precedes clipping, which does not change the .5 threshold.
    cm = np.bincount(2*y.astype(np.int64)+(p >= .5), weights=weights,
                     minlength=4).astype(np.int64).reshape(2, 2)
    n = int(cm.sum())
    scores = [2*cm[i, i]/d if (d := cm[i].sum()+cm[:, i].sum()) else 0. for i in (0, 1)]
    return dict(count=n, confusion_matrix_0_1=cm.tolist(), macro_f1=float(np.mean(scores)) if n else None,
                brier=float(np.sum(weights*(p-y)**2)/n) if n else None,
                marginal_nll=float(-np.sum(weights*(y*np.log(p)+(1-y)*np.log1p(-p)))/n) if n else None,
                false_negative_rate=float(cm[1, 0]/cm[1].sum()) if cm[1].sum() else None)


def cached_metrics(meta, prediction, indices=None, weights=None):
    ids = np.arange(len(meta['y'])) if indices is None else np.asarray(indices)
    w = np.ones(len(ids), np.int64) if weights is None else np.asarray(weights)
    if w.dtype != np.int64 or w.shape != ids.shape or np.any(w < 1):
        raise ValueError('Positive integer sampling weights required.')
    y = meta['y'][ids]; mask = meta['mask'][ids]; k = meta['K'][ids]
    p = prediction['marginals'][ids]; entity = meta['entity'][ids]
    deepest = {}; contributions = {}
    for s, name in enumerate(meta['entities'].tolist()):
        use = (entity == s) & mask[np.arange(len(ids)), k-1]
        deepest[name] = binary_metrics(y[use, k[use]-1], p[use, k[use]-1], w[use])
        parts = {}
        for label in (0, 1):
            use_labels = (entity[:, None] == s) & mask & (y == label)
            pp = np.clip(p.astype(np.float64), 1e-7, 1-1e-7)
            terms = -(y*np.log(pp)+(1-y)*np.log1p(-pp))
            # Same sample-equal reducer as the training contract, without auxiliary terms.
            denom = mask.sum(1).clip(min=1)
            parts[str(label)] = float(np.sum((terms*use_labels).sum(1)/denom*w))
        contributions[name] = parts
    complete = mask.sum(1) == k
    joint_n = int(w[complete].sum())
    per_interface = {}
    for col in range(meta['y'].shape[1]):
        use = mask[:, col]
        if use.any():per_interface[str(col+1)] = binary_metrics(y[use, col], p[use, col], w[use])
    valid = [x for x in deepest.values() if x['count']]
    return dict(evaluated_records=len(ids), population_weight=int(w.sum()),
                deepest_specimen_macro_f1=float(np.mean([x['macro_f1'] for x in valid])),
                deepest_brier=float(np.mean([x['brier'] for x in valid])),
                deepest_per_specimen=deepest, per_interface=per_interface,
                joint_nll=float(np.sum(prediction['true_nll'][ids][complete]*w[complete])/joint_n) if joint_n else None,
                joint_map_accuracy=float(np.sum(prediction['map_correct'][ids][complete]*w[complete])/joint_n) if joint_n else None,
                joint_complete_label_count=joint_n,
                hard_label_marginal_bce_contributions=contributions,
                bce_contribution_scope='sample-equal reliable-interface marginal BCE sum by entity and label; no auxiliary loss')


def infer_cache(model, data, output, *, batch_size, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    meta = metadata(data); n = len(data)
    prediction = dict(marginals=np.zeros((n, 6), np.float32), true_nll=np.zeros(n),
                      map_correct=np.zeros(n, bool))
    accumulator = PredictionAccumulator(); started = time.perf_counter(); seen = 0
    for batch, ids in data.batches(batch_size, device):
        names = [data.identity(i) for i in ids]
        small = TensorDataset(batch, tuple(x[0] for x in names), tuple(x[1] for x in names))
        _, rows = evaluate(model, small, batch_size=len(ids), device=device)
        for pos, row in zip(ids, rows):
            k = row['K']; accumulator.add(row)
            prediction['marginals'][pos, :k] = row['marginals']
            if all(row['label_mask']):
                label_index = sum(v << (k-j-1) for j, v in enumerate(row['y']))
                prediction['true_nll'][pos] = -row['log_probabilities'][label_index]
                prediction['map_correct'][pos] = row['map_states'] == row['y']
        seen += len(ids)
        if seen == n or seen % (100*batch_size) == 0:
            progress(output/'progress.json', phase='full_validation_cache', records=seen, total_records=n)
    if not np.isfinite(prediction['marginals']).all() or not np.isfinite(prediction['true_nll']).all():
        raise RuntimeError('Nonfinite cached predictions.')
    calculated = cached_metrics(meta, prediction)
    original = accumulator.result()
    for name, stats in calculated['deepest_per_specimen'].items():
        if stats['confusion_matrix_0_1'] != original['deepest_per_specimen'][name]['confusion_matrix_0_1']:
            raise RuntimeError('Cache aggregation differs from the public prediction accumulator.')
    if abs(calculated['joint_nll']-original['joint_nll']) > 1e-8:
        raise RuntimeError('Cache joint NLL differs from public accumulator.')
    np.savez_compressed(output/'metadata.npz', **meta)
    np.savez_compressed(output/'prediction.npz', **prediction)
    save(output/'metrics.json', calculated)
    save(output/'cache.json', dict(status='CACHED', records=n, seconds=time.perf_counter()-started,
                                 outer_opened=False, inference_passes=1))
    return calculated


def cache_selected(config_path, run, baseline_audit, queue_path, output):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    run = Path(run); resolved = json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    queue = json.loads(Path(queue_path).read_text(encoding='utf-8'))
    baseline = json.loads(Path(baseline_audit).read_text(encoding='utf-8'))
    checkpoint = run/'best.pt'; sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if summary['status'] != 'COMPLETED' or resolved['diagnostic'] or resolved['test_opened']:
        raise ValueError('Only completed original formal checkpoints are permitted.')
    if queue['status'] not in ('COMPLETED', 'STOPPED_AT_BOUNDARY') or any(j['status']=='RUNNING' for j in queue['jobs']):
        raise ValueError('Original training queue must be stopped.')
    jobs = [j for j in queue['jobs'] if Path(j['output']).resolve() == run.resolve()]
    if len(jobs) != 1 or jobs[0]['status'] != 'COMPLETED' or jobs[0]['checkpoint_sha256'] != sha:
        raise ValueError('Original checkpoint queue witness mismatch.')
    if baseline['checkpoint_sha256'] != sha or baseline['outer_opened'] or baseline['checkpoint_reselected']:
        raise ValueError('Full-audit identity mismatch.')
    catalog = json.loads(Path(cfg['catalog']).read_text(encoding='utf-8'))
    fold = next(f for f in cfg['folds'] if f['fold'] == resolved['fold']['fold'])
    validate_fold(fold, catalog)
    if fold != resolved['fold'] or catalog_signature(catalog) != resolved['catalog_signature']:
        raise ValueError('Scientific inputs differ from the fixed checkpoint.')
    c = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if c['resolved_config'] != resolved or c['epoch'] != summary['best_epoch']:
        raise ValueError('Selected epoch or resolved configuration mismatch.')
    model = build_model(ModelConfig(**resolved['model'])).to(cfg['training']['device'])
    model.load_state_dict(c['model_state_dict'])
    data = RFShardDataset(cfg['catalog'], fold['validation'])
    result = infer_cache(model, data, output, batch_size=cfg['evaluation_batch_size'], device=cfg['training']['device'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != sha:
        raise RuntimeError('Original selected checkpoint changed.')
    old = baseline['metrics']['reliable']
    if result['evaluated_records'] != baseline['evaluated_records']:
        raise RuntimeError('Full population count mismatch.')
    for entity, stats in result['deepest_per_specimen'].items():
        if stats['confusion_matrix_0_1'] != old['deepest_per_specimen'][entity]['confusion_matrix_0_1']:
            raise RuntimeError('Original full-audit confusion matrix mismatch.')
    if abs(result['deepest_specimen_macro_f1']-old['deepest_specimen_macro_f1']) > 1e-8:
        raise RuntimeError('Original full-audit F1 mismatch.')
    deltas = dict(joint_nll=result['joint_nll']-old['joint_nll'],
                  brier={e:s['brier']-old['deepest_per_specimen'][e]['brier'] for e,s in result['deepest_per_specimen'].items()})
    save(Path(output)/'reconciliation.json', dict(status='PASS', checkpoint_sha256=sha,
         counts_and_confusions_equal=True, aggregate_probability_metric_deltas=deltas,
         probability_warning=abs(deltas['joint_nll'])>1e-6 or any(abs(v)>1e-6 for v in deltas['brier'].values()),
         scope='aggregate reconciliation; historical audit has no row-level probability file'))
    return result


def sample_panel(meta, *, target, seed):
    ids = []; weights = []
    for s, _ in enumerate(meta['entities']):
        positions = np.flatnonzero(meta['entity'] == s); k = int(meta['K'][positions[0]])
        rows, w = select_validation_rows(meta['y'][positions, :k], meta['mask'][positions, :k],
                    meta['xy'][positions], meta['quality'][positions], target=target,
                    seed=seed, ordinal=int(meta['ordinal'][positions[0]]))
        ids.append(positions[rows]); weights.append(w)
    ids = np.concatenate(ids); weights = np.concatenate(weights)
    positive = (meta['mask'] & (meta['y'] == 1)).any(1)
    if (weights.sum() != len(meta['y']) or not np.isin(np.flatnonzero(positive), ids).all()
            or np.any(weights[positive[ids]] != 1)):
        raise RuntimeError('Sampling population or positive-row invariant failed.')
    return ids, weights


def sampling_design(meta, target):
    """Persist seed-independent strata/bins; selection still calls the original sampler."""
    arrays={};reports=[]
    for s,name in enumerate(meta['entities'].tolist()):
        positions=np.flatnonzero(meta['entity']==s);k=int(meta['K'][positions[0]])
        y=meta['y'][positions,:k];mask=meta['mask'][positions,:k]
        positive=(mask & (y==1)).any(1);negative=np.flatnonzero(~positive)
        codes=(mask.astype(np.int64)*(1<<np.arange(k))).sum(1)*2+meta['quality'][positions]
        groups=[negative[codes[negative]==c] for c in np.unique(codes[negative])]
        sizes=np.array([len(g) for g in groups],np.int64)
        if len(positions)<=target:quotas=sizes.copy()
        else:
            budget=min(len(negative),max(len(groups),target-int(positive.sum())))
            quotas=np.ones(len(groups),np.int64);remaining=budget-len(groups)
            if remaining:
                ideal=remaining*(sizes-1)/(sizes-1).sum();extra=np.floor(ideal).astype(np.int64)
                quotas+=extra;left=remaining-int(extra.sum())
                if left:quotas[np.argsort(-(ideal-extra),kind='stable')[:left]]+=1
        strata=[]
        for group,quota in zip(groups,quotas):
            xy=meta['xy'][positions[group]];order=group[np.lexsort((xy[:,0],xy[:,1]))]
            code=int(codes[group[0]]);prefix=f'e{s}_stratum{code}'
            edges=np.linspace(0,len(order),int(quota)+1,dtype=np.int64)
            arrays[prefix+'_ordered_indices']=positions[order]
            arrays[prefix+'_edges']=edges
            arrays[prefix+'_bin_sizes']=np.diff(edges)
            strata.append(dict(code=code,population=len(group),quota=int(quota),array_prefix=prefix))
        reports.append(dict(entity=name,ordinal=int(meta['ordinal'][positions[0]]),population=len(positions),
                            positive_count=int(positive.sum()),strata=strata))
    return arrays,reports


def validate_panel_bins(meta, ids, weights, design_arrays, reports):
    for report in reports:
        for stratum in report['strata']:
            prefix=stratum['array_prefix'];ordered=design_arrays[prefix+'_ordered_indices']
            inverse=np.full(len(meta['y']),-1,np.int64);inverse[ordered]=np.arange(len(ordered))
            locations=inverse[ids];use=locations>=0;edges=design_arrays[prefix+'_edges']
            bins=np.searchsorted(edges[1:],locations[use],side='right')
            if (not np.all(np.bincount(bins,minlength=len(edges)-1)==1)
                    or not np.array_equal(weights[use],np.diff(edges)[bins])):
                raise RuntimeError('Persisted stratum/bin design differs from original sampler output.')


def resample(cache_dirs, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    caches = {}; reference = None
    for head, folder in cache_dirs.items():
        folder = Path(folder)
        if json.loads((folder/'reconciliation.json').read_text(encoding='utf-8'))['status'] != 'PASS':
            raise ValueError('Every cache must reconcile before resampling.')
        with np.load(folder/'metadata.npz', allow_pickle=False) as z: meta = dict(z)
        with np.load(folder/'prediction.npz', allow_pickle=False) as z: prediction = dict(z)
        if reference is not None and any(not np.array_equal(reference[k], meta[k]) for k in reference):
            raise ValueError('Four-head cache row identity/metadata mismatch.')
        reference = meta; caches[head] = prediction
    full = {h:cached_metrics(reference, p) for h,p in caches.items()}
    results = []
    for target in (13334, 26668, 53336):
        design,report=sampling_design(reference,target)
        np.savez_compressed(output/f'design_{target}.npz',**design)
        save(output/f'design_{target}.json',dict(target_per_entity=target,entities=report,
            scope='seed-independent spatial sort, stratum quotas, bin edges and sizes in companion NPZ'))
        for seed in range(2026102301, 2026102321):
            ids, w = sample_panel(reference, target=target, seed=seed)
            validate_panel_bins(reference,ids,w,design,report)
            np.savez_compressed(output/f'panel_{target}_{seed}.npz', indices=ids, weights=w,
                                refs=reference['refs'][ids])
            metrics = {h:cached_metrics(reference, p, ids, w) for h,p in caches.items()}
            errors = {h:metrics[h]['deepest_specimen_macro_f1']-full[h]['deepest_specimen_macro_f1'] for h in caches}
            pv = metrics['proposed']['deepest_specimen_macro_f1']-metrics['vanilla_chain']['deepest_specimen_macro_f1']
            pv_full = full['proposed']['deepest_specimen_macro_f1']-full['vanilla_chain']['deepest_specimen_macro_f1']
            results.append(dict(target_per_entity=target, seed=seed, actual_rows=len(ids),
                    metrics=metrics, f1_errors=errors, paired_p_minus_v_error=pv-pv_full,
                    paired_sign_changed=bool(np.sign(pv)!=np.sign(pv_full))))
        save(output/'progress.json', dict(target_complete=target, panels_complete=len(results)))
    summary = []
    for target in (13334, 26668, 53336):
        group = [r for r in results if r['target_per_entity']==target]
        for name in [*caches, 'paired_p_minus_v']:
            values = np.abs([r['paired_p_minus_v_error'] if name=='paired_p_minus_v' else r['f1_errors'][name] for r in group])
            summary.append(dict(target_per_entity=target, metric=name, p50=float(np.quantile(values,.5)),
                     empirical_p95=float(np.quantile(values,.95)), maximum=float(values.max()),
                     exceeding_005=int((values>.005).sum()), replicates=20))
    save(output/'summary.json', dict(status='COMPLETED', full=full, empirical_errors=summary,
         repeats_are_not_scientific_replications=True, checkpoint_ranking_stability_not_tested=True,
         outer_opened=False, performance_claims_released=False))
    save(output/'panels.json', results)
    return summary
