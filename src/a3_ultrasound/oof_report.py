"""Prespecified aggregation of complete, separately evaluated outer runs."""
from __future__ import annotations
import numpy as np


def summarize_oof(config, results, *, expected_catalog_signature):
    if config.get('expected_runs')==60:
        from a3_ultrasound.phase60 import phase_cells
        phase_cells(config)
    expected={(f['fold'],s,h) for f in config['folds'] for s in config['seeds'] for h in config['heads']}
    registry={};signatures=set()
    for result in results:
        if result.get('status')!='EVALUATED':raise ValueError('Only outer EVALUATED results are accepted.')
        if config.get('protocol_signature') and result.get('protocol_signature')!=config['protocol_signature']:
            raise ValueError('Outer runs use a different training protocol.')
        key=(result['fold'],result['seed'],result['head'])
        if key not in expected or key in registry:raise ValueError('Unexpected or duplicate outer run.')
        signatures.add(result['catalog_signature']);registry[key]=result
    if set(registry)!=expected:raise ValueError(f'Incomplete matrix: {len(registry)}/{len(expected)}.')
    if len(signatures)!=1:raise ValueError('Outer runs use different data/input versions.')
    if signatures!={expected_catalog_signature}:raise ValueError('Outer results differ from active catalog.')
    entities=sorted({e for f in config['folds'] for e in f['test']})
    if sum(len(f['test']) for f in config['folds'])!=len(entities):raise ValueError('Repeated outer test entity.')
    scores={};calibration={}
    for head in config['heads']:
        scores[head]=[];calibration[head]=[]
        for seed in config['seeds']:
            per={};nll_sum=0.;nll_n=0
            for fold in config['folds']:
                r=registry[(fold['fold'],seed,head)]['metrics']['reliable']
                p=r['deepest_per_specimen']
                if set(p)!=set(fold['test']):raise ValueError('Missing or extra reliable deepest entity.')
                per.update(p);n=r['joint_complete_label_count']
                if n and r['joint_nll'] is None:raise ValueError('Missing complete-label joint NLL.')
                nll_sum+=(r['joint_nll'] or 0.)*n;nll_n+=n
            if not nll_n:raise ValueError('No complete-label outer cases.')
            scores[head].append([per[e]['macro_f1'] for e in entities])
            calibration[head].append({'joint_nll':nll_sum/nll_n,'deepest_brier':float(np.mean([per[e]['brier'] for e in entities]))})
    arrays={h:np.asarray(v,dtype=float) for h,v in scores.items()}
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite F1.')
    settings=config['statistics'];a,b=settings['primary_pair'];delta=arrays[a]-arrays[b]
    seed_diff=delta.mean(1);entity_diff=delta.mean(0)
    rng=np.random.default_rng(settings['bootstrap_seed'])
    indices=rng.integers(0,len(entities),size=(settings['bootstrap_replicates'],len(entities)))
    ci=np.quantile(entity_diff[indices].mean(1),[.025,.975]).tolist()
    guards={k:float(np.mean([r[k] for r in calibration[a]])-np.mean([r[k] for r in calibration[b]])) for k in ['joint_nll','deepest_brier']}
    f1_pass=bool(len(entities)>=5 and len(config['seeds'])>=5 and
                 (seed_diff>0).sum()>=settings['paired_positive_seeds_required'] and ci[0]>0)
    calibration_pass=all(guards[k]<=settings['calibration_guard_max_mean_delta_'+k] for k in guards)
    return {'status':'COMPLETE_OOF_SUMMARY','entities':entities,'seeds':config['seeds'],'heads':config['heads'],
        'deepest_f1_by_seed_entity':scores,'seed_macro_f1':{h:v.mean(1).tolist() for h,v in arrays.items()},
        'paired_difference_by_seed':seed_diff.tolist(),'paired_difference_mean':float(seed_diff.mean()),
        'entity_cluster_bootstrap_95ci':ci,'positive_seed_count':int((seed_diff>0).sum()),
        'calibration':calibration,'calibration_mean_deltas':guards,'f1_criterion_met':f1_pass,
        'no_degradation_guard_met':calibration_pass,'numerical_H2_criteria_met':f1_pass and calibration_pass,
        'claim_released':False,'interpretation':'Development OOF specimen contrasts; overlapping training folds. Not external/blind validation or full retraining uncertainty. Independent result audit and result-to-claim remain required.'}
