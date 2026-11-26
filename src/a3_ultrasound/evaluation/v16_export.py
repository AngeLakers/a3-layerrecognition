"""Turn finished v1.6 run directories into the calibration-stage JSONL.

Each run must be a completed guarded selection on its fold's validation panel.
Only complete reliable labels are exported (temperature fitting needs the true
joint state); partial-label rows stay in the run's own metrics.
"""
from __future__ import annotations

import json
from pathlib import Path


def export_rows(run_dirs, *, head, seed):
    rows=[];seen=set()
    for run in map(Path,run_dirs):
        resolved=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
        summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        fold=resolved['fold']['fold']
        if resolved['model']['head_type']!=head or resolved['seed']!=seed:
            raise ValueError(f'{run} is not {head}/seed{seed}.')
        if resolved.get('architecture_version')!='a3-v1.6' or resolved.get('diagnostic'):
            raise ValueError(f'{run} is not a v1.6 validation run.')
        if summary.get('status')!='COMPLETED' or summary.get('guard_violated') is not False \
                or summary.get('selected_checkpoint')!='best_guarded.pt':
            raise ValueError(f'{run} has no guarded checkpoint (FPR guard violated or incomplete).')
        if fold in seen:raise ValueError('Two runs for fold '+fold)
        seen.add(fold);validation=set(resolved['fold']['validation'])
        with open(run/'predictions.jsonl',encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line)
                if row['specimen_id'] not in validation:raise ValueError('Non-validation row in '+str(run))
                if not all(row['label_mask']):continue
                rows.append(dict(split='validation',fold=fold,head=head,seed=seed,
                    checkpoint_kind='guarded',guard_violated=False,sample_id=row['sample_id'],
                    specimen_id=row['specimen_id'],y=row['y'],label_mask=[bool(v) for v in row['label_mask']],
                    log_probabilities=row['log_probabilities'],sample_weight=row.get('sample_weight',1)))
    return rows
