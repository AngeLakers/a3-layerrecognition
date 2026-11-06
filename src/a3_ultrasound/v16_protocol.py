"""R3 development validation. Launch gates remain closed until D1/D2/D3 and review."""
from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy

HEADS=('local_independent','local_pre_independent','vanilla_chain')
TIMEBASE_SHA256='446ef99e74faa6ed7ef7dc2ef30667e988c8fdc7f1a840d1d50539c17285e487'
ROOT=Path(__file__).resolve().parents[2]


def template():
    # Versioned configuration is reviewable without private data artifacts.
    return json.loads((ROOT/'configs/experiment/v1_6_development.json').read_text(encoding='utf-8'))


def validate_config(cfg, *, fold_name, seed, head, require_permission=True):
    expected=template()
    runtime_keys=('catalog','epoch_selection_root','validation_selection_root','source_commit')
    candidate=deepcopy(cfg)
    for key in runtime_keys:
        if not isinstance(candidate.get(key),str) or not candidate[key]:raise ValueError('Missing runtime path or identity: '+key)
        candidate[key]=expected[key]
    channels=candidate['model'].get('input_channels'); channel=candidate.get('m3_channel')
    if (type(channels) is not int or (channels,channel) not in ((1,'none'),(2,'residual'),(2,'free_deep'))):
        raise ValueError('Inconsistent input channel declaration.')
    candidate['model']['input_channels']=1;candidate['m3_channel']='none'
    if channels==2 and head==HEADS[0]:
        raise ValueError('E4/E5 permit dual channels only for LP and V.')
    if candidate!=expected:raise ValueError('Configuration differs from registered V1.6 development contract.')
    allowed=[23,41,59] if head==HEADS[0] or channels==2 else [23,41,59,77,97]
    if head not in HEADS or seed not in allowed or fold_name not in [f['fold'] for f in cfg['folds']]:
        raise ValueError('Unregistered matrix cell.')
    if require_permission:
        raise ValueError('V1.6 development only: real launch awaits D1/D2/D3, reviewed runtime protocol and explicit authorization.')


def matrix(m3_channel='none'):
    if m3_channel not in ('none','residual','free_deep'):raise ValueError('Unknown M3 channel.')
    cells=[]
    for fold in range(1,6):
        for seed in (23,41,59,77,97):
            for head in HEADS:
                if head==HEADS[0] and seed in (77,97):continue
                cells.append(dict(fold=f'F{fold}',seed=seed,head=head,m3_channel='none'))
            if m3_channel!='none' and seed in (23,41,59):
                cells.extend(dict(fold=f'F{fold}',seed=seed,head=h,m3_channel=m3_channel) for h in HEADS[1:])
    return cells
