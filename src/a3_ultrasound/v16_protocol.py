"""R3 development validation. Launch gates remain closed until D1/D2/D3 and review."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from copy import deepcopy

HEADS=('local_independent','local_pre_independent','vanilla_chain')
TIMEBASE_SHA256='446ef99e74faa6ed7ef7dc2ef30667e988c8fdc7f1a840d1d50539c17285e487'
TIMEBASE_VERSION='v1.6-frozen-input-20261105'
M3_CHANNELS=('residual','free_deep')
M3_RUNTIME_KEYS=('m3_fit_manifest','m3_fit_manifest_sha256','m3_d2_decision','m3_d2_decision_sha256')
ROOT=Path(__file__).resolve().parents[2]
_SHA=re.compile(r'^[0-9a-f]{64}$')


def template():
    # Versioned configuration is reviewable without private data artifacts.
    return json.loads((ROOT/'configs/experiment/v1_6_development.json').read_text(encoding='utf-8'))


def is_v16(cfg):
    return cfg.get('model',{}).get('architecture_version')=='a3-v1.6'


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
    if channels==2 and head==HEADS[0]:
        raise ValueError('E4/E5 permit dual channels only for LP and V.')
    # The M3 fit manifest and the pre-training D2 decision are runtime artefacts. They
    # are mandatory for dual-channel cells and forbidden for single-channel cells.
    for key in M3_RUNTIME_KEYS:
        value=candidate.get(key)
        if channels==2:
            if not isinstance(value,str) or not value or (key.endswith('_sha256') and not _SHA.match(value)):
                raise ValueError('Dual-channel cells require a verified M3 fit manifest and D2 decision: '+key)
        elif value is not None:
            raise ValueError('Single-channel cells must not reference M3 artefacts: '+key)
        candidate[key]=expected.get(key)
    candidate['model']['input_channels']=1;candidate['m3_channel']='none'
    if candidate!=expected:raise ValueError('Configuration differs from registered V1.6 development contract.')
    allowed=[23,41,59] if head==HEADS[0] or channels==2 else [23,41,59,77,97]
    if head not in HEADS or seed not in allowed or fold_name not in [f['fold'] for f in cfg['folds']]:
        raise ValueError('Unregistered matrix cell.')
    if require_permission:
        raise ValueError('V1.6 development only: real launch awaits D1/D2/D3, reviewed runtime protocol and explicit authorization.')


def _read_verified(path, expected_sha256, name):
    raw=Path(path).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=expected_sha256:
        raise ValueError(f'{name} artifact changed (sha256 {digest}).')
    return json.loads(raw.decode('utf-8')),digest


def load_v16_timebase(cfg):
    """Return the frozen v1.6 centre table in the shape RFShardDataset expects.

    The v1.5 loader only accepts v1.5.2/v1.5.3 tables, so without this the
    board_timebase window policy has no artefact and dataset construction fails.
    """
    if cfg.get('timebase_calibration_sha256')!=TIMEBASE_SHA256:
        raise ValueError('V1.6 must use the frozen 2026-11-05 timebase.')
    value,digest=_read_verified(cfg['timebase_calibration_path'],TIMEBASE_SHA256,'V1.6 timebase')
    rows=value.get('rows',[])
    if (value.get('version')!=TIMEBASE_VERSION or value.get('status')!='FROZEN_V1_6_INPUT'
            or len(rows)!=13 or len({r.get('board') for r in rows})!=13):
        raise ValueError('Invalid frozen V1.6 timebase table.')
    for row in rows:
        tau=row.get('selected_tau');k=row.get('K')
        if (k not in (2,3) or not isinstance(tau,list) or len(tau)!=k
                or any(b<=a for a,b in zip(tau,tau[1:])) or not row.get('selected_bottom',0)>tau[-1]):
            raise ValueError('Invalid frozen centres for '+str(row.get('board')))
    value['_artifact_sha256']=digest
    return value


def verify_m6_inactive(cfg):
    """v1.5.3 decided M6 off (front-surface attention median 0.0118 < 0.5); v1.6 keeps that."""
    decision,_=_read_verified(cfg['m6_decision_path'],cfg['m6_decision_sha256'],'M6 decision')
    if (decision.get('enable_m6') is not False or decision.get('pre_exclude_front_end') is not None
            or cfg['model'].get('pre_exclude_front_end') is not None):
        raise ValueError('V1.6 inherits the v1.5.3 M6 decision: Pre front-surface exclusion stays off.')
    return decision


def load_m3_channel(cfg, fold_name):
    """Verify the per-fold M3 fit manifest and the D2 decision; return dataset config.

    Returns None for single-channel cells. The second channel is computed on the
    fly from each raw record with the fold's TRAIN-fitted parameters (no labels),
    so the manifest holds parameters, not multi-GB caches. Parameters must come
    from this fold's TRAIN boards and the frozen v1.6 centres, no family may be
    BOUNDARY_LIMITED, and D2 must have chosen exactly the configured channel.
    """
    channel=cfg.get('m3_channel','none')
    if channel=='none':return None
    if channel not in M3_CHANNELS or cfg['model'].get('input_channels')!=2:
        raise ValueError('Unknown or inconsistent M3 channel.')
    decision,_=_read_verified(cfg['m3_d2_decision'],cfg['m3_d2_decision_sha256'],'D2 decision')
    if decision.get('status')!='DECIDED' or decision.get('channel')!=channel:
        raise ValueError('D2 did not select the configured M3 channel; dual-channel cells are not permitted.')
    manifest,digest=_read_verified(cfg['m3_fit_manifest'],cfg['m3_fit_manifest_sha256'],'M3 fit manifest')
    if decision.get('fit_manifest_sha256',{}).get(fold_name)!=digest:
        raise ValueError('D2 decision was not made with this fold\'s M3 fit manifest.')
    fold=next(f for f in cfg['folds'] if f['fold']==fold_name)
    if (manifest.get('status')!='FITTED' or manifest.get('fold')!=fold_name
            or manifest.get('train_entities')!=fold['train']
            or manifest.get('timebase_sha256')!=TIMEBASE_SHA256):
        raise ValueError('M3 parameters were not fitted for this fold, TRAIN panel and frozen timebase.')
    params=manifest.get('family_params',{})
    if any(p.get('status')=='BOUNDARY_LIMITED' for p in params.values()):
        raise ValueError('BOUNDARY_LIMITED: M3 stage is paused.')
    entity_family=manifest.get('entity_family',{})
    return dict(channel=channel,fold=fold_name,manifest_sha256=digest,
                params_by_entity={e:params[f] for e,f in entity_family.items()})


def runtime_record(cfg, fold_name, output):
    """Copy the exact v1.6 input artefacts beside a run and return their identities."""
    from a3_ultrasound.runtime_supervisor import atomic_write
    items=[('timebase_calibration','timebase_calibration.json',TIMEBASE_SHA256),
           ('m6_decision','m6_decision.json',cfg['m6_decision_sha256']),
           ('speed_decision','speed_decision.json',cfg['speed_decision_sha256'])]
    if cfg.get('m3_channel','none')!='none':
        items+=[('m3_fit_manifest','m3_fit_manifest.json',cfg['m3_fit_manifest_sha256']),
                ('m3_d2_decision','m3_d2_decision.json',cfg['m3_d2_decision_sha256'])]
    record={}
    for key,name,expected in items:
        path=cfg[key+'_path'] if key+'_path' in cfg else cfg[key]
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(key+' artifact changed.')
        atomic_write(Path(output)/name,lambda handle,raw=raw:handle.write(raw))
        record[key+'_sha256']=expected
    m3=load_m3_channel(cfg,fold_name)
    record.update(timebase_version=TIMEBASE_VERSION,window_jitter=cfg['window_jitter'],
        window_jitter_identity='seed,epoch,catalog ordinal,source_row; common to all Heads; evaluation shift0',
        m6='inherited v1.5.3 decision: off',input_channels=cfg['model']['input_channels'],
        m3_channel=cfg.get('m3_channel','none'),
        m3_params_by_entity=None if m3 is None else m3['params_by_entity'])
    return record


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
