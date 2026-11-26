"""Per-fold M3 parameter fit (spec C3): TRAIN boards only, frozen v1.6 centres.

One manifest per fold holds the fitted scalars (f0, bandwidth, alpha, delta)
for each board family plus the family of every catalog entity. The transform
itself runs on the fly in the data pipeline, so no RF-sized cache is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from a3_ultrasound.data.m3_transform import fit_family, sample_train_records

LAMBDA_GRID=(1e-06,1e-04,1e-02,1.0,100.0)   # RULES_T2_R3 fit.lambda_grid
FAMILIES=('TC4-K2','TC4-K3','Al-K2')


def family_of(entry):
    k=int(entry['K'])
    if entry['group']=='AL':
        if k!=2:raise ValueError('Aluminium boards are registered as K=2 only.')
        return 'Al-K2'
    if k not in (2,3):raise ValueError('TC4 boards must be K=2 or K=3.')
    return f'TC4-K{k}'


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def load_train_records(catalog_path, entities, timebase_rows):
    """Eligible complete-label TRAIN rows; RF is not read here (x = row position)."""
    catalog_path=Path(catalog_path);catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    registry={e['entity']:e for e in catalog['entities']};records={};handles={}
    for entity in entities:
        entry=registry[entity];base=catalog_path.parent/entry['index_dir']
        arrays={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in
                ('source_row','offsets','lengths','y','label_mask','eligible')}
        ids=np.flatnonzero(np.asarray(arrays['eligible'])&np.asarray(arrays['label_mask']).all(1))
        records[entity]=dict(ordinal=int(entry['ordinal']),y=np.asarray(arrays['y'][ids]).astype(np.int64),
            row_ids=np.asarray(arrays['source_row'][ids]).astype(np.int64),x=ids,
            centers=[float(t) for t in timebase_rows[entity]['selected_tau']],family=family_of(entry))
        handles[entity]=dict(entry=entry,arrays=arrays,rf=np.load(entry['rf_path'],mmap_mode='r'))
    return records,handles


def load_rf(handle, positions):
    """Stored records zero-padded to the longest selected one; lengths kept."""
    a=handle['arrays'];offsets=np.asarray(a['offsets'][positions]).astype(np.int64)
    lengths=np.asarray(a['lengths'][positions]).astype(np.int64)
    x=np.zeros((len(positions),int(lengths.max()) if len(positions) else 0),np.float32)
    for j,(o,n) in enumerate(zip(offsets,lengths)):x[j,:n]=handle['rf'][o:o+n]
    return x,lengths


def fit_fold(cfg, fold_name, timebase, output_dir, *, lambda_grid=LAMBDA_GRID):
    """Fit all three families for one fold; returns (manifest, manifest_path)."""
    from a3_ultrasound.v16_protocol import TIMEBASE_SHA256
    from a3_ultrasound.stream_training import catalog_signature
    folds=[f['fold'] for f in cfg['folds']];fold=cfg['folds'][folds.index(fold_name)]
    fold_index=folds.index(fold_name)+1
    rows={r['board']:r for r in timebase['rows']}
    catalog=json.loads(Path(cfg['catalog']).read_text(encoding='utf-8'))
    records,handles=load_train_records(cfg['catalog'],fold['train'],rows)
    sampled=sample_train_records(records,fold['train'],fold_index)
    fit_records={};identities={};short={}
    for entity,row in sampled.items():
        normal=row['y'][:,-1]==0;positions=np.asarray(row['x'])[normal]
        x,lengths=load_rf(handles[entity],positions)
        need=round(row['centers'][-1])+81
        keep=lengths>=need;short[entity]=int((~keep).sum())
        fit_records[entity]=dict(x=x[keep],y=row['y'][normal][keep],centers=row['centers'],family=row['family'])
        identities[entity]=dict(sampled_source_rows=row['row_ids'].tolist(),
                                fit_source_rows=row['row_ids'][normal][keep].tolist())
    params={};cv={}
    for family in FAMILIES:
        result=fit_family(fit_records,fold['train'],family,lambda_grid=list(lambda_grid))
        cv[family]=result.pop('cv_grid',[]);params[family]=result
    boundary=[f for f,p in params.items() if p['status']=='BOUNDARY_LIMITED']
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    side={}
    for name,value in (('sampled_identities',identities),('cv_grid',cv)):
        raw=(json.dumps(value,ensure_ascii=False,indent=1)+'\n').encode('utf-8')
        path=output_dir/f'{fold_name}_m3_{name}.json';path.write_bytes(raw)
        side[name]=dict(path=path.name,sha256=_sha(raw))
    manifest=dict(status='PAUSED_BOUNDARY_LIMITED' if boundary else 'FITTED',
        boundary_limited_families=boundary,fold=fold_name,fold_index=fold_index,train_entities=fold['train'],
        timebase_sha256=TIMEBASE_SHA256,catalog_signature=catalog_signature(catalog),
        lambda_grid=list(lambda_grid),delta_grid=[-24,24],
        rules='RULES_T2_R3 numerical rules; delta integer [-24,24]; one fit per fold and family on TRAIN deep-normal records',
        family_params=params,entity_family={e['entity']:family_of(e) for e in catalog['entities']},
        excluded_short_records=short,side_files=side)
    raw=(json.dumps(manifest,ensure_ascii=False,indent=1)+'\n').encode('utf-8')
    path=output_dir/f'{fold_name}_m3_fit_manifest.json';path.write_bytes(raw)
    return manifest,path
