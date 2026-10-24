"""Exact phase membership and explicit per-record joint-posterior diagnostics."""
import json
from pathlib import Path
import numpy as np

HEADS=('local_independent','vanilla_chain','proposed','local_pre_independent')
SEEDS=(23,41,59)
FOLDS=('F1','F2','F3','F4','F5')


def phase_cells(cfg):
    if cfg['heads']!=list(HEADS) or cfg['seeds']!=list(SEEDS) or [f['fold'] for f in cfg['folds']]!=list(FOLDS):
        raise ValueError('Phase60 requires the exact ordered four heads, five folds and three seeds.')
    result=[(f,s,h) for s in SEEDS for f in FOLDS for h in HEADS]
    if len(set(result))!=60:raise ValueError('Duplicate matrix cell.')
    return result


def prediction_diagnostics(path):
    groups={};records=0;seen=set()
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line);k=int(row['K']);identity=(row['specimen_id'],row['sample_id'])
            if identity in seen:raise ValueError('Duplicate prediction sample identity.')
            seen.add(identity);records+=1
            y=np.asarray(row['y']);mask=np.asarray(row['label_mask'],bool)
            p=np.asarray(row['probabilities'],float);marginal=np.asarray(row['marginals'],float)
            predicted=np.asarray(row['map_states']);powers=2**np.arange(k-1,-1,-1)
            if y.shape!=(k,) or mask.shape!=(k,) or predicted.shape!=(k,) or marginal.shape!=(k,) or p.shape!=(2**k,):
                raise ValueError('Prediction contract shape mismatch.')
            if not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=2e-5):
                raise ValueError('Invalid joint posterior.')
            paths=(np.arange(2**k)[:,None]>>np.arange(k-1,-1,-1))&1
            if not np.allclose(p@paths,marginal,atol=2e-5):raise ValueError('Marginals disagree with joint posterior.')
            if not np.array_equal(paths[int(p.argmax())],predicted):raise ValueError('MAP differs from joint argmax.')
            if not np.isin(y[mask],(0,1)).all():raise ValueError('Invalid known true state.')
            key=str(row['specimen_id'])
            group=groups.setdefault(key,dict(K=k,records=0,complete=0,partial=0,confusion=np.zeros((2**k,2**k),np.int64),
                 weighted_confusion=np.zeros((2**k,2**k),np.int64)))
            if group['K']!=k:raise ValueError('Entity K changed.')
            group['records']+=1
            if mask.all():
                truth=int(y@powers);guess=int(predicted@powers);weight=int(row.get('sample_weight',1))
                if weight<1:raise ValueError('Invalid sample weight.')
                group['complete']+=1;group['confusion'][truth,guess]+=1;group['weighted_confusion'][truth,guess]+=weight
            else:group['partial']+=1
    for group in groups.values():
        k=group['K'];matrix=group['confusion'];states=(np.arange(2**k)[:,None]>>np.arange(k-1,-1,-1))&1
        multi=states.sum(1)>=2;single=states.sum(1)==1
        group['multi_to_single_count']=int(matrix[np.ix_(multi,single)].sum())
        group['multi_true_count']=int(matrix[multi].sum())
        group['single_wrong_layer_count']=int(matrix[np.ix_(single,single)].sum()-np.trace(matrix[np.ix_(single,single)]))
        group['state_order']=[format(i,f'0{k}b') for i in range(2**k)]
        group['support']=matrix.sum(1).tolist()
        group['recall']=[None if n==0 else float(matrix[i,i]/n) for i,n in enumerate(matrix.sum(1))]
        group['confusion']=matrix.tolist();group['weighted_confusion']=group['weighted_confusion'].tolist()
    return dict(status='PASS',records=records,entities=groups,predicted_state='joint_MAP_not_marginal_threshold',
                joint_path_order='lexicographic_I1_to_deepest',partial_labels_not_joint_scored=True)
