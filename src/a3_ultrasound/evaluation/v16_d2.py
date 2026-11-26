"""R3 D2 signal gate: envelope scores, AUC table and the pre-registered decision."""
import math
import numpy as np
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu

D2_SEED=20261106
PER_CLASS_CAP=5000


def envelope_peak(x, lengths, deep_center):
    """RULES_T2_R3: full-record Hilbert envelope, max over round(tau_K)-80..+80.

    Records are grouped by stored length so the Hilbert transform never sees an
    artificial zero tail. Python round() is ties-to-even, as in run_t2.py.
    """
    x=np.asarray(x,float);lengths=np.asarray(lengths,int);out=np.empty(len(x))
    t=round(float(deep_center))
    for n in np.unique(lengths):
        ids=np.flatnonzero(lengths==n);lo,hi=max(0,t-80),min(int(n),t+81)
        if hi<=lo:raise ValueError('Deep window outside the record.')
        out[ids]=np.abs(hilbert(x[ids,:int(n)],axis=1))[:,lo:hi].max(1)
    return out


def auc(abnormal, normal):
    """Signed Mann-Whitney U/(n1 n0); larger score = abnormal; ties count 1/2."""
    if not len(abnormal) or not len(normal):return None
    return float(mannwhitneyu(abnormal,normal).statistic/(len(abnormal)*len(normal)))


def average_precision(abnormal, normal):
    """Step-wise AP (precision at each distinct threshold times recall gain)."""
    if not len(abnormal) or not len(normal):return None
    score=np.r_[abnormal,normal];label=np.r_[np.ones(len(abnormal)),np.zeros(len(normal))]
    order=np.argsort(-score,kind='stable');score,label=score[order],label[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    tp=np.cumsum(label)[ends];total=ends+1.
    recall=tp/len(abnormal);precision=tp/total
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def sample_event(y, fold_index, board_ordinal):
    """Cap 5000 per K:upper|deep class within one validation board.

    One RNG per event: SeedSequence([20261106, fold_index, board_ordinal]);
    classes visited in sorted key order; returned positions ascending.
    """
    y=np.asarray(y,int);rng=np.random.default_rng(np.random.SeedSequence([D2_SEED,fold_index,board_ordinal]))
    keys=[''.join(map(str,row)) for row in y];chosen=[]
    for key in sorted(set(keys)):
        ids=np.flatnonzero(np.array(keys)==key)
        if len(ids)>PER_CLASS_CAP:ids=np.sort(rng.choice(ids,PER_CLASS_CAP,replace=False))
        chosen.append(ids)
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0,int)


def event_rows(fold, board, y, scores, support):
    """One row per upper-state group: AUC/AP of x, r, d for deep abnormal vs normal."""
    y=np.asarray(y,int);k=y.shape[1];rows=[]
    for upper in range(2**(k-1)):
        bits=format(upper,f'0{k-1}b');group=f'K{k}:{bits}'
        mask=np.all(y[:,:-1]==np.array(list(bits),int),axis=1)
        pos=mask&(y[:,-1]==1);neg=mask&(y[:,-1]==0)
        row=dict(fold=fold,board=board,K=k,group=group,n_abnormal=int(pos.sum()),n_normal=int(neg.sum()),
                 fit_normal_support=int(support.get(group,0)))
        for c in ('x','r','d'):
            row[c]=auc(scores[c][pos],scores[c][neg]);row['ap_'+c]=average_precision(scores[c][pos],scores[c][neg])
        rows.append(row)
    return rows


def choose_channel(rows, planned_events):
    """rows: fold,board,group,x,r,d AUCs and fit_normal_support.

    planned_events are all five validation fold-board events (TC4 only).
    K2:0 and K3:00 are checked separately; K3:01/10 separately. Missing
    classes/support/events returns INSUFFICIENT, never cancellation by default.
    """
    from a3_ultrasound.v16_protocol import template
    expected={(f['fold'],b,2 if b.startswith('TCL3-') else 3)
              for f in template()['folds'] for b in f['validation'] if b.startswith(('TCL3-','TCL4-'))}
    provided={(e['fold'],e['board'],e['K']) for e in planned_events}
    if provided!=expected or len(planned_events)!=len(expected):
        raise ValueError('D2 planned events must match the frozen five-fold TC4 validation panel.')
    needed={}
    for event in planned_events:
        for group in (('K2:0','K2:1') if event['K']==2 else ('K3:00','K3:01','K3:10')):
            needed[(event['fold'],event['board'],group)]=None
    seen=set()
    for row in rows:
        key=(row['fold'],row['board'],row['group'])
        if key in seen:raise ValueError('Duplicate D2 fold-board-group.')
        seen.add(key)
        if key in needed:needed[key]=row
    # Pre-registered insufficiency rule (2026-11-06, Claude; flagged for Brandon):
    # a row is valid when both classes exist (AUC defined) and the fit had >=200
    # deep-normal records with that upper state. K2:0/K2:1 need all five TC4-K2
    # events (the 4/5 sign count is defined on five). K3 groups use their valid
    # events if at least three remain; excluded events are reported, never imputed.
    def valid(row):
        return (row is not None and row.get('fit_normal_support',0)>=200
                and all(isinstance(row.get(c),(int,float)) and math.isfinite(row[c]) and 0<=row[c]<=1 for c in ('x','r','d')))
    groups={g:[] for g in ('K2:0','K2:1','K3:00','K3:01','K3:10')};excluded=[]
    for key,row in needed.items():
        if valid(row):groups[key[2]].append(row)
        else:excluded.append(key)
    short=[g for g,v in groups.items() if len(v)<(5 if g.startswith('K2') else 3)]
    if short:
        return dict(status='INSUFFICIENT',channel=None,insufficient_groups=short,excluded_events=excluded,
                    action='Stop before any E4/E5 training and report to Brandon; no default cancellation.')
    checks={}
    for channel in ('r','d'):
        def gain(group,other='x'):return np.array([r[channel]-r[other] for r in groups[group]])
        values=[np.median(gain('K2:1'))>=.03,int((gain('K2:1')>0).sum())>=4,
                np.median(gain('K2:0'))>=-.02,np.median(gain('K3:00'))>=-.02,
                np.median(gain('K3:01'))>=0,np.median(gain('K3:10'))>=0]
        if channel=='r':values.append(np.median(gain('K2:1','d'))>=-.02)
        checks[channel]=list(map(bool,values))
    channel='residual' if all(checks['r']) else 'free_deep' if all(checks['d']) else 'none'
    return dict(status='DECIDED',channel=channel,checks=checks,excluded_events=excluded,
                events_used={g:len(v) for g,v in groups.items()},
                matrix_cells=65 if channel=='none' else 95)


def run_d2(cfg, timebase, manifests):
    """All five folds: validation boards only, fold-fitted M3, then the decision.

    manifests: fold -> (manifest dict, sha256 of its exact bytes). Test boards are
    never opened. Aluminium and K3:11 rows are computed for the report only.
    """
    import json
    from pathlib import Path
    from a3_ultrasound.data.m3_fit import load_rf
    from a3_ultrasound.data.m3_transform import Decomposer
    from a3_ultrasound.v16_protocol import TIMEBASE_SHA256
    rows=[];planned=[];identities={};centres={r['board']:r for r in timebase['rows']}
    catalog_path=Path(cfg['catalog']);catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    registry={e['entity']:e for e in catalog['entities']}
    for fold_index,fold in enumerate(cfg['folds'],start=1):
        manifest,_=manifests[fold['fold']]
        if (manifest.get('status')!='FITTED' or manifest.get('fold')!=fold['fold']
                or manifest.get('train_entities')!=fold['train'] or manifest.get('timebase_sha256')!=TIMEBASE_SHA256):
            raise ValueError('D2 needs a FITTED manifest of the same fold, TRAIN panel and timebase: '+fold['fold'])
        for board in fold['validation']:
            entry=registry[board];base=catalog_path.parent/entry['index_dir']
            arrays={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in
                    ('source_row','offsets','lengths','y','label_mask','eligible')}
            ids=np.flatnonzero(np.asarray(arrays['eligible'])&np.asarray(arrays['label_mask']).all(1))
            y=np.asarray(arrays['y'][ids]).astype(int)
            take=sample_event(y,fold_index,int(entry['ordinal']));positions=ids[take];y=y[take]
            x,lengths=load_rf(dict(arrays=arrays,rf=np.load(entry['rf_path'],mmap_mode='r')),positions)
            family=manifest['entity_family'][board];params=manifest['family_params'][family]
            tau=[float(t) for t in centres[board]['selected_tau']]
            r=np.zeros_like(x);d=np.zeros_like(x)
            for n in np.unique(lengths):
                group=np.flatnonzero(lengths==n)
                r[group,:n],d[group,:n]=Decomposer(tau,int(n),params)(x[group,:n])
            scores={c:envelope_peak(v,lengths,tau[-1]) for c,v in (('x',x),('r',r),('d',d))}
            support=params.get('fit_normal_support_by_upper',{})
            rows+=[dict(row,family=family,param_status=params['status'])
                   for row in event_rows(fold['fold'],board,y,scores,support)]
            identities[f"{fold['fold']}:{board}"]=np.asarray(arrays['source_row'][positions]).astype(int).tolist()
            if entry['group']!='AL':planned.append(dict(fold=fold['fold'],board=board,K=int(entry['K'])))
    tc4=[r for r in rows if r['family']!='Al-K2']
    decision=choose_channel(tc4,planned)
    decision.update(fit_manifest_sha256={f:sha for f,(_,sha) in manifests.items()},
        timebase_sha256=TIMEBASE_SHA256,score='full-record Hilbert envelope peak, deep window round(tau_K)-80..+80',
        sampling=f'<= {PER_CLASS_CAP} per K:upper|deep class per validation board; SeedSequence([{D2_SEED}, fold_index, board_ordinal])',
        report_only=['K3:11','Al-K2'])
    return decision,rows,identities
