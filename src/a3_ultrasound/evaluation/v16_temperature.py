"""Global conditional-logit temperature, with specimen-grouped calibration holdout."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def tempered_log_joint(log_joint, temperature):
    """Recover each binary conditional from the joint, then re-enumerate.

    Unlike softmax(log_joint/T), this implements a single T at every chain
    step, including unobserved prefixes. Positive mass at every prefix is
    required: underflowed probabilities cannot recover conditionals.
    """
    q = np.asarray(log_joint, dtype=float)
    if (q.ndim != 2 or q.shape[1] not in (4, 8) or not np.isfinite(q).all()
            or not np.isfinite(temperature) or not .1 <= temperature <= 10
            or not np.allclose(logsumexp(q, axis=1), 0, atol=2e-6, rtol=0)):
        raise ValueError('Finite normalized K2/K3 log posteriors and T in [0.1,10] required.')
    result = np.zeros_like(q)
    states = q.shape[1]
    for level in range(int(np.log2(states))):
        width = states >> level; half = width // 2
        for start in range(0, states, width):
            a = logsumexp(q[:, start:start+half], axis=1)
            b = logsumexp(q[:, start+half:start+width], axis=1)
            logits = (b-a)/temperature
            result[:, start:start+half] += (-np.logaddexp(0, logits))[:,None]
            result[:, start+half:start+width] += (-np.logaddexp(0, -logits))[:,None]
    return result


def fit_temperature(log_joint, truth_index, weights):
    q = np.asarray(log_joint, float); y = np.asarray(truth_index); w = np.asarray(weights,float)
    tempered_log_joint(q, 1.)
    if (y.shape != (len(q),) or y.dtype.kind not in 'iu' or (y<0).any() or (y>=q.shape[1]).any()
            or w.shape != (len(q),) or not len(q) or not np.isfinite(w).all() or (w<=0).any()):
        raise ValueError('Invalid calibration cohort.')
    def objective(log_t):
        return float(-np.dot(w,tempered_log_joint(q,float(np.exp(log_t)))[np.arange(len(y)),y])/w.sum())
    lower, upper = np.log(.1),np.log(10.)
    opt = minimize_scalar(objective, bounds=(lower+1e-12,upper-1e-12), method='bounded')
    candidates=[(objective(t),t) for t in (lower+1e-12,upper-1e-12,0.,float(opt.x))]
    if max(v for v,_ in candidates)-min(v for v,_ in candidates)<=1e-12:
        return dict(temperature=1.,weighted_nll=objective(0.),success=True,status='UNIDENTIFIABLE_FLAT_OBJECTIVE')
    score, log_t = min(candidates)
    return dict(temperature=float(np.exp(log_t)), weighted_nll=score, success=bool(opt.success))


def specimen_oof(log_joint, truth_index, weights, specimen_ids):
    """Calibration-stage holdout only; does not remove backbone/selection exposure."""
    q=np.asarray(log_joint,float); y=np.asarray(truth_index); w=np.asarray(weights,float); ids=np.asarray(specimen_ids)
    if ids.shape != (len(q),) or len(np.unique(ids))<2:
        raise ValueError('At least two specimens are needed.')
    output=np.empty_like(q); records={}
    for board in np.unique(ids):
        held=ids==board
        fitted=fit_temperature(q[~held],y[~held],w[~held])
        output[held]=tempered_log_joint(q[held],fitted['temperature'])
        records[str(board)]=dict(**fitted,fit_specimens=np.unique(ids[~held]).tolist(),
                                held_records=int(held.sum()),fit_records=int((~held).sum()))
    return output, records


def mixed_specimen_oof(rows):
    """One temperature across K2 AND K3, per head/seed (never one T per K).

    Rows use saved prediction keys plus head and seed. All repeated fold
    appearances of a held specimen are excluded together.
    """
    if not rows or len({(r['head'],r['seed']) for r in rows})!=1:
        raise ValueError('Exactly one head/seed per calibration call required.')
    if rows[0]['head'] not in ('local_pre_independent','vanilla_chain'):
        raise ValueError('C6 only supports LP and V.')
    if any(len(r['label_mask'])!=len(r['y']) or not all(type(v) in (bool,np.bool_) and v for v in r['label_mask']) for r in rows):
        raise ValueError('Complete reliable boolean masks required.')
    groups={}
    for k in (2,3):
        indices=np.array([i for i,r in enumerate(rows) if len(r['y'])==k],int)
        if not len(indices):continue
        subset=[rows[i] for i in indices]
        q=np.array([r['log_probabilities'] for r in subset]); tempered_log_joint(q,1.)
        y=np.array([sum(v<<(k-j-1) for j,v in enumerate(r['y'])) for r in subset])
        w=np.array([r['sample_weight'] for r in subset],float)
        if not np.isfinite(w).all() or (w<=0).any() or any(not all(v in (0,1) for v in r['y']) for r in subset):
            raise ValueError('Invalid label or IPW.')
        groups[k]=(indices,q,y,w,np.array([r['specimen_id'] for r in subset]))
    if sum(len(g[0]) for g in groups.values())!=len(rows):raise ValueError('Only K2/K3 supported.')
    boards=sorted({r['specimen_id'] for r in rows})
    if len(boards)<2:raise ValueError('At least two specimens required.')
    def fit(excluded):
        def loss(log_t):
            total=mass=0.
            for _,q,y,w,ids in groups.values():
                keep=ids!=excluded if excluded is not None else np.ones(len(ids),bool)
                if not keep.any():continue
                logp=tempered_log_joint(q[keep],float(np.exp(log_t)))
                total-=float(np.dot(w[keep],logp[np.arange(keep.sum()),y[keep]]));mass+=float(w[keep].sum())
            return total/mass
        low,high=np.log(.1)+1e-12,np.log(10)-1e-12
        opt=minimize_scalar(loss,bounds=(low,high),method='bounded')
        if not opt.success:raise RuntimeError('Temperature optimizer failed.')
        candidates=[(loss(t),t) for t in (low,high,0.,float(opt.x))]
        flat=max(v for v,_ in candidates)-min(v for v,_ in candidates)<=1e-12
        value,logt=(loss(0.),0.) if flat else min(candidates)
        return dict(temperature=float(np.exp(logt)),weighted_nll=value,
                    status='UNIDENTIFIABLE_FLAT_OBJECTIVE' if flat else 'FITTED',
                    fit_specimens=[b for b in boards if b!=excluded])
    output=[None]*len(rows);records={}
    for board in boards:
        fitted=fit(board);records[board]=fitted
        for indices,q,_,_,ids in groups.values():
            keep=ids==board
            if keep.any():
                adjusted=tempered_log_joint(q[keep],fitted['temperature'])
                for i,value in zip(indices[keep],adjusted):output[i]=value.tolist()
    return output,records,fit(None)
