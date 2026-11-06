"""V1.6 fixed M3 decomposition; historical numerical rules retained.

Source: separation_feasibility_20261105/run_t2.py. No data reads on import.
Fit accepts explicitly authorized TRAIN records; transform has no labels.
"""
from __future__ import annotations
import math
import numpy as np

def omp(D,Y):
    # Batched exact OMP normal-equation solves, deterministic atom order/ties.
    G=D.T@D;C=Y@D;n=len(Y);atoms=D.shape[1];selected=np.zeros((n,8),dtype=int);coef=np.zeros((n,8));counts=np.zeros(n,dtype=int);alive=np.ones(n,dtype=bool);norm=(Y*Y).sum(1);relres=norm.copy()
    for step in range(8):
        ids=np.flatnonzero(alive)
        if not len(ids):break
        corr=C[ids].copy()
        if step:
            corr-=np.einsum('nks,ns->nk',G[:,selected[ids,:step]].transpose(1,0,2),coef[ids,:step])
            corr[np.arange(len(ids))[:,None],selected[ids,:step]]=0
        index=np.argmax(np.abs(corr),axis=1)
        if step:
            old=selected[ids,:step];GG=G[old[:,:,None],old[:,None,:]];cross=G[old,index[:,None]]
            solved=np.linalg.solve(GG,cross[...,None])[...,0];pivot=G[index,index]-(cross*solved).sum(1)
        else:pivot=G[index,index]
        good=pivot>1e-10
        alive[ids[~good]]=False;ids=ids[good];index=index[good]
        if not len(ids):continue
        selected[ids,step]=index;sel=selected[ids,:step+1];GG=G[sel[:,:,None],sel[:,None,:]];cc=C[ids[:,None],sel]
        weights=np.linalg.solve(GG,cc[...,None])[...,0];coef[ids,:step+1]=weights;counts[ids]=step+1
        relres[ids]=np.maximum(0,norm[ids]-(weights*cc).sum(1));alive[ids[relres[ids]<=1e-12*norm[ids]]]=False
    return selected,coef,counts

def dictionary(centers, length, params):
    tau=np.asarray(centers,float)
    if len(tau) not in (2,3) or not np.isfinite(tau).all() or (np.diff(tau)<=48).any():
        raise ValueError('Ordered K2/K3 centers with gap >48 required.')
    delta=params['delta']
    if not np.isfinite(delta) or int(delta)!=delta or abs(delta)>24:
        raise ValueError('delta must be a finite integer in [-24,24].')
    if params.get('status') == 'BOUNDARY_LIMITED' or abs(delta) == 24:
        raise ValueError('BOUNDARY_LIMITED: M3 paused; do not widen delta automatically.')
    gap=float(tau[-1]-tau[-2]); delta=params['delta']; alpha=params['alpha']
    f0=params['f0_cycles_per_sample']; sigma=params['sigma_samples']
    if not (0<f0<.5 and sigma>0 and np.isfinite(sigma) and -1<=alpha<=1):
        raise ValueError('Invalid fitted Gabor parameters.')
    lo=max(0,round(tau[0])-80); dl=max(0,round(tau[-1])-80); hi=min(length,round(tau[-1])+81)
    if hi<=dl or round(tau[-1])+80>=length:
        raise ValueError('Full deep window required.')
    tt=np.arange(length,dtype=float)
    def g(c):return np.exp(-.5*((tt-c)/sigma)**2)*np.cos(2*np.pi*f0*(tt-c))
    atoms=[];copies=[];deep=[]
    for center in range(lo,hi):
        isdeep=center>=dl; copy=np.zeros(length)
        if not isdeep:
            copy=alpha*g(center+gap+delta)
            if len(tau)==3:copy+=alpha**2*g(center+2*gap+2*delta)
        atom=g(center)+copy; norm=np.linalg.norm(atom[lo:hi])
        if norm<=1e-15:continue
        atoms.append(atom/norm);copies.append(copy/norm);deep.append(isdeep)
    return np.stack(atoms,1),np.stack(copies,1),np.asarray(deep),lo,hi


def transform(x, centers, params):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite [N,T] RF required.')
    if params.get('status')=='IDENTITY_FALLBACK':
        return x.astype(np.float32),np.zeros_like(x,dtype=np.float32)
    D,C,deep,lo,hi=dictionary(centers,x.shape[1],params)
    selected,coef,counts=omp(D[lo:hi],x[:,lo:hi])
    multiple=np.zeros_like(x);free=np.zeros_like(x)
    for i,count in enumerate(counts):
        ids=selected[i,:count]; weights=coef[i,:count]
        multiple[i,lo:hi]=C[lo:hi,ids]@weights
        mask=deep[ids]
        if mask.any():free[i]=D[:,ids[mask]]@weights[mask]
    return (x-multiple).astype(np.float32),free.astype(np.float32)


def _features(x, centers, delta):
    lo=round(centers[-1])-80;hi=round(centers[-1])+81
    if lo<0 or hi>x.shape[1]:raise ValueError('Full deep fitting window required.')
    times=np.arange(lo,hi,dtype=float);gap=centers[-1]-centers[-2];cols=[]
    for j in range(1,len(centers)):
        for lag in range(-4,5):
            q=times-j*gap-lag-delta;left=np.floor(q).astype(int);frac=q-left
            val=np.zeros((len(x),len(q)))
            for index,weight in [(left,1-frac),(left+1,frac)]:
                ok=(index>=0)&(index<x.shape[1]);val[:,ok]+=x[:,index[ok]]*weight[ok]
            cols.append(val)
    return np.stack(cols,-1),x[:,lo:hi]


def fit_family(records, train_entities, family, *, lambda_grid):
    """records: board->{x, y, centers, family}; arrays are complete reliable sampled TRAIN.

    Caller retains sampled IDs. Pooled fallback uses TRAIN K2 even for a K3
    target; only scalar alpha/delta/frequency are transferred (FIR h is fit-only).
    """
    if set(records)-set(train_entities):raise ValueError('Non-TRAIN board in fit.')
    for row in records.values():
        x=np.asarray(row['x']); y=np.asarray(row['y']); k=len(row['centers'])
        if x.ndim!=2 or y.shape!=(len(x),k) or not np.isin(y,[0,1]).all() or not np.isfinite(x).all():
            raise ValueError('Complete reliable TRAIN records required.')
    def choose(names):
        return [(b,np.asarray(records[b]['x'],float)[np.asarray(records[b]['y'])[:,-1]==0],records[b]['centers']) for b in names]
    items=choose([b for b in records if records[b]['family']==family]);status='FITTED'
    if sum(len(x) for _,x,_ in items)<500:
        items=choose([b for b in records if len(records[b]['centers'])==2]);status='POOLED_FIT'
    nfit=sum(len(x) for _,x,_ in items)
    support={}
    for b,x,tau in items:
        labels=np.asarray(records[b]['y'])
        for row in labels[labels[:,-1]==0]:
            key=f'K{len(row)}:'+''.join(map(str,row[:-1].astype(int)))
            support[key]=support.get(key,0)+1
    support_info=dict(fit_normal_support_by_upper=support,
                      evaluation_support_minimum=200,
                      support_scope='Only evaluation eligibility; never truth-routed transformation')
    if nfit<500:return dict(status='IDENTITY_FALLBACK',fit_records=nfit,fit_boards=[b for b,x,_ in items if len(x)],**support_info)
    items=[row for row in items if len(row[1])]
    p=9*(len(items[0][2])-1);chosen=None;curve=[]
    for delta in range(-24,25):
        G=np.zeros((5,p,p));v=np.zeros((5,p));yy=np.zeros(5);count=np.zeros(5)
        for b,x,tau in items:
            for fold,block in enumerate(np.array_split(x,5)):
                for sub in np.array_split(block,max(1,math.ceil(len(block)/128))):
                    if not len(sub):continue
                    A,Y=_features(sub,tau,delta);A=A.reshape(-1,p);Y=Y.ravel()
                    G[fold]+=A.T@A;v[fold]+=A.T@Y;yy[fold]+=Y@Y;count[fold]+=len(Y)
        for lam in lambda_grid:
            errors=[]
            for q in range(5):
                n=count.sum()-count[q];h=np.linalg.solve((G.sum(0)-G[q])/n+lam*np.eye(p),(v.sum(0)-v[q])/n)
                errors.append(yy[q]-2*h@v[q]+h@G[q]@h)
            score=float(np.sum(errors)/count.sum());curve.append(dict(delta=delta,lambda_value=lam,cv_mse=score))
            rank=(score,-lam,abs(delta),delta)
            if chosen is None or rank<chosen[0]:chosen=(rank,delta,lam)
    _,delta,lam=chosen
    dot=norm=0.;maxlen=max(x.shape[1] for _,x,_ in items);nfft=2**math.ceil(math.log2(maxlen));spec=np.zeros(nfft//2+1)
    for b,x,tau in items:
        for sub in np.array_split(x,max(1,math.ceil(len(x)/128))):
            A,Y=_features(sub,tau,delta);up=A[:,:,4];dot+=float((up*Y).sum());norm+=float((up*up).sum())
            spec+=np.abs(np.fft.rfft(sub,n=nfft,axis=1)).sum(0)
    if norm==0:raise ValueError('Zero upstream fit energy.')
    spec/=nfit;index=int(np.argmax(spec[1:-1]))+1
    if spec[index]<=0:raise ValueError('Zero fitting frequency peak.')
    low=high=index
    while low>1 and spec[low-1]>=spec[index]/2:low-=1
    while high<len(spec)-2 and spec[high+1]>=spec[index]/2:high+=1
    bandwidth=max(1,high-low)/nfft
    return dict(status='BOUNDARY_LIMITED' if abs(delta)==24 else status,fit_scope=status,
        **support_info,
        fit_records=nfit,fit_boards=[b for b,_,_ in items],family=family,
        alpha_raw=dot/norm,alpha=float(np.clip(dot/norm,-1,1)),delta=delta,lambda_value=lam,
        f0_cycles_per_sample=index/nfft,bandwidth_cycles_per_sample=bandwidth,
        sigma_samples=math.sqrt(2*math.log(2))/(math.pi*bandwidth),cv_grid=curve)


def sample_train_records(records, train_entities, fold_index):
    """Fold-local old 5000-per-full-state cap; records must already be eligible.

    Catalog ordinal then ascending source row defines insertion order. One
    RNG per fold SeedSequence([20261105, fold_index]); all sampled identities
    are returned. Never reuse validation diagnostic samples for fitting.
    """
    if set(records)!=set(train_entities) or type(fold_index) is not int or not 1<=fold_index<=5:
        raise ValueError('Exactly the registered TRAIN entities and 1-based fold index required.')
    groups={};ordered=sorted(records,key=lambda b:records[b]['ordinal'])
    for b in ordered:
        row=records[b];y=np.asarray(row['y']);ids=np.asarray(row['row_ids'])
        if y.ndim!=2 or y.shape[1] not in (2,3) or not np.isin(y,[0,1]).all() or ids.shape!=(len(y),) or len(np.unique(ids))!=len(ids):
            raise ValueError('Unique source rows and complete reliable labels required.')
        for i in np.argsort(ids,kind='stable'):
            key=(y.shape[1],tuple(y[i].astype(int)))
            groups.setdefault(key,[]).append((b,int(i)))
    rng=np.random.default_rng(np.random.SeedSequence([20261105,fold_index]));selected={b:[] for b in ordered}
    for group in groups.values():
        take=rng.choice(len(group),min(5000,len(group)),replace=False) if len(group)>5000 else np.arange(len(group))
        for j in take:
            b,i=group[j];selected[b].append(i)
    result={}
    for b in ordered:
        row=records[b];ix=np.array(sorted(selected[b],key=lambda i:row['row_ids'][i]),int)
        result[b]=dict(row,x=np.asarray(row['x'])[ix],y=np.asarray(row['y'])[ix],row_ids=np.asarray(row['row_ids'])[ix])
    return result
