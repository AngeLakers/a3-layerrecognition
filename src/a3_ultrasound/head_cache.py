"""Limited legacy-frontend Head diagnostic. Never a V1.4 frontend result."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from a3_ultrasound.inference.enumerator import enumerate_pcc
from a3_ultrasound.models.losses import pcc_training_loss
from a3_ultrasound.models.pcc import PCCConditionalStep
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.selection import GuardedSelector


def write_json(path,value):
    from a3_ultrasound.runtime_supervisor import atomic_json
    atomic_json(path,value)


@torch.no_grad()
def cache_frontend(model,data,*,batch_size,device,progress_path=None):
    model.eval();blocks={};sample_ids=[];specimen_ids=[];weights=[]
    for batch,indices in data.batches(batch_size,device):
        encoded=model.encode(batch);tokens=encoded.responsibility_tokens
        values=dict(r_loc=tokens.r_loc,r_ctx=tokens.r_ctx,r_gate=tokens.r_gate,query=encoded.query,
                    interface_mask=batch.interface_mask,label_mask=batch.label_mask,y=batch.y)
        for name,value in values.items():blocks.setdefault(name,[]).append(value.cpu())
        names=[data.identity(i) for i in indices]
        sample_ids.extend(n[0] for n in names);specimen_ids.extend(n[1] for n in names)
        batch_weights=[a.get('sample_weight',1) for a in data.annotations(indices)]
        if any(type(w) is not int or w<1 for w in batch_weights):
            raise ValueError('Cache weights must be positive integer IPW multiplicities; no silent cast.')
        weights.extend(batch_weights)
        if progress_path and len(weights)%(batch_size*100)==0:
            write_json(progress_path,dict(phase='cache_frontend',records=len(weights),heartbeat_epoch=time.time()))
    result={name:torch.cat(value) for name,value in blocks.items()}
    result.update(sample_ids=sample_ids,specimen_ids=specimen_ids,weights=torch.tensor(weights,dtype=torch.long))
    return result


def cache_batches(cache,size,device,order=None):
    order=torch.arange(len(cache['y'])) if order is None else order
    for start in range(0,len(order),size):
        ids=order[start:start+size]
        yield {k:v[ids].to(device) for k,v in cache.items() if isinstance(v,torch.Tensor)},ids.tolist()


def head_args(batch):
    return tuple(batch[k] for k in ('r_loc','r_ctx','r_gate','query'))


def validate_cache_weights(cache, *, training=False):
    weights=cache['weights']
    if weights.dtype!=torch.long or weights.shape!=(len(cache['y']),) or torch.any(weights<1):
        raise ValueError('Cache needs positive int64 row weights.')
    if training and not torch.all(weights==1):
        raise ValueError('Cached training is sample-equal and requires unit weights.')


@torch.no_grad()
def evaluate_cache(head,cache,*,batch_size=128,device='cpu',prediction_path=None):
    validate_cache_weights(cache)
    head.eval();acc=PredictionAccumulator()
    handle=Path(prediction_path).open('w',encoding='utf-8') if prediction_path else None
    try:
        for b,ids in cache_batches(cache,batch_size,device):
            args=head_args(b);exact=enumerate_pcc(head,*args,b['interface_mask'])
            diag=head(*args,exact.map_states,b['interface_mask'])
            fields=dict(y=b['y'],label_mask=b['label_mask'],marginals=exact.marginals,map_states=exact.map_states,
                base_logodds=diag.base_logodds,base_probabilities=diag.base_logodds.sigmoid(),
                correction=diag.correction,gate=diag.gate,gate_times_correction=diag.gate*diag.correction)
            fields={k:v.cpu().tolist() for k,v in fields.items()}
            probs=exact.probabilities.cpu();logs=exact.log_probabilities.cpu();paths=exact.path_mask.cpu()
            logz=exact.raw_log_normalizer.cpu().tolist()
            counts=b['interface_mask'].sum(1).cpu().tolist();weights=b['weights'].cpu().tolist()
            for i,index in enumerate(ids):
                row={k:v[i][:counts[i]] for k,v in fields.items()}
                row.update(sample_id=cache['sample_ids'][index],specimen_id=cache['specimen_ids'][index],
                    K=counts[i],rho=head.rho,sample_weight=weights[i],raw_logZ=logz[i],
                    probabilities=probs[i,paths[i]].tolist(),log_probabilities=logs[i,paths[i]].tolist(),
                    correction_diagnostic_path='inferred_joint_MAP')
                acc.add(row)
                if handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    finally:
        if handle:handle.close()
    return acc.result()


def fresh_head(model_config,rho,seed):
    torch.manual_seed(seed)
    c=model_config
    return PCCConditionalStep(local_dim=c['token_dim'],context_dim=c['token_dim'],gate_dim=c['token_dim'],
        query_dim=c['query_dim'],prefix_dim=c['prefix_dim'],hidden_dim=c['pcc_hidden_dim'],rho=rho,
        max_interfaces=c['max_interfaces'],local_base_type=c['local_base_type'],
        local_base_hidden_dim=c['local_base_hidden_dim'])


def train_cached_head(train_cache,validation_cache,model_config,output,*,rho,shadow=False,seed=23,
                      epochs=15,batch_size=16,device='cpu'):
    validate_cache_weights(train_cache,training=True);validate_cache_weights(validation_cache)
    if not torch.equal(train_cache['label_mask'],train_cache['interface_mask']):
        raise ValueError('Cached Head training requires complete reliable labels.')
    if set(train_cache['specimen_ids']) & set(validation_cache['specimen_ids']):
        raise ValueError('Cached train/validation entity overlap.')
    if epochs>15:raise ValueError('Stage1 budget is at most 15 epochs.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    head=fresh_head(model_config,rho,seed).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0)
    selector=GuardedSelector();generator=torch.Generator().manual_seed(seed)
    history=[];started=time.perf_counter()
    resolved=dict(stage=1,frontend='frozen legacy V1.3 BN/legacy3; not V1.4 frontend',rho=rho,shadow=shadow,
                  seed=seed,epochs=epochs,batch_size=batch_size,learning_rate=.001,lambda_local=.3,
                  model_config=model_config,train_normal_sampling='fixed cached seed23 panel; explicit stage1 exception',
                  final_comparison_epoch=epochs,outer_opened=False,formal_result=False)
    resolved['weight_policy']='train unit/equal samples; validation frozen inverse-probability positive integer multiplicities'
    write_json(output/'resolved_config.json',resolved)
    for epoch in range(1,epochs+1):
        head.train();total=seen=0
        order=torch.randperm(len(train_cache['y']),generator=generator)
        for b,ids in cache_batches(train_cache,batch_size,device,order):
            optimizer.zero_grad(set_to_none=True)
            cond=head(*head_args(b),b['y'],b['interface_mask'])
            loss=pcc_training_loss(cond.logits,cond.base_logodds,b['y'],b['interface_mask'],
                label_mask=b['label_mask'],lambda_local=.3,shadow_auxiliary=shadow)
            if not torch.isfinite(loss.total):raise RuntimeError('Nonfinite cached loss.')
            loss.total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('Nonfinite cached gradient.')
            optimizer.step();total+=float(loss.total.detach())*len(ids);seen+=len(ids)
            if seen%(batch_size*100)==0:
                write_json(output/'progress.json',dict(phase='training',epoch=epoch,records=seen,total_records=len(order),heartbeat_epoch=time.time()))
        metrics=evaluate_cache(head,validation_cache,device=device)
        decision=selector.update(epoch,metrics)
        record=dict(epoch=epoch,train_loss=total/seen,validation=metrics,selection=decision)
        history.append(record);write_json(output/'history.json',history)
        payload=dict(head_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,resolved_config=resolved,selection_state=selector.state_dict(),shuffle_rng_state=generator.get_state(),
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        torch.save(payload,output/f'epoch_{epoch:02d}.pt')
        if decision['guarded_updated']:torch.save(payload,output/'best_guarded.pt')
        if decision['fallback_updated']:torch.save(payload,output/'best_fallback.pt')
        print(json.dumps(dict(stage=1,rho=rho,shadow=shadow,epoch=epoch,positive_accuracy=metrics['positive_joint_map_accuracy'])),flush=True)
    final=evaluate_cache(head,validation_cache,device=device,prediction_path=output/'final_predictions.jsonl')
    result=dict(status='COMPLETED_DIAGNOSTIC',epochs=epochs,final_metrics=final,
        rho=rho,shadow=shadow,seed=seed,
        guarded_best=selector.guarded,fallback_best=selector.fallback,guard_violated=selector.guarded is None,
        seconds=time.perf_counter()-started,formal_result=False,outer_opened=False)
    write_json(output/'summary.json',result)
    return result


def select_stage1_candidates(rho1,rho3,shadow):
    """Fixed endpoint screening, not selection of the best-looking epoch."""
    for result,(rho,is_shadow) in zip((rho1,rho3,shadow),((1,False),(3,False),(3,True))):
        if (result.get('status')!='COMPLETED_DIAGNOSTIC' or result.get('epochs')!=15 or
            result.get('rho')!=rho or result.get('shadow')!=is_shadow or result.get('seed')!=23):
            raise ValueError('Stage1 screening requires all three registered seed23 variants at epoch15.')
    a,b,c=[x['final_metrics'] for x in (rho1,rho3,shadow)]
    passed=(b['correction_saturation_fraction_MAP_path'] is not None and
            b['correction_saturation_fraction_MAP_path']<.5 and
            b['correction_saturation_fraction_MAP_path']<a['correction_saturation_fraction_MAP_path'] and
            b['positive_joint_map_accuracy']>=a['positive_joint_map_accuracy'] and
            b['normal_joint_false_positive_rate']<=.01)
    result=[]
    if passed:
        result.append('proposed_rho3')
        if c['positive_joint_map_accuracy']>=b['positive_joint_map_accuracy'] and c['normal_joint_false_positive_rate']<=.01:
            result.append('proposed_shadow_aux_rho3')
    return result
