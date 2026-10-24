"""One user-requested Vanilla control on frozen Stage1 caches, no RF/outer access."""
from pathlib import Path
import hashlib
import json
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.head_cache import cache_batches,head_args,validate_cache_weights,write_json
from a3_ultrasound.inference.enumerator import enumerate_pcc
from a3_ultrasound.models.baselines import VanillaClassifierChainStep
from a3_ultrasound.models.losses import pcc_training_loss
from a3_ultrasound.selection import GuardedSelector
from a3_ultrasound.stream_metrics import PredictionAccumulator


def build_head(mc,seed=23):
    torch.manual_seed(seed)
    return VanillaClassifierChainStep(local_dim=mc['token_dim'],context_dim=mc['token_dim'],
        query_dim=mc['query_dim'],prefix_dim=mc['prefix_dim'],hidden_dim=mc['pcc_hidden_dim'],
        max_interfaces=mc['max_interfaces'],local_base_hidden_dim=mc['local_base_hidden_dim'],
        local_base_type=mc['local_base_type'])


@torch.no_grad()
def evaluate(head,cache,*,device='cpu',batch_size=128,prediction_path=None):
    validate_cache_weights(cache)
    head.eval();acc=PredictionAccumulator()
    handle=Path(prediction_path).open('w',encoding='utf-8') if prediction_path else None
    try:
        for b,ids in cache_batches(cache,batch_size,device):
            args=head_args(b)
            exact=enumerate_pcc(head,*args,b['interface_mask'])
            diag=head(*args,exact.map_states,b['interface_mask'])
            fields=dict(y=b['y'],label_mask=b['label_mask'],marginals=exact.marginals,
                map_states=exact.map_states,base_logodds=diag.base_logodds,
                base_probabilities=diag.base_logodds.sigmoid())
            fields={k:v.cpu().tolist() for k,v in fields.items()}
            probs=exact.probabilities.cpu();logs=exact.log_probabilities.cpu();paths=exact.path_mask.cpu()
            logz=exact.raw_log_normalizer.cpu().tolist()
            counts=b['interface_mask'].sum(1).cpu().tolist();weights=b['weights'].cpu().tolist()
            for i,index in enumerate(ids):
                row={k:v[i][:counts[i]] for k,v in fields.items()}
                # Vanilla has no gate/correction; absence is explicit, never fabricated zero values.
                row.update(sample_id=cache['sample_ids'][index],specimen_id=cache['specimen_ids'][index],
                    K=counts[i],rho=None,correction=None,gate=None,gate_times_correction=None,
                    sample_weight=weights[i],raw_logZ=logz[i],probabilities=probs[i,paths[i]].tolist(),
                    log_probabilities=logs[i,paths[i]].tolist(),head='vanilla_chain',
                    correction_diagnostic_path=None)
                acc.add(row)
                if handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    finally:
        if handle:handle.close()
    return acc.result()


def train_one(train_cache,val_cache,mc,output,*,device='cuda',epochs=15,seed=23):
    validate_cache_weights(train_cache,training=True);validate_cache_weights(val_cache)
    if epochs!=15 or seed!=23:raise ValueError('Only the fixed epoch15/seed23 supplement is authorized.')
    if not torch.equal(train_cache['label_mask'],train_cache['interface_mask']):
        raise ValueError('Training needs complete reliable labels.')
    if set(train_cache['specimen_ids']) & set(val_cache['specimen_ids']):
        raise ValueError('Entity overlap.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    head=build_head(mc,seed).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0)
    selector=GuardedSelector();generator=torch.Generator().manual_seed(seed)
    resolved=dict(stage='1 supplemental control',head='vanilla_chain',seed=seed,epochs=epochs,
        frontend='same frozen V1.3 BN/legacy3 cache as original Stage1; Labels V2',
        model_config=mc,batch_size=16,learning_rate=.001,lambda_local=.3,
        train_normal_sampling='same fixed cached panel; no resampling',
        weight_policy='train unit; validation frozen positive integer IPW',
        formal_result=False,outer_opened=False,final_comparison_epoch=15,
        parameters=sum(p.numel() for p in head.parameters()))
    write_json(output/'resolved_config.json',resolved)
    history=[];started=time.perf_counter()
    for epoch in range(1,16):
        head.train();total=seen=0
        order=torch.randperm(len(train_cache['y']),generator=generator)
        order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        for b,ids in cache_batches(train_cache,16,device,order):
            optimizer.zero_grad(set_to_none=True)
            cond=head(*head_args(b),b['y'],b['interface_mask'])
            loss=pcc_training_loss(cond.logits,cond.base_logodds,b['y'],b['interface_mask'],
                label_mask=b['label_mask'],lambda_local=.3,shadow_auxiliary=False)
            if not torch.isfinite(loss.total):raise RuntimeError('Nonfinite loss.')
            loss.total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('Nonfinite gradient.')
            optimizer.step();total+=float(loss.total.detach())*len(ids);seen+=len(ids)
            if seen%1600==0:
                write_json(output/'progress.json',dict(phase='training',epoch=epoch,records=seen,
                    total_records=len(order),heartbeat_epoch=time.time()))
        metrics=evaluate(head,val_cache,device=device)
        decision=selector.update(epoch,metrics)
        history.append(dict(epoch=epoch,train_loss=total/seen,validation=metrics,
            selection=decision,shuffle_order_sha256=order_hash))
        write_json(output/'history.json',history)
        payload=dict(head_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,resolved_config=resolved,selection_state=selector.state_dict(),
            shuffle_rng_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        torch.save(payload,output/f'epoch_{epoch:02d}.pt')
        if decision['guarded_updated']:torch.save(payload,output/'best_guarded.pt')
        if decision['fallback_updated']:torch.save(payload,output/'best_fallback.pt')
        print(json.dumps(dict(head='vanilla_chain',epoch=epoch,positive_accuracy=metrics['positive_joint_map_accuracy'],
            normal_fpr=metrics['normal_joint_false_positive_rate'])),flush=True)
    final=evaluate(head,val_cache,device=device,prediction_path=output/'final_predictions.jsonl')
    result=dict(status='COMPLETED_DIAGNOSTIC',head='vanilla_chain',epochs=15,seed=23,
        final_metrics=final,guarded_best=selector.guarded,fallback_best=selector.fallback,
        guard_violated=selector.guarded is None,seconds=time.perf_counter()-started,
        formal_result=False,outer_opened=False,automatic_next_stage=False)
    write_json(output/'summary.json',result)
    return result


def main(config_path,output):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if not cfg['development_training_permitted'] or cfg['formal_training_permitted'] or cfg['outer_evaluation_permitted']:
        raise ValueError('Only the explicit Vanilla cached control is allowed.')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    caches={}
    for name,item in cfg['caches'].items():
        p=Path(item['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Cache changed: '+name)
        caches[name]=torch.load(p,map_location='cpu',weights_only=True)
    assert len(caches['train']['y'])==100732 and len(caches['validation']['y'])==40002
    provenance=Path(cfg['original_stage1'])/'provenance.json'
    if hashlib.sha256(provenance.read_bytes()).hexdigest()!=cfg['provenance_sha256']:raise ValueError('Cache provenance changed.')
    write_json(root/'provenance.json',dict(config=cfg,source_provenance=json.loads(provenance.read_text(encoding='utf-8')),
        formal_result=False,outer_opened=False))
    cp=Path(cfg['reference_head_checkpoint'])
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=cfg['reference_head_checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed.')
    mc=torch.load(cp,map_location='cpu',weights_only=True)['resolved_config']['model_config']
    result=train_one(caches['train'],caches['validation'],mc,root/'vanilla_chain',device='cuda')
    write_json(root/'stage1_result.json',dict(status='COMPLETED',supplement='single Vanilla cached control',
        result=result,original_stage1_screening_unchanged=True,automatic_next_stage=False,
        formal_result=False,outer_opened=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.config,a.output)
