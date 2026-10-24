"""Explicit, review-gated stage1 diagnostic only; never launches stage2/3/4."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.training import ModelConfig,build_model
from a3_ultrasound.data.streaming import RFShardDataset
from a3_ultrasound.stream_training import catalog_signature,validate_fold
from a3_ultrasound.head_cache import cache_frontend,train_cached_head,select_stage1_candidates,write_json


def main(config_path,output):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if not cfg.get('development_training_permitted') or cfg.get('formal_training_permitted') or cfg.get('outer_evaluation_permitted'):
        raise ValueError('Only reviewed limited diagnostics are allowed; formal and outer remain closed.')
    if cfg.get('development_stage')!=1 or cfg['seed']!=23 or cfg['fold']['fold']!='F1':
        raise ValueError('This entrypoint is only stage1 F1/seed23.')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cp=Path(cfg['legacy_checkpoint'])
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=cfg['legacy_checkpoint_sha256']:
        raise ValueError('Legacy checkpoint hash mismatch.')
    saved=torch.load(cp,map_location='cpu',weights_only=True)
    if saved['resolved_config']['fold']!=cfg['fold'] or saved['resolved_config']['seed']!=23:
        raise ValueError('Legacy checkpoint fold/seed differs from cache protocol.')
    if saved['resolved_config']['catalog_signature']!=cfg['catalog_signature']:
        raise ValueError('Legacy checkpoint label/input catalog differs from cache protocol.')
    mc=saved['resolved_config']['model'];model=build_model(ModelConfig(**mc))
    if model.query_encoder.network[0].in_features!=3:raise ValueError('Stage1 expects explicit legacy frontend.')
    model.load_state_dict(saved['model_state_dict']);device=cfg['device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model.to(device);model.eval()
    catalog=json.loads(Path(cfg['catalog']).read_text(encoding='utf-8'))
    validate_fold(cfg['fold'],catalog)
    if catalog_signature(catalog)!=cfg['catalog_signature']:raise ValueError('Catalog mismatch.')
    train=RFShardDataset(cfg['catalog'],cfg['fold']['train'],training=True,seed=23)
    import numpy as np
    frozen=np.load(cfg['training_selection'],allow_pickle=False)
    if not np.array_equal(frozen,train.refs):raise ValueError('Legacy cache panel differs from frozen training refs.')
    val=RFShardDataset(cfg['catalog'],cfg['fold']['validation'])
    vp=Path(cfg['validation_selection'])
    if hashlib.sha256(vp.read_bytes()).hexdigest()!=cfg['validation_selection_sha256']:raise ValueError('Validation selection mismatch.')
    val.use_frozen_evaluation_selection(vp)
    caches={}
    for name,data in [('train',train),('validation',val)]:
        caches[name]=cache_frontend(model,data,batch_size=128,device=device,progress_path=root/'progress.json')
        torch.save(caches[name],root/f'{name}_cache.pt')
    del model
    if device=='cuda':torch.cuda.empty_cache()
    provenance=dict(config=cfg,source_checkpoint_sha256=cfg['legacy_checkpoint_sha256'],
        cached_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*_cache.pt')},outer_opened=False)
    write_json(root/'provenance.json',provenance)
    results=[]
    for name,rho,shadow in [('proposed_rho1',1,False),('proposed_rho3',3,False),('proposed_shadow_aux_rho3',3,True)]:
        results.append(train_cached_head(caches['train'],caches['validation'],mc,root/name,rho=rho,shadow=shadow,device=device))
    passed=select_stage1_candidates(*results)
    write_json(root/'stage1_result.json',dict(status='COMPLETED',passed_candidates=passed,results=results,
        automatic_next_stage=False,formal_result=False,frontend_effect_not_tested=True,
        fold=cfg['fold'],seed=23,catalog_signature=cfg['catalog_signature'],legacy_checkpoint_sha256=cfg['legacy_checkpoint_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.config,a.output)
