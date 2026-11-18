"""A5 G1D adapter over the immutable v1.6 fixed40 runtime."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse,copy,hashlib,importlib.util,json,os,sys,tarfile

def install(original, model_path, runtime_root):
    sys.path.insert(0,str(original/'src'))
    import a3_ultrasound.training as training
    import a3_ultrasound.stream_training as stream
    import a3_ultrasound.v16_protocol as protocol
    from a3_ultrasound.models.backbone import TsaiTimePreservingBackbone
    spec=importlib.util.spec_from_file_location('a5_g1d_model',model_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old_factory=training.build_model;old_evaluate=training.evaluate;old_loss=stream.head_training_loss;old_validator=protocol.validate_config
    def factory(config):
        if config.head_type!='g1d':return old_factory(config)
        if config.backbone!='resnet1d' or config.backbone_norm!='group8' or config.input_channels!=1:raise ValueError('G1D frozen backbone differs')
        backbone=TsaiTimePreservingBackbone(config.backbone,feature_stage=config.backbone_stage,backbone_norm=config.backbone_norm,canonical_per_trace=config.canonical_per_trace,input_channels=1)
        return module.G1DModel(backbone,attention_dim=config.attention_dim,hidden_dim=config.local_base_hidden_dim)
    def evaluate(model,*args,**kwargs):
        summary,rows=old_evaluate(model,*args,**kwargs)
        if model.head_type=='g1d':
            for row in rows:
                row['global_query_token_counts']=row.pop('indexed_local_token_counts',None)
                row['indexed_local_token_counts']=None
                row['readout']='three learned global queries; physical interface metadata ignored'
        return summary,rows
    def loss(output,batch,*,head_type,**kwargs):
        return old_loss(output,batch,head_type='local_independent' if head_type=='g1d' else head_type,**kwargs)
    def validate(cfg,*,fold_name,seed,head,require_permission=True):
        if head!='g1d':return old_validator(cfg,fold_name=fold_name,seed=seed,head=head,require_permission=require_permission)
        if seed not in (23,41,59) or fold_name not in ('F1','F2','F3','F4','F5'):raise ValueError('Unregistered A5 cell')
        auth=Path(cfg['a5_authorization']);raw=auth.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg['a5_authorization_sha256']:raise ValueError('A5 authorization changed')
        permit=json.loads(raw)
        if permit.get('task')!='A5' or permit.get('authorized') is not True:raise ValueError('A5 launch not authorized')
        reference=json.loads((runtime_root/'runtime'/f'{fold_name}_s{seed}.json').read_text())
        allowed={'heads','seeds','a5_authorization','a5_authorization_sha256','a5_stage'}
        for key,value in reference.items():
            if key in allowed:continue
            candidate=cfg.get(key)
            if key=='training' and cfg.get('a5_stage')=='smoke':
                candidate=dict(candidate);candidate['epochs']=value['epochs']
            if candidate!=value:raise ValueError('A5 differs from fixed40 '+key)
        if cfg['heads']!=['g1d'] or cfg['seeds']!=[23,41,59]:raise ValueError('A5 matrix differs')
        if cfg.get('a5_stage')=='smoke' and (fold_name!='F1' or seed!=23 or cfg['training']['epochs']!=1):raise ValueError('Unregistered A5 smoke')
        if cfg.get('a5_stage')=='full' and cfg['training']['epochs']!=40:raise ValueError('A5 must train40 epochs')
        if cfg.get('outer_evaluation_permitted') or cfg['training'].get('outer_evaluation_permitted'):raise ValueError('Outer remains closed')
    training.build_model=factory;stream.build_model=factory
    training.evaluate=evaluate;stream.evaluate=evaluate
    stream.head_training_loss=loss;protocol.validate_config=validate
    return training,stream

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--fold',required=True)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.root.resolve();args.old_root=args.old_root.resolve()
    # Preserve the original runtime's artifacts/... paths and training cwd.
    os.chdir(args.old_root/'code-direct')
    receipt=json.loads((root/'G1D_PACKAGE_RECEIPT.json').read_text())
    package=root/'g1d_code.tar'
    if hashlib.sha256(package.read_bytes()).hexdigest()!=receipt['sha256']:
        raise RuntimeError('G1D package SHA256 mismatch')
    with tarfile.open(package) as archive:
        for entry in archive.getmembers():
            if entry.isfile():
                path=root/entry.name
                expected=hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('G1D extracted code differs '+entry.name)
    training,stream=install(root/'original_fixed40',root/'g1d_model.py',args.old_root)
    cfg=json.loads((args.old_root/'runtime'/f'{args.fold}_s{args.seed}.json').read_text())
    cfg.update(heads=['g1d'],seeds=[23,41,59],a5_authorization=str(root/'A5_AUTHORIZATION.json'),a5_authorization_sha256=hashlib.sha256((root/'A5_AUTHORIZATION.json').read_bytes()).hexdigest(),a5_stage='smoke' if args.smoke else 'full')
    if args.smoke:
        cfg['training']['epochs']=1
        original_validate_training = stream._validate_training_config
        def validate_smoke_training(training_config):
            if training_config.epochs != 1:
                raise ValueError('A5 smoke must run exactly one epoch')
            # Validate the frozen 40-epoch schedule while executing its first epoch.
            return original_validate_training(replace(training_config, epochs=40))
        stream._validate_training_config = validate_smoke_training
    configs=root/'a5_configs';configs.mkdir(exist_ok=True)
    path=configs/(f'{args.fold}_s{args.seed}_'+cfg['a5_stage']+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
    outputs=root/('a5_smoke' if args.smoke else 'a5_runs');outputs.mkdir(exist_ok=True)
    stream.train(path,outputs/f'{args.fold}_s{args.seed}_g1d',fold_name=args.fold,head='g1d',seed=args.seed)

if __name__=='__main__':main()
