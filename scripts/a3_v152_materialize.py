"""Bind the frozen V1.5.2 template to existing data without enabling training."""
from pathlib import Path
import argparse,hashlib,json,re,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--epoch-root',type=Path,required=True)
    p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from a3_ultrasound.stream_training import catalog_signature
    from a3_ultrasound.freeze_protocol import verify_panels
    from a3_ultrasound.v15_protocol import validate_config,load_timebase_artifact,load_physical_artifacts
    cfg=json.loads(a.template.read_text(encoding='utf-8'))
    if cfg.get('execution_version')!='v1.5.2' or cfg.get('freeze_training_permitted') is not False:
        raise ValueError('Expected a stopped V1.5.2 template.')
    if not re.fullmatch('[0-9a-f]{40}',a.source_commit):raise ValueError('Exact source commit required.')
    cfg.update(catalog=str(a.catalog.resolve()),epoch_selection_root=str(a.epoch_root.resolve()),
        validation_selection_root=str(a.validation_root.resolve()),source_commit=a.source_commit,freeze_training_permitted=False)
    for name in ('timebase_calibration','physical_statistics','m6_decision'):
        path=Path(cfg[name+'_path']);path=path if path.is_absolute() else root/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg[name+'_sha256']:raise ValueError(name+' checksum mismatch')
        cfg[name+'_path']=str(path.resolve())
    cat=json.loads(a.catalog.read_text(encoding='utf-8'));verify_panels(cfg,catalog_signature(cat))
    load_timebase_artifact(cfg);load_physical_artifacts(cfg)
    validate_config(cfg,fold_name='F1',seed=23,head='local_independent',require_permission=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Use a new runtime config output; originals are preserved.')
    a.output.write_bytes((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status='MATERIALIZED_STOPPED',source_commit=a.source_commit,training_permitted=False,
        runtime_config=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()),ensure_ascii=True))

if __name__=='__main__':main()
