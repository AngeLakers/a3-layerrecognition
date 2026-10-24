"""Assemble immutable remote metadata/code; raw RF remains at the existing staging path."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.stream_training import catalog_signature
from a3_ultrasound.phase60 import phase_cells

REPO=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--rf-paths',type=Path,required=True)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    remote=Path(args.remote_root);rf=json.loads(args.rf_paths.read_text('utf8'));cfg=json.loads((args.data/'local_training.json').read_text('utf8'))
    for folder in ('catalog','frozen_training_selections','validation_selection'):
        shutil.copytree(args.data/folder,root/'data'/folder)
    cat=json.loads((root/'data/catalog/catalog.json').read_text('utf8'))
    for e in cat['entities']:e['rf_path']=rf[e['entity']]
    save(root/'data/catalog/catalog.json',cat)
    for p in (REPO/'src').rglob('*.py'):
        dest=root/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in (REPO/'tests').glob('test_*.py'):
        dest=root/'code/tests'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(REPO/'scripts/a3_phase60_runtime.py',root/'runtime.py')
    (root/'start_supervisor.py').write_text('''from pathlib import Path
import runpy,sys,traceback
root=Path(__file__).resolve().parent
with (root/'supervisor_launch.log').open('a',encoding='utf8',buffering=1) as log:
    sys.stdout=log;sys.stderr=log;sys.argv=[str(root/'runtime.py')]
    try:runpy.run_path(sys.argv[0],run_name='__main__')
    except BaseException:
        traceback.print_exc();raise
''','utf8')
    cfg.update(catalog=str(remote/'data/catalog/catalog.json'),
        frozen_selection_root=str(remote/'data/frozen_training_selections'),validation_selection_root=str(remote/'data/validation_selection'),
        formal_training_permitted=True,outer_evaluation_permitted=False,save_epoch_checkpoints=True,
        readiness_scope='V2 nominal labels, fresh initialization; launch additionally requires same-package reviews and remote preflight.')
    scientific={key:cfg[key] for key in ('model','training','normal_to_positive','folds','seeds','heads','statistics','evaluation_batch_size')}
    scientific['catalog_signature']=catalog_signature(cat)
    scientific['training_selections']={p.name:sha(p) for p in (root/'data/frozen_training_selections').glob('*.npy')}
    scientific['validation_manifest_sha256']=sha(root/'data/validation_selection/manifest.json')
    signature=hashlib.sha256(json.dumps(scientific,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cfg.update(protocol_signature=signature,expected_runs=60,label_version='V2')
    save(root/'config_train.json',cfg);outer=copy.deepcopy(cfg);outer['outer_evaluation_permitted']=True;save(root/'config_outer.json',outer)
    save(root/'scientific_protocol.json',scientific)
    evidence=Path('I:/programming/实验资料/A3')
    environment=root/'environment';environment.mkdir()
    source=evidence/'a3/local/remote3080ti_setup_20261015_022701'
    for name in ('ENVIRONMENT_LEDGER.md','env-spec.environment-only.json'):
        shutil.copy2(source/name,environment/name)
    expected=json.loads((evidence/'experiments/remote3080ti_stage_20261023/environment_audit.json').read_text('utf8'))
    save(environment/'expected.json',dict(hostname='ShinyRuo',python=expected['python'],versions=expected['versions']))
    jobs=[]
    for fold,seed,head in phase_cells(cfg):
        name=f'{fold}_{head}_s{seed}'
        jobs.append(dict(id=name,fold=fold,seed=seed,head=head,
            output=f'E:/programming/A3/runs/labels_v2_phase60_20261024/{name}',
            outer_output=f'E:/programming/A3/runs/labels_v2_phase60_20261024/{name}_outer'))
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    manifest=dict(schema='a3-labels-v2-phase60-v1',python='E:/programming/anaconda/envs/aislee-a3-layercongition/python.exe',
        code_root=str(remote/'code'),config=str(remote/'config_train.json'),files=files,jobs=jobs,
        output_root='E:/programming/A3/runs/labels_v2_phase60_20261024',
        original_queue='E:/programming/A3/operations/run_20261023/queue_state.json',
        protocol_signature=signature,catalog_signature=catalog_signature(cat),minimum_gpu_free_mib=5120,
        outer_after_all_training=True,old_runs_reused=False,
        runtime_scope='single GPU, 60 fresh training cells then separately invoked 60 outer evaluations, immutable receipts, no interrupted rerun')
    save(root/'manifest.json',manifest)
    with zipfile.ZipFile(root.parent/'phase60_release.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    save(root.parent/'release_identity.json',dict(protocol_signature=signature,catalog_signature=catalog_signature(cat),
        manifest_sha256=sha(root/'manifest.json'),zip_sha256=sha(root.parent/'phase60_release.zip'),files=len(files),cells=len(jobs)))
    print(json.dumps(json.loads((root.parent/'release_identity.json').read_text('utf8'))))


if __name__=='__main__':main()
