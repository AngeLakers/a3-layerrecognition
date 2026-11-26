"""V1.6 C3: fit the fixed M3 parameters per fold on TRAIN boards (no training, no GPU).

Writes <output>/<fold>_m3_fit_manifest.json (+ sampled identities and CV grid).
Exit code 3 means a family hit delta = +-24: M3 is paused for review.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.data.m3_fit import fit_fold
from a3_ultrasound.v16_protocol import is_v16,load_v16_timebase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='v1.6 runtime config with the real catalog path')
    parser.add_argument('--output',required=True)
    parser.add_argument('--folds',nargs='+',default=['F1','F2','F3','F4','F5'])
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not is_v16(cfg):raise ValueError('A v1.6 configuration is required.')
    timebase=load_v16_timebase(cfg);paused=[]
    for fold in args.folds:
        started=time.perf_counter()
        manifest,path=fit_fold(cfg,fold,timebase,args.output)
        summary={f:dict(status=p['status'],delta=p.get('delta'),alpha=p.get('alpha'),fit_records=p['fit_records'])
                 for f,p in manifest['family_params'].items()}
        print(json.dumps(dict(fold=fold,status=manifest['status'],seconds=round(time.perf_counter()-started,1),
                              manifest=str(path),families=summary),ensure_ascii=False),flush=True)
        if manifest['status']!='FITTED':paused.append(fold)
    if paused:
        print('BOUNDARY_LIMITED in '+', '.join(paused)+': M3 paused; report delta curves, do not widen.',file=sys.stderr)
        sys.exit(3)


if __name__=='__main__':
    main()
