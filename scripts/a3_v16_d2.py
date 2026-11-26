"""V1.6 D2: pre-training M3 signal gate on each fold's validation boards.

Needs the five FITTED manifests from a3_v16_m3_fit.py. Writes the decision JSON
(referenced by dual-channel configs), the AUC table and the sampled identities.
Exit code 4 means INSUFFICIENT: stop and report before any E4/E5 training.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.evaluation.v16_d2 import run_d2
from a3_ultrasound.v16_protocol import is_v16,load_v16_timebase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--manifests',required=True,help='directory holding F1..F5 _m3_fit_manifest.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not is_v16(cfg):raise ValueError('A v1.6 configuration is required.')
    timebase=load_v16_timebase(cfg);manifests={}
    for fold in [f['fold'] for f in cfg['folds']]:
        raw=(Path(args.manifests)/f'{fold}_m3_fit_manifest.json').read_bytes()
        manifests[fold]=(json.loads(raw.decode('utf-8')),hashlib.sha256(raw).hexdigest())
    decision,rows,identities=run_d2(cfg,timebase,manifests)
    output.mkdir(parents=True)
    raw=(json.dumps(identities,ensure_ascii=False)+'\n').encode('utf-8')
    (output/'D2_sampled_identities.json').write_bytes(raw)
    decision['sampled_identities_sha256']=hashlib.sha256(raw).hexdigest()
    with open(output/'D2_auc_table.csv','w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=(json.dumps(dict(decision,rows=rows),ensure_ascii=False,indent=1,default=list)+'\n').encode('utf-8')
    (output/'D2_decision.json').write_bytes(raw)
    print(json.dumps(dict(status=decision['status'],channel=decision.get('channel'),
                          decision=str(output/'D2_decision.json'),sha256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
    if decision['status']!='DECIDED':sys.exit(4)


if __name__=='__main__':
    main()
