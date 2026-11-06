"""Calibrate one head/seed across mixed K; complete validation rows only."""
from pathlib import Path
import sys, argparse, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.evaluation.v16_temperature import mixed_specimen_oof

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSONL with prediction keys plus head/seed/fold')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    with open(args.input,encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
    if any(r.get('split')!='validation' for r in rows):raise ValueError('Explicit validation provenance required.')
    from a3_ultrasound.v16_protocol import template
    folds={f['fold']:f for f in template()['folds']}
    expected={(f,b) for f,definition in folds.items() for b in definition['validation']}
    if {(r.get('fold'),r.get('specimen_id')) for r in rows}!=expected:
        raise ValueError('All five folds and every registered validation board event required.')
    identities=set()
    for r in rows:
        if len(r['label_mask'])!=len(r['y']) or not all(type(v) is bool for v in r['label_mask']):
            raise ValueError('Boolean mask must match K.')
        if (r.get('checkpoint_kind')!='guarded' or r.get('guard_violated') is not False
                or r.get('fold') not in folds
                or r['specimen_id'] not in folds[r['fold']]['validation']):
            raise ValueError('Guarded checkpoint and registered validation membership required.')
        identity=(r['head'],r['seed'],r['fold'],r['sample_id'])
        if identity in identities:raise ValueError('Duplicate prediction within a fold.')
        identities.add(identity)
    rows=[r for r in rows if all(r['label_mask'])]
    if {(r['fold'],r['specimen_id']) for r in rows}!=expected:
        raise ValueError('Every validation board event needs complete reliable calibration records.')
    q,records,final=mixed_specimen_oof(rows)
    output.mkdir(parents=True)
    with (output/'oof_predictions.jsonl').open('w',encoding='utf-8') as f:
        for row,value in zip(rows,q):f.write(json.dumps(dict(row,calibrated_log_probabilities=value),ensure_ascii=False)+'\n')
    (output/'calibration.json').write_text(json.dumps(dict(specimens=records,all_validation_temperature_candidate=final,scope='calibration-stage OOF; no backbone or selection leakage claim; outer not evaluated or authorized'),indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':main()
