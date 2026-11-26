"""Collect five guarded v1.6 validation runs (one head/seed) into calibration JSONL."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.evaluation.v16_export import export_rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--head',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    rows=export_rows(args.runs,head=args.head,seed=args.seed)
    with open(output,'w',encoding='utf-8') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(dict(rows=len(rows),output=str(output))))

if __name__=='__main__':
    main()
