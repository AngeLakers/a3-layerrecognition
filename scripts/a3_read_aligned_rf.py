"""Export one derived RF record without changing its source or reading labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.data.center_alignment import AlignedRFView


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--board',required=True);p.add_argument('--row',type=int,required=True)
    p.add_argument('--mode',choices=['board','record'],default='board')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=AlignedRFView(a.catalog,mode=a.mode).read(a.board,a.row)
    arrays={key:value.pop(key) for key in ('x','source_mask','relative_sample')}
    with a.output.open('xb') as f:
        np.savez_compressed(f,**arrays,metadata_json=json.dumps(value,ensure_ascii=False))
    print(json.dumps(dict(status='EXPORTED',**value),ensure_ascii=True))


if __name__=='__main__':
    main()
