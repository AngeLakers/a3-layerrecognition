"""Local diagnostics and explicitly separated formal train/outer evaluation."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.stream_training import train,test_checkpoint
from a3_ultrasound.models.integrated import HEAD_TYPES

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['diagnostic','train','test'],required=True);p.add_argument('--fold');p.add_argument('--head',choices=HEAD_TYPES);p.add_argument('--seed',type=int,default=23);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.mode=='test':
        if a.checkpoint is None:p.error('--checkpoint required for test')
        result=test_checkpoint(a.config,a.checkpoint,a.output)
    else:
        if not a.fold or not a.head:p.error('--fold and --head required')
        result=train(a.config,a.output,fold_name=a.fold,head=a.head,seed=a.seed,diagnostic=a.mode=='diagnostic')
    print(json.dumps(result,ensure_ascii=False,indent=2))
