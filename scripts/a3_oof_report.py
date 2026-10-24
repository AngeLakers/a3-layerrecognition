"""Aggregate the complete frozen outer matrix without launching training."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.oof_report import summarize_oof
from a3_ultrasound.stream_training import catalog_signature

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--outer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    files=sorted(a.outer_root.glob('*/summary.json'))
    signature=catalog_signature(json.loads(Path(cfg['catalog']).read_text(encoding='utf-8')))
    result=summarize_oof(cfg,[json.loads(f.read_text(encoding='utf-8')) for f in files],expected_catalog_signature=signature)
    with a.output.open('x',encoding='utf-8') as handle:json.dump(dict(result,sources=[str(f) for f in files]),handle,ensure_ascii=False,indent=2,allow_nan=False)
