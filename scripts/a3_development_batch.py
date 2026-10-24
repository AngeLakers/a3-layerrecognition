"""Explicit B1/B3 entrypoints; original formal training runner stays unchanged."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    c=sub.add_parser('cache');c.add_argument('--config',required=True);c.add_argument('--run',required=True)
    c.add_argument('--baseline-audit',required=True);c.add_argument('--queue',required=True);c.add_argument('--output',required=True)
    r=sub.add_parser('resample');r.add_argument('--cache-map',required=True);r.add_argument('--output',required=True)
    t=sub.add_parser('pilot');t.add_argument('--config',required=True);t.add_argument('--arm',choices=['C','L','LW'],required=True);t.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='cache':
        from a3_ultrasound.validation_sampling_audit import cache_selected
        cache_selected(a.config,a.run,a.baseline_audit,a.queue,a.output)
    elif a.command=='resample':
        from a3_ultrasound.validation_sampling_audit import resample
        resample(json.loads(Path(a.cache_map).read_text(encoding='utf-8')),a.output)
    else:
        from a3_ultrasound.optimization_pilot import train_pilot
        train_pilot(a.config,a.output,arm=a.arm)


if __name__=='__main__':main()
