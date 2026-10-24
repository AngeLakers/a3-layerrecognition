"""Audit a completed selected checkpoint on the original validation population."""
from pathlib import Path
import argparse
import json
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue-state', type=Path)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['training']['device'] != 'cpu':
        # Validate the resource gate before initializing CUDA.
        if args.queue_state is None:
            parser.error('--queue-state is required for a GPU audit')
        q = json.loads(args.queue_state.read_text(encoding='utf-8'))
        if q['status'] not in ('COMPLETED','STOPPED_AT_BOUNDARY') or any(j['status']=='RUNNING' for j in q['jobs']):
            parser.error('training queue is still active')
        if '3080 Ti' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Wrong audit GPU')
        torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    from a3_ultrasound.validation_audit import audit_validation
    result = audit_validation(args.config,args.run,args.output,queue_state=args.queue_state)
    print(json.dumps({k:result[k] for k in ('status','evaluated_records','full_minus_compact','seconds')},indent=2))
