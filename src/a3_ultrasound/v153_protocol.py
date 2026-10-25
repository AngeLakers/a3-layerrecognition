"""User-prescribed V1.5.3 release defaults; exploratory knobs cannot drift."""
from pathlib import Path
import hashlib,json

RELEASE='v1.5.3-fixed-batch16-v1'
FROZEN_SPEED = {'workers': 2, 'prefetch_factor': 2, 'pin_memory': True, 'non_blocking': True, 'coalesced_finite': True, 'cpu_structure': True, 'validation_cache': True, 'cpu_correction': True, 'skip_minibatch_summary': True, 'fast_nominal_fixed': True, 'fused_adam': False, 'tf32': True, 'bf16_autocast': False, 'cudnn_benchmark': False, 'batch_size': 16, 'length_buckets': False}
FROZEN_BACKEND = {'dtype': 'float32', 'tf32': True, 'cudnn_benchmark': False, 'cudnn_deterministic': True, 'deterministic_algorithms': True}
FROZEN_BATCHING = {'name': 'frozen_epoch_order_v1', 'seed': 23, 'drop_last': False, 'head_independent': True}
DECISION_SHA256 = '405f579a1de0ecc22e5fe500efe0ece103a37a148657b298df70e7fe3756a0ba'

def validate_speed_config(cfg):
    if (cfg.get('speed_release')!=RELEASE or cfg.get('speed')!=FROZEN_SPEED
            or cfg.get('speed_backend_policy')!=FROZEN_BACKEND or cfg.get('speed_batching_policy')!=FROZEN_BATCHING
            or cfg.get('speed_decision_sha256')!=DECISION_SHA256
            or Path(cfg.get('speed_decision_path','')).name!='speed_decision.json'):
        raise ValueError('Configuration differs from locked V1.5.3 batch16 speed protocol')
    if cfg['training']['batch_size']!=16 or cfg['speed']['fused_adam'] or cfg['speed']['bf16_autocast']:
        raise ValueError('V1.5.3 excludes fused, BF16 and large training batches')

def verify_speed_decision(cfg):
    raw=Path(cfg['speed_decision_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DECISION_SHA256:raise ValueError('V1.5.3 speed decision changed')
    return json.loads(raw)
