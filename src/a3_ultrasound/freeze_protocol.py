"""The v1.4-freeze contract; launching remains a separate operator action."""
import hashlib
import json
import re
from pathlib import Path


PROTOCOL = 'V1.4-freeze'
HEADS = ('proposed', 'vanilla_chain', 'local_independent')


def validate_freeze_config(cfg, *, fold_name, seed, head, require_permission=True):
    t, m = cfg['training'], cfg['model']
    fixed = dict(epochs=40, min_epochs=15, patience=10, min_delta=.002,
        normal_fpr_limit=.02, selection_rule='guarded_joint', batch_size=16,
        learning_rate=.001, weight_decay=0., lambda_local=.3,
        scheduler_step_size=None, disable_local_auxiliary=False, positive_class_weight=None,
        outer_evaluation_permitted=False, device='cuda', real_training_permitted=True)
    if (cfg.get('protocol') != PROTOCOL or cfg.get('heads') != list(HEADS)
            or cfg.get('development_stage') != 'v1.4-freeze'
            or cfg.get('seeds') != [23] or fold_name != 'F1' or seed != 23 or head not in HEADS
            or len(cfg.get('folds', [])) != 1 or cfg['folds'][0]['fold'] != 'F1'
            or cfg.get('outer_evaluation_permitted') is not False
            or cfg.get('formal_training_permitted') is not False
            or cfg.get('automatic_screening') is not False
            or cfg.get('normal_to_positive') != 3 or cfg.get('label_version') != 'V2'
            or cfg.get('evaluation_batch_size') != 128 or not cfg.get('save_epoch_checkpoints')
            or not cfg.get('save_best_correction_npz')
            or m.get('architecture_version') != 'a3-v1.4' or m.get('backbone') != 'resnet1d'
            or m.get('backbone_norm') != 'group8' or m.get('query_view') != 'position2' or m.get('rho') != 3.
            or any(t.get(k) != v for k, v in fixed.items())):
        raise ValueError('Configuration differs from v1.4-freeze.')
    if any('checkpoint' in k.lower() and k != 'save_epoch_checkpoints' for k in cfg):
        raise ValueError('Frozen runs always start from fresh weights.')
    if require_permission and cfg.get('freeze_training_permitted') is not True:
        raise ValueError('Waiting for the user to explicitly restart training.')


def verify_panels(cfg, catalog_signature):
    if cfg.get('catalog_signature') != catalog_signature:
        raise ValueError('Frozen catalog scientific identity changed.')
    root = Path(cfg['epoch_selection_root'])
    manifest_path = root/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != cfg['epoch_manifest_sha256']:
        raise ValueError('Frozen epoch manifest changed.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['positive_fraction'] != 1. or manifest['epochs'] != 40 or manifest['seed'] != 23
            or manifest['normal_ratio'] != 3 or manifest['catalog_signature'] != catalog_signature):
        raise ValueError('Frozen runs require all positives and 40 shared 3:1 epoch selections.')
    validation = Path(cfg['validation_selection_root'])/'F1.npz'
    if hashlib.sha256(validation.read_bytes()).hexdigest() != cfg['validation_sha256']:
        raise ValueError('Frozen validation panel changed.')


def commit_identity(cfg, code_version):
    actual = code_version.get('commit')
    expected = cfg.get('source_commit')
    if not actual or not re.fullmatch(r'[0-9a-f]{40}', actual) or actual != expected:
        raise ValueError('Frozen source commit identity is missing or mismatched.')
    return actual
