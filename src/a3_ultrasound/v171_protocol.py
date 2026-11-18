"""V1.7.1 matrix, role views, and phase authorization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from a3_ultrasound.data.v171_cohort import file_sha

PROTOCOL = "A3-v1.7.1-fixed40"
LP = "local_pre_independent"
LOCAL = "local_independent"
GLOBAL = "g1d"
FOLDS = tuple(f"F{i}" for i in range(1, 6))
SEEDS = (23, 41, 59)
TARGETS = ("101", "001")
PATH_KEYS = (
    "catalog", "epoch_selection_root", "validation_selection_root",
    "timebase_calibration_path", "m6_decision_path", "speed_decision_path", "role_root",
)
FIXED_TRAINING = {
    "epochs": 40, "batch_size": 16, "patience": 40, "min_epochs": 15,
    "learning_rate": .001, "weight_decay": 0.0, "lr_schedule": "cosine",
    "lr_warmup_epochs": 2, "lr_min": .00001, "min_delta": .002,
    "normal_fpr_limit": .02, "selection_rule": "guarded_joint", "lambda_local": .3,
    "positive_class_weight": None, "disable_local_auxiliary": False,
    "local_aux_mode": "clean_upstream", "prefix_replacement_probability": .2,
    "scheduler_step_size": None,
}
FIXED_MODEL = {
    "architecture_version": "a3-v1.6", "backbone_stage": "final_pre_pool",
    "backbone_norm": "group8", "canonical_per_trace": False, "query_view": "position2",
    "input_channels": 1, "physical_features": False, "pre_exclude_front_end": None,
    "query_dim": 64, "attention_dim": 64, "token_dim": 64, "prefix_dim": 32,
    "pcc_hidden_dim": 64, "common_feature_dim": 64, "max_interfaces": 6, "rho": 3.0,
    "local_base_type": "mlp", "local_base_hidden_dim": 32,
    "local_prior_sigma_samples": 40.0, "local_prior_strength": .5, "pre_query_conditioned": False,
}
FIXED_SPEED = {
    "workers": 2, "prefetch_factor": 2, "pin_memory": True, "non_blocking": True,
    "coalesced_finite": True, "cpu_structure": True, "validation_cache": True,
    "cpu_correction": True, "skip_minibatch_summary": True, "fast_nominal_fixed": True,
    "fused_adam": False, "tf32": True, "bf16_autocast": False, "cudnn_benchmark": False,
    "batch_size": 16, "length_buckets": False,
}


@dataclass(frozen=True)
class Cell:
    block: str
    backbone: str
    head: str
    fold: str
    seed: int
    arm: str | None = None

    @property
    def run_id(self):
        label = {LP: "LP", GLOBAL: "G1D", LOCAL: "L-A"}[self.head]
        if self.block == "B1" and self.head == GLOBAL:
            label = "GLOBAL"
        if self.block == "B4":
            return f"v171_B4_{label}_{self.arm}_{self.fold}_s{self.seed}"
        return f"v171_{self.backbone}_{label}_{self.fold}_s{self.seed}"


def matrix():
    b1 = [Cell("B1", backbone, head, fold, seed)
          for backbone, head, fold, seed in product(("tcn", "inceptiontime"), (GLOBAL, LP), FOLDS, SEEDS)]
    b2 = [Cell("B2", "resnet1d", LOCAL, fold, seed) for fold, seed in product(FOLDS, SEEDS)]
    b4 = [Cell("B4", "resnet1d", head, fold, seed, arm)
          for head, fold, seed, arm in product((LP, GLOBAL), FOLDS[:4], SEEDS[:2], ("B", "C"))]
    return b1 + b2 + b4


def cell_identity(cfg):
    cell = Cell(**cfg["cell"])
    if cell not in matrix():
        raise ValueError(f"Unregistered V1.7.1 cell: {cell}")
    return {**cfg["cell"], "run_id": cell.run_id, "target": cfg.get("held_out_target")}


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("protocol") == PROTOCOL:
        for name in PATH_KEYS:
            if cfg.get(name):
                cfg[name] = str((path.parent / cfg[name]).resolve())
    return cfg


def dataset_row_view(cfg, split):
    identity = cell_identity(cfg)
    if identity["block"] != "B4" or split == "test":
        return None
    if split not in ("train", "validation"):
        raise ValueError("Unexpected B4 input role.")
    role = identity["arm"] if split == "train" else "validation"
    names = cfg.get("role_sha256", {})
    digest = names.get(role)
    if not cfg.get("role_root") or not digest:
        raise ValueError(f"B4 requires frozen {role} row view.")
    return {"path": str(Path(cfg["role_root"]) / f"{role}.json"), "sha256": digest,
            "role": role, "fold": identity["fold"], "target": identity["target"]}


def selection_artifacts(cfg):
    identity = cell_identity(cfg)
    epoch = Path(cfg["epoch_selection_root"]) / "manifest.json"
    validation = Path(cfg["validation_selection_root"]) / f"{identity['fold']}.npz"
    pairs = {"epoch_manifest_sha256": epoch, "validation_sha256": validation,
             "validation_manifest_sha256": validation.parent / "manifest.json"}
    out = {}
    for key, path in pairs.items():
        out[key] = file_sha(path)
        if out[key] != cfg.get(key):
            raise ValueError(f"V1.7.1 frozen selection differs: {key}")
    for split in ("train", "validation"):
        view = dataset_row_view(cfg, split)
        if view is not None:
            if file_sha(view["path"]) != view["sha256"]:
                raise ValueError(f"B4 frozen {split} view changed.")
            out[f"{split}_view_sha256"] = view["sha256"]
    return out


def _fixed(actual, expected, description):
    mismatch = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatch:
        raise ValueError(f"V1.7.1 {description} differs: {', '.join(mismatch)}")


def validate_config(cfg, *, fold_name, seed, head, phase, config_path=None,
                    authorization=None, require_permission=True):
    identity = cell_identity(cfg)
    if cfg.get("protocol") != PROTOCOL or phase not in ("train", "outer"):
        raise ValueError("Unknown V1.7.1 protocol or phase.")
    if (fold_name, seed, head) != (identity["fold"], identity["seed"], identity["head"]):
        raise ValueError("Requested arguments do not match the registered cell.")
    if require_permission:
        if not authorization or not config_path:
            raise PermissionError(f"V1.7.1 {phase} requires separate explicit authorization.")
        receipt = json.loads(Path(authorization).read_text(encoding="utf-8"))
        if (receipt.get("protocol") != PROTOCOL or receipt.get("phase") != phase
                or receipt.get("authorized") is not True
                or receipt.get("cells", {}).get(identity["run_id"]) != file_sha(config_path)):
            raise PermissionError(f"Authorization does not cover this exact {phase} cell.")
    _fixed(cfg["training"], FIXED_TRAINING, "training")
    _fixed(cfg["model"], FIXED_MODEL, "model")
    _fixed(cfg["speed"], FIXED_SPEED, "speed")
    expected_scope = "full_waveform" if identity["block"] == "B2" else "indexed"
    if (cfg["model"]["backbone"] != identity["backbone"] or cfg["model"]["readout_scope"] != expected_scope
            or cfg["training"]["seed"] != seed or cfg.get("heads") != [head] or cfg.get("seeds") != [seed]
            or cfg.get("normal_to_positive") != 3 or cfg.get("evaluation_batch_size") != 128):
        raise ValueError("V1.7.1 model, batch size or role differs from the active matrix.")
    if (cfg.get("m3_channel") != "none" or cfg.get("label_version") != "V2"
            or cfg.get("formal_training_permitted") is not True
            or cfg.get("development_stage") is not None or cfg.get("outer_evaluation_permitted") is not False
            or cfg["training"].get("outer_evaluation_permitted") is not False):
        raise ValueError("V1.7.1 training must retain V2, single RF and a closed outer phase.")
    if (cfg.get("input_tail_policy") != "nominal_fixed"
            or cfg.get("input_window_policy") != {"name": "board_timebase", "halfwidth_samples": 80}):
        raise ValueError("V1.7.1 must retain the registered 80-point nominal window.")
    if cfg.get("window_jitter") != {"enabled": True, "max_shift_samples": 24, "seed": 23, "distribution": "uniform"}:
        raise ValueError("V1.7.1 jitter differs from the fixed sampler.")
    if identity["block"] == "B4":
        if identity["target"] not in TARGETS:
            raise ValueError("B4 needs a registered held-out target.")
    elif identity["target"] is not None or cfg.get("role_root"):
        raise ValueError("B1/B2 must keep the original cohort.")
    from a3_ultrasound.stream_training import catalog_signature, validate_fold
    from a3_ultrasound.v16_protocol import load_v16_timebase, verify_m6_inactive
    catalog = json.loads(Path(cfg["catalog"]).read_text(encoding="utf-8"))
    if catalog_signature(catalog) != cfg["catalog_signature"]:
        raise ValueError("V1.7.1 input catalog differs from the formally archived input.")
    fold = next(fold for fold in cfg["folds"] if fold["fold"] == fold_name)
    validate_fold(fold, catalog)
    load_v16_timebase(cfg)
    verify_m6_inactive(cfg)
    if file_sha(cfg["speed_decision_path"]) != cfg["speed_decision_sha256"]:
        raise ValueError("V1.7.1 speed decision changed.")
    selection_artifacts(cfg)
