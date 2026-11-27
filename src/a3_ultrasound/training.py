"""Small common training/evaluation runner for the four A3 V1.3 Heads.

Input tensors are already phase-preserving preprocessed and metadata-normalized.
Sample/specimen identifiers remain evaluation-side data, never model features.
"""

from __future__ import annotations

import copy
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from a3_ultrasound.models.backbone import LengthGroupedTCNBackbone, TsaiTimePreservingBackbone
from a3_ultrasound.models.integrated import A3Model, HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss, validate_complete_training_batch
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.selection import GuardedSelector
from a3_ultrasound.stream_metrics import PredictionAccumulator
from a3_ultrasound.prefix_perturbation import training_conditional


@dataclass(frozen=True)
class ModelConfig:
    architecture_version: str = "a3-v1.3"
    backbone_norm: str = "batch"
    canonical_per_trace: bool = True
    query_view: str = "legacy3"
    head_type: str = "proposed"
    backbone: str = "resnet1d"
    backbone_stage: str = "final_pre_pool"
    query_dim: int = 32
    attention_dim: int = 64
    token_dim: int = 32
    prefix_dim: int = 32
    pcc_hidden_dim: int = 64
    common_feature_dim: int = 64
    max_interfaces: int = 6
    rho: float = 1.0
    local_base_type: str = "mlp"
    local_base_hidden_dim: int = 32
    readout_scope: str = "indexed"
    relative_time_attention: bool = False
    pre_query_conditioned: bool = False
    local_prior_sigma_samples: float | None = None
    local_prior_strength: float = 1.0
    physical_features: bool = False
    pre_exclude_front_end: int | None = None
    reference_bias: str = "none"
    input_channels: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 23
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_local: float = 0.3
    positive_class_weight: float | None = None
    disable_local_auxiliary: bool = False
    patience: int = 5
    scheduler_step_size: int | None = None
    scheduler_gamma: float = 0.5
    device: str = "cpu"
    real_training_permitted: bool = False
    selection_rule: str = "legacy_deepest"
    min_epochs: int = 15
    min_delta: float = 0.002
    normal_fpr_limit: float = 0.01
    outer_evaluation_permitted: bool = False
    local_aux_mode: str | None = None
    prefix_replacement_probability: float = 0.0
    lr_schedule: str | None = None
    lr_warmup_epochs: int = 0
    lr_min: float = 0.0
    v185_loss: str = "nll"
    layer_weights: tuple[float, ...] | None = None


def scheduled_learning_rate(config: TrainingConfig, progress: float) -> float:
    """Learning rate at fractional epoch progress u in [0, epochs).

    None keeps the historical constant rate. 'cosine': linear warmup over W
    epochs (lr0*(u+1)/(W+1)) to lr0, then lr_min + (lr0-lr_min)/2*(1+cos(pi*(u-W)/(E-W))).
    Depends only on (epoch, step), never on validation, so all Heads share it.
    """
    lr0=config.learning_rate
    if config.lr_schedule is None:return lr0
    w=config.lr_warmup_epochs;e=config.epochs
    if progress<w:return lr0*(progress+1)/(w+1)
    ratio=min(1.,(progress-w)/max(e-w,1e-12))
    return config.lr_min+(lr0-config.lr_min)*.5*(1+math.cos(math.pi*ratio))


def build_model(config: ModelConfig, *, v171: bool = False) -> A3Model:
    if config.architecture_version not in ("a3-v1.3", "a3-v1.4", "a3-v1.5", "a3-v1.6", "a3-v1.8"):
        raise ValueError("Unknown architecture_version.")
    if config.architecture_version in ("a3-v1.4", "a3-v1.5", "a3-v1.6", "a3-v1.8") and (config.backbone_norm != "group8" or config.query_view != "position2"):
        raise ValueError("V1.4 requires group8 and position2.")
    if type(config.input_channels) is not int or config.input_channels not in (1, 2):
        raise ValueError("input_channels must be 1 or 2.")
    if config.architecture_version != "a3-v1.6" and config.input_channels != 1:
        raise ValueError("Legacy architectures require one channel.")
    v18 = config.architecture_version == "a3-v1.8"
    if config.head_type in ("state_conditioned_gru", "waveform_only_gru") and not v18:
        raise ValueError("The state-conditioned GRU requires a3-v1.8.")
    v18_heads = ("local_pre_independent", "state_conditioned_gru", "waveform_only_gru")
    if config.relative_time_attention:
        if not v18 or config.backbone not in ("resnet1d", "patchtst", "tcn", "inceptiontime") or config.head_type != "local_independent":
            raise ValueError("M1 relative time requires a registered v1.8 temporal encoder and independent head.")
        v18_heads += ("local_independent",)
    if v18 and (config.physical_features or config.head_type not in v18_heads):
        raise ValueError("V1.8 supports the single-RF LP and state-conditioned GRU pair.")
    global_head = v171 and config.head_type == 'g1d'
    if config.architecture_version == "a3-v1.6":
        legacy_head = config.head_type in ("local_independent", "local_pre_independent", "vanilla_chain")
        if config.physical_features or not (legacy_head or global_head):
            raise ValueError("V1.6 permits only L/LP/V without physical features.")
    if config.head_type not in HEAD_TYPES and not global_head:
        raise ValueError("Unknown head_type.")
    if config.rho not in (1.0, 2.0, 3.0):
        raise ValueError("rho must use the registered candidates 1, 2, 3.")
    wide_readout = config.relative_time_attention and config.readout_scope == "full_waveform"
    allowed_feature_dims = (64, 128) if wide_readout else (64,)
    if config.common_feature_dim not in allowed_feature_dims:
        raise ValueError(f"The common adapter width must be one of {allowed_feature_dims}.")
    if config.backbone == "patchtst":
        if not v18 or config.backbone_stage != "final_pre_pool":
            raise ValueError("PatchTST requires v1.8 pre-head temporal tokens.")
        from a3_ultrasound.models.patchtst import PatchTSTBackbone
        backbone = PatchTSTBackbone()
    elif config.backbone == "tcn":
        if config.backbone_stage != "final_pre_pool":
            raise ValueError("TCN smoke supports final_pre_pool only.")
        if v171 or v18:
            backbone = LengthGroupedTCNBackbone(channels=(128, 128, 128), kernel_size=5,
                                               dilations=(1, 2, 4), use_norm='group8', input_channels=1)
        else:
            backbone = LengthGroupedTCNBackbone(channels=(8, 8), kernel_size=3,
                                               dilations=(1, 2), use_norm=None, input_channels=config.input_channels)
    else:
        backbone = TsaiTimePreservingBackbone(config.backbone, feature_stage=config.backbone_stage,
                                              backbone_norm=config.backbone_norm,
                                              canonical_per_trace=config.canonical_per_trace, input_channels=config.input_channels)
    if global_head:
        from a3_ultrasound.models.g1d import G1DModel
        return G1DModel(backbone, attention_dim=config.attention_dim, hidden_dim=config.local_base_hidden_dim)
    kwargs = asdict(config)
    kwargs.pop("input_channels")
    kwargs.pop("backbone")
    kwargs.pop("backbone_stage")
    kwargs.pop("backbone_norm")
    kwargs.pop("canonical_per_trace")
    kwargs.pop("architecture_version")
    return A3Model(backbone=backbone, **kwargs)


@dataclass(frozen=True)
class TensorDataset:
    batch: InterfaceBatch
    sample_ids: tuple[str, ...]
    specimen_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n = self.batch.x_raw.shape[0]
        if n == 0 or len(self.sample_ids) != n or len(self.specimen_ids) != n:
            raise ValueError("A nonempty dataset needs one sample_id and specimen_id per row.")
        if len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be unique within a split.")
        if any(not isinstance(value, str) or not value.strip() for value in (*self.sample_ids, *self.specimen_ids)):
            raise ValueError("Sample/specimen IDs must be nonempty strings.")

    def batches(self, size: int, device: str, *, generator: torch.Generator | None = None):
        n = len(self.sample_ids)
        indices = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
        for start in range(0, n, size):
            selected = indices[start:start + size]
            tensors = {field.name: (None if (value := getattr(self.batch, field.name)) is None
                                    else value[selected].to(device)) for field in fields(InterfaceBatch)}
            yield InterfaceBatch(**tensors), selected.tolist()


def audit_splits(datasets: dict[str, TensorDataset]) -> dict[str, Any]:
    if set(datasets) != {"train", "validation", "test"}:
        raise ValueError("Exactly train, validation and test splits are required.")
    names = tuple(datasets)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(datasets[name].specimen_ids) & set(datasets[other].specimen_ids):
                raise ValueError(f"specimen leakage between {name} and {other}.")
            if set(datasets[name].sample_ids) & set(datasets[other].sample_ids):
                raise ValueError(f"sample leakage between {name} and {other}.")
    return {name: {"samples": len(data.sample_ids), "specimens": len(set(data.specimen_ids))}
            for name, data in datasets.items()}


def build_optimizer(model: A3Model, config: TrainingConfig, *, fused: bool = False) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    """Cover each trainable parameter once, including adapter/readouts/new Head."""
    groups = []
    counts = {}
    parameter_ids = []
    for name, child in model.named_children():
        parameters = [parameter for parameter in child.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "name": name})
            counts[name] = sum(parameter.numel() for parameter in parameters)
            parameter_ids.extend(map(id, parameters))
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(parameter_ids) != expected or len(parameter_ids) != len(expected):
        raise RuntimeError("Optimizer groups must cover every trainable parameter exactly once.")
    kwargs={'fused':True} if fused else {}
    return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay,**kwargs), counts


def _binary_metrics(truth: list[int], prediction: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not truth:
        return {"count": 0, "macro_f1": None, "accuracy": None, "brier": None, "marginal_nll": None}
    y, pred, prob = np.asarray(truth), np.asarray(prediction), np.asarray(probabilities)
    f1 = []
    for label in (0, 1):
        tp = int(((y == label) & (pred == label)).sum())
        fp = int(((y != label) & (pred == label)).sum())
        fn = int(((y == label) & (pred != label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return {"count": len(truth), "macro_f1": float(np.mean(f1)),
            "accuracy": float((y == pred).mean()), "brier": float(((prob - y) ** 2).mean()),
            "marginal_nll": float(-(y * np.log(prob) + (1 - y) * np.log1p(-prob)).mean()),
            "confusion_matrix_0_1": [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]}


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interfaces: dict[str, list[tuple[int, int, float]]] = {}
    deepest: dict[str, list[tuple[int, int, float]]] = {}
    upper_groups: dict[str, list[tuple[int, int, float]]] = {}
    joint_correct, joint_nll, base_errors, bound_errors = [], [], 0, 0
    wrong_to_right = right_to_wrong = labeled = 0
    for row in rows:
        y, mask, p = row["y"], row["label_mask"], row["marginals"]
        if y is None:
            continue
        for k, reliable in enumerate(mask):
            if reliable:
                value = (y[k], int(p[k] >= 0.5), p[k])
                interfaces.setdefault(str(k + 1), []).append(value)
                base_wrong = int(row["base_probabilities"][k] >= 0.5) != y[k]
                final_wrong = value[1] != y[k]
                labeled += 1
                base_errors += int(base_wrong)
                wrong_to_right += int(base_wrong and not final_wrong)
                right_to_wrong += int(not base_wrong and final_wrong)
                if row["rho"] is not None:
                    bound_errors += int(base_wrong and abs(row["base_logodds"][k]) > row["rho"])
        if mask[-1]:
            value = (y[-1], int(p[-1] >= 0.5), p[-1])
            deepest.setdefault(row["specimen_id"], []).append(value)
            if all(mask[:-1]):
                key = f"K{len(y)}:" + "".join(map(str, y[:-1]))
                upper_groups.setdefault(key, []).append(value)
        if all(mask):
            joint_correct.append(row["map_states"] == y)
            path_index = sum(state << (len(y) - k - 1) for k, state in enumerate(y))
            joint_nll.append(-row["log_probabilities"][path_index])
    def metrics(values):
        return _binary_metrics(*map(list, zip(*values)))
    per_specimen = {key: metrics(value) for key, value in deepest.items()}
    correction_values = [value for row in rows for value in (row["gate_times_correction"] or [])[1:]]
    saturation = [abs(value) >= 0.95 * row["rho"] for row in rows
                  if row["rho"] is not None for value in row["correction"][1:]]
    prefix_ranges = [value for row in rows for value in row["conditional_probability_range"][1:]]
    return {
        "sample_count": len(rows), "labeled_interface_count": labeled,
        "per_interface": {key: metrics(value) for key, value in interfaces.items()},
        "deepest_per_specimen": per_specimen,
        "deepest_specimen_macro_f1": float(np.mean([m["macro_f1"] for m in per_specimen.values()])) if per_specimen else None,
        "deepest_by_true_upper_states": {key: metrics(value) for key, value in upper_groups.items()},
        "joint_complete_label_count": len(joint_correct),
        "joint_map_accuracy": float(np.mean(joint_correct)) if joint_correct else None,
        "joint_nll": float(np.mean(joint_nll)) if joint_nll else None,
        "base_wrong_to_exact_right": wrong_to_right, "base_right_to_exact_wrong": right_to_wrong,
        "base_error_count": base_errors,
        "base_wrong_beyond_rho_count": bound_errors if any(row["rho"] is not None for row in rows) else None,
        "base_wrong_beyond_rho_fraction_of_errors": bound_errors / base_errors if base_errors and any(row["rho"] is not None for row in rows) else None,
        "raw_logZ_max_abs": max(abs(row["raw_logZ"]) for row in rows),
        "gate_times_correction_quantiles_MAP_path": dict(zip(("min", "q25", "median", "q75", "max"),
                                                            np.quantile(correction_values, [0, .25, .5, .75, 1]).tolist())) if correction_values else None,
        "correction_saturation_fraction_MAP_path": float(np.mean(saturation)) if saturation else None,
        "conditional_probability_range_mean_nonfirst": float(np.mean(prefix_ranges)) if prefix_ranges else None,
        "threshold": 0.5, "macro_f1_labels": [0, 1], "macro_f1_zero_division": 0,
    }


@torch.no_grad()
def evaluate(model: A3Model, data: TensorDataset, *, batch_size: int = 16,
             device: str = "cpu", speed_options=None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """All predictions use exact inference. Labels are consulted only for metrics."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    was_training = model.training
    model.eval()
    rows = []
    try:
        from a3_ultrasound.speed import speed_context,precision_context
        optimized=speed_options is not None and speed_options.cpu_structure
        batches=[(data.batch,list(range(len(data.sample_ids))))] if optimized and batch_size>=len(data.sample_ids) else data.batches(batch_size,device)
        for batch, indices in batches:
            with speed_context(speed_options,batch):
                with precision_context(speed_options):encoded = model.encode(batch)
                exact = model.infer_encoded(encoded)
                # MAP-path correction is a prediction-side diagnostic, not truth forcing.
                diagnostic = model.conditional(encoded, exact.map_states)
            if not optimized and batch.label_mask is not None and torch.any(batch.label_mask & ~batch.interface_mask):
                raise ValueError("label_mask cannot supervise padded interfaces.")
            # Copy each field once per batch instead of synchronizing the GPU
            # for every scalar/row. GPU arithmetic and row order stay unchanged.
            base_probabilities = diagnostic.base_logodds.sigmoid()
            gate_correction = diagnostic.gate * diagnostic.correction if diagnostic.gate is not None else None
            raw_gate_correction = gate_correction
            if diagnostic.upstream_flag is not None:
                gate_correction = gate_correction * diagnostic.upstream_flag
            tensors = (batch.y, batch.label_mask, exact.marginals, exact.map_states,
                       diagnostic.base_logodds, base_probabilities,
                       exact.conditional_probability_range, encoded.local_token_diagnostics.counts,
                       diagnostic.gate, diagnostic.correction, gate_correction,
                       raw_gate_correction, diagnostic.upstream_flag)
            cpu_fields = {id(tensor): tensor.cpu() for tensor in tensors if tensor is not None}
            counts = batch.interface_mask.sum(1).cpu().tolist()
            path_masks = exact.path_mask.cpu()
            probabilities = exact.probabilities.cpu()
            log_probabilities = exact.log_probabilities.cpu()
            raw_logz = exact.raw_log_normalizer.cpu().tolist()
            for i, source_index in enumerate(indices):
                k = counts[i]
                valid_paths = path_masks[i]
                def values(tensor):
                    return cpu_fields[id(tensor)][i, :k].tolist()
                y = values(batch.y) if batch.y is not None else None
                label_mask = values(batch.label_mask) if batch.label_mask is not None else [y is not None] * k
                if y is not None and any(reliable and state not in (0, 1) for state, reliable in zip(y, label_mask)):
                    raise ValueError("Reliable evaluation labels must be binary.")
                row = {
                    "sample_id": data.sample_ids[source_index], "specimen_id": data.specimen_ids[source_index],
                    "K": k, "y": y, "label_mask": label_mask,
                    "marginals": values(exact.marginals), "map_states": values(exact.map_states),
                    "probabilities": probabilities[i, valid_paths].tolist(),
                    "log_probabilities": log_probabilities[i, valid_paths].tolist(),
                    "base_logodds": values(diagnostic.base_logodds),
                    "base_probabilities": values(base_probabilities),
                    "raw_logZ": raw_logz[i],
                    "conditional_probability_range": values(exact.conditional_probability_range),
                    "indexed_local_token_counts": values(encoded.local_token_diagnostics.counts),
                    "rho": getattr(model.head, "rho", None),
                    "correction_diagnostic_path": "inferred_joint_MAP" if diagnostic.gate is not None else None,
                    "gate": values(diagnostic.gate) if diagnostic.gate is not None else None,
                    "correction": values(diagnostic.correction) if diagnostic.correction is not None else None,
                    "gate_times_correction": values(gate_correction) if gate_correction is not None else None,
                }
                if diagnostic.correction is not None:
                    eligible = (cpu_fields[id(diagnostic.correction)] if speed_options is not None and speed_options.cpu_correction else diagnostic.correction)[i, 1:k]
                    row["correction_saturation_fraction"] = (float((eligible.abs() >= 0.95 * model.head.rho).float().mean())
                                                             if eligible.numel() else None)
                if diagnostic.upstream_flag is not None:
                    row['upstream_flag'] = values(diagnostic.upstream_flag)
                    row['gate_times_raw_correction'] = values(raw_gate_correction)
                rows.append(row)
        if speed_options is not None and speed_options.skip_minibatch_summary:return {},rows
        metrics=summarize_predictions(rows)
        accumulator=PredictionAccumulator()
        for row in rows:accumulator.add(row)
        metrics.update(accumulator.result())
        return metrics, rows
    finally:
        model.train(was_training)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _code_version() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_path_count": len(status)}
    except (OSError, subprocess.CalledProcessError):
        revision = repository / 'SOURCE_REVISION.json'
        if revision.is_file():
            value = json.loads(revision.read_text(encoding='utf-8'))
            return {"commit": value.get("commit"), "dirty": bool(value.get("dirty",False)),
                    "status": "uncommitted source snapshot metadata" if value.get("dirty",False) else "frozen SOURCE_REVISION metadata"}
        return {"commit": None, "dirty": None, "status": "git unavailable"}


def _validate_training_config(config: TrainingConfig) -> None:
    if config.local_aux_mode not in (None, 'all', 'clean_upstream'):
        raise ValueError('Unknown local_aux_mode.')
    if not math.isfinite(config.prefix_replacement_probability) or not 0 <= config.prefix_replacement_probability <= 1:
        raise ValueError('prefix_replacement_probability must be in [0,1].')
    if config.selection_rule not in ("legacy_deepest", "guarded_joint"):
        raise ValueError("Unknown selection rule.")
    GuardedSelector(config.min_epochs,config.patience,config.min_delta,config.normal_fpr_limit)
    for name in ("epochs", "batch_size", "patience"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if config.scheduler_step_size is not None and (type(config.scheduler_step_size) is not int or config.scheduler_step_size < 1):
        raise ValueError("scheduler_step_size must be a positive integer or null.")
    if not math.isfinite(config.scheduler_gamma) or not 0 < config.scheduler_gamma <= 1:
        raise ValueError("scheduler_gamma must lie in (0, 1].")
    if config.lr_schedule not in (None, "cosine"):
        raise ValueError("lr_schedule must be null or cosine.")
    if type(config.lr_warmup_epochs) is not int or not 0 <= config.lr_warmup_epochs < config.epochs:
        raise ValueError("lr_warmup_epochs must be an integer in [0, epochs).")
    if not math.isfinite(config.lr_min) or not 0 <= config.lr_min < config.learning_rate:
        raise ValueError("lr_min must lie in [0, learning_rate).")
    if config.lr_schedule is not None and config.scheduler_step_size is not None:
        raise ValueError("Use either StepLR or lr_schedule, not both.")


def run_training(model_config: ModelConfig, config: TrainingConfig,
                 datasets: dict[str, TensorDataset], output_dir: Path, *,
                 data_kind: str, input_manifest: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    _validate_training_config(config)
    if data_kind not in ("synthetic", "real"):
        raise ValueError("data_kind must be synthetic or real.")
    if data_kind == "real":
        if not config.real_training_permitted:
            raise ValueError("Real training remains GATED; explicit approved config is required.")
        if model_config.backbone == "tcn":
            raise ValueError("TCN is smoke/debug-only, not a formal training Backbone.")
    audit = audit_splits(datasets)
    # Only teacher-forced training requires complete labels. Test/inference may
    # be unlabeled; validation needs a reliable deepest label for model selection.
    validate_complete_training_batch(datasets["train"].batch)
    validation_batch = datasets["validation"].batch
    if validation_batch.y is None:
        raise ValueError("Validation requires reliable deepest-interface labels for model selection.")
    validation_mask = (validation_batch.interface_mask if validation_batch.label_mask is None
                       else validation_batch.label_mask & validation_batch.interface_mask)
    last = validation_batch.interface_mask.sum(dim=1) - 1
    if torch.any(last < 0) or not validation_mask.gather(1, last[:, None]).any():
        raise ValueError("Validation requires reliable deepest-interface labels for model selection.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = build_model(model_config).to(config.device)
    # Validate loss configuration before creating any run artifacts.
    first_batch, _ = next(datasets["train"].batches(config.batch_size, config.device))
    model.eval()
    with torch.no_grad():
        first_output = model(first_batch, first_batch.y).conditional
        head_training_loss(first_output, first_batch, head_type=model.head_type,
                           lambda_local=config.lambda_local, positive_class_weight=config.positive_class_weight,
                           disable_local_auxiliary=config.disable_local_auxiliary,
                           local_aux_mode=config.local_aux_mode, v185_loss=config.v185_loss, layer_weights=config.layer_weights)
    optimizer, parameter_counts = build_optimizer(model, config)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_step_size, config.scheduler_gamma)
                 if config.scheduler_step_size is not None else None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = {"architecture_version": model_config.architecture_version, "model": asdict(model_config), "training": asdict(config),
                "data_kind": data_kind, "effective_lambda_local": 0.0 if model.head_type == "local_independent" else config.lambda_local,
                "optimizer": "AdamW", "scheduler": "StepLR" if scheduler else None,
                "selection": "validation deepest specimen macro-F1, tie: lower joint NLL",
                "parameter_counts": parameter_counts, "total_parameters": sum(parameter_counts.values()),
                "input_manifest": input_manifest, "split_audit": audit,
                "split_membership": {name: {"sample_ids": list(data.sample_ids),
                                             "specimen_ids": sorted(set(data.specimen_ids))} for name, data in datasets.items()},
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    resolved["code_version"] = _code_version()
    resolved["python_version"] = sys.version.split()[0]
    _write_json(output_dir / "resolved_config.json", resolved)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    best_key = (-math.inf, -math.inf)
    stale = 0
    selector = GuardedSelector(config.min_epochs,config.patience,config.min_delta,config.normal_fpr_limit) if config.selection_rule == "guarded_joint" else None
    resolved["selection"] = config.selection_rule
    _write_json(output_dir / "resolved_config.json", resolved)
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            total = chain_total = local_total = 0.0
            count = 0
            perturbation = dict(attempted=0, changed=0, eligible=0)
            for step, (batch, indices) in enumerate(datasets["train"].batches(config.batch_size, config.device, generator=generator)):
                validate_complete_training_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                output, attempt = training_conditional(model, batch,
                    probability=config.prefix_replacement_probability, seed=config.seed, epoch=epoch, step=step)
                for name in perturbation:
                    perturbation[name] += attempt[name]
                loss = head_training_loss(output, batch, head_type=model.head_type,
                                          lambda_local=config.lambda_local, positive_class_weight=config.positive_class_weight,
                                          disable_local_auxiliary=config.disable_local_auxiliary,
                                          local_aux_mode=config.local_aux_mode, v185_loss=config.v185_loss, layer_weights=config.layer_weights)
                if not torch.isfinite(loss.total):
                    raise RuntimeError("Non-finite training loss.")
                loss.total.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise RuntimeError("Non-finite parameter gradient.")
                optimizer.step()
                n = len(indices)
                total += float(loss.total.detach()) * n
                chain_total += float(loss.chain.detach()) * n
                local_total += float(loss.local.detach()) * n
                count += n
            validation, _ = evaluate(model, datasets["validation"], batch_size=config.batch_size, device=config.device)
            key = (validation["deepest_specimen_macro_f1"],
                   -validation["joint_nll"] if validation["joint_nll"] is not None else -math.inf)
            record = {"epoch": epoch, "total_loss": total / count, "main_loss": chain_total / count,
                      "local_loss": local_total / count, "learning_rate": optimizer.param_groups[0]["lr"],
                      "validation": validation}
            if config.prefix_replacement_probability:
                record['prefix_perturbation'] = perturbation
            if selector:
                record["selection"] = selector.update(epoch,validation)
                payload = {"architecture_version":model_config.architecture_version,"epoch":epoch,
                    "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                    "resolved_config":resolved,"validation":validation,"selection_state":selector.state_dict()}
                if record["selection"]["guarded_updated"]:torch.save(payload,output_dir/"best_guarded.pt")
                if record["selection"]["fallback_updated"]:torch.save(payload,output_dir/"best_fallback.pt")
            history.append(record)
            _write_json(output_dir / "history.json", history)
            if scheduler is not None:
                scheduler.step()
            if selector is None and key > best_key:
                best_key, stale = key, 0
                checkpoint = {"architecture_version": model_config.architecture_version, "epoch": epoch,
                              "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                              "resolved_config": resolved, "validation": validation}
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                stale += 1
            if (record["selection"]["stop"] if selector else stale >= config.patience):
                break
        selected_name = ("best_guarded.pt" if selector.guarded else "best_fallback.pt") if selector else "best.pt"
        checkpoint = torch.load(output_dir / selected_name, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = {}
        for name in (("validation", "test") if data_kind == "synthetic" or config.outer_evaluation_permitted else ("validation",)):
            metrics[name], rows = evaluate(model, datasets[name], batch_size=config.batch_size, device=config.device)
            _write_predictions(output_dir / f"{name}_predictions.jsonl", rows)
        summary = {"status": "completed", "data_kind": data_kind, "best_epoch": checkpoint["epoch"],
                   "epochs_run": len(history), "metrics": metrics,
                   "wall_time_seconds": time.perf_counter() - started,
                   "performance_claims_released": False, "real_training_gate_changed": False}
        summary.update(selected_checkpoint=selected_name, guard_violated=selector.guarded is None if selector else None,
                       test_opened="test" in metrics)
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        _write_json(output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def load_tensor_manifest(path: Path) -> tuple[dict[str, TensorDataset], dict[str, Any]]:
    """NPZ fields follow InterfaceBatch; IDs are separate Unicode arrays, no pickle."""
    path = Path(path)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "a3-tensor-dataset-v1":
        raise ValueError("Expected a3-tensor-dataset-v1 manifest.")
    if not manifest.get("preprocessing_id") or not manifest.get("split_id"):
        raise ValueError("A versioned preprocessing_id and split_id are required.")
    datasets = {}
    for name, relative_path in manifest["splits"].items():
        with np.load(path.parent / relative_path, allow_pickle=False) as arrays:
            required = {"x_raw", "length_mask", "interface_meta", "interface_mask", "sample_ids", "specimen_ids"}
            if not required.issubset(arrays.files):
                raise ValueError(f"Missing tensor dataset fields: {sorted(required - set(arrays.files))}.")
            kwargs = {field.name: torch.from_numpy(arrays[field.name].copy()) if field.name in arrays else None
                      for field in fields(InterfaceBatch)}
            datasets[name] = TensorDataset(InterfaceBatch(**kwargs), tuple(arrays["sample_ids"].tolist()),
                                           tuple(arrays["specimen_ids"].tolist()))
    audit_splits(datasets)
    return datasets, manifest


def synthetic_datasets(seed: int, samples_per_split: int = 8) -> dict[str, TensorDataset]:
    """Explicit toy signals only: mixed K=2/3 and distinct synthetic specimens."""
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name in ("train", "validation", "test"):
        n, t, kmax = samples_per_split, 64, 3
        x = torch.randn(n, t, generator=generator) * 0.1
        mask = torch.arange(kmax).unsqueeze(0) < torch.tensor([2 + i % 2 for i in range(n)]).unsqueeze(1)
        lengths = torch.tensor([56 + 8 * (i % 2) for i in range(n)])
        length_mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        meta = torch.zeros(n, kmax, 5)
        y = torch.randint(0, 2, (n, kmax), generator=generator)
        y[~mask] = -1
        for i in range(n):
            k = int(mask[i].sum())
            for j in range(k):
                tau = 12 + j * 18
                meta[i, j] = torch.tensor([(j + 1) / k, tau / lengths[i], (j + 1) / (k + 1),
                                           3 / lengths[i], 3 / lengths[i]])
                x[i, tau - 2:tau + 3] += (2 * y[i, j] - 1) * torch.tensor([0.2, -0.5, 1.0, -0.5, 0.2])
        result[name] = TensorDataset(InterfaceBatch(x, length_mask, meta, mask, mask.clone(), y),
                                    tuple(f"synthetic-{name}-{i}" for i in range(n)),
                                    tuple(f"synthetic-{name}-specimen-{i % 2}" for i in range(n)))
    return result


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[A3Model, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture_version") not in ("a3-v1.3", "a3-v1.4", "a3-v1.5", "a3-v1.6", "a3-v1.8"):
        raise ValueError("Expected a V1.3 checkpoint from the common runner.")
    model = build_model(ModelConfig(**checkpoint["resolved_config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_from_config(path: Path, output_dir: Path, *, head_type: str | None = None,
                    checkpoint_path: Path | None = None, evaluate_split: str = "test") -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") not in ("a3-run-v1.3", "a3-run-v1.4"):
        raise ValueError("Expected a3-run-v1.3 config.")
    model_values = copy.deepcopy(config.get("model", {}))
    if head_type is not None:
        model_values["head_type"] = head_type
    model_config = ModelConfig(**model_values)
    training_config = TrainingConfig(**config.get("training", {}))
    data_config = config["data"]
    if data_config["source"] == "synthetic":
        datasets = synthetic_datasets(training_config.seed, data_config.get("samples_per_split", 8))
        manifest = {"data_kind": "synthetic", "source": "a3.training.synthetic_datasets",
                    "seed": training_config.seed, "samples_per_split": data_config.get("samples_per_split", 8),
                    "preprocessing_id": "synthetic-identity-v1", "split_id": "synthetic-disjoint-v1"}
    elif data_config["source"] == "tensor_manifest":
        datasets, manifest = load_tensor_manifest(path.parent / data_config["manifest"])
    else:
        raise ValueError("data.source must be synthetic or tensor_manifest.")
    if checkpoint_path is None:
        return run_training(model_config, training_config, datasets, output_dir,
                            data_kind=manifest["data_kind"], input_manifest=manifest)
    model, checkpoint = load_checkpoint(checkpoint_path, device=training_config.device)
    if asdict(model_config) != asdict(ModelConfig(**checkpoint["resolved_config"]["model"])):
        raise ValueError("Evaluation model config must match checkpoint; no ignored Head overrides.")
    if evaluate_split == "test" and manifest["data_kind"] == "real" and not training_config.outer_evaluation_permitted:
        raise ValueError("Outer evaluation remains closed.")
    if evaluate_split != "train":
        training_membership = checkpoint["resolved_config"]["split_membership"]["train"]
        if set(datasets[evaluate_split].specimen_ids) & set(training_membership["specimen_ids"]):
            raise ValueError("Evaluation specimens overlap the checkpoint training split.")
        if set(datasets[evaluate_split].sample_ids) & set(training_membership["sample_ids"]):
            raise ValueError("Evaluation samples overlap the checkpoint training split.")
    metrics, rows = evaluate(model, datasets[evaluate_split], batch_size=training_config.batch_size,
                             device=training_config.device)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_predictions(output_dir / f"{evaluate_split}_predictions.jsonl", rows)
    result = {"status": "evaluated", "split": evaluate_split, "data_kind": manifest["data_kind"], "metrics": metrics,
              "checkpoint_epoch": checkpoint["epoch"], "performance_claims_released": False}
    _write_json(output_dir / "resolved_config.json", {"checkpoint_config": checkpoint["resolved_config"],
                                                     "evaluation_manifest": manifest})
    _write_json(output_dir / "summary.json", result)
    return result
