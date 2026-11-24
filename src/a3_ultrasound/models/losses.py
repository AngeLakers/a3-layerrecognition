"""Equal-supervised-sample losses for the A3 V1.2 training contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from a3_ultrasound.models.pcc import HeadStepOutput, validate_prefix_states
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.speed import complete_training,fast_structure,finite_tensor


@dataclass(frozen=True)
class PCCLossOutput:
    """Total, primary, and local auxiliary losses with audit components."""

    total: Tensor
    chain: Tensor
    local: Tensor
    per_sample_chain: Tensor
    per_sample_local: Tensor
    supervised_sample_mask: Tensor


def pcc_training_loss(
    conditional_logits: Tensor,
    base_logodds: Tensor,
    y: Tensor,
    interface_mask: Tensor,
    *,
    label_mask: Tensor | None = None,
    positive_class_weight: float | Tensor | None = None,
    lambda_local: float = 0.3,
    shadow_auxiliary: bool = False,
) -> PCCLossOutput:
    """Compute truth-prefix PCC and local losses on the frozen scale.

    Each supervised A-scan is divided by its supervised-interface count
    ``N_i^sup`` before averaging, so every sample with at least one reliable
    label has equal weight even when ``label_mask`` is partial.
    """

    if conditional_logits.ndim != 2:
        raise ValueError("conditional_logits must have shape [B, Kmax].")
    expected_shape = conditional_logits.shape
    if base_logodds.shape != expected_shape or y.shape != expected_shape:
        raise ValueError("base_logodds and y must match conditional_logits.")
    if interface_mask.shape != expected_shape or interface_mask.dtype != torch.bool:
        raise ValueError("interface_mask must be bool with shape [B, Kmax].")
    if y.dtype != torch.int64:
        raise TypeError("y must be int64 before loss computation.")
    if label_mask is not None:
        if label_mask.shape != expected_shape or label_mask.dtype != torch.bool:
            raise ValueError("label_mask must be bool with shape [B, Kmax].")
        if not fast_structure() and torch.any(label_mask & ~interface_mask):
            raise ValueError("label_mask cannot supervise padded interface slots.")
        supervision_mask = interface_mask & label_mask
    else:
        supervision_mask = interface_mask

    if not fast_structure() and torch.any(interface_mask.sum(dim=1) == 0):
        raise ValueError("Every loss sample must contain at least one valid interface.")
    if not complete_training():
        supervised_values = y[supervision_mask]
        if supervised_values.numel() == 0:
            raise ValueError("At least one interface label is required for PCC loss.")
        if torch.any((supervised_values != 0) & (supervised_values != 1)):
            raise ValueError("Supervised y values must use good=0/abnormal=1.")
    if fast_structure():
        finite_tensor(conditional_logits,interface_mask);finite_tensor(base_logodds,interface_mask)
    elif not torch.isfinite(conditional_logits[interface_mask]).all() or not torch.isfinite(base_logodds[interface_mask]).all():
        raise ValueError("Valid PCC logits must be finite.")

    try:
        resolved_lambda = float(lambda_local)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda_local must be finite and >= 0.") from exc
    if not math.isfinite(resolved_lambda) or resolved_lambda < 0:
        raise ValueError("lambda_local must be finite and >= 0.")

    pos_weight = None
    if positive_class_weight is not None:
        pos_weight = torch.as_tensor(
            positive_class_weight,
            dtype=conditional_logits.dtype,
            device=conditional_logits.device,
        )
        if pos_weight.numel() != 1 or not torch.isfinite(pos_weight).all() or pos_weight.item() <= 0:
            raise ValueError("positive_class_weight must be one finite positive scalar.")

    # Keep the low-level masked reducer available for legacy diagnostics. It
    # does not authorize teacher forcing with unknown states; use the wrapper below.
    target = torch.where(supervision_mask, y, 0).to(dtype=conditional_logits.dtype)
    safe_logits = torch.where(supervision_mask, conditional_logits, 0.0)
    safe_base = torch.where(supervision_mask, base_logodds, 0.0)
    chain_terms = F.binary_cross_entropy_with_logits(
        safe_logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    local_terms = F.binary_cross_entropy_with_logits(
        safe_base,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    active = supervision_mask.to(dtype=conditional_logits.dtype)
    supervised_counts = supervision_mask.sum(dim=1).to(dtype=conditional_logits.dtype)
    supervised_sample_mask = supervision_mask.any(dim=1)
    safe_counts = supervised_counts.clamp_min(1.0)
    per_sample_chain = (chain_terms * active).sum(dim=1) / safe_counts
    local_mask = supervision_mask
    if shadow_auxiliary:
        known = supervision_mask | ~interface_mask
        upstream_unknown = ((~known).long().cumsum(1) - (~known).long()) > 0
        positive = ((y == 1) & supervision_mask).long()
        upstream_bad = (positive.cumsum(1) - positive) > 0
        local_mask = supervision_mask & ~upstream_bad & ~upstream_unknown
    local_counts = local_mask.sum(1).clamp_min(1).to(conditional_logits.dtype)
    per_sample_local = (local_terms * local_mask).sum(dim=1) / local_counts
    chain = per_sample_chain.mean() if complete_training() else per_sample_chain[supervised_sample_mask].mean()
    local_samples = local_mask.any(1)
    local = per_sample_local.mean() if complete_training() else (per_sample_local[local_samples].mean() if local_samples.any() else per_sample_local.sum() * 0.0)
    total = chain + resolved_lambda * local
    return PCCLossOutput(
        total=total,
        chain=chain,
        local=local,
        per_sample_chain=per_sample_chain,
        per_sample_local=per_sample_local,
        supervised_sample_mask=supervised_sample_mask,
    )


def validate_complete_training_batch(batch: InterfaceBatch) -> None:
    """The common four-Head main-training cohort requires all valid labels."""
    if batch.y is None:
        raise ValueError("Main training requires complete reliable labels (y).")
    if not fast_structure() and batch.label_mask is not None and not torch.equal(batch.label_mask, batch.interface_mask):
        raise ValueError("Main training requires complete label_mask coverage of all valid interfaces.")
    validate_prefix_states(batch.y, batch.interface_mask)


def head_training_loss(output: HeadStepOutput, batch: InterfaceBatch, *, head_type: str,
                       lambda_local: float = 0.3,
                       positive_class_weight: float | Tensor | None = None,
                       disable_local_auxiliary: bool = False,
                       local_aux_mode: str | None = None) -> PCCLossOutput:
    """Dispatch the main loss; lambda=0 is an explicitly named ablation only."""
    validate_complete_training_batch(batch)
    if head_type not in ("local_independent", "local_pre_independent", "vanilla_chain", "proposed", "proposed_shadow_aux", "proposed_v15", "state_conditioned_gru"):
        raise ValueError("Unknown head_type for training loss.")
    if local_aux_mode not in (None, "all", "clean_upstream"):
        raise ValueError("local_aux_mode must be all or clean_upstream.")
    clean_upstream = (head_type in ("proposed_shadow_aux", "state_conditioned_gru") if local_aux_mode is None
                      else local_aux_mode == "clean_upstream")
    if disable_local_auxiliary:
        if lambda_local != 0 or head_type == "local_independent":
            raise ValueError("The no-local-auxiliary ablation requires lambda_local=0 and an auxiliary Head.")
    elif lambda_local not in (0.1, 0.3, 1.0):
        raise ValueError("Main lambda_local candidates are 0.1, 0.3, 1.0; zero requires explicit ablation.")
    effective_lambda = 0.0 if head_type == "local_independent" else lambda_local
    return pcc_training_loss(output.logits, output.base_logodds, batch.y, batch.interface_mask,
                             label_mask=batch.label_mask, positive_class_weight=positive_class_weight,
                             lambda_local=effective_lambda, shadow_auxiliary=clean_upstream)


__all__ = ["PCCLossOutput", "pcc_training_loss", "head_training_loss", "validate_complete_training_batch"]
