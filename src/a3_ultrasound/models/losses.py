"""Equal-supervised-sample losses for the A3 V1.2 training contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


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
        if torch.any(label_mask & ~interface_mask):
            raise ValueError("label_mask cannot supervise padded interface slots.")
        supervision_mask = interface_mask & label_mask
    else:
        supervision_mask = interface_mask

    if torch.any(interface_mask.sum(dim=1) == 0):
        raise ValueError("Every loss sample must contain at least one valid interface.")
    supervised_values = y[supervision_mask]
    if supervised_values.numel() == 0:
        raise ValueError("At least one interface label is required for PCC loss.")
    if torch.any((supervised_values != 0) & (supervised_values != 1)):
        raise ValueError("Supervised y values must use good=0/abnormal=1.")
    if not torch.isfinite(conditional_logits[interface_mask]).all() or not torch.isfinite(
        base_logodds[interface_mask]
    ).all():
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

    target = y.to(dtype=conditional_logits.dtype)
    chain_terms = F.binary_cross_entropy_with_logits(
        conditional_logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    local_terms = F.binary_cross_entropy_with_logits(
        base_logodds,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    active = supervision_mask.to(dtype=conditional_logits.dtype)
    supervised_counts = supervision_mask.sum(dim=1).to(dtype=conditional_logits.dtype)
    supervised_sample_mask = supervision_mask.any(dim=1)
    safe_counts = supervised_counts.clamp_min(1.0)
    per_sample_chain = (chain_terms * active).sum(dim=1) / safe_counts
    per_sample_local = (local_terms * active).sum(dim=1) / safe_counts
    chain = per_sample_chain[supervised_sample_mask].mean()
    local = per_sample_local[supervised_sample_mask].mean()
    total = chain + resolved_lambda * local
    return PCCLossOutput(
        total=total,
        chain=chain,
        local=local,
        per_sample_chain=per_sample_chain,
        per_sample_local=per_sample_local,
        supervised_sample_mask=supervised_sample_mask,
    )


__all__ = ["PCCLossOutput", "pcc_training_loss"]
