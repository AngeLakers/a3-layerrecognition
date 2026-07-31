"""Stable tensor schema for the A3 V1.1 prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


INTERFACE_META_SCHEMA_VERSION = "a3-interface-meta-v3"
INTERFACE_META_FIELDS = (
    "relative_order",
    "normalized_tof",
    "normalized_depth",
    "normalized_left_tolerance",
    "normalized_right_tolerance",
)
INTERFACE_META_DIM = len(INTERFACE_META_FIELDS)
QUERY_META_FIELDS = INTERFACE_META_FIELDS[:3]
QUERY_META_INDICES = (0, 1, 2)
QUERY_META_DIM = len(QUERY_META_FIELDS)
PRIOR_META_FIELDS = (
    "normalized_tof",
    "normalized_left_tolerance",
    "normalized_right_tolerance",
)
PRIOR_META_INDICES = (1, 3, 4)
PRIOR_META_DIM = len(PRIOR_META_FIELDS)


@dataclass(frozen=True)
class InterfaceBatch:
    """Validated batch tensors without any audit-only metadata.

    Network-facing fields deliberately exclude specimen ID, scan coordinate,
    region ID, label source, and paths. Those belong only in split/audit data.
    """

    x_raw: Tensor
    length_mask: Tensor
    interface_meta: Tensor
    interface_mask: Tensor
    label_mask: Tensor | None = None
    y: Tensor | None = None

    def __post_init__(self) -> None:
        if self.x_raw.ndim != 2:
            raise ValueError("x_raw must have shape [B, T].")
        batch_size, trace_length = self.x_raw.shape
        if self.x_raw.dtype != torch.float32:
            raise TypeError("x_raw must be float32 to preserve the RF contract.")
        if self.length_mask.shape != (batch_size, trace_length):
            raise ValueError("length_mask must have shape [B, T].")
        if self.length_mask.dtype != torch.bool:
            raise TypeError("length_mask must be bool.")
        if self.interface_meta.ndim != 3 or self.interface_meta.shape[0] != batch_size:
            raise ValueError("interface_meta must have shape [B, Kmax, 5].")
        if self.interface_meta.shape[-1] != INTERFACE_META_DIM:
            raise ValueError(
                "interface_meta must use a3-interface-meta-v3 five-field order."
            )
        if self.interface_meta.dtype != torch.float32:
            raise TypeError("interface_meta must be float32.")
        if self.interface_mask.shape != self.interface_meta.shape[:2]:
            raise ValueError("interface_mask must have shape [B, Kmax].")
        if self.interface_mask.dtype != torch.bool:
            raise TypeError("interface_mask must be bool.")
        if self.label_mask is not None:
            if self.label_mask.shape != self.interface_mask.shape or self.label_mask.dtype != torch.bool:
                raise ValueError("label_mask must be bool with shape [B, Kmax].")
        if self.y is not None:
            if self.y.shape != self.interface_mask.shape or self.y.dtype != torch.int64:
                raise ValueError("y must be int64 with shape [B, Kmax].")

    @property
    def valid_interface_counts(self) -> Tensor:
        return self.interface_mask.sum(dim=1)
