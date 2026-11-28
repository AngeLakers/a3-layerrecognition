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
QUERY_META_INDICES = (0, 2)
QUERY_META_FIELDS = tuple(INTERFACE_META_FIELDS[i] for i in QUERY_META_INDICES)
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
    phys_loc: Tensor | None = None
    phys_ctx: Tensor | None = None
    board_interval_samples: Tensor | None = None

    def __post_init__(self) -> None:
        if self.x_raw.ndim not in (2, 3):
            raise ValueError("x_raw must have shape [B, T] or [B, C, T].")
        batch_size, trace_length = self.x_raw.shape[0], self.x_raw.shape[-1]
        interval = self.board_interval_samples
        if interval is not None and (
            interval.shape != (batch_size,) or interval.dtype != torch.float32
            or not torch.isfinite(interval).all() or not (interval > 0).all()
        ):
            raise ValueError("Board intervals must be finite positive float32 samples, one per trace.")
        if self.x_raw.ndim == 3 and self.x_raw.shape[1] not in (1, 2):
            raise ValueError("Expected one or two RF channels.")
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
        if (self.phys_loc is None) != (self.phys_ctx is None):
            raise ValueError("Physical local and context tensors must be supplied together.")
        for value, width in ((self.phys_loc, 1), (self.phys_ctx, 3)):
            if value is not None:
                if (value.shape != (*self.interface_mask.shape, width)
                        or value.dtype != torch.float32 or not torch.isfinite(value).all()
                        or torch.any(value[~self.interface_mask] != 0)):
                    raise ValueError("Physical features must be finite float32 with masked padding zero.")

    @property
    def valid_interface_counts(self) -> Tensor:
        return self.interface_mask.sum(dim=1)

    def pin_memory(self):
        """DataLoader custom-batch hook; CPU validation precedes H2D."""
        from dataclasses import fields
        result=object.__new__(type(self))
        for field in fields(self):
            value=getattr(self,field.name)
            object.__setattr__(result,field.name,None if value is None else value.pin_memory())
        return result
