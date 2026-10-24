"""V1.4 two-field Query, with explicit legacy three-field checkpoint support."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from a3_ultrasound.schema import INTERFACE_META_DIM, QUERY_META_INDICES


class InterfaceQueryEncoder(nn.Module):
    """Encode interface identity/position without Prior tolerance fields.

    No categorical absolute-interface ID or Query LayerNorm is accepted or
    created here. Tolerances remain in the packed API but are routed only to
    Prior/support construction.
    """

    def __init__(self, query_dim: int, hidden_dim: int | None = None,
                 *, query_view: str = "position2") -> None:
        super().__init__()
        if query_view not in ("position2", "legacy3"):
            raise ValueError("query_view must be position2 or legacy3.")
        self.meta_indices = QUERY_META_INDICES if query_view == "position2" else (0, 1, 2)
        hidden_dim = hidden_dim or query_dim
        self.network = nn.Sequential(
            nn.Linear(len(self.meta_indices), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, interface_meta: Tensor, interface_mask: Tensor) -> Tensor:
        if interface_meta.ndim != 3 or interface_meta.shape[-1] != INTERFACE_META_DIM:
            raise ValueError("interface_meta must have shape [B, Kmax, 5].")
        if interface_mask.shape != interface_meta.shape[:2] or interface_mask.dtype != torch.bool:
            raise ValueError("interface_mask must be bool with shape [B, Kmax].")
        query_meta = interface_meta[..., list(self.meta_indices)]
        query = self.network(query_meta)
        return query * interface_mask.unsqueeze(-1).to(dtype=query.dtype)
