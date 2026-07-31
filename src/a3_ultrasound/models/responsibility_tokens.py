"""Projected responsibility tokens for the A3 V1.1 structured Head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResponsibilityTokens:
    r_loc: Tensor
    r_ctx: Tensor
    r_gate: Tensor


class ResponsibilityTokenProjector(nn.Module):
    """Fuse heterogeneous sources through independent learned projections."""

    def __init__(
        self,
        *,
        local_dim: int,
        context_dim: int,
        query_dim: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "local_dim": local_dim,
            "context_dim": context_dim,
            "query_dim": query_dim,
            "token_dim": token_dim,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions.values()
        ):
            raise ValueError("Responsibility dimensions must be positive integers.")
        if token_dim <= 1:
            raise ValueError(
                "token_dim (D_r) must be > 1 because LayerNorm over one feature "
                "erases all source-dependent variation."
            )
        self.local_evidence_projection = nn.Linear(local_dim, token_dim, bias=False)
        self.local_query_projection = nn.Linear(query_dim, token_dim, bias=False)
        self.context_projection = nn.Linear(context_dim, token_dim, bias=False)
        self.context_query_projection = nn.Linear(query_dim, token_dim, bias=False)
        self.gate_evidence_projection = nn.Linear(local_dim, token_dim, bias=False)
        self.gate_context_projection = nn.Linear(context_dim, token_dim, bias=False)
        self.gate_query_projection = nn.Linear(query_dim, token_dim, bias=False)

        self.local_bias = nn.Parameter(torch.zeros(token_dim))
        self.context_bias = nn.Parameter(torch.zeros(token_dim))
        self.gate_bias = nn.Parameter(torch.zeros(token_dim))

        self.local_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.gate_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        e_loc: Tensor,
        c_pre: Tensor,
        query: Tensor,
        interface_mask: Tensor,
    ) -> ResponsibilityTokens:
        tensors = (e_loc, c_pre, query)
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("e_loc, c_pre, and query must each have shape [B, Kmax, D].")
        batch_size, kmax, _ = e_loc.shape
        if any(tensor.shape[:2] != (batch_size, kmax) for tensor in tensors[1:]):
            raise ValueError("Responsibility-token sources must agree on [B, Kmax].")
        if interface_mask.shape != (batch_size, kmax) or interface_mask.dtype != torch.bool:
            raise ValueError("interface_mask must be bool with shape [B, Kmax].")

        u_loc = (
            self.local_evidence_projection(e_loc)
            + self.local_query_projection(query)
            + self.local_bias
        )
        u_ctx = (
            self.context_projection(c_pre)
            + self.context_query_projection(query)
            + self.context_bias
        )
        u_gate = (
            self.gate_evidence_projection(e_loc)
            + self.gate_context_projection(c_pre)
            + self.gate_query_projection(query)
            + self.gate_bias
        )
        valid = interface_mask.unsqueeze(-1).to(dtype=e_loc.dtype)
        return ResponsibilityTokens(
            r_loc=self.local_norm(u_loc) * valid,
            r_ctx=self.context_norm(u_ctx) * valid,
            r_gate=self.gate_norm(u_gate) * valid,
        )
