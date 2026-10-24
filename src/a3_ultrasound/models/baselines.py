"""Task-aligned V1.2 Head baselines with parameters shared across K."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from a3_ultrasound.models.pcc import (
    HeadStepOutput, LocalBaseScorer, LocalBaseType, SharedPrefixGRUEncoder,
    validate_prefix_states,
)


BaselineStepOutput = HeadStepOutput


class IndependentConditionalStep(nn.Module):
    """Per-interface local prediction without prefix or context dependency."""

    def __init__(self, local_dim: int, *, hidden_dim: int = 32,
                 local_base_type: LocalBaseType = "mlp") -> None:
        super().__init__()
        self.base = LocalBaseScorer(
            local_dim,
            scorer_type=local_base_type,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        r_loc: Tensor,
        r_ctx: Tensor,
        r_gate: Tensor,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> BaselineStepOutput:
        validate_prefix_states(prefix_states, interface_mask)
        del r_ctx, r_gate, query
        base_logodds = self.base(r_loc) * interface_mask.to(dtype=r_loc.dtype)
        return BaselineStepOutput(
            logits=base_logodds,
            base_logodds=base_logodds,
        )


class LocalPreIndependentStep(nn.Module):
    """Fuse Local and Pre observations without output-state dependencies."""

    def __init__(self, *, local_dim: int, context_dim: int, hidden_dim: int = 64,
                 local_base_hidden_dim: int = 32,
                 local_base_type: LocalBaseType = "mlp") -> None:
        super().__init__()
        self.base = LocalBaseScorer(local_dim, scorer_type=local_base_type,
                                    hidden_dim=local_base_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(local_dim + context_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r_loc: Tensor, r_ctx: Tensor, r_gate: Tensor, query: Tensor,
                prefix_states: Tensor, interface_mask: Tensor) -> HeadStepOutput:
        validate_prefix_states(prefix_states, interface_mask)
        del r_gate, query
        valid = interface_mask.to(dtype=r_loc.dtype)
        return HeadStepOutput(
            logits=self.classifier(torch.cat((r_loc, r_ctx), dim=-1)).squeeze(-1) * valid,
            base_logodds=self.base(r_loc) * valid,
        )


class VanillaClassifierChainStep(nn.Module):
    """Shared classifier chain without the proposed Base/Gate/Correction split."""

    def __init__(
        self,
        *,
        local_dim: int,
        context_dim: int,
        query_dim: int,
        prefix_dim: int = 32,
        hidden_dim: int = 64,
        max_interfaces: int = 6,
        local_base_hidden_dim: int = 32,
        local_base_type: LocalBaseType = "mlp",
    ) -> None:
        super().__init__()
        if not 1 <= max_interfaces <= 6:
            raise ValueError("max_interfaces must lie in [1, 6].")
        self.max_interfaces = max_interfaces
        self.prefix_encoder = SharedPrefixGRUEncoder(
            query_dim=query_dim,
            state_embedding_dim=prefix_dim,
            hidden_dim=prefix_dim,
        )
        self.base = LocalBaseScorer(
            local_dim,
            scorer_type=local_base_type,
            hidden_dim=local_base_hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(local_dim + context_dim + prefix_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        r_loc: Tensor,
        r_ctx: Tensor,
        r_gate: Tensor,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> BaselineStepOutput:
        del r_gate
        if interface_mask.shape[1] > self.max_interfaces:
            raise ValueError("Kmax exceeds the classifier-chain engineering boundary.")
        prefix_hidden = self.prefix_encoder(query, prefix_states, interface_mask)
        base_logodds = self.base(r_loc)
        logits = self.classifier(
            torch.cat((r_loc, r_ctx, prefix_hidden), dim=-1)
        ).squeeze(-1)
        valid = interface_mask.to(dtype=logits.dtype)
        return BaselineStepOutput(
            logits=logits * valid,
            base_logodds=base_logodds * valid,
            prefix_hidden=prefix_hidden,
        )


__all__ = [
    "BaselineStepOutput",
    "IndependentConditionalStep",
    "LocalPreIndependentStep",
    "VanillaClassifierChainStep",
]
