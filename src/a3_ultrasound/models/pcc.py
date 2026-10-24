"""Variable-K propagation-conditioned binary head for A3 V1.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


LocalBaseType = Literal["linear", "mlp"]


@dataclass(frozen=True)
class HeadStepOutput:
    logits: Tensor
    base_logodds: Tensor
    correction: Tensor | None = None
    gate: Tensor | None = None
    prefix_hidden: Tensor | None = None
    upstream_flag: Tensor | None = None


# Keep the historical constructor/output name for existing PCC callers.
PCCStepOutput = HeadStepOutput


def validate_prefix_states(prefix_states: Tensor, interface_mask: Tensor) -> Tensor:
    """Reject invalid real states; replace padding only before embedding lookup."""
    if interface_mask.ndim != 2 or interface_mask.dtype != torch.bool:
        raise ValueError("interface_mask must be bool with shape [B, Kmax].")
    if prefix_states.shape != interface_mask.shape:
        raise ValueError("prefix_states must have shape [B, Kmax].")
    if prefix_states.device != interface_mask.device:
        raise ValueError("prefix_states and interface_mask must share a device.")
    if prefix_states.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        raise ValueError("prefix_states must be integer binary states on valid slots.")
    valid_states = prefix_states[interface_mask]
    if torch.any((valid_states != 0) & (valid_states != 1)):
        raise ValueError("prefix_states must be binary on valid interface slots.")
    return torch.where(interface_mask, prefix_states, 0).to(dtype=torch.long)


class LocalBaseScorer(nn.Module):
    """Interface-local log-odds scorer with a retained linear ablation."""

    def __init__(
        self,
        input_dim: int,
        *,
        scorer_type: LocalBaseType = "mlp",
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (input_dim, hidden_dim)
        ):
            raise ValueError("Local-base dimensions must be positive integers.")
        if scorer_type == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif scorer_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError("scorer_type must be 'linear' or 'mlp'.")
        self.scorer_type = scorer_type

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).squeeze(-1)


class SharedPrefixGRUEncoder(nn.Module):
    """Encode only previous valid Query/state pairs with shared parameters."""

    def __init__(self, query_dim: int, state_embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (query_dim, state_embedding_dim, hidden_dim)
        ):
            raise ValueError("Prefix dimensions must be positive integers.")
        self.state_embedding = nn.Embedding(2, state_embedding_dim)
        self.gru_cell = nn.GRUCell(query_dim + state_embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> Tensor:
        if query.ndim != 3:
            raise ValueError("query must have shape [B, Kmax, Dq].")
        batch_size, kmax, _ = query.shape
        if prefix_states.shape != (batch_size, kmax):
            raise ValueError("prefix_states must have shape [B, Kmax].")
        if interface_mask.shape != (batch_size, kmax) or interface_mask.dtype != torch.bool:
            raise ValueError("interface_mask must be bool with shape [B, Kmax].")

        safe_states = validate_prefix_states(prefix_states, interface_mask)
        embedded = self.state_embedding(safe_states)
        hidden = query.new_zeros((batch_size, self.hidden_dim))
        history: list[Tensor] = []
        for source_index in range(kmax):
            # h_<k is captured before consuming the current interface state.
            history.append(hidden)
            update = self.gru_cell(
                torch.cat((query[:, source_index], embedded[:, source_index]), dim=-1),
                hidden,
            )
            valid = interface_mask[:, source_index].unsqueeze(-1)
            hidden = torch.where(valid, update, hidden)
        return torch.stack(history, dim=1) * interface_mask.unsqueeze(-1).to(
            dtype=query.dtype
        )


class PCCConditionalStep(nn.Module):
    """Shared variable-K conditional Bernoulli step.

    Prefix, Base, Correction, and Gate parameters are shared across interface
    positions and do not contain a ``Kmax * feature_dim`` trainable layer.
    ``max_interfaces`` is only an exact-inference and validation guard.
    """

    def __init__(
        self,
        *,
        local_dim: int,
        context_dim: int,
        gate_dim: int,
        query_dim: int,
        prefix_dim: int,
        hidden_dim: int,
        max_interfaces: int = 6,
        rho: float = 1.0,
        local_base_type: LocalBaseType = "mlp",
        local_base_hidden_dim: int = 32,
        correction_mode: Literal["bidirectional", "suppress_only"] = "bidirectional",
    ) -> None:
        super().__init__()
        if (
            not isinstance(max_interfaces, int)
            or isinstance(max_interfaces, bool)
            or not 1 <= max_interfaces <= 6
        ):
            raise ValueError("max_interfaces must be an integer in [1, 6].")
        try:
            resolved_rho = float(rho)
        except (TypeError, ValueError) as exc:
            raise ValueError("rho must be a finite positive number.") from exc
        if not math.isfinite(resolved_rho) or resolved_rho <= 0:
            raise ValueError(
                "rho must be finite and > 0 so it remains a positive correction bound."
            )
        self.rho = resolved_rho
        if correction_mode not in ("bidirectional", "suppress_only"):
            raise ValueError("Unknown correction_mode.")
        self.correction_mode = correction_mode
        self.max_interfaces = max_interfaces
        self.query_dim = int(query_dim)
        self.prefix_dim = int(prefix_dim)
        self.base = LocalBaseScorer(
            local_dim,
            scorer_type=local_base_type,
            hidden_dim=local_base_hidden_dim,
        )
        self.prefix_encoder = SharedPrefixGRUEncoder(
            query_dim=query_dim,
            state_embedding_dim=prefix_dim,
            hidden_dim=prefix_dim,
        )
        self.correction = nn.Sequential(
            nn.Linear(context_dim + prefix_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final_correction = self.correction[-1]
        assert isinstance(final_correction, nn.Linear)
        nn.init.zeros_(final_correction.weight)
        nn.init.zeros_(final_correction.bias)
        if correction_mode == "suppress_only":
            nn.init.constant_(final_correction.bias, -3.0)

    def prefix_history(
        self,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> Tensor:
        """Return the recurrent state available before every interface."""

        if query.ndim != 3:
            raise ValueError("query must have shape [B, Kmax, Dq].")
        if query.shape[1] > self.max_interfaces:
            raise ValueError(
                f"Kmax={query.shape[1]} exceeds the engineering boundary of "
                f"{self.max_interfaces}."
            )
        return self.prefix_encoder(query, prefix_states, interface_mask)

    def forward(
        self,
        r_loc: Tensor,
        r_ctx: Tensor,
        r_gate: Tensor,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> PCCStepOutput:
        tensors = (r_loc, r_ctx, r_gate, query)
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("Representations and query must each have shape [B, Kmax, D].")
        batch_size, kmax, _ = r_loc.shape
        if any(tensor.shape[:2] != (batch_size, kmax) for tensor in tensors[1:]):
            raise ValueError("All PCC inputs must share [B, Kmax].")
        if interface_mask.shape != (batch_size, kmax) or interface_mask.dtype != torch.bool:
            raise ValueError("interface_mask must be bool with shape [B, Kmax].")

        prefix_hidden = self.prefix_history(query, prefix_states, interface_mask)
        base_logodds = self.base(r_loc)
        raw_correction = self.correction(torch.cat((r_ctx, prefix_hidden), dim=-1)).squeeze(-1)
        correction = (self.rho * torch.tanh(raw_correction)
                      if self.correction_mode == "bidirectional"
                      else -self.rho * torch.sigmoid(raw_correction))
        gate = torch.sigmoid(self.gate(r_gate).squeeze(-1))
        has_prefix = interface_mask.cumsum(dim=1) > 1
        correction = torch.where(has_prefix, correction, torch.zeros_like(correction))
        positive = ((prefix_states == 1) & interface_mask).long()
        upstream_flag = ((positive.cumsum(1) - positive) > 0) & interface_mask
        effective_correction = correction
        if self.correction_mode == "suppress_only":
            effective_correction = correction * upstream_flag.to(correction.dtype)
        logits = base_logodds + gate * effective_correction
        valid = interface_mask.to(dtype=logits.dtype)
        return PCCStepOutput(
            logits=logits * valid,
            base_logodds=base_logodds * valid,
            correction=correction * valid,
            gate=gate * valid,
            prefix_hidden=prefix_hidden * valid.unsqueeze(-1),
            upstream_flag=upstream_flag if self.correction_mode == "suppress_only" else None,
        )


__all__ = [
    "HeadStepOutput",
    "LocalBaseScorer",
    "LocalBaseType",
    "PCCConditionalStep",
    "PCCStepOutput",
    "SharedPrefixGRUEncoder",
    "validate_prefix_states",
]
