"""Scalar Bernoulli propagation-conditioned classifier (PCC) step."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PCCStepOutput:
    logits: Tensor
    base_logodds: Tensor
    correction: Tensor
    gate: Tensor
    prefix_hidden: Tensor


class PCCConditionalStep(nn.Module):
    """Shared conditional Bernoulli step for truth-prefix training and enumeration.

    The correction pathway receives only ``r_ctx`` and ordered prefix history.
    The Gate receives only ``r_gate`` and is path-independent. Correction has no
    direct tensor input from ``r_loc``/local evidence.
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
        max_interfaces: int = 3,
        rho: float = 1.0,
    ) -> None:
        super().__init__()
        if max_interfaces <= 0:
            raise ValueError("max_interfaces must be positive.")
        try:
            resolved_rho = float(rho)
        except (TypeError, ValueError) as exc:
            raise ValueError("rho must be a finite positive number.") from exc
        if not math.isfinite(resolved_rho) or resolved_rho <= 0:
            raise ValueError(
                "rho must be finite and > 0 so it remains a positive correction bound."
            )
        self.rho = resolved_rho
        self.max_interfaces = int(max_interfaces)
        self.query_dim = int(query_dim)
        self.prefix_dim = int(prefix_dim)
        self.base = nn.Linear(local_dim, 1)
        self.prefix_embedding = nn.Embedding(2, prefix_dim)
        prefix_input_dim = self.max_interfaces * (query_dim + prefix_dim + 1)
        self.prefix_network = nn.Sequential(
            nn.Linear(prefix_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prefix_dim),
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

    def prefix_history(self, query: Tensor, prefix_states: Tensor, interface_mask: Tensor) -> Tensor:
        """Encode ordered valid states before every target interface with an MLP."""

        batch_size, kmax, _ = query.shape
        if kmax > self.max_interfaces:
            raise ValueError(
                f"Kmax={kmax} exceeds the Prefix MLP boundary of {self.max_interfaces}."
            )
        if prefix_states.shape != (batch_size, kmax):
            raise ValueError("prefix_states must have shape [B, Kmax].")
        safe_states = prefix_states.to(dtype=torch.long).clamp(min=0, max=1)
        slot_valid = interface_mask.unsqueeze(-1).to(dtype=query.dtype)
        slot_features = torch.cat(
            (
                query,
                self.prefix_embedding(safe_states),
                slot_valid,
            ),
            dim=-1,
        )
        if kmax < self.max_interfaces:
            padding = query.new_zeros(
                (
                    batch_size,
                    self.max_interfaces - kmax,
                    slot_features.shape[-1],
                )
            )
            slot_features = torch.cat((slot_features, padding), dim=1)
            padded_mask = torch.cat(
                (
                    interface_mask,
                    torch.zeros(
                        (batch_size, self.max_interfaces - kmax),
                        device=interface_mask.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=1,
            )
        else:
            padded_mask = interface_mask

        source_index = torch.arange(self.max_interfaces, device=query.device).view(1, 1, -1)
        target_index = torch.arange(kmax, device=query.device).view(1, -1, 1)
        prefix_mask = (source_index < target_index) & padded_mask.unsqueeze(1)
        ordered = slot_features.unsqueeze(1) * prefix_mask.unsqueeze(-1).to(dtype=query.dtype)
        flat_prefix = ordered.flatten(start_dim=2)
        history = self.prefix_network(flat_prefix)
        has_prefix = prefix_mask.any(dim=-1)
        return torch.where(has_prefix.unsqueeze(-1), history, torch.zeros_like(history))

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
        base_logodds = self.base(r_loc).squeeze(-1)
        correction = self.rho * torch.tanh(
            self.correction(torch.cat((r_ctx, prefix_hidden), dim=-1)).squeeze(-1)
        )
        gate = torch.sigmoid(self.gate(r_gate).squeeze(-1))
        has_prefix = interface_mask.cumsum(dim=1) > 1
        correction = torch.where(has_prefix, correction, torch.zeros_like(correction))
        logits = base_logodds + gate * correction
        valid = interface_mask.to(dtype=logits.dtype)
        return PCCStepOutput(
            logits=logits * valid,
            base_logodds=base_logodds * valid,
            correction=correction * valid,
            gate=gate * valid,
            prefix_hidden=prefix_hidden * valid.unsqueeze(-1),
        )
