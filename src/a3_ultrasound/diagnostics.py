"""Label-free diagnostics for exercising the integrated A3 computation graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from a3_ultrasound.models.integrated import IntegratedPCCOutput


PROBE_COMPONENT_WEIGHTS: Mapping[str, float] = {
    "conditional_logits_squared_mean": 1.0,
    "base_logodds_squared_mean": 0.5,
    "correction_mean": 0.25,
    "gate_squared_mean": 0.125,
    "prefix_hidden_squared_mean": 0.0625,
    "query_squared_mean": 0.03125,
    "local_token_squared_mean": 0.015625,
    "pre_token_squared_mean": 0.015625,
    "r_loc_squared_mean": 0.0078125,
    "r_ctx_squared_mean": 0.0078125,
    "r_gate_squared_mean": 0.0078125,
}


@dataclass(frozen=True)
class LabelFreeAutogradProbe:
    """A deterministic scalar used only to inspect autograd connectivity.

    The scalar has no target and no statistical or physical interpretation. It
    must not be reused as a training objective, model-selection metric, or
    performance measure.
    """

    scalar: Tensor
    components: Mapping[str, Tensor]
    weights: Mapping[str, float]


def _valid_mean(values: Tensor, interface_mask: Tensor, *, squared: bool) -> Tensor:
    if values.shape[:2] != interface_mask.shape:
        raise ValueError("Probe tensors must share the [B, Kmax] interface shape.")
    selected = values[interface_mask]
    if selected.numel() == 0:
        raise ValueError("The label-free autograd probe needs at least one valid interface.")
    if squared:
        selected = selected.square()
    return selected.mean()


def build_label_free_autograd_probe(
    output: IntegratedPCCOutput,
    interface_mask: Tensor,
) -> LabelFreeAutogradProbe:
    """Build a target-free scalar that touches every integrated model branch."""

    if interface_mask.dtype != torch.bool or interface_mask.ndim != 2:
        raise ValueError("interface_mask must be bool with shape [B, Kmax].")
    components = {
        "conditional_logits_squared_mean": _valid_mean(
            output.conditional.logits, interface_mask, squared=True
        ),
        "base_logodds_squared_mean": _valid_mean(
            output.conditional.base_logodds, interface_mask, squared=True
        ),
        # Correction is zero-initialized by contract. A linear mean is used so
        # that the diagnostic still reaches its final projection at step zero.
        "correction_mean": _valid_mean(
            output.conditional.correction, interface_mask, squared=False
        ),
        "gate_squared_mean": _valid_mean(
            output.conditional.gate, interface_mask, squared=True
        ),
        "prefix_hidden_squared_mean": _valid_mean(
            output.conditional.prefix_hidden, interface_mask, squared=True
        ),
        "query_squared_mean": _valid_mean(output.query, interface_mask, squared=True),
        "local_token_squared_mean": _valid_mean(
            output.local_readout.token, interface_mask, squared=True
        ),
        "pre_token_squared_mean": _valid_mean(
            output.pre_readout.token, interface_mask, squared=True
        ),
        "r_loc_squared_mean": _valid_mean(
            output.responsibility_tokens.r_loc, interface_mask, squared=True
        ),
        "r_ctx_squared_mean": _valid_mean(
            output.responsibility_tokens.r_ctx, interface_mask, squared=True
        ),
        "r_gate_squared_mean": _valid_mean(
            output.responsibility_tokens.r_gate, interface_mask, squared=True
        ),
    }
    scalar = sum(
        components[name] * weight
        for name, weight in PROBE_COMPONENT_WEIGHTS.items()
    )
    return LabelFreeAutogradProbe(
        scalar=scalar,
        components=components,
        weights=dict(PROBE_COMPONENT_WEIGHTS),
    )


def summarize_parameter_gradients(
    model: nn.Module,
    group_prefixes: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, int | float | bool]]:
    """Summarize finite/nonzero gradients without exposing parameter values."""

    named_parameters = tuple(model.named_parameters())
    summary: dict[str, dict[str, int | float | bool]] = {}
    for group_name, prefixes in group_prefixes.items():
        members = [
            parameter
            for name, parameter in named_parameters
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        gradients = [parameter.grad for parameter in members if parameter.grad is not None]
        finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        max_abs = max(
            (float(gradient.detach().abs().max()) for gradient in gradients),
            default=0.0,
        )
        summary[group_name] = {
            "parameter_tensors": len(members),
            "parameter_values": sum(parameter.numel() for parameter in members),
            "gradient_tensors": len(gradients),
            "all_gradients_finite": finite,
            "nonzero_gradient_present": finite and max_abs > 0.0,
            "gradient_max_abs": max_abs,
        }
    return summary


__all__ = [
    "LabelFreeAutogradProbe",
    "PROBE_COMPONENT_WEIGHTS",
    "build_label_free_autograd_probe",
    "summarize_parameter_gradients",
]
