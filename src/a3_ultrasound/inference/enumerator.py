"""Exact log-space enumeration of PCC Bernoulli state paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F


class ConditionalLogitOutput(Protocol):
    logits: Tensor


class BinaryConditionalStep(Protocol):
    def __call__(
        self,
        r_loc: Tensor,
        r_ctx: Tensor,
        r_gate: Tensor,
        query: Tensor,
        prefix_states: Tensor,
        interface_mask: Tensor,
    ) -> ConditionalLogitOutput: ...


@dataclass(frozen=True)
class ExactInferenceOutput:
    path_states: Tensor
    path_mask: Tensor
    log_probabilities: Tensor
    probabilities: Tensor
    marginals: Tensor
    map_states: Tensor
    raw_log_normalizer: Tensor
    raw_log_probabilities: Tensor
    conditional_probability_range: Tensor


def lexicographic_binary_paths(interface_count: int, *, device: torch.device) -> Tensor:
    """Return ``[2**K, K]`` paths in fixed bit-string lexical order."""

    if not 1 <= interface_count <= 6:
        raise ValueError("Exact enumeration is supported only for 1 <= K <= 6.")
    values = torch.arange(2**interface_count, device=device, dtype=torch.long)
    shifts = torch.arange(interface_count - 1, -1, -1, device=device, dtype=torch.long)
    return ((values.unsqueeze(1) >> shifts) & 1).to(dtype=torch.long)


def enumerate_binary_chain(
    step: BinaryConditionalStep,
    r_loc: Tensor,
    r_ctx: Tensor,
    r_gate: Tensor,
    query: Tensor,
    interface_mask: Tensor,
) -> ExactInferenceOutput:
    """Enumerate all valid paths using the same PCC step used in training.

    Samples are compacted by their valid interface slots, so mixed K batches do
    not leak padded slots into probabilities or marginals.
    """

    if interface_mask.ndim != 2 or interface_mask.dtype != torch.bool:
        raise ValueError("interface_mask must be bool with shape [B, Kmax].")
    batch_size, kmax = interface_mask.shape
    if kmax > 6:
        raise ValueError("Kmax exceeds the prototype enumeration boundary of 6.")
    if any(tensor.shape[:2] != (batch_size, kmax) for tensor in (r_loc, r_ctx, r_gate, query)):
        raise ValueError("PCC tensors must agree with interface_mask on [B, Kmax].")

    counts = interface_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every enumerated sample must contain at least one valid interface.")
    expected_mask = (
        torch.arange(kmax, device=interface_mask.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    if not torch.equal(interface_mask, expected_mask):
        raise ValueError("interface_mask must describe one contiguous ordered prefix.")
    max_paths = 2 ** int(counts.max().item())
    device = r_loc.device
    states = torch.full((batch_size, max_paths, kmax), -1, device=device, dtype=torch.long)
    path_mask = torch.zeros((batch_size, max_paths), device=device, dtype=torch.bool)
    log_probabilities = torch.full((batch_size, max_paths), float("-inf"), device=device, dtype=r_loc.dtype)
    probabilities = torch.zeros((batch_size, max_paths), device=device, dtype=r_loc.dtype)
    marginals = torch.zeros((batch_size, kmax), device=device, dtype=r_loc.dtype)
    map_states = torch.full((batch_size, kmax), -1, device=device, dtype=torch.long)
    raw_log_probabilities = torch.full_like(log_probabilities, float("-inf"))
    raw_log_normalizer = r_loc.new_zeros(batch_size)
    conditional_probability_range = r_loc.new_zeros((batch_size, kmax))

    # Equal-K samples share the same path lattice. Fold sample and path axes
    # into the Head's batch axis; no reduction ever crosses sample boundaries.
    for interface_count in sorted(set(counts.cpu().tolist())):
        batch_indices = torch.nonzero(counts == interface_count, as_tuple=False).squeeze(1)
        group_size = batch_indices.numel()
        paths = lexicographic_binary_paths(interface_count, device=device)
        path_count = paths.shape[0]
        def repeated(tensor):
            selected = tensor.index_select(0, batch_indices)[:, :interface_count]
            return selected[:, None].expand(-1, path_count, -1, -1).reshape(
                group_size * path_count, interface_count, selected.shape[-1])
        grouped_paths = paths[None].expand(group_size, -1, -1)
        compact_mask = torch.ones((group_size * path_count, interface_count), device=device, dtype=torch.bool)
        output = step(
            repeated(r_loc),
            repeated(r_ctx),
            repeated(r_gate),
            repeated(query),
            grouped_paths.reshape(-1, interface_count),
            compact_mask,
        )
        logits = output.logits.reshape(group_size, path_count, interface_count)
        path_values = grouped_paths.to(dtype=logits.dtype)
        unnormalized_log_probability = (
            path_values * F.logsigmoid(logits)
            + (1.0 - path_values) * F.logsigmoid(-logits)
        ).sum(dim=2)
        log_z = torch.logsumexp(unnormalized_log_probability, dim=1)
        log_probability = unnormalized_log_probability - log_z[:, None]
        probability = log_probability.exp()
        compact_marginal = (probability.unsqueeze(2) * path_values).sum(dim=1)
        best_path = paths[log_probability.argmax(dim=1)]

        states[batch_indices, :path_count, :interface_count] = grouped_paths
        path_mask[batch_indices, :path_count] = True
        log_probabilities[batch_indices, :path_count] = log_probability
        probabilities[batch_indices, :path_count] = probability
        marginals[batch_indices, :interface_count] = compact_marginal
        map_states[batch_indices, :interface_count] = best_path
        raw_log_normalizer[batch_indices] = log_z
        raw_log_probabilities[batch_indices, :path_count] = unnormalized_log_probability
        conditional_p = logits.sigmoid()
        conditional_probability_range[batch_indices, :interface_count] = (
            conditional_p.amax(dim=1) - conditional_p.amin(dim=1))

    return ExactInferenceOutput(
        path_states=states,
        path_mask=path_mask,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        marginals=marginals,
        map_states=map_states,
        raw_log_normalizer=raw_log_normalizer,
        raw_log_probabilities=raw_log_probabilities,
        conditional_probability_range=conditional_probability_range,
    )


# Compatibility name retained for V1.1 callers.
enumerate_pcc = enumerate_binary_chain


__all__ = [
    "BinaryConditionalStep",
    "ExactInferenceOutput",
    "enumerate_binary_chain",
    "enumerate_pcc",
    "lexicographic_binary_paths",
]
