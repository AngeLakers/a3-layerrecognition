"""Exact log-space enumeration of PCC Bernoulli state paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from a3_ultrasound.models.pcc import PCCConditionalStep


@dataclass(frozen=True)
class ExactInferenceOutput:
    path_states: Tensor
    path_mask: Tensor
    log_probabilities: Tensor
    probabilities: Tensor
    marginals: Tensor
    map_states: Tensor


def lexicographic_binary_paths(interface_count: int, *, device: torch.device) -> Tensor:
    """Return ``[2**K, K]`` paths in fixed bit-string lexical order."""

    if not 1 <= interface_count <= 6:
        raise ValueError("Exact enumeration is supported only for 1 <= K <= 6.")
    values = torch.arange(2**interface_count, device=device, dtype=torch.long)
    shifts = torch.arange(interface_count - 1, -1, -1, device=device, dtype=torch.long)
    return ((values.unsqueeze(1) >> shifts) & 1).to(dtype=torch.long)


def enumerate_pcc(
    step: PCCConditionalStep,
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

    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(interface_mask[batch_index], as_tuple=False).squeeze(1)
        interface_count = int(valid_indices.numel())
        paths = lexicographic_binary_paths(interface_count, device=device)
        path_count = paths.shape[0]
        repeated = lambda tensor: tensor[batch_index : batch_index + 1, valid_indices].expand(path_count, -1, -1)
        compact_mask = torch.ones((path_count, interface_count), device=device, dtype=torch.bool)
        output = step(
            repeated(r_loc),
            repeated(r_ctx),
            repeated(r_gate),
            repeated(query),
            paths,
            compact_mask,
        )
        path_values = paths.to(dtype=output.logits.dtype)
        unnormalized_log_probability = (
            path_values * F.logsigmoid(output.logits)
            + (1.0 - path_values) * F.logsigmoid(-output.logits)
        ).sum(dim=1)
        log_probability = (
            unnormalized_log_probability
            - torch.logsumexp(unnormalized_log_probability, dim=0)
        )
        probability = log_probability.exp()
        compact_marginal = (probability.unsqueeze(1) * path_values).sum(dim=0)
        best_path = paths[log_probability.argmax(dim=0)]

        states[batch_index, :path_count, valid_indices] = paths
        path_mask[batch_index, :path_count] = True
        log_probabilities[batch_index, :path_count] = log_probability
        probabilities[batch_index, :path_count] = probability
        marginals[batch_index, valid_indices] = compact_marginal
        map_states[batch_index, valid_indices] = best_path

    return ExactInferenceOutput(
        path_states=states,
        path_mask=path_mask,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        marginals=marginals,
        map_states=map_states,
    )
