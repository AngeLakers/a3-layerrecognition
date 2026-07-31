"""Feature-center indexed Local and Pre-interface masked-attention Readout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ReadoutOutput:
    token: Tensor
    attention: Tensor


def conv1d_feature_centers(
    input_length: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    device: torch.device | None = None,
) -> Tensor:
    """Return exact receptive-field centres for one Conv1d layer.

    Centres are derived from kernel/stride/padding/dilation rather than an
    approximate downsampling ratio. Stacked-backbone composition is deferred
    until its kernel stack is frozen by RF/environment smoke.
    """

    if min(input_length, kernel_size, stride, dilation) <= 0 or padding < 0:
        raise ValueError("Conv1d geometry values must be valid positive integers.")
    receptive_width = dilation * (kernel_size - 1) + 1
    output_length = (input_length + 2 * padding - receptive_width) // stride + 1
    if output_length <= 0:
        raise ValueError("Conv1d geometry produces no output features.")
    index = torch.arange(output_length, device=device, dtype=torch.float32)
    return index * stride - padding + (receptive_width - 1) / 2.0


def build_indexed_supports(
    feature_centers: Tensor,
    window_start: Tensor,
    window_end: Tensor,
    feature_mask: Tensor,
    interface_mask: Tensor,
    *,
    prior_center: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build Local/Pre masks and a normalized Gaussian local prior.

    ``window_start`` and ``window_end`` are already frozen TOF/data-contract
    quantities. All masks are feature-centre masks; this does not claim raw-RF
    causal isolation for a noncausal backbone.
    """

    if window_start.shape != window_end.shape or window_start.ndim != 2:
        raise ValueError("window bounds must have shape [B, Kmax].")
    batch_size, kmax = window_start.shape
    if feature_mask.ndim != 2 or feature_mask.shape[0] != batch_size:
        raise ValueError("feature_mask must have shape [B, L].")
    if feature_mask.dtype != torch.bool or interface_mask.shape != (batch_size, kmax):
        raise ValueError("feature/interface masks have incompatible shapes.")
    if interface_mask.dtype != torch.bool:
        raise TypeError("interface_mask must be bool.")
    if feature_centers.ndim == 1:
        centers = feature_centers.view(1, 1, -1)
    elif feature_centers.ndim == 2 and feature_centers.shape[0] == batch_size:
        centers = feature_centers.unsqueeze(1)
    else:
        raise ValueError("feature_centers must have shape [L] or [B, L].")
    if centers.shape[-1] != feature_mask.shape[1]:
        raise ValueError("feature_centers and feature_mask disagree on L.")

    starts = window_start.unsqueeze(-1)
    ends = window_end.unsqueeze(-1)
    if prior_center is None:
        centers_of_prior = (starts + ends) / 2.0
    else:
        if prior_center.shape != window_start.shape:
            raise ValueError("prior_center must have shape [B, Kmax].")
        centers_of_prior = prior_center.unsqueeze(-1)
    valid_feature = feature_mask.unsqueeze(1)
    valid_interface = interface_mask.unsqueeze(-1)
    local_mask = (centers >= starts) & (centers <= ends) & valid_feature & valid_interface
    pre_mask = (centers < starts) & valid_feature & valid_interface

    scale = ((ends - starts).abs() / 2.0).clamp_min(1.0)
    gaussian = torch.exp(-0.5 * ((centers - centers_of_prior) / scale).square())
    local_prior = gaussian * local_mask.to(dtype=gaussian.dtype)
    local_prior = local_prior / local_prior.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gaussian.dtype).eps)
    return local_mask, pre_mask, local_prior


class IndexedReadout(nn.Module):
    """One masked-attention branch with explicit Query responsibility.

    Local Readout sets ``query_dim`` and requires Query. Pre-interface Readout
    sets ``query_dim=None`` and rejects Query, so its attention score is
    feature-only inside the target-specific support.
    """

    def __init__(
        self,
        feature_dim: int,
        query_dim: int | None,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_projection = (
            nn.Linear(query_dim, attention_dim, bias=False)
            if query_dim is not None
            else None
        )
        self.score_projection = nn.Linear(attention_dim, 1, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(attention_dim))
        self.empty_context = nn.Parameter(torch.zeros(feature_dim))

    def forward(
        self,
        features: Tensor,
        query: Tensor | None,
        support_mask: Tensor,
        interface_mask: Tensor,
        *,
        local_prior: Tensor | None = None,
        allow_empty_context: bool = False,
    ) -> ReadoutOutput:
        if features.ndim != 3 or support_mask.ndim != 3:
            raise ValueError("features/support_mask must have shapes [B, L, D] and [B, K, L].")
        batch_size, feature_count, _ = features.shape
        kmax = support_mask.shape[1]
        if support_mask.shape != (batch_size, kmax, feature_count):
            raise ValueError("support_mask must have shape [B, Kmax, L].")
        if interface_mask.shape != (batch_size, kmax) or interface_mask.dtype != torch.bool:
            raise ValueError("interface_mask must be bool with shape [B, Kmax].")
        if self.query_projection is None:
            if query is not None:
                raise ValueError("Query-free Pre-interface Readout must receive query=None.")
        elif query is None or query.ndim != 3 or query.shape[:2] != (batch_size, kmax):
            raise ValueError("Query-conditioned Local Readout requires query [B, Kmax, Dq].")
        if support_mask.dtype != torch.bool:
            raise TypeError("support_mask must be bool.")
        if local_prior is not None and local_prior.shape != support_mask.shape:
            raise ValueError("local_prior must have shape [B, Kmax, L].")

        valid = support_mask & interface_mask.unsqueeze(-1)
        has_support = valid.any(dim=-1)
        if not allow_empty_context and torch.any(interface_mask & ~has_support):
            raise ValueError("An effective Local support is empty; check TOF/stride contract.")

        hidden = self.feature_projection(features).unsqueeze(1) + self.score_bias
        if self.query_projection is not None:
            assert query is not None
            hidden = hidden + self.query_projection(query).unsqueeze(2)
        scores = self.score_projection(torch.tanh(hidden)).squeeze(-1)
        if local_prior is not None:
            eps = torch.finfo(scores.dtype).eps
            scores = scores + torch.log(local_prior.clamp_min(eps))
        masked_scores = scores.masked_fill(~valid, float("-inf"))
        attention = torch.softmax(masked_scores, dim=-1)
        attention = torch.where(has_support.unsqueeze(-1), attention, torch.zeros_like(attention))
        token = torch.einsum("bkl,bld->bkd", attention, features)

        empty_context = interface_mask & ~has_support
        if allow_empty_context:
            token = torch.where(empty_context.unsqueeze(-1), self.empty_context.view(1, 1, -1), token)
        token = token * interface_mask.unsqueeze(-1).to(dtype=token.dtype)
        return ReadoutOutput(token=token, attention=attention)
