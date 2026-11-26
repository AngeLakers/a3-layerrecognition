"""Expected unit FN/FP and equal-count location errors of independent binary heads."""
import torch
from torch import Tensor


def combination_cost(logits: Tensor, truth: Tensor, valid: Tensor, *, exchange: bool) -> Tensor:
    """Return per-record cost / K. Padding contributes neither counts nor gradients."""
    probabilities = logits.float().masked_fill(~valid, 0).sigmoid().masked_fill(~valid, 0)
    positive = truth.eq(1) & valid
    errors = torch.where(positive, 1 - probabilities, probabilities)
    cost = errors.sum(dim=1)
    if exchange:
        # Equal predicted and true cardinality, excluding the exactly correct state.
        mass = torch.ones_like(probabilities[:, :1])
        for p in probabilities.unbind(dim=1):
            p = p[:, None]
            zero = torch.zeros_like(p)
            mass = (torch.cat((mass * (1 - p), zero), 1)
                    + torch.cat((zero, mass * p), 1))
        same_count = mass.gather(1, positive.sum(1, keepdim=True)).squeeze(1)
        exact = (1 - errors).prod(1)
        cost = cost + (same_count - exact).clamp_min(0)
    return cost / valid.sum(1)

