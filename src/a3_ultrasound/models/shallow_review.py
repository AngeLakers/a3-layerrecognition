"""Frozen first-pass features; one parallel residual update at eligible deeper slots."""
import torch
from torch import nn


def eligible_donors(valid, initial_report, probability):
    confidence = torch.maximum(probability, 1-probability)
    trusted = valid & initial_report.ne(-1) & confidence.ge(.9)
    k = valid.size(1)
    prior = torch.ones(k, k, device=valid.device, dtype=torch.bool).tril(-1)
    return prior & trusted[:, None, :] & valid[:, :, None]


class ShallowReview(nn.Module):
    def __init__(self, dim, mode, hidden=32):
        super().__init__()
        if mode not in ("self", "scores", "features"):
            raise ValueError("Review mode must be self, scores or features.")
        self.mode = mode
        self.gate = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())
        self.update = nn.Sequential(nn.Linear(2*dim + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, h, logits, tau, valid, initial_report, calibrated_p, *, shallow_features=None):
        h, z, tau, p = (t.detach() for t in (h, logits, tau, calibrated_p))
        eligible = eligible_donors(valid, initial_report, p)
        dt = tau.unsqueeze(2) - tau.unsqueeze(1)
        prev_score = p[:, None, :].expand_as(dt)
        weight = self.gate(torch.stack((dt, prev_score), -1)).squeeze(-1)
        weight = weight.masked_fill(~eligible, 0) / eligible.sum(-1, keepdim=True).clamp_min(1)
        source = h if shallow_features is None else shallow_features.detach()
        ctx = weight @ source if self.mode == "features" else torch.zeros_like(h)
        soft = (weight * prev_score).sum(-1) if self.mode != "self" else torch.zeros_like(z)
        delta = self.update(torch.cat((h, ctx, soft.unsqueeze(-1), z.unsqueeze(-1)), -1)).squeeze(-1)
        active = valid & initial_report.eq(-1) & eligible.any(-1)
        return z + torch.where(active, delta, 0.), active

