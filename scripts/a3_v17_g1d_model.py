"""Generic waveform readout with three learned queries and no physical metadata."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace
from torch.nn import functional as F

import torch
from torch import Tensor, nn

from a3_ultrasound.inference.enumerator import enumerate_pcc
from a3_ultrasound.models.baselines import IndependentConditionalStep
from a3_ultrasound.models.pcc import HeadStepOutput
from a3_ultrasound.schema import InterfaceBatch


@dataclass(frozen=True)
class G1DEncoding:
    interface_mask: Tensor
    tokens: Tensor
    attention: Tensor
    local_token_diagnostics: object


@dataclass(frozen=True)
class G1DOutput:
    encoding: G1DEncoding
    conditional: HeadStepOutput


class G1DModel(nn.Module):
    """Pool the whole valid waveform independently for each learned output query.

    The supplied backbone is resolved by the explicit experiment configuration.
    Only RF, RF validity and the K=2/3 output mask enter this model.
    """

    head_type = "g1d"
    max_interfaces = 3

    def __init__(self, backbone: nn.Module, *, attention_dim: int = 64,
                 hidden_dim: int = 32) -> None:
        super().__init__()
        feature_dim = backbone.output_dim
        self.backbone = backbone
        self.queries = nn.Embedding(self.max_interfaces, attention_dim)
        self.keys = nn.Linear(feature_dim, attention_dim)
        self.head = IndependentConditionalStep(feature_dim, hidden_dim=hidden_dim)
        self.attention_dim = attention_dim

    def encode(self, batch: InterfaceBatch) -> G1DEncoding:
        mask = batch.interface_mask
        if mask.ndim != 2 or mask.dtype != torch.bool or not 2 <= mask.shape[1] <= 6:
            raise ValueError("G1D expects a bool padded mask with 2/3 active interfaces.")
        counts = mask.sum(1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < counts[:, None]
        if not torch.equal(mask, expected) or torch.any((counts < 2) | (counts > 3)):
            raise ValueError("G1D requires an ordered K=2/3 output mask.")
        encoded = self.backbone(batch.x_raw, batch.length_mask)
        keys = self.keys(encoded.features)
        query_count = min(3, mask.shape[1])
        query = self.queries.weight[:query_count]
        scores = torch.einsum("bld,kd->bkl", keys, query) / sqrt(self.attention_dim)
        scores = scores.masked_fill(~encoded.feature_mask[:, None], -torch.inf)
        attention = scores.softmax(-1) * mask[:, :query_count, None]
        if mask.shape[1] > query_count:
            attention = F.pad(attention, (0, 0, 0, mask.shape[1]-query_count))
        tokens = torch.einsum("bkl,bld->bkd", attention, encoded.features)
        counts = encoded.feature_mask.sum(1)[:, None].expand_as(mask) * mask
        return G1DEncoding(mask, tokens, attention, SimpleNamespace(counts=counts))

    def conditional(self, encoded: G1DEncoding, prefix_states: Tensor) -> HeadStepOutput:
        tokens = encoded.tokens
        return self.head(tokens, tokens, tokens, tokens, prefix_states, encoded.interface_mask)

    def forward(self, batch: InterfaceBatch, prefix_states: Tensor) -> G1DOutput:
        encoded = self.encode(batch)
        return G1DOutput(encoded, self.conditional(encoded, prefix_states))

    def infer_encoded(self, encoded: G1DEncoding):
        tokens = encoded.tokens
        return enumerate_pcc(self.head, tokens, tokens, tokens, tokens, encoded.interface_mask)

    def infer_exact(self, batch: InterfaceBatch):
        return self.infer_encoded(self.encode(batch))
