"""Conditional waveform memory for single-RF interface classification."""
import torch
from torch import Tensor, nn

from a3_ultrasound.models.pcc import HeadStepOutput, LocalBaseScorer, LocalBaseType, validate_prefix_states


class StateConditionedGRUStep(nn.Module):
    """Score from the local window and preceding history, then write W_s z."""

    def __init__(self, wave_dim: int, query_dim: int, token_dim: int, prefix_dim: int,
                 hidden_dim: int, local_base_type: LocalBaseType, local_base_hidden_dim: int):
        super().__init__()
        # As in LP, the classifier sees local evidence and interface position.
        self.local_token = nn.Sequential(nn.Linear(wave_dim + query_dim, token_dim), nn.LayerNorm(token_dim))
        self.base = LocalBaseScorer(token_dim, scorer_type=local_base_type, hidden_dim=local_base_hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(token_dim + prefix_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 1))
        self.write_zero = nn.Linear(wave_dim, prefix_dim, bias=False)
        self.write_one = nn.Linear(wave_dim, prefix_dim, bias=False)
        self.gru = nn.GRUCell(query_dim + prefix_dim, prefix_dim)

    def forward(self, r_loc: Tensor, r_ctx: Tensor, r_gate: Tensor, query: Tensor,
                prefix_states: Tensor, interface_mask: Tensor) -> HeadStepOutput:
        states = validate_prefix_states(prefix_states, interface_mask)
        # Only the actual local waveform vector enters the state-specific projection.
        previous = r_loc[:, :-1]
        writes = torch.where(states[:, :-1, None].bool(), self.write_one(previous),
                             self.write_zero(previous))
        inputs = torch.cat((query[:, :-1], writes), dim=-1)
        hidden = r_loc.new_zeros((r_loc.size(0), self.gru.hidden_size))
        history = [hidden]
        for index in range(inputs.size(1)):
            updated = self.gru(inputs[:, index], hidden)
            hidden = torch.where(interface_mask[:, index, None], updated, hidden)
            history.append(hidden)
        history = torch.stack(history, dim=1) * interface_mask.unsqueeze(-1)
        local = self.local_token(torch.cat((r_loc, query), dim=-1))
        logits = self.classifier(torch.cat((local, history), dim=-1)).squeeze(-1)
        return HeadStepOutput(logits=logits * interface_mask, base_logodds=self.base(local) * interface_mask,
                              prefix_hidden=history)
