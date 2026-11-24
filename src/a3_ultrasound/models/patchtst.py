"""PatchTST pre-head tokens expressed in original RF sample coordinates."""
import torch
from torch import nn
from torch.nn import functional as F
from tsai.models.PatchTST import _TSTiEncoder

from a3_ultrasound.models.backbone import TemporalBackboneOutput, _validate_temporal_inputs
from a3_ultrasound.speed import cpu_batch, fast_structure


class PatchTSTBackbone(nn.Module):
    """Reuse tsai's channel-independent PatchTST layers on length-grouped single RF."""

    output_dim = 128
    patch_len = 16
    stride = 4

    def __init__(self):
        super().__init__()
        # One shared position table for the variable lengths in this acquisition.
        self.encoder = _TSTiEncoder(1, 1024, self.patch_len, n_layers=3, d_model=self.output_dim,
                                    n_heads=16, d_ff=256, norm="LayerNorm", dropout=0.,
                                    res_attention=True)

    def forward(self, x_raw, length_mask):
        lengths = _validate_temporal_inputs(x_raw, length_mask)
        if x_raw.ndim == 3:
            if x_raw.size(1) != 1:
                raise ValueError("PatchTST expects single-channel RF.")
            x_raw = x_raw[:, 0]
        host = cpu_batch().length_mask.sum(1).tolist() if fast_structure() else lengths.tolist()
        groups = {}
        for row, length in enumerate(host):
            groups.setdefault(length, []).append(row)
        by_row = [None] * len(host)
        for length, rows in groups.items():
            indices = torch.tensor(rows, device=x_raw.device)
            waves = x_raw.index_select(0, indices)[:, :length]
            # Full patches plus one right-padded patch only for a remaining tail.
            steps = max(0, (length - self.patch_len + self.stride - 1) // self.stride)
            padding = steps * self.stride + self.patch_len - length
            patches = F.pad(waves, (0, padding)).unfold(-1, self.patch_len, self.stride)
            count = patches.size(1)
            if count > self.encoder.W_pos.size(0):
                raise ValueError("RF exceeds the registered PatchTST position table.")
            tokens = self.encoder.dropout(self.encoder.W_P(patches) + self.encoder.W_pos[:count])
            scores = None
            for layer in self.encoder.layers:
                tokens, scores = layer(tokens, prev=scores)
            for row, features in zip(rows, tokens.unbind()):
                by_row[row] = features
        features = nn.utils.rnn.pad_sequence(by_row, batch_first=True)
        starts = torch.arange(features.size(1), device=x_raw.device) * self.stride
        valid_counts = torch.div((lengths - self.patch_len).clamp_min(0) + self.stride - 1,
                                 self.stride, rounding_mode="floor") + 1
        mask = starts[None, :] < valid_counts[:, None] * self.stride
        # Tail centers refer to measured samples, not the padded part of a patch.
        ends = torch.minimum(starts + self.patch_len - 1, lengths[:, None] - 1)
        centers = (starts + ends).to(features.dtype) / 2
        return TemporalBackboneOutput(features, centers, mask, effective_stride=float(self.stride))
