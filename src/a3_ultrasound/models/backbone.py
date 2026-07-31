"""Time-index-preserving third-party backbone adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from pytorch_tcn import TCN
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BackboneOutput:
    """Backbone features and their exact input-sample support metadata."""

    features: Tensor
    feature_centers: Tensor
    feature_mask: Tensor


class LengthGroupedTCNBackbone(nn.Module):
    """Noncausal, same-length TCN with padding-safe length grouping.

    ``pytorch-tcn`` preserves temporal length but its biased convolutions can
    turn a zero-padded tail into nonzero hidden activations. Those activations
    can then leak back into the valid boundary of a shorter trace. This adapter
    groups traces by their valid prefix length, crops before invoking the
    third-party model, and pads only the final features.

    The adapter deliberately exposes the complete temporal feature map. It
    does not add global pooling, a classifier, or a task head.
    """

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 32),
        *,
        kernel_size: int = 5,
        dilations: Sequence[int] | None = (1, 2, 4),
        dropout: float = 0.0,
        use_norm: str | None = "weight_norm",
    ) -> None:
        super().__init__()
        channel_list = list(channels)
        if not channel_list or any(
            not isinstance(channel, int) or isinstance(channel, bool) or channel <= 0
            for channel in channel_list
        ):
            raise ValueError("channels must contain positive integers.")
        if (
            not isinstance(kernel_size, int)
            or isinstance(kernel_size, bool)
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        dilation_list = None if dilations is None else list(dilations)
        if dilation_list is not None and (
            len(dilation_list) != len(channel_list)
            or any(
                not isinstance(dilation, int)
                or isinstance(dilation, bool)
                or dilation <= 0
                for dilation in dilation_list
            )
        ):
            raise ValueError(
                "dilations must contain one positive integer per TCN block."
            )

        self.output_dim = channel_list[-1]
        self.tcn = TCN(
            num_inputs=1,
            num_channels=channel_list,
            kernel_size=kernel_size,
            dilations=dilation_list,
            dropout=dropout,
            causal=False,
            use_norm=use_norm,
            input_shape="NCL",
        )

    def forward(self, x_raw: Tensor, length_mask: Tensor) -> BackboneOutput:
        """Encode ``x_raw [B,T]`` without allowing padded tails into features."""

        if x_raw.ndim != 2:
            raise ValueError("x_raw must have shape [B, T].")
        if length_mask.shape != x_raw.shape or length_mask.dtype != torch.bool:
            raise ValueError("length_mask must be bool with the same [B, T] shape.")
        if x_raw.shape[1] == 0:
            raise ValueError("x_raw must contain at least one time sample.")

        valid_lengths = length_mask.sum(dim=1)
        if torch.any(valid_lengths == 0):
            raise ValueError("every trace must contain at least one valid sample.")
        expected_mask = (
            torch.arange(x_raw.shape[1], device=x_raw.device).unsqueeze(0)
            < valid_lengths.unsqueeze(1)
        )
        if not torch.equal(length_mask, expected_mask):
            raise ValueError("length_mask must describe one contiguous valid prefix.")

        batch_outputs: list[Tensor | None] = [None] * x_raw.shape[0]
        for valid_length_tensor in torch.unique(valid_lengths, sorted=True):
            valid_length = int(valid_length_tensor.item())
            batch_indices = torch.nonzero(
                valid_lengths == valid_length_tensor,
                as_tuple=False,
            ).squeeze(1)
            group_input = x_raw.index_select(0, batch_indices)[:, None, :valid_length]
            group_features = self.tcn(group_input).transpose(1, 2)
            group_features = F.pad(
                group_features,
                (0, 0, 0, x_raw.shape[1] - valid_length),
            )
            for group_index, batch_index in enumerate(batch_indices.tolist()):
                batch_outputs[batch_index] = group_features[group_index]

        if any(output is None for output in batch_outputs):
            raise RuntimeError("internal length grouping failed to encode every trace.")
        features = torch.stack(
            [output for output in batch_outputs if output is not None],
            dim=0,
        )
        feature_centers = torch.arange(
            x_raw.shape[1],
            device=x_raw.device,
            dtype=x_raw.dtype,
        )
        return BackboneOutput(
            features=features,
            feature_centers=feature_centers,
            feature_mask=length_mask,
        )


__all__ = ["BackboneOutput", "LengthGroupedTCNBackbone"]
