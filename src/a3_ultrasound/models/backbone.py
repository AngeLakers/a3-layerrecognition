"""Replaceable temporal backbones and the A3 V1.2 feature adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from pytorch_tcn import TCN
from torch import Tensor, nn
from a3_ultrasound.speed import fast_structure,cpu_batch


@dataclass(frozen=True)
class TemporalBackboneOutput:
    """Temporal features expressed in original-RF sample coordinates."""

    features: Tensor
    feature_centers: Tensor
    feature_mask: Tensor
    effective_stride: float | None = None
    theoretical_rf_size: float | None = None

    @property
    def temporal_length(self) -> int:
        return int(self.features.shape[1])


# Backward-compatible public name used by the V1.1 review packet and scripts.
BackboneOutput = TemporalBackboneOutput


class CommonTemporalBackboneAdapter(nn.Module):
    """Project only feature channels while preserving temporal geometry."""

    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (input_dim, output_dim)
        ):
            raise ValueError("Adapter dimensions must be positive integers.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_projection = nn.Linear(input_dim, output_dim)

    def forward(self, output: TemporalBackboneOutput) -> TemporalBackboneOutput:
        if output.features.ndim != 3 or output.features.shape[-1] != self.input_dim:
            raise ValueError(
                "Backbone features must have shape [B, L, input_dim]."
            )
        batch_size, temporal_length, _ = output.features.shape
        if output.feature_mask.shape != (batch_size, temporal_length):
            raise ValueError("feature_mask must agree with backbone features on [B, L].")
        if output.feature_mask.dtype != torch.bool:
            raise TypeError("feature_mask must be bool.")
        if output.feature_centers.ndim == 1:
            if output.feature_centers.shape[0] != temporal_length:
                raise ValueError("feature_centers must agree with temporal length L.")
        elif output.feature_centers.shape != (batch_size, temporal_length):
            raise ValueError("feature_centers must have shape [L] or [B, L].")

        return TemporalBackboneOutput(
            features=self.channel_projection(output.features),
            feature_centers=output.feature_centers,
            feature_mask=output.feature_mask,
            effective_stride=output.effective_stride,
            theoretical_rf_size=output.theoretical_rf_size,
        )


def _validate_temporal_inputs(x_raw: Tensor, length_mask: Tensor) -> Tensor:
    if x_raw.ndim != 2:
        raise ValueError("x_raw must have shape [B, T].")
    if length_mask.shape != x_raw.shape or length_mask.dtype != torch.bool:
        raise ValueError("length_mask must be bool with the same [B, T] shape.")
    if x_raw.shape[1] == 0:
        raise ValueError("x_raw must contain at least one time sample.")

    valid_lengths = length_mask.sum(dim=1)
    if not fast_structure() and torch.any(valid_lengths == 0):
        raise ValueError("every trace must contain at least one valid sample.")
    expected_mask = (
        torch.arange(x_raw.shape[1], device=x_raw.device).unsqueeze(0)
        < valid_lengths.unsqueeze(1)
    )
    if not fast_structure() and not torch.equal(length_mask, expected_mask):
        raise ValueError("length_mask must describe one contiguous valid prefix.")
    return valid_lengths


def _encode_by_valid_length(
    x_raw: Tensor,
    length_mask: Tensor,
    encoder: nn.Module,
    *,
    output_dim: int,
    canonical_per_trace: bool = False,
) -> TemporalBackboneOutput:
    """Crop before encoding so padded tails cannot affect valid features."""

    valid_lengths = _validate_temporal_inputs(x_raw, length_mask)
    encoded_by_sample: list[Tensor | None] = [None] * x_raw.shape[0]
    centers_by_sample: list[Tensor | None] = [None] * x_raw.shape[0]

    host_lengths=cpu_batch().length_mask.sum(1).tolist() if fast_structure() else None
    groups=sorted(set(host_lengths)) if host_lengths is not None else torch.unique(valid_lengths,sorted=True)
    for valid_length_tensor in groups:
        valid_length = int(valid_length_tensor) if host_lengths is not None else int(valid_length_tensor.item())
        batch_indices = (torch.tensor([i for i,n in enumerate(host_lengths) if n==valid_length],device=x_raw.device)
            if host_lengths is not None else torch.nonzero(valid_lengths==valid_length_tensor,as_tuple=False).squeeze(1))
        group_input = x_raw.index_select(0, batch_indices)[:, None, :valid_length]
        # GN is sample-independent mathematically. Use identical trunk call
        # shapes for bit-stable FP32 features across batch compositions; some
        # batched convolution kernels otherwise differ by >1e-5 on real RF.
        group_features = (torch.cat([encoder(x.unsqueeze(0)) for x in group_input],dim=0)
                          if canonical_per_trace else encoder(group_input))
        if group_features.ndim != 3:
            raise RuntimeError("Temporal backbone trunk must return [B, C, L].")
        if group_features.shape[1] != output_dim:
            raise RuntimeError("Temporal backbone output channels disagree with output_dim.")
        group_features = group_features.transpose(1, 2)
        # V1.2 formal candidates are deliberately time-preserving. A future
        # downsampled wrapper must provide its own exact center propagation.
        if group_features.shape[1] != valid_length:
            raise RuntimeError(
                "This wrapper requires a time-preserving pre-pooling feature stage."
            )
        centers = torch.arange(
            valid_length,
            device=x_raw.device,
            dtype=x_raw.dtype,
        )
        host_indices=[i for i,n in enumerate(host_lengths) if n==valid_length] if host_lengths is not None else batch_indices.tolist()
        for group_index, batch_index in enumerate(host_indices):
            encoded_by_sample[batch_index] = group_features[group_index]
            centers_by_sample[batch_index] = centers

    if any(value is None for value in encoded_by_sample) or any(
        value is None for value in centers_by_sample
    ):
        raise RuntimeError("Length grouping failed to encode every trace.")

    max_length = max(int(value.shape[0]) for value in encoded_by_sample if value is not None)
    features = x_raw.new_zeros((x_raw.shape[0], max_length, output_dim))
    feature_centers = x_raw.new_zeros((x_raw.shape[0], max_length))
    feature_mask = torch.zeros(
        (x_raw.shape[0], max_length),
        device=x_raw.device,
        dtype=torch.bool,
    )
    for index, (sample_features, sample_centers) in enumerate(
        zip(encoded_by_sample, centers_by_sample, strict=True)
    ):
        assert sample_features is not None and sample_centers is not None
        sample_length = sample_features.shape[0]
        features[index, :sample_length] = sample_features
        feature_centers[index, :sample_length] = sample_centers
        feature_mask[index, :sample_length] = True

    return TemporalBackboneOutput(
        features=features,
        feature_centers=feature_centers,
        feature_mask=feature_mask,
        effective_stride=1.0,
    )


class LengthGroupedTCNBackbone(nn.Module):
    """Padding-safe TCN retained only as a lightweight smoke backbone."""

    training_role = "smoke_only"
    feature_stage = "final_pre_pool"

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

    def forward(self, x_raw: Tensor, length_mask: Tensor) -> TemporalBackboneOutput:
        return _encode_by_valid_length(
            x_raw,
            length_mask,
            self.tcn,
            output_dim=self.output_dim,
        )


TsaiBackboneName = Literal["resnet1d", "inceptiontime", "fcn1d"]
TsaiFeatureStage = Literal["stage1", "stage2", "stage3", "final_pre_pool"]


class TsaiTimePreservingBackbone(nn.Module):
    """Expose the final pre-pooling temporal trunk from a tsai classifier."""

    training_role = "formal_candidate"
    def __init__(
        self,
        name: TsaiBackboneName,
        *,
        feature_stage: TsaiFeatureStage = "final_pre_pool",
        inception_nf: int = 32,
        fcn_layers: Sequence[int] = (128, 256, 128),
        backbone_norm: str = "batch",
        canonical_per_trace: bool = True,
    ) -> None:
        super().__init__()
        resolved_stage = "stage3" if feature_stage == "final_pre_pool" else feature_stage
        stage_index = {"stage1": 1, "stage2": 2, "stage3": 3}.get(resolved_stage)
        if stage_index is None:
            raise ValueError("feature_stage must be stage1/stage2/stage3/final_pre_pool.")
        if name == "resnet1d":
            from tsai.models.ResNet import ResNet

            classifier = ResNet(1, 2)
            available_modules = (
                classifier.resblock1,
                classifier.resblock2,
                classifier.resblock3,
            )
            output_dims = (64, 128, 128)
        elif name == "inceptiontime":
            from tsai.models.InceptionTime import InceptionTime

            classifier = InceptionTime(1, 2, nf=inception_nf)
            if resolved_stage != "stage3":
                raise ValueError(
                    "The current InceptionTime wrapper exposes only final_pre_pool."
                )
            available_modules = (classifier.inceptionblock,)
            output_dims = (int(classifier.fc.in_features),)
            stage_index = 1
        elif name == "fcn1d":
            from tsai.models.FCN import FCN

            layer_list = list(fcn_layers)
            if len(layer_list) != 3 or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in layer_list
            ):
                raise ValueError("fcn_layers must contain three positive integers.")
            classifier = FCN(1, 2, layers=layer_list)
            available_modules = (
                classifier.convblock1,
                classifier.convblock2,
                classifier.convblock3,
            )
            output_dims = tuple(layer_list)
        else:
            raise ValueError(f"Unsupported tsai backbone: {name!r}.")

        if not isinstance(classifier.fc, nn.Linear):
            raise RuntimeError("tsai classifier head no longer exposes a Linear fc layer.")
        self.name = name
        self.feature_stage = feature_stage
        self.output_dim = int(output_dims[stage_index - 1])
        self.trunk = nn.Sequential(*available_modules[:stage_index])
        if backbone_norm not in ("batch", "group8"):
            raise ValueError("backbone_norm must be batch or group8.")
        self.backbone_norm = backbone_norm
        if type(canonical_per_trace) is not bool:
            raise ValueError("canonical_per_trace must be bool.")
        self.canonical_per_trace = canonical_per_trace
        if backbone_norm == "group8":
            self._replace_batch_norm(self.trunk)

    @staticmethod
    def _replace_batch_norm(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                if child.num_features % 8:
                    raise ValueError("group8 requires channels divisible by 8.")
                norm = nn.GroupNorm(8, child.num_features, eps=1e-5, affine=True)
                if child.affine:
                    with torch.no_grad():
                        norm.weight.copy_(child.weight)
                        norm.bias.copy_(child.bias)
                setattr(module, name, norm)
            else:
                TsaiTimePreservingBackbone._replace_batch_norm(child)

    def forward(self, x_raw: Tensor, length_mask: Tensor) -> TemporalBackboneOutput:
        return _encode_by_valid_length(
            x_raw,
            length_mask,
            self.trunk,
            output_dim=self.output_dim,
            canonical_per_trace=self.backbone_norm == "group8" and self.canonical_per_trace,
        )


__all__ = [
    "BackboneOutput",
    "CommonTemporalBackboneAdapter",
    "LengthGroupedTCNBackbone",
    "TemporalBackboneOutput",
    "TsaiBackboneName",
    "TsaiFeatureStage",
    "TsaiTimePreservingBackbone",
]
