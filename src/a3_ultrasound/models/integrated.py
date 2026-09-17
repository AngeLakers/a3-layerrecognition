"""Integrated synthetic-forward contract for the A3 PCC model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from a3_ultrasound.inference.enumerator import ExactInferenceOutput, enumerate_pcc
from a3_ultrasound.models.backbone import (
    BackboneOutput,
    CommonTemporalBackboneAdapter,
)
from a3_ultrasound.models.indexed_readout import (
    IndexedReadout,
    LocalTokenDiagnostics,
    ReadoutOutput,
    build_indexed_supports,
    summarize_local_tokens,
)
from a3_ultrasound.models.pcc import LocalBaseType, PCCConditionalStep, PCCStepOutput
from a3_ultrasound.models.query import InterfaceQueryEncoder
from a3_ultrasound.models.responsibility_tokens import (
    ResponsibilityTokenProjector,
    ResponsibilityTokens,
)
from a3_ultrasound.schema import InterfaceBatch


@dataclass(frozen=True)
class IntegratedPCCOutput:
    """Conditional forward outputs plus the tensors needed for structure audits."""

    backbone: BackboneOutput
    query: Tensor
    window_start: Tensor
    window_end: Tensor
    local_support: Tensor
    pre_support: Tensor
    local_prior: Tensor
    local_token_diagnostics: LocalTokenDiagnostics
    local_readout: ReadoutOutput
    pre_readout: ReadoutOutput
    responsibility_tokens: ResponsibilityTokens
    conditional: PCCStepOutput


class A3PCCModel(nn.Module):
    """Compose the time-indexed Backbone, Query, Readouts, projector, and PCC.

    The current M1 contract intentionally uses the input RF tensor unchanged.
    A frozen phase-preserving preprocessing rule can be placed before this
    model later; this class does not estimate normalization statistics.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        query_dim: int,
        attention_dim: int,
        token_dim: int,
        prefix_dim: int,
        pcc_hidden_dim: int,
        common_feature_dim: int = 64,
        max_interfaces: int = 6,
        rho: float = 1.0,
        local_base_type: LocalBaseType = "mlp",
        local_base_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (
                query_dim,
                attention_dim,
                token_dim,
                prefix_dim,
                pcc_hidden_dim,
                common_feature_dim,
                max_interfaces,
            )
        ):
            raise ValueError("All model dimensions and max_interfaces must be positive integers.")

        feature_dim = getattr(backbone, "output_dim", None)
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("backbone must expose one positive integer output_dim.")
        self.max_interfaces = max_interfaces
        self.backbone = backbone
        self.backbone_adapter = CommonTemporalBackboneAdapter(
            feature_dim,
            common_feature_dim,
        )
        self.query_encoder = InterfaceQueryEncoder(query_dim=query_dim)
        self.local_readout = IndexedReadout(
            feature_dim=common_feature_dim,
            query_dim=query_dim,
            attention_dim=attention_dim,
        )
        self.pre_readout = IndexedReadout(
            feature_dim=common_feature_dim,
            query_dim=None,
            attention_dim=attention_dim,
        )
        self.responsibility_projector = ResponsibilityTokenProjector(
            local_dim=common_feature_dim,
            context_dim=common_feature_dim,
            query_dim=query_dim,
            token_dim=token_dim,
        )
        self.pcc = PCCConditionalStep(
            local_dim=token_dim,
            context_dim=token_dim,
            gate_dim=token_dim,
            query_dim=query_dim,
            prefix_dim=prefix_dim,
            hidden_dim=pcc_hidden_dim,
            max_interfaces=max_interfaces,
            rho=rho,
            local_base_type=local_base_type,
            local_base_hidden_dim=local_base_hidden_dim,
        )

    def forward(
        self,
        batch: InterfaceBatch,
        prefix_states: Tensor,
    ) -> IntegratedPCCOutput:
        """Run one truth-prefix or explicitly supplied conditional path."""

        self._validate_structural_inputs(batch, prefix_states)
        raw_backbone_output = self.backbone(batch.x_raw, batch.length_mask)
        backbone_output = self.backbone_adapter(raw_backbone_output)
        query = self.query_encoder(batch.interface_meta, batch.interface_mask)
        window_start, window_end, nominal_tof = self._sample_windows(batch)
        local_support, pre_support, local_prior = build_indexed_supports(
            backbone_output.feature_centers,
            window_start,
            window_end,
            backbone_output.feature_mask,
            batch.interface_mask,
            prior_center=nominal_tof,
        )
        local_token_diagnostics = summarize_local_tokens(
            local_support,
            batch.interface_mask,
        )
        local_output = self.local_readout(
            backbone_output.features,
            query,
            local_support,
            batch.interface_mask,
            local_prior=local_prior,
        )
        pre_output = self.pre_readout(
            backbone_output.features,
            None,
            pre_support,
            batch.interface_mask,
            allow_empty_context=True,
        )
        tokens = self.responsibility_projector(
            local_output.token,
            pre_output.token,
            query,
            batch.interface_mask,
        )
        conditional = self.pcc(
            tokens.r_loc,
            tokens.r_ctx,
            tokens.r_gate,
            query,
            prefix_states,
            batch.interface_mask,
        )
        return IntegratedPCCOutput(
            backbone=backbone_output,
            query=query,
            window_start=window_start,
            window_end=window_end,
            local_support=local_support,
            pre_support=pre_support,
            local_prior=local_prior,
            local_token_diagnostics=local_token_diagnostics,
            local_readout=local_output,
            pre_readout=pre_output,
            responsibility_tokens=tokens,
            conditional=conditional,
        )

    def infer_exact(self, batch: InterfaceBatch) -> ExactInferenceOutput:
        """Run exact inference for the engineering range 1 <= K <= 6."""

        reference_prefix = torch.zeros_like(batch.interface_mask, dtype=torch.long)
        encoded = self.forward(batch, reference_prefix)
        tokens = encoded.responsibility_tokens
        return enumerate_pcc(
            self.pcc,
            tokens.r_loc,
            tokens.r_ctx,
            tokens.r_gate,
            encoded.query,
            batch.interface_mask,
        )

    def _validate_structural_inputs(
        self,
        batch: InterfaceBatch,
        prefix_states: Tensor,
    ) -> None:
        batch_size, kmax = batch.interface_mask.shape
        if kmax > self.max_interfaces:
            raise ValueError(
                f"Kmax={kmax} exceeds the configured boundary of {self.max_interfaces}."
            )
        valid_counts = batch.interface_mask.sum(dim=1)
        expected_mask = (
            torch.arange(kmax, device=batch.interface_mask.device).unsqueeze(0)
            < valid_counts.unsqueeze(1)
        )
        if not torch.equal(batch.interface_mask, expected_mask):
            raise ValueError("interface_mask must describe one contiguous ordered prefix.")
        if prefix_states.shape != (batch_size, kmax):
            raise ValueError("prefix_states must have shape [B, Kmax].")
        if prefix_states.device != batch.interface_mask.device:
            raise ValueError("prefix_states and batch tensors must share a device.")
        valid_states = prefix_states[batch.interface_mask]
        if torch.is_floating_point(valid_states) or torch.any(
            (valid_states != 0) & (valid_states != 1)
        ):
            raise ValueError("prefix_states must be binary on valid interface slots.")

    @staticmethod
    def _sample_windows(batch: InterfaceBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Invert the collate normalization into raw-sample window coordinates."""

        valid_lengths = batch.length_mask.sum(dim=1).to(dtype=batch.interface_meta.dtype)
        scale = valid_lengths.unsqueeze(1)
        tau = batch.interface_meta[..., 1] * scale
        left = batch.interface_meta[..., 3] * scale
        right = batch.interface_meta[..., 4] * scale
        window_start = tau - left
        window_end = tau + right

        valid = batch.interface_mask
        last_sample = valid_lengths.unsqueeze(1) - 1.0
        invalid_window = (
            (window_start < -1e-4)
            | (window_end > last_sample + 1e-4)
            | (window_start > window_end)
        )
        if torch.any(valid & invalid_window):
            raise ValueError(
                "Every valid interface window must lie inside its trace and have start <= end."
            )
        return window_start, window_end, tau


__all__ = ["A3PCCModel", "IntegratedPCCOutput"]
