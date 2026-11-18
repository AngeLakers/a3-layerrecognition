"""Shared frontend and replaceable conditional heads for A3 V1.3."""

from __future__ import annotations
from a3_ultrasound.speed import fast_structure,options

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from a3_ultrasound.inference.enumerator import ExactInferenceOutput, enumerate_pcc
from a3_ultrasound.models.backbone import (
    BackboneOutput,
    CommonTemporalBackboneAdapter,
)
from a3_ultrasound.models.baselines import (
    IndependentConditionalStep, LocalPreIndependentStep, VanillaClassifierChainStep,
)
from a3_ultrasound.models.indexed_readout import (
    IndexedReadout,
    LocalTokenDiagnostics,
    ReadoutOutput,
    build_indexed_supports,
    summarize_local_tokens,
)
from a3_ultrasound.models.pcc import (
    HeadStepOutput, LocalBaseType, PCCConditionalStep, validate_prefix_states,
)
from a3_ultrasound.models.query import InterfaceQueryEncoder
from a3_ultrasound.models.responsibility_tokens import (
    ResponsibilityTokenProjector,
    ResponsibilityTokens,
)
from a3_ultrasound.schema import InterfaceBatch


HeadType = Literal["local_independent", "local_pre_independent", "vanilla_chain", "proposed", "proposed_shadow_aux", "proposed_v15"]
HEAD_TYPES = ("local_independent", "local_pre_independent", "vanilla_chain", "proposed", "proposed_shadow_aux", "proposed_v15")


@dataclass(frozen=True)
class EncodedInterfaces:
    """Label-free features and support diagnostics reusable across candidate paths."""

    interface_mask: Tensor
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


@dataclass(frozen=True)
class IntegratedPCCOutput(EncodedInterfaces):
    conditional: HeadStepOutput


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
        head_type: HeadType = "proposed",
        readout_scope: Literal["indexed", "full_waveform"] = "indexed",
        pre_query_conditioned: bool = False,
        local_prior_sigma_samples: float | None = None,
        local_prior_strength: float = 1.0,
        query_view: str = "position2",
        physical_features: bool = False,
        pre_exclude_front_end: int | None = None,
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
        if not 1 <= max_interfaces <= 6:
            raise ValueError("max_interfaces must lie in [1, 6].")
        if head_type not in HEAD_TYPES:
            raise ValueError(f"Unknown head_type: {head_type!r}.")
        if readout_scope not in ("indexed", "full_waveform"):
            raise ValueError("readout_scope must be indexed or full_waveform.")
        if readout_scope == "full_waveform" and head_type != "local_independent":
            raise ValueError("full_waveform ablation is restricted to local_independent.")
        self.head_type = head_type
        self.physical_features = physical_features
        self.pre_exclude_front_end = pre_exclude_front_end
        self.readout_scope = readout_scope
        self.pre_query_conditioned = pre_query_conditioned
        self.local_prior_sigma_samples = local_prior_sigma_samples
        self.backbone = backbone
        self.backbone_adapter = CommonTemporalBackboneAdapter(
            feature_dim,
            common_feature_dim,
        )
        self.query_encoder = InterfaceQueryEncoder(query_dim=query_dim, query_view=query_view)
        self.local_readout = IndexedReadout(
            feature_dim=common_feature_dim,
            query_dim=query_dim,
            attention_dim=attention_dim,
            prior_strength=local_prior_strength,
        )
        self.pre_readout = IndexedReadout(
            feature_dim=common_feature_dim,
            query_dim=query_dim if pre_query_conditioned else None,
            attention_dim=attention_dim,
            owns_empty_context=True,
        )
        self.responsibility_projector = ResponsibilityTokenProjector(
            local_dim=common_feature_dim,
            context_dim=common_feature_dim,
            query_dim=query_dim,
            token_dim=token_dim,
        )
        local_dim = token_dim + int(physical_features)
        context_dim = token_dim + 3*int(physical_features)
        gate_dim = token_dim + 4*int(physical_features)
        common = dict(local_dim=local_dim, local_base_type=local_base_type,
                      local_base_hidden_dim=local_base_hidden_dim)
        # Retain the 'pcc' module key for compatibility with historical state_dicts.
        if head_type == "local_independent":
            self.pcc = IndependentConditionalStep(
                local_dim, hidden_dim=local_base_hidden_dim, local_base_type=local_base_type)
        elif head_type == "local_pre_independent":
            self.pcc = LocalPreIndependentStep(
                **common, context_dim=context_dim, hidden_dim=pcc_hidden_dim)
        elif head_type == "vanilla_chain":
            self.pcc = VanillaClassifierChainStep(
                **common, context_dim=context_dim, query_dim=query_dim,
                prefix_dim=prefix_dim, hidden_dim=pcc_hidden_dim, max_interfaces=max_interfaces)
        else:
            self.pcc = PCCConditionalStep(
                local_dim=local_dim,
                context_dim=context_dim,
                gate_dim=gate_dim,
                query_dim=query_dim,
                prefix_dim=prefix_dim,
                hidden_dim=pcc_hidden_dim,
                max_interfaces=max_interfaces,
                rho=rho,
                local_base_type=local_base_type,
                local_base_hidden_dim=local_base_hidden_dim,
                correction_mode="suppress_only" if head_type == "proposed_v15" else "bidirectional",
            )

    def forward(
        self,
        batch: InterfaceBatch,
        prefix_states: Tensor,
    ) -> IntegratedPCCOutput:
        """Run one truth-prefix or explicitly supplied conditional path."""

        encoded = self.encode(batch)
        return IntegratedPCCOutput(**vars(encoded), conditional=self.conditional(encoded, prefix_states))

    @property
    def head(self) -> nn.Module:
        return self.pcc

    def encode(self, batch: InterfaceBatch) -> EncodedInterfaces:
        """Encode only waveform, structure and validity masks, never labels."""
        self._validate_structural_inputs(batch)
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
            prior_sigma_samples=self.local_prior_sigma_samples,
            pre_exclude_front_end=self.pre_exclude_front_end,
        )
        if self.readout_scope == "full_waveform":
            local_support = (backbone_output.feature_mask.unsqueeze(1)
                             & batch.interface_mask.unsqueeze(-1))
            local_prior = local_support.to(backbone_output.features.dtype)
            local_prior = local_prior / local_prior.sum(-1, keepdim=True).clamp_min(1)
        local_token_diagnostics = summarize_local_tokens(local_support, batch.interface_mask)
        local_output = self.local_readout(
            backbone_output.features,
            query,
            local_support,
            batch.interface_mask,
            local_prior=local_prior if self.readout_scope == "indexed" else None,
        )
        pre_output = self.pre_readout(
            backbone_output.features,
            query if self.pre_query_conditioned else None,
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
        if self.physical_features:
            if batch.phys_loc is None or batch.phys_ctx is None:
                raise ValueError("This model requires training-normalized physical features.")
            tokens = type(tokens)(
                r_loc=torch.cat((tokens.r_loc,batch.phys_loc),dim=-1),
                r_ctx=torch.cat((tokens.r_ctx,batch.phys_ctx),dim=-1),
                r_gate=torch.cat((tokens.r_gate,batch.phys_loc,batch.phys_ctx),dim=-1),
            )
        return EncodedInterfaces(
            interface_mask=batch.interface_mask,
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
        )

    def conditional(self, encoded: EncodedInterfaces, prefix_states: Tensor) -> HeadStepOutput:
        validate_prefix_states(prefix_states, encoded.interface_mask)
        tokens = encoded.responsibility_tokens
        if options().bf16_autocast:
            with torch.autocast(encoded.query.device.type,enabled=False):
                return self.head(tokens.r_loc.float(),tokens.r_ctx.float(),tokens.r_gate.float(),encoded.query.float(),prefix_states,encoded.interface_mask)
        return self.head(tokens.r_loc, tokens.r_ctx, tokens.r_gate, encoded.query,prefix_states, encoded.interface_mask)

    def infer_encoded(self, encoded: EncodedInterfaces) -> ExactInferenceOutput:
        """Enumerate the selected Head without re-encoding the waveform."""
        tokens = encoded.responsibility_tokens
        if options().bf16_autocast:
            with torch.autocast(encoded.query.device.type,enabled=False):
                return enumerate_pcc(self.head,tokens.r_loc.float(),tokens.r_ctx.float(),tokens.r_gate.float(),encoded.query.float(),encoded.interface_mask)
        return enumerate_pcc(self.head, tokens.r_loc, tokens.r_ctx, tokens.r_gate,
                             encoded.query, encoded.interface_mask)

    def infer_exact(self, batch: InterfaceBatch) -> ExactInferenceOutput:
        """Run exact inference for the engineering range 1 <= K <= 6."""

        return self.infer_encoded(self.encode(batch))

    def _validate_structural_inputs(
        self,
        batch: InterfaceBatch,
    ) -> None:
        batch_size, kmax = batch.interface_mask.shape
        if kmax > self.max_interfaces:
            raise ValueError(
                f"Kmax={kmax} exceeds the configured boundary of {self.max_interfaces}."
            )
        if fast_structure():return
        valid_counts = batch.interface_mask.sum(dim=1)
        if batch_size == 0 or torch.any(valid_counts == 0):
            raise ValueError("Every sample must have at least one valid interface.")
        expected_mask = (
            torch.arange(kmax, device=batch.interface_mask.device).unsqueeze(0)
            < valid_counts.unsqueeze(1)
        )
        if not torch.equal(batch.interface_mask, expected_mask):
            raise ValueError("interface_mask must describe one contiguous ordered prefix.")

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
        if not fast_structure() and torch.any(valid & invalid_window):
            raise ValueError(
                "Every valid interface window must lie inside its trace and have start <= end."
            )
        return window_start, window_end, tau


A3Model = A3PCCModel

__all__ = ["A3Model", "A3PCCModel", "EncodedInterfaces", "HEAD_TYPES", "HeadType", "IntegratedPCCOutput"]
