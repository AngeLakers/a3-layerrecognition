"""A3 model-side structural components."""

from .backbone import BackboneOutput, LengthGroupedTCNBackbone
from .capacity import (
    CapacityMatch,
    HeadCapacity,
    matched_joint_head_parameter_count,
    pcc_head_parameter_count,
    responsibility_projector_parameter_count,
    select_matched_joint_hidden_width,
)
from .integrated import A3PCCModel, IntegratedPCCOutput
from .indexed_readout import IndexedReadout, ReadoutOutput, build_indexed_supports
from .losses import PCCLossOutput, pcc_training_loss
from .pcc import PCCConditionalStep, PCCStepOutput
from .query import InterfaceQueryEncoder
from .responsibility_tokens import ResponsibilityTokenProjector, ResponsibilityTokens

__all__ = [
    "A3PCCModel",
    "BackboneOutput",
    "CapacityMatch",
    "HeadCapacity",
    "IndexedReadout",
    "IntegratedPCCOutput",
    "InterfaceQueryEncoder",
    "LengthGroupedTCNBackbone",
    "PCCConditionalStep",
    "PCCLossOutput",
    "PCCStepOutput",
    "ReadoutOutput",
    "ResponsibilityTokenProjector",
    "ResponsibilityTokens",
    "build_indexed_supports",
    "matched_joint_head_parameter_count",
    "pcc_head_parameter_count",
    "pcc_training_loss",
    "responsibility_projector_parameter_count",
    "select_matched_joint_hidden_width",
]
