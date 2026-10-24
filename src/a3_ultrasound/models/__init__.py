"""A3 model-side structural components."""

from .backbone import (
    BackboneOutput,
    CommonTemporalBackboneAdapter,
    LengthGroupedTCNBackbone,
    TemporalBackboneOutput,
    TsaiFeatureStage,
    TsaiTimePreservingBackbone,
)
from .baselines import IndependentConditionalStep, LocalPreIndependentStep, VanillaClassifierChainStep
from .capacity import (
    BaselineHeadCapacity,
    CapacityMatch,
    HeadCapacity,
    head_baseline_parameter_counts,
    matched_joint_head_parameter_count,
    pcc_head_parameter_count,
    responsibility_projector_parameter_count,
    select_matched_joint_hidden_width,
)
from .integrated import A3Model, A3PCCModel, EncodedInterfaces, IntegratedPCCOutput
from .indexed_readout import (
    IndexedReadout,
    LocalTokenDiagnostics,
    ReadoutOutput,
    build_indexed_supports,
    summarize_local_tokens,
)
from .losses import PCCLossOutput, head_training_loss, pcc_training_loss, validate_complete_training_batch
from .pcc import (
    LocalBaseScorer,
    HeadStepOutput,
    PCCConditionalStep,
    PCCStepOutput,
    SharedPrefixGRUEncoder,
)
from .query import InterfaceQueryEncoder
from .responsibility_tokens import ResponsibilityTokenProjector, ResponsibilityTokens

__all__ = [
    "A3Model",
    "EncodedInterfaces",
    "HeadStepOutput",
    "LocalPreIndependentStep",
    "head_training_loss",
    "validate_complete_training_batch",
    "A3PCCModel",
    "BackboneOutput",
    "BaselineHeadCapacity",
    "CommonTemporalBackboneAdapter",
    "CapacityMatch",
    "HeadCapacity",
    "IndexedReadout",
    "IndependentConditionalStep",
    "IntegratedPCCOutput",
    "InterfaceQueryEncoder",
    "LengthGroupedTCNBackbone",
    "LocalBaseScorer",
    "LocalTokenDiagnostics",
    "PCCConditionalStep",
    "PCCLossOutput",
    "PCCStepOutput",
    "ReadoutOutput",
    "ResponsibilityTokenProjector",
    "ResponsibilityTokens",
    "SharedPrefixGRUEncoder",
    "TemporalBackboneOutput",
    "TsaiFeatureStage",
    "TsaiTimePreservingBackbone",
    "VanillaClassifierChainStep",
    "build_indexed_supports",
    "head_baseline_parameter_counts",
    "matched_joint_head_parameter_count",
    "pcc_head_parameter_count",
    "pcc_training_loss",
    "responsibility_projector_parameter_count",
    "select_matched_joint_hidden_width",
    "summarize_local_tokens",
]
