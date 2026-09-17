"""Data-contract utilities for the A3 research prototype."""

from .collate import (
    DEFAULT_KMAX,
    ENGINEERING_K_CAP,
    InterfaceSample,
    NetworkApiDerivationError,
    collate_interface_samples,
    derive_interface_tensors,
)
from .time_prior import (
    DerivedTimePrior,
    TimePriorDerivationError,
    derive_time_prior_samples,
)
from .real_rf import (
    FIVE_POINT_SPATIAL_STENCIL,
    LabelFreeRealRFBatch,
    RealRFLoadError,
    build_label_free_real_rf_batch,
    load_label_free_network_manifest,
    select_five_point_spatial_stencil,
)
from .splits import (
    SPLIT_ROLES,
    SplitAudit,
    require_train_only_statistics_source,
    validate_specimen_split_disjointness,
)

__all__ = [
    "DEFAULT_KMAX",
    "ENGINEERING_K_CAP",
    "DerivedTimePrior",
    "FIVE_POINT_SPATIAL_STENCIL",
    "InterfaceSample",
    "LabelFreeRealRFBatch",
    "NetworkApiDerivationError",
    "RealRFLoadError",
    "SPLIT_ROLES",
    "SplitAudit",
    "TimePriorDerivationError",
    "collate_interface_samples",
    "build_label_free_real_rf_batch",
    "derive_interface_tensors",
    "derive_time_prior_samples",
    "load_label_free_network_manifest",
    "require_train_only_statistics_source",
    "select_five_point_spatial_stencil",
    "validate_specimen_split_disjointness",
]
