"""Data-contract utilities for the A3 research prototype."""

from .collate import (
    DEFAULT_KMAX,
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
from .splits import (
    SPLIT_ROLES,
    SplitAudit,
    require_train_only_statistics_source,
    validate_specimen_split_disjointness,
)

__all__ = [
    "DEFAULT_KMAX",
    "DerivedTimePrior",
    "InterfaceSample",
    "NetworkApiDerivationError",
    "SPLIT_ROLES",
    "SplitAudit",
    "TimePriorDerivationError",
    "collate_interface_samples",
    "derive_interface_tensors",
    "derive_time_prior_samples",
    "require_train_only_statistics_source",
    "validate_specimen_split_disjointness",
]
