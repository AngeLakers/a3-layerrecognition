"""A3 V1.1 fixed-interface state-inference prototype.

This package only contains the structure/API prototype and synthetic-test
components. It intentionally excludes real-data training and experiment claims.
"""

from .schema import (
    INTERFACE_META_DIM,
    INTERFACE_META_FIELDS,
    INTERFACE_META_SCHEMA_VERSION,
    PRIOR_META_DIM,
    PRIOR_META_FIELDS,
    PRIOR_META_INDICES,
    QUERY_META_DIM,
    QUERY_META_FIELDS,
    QUERY_META_INDICES,
    InterfaceBatch,
)

__all__ = [
    "INTERFACE_META_DIM",
    "INTERFACE_META_FIELDS",
    "INTERFACE_META_SCHEMA_VERSION",
    "PRIOR_META_DIM",
    "PRIOR_META_FIELDS",
    "PRIOR_META_INDICES",
    "QUERY_META_DIM",
    "QUERY_META_FIELDS",
    "QUERY_META_INDICES",
    "InterfaceBatch",
]
