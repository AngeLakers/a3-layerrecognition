"""A3 V1.4 fixed-interface inference and versioned training utilities.

Software validation does not establish research performance claims.
"""

__version__ = "1.4.0"

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
