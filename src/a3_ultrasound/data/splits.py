"""Specimen-level split guards for synthetic contract fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


SPLIT_ROLES = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class SplitAudit:
    counts: dict[str, int]
    all_specimens: tuple[str, ...]


def _specimen_ids(role: str, values: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{role} must contain at least one specimen.")
    if any(
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        for value in resolved
    ):
        raise ValueError(
            f"{role} specimen IDs must be nonempty strings without outer whitespace."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{role} contains duplicate specimen IDs.")
    return resolved


def validate_specimen_split_disjointness(
    split: Mapping[str, Iterable[str]],
) -> SplitAudit:
    """Fail fast unless Train/Validation/Calibration/Test are pairwise disjoint."""

    if set(split) != set(SPLIT_ROLES):
        raise ValueError(
            "split must contain exactly train, validation, calibration, and test."
        )
    resolved = {role: _specimen_ids(role, split[role]) for role in SPLIT_ROLES}
    for index, left_role in enumerate(SPLIT_ROLES):
        left = set(resolved[left_role])
        for right_role in SPLIT_ROLES[index + 1 :]:
            overlap = left.intersection(resolved[right_role])
            if overlap:
                raise ValueError(
                    f"Specimen IDs must be disjoint across {left_role} and {right_role}."
                )
    all_specimens = tuple(
        specimen for role in SPLIT_ROLES for specimen in resolved[role]
    )
    return SplitAudit(
        counts={role: len(resolved[role]) for role in SPLIT_ROLES},
        all_specimens=all_specimens,
    )


def require_train_only_statistics_source(split_role: str) -> None:
    """Reject shortcut/statistic fitting from Validation, Calibration, or Test."""

    if split_role != "train":
        raise ValueError("Statistics and shortcut priors may be fitted from Train only.")


__all__ = [
    "SPLIT_ROLES",
    "SplitAudit",
    "require_train_only_statistics_source",
    "validate_specimen_split_disjointness",
]
