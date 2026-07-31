"""Specimen-level binary macro-F1 with the frozen present-class convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SpecimenMacroF1:
    """One specimen's macro-F1 plus the classes actually present in truth."""

    macro_f1: float
    per_class_f1: dict[int, float]
    present_classes: tuple[int, ...]
    single_class_specimen: bool
    zero_division: int = 0


def _binary_vector(name: str, values: Iterable[int]) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{name} must be nonempty.")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in (0, 1)
        for value in resolved
    ):
        raise ValueError(f"{name} must contain only integer binary labels.")
    return resolved


def specimen_macro_f1_present_classes(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> SpecimenMacroF1:
    """Compute macro-F1 only over truth-present classes with zero_division=0."""

    truth = _binary_vector("y_true", y_true)
    prediction = _binary_vector("y_pred", y_pred)
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have the same length.")

    present_classes = tuple(sorted(set(truth)))
    per_class: dict[int, float] = {}
    for label in present_classes:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, prediction, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label] = (
            0.0 if denominator == 0 else (2.0 * true_positive) / denominator
        )
    macro_f1 = sum(per_class.values()) / len(present_classes)
    return SpecimenMacroF1(
        macro_f1=macro_f1,
        per_class_f1=per_class,
        present_classes=present_classes,
        single_class_specimen=len(present_classes) == 1,
    )
