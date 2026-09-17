"""Derive the frozen A3 network batch from validated acquisition metadata.

This module only converts in-memory traces and per-acquisition structural
metadata into the public ``InterfaceBatch`` tensor contract. It deliberately
does not resolve file paths, specimen identities, scan coordinates, or split
membership.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from a3_ultrasound.schema import (
    INTERFACE_META_DIM,
    INTERFACE_META_SCHEMA_VERSION,
    InterfaceBatch,
)
from a3_ultrasound.data.time_prior import (
    TimePriorDerivationError,
    derive_time_prior_samples,
)


ENGINEERING_K_CAP = 6
DEFAULT_KMAX = ENGINEERING_K_CAP


class NetworkApiDerivationError(ValueError):
    """Raised when metadata cannot be safely converted to network tensors."""


@dataclass(frozen=True)
class InterfaceSample:
    """One in-memory RF trace with optional per-interface supervision.

    Audit-only fields are intentionally absent. ``y`` and ``label_mask`` use
    the valid-interface length ``K_i`` rather than padded ``Kmax``.
    """

    x_raw: Tensor | Sequence[float]
    y: Tensor | Sequence[int] | None = None
    label_mask: Tensor | Sequence[bool] | None = None


def derive_interface_tensors(
    manifest: Mapping[str, Any],
    *,
    trace_length: int,
    kmax: int = DEFAULT_KMAX,
) -> tuple[Tensor, Tensor]:
    """Derive packed v3 metadata and its valid-interface mask for one trace.

    The frozen packed order is::

        [k/K_i, tau/T_i, z/zmax, Delta-/T_i, Delta+/T_i]

    ``T_i`` is the trace's valid sample count. ``zmax`` is the acquisition
    batch's registered specimen setting ``structure.total_thickness_mm``.
    """

    if not isinstance(trace_length, int) or isinstance(trace_length, bool) or trace_length <= 0:
        raise NetworkApiDerivationError("trace_length must be a positive integer.")
    if not isinstance(kmax, int) or isinstance(kmax, bool) or kmax <= 0:
        raise NetworkApiDerivationError("kmax must be a positive integer.")

    structure = _require_mapping(manifest, "structure")
    interface_count = _require_int(structure, "K", path="structure.K")
    if interface_count not in {2, 3}:
        raise NetworkApiDerivationError("structure.K must be 2 or 3 for A3 V1.")
    if interface_count > kmax:
        raise NetworkApiDerivationError(
            f"kmax={kmax} cannot hold structure.K={interface_count} valid interfaces."
        )
    if kmax > ENGINEERING_K_CAP:
        raise NetworkApiDerivationError(
            f"kmax>{ENGINEERING_K_CAP} exceeds the A3 V1.2 engineering boundary."
        )

    interfaces = structure.get("interfaces")
    if not isinstance(interfaces, list) or len(interfaces) != interface_count:
        raise NetworkApiDerivationError(
            "structure.interfaces must be a list whose length equals structure.K."
        )

    zmax = _require_finite_number(
        structure,
        "total_thickness_mm",
        path="structure.total_thickness_mm",
        positive=True,
    )
    try:
        time_prior = derive_time_prior_samples(
            manifest,
            trace_length=trace_length,
        )
    except TimePriorDerivationError as exc:
        raise NetworkApiDerivationError(str(exc)) from exc

    metadata = torch.zeros((kmax, INTERFACE_META_DIM), dtype=torch.float32)
    interface_mask = torch.zeros((kmax,), dtype=torch.bool)
    for index, interface in enumerate(interfaces):
        if not isinstance(interface, Mapping):
            raise NetworkApiDerivationError(
                f"structure.interfaces[{index}] must be a mapping."
            )
        expected_order = index + 1
        if interface.get("interface_order") != expected_order:
            raise NetworkApiDerivationError(
                "structure.interfaces must be ordered shallow-to-deep with "
                f"interface_order={expected_order} at index {index}."
            )

        tau_samples = time_prior.nominal_tof_samples[index]
        depth_mm = _require_finite_number(
            interface,
            "nominal_depth_mm",
            path=f"structure.interfaces[{index}].nominal_depth_mm",
            positive=True,
        )
        left_samples = time_prior.window_left_samples
        right_samples = time_prior.window_right_samples

        if depth_mm >= zmax:
            raise NetworkApiDerivationError(
                f"structure.interfaces[{index}].nominal_depth_mm must be < "
                "structure.total_thickness_mm for an internal interface."
            )
        if tau_samples < 0 or tau_samples >= trace_length:
            raise NetworkApiDerivationError(
                f"structure.interfaces[{index}] nominal TOF must lie within the valid trace."
            )
        if tau_samples - left_samples < 0 or tau_samples + right_samples >= trace_length:
            raise NetworkApiDerivationError(
                f"structure.interfaces[{index}] TOF window must lie within the valid trace."
            )

        metadata[index] = torch.tensor(
            (
                expected_order / interface_count,
                tau_samples / trace_length,
                depth_mm / zmax,
                left_samples / trace_length,
                right_samples / trace_length,
            ),
            dtype=torch.float32,
        )
        interface_mask[index] = True

    return metadata, interface_mask


def collate_interface_samples(
    samples: Sequence[InterfaceSample],
    manifest: Mapping[str, Any],
    *,
    kmax: int = DEFAULT_KMAX,
) -> InterfaceBatch:
    """Pad traces and derive a network-only ``InterfaceBatch``.

    The caller must first validate ``manifest`` at
    ``before_network_api_derivation`` with the project metadata validator.
    This function repeats all safety-critical derivation checks so it cannot
    silently manufacture missing TOF, depth, window, or normalization values.
    """

    if not samples:
        raise NetworkApiDerivationError("samples must contain at least one trace.")

    traces = [_coerce_trace(sample.x_raw, index) for index, sample in enumerate(samples)]
    _validate_registered_trace_lengths(manifest, traces)

    batch_size = len(traces)
    padded_length = max(trace.numel() for trace in traces)
    x_raw = torch.zeros((batch_size, padded_length), dtype=torch.float32)
    length_mask = torch.zeros((batch_size, padded_length), dtype=torch.bool)
    interface_meta = torch.zeros(
        (batch_size, kmax, INTERFACE_META_DIM),
        dtype=torch.float32,
    )
    interface_mask = torch.zeros((batch_size, kmax), dtype=torch.bool)

    for index, trace in enumerate(traces):
        trace_length = trace.numel()
        x_raw[index, :trace_length] = trace
        length_mask[index, :trace_length] = True
        sample_meta, sample_mask = derive_interface_tensors(
            manifest,
            trace_length=trace_length,
            kmax=kmax,
        )
        interface_meta[index] = sample_meta
        interface_mask[index] = sample_mask

    y, label_mask = _collate_labels(samples, interface_mask)
    return InterfaceBatch(
        x_raw=x_raw,
        length_mask=length_mask,
        interface_meta=interface_meta,
        interface_mask=interface_mask,
        label_mask=label_mask,
        y=y,
    )


def _coerce_trace(value: Tensor | Sequence[float], index: int) -> Tensor:
    try:
        trace = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise NetworkApiDerivationError(f"samples[{index}].x_raw is not numeric.") from exc
    if trace.ndim != 1 or trace.numel() == 0:
        raise NetworkApiDerivationError(
            f"samples[{index}].x_raw must be a non-empty one-dimensional trace."
        )
    if not torch.isfinite(trace).all():
        raise NetworkApiDerivationError(
            f"samples[{index}].x_raw must contain only finite values."
        )
    return trace


def _validate_registered_trace_lengths(
    manifest: Mapping[str, Any],
    traces: Sequence[Tensor],
) -> None:
    raw_layout = manifest.get("raw_layout")
    if not isinstance(raw_layout, Mapping):
        raise NetworkApiDerivationError("raw_layout must be a mapping.")
    mode = raw_layout.get("trace_length_mode")
    if mode == "fixed_within_batch":
        expected = raw_layout.get("trace_samples")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise NetworkApiDerivationError(
                "raw_layout.trace_samples must be a positive integer for fixed traces."
            )
        actual = [trace.numel() for trace in traces]
        if any(length != expected for length in actual):
            raise NetworkApiDerivationError(
                f"all traces must match raw_layout.trace_samples={expected}; got {actual}."
            )
    elif mode != "variable_within_batch":
        raise NetworkApiDerivationError(
            "raw_layout.trace_length_mode must be fixed_within_batch or variable_within_batch."
        )


def _collate_labels(
    samples: Sequence[InterfaceSample],
    interface_mask: Tensor,
) -> tuple[Tensor | None, Tensor | None]:
    if all(sample.y is None and sample.label_mask is None for sample in samples):
        return None, None
    if any(sample.y is None and sample.label_mask is not None for sample in samples):
        raise NetworkApiDerivationError("label_mask cannot be provided without y.")

    batch_size, kmax = interface_mask.shape
    y = torch.zeros((batch_size, kmax), dtype=torch.int64)
    label_mask = torch.zeros((batch_size, kmax), dtype=torch.bool)
    valid_count = int(interface_mask[0].sum().item())

    for index, sample in enumerate(samples):
        if sample.y is None:
            continue
        try:
            raw_y = torch.as_tensor(sample.y)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise NetworkApiDerivationError(
                f"samples[{index}].y must contain integer labels."
            ) from exc
        if raw_y.ndim != 1 or raw_y.numel() != valid_count:
            raise NetworkApiDerivationError(
                f"samples[{index}].y must contain exactly K={valid_count} labels."
            )
        if sample.label_mask is None:
            sample_label_mask = torch.ones((valid_count,), dtype=torch.bool)
        else:
            try:
                raw_label_mask = torch.as_tensor(sample.label_mask)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise NetworkApiDerivationError(
                    f"samples[{index}].label_mask must contain boolean values."
                ) from exc
            if (
                raw_label_mask.ndim != 1
                or raw_label_mask.numel() != valid_count
                or raw_label_mask.dtype != torch.bool
            ):
                raise NetworkApiDerivationError(
                    f"samples[{index}].label_mask must contain exactly K={valid_count} "
                    "boolean values."
                )
            sample_label_mask = raw_label_mask
        if raw_y.dtype == torch.bool or torch.is_floating_point(raw_y) or torch.is_complex(raw_y):
            raise NetworkApiDerivationError(
                f"samples[{index}].y must use an integer dtype before collation."
            )
        sample_y = raw_y.to(dtype=torch.int64)
        supervised_values = sample_y[sample_label_mask]
        if supervised_values.numel() and not torch.all(
            (supervised_values == 0) | (supervised_values == 1)
        ):
            raise NetworkApiDerivationError(
                f"samples[{index}].y must use good=0/abnormal=1 where label_mask is true."
            )
        y[index, :valid_count] = sample_y
        label_mask[index, :valid_count] = sample_label_mask

    return y, label_mask


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise NetworkApiDerivationError(f"{key} must be a mapping.")
    return child


def _require_int(value: Mapping[str, Any], key: str, *, path: str) -> int:
    child = value.get(key)
    if not isinstance(child, int) or isinstance(child, bool):
        raise NetworkApiDerivationError(f"{path} must be an integer.")
    return child


def _require_finite_number(
    value: Mapping[str, Any],
    key: str,
    *,
    path: str,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    number = _optional_finite_number(value.get(key), path, positive=positive)
    if number is None:
        raise NetworkApiDerivationError(f"{path} must be a finite number.")
    if minimum is not None and number < minimum:
        raise NetworkApiDerivationError(f"{path} must be >= {minimum}.")
    return number


def _optional_finite_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NetworkApiDerivationError(f"{path} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise NetworkApiDerivationError(f"{path} must be finite.")
    if positive and number <= 0:
        raise NetworkApiDerivationError(f"{path} must be > 0.")
    return number


__all__ = [
    "DEFAULT_KMAX",
    "ENGINEERING_K_CAP",
    "INTERFACE_META_SCHEMA_VERSION",
    "InterfaceSample",
    "NetworkApiDerivationError",
    "collate_interface_samples",
    "derive_interface_tensors",
]
