"""Derive batch-referenced nominal TOFs and one shared acquisition window.

The source of truth is deliberately hierarchical:

* one acquisition/window profile supplies ``Delta-`` and ``Delta+``;
* one acquisition batch supplies the front-surface reference ``b_ref``;
* specimen geometry and sound speed supply propagation time;
* interfaces contain order and nominal depth only.

Per-interface TOF/window overrides are rejected so that the five-field network
metadata cannot silently drift away from the physical acquisition contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_FORBIDDEN_INTERFACE_TIME_FIELDS = frozenset(
    {
        "nominal_tof_samples",
        "nominal_tof_s",
        "window_left_samples",
        "window_right_samples",
    }
)
_REPRESENTATION_TOLERANCE_SAMPLES = 1e-6


class TimePriorDerivationError(ValueError):
    """Raised when physical time-prior facts are missing or inconsistent."""


@dataclass(frozen=True)
class DerivedTimePrior:
    """Derived raw-sample coordinates for all valid internal interfaces."""

    nominal_tof_samples: tuple[float, ...]
    window_left_samples: float
    window_right_samples: float
    window_start_samples: tuple[float, ...]
    window_end_samples: tuple[float, ...]
    batch_reference_samples: float
    backwall_tof_samples: float


def derive_time_prior_samples(
    manifest: Mapping[str, Any],
    *,
    trace_length: int,
) -> DerivedTimePrior:
    """Derive nominal TOFs and shared supports from acquisition facts.

    Homogeneous propagation uses

    ``tau_k = b_ref + 2 * fs * z_k / c``.

    When every layer declares ``sound_speed_mps``, the cumulative layered form

    ``tau_k = b_ref + 2 * fs * sum_l(d_l / c_l)``

    is used instead.
    """

    if not isinstance(trace_length, int) or isinstance(trace_length, bool) or trace_length <= 0:
        raise TimePriorDerivationError("trace_length must be a positive integer.")

    acquisition = _require_mapping(manifest, "acquisition")
    sampling_rate_hz = _require_number(
        acquisition,
        "sampling_rate_hz",
        path="acquisition.sampling_rate_hz",
        positive=True,
    )
    window_profile = _require_mapping(acquisition, "window_profile")
    left_samples = _resolve_sample_quantity(
        window_profile,
        sample_key="window_left_samples",
        second_key="window_left_s",
        sampling_rate_hz=sampling_rate_hz,
        path="acquisition.window_profile",
        positive=True,
    )
    right_samples = _resolve_sample_quantity(
        window_profile,
        sample_key="window_right_samples",
        second_key="window_right_s",
        sampling_rate_hz=sampling_rate_hz,
        path="acquisition.window_profile",
        positive=True,
    )

    time_reference = _require_mapping(manifest, "time_reference")
    batch_reference = _resolve_sample_quantity(
        time_reference,
        sample_key="batch_reference_samples",
        second_key="batch_reference_s",
        sampling_rate_hz=sampling_rate_hz,
        path="time_reference",
        positive=False,
    )

    structure = _require_mapping(manifest, "structure")
    interfaces = structure.get("interfaces")
    if not isinstance(interfaces, list) or not interfaces:
        raise TimePriorDerivationError("structure.interfaces must be a non-empty list.")
    for index, interface in enumerate(interfaces):
        if not isinstance(interface, Mapping):
            raise TimePriorDerivationError(
                f"structure.interfaces[{index}] must be a mapping."
            )
        forbidden = _FORBIDDEN_INTERFACE_TIME_FIELDS.intersection(interface)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise TimePriorDerivationError(
                f"structure.interfaces[{index}] contains forbidden per-interface "
                f"TOF/window override field(s): {fields}."
            )

    depths_mm = tuple(
        _require_number(
            interface,
            "nominal_depth_mm",
            path=f"structure.interfaces[{index}].nominal_depth_mm",
            positive=True,
        )
        for index, interface in enumerate(interfaces)
    )
    if any(later <= earlier for earlier, later in zip(depths_mm, depths_mm[1:])):
        raise TimePriorDerivationError(
            "structure.interfaces nominal depths must be strictly increasing."
        )

    total_thickness_mm = _require_number(
        structure,
        "total_thickness_mm",
        path="structure.total_thickness_mm",
        positive=True,
    )
    if depths_mm[-1] >= total_thickness_mm:
        raise TimePriorDerivationError(
            "Every target interface depth must be strictly below total thickness."
        )

    propagation_seconds, backwall_seconds = _propagation_times_seconds(
        structure,
        depths_mm=depths_mm,
        total_thickness_mm=total_thickness_mm,
    )
    nominal_tofs = tuple(
        batch_reference + sampling_rate_hz * seconds
        for seconds in propagation_seconds
    )
    backwall_tof = batch_reference + sampling_rate_hz * backwall_seconds
    starts = tuple(tau - left_samples for tau in nominal_tofs)
    ends = tuple(tau + right_samples for tau in nominal_tofs)

    if any(later <= earlier for earlier, later in zip(nominal_tofs, nominal_tofs[1:])):
        raise TimePriorDerivationError("Derived nominal TOFs must be strictly increasing.")
    if starts[0] <= batch_reference:
        raise TimePriorDerivationError(
            "The first interface window overlaps the batch front-surface reference."
        )
    if any(start <= previous_end for previous_end, start in zip(ends, starts[1:])):
        raise TimePriorDerivationError(
            "Shared interface windows overlap; one acquisition-profile window is not feasible."
        )
    if ends[-1] >= backwall_tof:
        raise TimePriorDerivationError(
            "The deepest interface window overlaps the nominal backwall reference."
        )
    if starts[0] < 0 or ends[-1] >= trace_length:
        raise TimePriorDerivationError(
            "Every derived interface window must lie inside [0, trace_length - 1]."
        )

    return DerivedTimePrior(
        nominal_tof_samples=nominal_tofs,
        window_left_samples=left_samples,
        window_right_samples=right_samples,
        window_start_samples=starts,
        window_end_samples=ends,
        batch_reference_samples=batch_reference,
        backwall_tof_samples=backwall_tof,
    )


def _propagation_times_seconds(
    structure: Mapping[str, Any],
    *,
    depths_mm: Sequence[float],
    total_thickness_mm: float,
) -> tuple[tuple[float, ...], float]:
    layers = structure.get("layers")
    if isinstance(layers, list) and layers:
        layer_speeds = [
            _optional_number(
                layer.get("sound_speed_mps") if isinstance(layer, Mapping) else None,
                f"structure.layers[{index}].sound_speed_mps",
                positive=True,
            )
            for index, layer in enumerate(layers)
        ]
        if any(speed is not None for speed in layer_speeds):
            if any(speed is None for speed in layer_speeds):
                raise TimePriorDerivationError(
                    "Layered propagation requires sound_speed_mps on every layer."
                )
            thicknesses = [
                _require_number(
                    layer,
                    "thickness_mm",
                    path=f"structure.layers[{index}].thickness_mm",
                    positive=True,
                )
                for index, layer in enumerate(layers)
                if isinstance(layer, Mapping)
            ]
            if len(thicknesses) != len(layers):
                raise TimePriorDerivationError("Every structure layer must be a mapping.")
            cumulative_depths: list[float] = []
            cumulative_times: list[float] = []
            depth_mm = 0.0
            one_way_seconds = 0.0
            for thickness_mm, speed_mps in zip(thicknesses, layer_speeds):
                assert speed_mps is not None
                depth_mm += thickness_mm
                one_way_seconds += thickness_mm * 1e-3 / speed_mps
                cumulative_depths.append(depth_mm)
                cumulative_times.append(one_way_seconds)
            if not math.isclose(
                cumulative_depths[-1],
                total_thickness_mm,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise TimePriorDerivationError(
                    "Layer thickness sum must equal structure.total_thickness_mm."
                )
            interface_times: list[float] = []
            for index, depth in enumerate(depths_mm):
                if index >= len(cumulative_depths) - 1 or not math.isclose(
                    depth,
                    cumulative_depths[index],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise TimePriorDerivationError(
                        "Layered interface depths must equal cumulative layer boundaries."
                    )
                interface_times.append(2.0 * cumulative_times[index])
            return tuple(interface_times), 2.0 * cumulative_times[-1]

    sound_speed_mps = _require_number(
        structure,
        "sound_speed_mps",
        path="structure.sound_speed_mps",
        positive=True,
    )
    interface_times = tuple(
        2.0 * depth_mm * 1e-3 / sound_speed_mps for depth_mm in depths_mm
    )
    backwall_time = 2.0 * total_thickness_mm * 1e-3 / sound_speed_mps
    return interface_times, backwall_time


def _resolve_sample_quantity(
    value: Mapping[str, Any],
    *,
    sample_key: str,
    second_key: str,
    sampling_rate_hz: float,
    path: str,
    positive: bool,
) -> float:
    samples = _optional_number(
        value.get(sample_key),
        f"{path}.{sample_key}",
        positive=positive,
        nonnegative=not positive,
    )
    seconds = _optional_number(
        value.get(second_key),
        f"{path}.{second_key}",
        positive=positive,
        nonnegative=not positive,
    )
    if samples is None and seconds is None:
        raise TimePriorDerivationError(
            f"{path} requires {sample_key} or {second_key}."
        )
    converted = None if seconds is None else seconds * sampling_rate_hz
    if samples is not None and converted is not None and not math.isclose(
        samples,
        converted,
        rel_tol=0.0,
        abs_tol=_REPRESENTATION_TOLERANCE_SAMPLES,
    ):
        raise TimePriorDerivationError(
            f"{path}.{sample_key} and {path}.{second_key} disagree by more "
            f"than {_REPRESENTATION_TOLERANCE_SAMPLES} sample."
        )
    return samples if samples is not None else float(converted)


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise TimePriorDerivationError(f"{key} must be a mapping.")
    return child


def _require_number(
    value: Mapping[str, Any],
    key: str,
    *,
    path: str,
    positive: bool = False,
) -> float:
    number = _optional_number(value.get(key), path, positive=positive)
    if number is None:
        raise TimePriorDerivationError(f"{path} must be a finite number.")
    return number


def _optional_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TimePriorDerivationError(f"{path} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise TimePriorDerivationError(f"{path} must be finite.")
    if positive and number <= 0:
        raise TimePriorDerivationError(f"{path} must be > 0.")
    if nonnegative and number < 0:
        raise TimePriorDerivationError(f"{path} must be >= 0.")
    return number


__all__ = [
    "DerivedTimePrior",
    "TimePriorDerivationError",
    "derive_time_prior_samples",
]
