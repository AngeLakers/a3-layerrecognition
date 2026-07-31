"""Strict label-free TXT loading for the A3 real-RF batch contract."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from a3_ultrasound.data.collate import (
    DEFAULT_KMAX,
    InterfaceSample,
    collate_interface_samples,
)
from a3_ultrasound.data.metadata import (
    load_yaml_mapping,
    validate_manifest,
)
from a3_ultrasound.schema import InterfaceBatch


FIVE_POINT_SPATIAL_STENCIL = "five_point_spatial_stencil_v1"

_LABEL_FREE_PLACEHOLDER = {
    "schema": "good_abnormal_v1",
    "source_type": "embedded_defect_design",
    "embedded_defect_manifest_ref": None,
    "good_region_ref": None,
    "uncertain_policy": "mask",
    "uncertain_boundary_ref": None,
}


class RealRFLoadError(ValueError):
    """Raised when a selected real-RF trace violates the registered contract."""


@dataclass(frozen=True)
class LabelFreeRealRFBatch:
    """A network batch plus non-network audit facts about its construction."""

    batch: InterfaceBatch
    selection_policy: str
    trace_count: int
    trace_lengths: tuple[int, ...]
    normalization_divisor: float
    raw_min: int
    raw_max: int


def load_label_free_network_manifest(
    manifest_path: str | Path,
    metadata_schema_path: str | Path,
) -> dict[str, Any]:
    """Validate the network-stage manifest without consuming registered labels."""

    manifest = load_yaml_mapping(manifest_path)
    schema = load_yaml_mapping(metadata_schema_path)
    label_free_view = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "labels"
    }
    label_free_view["labels"] = copy.deepcopy(_LABEL_FREE_PLACEHOLDER)
    validate_manifest(
        label_free_view,
        schema,
        stage="before_network_api_derivation",
    )
    label_free_view.pop("labels", None)
    return label_free_view


def select_five_point_spatial_stencil(
    manifest: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Select corners plus centre without enumerating the registered scan."""

    geometry = _require_mapping(manifest, "scan_geometry")
    if geometry.get("layout") != "regular_grid":
        raise RealRFLoadError(
            "five-point selection requires scan_geometry.layout=regular_grid."
        )
    grid_shape = geometry.get("grid_shape")
    if (
        not isinstance(grid_shape, Sequence)
        or isinstance(grid_shape, (str, bytes))
        or len(grid_shape) != 2
    ):
        raise RealRFLoadError("scan_geometry.grid_shape must contain two dimensions.")
    grid_x, grid_y = grid_shape
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 1
        for value in (grid_x, grid_y)
    ):
        raise RealRFLoadError(
            "five-point selection requires both grid dimensions to exceed one."
        )
    coordinates = (
        (0, 0),
        (0, grid_y - 1),
        (grid_x // 2, grid_y // 2),
        (grid_x - 1, 0),
        (grid_x - 1, grid_y - 1),
    )
    if len(set(coordinates)) != len(coordinates):
        raise RealRFLoadError("grid is too small for five unique stencil points.")
    return coordinates


def build_label_free_real_rf_batch(
    raw_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    coordinates: Sequence[tuple[int, int]] | None = None,
    kmax: int = DEFAULT_KMAX,
) -> LabelFreeRealRFBatch:
    """Read selected strict-integer TXT traces and build a label-free batch.

    The only amplitude transform is the manifest's one fixed batch divisor.
    No labels, scan coordinates, specimen IDs, paths, denoising, per-trace
    statistics, or peak normalization enter the returned network batch.
    """

    root = Path(raw_root)
    if not root.is_dir():
        raise RealRFLoadError("raw_root must be an existing directory.")

    raw_layout = _require_mapping(manifest, "raw_layout")
    if raw_layout.get("raw_dtype") != "signed_integer_text":
        raise RealRFLoadError(
            "raw_layout.raw_dtype must be signed_integer_text for the TXT loader."
        )
    if raw_layout.get("raw_unit") != "level":
        raise RealRFLoadError("raw_layout.raw_unit must be level.")
    file_pattern = raw_layout.get("file_pattern")
    if not isinstance(file_pattern, str) or not file_pattern:
        raise RealRFLoadError("raw_layout.file_pattern must be a non-empty string.")

    divisor = _fixed_batch_divisor(manifest)
    selected = (
        tuple(coordinates)
        if coordinates is not None
        else select_five_point_spatial_stencil(manifest)
    )
    _validate_coordinates(selected, manifest)

    traces: list[Tensor] = []
    trace_lengths: list[int] = []
    raw_min: int | None = None
    raw_max: int | None = None
    for coordinate in selected:
        path = _resolve_trace_path(root, file_pattern, coordinate)
        raw_values = _read_strict_integer_trace(path)
        trace_lengths.append(len(raw_values))
        current_min = min(raw_values)
        current_max = max(raw_values)
        raw_min = current_min if raw_min is None else min(raw_min, current_min)
        raw_max = current_max if raw_max is None else max(raw_max, current_max)
        trace = torch.tensor(raw_values, dtype=torch.float32) / divisor
        if not torch.isfinite(trace).all():
            raise RealRFLoadError(
                "fixed batch normalization produced a non-finite value."
            )
        traces.append(trace)

    batch = collate_interface_samples(
        [InterfaceSample(x_raw=trace) for trace in traces],
        manifest,
        kmax=kmax,
    )
    if batch.y is not None or batch.label_mask is not None:
        raise RuntimeError("internal error: label-free loader created supervision tensors.")
    assert raw_min is not None and raw_max is not None
    return LabelFreeRealRFBatch(
        batch=batch,
        selection_policy=(
            FIVE_POINT_SPATIAL_STENCIL
            if coordinates is None
            else "caller_supplied_audit_coordinates"
        ),
        trace_count=len(traces),
        trace_lengths=tuple(trace_lengths),
        normalization_divisor=divisor,
        raw_min=raw_min,
        raw_max=raw_max,
    )


def _fixed_batch_divisor(manifest: Mapping[str, Any]) -> float:
    preprocessing = _require_mapping(manifest, "preprocessing")
    normalization = preprocessing.get("amplitude_normalization")
    if not isinstance(normalization, Mapping):
        raise RealRFLoadError(
            "preprocessing.amplitude_normalization must be a mapping."
        )
    if normalization.get("method") != "fixed_batch_reference_division":
        raise RealRFLoadError(
            "M2-01 requires fixed_batch_reference_division normalization."
        )
    if normalization.get("raw_unit") != "level":
        raise RealRFLoadError("normalization raw_unit must match level.")
    value = normalization.get("divisor_level")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RealRFLoadError("normalization divisor_level must be numeric.")
    divisor = float(value)
    if not math.isfinite(divisor) or divisor <= 0:
        raise RealRFLoadError(
            "normalization divisor_level must be finite and positive."
        )
    return divisor


def _validate_coordinates(
    coordinates: Sequence[tuple[int, int]],
    manifest: Mapping[str, Any],
) -> None:
    if not coordinates:
        raise RealRFLoadError("at least one audit coordinate is required.")
    geometry = _require_mapping(manifest, "scan_geometry")
    grid_shape = geometry.get("grid_shape")
    if (
        not isinstance(grid_shape, Sequence)
        or isinstance(grid_shape, (str, bytes))
        or len(grid_shape) != 2
    ):
        raise RealRFLoadError("scan_geometry.grid_shape must contain two dimensions.")
    grid_x, grid_y = grid_shape
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (grid_x, grid_y)
    ):
        raise RealRFLoadError("scan_geometry.grid_shape values must be positive integers.")
    if len(set(coordinates)) != len(coordinates):
        raise RealRFLoadError("audit coordinates must be unique.")
    for coordinate in coordinates:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in coordinate
            )
        ):
            raise RealRFLoadError("each audit coordinate must be an integer (x, y) tuple.")
        if not (0 <= coordinate[0] < grid_x and 0 <= coordinate[1] < grid_y):
            raise RealRFLoadError("an audit coordinate lies outside the registered grid.")


def _resolve_trace_path(
    root: Path,
    file_pattern: str,
    coordinate: tuple[int, int],
) -> Path:
    try:
        relative_name = file_pattern.format(ix=coordinate[0], iy=coordinate[1])
    except (IndexError, KeyError, ValueError) as exc:
        raise RealRFLoadError(
            "raw_layout.file_pattern must format with {ix} and {iy}."
        ) from exc
    relative_path = Path(relative_name)
    if relative_path.is_absolute():
        raise RealRFLoadError("raw_layout.file_pattern must remain relative.")
    root_resolved = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root_resolved):
        raise RealRFLoadError("raw_layout.file_pattern escapes raw_root.")
    if not path.is_file():
        raise RealRFLoadError("a selected trace file is missing.")
    return path


def _read_strict_integer_trace(path: Path) -> list[int]:
    try:
        tokens = path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError) as exc:
        raise RealRFLoadError("a selected trace file is not readable UTF-8 text.") from exc
    if not tokens:
        raise RealRFLoadError("a selected trace file is empty.")
    try:
        return [int(token, 10) for token in tokens]
    except ValueError as exc:
        raise RealRFLoadError(
            "a selected trace contains a non-integer token."
        ) from exc


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise RealRFLoadError(f"{key} must be a mapping.")
    return child


__all__ = [
    "FIVE_POINT_SPATIAL_STENCIL",
    "LabelFreeRealRFBatch",
    "RealRFLoadError",
    "build_label_free_real_rf_batch",
    "load_label_free_network_manifest",
    "select_five_point_spatial_stencil",
]
