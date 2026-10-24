"""Offline, phase-preserving A3 RF preparation; no label or time-coordinate changes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy import signal
import yaml


@dataclass(frozen=True)
class RFPreprocessing:
    sample_rate_hz: float
    normalization_divisor: float
    padlen: int

    def __post_init__(self):
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 40_000_000:
            raise ValueError("sample_rate_hz must place 20 MHz below Nyquist.")
        if not np.isfinite(self.normalization_divisor) or self.normalization_divisor <= 0:
            raise ValueError("normalization_divisor must be finite and positive.")
        if isinstance(self.padlen, bool) or not isinstance(self.padlen, int) or self.padlen < 0:
            raise ValueError("padlen must be an explicit nonnegative integer.")

    def record(self) -> dict:
        return {"algorithm": "a3-signed-rf-fixed-divisor-lp20-v1", **asdict(self),
                "cutoff_hz": 20_000_000, "butterworth_prototype_order": 3,
                "filter": "sosfiltfilt", "padtype": "odd", "output_dtype": "float32",
                "baseline_subtracted": False, "envelope_features": False,
                "spatial_smoothing": False, "time_shift_samples": 0}


def preprocess_rf(raw: np.ndarray, config: RFPreprocessing,
                  length_mask: np.ndarray | None = None) -> np.ndarray:
    """Filter each valid prefix before padding; input is unnormalized instrument values."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or not all(raw.shape) or raw.dtype.kind not in "iuf":
        raise ValueError("raw must be a nonempty real numeric [N,T] array.")
    mask = np.ones(raw.shape, dtype=bool) if length_mask is None else np.asarray(length_mask)
    if mask.dtype != np.bool_ or mask.shape != raw.shape:
        raise ValueError("length_mask must be bool [N,T].")
    lengths = mask.sum(axis=1)
    if not np.array_equal(mask, np.arange(raw.shape[1])[None, :] < lengths[:, None]):
        raise ValueError("length_mask must describe a contiguous valid prefix.")
    if np.any(lengths <= config.padlen) or np.any(lengths < 2):
        raise ValueError("Each valid trace must have at least two samples and exceed padlen.")
    if not np.isfinite(raw[mask]).all():
        raise ValueError("Valid RF samples must be finite.")
    sos = signal.butter(3, 20_000_000, fs=config.sample_rate_hz, output="sos")
    output = np.zeros(raw.shape, dtype=np.float32)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        values = raw[rows, :length].astype(np.float64) / config.normalization_divisor
        filtered = signal.sosfiltfilt(sos, values, axis=-1, padtype="odd", padlen=config.padlen)
        if not np.isfinite(filtered).all() or np.any(np.abs(filtered) > np.finfo(np.float32).max):
            raise ValueError("Preprocessing produced nonfinite or float32-overflow values.")
        output[rows, :length] = filtered
    return output


def preprocess_npz(source: Path, destination: Path, profile_path: Path,
                   *, signal_key: str = "x_raw") -> dict:
    """Prepare one acquisition batch; carry all other arrays through unchanged."""
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != "a3-rf-preprocessing-v1":
        raise ValueError("Expected a3-rf-preprocessing-v1 profile.")
    if set(profile) != {"schema_version", "preprocessing_id", "acquisition", "processing"}:
        raise ValueError("Unknown/missing profile fields; no silently ignored options.")
    if not isinstance(profile["preprocessing_id"], str) or not profile["preprocessing_id"].strip():
        raise ValueError("preprocessing_id must be a nonempty versioned identifier.")
    acquisition = profile["acquisition"]
    if set(acquisition) != {"probe_mhz", "gain_db", "batch_id"}:
        raise ValueError("Record probe_mhz, gain_db and batch_id for this file.")
    if acquisition["probe_mhz"] not in (5, 10) or not np.isfinite(acquisition["gain_db"]):
        raise ValueError("This adopted profile supports 5/10 MHz probes and finite gain_db.")
    if not isinstance(acquisition["batch_id"], str) or not acquisition["batch_id"].strip():
        raise ValueError("batch_id must be nonempty.")
    config = RFPreprocessing(**profile["processing"])
    with np.load(source, allow_pickle=False) as source_arrays:
        if "preprocessing_record" in source_arrays:
            raise ValueError("Already processed: refusing repeated normalization/filtering.")
        arrays = {key: source_arrays[key].copy() for key in source_arrays.files}
    arrays[signal_key] = preprocess_rf(arrays[signal_key], config, arrays.get("length_mask"))
    record = {"schema_version": profile["schema_version"],
              "preprocessing_id": profile["preprocessing_id"], "acquisition": acquisition,
              "processing": config.record(), "signal_key": signal_key,
              "trace_count": len(arrays[signal_key]),
              "numpy_version": np.__version__}
    import scipy
    record["scipy_version"] = scipy.__version__
    arrays["preprocessing_record"] = np.array(json.dumps(record, ensure_ascii=False, allow_nan=False))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return record
