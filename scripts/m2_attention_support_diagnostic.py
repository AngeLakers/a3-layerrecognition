"""Run the M2-03 label-free attention/support diagnostic on real RF."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from a3_ultrasound.data import (  # noqa: E402
    build_label_free_real_rf_batch,
    load_label_free_network_manifest,
)

from m2_real_rf_forward_smoke import (  # noqa: E402
    SEED,
    _build_model,
    _git_state,
    _sha256,
    _to_device,
)


def _max_abs(values: torch.Tensor) -> float:
    return float(values.abs().max()) if values.numel() else 0.0


def _weighted_sample(
    attention: torch.Tensor,
    centers: torch.Tensor,
    support: torch.Tensor,
) -> float | None:
    supported_attention = attention[support]
    if supported_attention.numel() == 0:
        return None
    return float((supported_attention * centers[support]).sum())


def _peak_sample(
    attention: torch.Tensor,
    centers: torch.Tensor,
    support: torch.Tensor,
) -> float | None:
    supported_attention = attention[support]
    if supported_attention.numel() == 0:
        return None
    supported_centers = centers[support]
    return float(supported_centers[supported_attention.argmax()])


def _build_figure(
    *,
    local_attention: np.ndarray,
    pre_attention: np.ndarray,
    local_support: np.ndarray,
    pre_support: np.ndarray,
    interface_mask: np.ndarray,
    feature_centers: np.ndarray,
    output_path: Path,
) -> None:
    valid_interface_count = int(interface_mask.sum(axis=1).max())
    figure, axes = plt.subplots(
        2,
        valid_interface_count,
        figsize=(6.2 * valid_interface_count, 7.0),
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]

    for interface_index in range(valid_interface_count):
        valid_samples = np.flatnonzero(interface_mask[:, interface_index])
        local_axis = axes[0, interface_index]
        pre_axis = axes[1, interface_index]

        local_left = float("inf")
        local_right = float("-inf")
        pre_right = 0.0
        for sample_index in valid_samples:
            color = colors(sample_index % 10)
            label = f"trace {sample_index + 1}"
            local_mask = local_support[sample_index, interface_index]
            pre_mask = pre_support[sample_index, interface_index]
            local_axis.plot(
                feature_centers,
                local_attention[sample_index, interface_index],
                color=color,
                linewidth=1.1,
                alpha=0.82,
                label=label,
            )
            pre_axis.plot(
                feature_centers,
                pre_attention[sample_index, interface_index],
                color=color,
                linewidth=1.0,
                alpha=0.82,
                label=label,
            )
            if local_mask.any():
                local_centers = feature_centers[local_mask]
                local_left = min(local_left, float(local_centers.min()))
                local_right = max(local_right, float(local_centers.max()))
            if pre_mask.any():
                pre_right = max(pre_right, float(feature_centers[pre_mask].max()))

        if math.isfinite(local_left) and math.isfinite(local_right):
            local_axis.axvspan(
                local_left,
                local_right,
                color="#f6c85f",
                alpha=0.18,
                label="hard local support",
            )
            local_axis.set_xlim(max(0.0, local_left - 30.0), local_right + 30.0)
        if pre_right > 0:
            pre_axis.axvspan(
                0.0,
                pre_right,
                color="#6f4e7c",
                alpha=0.10,
                label="hard pre support",
            )
            pre_axis.set_xlim(0.0, pre_right + 30.0)

        local_axis.set_title(f"Interface {interface_index + 1}: Local attention")
        pre_axis.set_title(f"Interface {interface_index + 1}: Pre attention")
        local_axis.set_ylabel("attention weight")
        pre_axis.set_ylabel("attention weight")
        pre_axis.set_xlabel("feature-center sample")
        local_axis.grid(alpha=0.2)
        pre_axis.grid(alpha=0.2)
        local_axis.legend(fontsize=8, ncol=2)
        pre_axis.legend(fontsize=8, ncol=2)

    figure.suptitle(
        "M2-03 label-free attention/support diagnostic\n"
        "seeded random weights; no labels, loss, training, or performance claim",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    *,
    raw_root: Path,
    manifest_path: Path,
    metadata_schema_path: Path,
    output_json: Path,
    output_npz: Path,
    output_figure: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    manifest = load_label_free_network_manifest(manifest_path, metadata_schema_path)
    loaded = build_label_free_real_rf_batch(raw_root, manifest)
    if loaded.batch.y is not None or loaded.batch.label_mask is not None:
        raise RuntimeError("M2-03 must receive a label-free real-RF batch.")
    batch = _to_device(loaded.batch, device)
    model = _build_model(device).eval()
    prefix_states = torch.zeros_like(batch.interface_mask, dtype=torch.long)

    with torch.inference_mode():
        inference_mode_enabled = not torch.is_grad_enabled()
        output = model(batch, prefix_states)

    valid = batch.interface_mask
    invalid = ~valid
    centers = output.backbone.feature_centers
    if centers.ndim != 1:
        raise RuntimeError("M2-03 expects one shared feature-center axis.")
    expanded_centers = centers.view(1, 1, -1)
    feature_mask = output.backbone.feature_mask.unsqueeze(1)
    expected_local = (
        (expanded_centers >= output.window_start.unsqueeze(-1))
        & (expanded_centers <= output.window_end.unsqueeze(-1))
        & feature_mask
        & valid.unsqueeze(-1)
    )
    expected_pre = (
        (expanded_centers < output.window_start.unsqueeze(-1))
        & feature_mask
        & valid.unsqueeze(-1)
    )

    local_attention = output.local_readout.attention
    pre_attention = output.pre_readout.attention
    local_sum = local_attention.sum(dim=-1)
    pre_sum = pre_attention.sum(dim=-1)
    local_has_support = output.local_support.any(dim=-1)
    pre_has_support = output.pre_support.any(dim=-1)
    local_normalization_error = _max_abs(local_sum[valid & local_has_support] - 1.0)
    pre_normalization_error = _max_abs(pre_sum[valid & pre_has_support] - 1.0)
    local_outside_error = _max_abs(local_attention[~output.local_support])
    pre_outside_error = _max_abs(pre_attention[~output.pre_support])
    invalid_attention_error = max(
        _max_abs(local_attention[invalid]),
        _max_abs(pre_attention[invalid]),
    )

    diagnostics: list[dict[str, Any]] = []
    for batch_index in range(batch.x_raw.shape[0]):
        for interface_index in range(batch.interface_mask.shape[1]):
            if not bool(valid[batch_index, interface_index]):
                continue
            local_mask = output.local_support[batch_index, interface_index]
            pre_mask = output.pre_support[batch_index, interface_index]
            local_weights = local_attention[batch_index, interface_index]
            pre_weights = pre_attention[batch_index, interface_index]
            diagnostics.append(
                {
                    "trace_ordinal": batch_index + 1,
                    "interface_order": interface_index + 1,
                    "window_start_sample": float(
                        output.window_start[batch_index, interface_index]
                    ),
                    "window_end_sample": float(
                        output.window_end[batch_index, interface_index]
                    ),
                    "local_support_count": int(local_mask.sum()),
                    "pre_support_count": int(pre_mask.sum()),
                    "local_attention_sum": float(local_weights.sum()),
                    "pre_attention_sum": float(pre_weights.sum()),
                    "local_weighted_center_sample": _weighted_sample(
                        local_weights, centers, local_mask
                    ),
                    "pre_weighted_center_sample": _weighted_sample(
                        pre_weights, centers, pre_mask
                    ),
                    "local_peak_sample": _peak_sample(
                        local_weights, centers, local_mask
                    ),
                    "pre_peak_sample": _peak_sample(pre_weights, centers, pre_mask),
                    "empty_pre_support": not bool(pre_mask.any()),
                }
            )

    arrays = {
        "feature_centers": centers.detach().cpu().numpy().astype(np.float32),
        "local_attention": local_attention.detach().cpu().numpy().astype(np.float32),
        "pre_attention": pre_attention.detach().cpu().numpy().astype(np.float32),
        "local_support": output.local_support.detach().cpu().numpy(),
        "pre_support": output.pre_support.detach().cpu().numpy(),
        "interface_mask": valid.detach().cpu().numpy(),
        "window_start": output.window_start.detach().cpu().numpy().astype(np.float32),
        "window_end": output.window_end.detach().cpu().numpy().astype(np.float32),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    _build_figure(
        local_attention=arrays["local_attention"],
        pre_attention=arrays["pre_attention"],
        local_support=arrays["local_support"],
        pre_support=arrays["pre_support"],
        interface_mask=arrays["interface_mask"],
        feature_centers=arrays["feature_centers"],
        output_path=output_figure,
    )

    checks = {
        "label_free_batch": batch.y is None and batch.label_mask is None,
        "all_attention_finite": bool(
            torch.isfinite(local_attention).all()
            and torch.isfinite(pre_attention).all()
        ),
        "local_support_matches_hard_contract": bool(
            torch.equal(output.local_support, expected_local)
        ),
        "pre_support_matches_hard_contract": bool(
            torch.equal(output.pre_support, expected_pre)
        ),
        "local_support_nonempty_for_valid_interfaces": bool(
            local_has_support[valid].all()
        ),
        "local_attention_normalized": local_normalization_error < 1e-5,
        "pre_attention_normalized_where_supported": pre_normalization_error < 1e-5,
        "local_attention_zero_outside_support": local_outside_error == 0.0,
        "pre_attention_zero_outside_support": pre_outside_error == 0.0,
        "invalid_attention_zero": invalid_attention_error == 0.0,
        "no_gradient_tracking": inference_mode_enabled,
        "raw_rf_not_serialized_to_npz": "x_raw" not in arrays,
        "diagnostic_npz_written": output_npz.is_file() and output_npz.stat().st_size > 0,
        "diagnostic_figure_written": (
            output_figure.is_file() and output_figure.stat().st_size > 0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"M2-03 attention/support checks failed: {checks}")

    repository = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": "a3-m2-03-attention-support-diagnostic-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-03",
        "status": "PASS",
        "evidence_class": "sampled_real_rf_label_free_attention_support_diagnostic",
        "scope": {
            "reads_real_rf": True,
            "reads_registered_labels": False,
            "serializes_raw_rf": False,
            "performs_network_forward": True,
            "performs_loss": False,
            "performs_backward": False,
            "performs_optimizer_step": False,
            "performs_training": False,
            "performs_model_selection": False,
            "performs_performance_evaluation": False,
        },
        "resolved_config": {
            "seed": SEED,
            "device": str(device),
            "model_weight_state": "seeded_random_initialization_no_checkpoint",
            "batch_size": int(batch.x_raw.shape[0]),
            "trace_length": int(batch.x_raw.shape[1]),
            "valid_interface_counts": valid.sum(dim=1).tolist(),
        },
        "registered_contract": {
            "dataset_id": manifest["identity"]["dataset_id"],
            "acquisition_batch_id": manifest["identity"]["acquisition_batch_id"],
            "manifest_version": manifest.get("manifest_version"),
            "K": manifest["structure"]["K"],
            "labels_absent": True,
        },
        "input_fingerprint": loaded.input_fingerprint(),
        "global_audit": {
            "valid_interface_rows": int(valid.sum()),
            "empty_local_support_rows": int((valid & ~local_has_support).sum()),
            "empty_pre_support_rows": int((valid & ~pre_has_support).sum()),
            "local_attention_max_abs_normalization_error": local_normalization_error,
            "pre_attention_max_abs_normalization_error": pre_normalization_error,
            "local_attention_max_abs_outside_support": local_outside_error,
            "pre_attention_max_abs_outside_support": pre_outside_error,
            "invalid_attention_max_abs": invalid_attention_error,
        },
        "per_trace_interface_audit": diagnostics,
        "artifacts": {
            "arrays": output_npz.name,
            "figure": output_figure.name,
            "arrays_sha256": _sha256(output_npz),
            "figure_sha256": _sha256(output_figure),
            "arrays_contain_raw_rf": False,
            "arrays_contain_scan_coordinates": False,
        },
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "device_type": device.type,
            "compute_ledger_present": False,
        },
        "source_repository": _git_state(repository),
        "source_sha256": {
            "m2_real_rf_forward_smoke.py": _sha256(
                Path(__file__).with_name("m2_real_rf_forward_smoke.py")
            ),
            "m2_attention_support_diagnostic.py": _sha256(Path(__file__).resolve()),
            manifest_path.name: _sha256(manifest_path),
            metadata_schema_path.name: _sha256(metadata_schema_path),
        },
        "interpretation_boundary": (
            "PASS proves that Local and Pre attention from five selected label-free "
            "real RF traces stays inside the deterministic hard supports, normalizes "
            "on supported valid rows, and remains zero on invalid slots. The seeded "
            "random weights make attention locations diagnostic-only; this does not "
            "prove learned evidence allocation, performance, or mechanism."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-schema", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        metadata_schema_path=args.metadata_schema,
        output_json=args.output_json,
        output_npz=args.output_npz,
        output_figure=args.output_figure,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "device": result["resolved_config"]["device"],
                "json": args.output_json.as_posix(),
                "figure": args.output_figure.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
