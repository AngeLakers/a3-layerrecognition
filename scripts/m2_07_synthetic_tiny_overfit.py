"""Run the M2-07 fixed synthetic tiny-overfit learning-path diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from a3_ultrasound.models import pcc_training_loss
from a3_ultrasound.runtime_contract import build_runtime_acceptance
from a3_ultrasound.schema import InterfaceBatch
from m2_04_gpu_profile import _nvidia_smi_snapshot
from m2_real_rf_forward_smoke import _build_model, _git_state, _sha256


SEED = 3117
TRACE_LENGTH = 2300
UNIQUE_SAMPLE_COUNT = 16
MAX_STEPS = 240
EVALUATION_INTERVAL = 10
LEARNING_RATE = 3e-3
LAMBDA_LOCAL = 0.3
LOSS_RATIO_THRESHOLD = 0.25
VALID_ACCURACY_THRESHOLD = 1.0
MIB = 1024**2


def _make_unique_samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create 16 deterministic traces with mixed K and both binary classes."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    positions = torch.arange(TRACE_LENGTH, dtype=torch.float32)
    nominal_tofs = (245.026, 455.553, 666.079)
    label_patterns: list[tuple[int, tuple[int, ...]]] = []
    label_patterns.extend((3, tuple(bits)) for bits in product((0, 1), repeat=3))
    label_patterns.extend((2, tuple(bits)) for _ in range(2) for bits in product((0, 1), repeat=2))
    if len(label_patterns) != UNIQUE_SAMPLE_COUNT:
        raise RuntimeError("M2-07 synthetic pattern construction changed unexpectedly.")

    x_raw = torch.zeros((UNIQUE_SAMPLE_COUNT, TRACE_LENGTH), dtype=torch.float32)
    interface_meta = torch.zeros((UNIQUE_SAMPLE_COUNT, 3, 5), dtype=torch.float32)
    interface_mask = torch.zeros((UNIQUE_SAMPLE_COUNT, 3), dtype=torch.bool)
    labels = torch.zeros((UNIQUE_SAMPLE_COUNT, 3), dtype=torch.int64)

    for sample_index, (interface_count, pattern) in enumerate(label_patterns):
        trace = 0.025 * torch.randn(TRACE_LENGTH, generator=generator)
        trace += 0.02 * torch.sin(
            positions * (0.011 + 0.0003 * sample_index) + 0.17 * sample_index
        )
        interface_mask[sample_index, :interface_count] = True
        labels[sample_index, :interface_count] = torch.tensor(pattern, dtype=torch.int64)
        for interface_index, label in enumerate(pattern):
            center = nominal_tofs[interface_index] + float((sample_index % 3) - 1) * 2.0
            sigma = 11.0 + float((sample_index + interface_index) % 3)
            signed_amplitude = 2.5 if label == 1 else -2.5
            local_coordinate = positions - center
            envelope = torch.exp(-0.5 * (local_coordinate / sigma) ** 2)
            pulse = signed_amplitude * envelope
            ring = 0.35 * signed_amplitude * envelope * torch.cos(0.32 * local_coordinate)
            trace += pulse + ring
            interface_meta[sample_index, interface_index] = torch.tensor(
                [
                    (interface_index + 1) / interface_count,
                    nominal_tofs[interface_index] / TRACE_LENGTH,
                    (interface_index + 1) / (interface_count + 1),
                    88.0 / TRACE_LENGTH,
                    104.0 / TRACE_LENGTH,
                ],
                dtype=torch.float32,
            )
        x_raw[sample_index] = trace

    return x_raw, interface_meta, interface_mask, labels


def _make_training_batch(
    device: torch.device,
    micro_batch_size: int,
) -> tuple[InterfaceBatch, torch.Tensor]:
    x_raw, interface_meta, interface_mask, labels = _make_unique_samples()
    repeat_factor = micro_batch_size // UNIQUE_SAMPLE_COUNT
    if repeat_factor * UNIQUE_SAMPLE_COUNT != micro_batch_size:
        raise RuntimeError("M2-07 micro batch must be divisible by the unique sample count.")
    x_raw = x_raw.repeat((repeat_factor, 1)).to(device)
    interface_meta = interface_meta.repeat((repeat_factor, 1, 1)).to(device)
    interface_mask = interface_mask.repeat((repeat_factor, 1)).to(device)
    labels = labels.repeat((repeat_factor, 1)).to(device)
    return (
        InterfaceBatch(
            x_raw=x_raw,
            length_mask=torch.ones_like(x_raw, dtype=torch.bool),
            interface_meta=interface_meta,
            interface_mask=interface_mask,
        ),
        labels,
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    batch: InterfaceBatch,
    labels: torch.Tensor,
    step: int,
) -> dict[str, Any]:
    model.eval()
    output = model(batch, labels)
    loss = pcc_training_loss(
        output.conditional.logits,
        output.conditional.base_logodds,
        labels,
        batch.interface_mask,
        lambda_local=LAMBDA_LOCAL,
    )
    valid = batch.interface_mask
    conditional_predictions = (output.conditional.logits >= 0).to(dtype=torch.int64)
    local_predictions = (output.conditional.base_logodds >= 0).to(dtype=torch.int64)
    conditional_correct = conditional_predictions[valid] == labels[valid]
    local_correct = local_predictions[valid] == labels[valid]
    invalid = ~valid
    return {
        "step": step,
        "total_loss": float(loss.total.detach()),
        "chain_loss": float(loss.chain.detach()),
        "local_loss": float(loss.local.detach()),
        "conditional_valid_accuracy": float(conditional_correct.float().mean()),
        "local_valid_accuracy": float(local_correct.float().mean()),
        "conditional_positive_accuracy": float(
            conditional_correct[labels[valid] == 1].float().mean()
        ),
        "conditional_negative_accuracy": float(
            conditional_correct[labels[valid] == 0].float().mean()
        ),
        "invalid_logit_max_abs": float(
            output.conditional.logits[invalid].abs().max().detach()
        ),
        "all_valid_logits_finite": bool(
            torch.isfinite(output.conditional.logits[valid]).all()
        ),
    }


def _criteria(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, bool]:
    return {
        "loss_finite": math.isfinite(final["total_loss"]),
        "loss_ratio_at_most_threshold": (
            final["total_loss"] / initial["total_loss"] <= LOSS_RATIO_THRESHOLD
        ),
        "conditional_accuracy_reaches_threshold": (
            final["conditional_valid_accuracy"] >= VALID_ACCURACY_THRESHOLD
        ),
        "local_accuracy_reaches_threshold": (
            final["local_valid_accuracy"] >= VALID_ACCURACY_THRESHOLD
        ),
        "both_classes_correct": (
            final["conditional_positive_accuracy"] >= VALID_ACCURACY_THRESHOLD
            and final["conditional_negative_accuracy"] >= VALID_ACCURACY_THRESHOLD
        ),
        "valid_logits_finite": final["all_valid_logits_finite"],
        "invalid_logits_zero": final["invalid_logit_max_abs"] == 0.0,
    }


def _frontend_contract() -> dict[str, Any]:
    return {
        "entrypoint": "scripts/m2_07_synthetic_tiny_overfit.py",
        "required_parameters": [
            "--runtime-config <RUNTIME_CONFIG>",
            "--output <RESULT_JSON>",
            "--log <RUN_LOG>",
        ],
        "pycharm": {
            "project": "a3-layerrecognition",
            "interpreter": "aislee-a3-layercongition",
            "same_cli_contract_available": True,
            "execution_recorded_by_this_run": False,
        },
        "codex_cli": {
            "same_cli_contract_available": True,
            "execution_recorded_by_this_run": True,
        },
        "contains_machine_absolute_paths": False,
        "codex_chat_state_required": False,
    }


def run(
    output_path: Path,
    runtime_config_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered A3 environment.")
    runtime_acceptance = build_runtime_acceptance(runtime_config_path)
    resolved_runtime = runtime_acceptance["resolved_runtime"]
    micro_batch_size = int(resolved_runtime["micro_batch_size"])
    gradient_accumulation_steps = int(
        resolved_runtime["gradient_accumulation_steps"]
    )
    if gradient_accumulation_steps != 1:
        raise RuntimeError("M2-07 currently requires gradient_accumulation_steps=1.")
    if resolved_runtime["precision"] != "float32" or resolved_runtime["amp_enabled"]:
        raise RuntimeError("M2-07 currently requires the frozen float32/AMP-off runtime.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    model = _build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    batch, labels = _make_training_batch(device, micro_batch_size)
    unique_x, _, unique_mask, unique_labels = _make_unique_samples()
    preflight = _nvidia_smi_snapshot()
    frontend_contract = _frontend_contract()

    initial = _evaluate(model, batch, labels, step=0)
    trace: list[dict[str, Any]] = [initial]
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    stop_reason = "max_steps"
    completed_steps = 0

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels)
        loss = pcc_training_loss(
            output.conditional.logits,
            output.conditional.base_logodds,
            labels,
            batch.interface_mask,
            lambda_local=LAMBDA_LOCAL,
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"M2-07 non-finite loss at step {step}.")
        loss.total.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"M2-07 missing or non-finite gradients at step {step}.")
        optimizer.step()
        completed_steps = step

        if step % EVALUATION_INTERVAL == 0 or step == MAX_STEPS:
            snapshot = _evaluate(model, batch, labels, step=step)
            trace.append(snapshot)
            if all(_criteria(initial, snapshot).values()):
                stop_reason = "preregistered_criteria_met"
                break

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - wall_started
    final = trace[-1]
    checks = _criteria(initial, final)
    checks.update(
        {
            "mixed_k_present": set(unique_mask.sum(dim=1).tolist()) == {2, 3},
            "both_label_classes_present": set(unique_labels[unique_mask].tolist()) == {0, 1},
            "micro_batch_matches_runtime_freeze": (
                batch.x_raw.shape[0] == micro_batch_size
            ),
            "gradient_accumulation_matches_runtime_freeze": (
                gradient_accumulation_steps == 1
            ),
            "runtime_acceptance_passed": runtime_acceptance["status"] == "PASS",
            "float32_without_amp": next(model.parameters()).dtype == torch.float32,
            "completed_at_least_one_step": completed_steps > 0,
            "frontend_contract_has_no_machine_absolute_paths": (
                frontend_contract["contains_machine_absolute_paths"] is False
            ),
            "registered_conda_environment_matches": (
                Path(sys.prefix).name == "aislee-a3-layercongition"
            ),
            "pythonpath_environment_variable_absent": not bool(
                os.environ.get("PYTHONPATH")
            ),
        }
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    repository = Path(__file__).resolve().parents[1]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "a3-m2-07-synthetic-tiny-overfit-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_id": "M2-07",
        "status": status,
        "evidence_class": "synthetic_learning_path_only",
        "scope": {
            "reads_real_rf": False,
            "reads_real_labels": False,
            "uses_synthetic_labels": True,
            "performs_optimizer_steps": True,
            "saves_checkpoint": False,
            "performs_model_selection": False,
            "runs_exact_enumeration": False,
        },
        "resolved_config": {
            "seed": SEED,
            "device": "cuda:0",
            "trace_length": TRACE_LENGTH,
            "unique_sample_count": UNIQUE_SAMPLE_COUNT,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "precision": resolved_runtime["precision"],
            "amp_enabled": resolved_runtime["amp_enabled"],
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "lambda_local": LAMBDA_LOCAL,
            "max_steps": MAX_STEPS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "loss_ratio_threshold": LOSS_RATIO_THRESHOLD,
            "valid_accuracy_threshold": VALID_ACCURACY_THRESHOLD,
        },
        "data_contract": {
            "interface_counts": sorted(set(unique_mask.sum(dim=1).tolist())),
            "valid_interface_count": int(unique_mask.sum()),
            "positive_valid_labels": int((unique_labels[unique_mask] == 1).sum()),
            "negative_valid_labels": int((unique_labels[unique_mask] == 0).sum()),
            "synthetic_signal_rule": (
                "Label-dependent signed Gaussian/ringing pulses inside fixed prior windows, "
                "plus deterministic low-amplitude noise and baseline variation."
            ),
            "unique_x_sha256": _tensor_sha256(unique_x),
        },
        "training": {
            "completed_steps": completed_steps,
            "stop_reason": stop_reason,
            "elapsed_seconds": elapsed_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "initial": initial,
            "final": final,
            "loss_ratio": final["total_loss"] / initial["total_loss"],
            "trace": trace,
        },
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": properties.name,
            "device_total_memory_mib": properties.total_memory / MIB,
            "compute_ledger_present": False,
            "registered_conda_environment": "aislee-a3-layercongition",
            "resolved_environment_name": Path(sys.prefix).name,
            "pythonpath_environment_variable_present": bool(
                os.environ.get("PYTHONPATH")
            ),
        },
        "preflight": preflight,
        "runtime_contract": {
            "config_name": runtime_config_path.name,
            "config_sha256": _sha256(runtime_config_path),
            "resolved_runtime": resolved_runtime,
            "runtime_acceptance_input_sha256": runtime_acceptance["input_sha256"],
        },
        "execution_frontend_contract": frontend_contract,
        "artifact_contract": {
            "result_name": output_path.name,
            "companion_log_name": log_path.name,
            "companion_log_generated_by_runner": True,
        },
        "source_repository": _git_state(repository),
        "source_sha256": {
            "integrated.py": _sha256(
                repository / "src" / "a3_ultrasound" / "models" / "integrated.py"
            ),
            "losses.py": _sha256(
                repository / "src" / "a3_ultrasound" / "models" / "losses.py"
            ),
            "runtime_contract.py": _sha256(
                repository / "src" / "a3_ultrasound" / "runtime_contract.py"
            ),
            "m2_real_rf_forward_smoke.py": _sha256(
                repository / "scripts" / "m2_real_rf_forward_smoke.py"
            ),
            "m2_04_gpu_profile.py": _sha256(
                repository / "scripts" / "m2_04_gpu_profile.py"
            ),
            "m2_07_synthetic_tiny_overfit.py": _sha256(Path(__file__).resolve()),
        },
        "interpretation_boundary": (
            "PASS means only that the full PCC truth-prefix loss and optimizer path can "
            "memorize one fixed synthetic batch. It is not real-RF performance, "
            "generalization, training stability, model selection, matched-joint parity, "
            "or propagation-mechanism evidence."
        ),
    }
    summary = json.dumps(
        {
            "run_id": result["run_id"],
            "status": result["status"],
            "completed_steps": result["training"]["completed_steps"],
            "initial_total_loss": result["training"]["initial"]["total_loss"],
            "final_total_loss": result["training"]["final"]["total_loss"],
            "loss_ratio": result["training"]["loss_ratio"],
            "conditional_valid_accuracy": result["training"]["final"][
                "conditional_valid_accuracy"
            ],
            "output": output_path.name,
        },
        ensure_ascii=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(summary + "\n", encoding="utf-8")
    result["artifact_contract"]["companion_log_sha256"] = _sha256(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError(f"M2-07 checks failed: {checks}")
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output, args.runtime_config, args.log)
    summary = json.dumps(
        {
            "run_id": result["run_id"],
            "status": result["status"],
            "completed_steps": result["training"]["completed_steps"],
            "initial_total_loss": result["training"]["initial"]["total_loss"],
            "final_total_loss": result["training"]["final"]["total_loss"],
            "loss_ratio": result["training"]["loss_ratio"],
            "conditional_valid_accuracy": result["training"]["final"][
                "conditional_valid_accuracy"
            ],
            "output": args.output.name,
        },
        ensure_ascii=False,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
