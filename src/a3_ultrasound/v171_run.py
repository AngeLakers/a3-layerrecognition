"""Prepare, train and evaluate the authorized V1.7.1 fixed-budget matrix."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from a3_ultrasound.data.epoch_sampling import freeze_epochs
from a3_ultrasound.data.v171_cohort import InsufficientSupport, complete_target, file_sha, prepare_roles, read_index, write_json
from a3_ultrasound.v171_protocol import (
    PATH_KEYS, PROTOCOL, TARGETS, cell_identity, load_config, matrix, selection_artifacts, validate_config,
)


def _relocate(cfg, directory):
    copy = dict(cfg)
    for name in PATH_KEYS:
        if copy.get(name):
            copy[name] = Path(os.path.relpath(copy[name], directory)).as_posix()
    return copy


def _freeze_arm(catalog_path, entries, fold, role_path, output, seed, signature):
    role = json.loads(role_path.read_bytes())
    shards = []
    for board in fold["train"]:
        entry = entries[board]
        arrays = read_index(catalog_path, entry)
        mask = np.asarray(arrays["eligible"], dtype=bool).copy()
        for position, source_row in role["excluded_rows"][board]:
            if arrays["source_row"][position] != source_row or not mask[position]:
                raise ValueError("Frozen TRAIN exclusion does not match the original row.")
            mask[position] = False
        shards.append({**entry, "arrays": arrays, "sampling_eligible": mask})
    data = SimpleNamespace(shards=shards, entities=fold["train"], row_view_sha256=file_sha(role_path))
    freeze_epochs(data, output, seed=seed, epochs=40, catalog_signature=signature)


def prepare(reference, catalog_path, fixed_artifacts, saved_panels, output, *, target="101"):
    """Prepare exact runnable cells. Only Labels V2 indexes are read."""
    from a3_ultrasound.stream_training import catalog_signature, validate_fold

    reference = json.loads(Path(reference).read_bytes())
    output, catalog_path, fixed_artifacts, saved_panels = [
        Path(p).resolve() for p in (output, catalog_path, fixed_artifacts, saved_panels)
    ]
    catalog = json.loads(catalog_path.read_bytes())
    signature = catalog_signature(catalog)
    if signature != reference["catalog_signature"]:
        raise ValueError("Local catalog is not the archived fixed40 catalog.")
    folds = {fold["fold"]: fold for fold in reference["folds"]}
    entries = {entry["entity"]: entry for entry in catalog["entities"]}
    for fold in folds.values():
        validate_fold(fold, catalog)
    support_failures = {}
    for fold_id in sorted({cell.fold for cell in matrix() if cell.block == "B4"}):
        fold = folds[fold_id]
        k3_boards = [entries[board] for board in fold["test"] if entries[board]["K"] == 3]
        if len(k3_boards) != 1:
            raise ValueError(f"{fold_id} must have one registered K3 outer specimen.")
        index = read_index(catalog_path, k3_boards[0])
        if not (complete_target(k3_boards[0], index, target) & index["eligible"]).any():
            support_failures[fold_id] = f"{fold_id} has no complete held-out target label support."
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "configs"
    configs.mkdir()
    path_map = {
        "timebase_calibration_path": fixed_artifacts / "timebase_calibration.json",
        "m6_decision_path": fixed_artifacts / "m6_decision.json",
        "speed_decision_path": fixed_artifacts / "speed_decision.json",
    }
    excluded = {
        "direct_core_authorization", "direct_core_authorization_sha256", "a5_authorization",
        "a5_authorization_sha256", "a5_stage", "protocol_signature",
    }
    common = {k: v for k, v in reference.items() if k not in excluded}
    common.update({k: str(path) for k, path in path_map.items()})
    common.update(protocol=PROTOCOL, catalog=str(catalog_path), outer_evaluation_permitted=False, development_stage=None)
    files = []
    prepared = set()
    for cell in matrix():
        cfg = dict(common, cell=asdict(cell), heads=[cell.head], seeds=[cell.seed],
                   held_out_target=target if cell.block == "B4" else None)
        cfg["model"] = dict(common["model"], backbone=cell.backbone,
                            readout_scope="full_waveform" if cell.block == "B2" else "indexed")
        cfg["training"] = dict(common["training"], seed=cell.seed, outer_evaluation_permitted=False)
        if cell.block == "B4":
            fold = folds[cell.fold]
            role_root = output / "cohorts" / f"{cell.fold}_s{cell.seed}"
            key = cell.fold, cell.seed
            if key not in prepared and key not in support_failures and cell.fold not in support_failures:
                try:
                    prepare_roles(catalog_path, catalog, fold, target, cell.seed, role_root)
                except InsufficientSupport as exc:
                    support_failures[key] = str(exc)
            reason = support_failures.get(cell.fold) or support_failures.get(key)
            if reason:
                files.append({"run_id": cell.run_id, "block": cell.block,
                              "status": "NOT_ESTIMABLE", "reason": reason})
                continue
            if key not in prepared:
                for arm in ("B", "C"):
                    _freeze_arm(catalog_path, entries, fold, role_root / f"{arm}.json",
                                role_root / f"epochs_{arm}", cell.seed, signature)
                prepared.add(key)
                print(f"Prepared paired B/C manifest and train/validation support for {cell.fold}/s{cell.seed}", flush=True)
            cfg.update(role_root=str(role_root), epoch_selection_root=str(role_root / f"epochs_{cell.arm}"),
                       validation_selection_root=str(role_root),
                       role_sha256={arm: file_sha(role_root/f"{arm}.json") for arm in (cell.arm, "validation")})
        else:
            cfg.update(epoch_selection_root=str(saved_panels / "epochs" / f"{cell.fold}_s{cell.seed}"),
                       validation_selection_root=str(saved_panels / "validation_selection"))
        cfg["epoch_manifest_sha256"] = file_sha(Path(cfg["epoch_selection_root"]) / "manifest.json")
        validation = Path(cfg["validation_selection_root"])
        cfg["validation_sha256"] = file_sha(validation / f"{cell.fold}.npz")
        cfg["validation_manifest_sha256"] = file_sha(validation / "manifest.json")
        validate_config(cfg, fold_name=cell.fold, seed=cell.seed, head=cell.head, phase="train", require_permission=False)
        path = configs / f"{cell.run_id}.json"
        write_json(path, _relocate(cfg, configs))
        files.append({"run_id": cell.run_id, "block": cell.block, "config": path.relative_to(output).as_posix(),
                      "sha256": file_sha(path), "status": "PREPARED_NOT_AUTHORIZED"})
    write_json(output / "matrix.json", {"protocol": PROTOCOL, "catalog_signature": signature,
                                      "target": target, "cells": files, "training_authorized": False,
                                      "outer_authorized": False})
    return files


def train(config, output, authorization):
    from a3_ultrasound.stream_training import train as train_loop

    cfg = load_config(config)
    cell = cell_identity(cfg)
    summary = train_loop(config, output, fold_name=cell["fold"], head=cell["head"],
                         seed=cell["seed"], authorization=authorization)
    if summary["status"] == "COMPLETED" and not summary["guard_violated"] and summary["epochs"] == 40:
        path = Path(output)
        write_json(path / "guarded_receipt.json", {
            "cell": cell, "selection": selection_artifacts(cfg), "config_sha256": file_sha(config),
            "weights_sha256": file_sha(path / "best_guarded.pt"), "guarded": summary["best_guarded"],
        })
    return summary


def _guarded_checkpoint(cfg, config, run, *, device):
    import torch
    from a3_ultrasound.training import ModelConfig

    run = Path(run)
    receipt = json.loads((run / "guarded_receipt.json").read_bytes())
    path = run / "best_guarded.pt"
    if (receipt["cell"] != cell_identity(cfg) or receipt["selection"] != selection_artifacts(cfg)
            or receipt["config_sha256"] != file_sha(config) or receipt["weights_sha256"] != file_sha(path)):
        raise ValueError("Guarded receipt and active cell, frozen panel or weights disagree.")
    c = torch.load(path, map_location=device, weights_only=True)
    resolved = c["resolved_config"]
    guarded = c["selection_state"]["guarded"]
    expected_model = asdict(ModelConfig(**dict(cfg["model"], head_type=receipt["cell"]["head"])))
    if (not guarded or guarded != receipt["guarded"] or guarded["epoch"] != c["epoch"]
            or not 0 <= guarded["normal_fpr"] <= .02 or resolved.get("diagnostic")
            or resolved.get("v171_cell") != receipt["cell"]
            or resolved.get("v171_selection") != receipt["selection"]
            or resolved.get("config_sha256") != file_sha(config)
            or resolved.get("speed") != cfg["speed"] or resolved.get("model") != expected_model):
        raise ValueError("The supplied epoch is not the selected, eligible V1.7.1 checkpoint.")
    return c


def evaluate(config, run, output, authorization, *, phase):
    from a3_ultrasound.speed import SpeedOptions, configure_backend, configure_model, dataset_from_config
    from a3_ultrasound.stream_training import evaluate_stream
    from a3_ultrasound.training import ModelConfig, build_model

    if phase not in ("outer", "calibration-validation"):
        raise ValueError("Unknown evaluation phase.")
    cfg = load_config(config)
    cell = cell_identity(cfg)
    validate_config(cfg, fold_name=cell["fold"], seed=cell["seed"], head=cell["head"],
                    phase="outer" if phase=="outer" else "train", config_path=config,
                    authorization=authorization)
    device = cfg["training"]["device"]
    c = _guarded_checkpoint(cfg, config, run, device=device)
    options = SpeedOptions(**cfg["speed"])
    configure_backend(options)
    model = build_model(ModelConfig(**c["resolved_config"]["model"]), v171=True).to(device)
    configure_model(model, options)
    model.load_state_dict(c["model_state_dict"], strict=True)
    split = "test" if phase == "outer" else "validation"
    runtime = dict(cfg, fold_name=cell["fold"], outer_evaluation_permitted=phase == "outer")
    data = dataset_from_config(runtime, split, sampled_validation=False)
    fold = next(item for item in runtime["folds"] if item["fold"] == cell["fold"])
    if data.entities != fold[split]:
        raise ValueError("Evaluation specimen identities differ from the active fold.")
    if phase == "outer" and (data.row_view_sha256 is not None or set(data.entities) & set(fold["train"] + fold["validation"])):
        raise ValueError("Outer inputs must be the original, isolated cohort.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metrics = evaluate_stream(model, data, batch_size=runtime["evaluation_batch_size"], device=device,
                              prediction_path=output / "predictions.jsonl", phase=phase, speed_options=options)
    result = {"status": "EVALUATED", "phase": phase, "cell": cell, "epoch": c["epoch"],
              "records": len(data), "config_sha256": file_sha(config),
              "predictions_sha256": file_sha(output / "predictions.jsonl"),
              "guarded_weights_sha256": file_sha(Path(run) / "best_guarded.pt"), "metrics": metrics}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    for name in ("reference", "catalog", "fixed-artifacts", "saved-panels", "output"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--target", choices=TARGETS, default="101")
    for phase in ("train", "outer", "calibration-validation"):
        command = sub.add_parser(phase)
        for name in ("config", "output", "authorization"):
            command.add_argument(f"--{name}", type=Path, required=True)
        if phase != "train":
            command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        cells = prepare(args.reference, args.catalog, args.fixed_artifacts, args.saved_panels, args.output,
                        target=args.target)
        ready = sum(cell["status"] == "PREPARED_NOT_AUTHORIZED" for cell in cells)
        print(f"{ready}/{len(cells)} cells prepared. Train and outer remain closed.")
    elif args.phase == "train":
        train(args.config, args.output, args.authorization)
    else:
        evaluate(args.config, args.run, args.output, args.authorization, phase=args.phase)


if __name__ == "__main__":
    main()
