"""Cloud-only campaign preparation and model fitting for the mixed-K v1.8.5 plan."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from a3_ultrasound.data.epoch_sampling import freeze_epochs
from a3_ultrasound.speed import dataset_from_config
from a3_ultrasound.stream_training import catalog_signature, train
from a3_ultrasound.validation_sampling_audit import metadata, sample_panel


BACKBONES = ("resnet1d", "inceptiontime", "tcn", "patchtst")
ROLE_NAMES = ("base_fit", "early_stop", "residual_fit", "calibration_workpoint", "dev_evaluation", "sealed")


def loss_arms(backbone):
    return ("nll", "weighted", "additive", "exchange") if backbone == "resnet1d" else ("nll", "exchange")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_base_schedule(cfg):
    required = dict(selection_rule="guarded_joint", epochs=40, min_epochs=15, patience=10,
                    normal_fpr_limit=.02, lr_schedule="cosine", lr_warmup_epochs=2,
                    learning_rate=.001, lr_min=.00001, scheduler_step_size=None, min_delta=.002)
    for field, expected in required.items():
        if cfg["training"].get(field) != expected:
            raise ValueError(f"v1.8.5 base training requires {field}={expected}")


def class_support(dataset):
    width = max(shard["K"] for shard in dataset.shards)
    pos = np.zeros(width, dtype=np.int64)
    counts = np.zeros_like(pos)
    for ordinal, shard in enumerate(dataset.shards):
        ids = dataset.refs[dataset.refs[:, 0] == ordinal, 1]
        k = shard["K"]
        y = shard["arrays"]["y"][ids, :k]
        if not shard["arrays"]["label_mask"][ids, :k].all() or not np.isin(y, (0, 1)).all():
            raise ValueError("Incomplete labels in training cohort")
        pos[:k] += y.sum(0, dtype=np.int64)
        counts[:k] += len(ids)
    negative = counts - pos
    supported = bool(np.all((pos > 0) & (negative > 0)))
    return dict(positive=pos.tolist(), negative=negative.tolist(),
                weights=(negative / pos).tolist() if supported else None,
                status="ready" if supported else "skipped_missing_class")


def prepare(template, roles_csv, target):
    cfg = read(template)
    if cfg.get("protocol") != "A3-v1.8-M1-paired-v1" or cfg["m1_arm"] != "G":
        raise ValueError("Expected a frozen M1 G template")
    check_base_schedule(cfg)
    model = cfg["model"]
    if model.get("relative_time_attention") is not True or model.get("readout_scope") != "full_waveform":
        raise ValueError("Expected a full-waveform relative-time readout in the M1 G template")
    if (model.get("query_dim"), model.get("attention_dim")) != (64, 64):
        raise ValueError("Expected 64-dimensional query and attention in the M1 G template")
    # Each M1 template contains the same five splits. Its m1_fold selects the run.
    fold = cfg["m1_fold"]
    original, = [entry for entry in cfg["folds"] if entry["fold"] == fold]
    with open(roles_csv, encoding="utf-8-sig", newline="") as f:
        row, = [r for r in csv.DictReader(f) if r["fold"] == fold]
    roles = {k: row[k].split(" / ") for k in ROLE_NAMES}
    flat = sum(roles.values(), [])
    if (len(flat) != len(set(flat)) or set(roles["sealed"]) != set(original["test"])
            or set(roles["dev_evaluation"]) != set(original["validation"])
            or set(flat) != set(original["train"] + original["validation"] + original["test"])):
        raise ValueError("Role split overlaps or differs from original fold")
    # The historical runner calls its unopened complement 'test'. Store the six true roles separately.
    splits = dict(fold=fold, train=roles["base_fit"], validation=roles["early_stop"],
                  test=sorted(set(flat) - set(roles["base_fit"] + roles["early_stop"])))
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg.update(protocol="A3-v1.8.5", execution_version="v1.8.5", folds=[splits], roles=roles,
               outer_evaluation_permitted=False, epoch_selection_root=None, validation_selection_root=None)
    model.update(common_feature_dim=128, token_dim=128)
    if cfg["catalog_signature"] != catalog_signature(read(cfg["catalog"])):
        raise ValueError("Catalog signature mismatch")
    for path_key, sha_key in (("timebase_calibration_path","timebase_calibration_sha256"),
                              ("m6_decision_path","m6_decision_sha256")):
        if hashlib.sha256(Path(cfg[path_key]).read_bytes()).hexdigest() != cfg[sha_key]:
            raise ValueError(f"Input mismatch: {path_key}")
    with TemporaryDirectory(dir=target.parent) as temp:
        work = Path(temp) / "prepared"
        work.mkdir()
        data = dataset_from_config(cfg, "train")
        support = class_support(data)
        write(work/"weighted_support.json", support)
        freeze_epochs(data, work/"epochs", seed=23, epochs=cfg["training"]["epochs"],
                      catalog_signature=cfg["catalog_signature"])
        for role in ("early_stop", "residual_fit", "calibration_workpoint", "dev_evaluation"):
            block = role_dataset(cfg, role)
            meta = metadata(block)
            idx, weight = sample_panel(meta, target=13334, seed=23)
            directory = work / role
            directory.mkdir()
            panel = directory / f"{fold}.npz"
            np.savez(panel, indices=idx, weights=weight, refs=block.refs[idx], entities=block.entities)
            digest = hashlib.sha256(panel.read_bytes()).hexdigest()
            write(directory/"manifest.json", dict(catalog_signature=cfg["catalog_signature"],
                  policy="stratified per entity, inverse sampling weights", seed=23,
                  folds={fold:dict(sha256=digest)}))
        cfg.update(epoch_selection_root=str(target/"epochs"), validation_selection_root=str(target/"early_stop"))
        cfg["epoch_manifest_sha256"] = hashlib.sha256((work/"epochs/manifest.json").read_bytes()).hexdigest()
        cfg["validation_manifest_sha256"] = hashlib.sha256((work/"early_stop/manifest.json").read_bytes()).hexdigest()
        cfg["validation_sha256"] = read(work/"early_stop/manifest.json")["folds"][fold]["sha256"]
        for backbone in BACKBONES:
            for arm in loss_arms(backbone):
                if arm == "weighted" and support["weights"] is None:
                    continue
                cell = dict(cfg, model=dict(cfg["model"],backbone=backbone),
                            training=dict(cfg["training"],v185_loss=arm,layer_weights=support["weights"] if arm=="weighted" else None))
                write(work/f"{backbone}_{arm}.json",cell)
        work.rename(target)


def role_dataset(cfg, role):
    if role not in ("early_stop", "residual_fit", "calibration_workpoint", "dev_evaluation"):
        raise ValueError("Role not open for feature extraction")
    fold = dict(cfg["folds"][0], validation=cfg["roles"][role])
    return dataset_from_config(dict(cfg, folds=[fold]), "validation", sampled_validation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("template", "roles_csv", "output"):
        prep.add_argument(key,type=Path)
    fit = sub.add_parser("fit")
    fit.add_argument("config",type=Path)
    fit.add_argument("output",type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.template,args.roles_csv,args.output)
    else:
        cfg=read(args.config)
        if cfg.get("protocol")!="A3-v1.8.5" or cfg.get("outer_evaluation_permitted") is not False:
            raise ValueError("Expected a prepared v1.8.5 development config")
        check_base_schedule(cfg)
        # The historical M1 launcher supplied this before the deterministic CUDA runner.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        train(args.config,args.output,fold_name=cfg["folds"][0]["fold"],head="local_independent",seed=23)


if __name__ == "__main__":
    main()

