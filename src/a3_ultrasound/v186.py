"""Prepare and fit the two Stage 1 arms using the original recovery configuration bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from a3_ultrasound.training import _code_version


ARMS = ("score", "feature")
PROTOCOL = "A3-v1.8.6-stage1"


def registry():
    return json.loads(Path(__file__).with_name("v186_inputs.json").read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Each prediction export belongs to exactly one named checkpoint.
EXPORTS = {"best": ("predictions.jsonl", "best_guarded.pt"),
           "fixed15": ("predictions_fixed_epoch15.jsonl", "epoch_15.pt"),
           "last": ("predictions_last_epoch.jsonl", "last_epoch.pt")}


def export_receipts(output, summary):
    epochs = {"best": summary["best_epoch"], "fixed15": 15, "last": summary["epochs"]}
    return {key: {"predictions": predictions, "predictions_sha256": digest(output / predictions),
                  "checkpoint": weights, "checkpoint_sha256": digest(output / weights), "epoch": epochs[key]}
            for key, (predictions, weights) in EXPORTS.items() if epochs[key] is not None}



def configuration(baseline_dir, fold, arm):
    source = baseline_dir / f"{fold}_resnet1d_nll_128_s23.json"
    expected = registry()[fold]["config"]
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or arm not in ARMS:
        raise ValueError("Unregistered recovery configuration or donor arm")
    config = json.loads(raw)
    config["model"]["stage1_donor"] = arm
    config.update(protocol=PROTOCOL, execution_version=PROTOCOL, recovery_config_sha256=expected,
                  recovery_source_commit=config["source_commit"], source_commit=_code_version()["commit"])
    return config


def validate_configuration(cfg, fold, arm):
    if (cfg["protocol"] != PROTOCOL or cfg["execution_version"] != PROTOCOL
            or cfg["folds"][0]["fold"] != fold or arm not in ARMS
            or cfg["model"]["stage1_donor"] != arm):
        raise ValueError("Unregistered Stage 1 run")
    settings = dict(cfg)
    settings["model"] = dict(cfg["model"])
    del settings["model"]["stage1_donor"]
    for key in ("protocol", "execution_version", "source_commit", "recovery_source_commit", "recovery_config_sha256"):
        del settings[key]
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    entry = registry()[fold]
    if cfg["recovery_config_sha256"] != entry["config"] or hashlib.sha256(canonical).hexdigest() != entry["settings"]:
        raise ValueError("Run settings differ from the frozen control")


def verify_inputs(cfg):
    from a3_ultrasound.stream_training import catalog_signature

    fold = cfg["folds"][0]["fold"]
    files = [(Path(cfg["epoch_selection_root"]) / "manifest.json", "epoch_manifest_sha256"),
             (Path(cfg["validation_selection_root"]) / "manifest.json", "validation_manifest_sha256"),
             (Path(cfg["validation_selection_root"]) / f"{fold}.npz", "validation_sha256")]
    for key in ("timebase_calibration", "m6_decision", "speed_decision"):
        files.append((Path(cfg[f"{key}_path"]), f"{key}_sha256"))
    for path, key in files:
        if digest(path) != cfg[key]:
            raise ValueError(f"Frozen input changed: {key}")
    if catalog_signature(json.loads(Path(cfg["catalog"]).read_bytes())) != cfg["catalog_signature"]:
        raise ValueError("Frozen catalog changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("fold", choices=tuple(registry()))
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    cfg = configuration(args.baseline_dir, args.fold, args.arm)
    path = args.job / "config.json"
    if args.command == "prepare":
        args.job.mkdir(parents=True, exist_ok=False)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        return
    if json.loads(path.read_bytes()) != cfg:
        raise ValueError("Prepared configuration or code revision changed")
    from a3_ultrasound.stream_training import train
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    train(path, args.job / "run", fold_name=args.fold, head="local_independent", seed=23)


if __name__ == "__main__":
    main()
