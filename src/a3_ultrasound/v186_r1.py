"""v1.8.6 R1 frozen encoder setup for paired readout experiments."""

import json
from pathlib import Path

import torch

from a3_ultrasound.v186 import digest


DEVELOPMENT_PROTOCOL = "A3-v1.8.6-no-thin"


def load_frozen_encoder(model, source, fold, model_config):
    path = Path(source["checkpoint"])
    if digest(path) != source["sha256"]:
        raise ValueError("Frozen encoder checkpoint hash differs from the experiment manifest.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != 15:
        raise ValueError("R1 uses the prespecified epoch-15 encoder checkpoint.")
    resolved = checkpoint.get("resolved_config")
    if resolved is None:
        sidecar = path.with_name("resolved_config.json")
        if not source.get("resolved_sha256") or digest(sidecar) != source["resolved_sha256"]:
            raise ValueError("Recovery encoder requires the verified resolved-config sidecar.")
        resolved = json.loads(sidecar.read_bytes())
        if not checkpoint.get("config_sha256") or checkpoint["config_sha256"] != resolved.get("config_sha256"):
            raise ValueError("Recovery encoder and its sidecar have different run configs.")
    source_fold = resolved["fold"]
    if (resolved.get("diagnostic") is not False or resolved.get("test_opened") is not False
            or checkpoint.get("diagnostic_weights_never_reuse", False)
            or source_fold["fold"] != fold["fold"]
            or any(set(source_fold[role]) != set(fold[role]) for role in ("train", "validation", "test"))
            or set(resolved["selection_entities"]) != set(fold["train"])):
        raise ValueError("Encoder checkpoint roles differ from this development fold.")
    for field in ("architecture_version", "backbone", "backbone_norm", "backbone_stage",
                  "canonical_per_trace", "input_channels", "common_feature_dim"):
        if resolved["model"].get(field, 1 if field == "input_channels" else None) != getattr(model_config, field):
            raise ValueError(f"Encoder checkpoint architecture differs at {field}.")
    state = checkpoint["model_state_dict"]
    for name in ("backbone", "backbone_adapter"):
        module = getattr(model, name)
        prefix = f"{name}."
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                if key.startswith(prefix)}, strict=True)
        module.requires_grad_(False)
        module.eval()
    return model.backbone, model.backbone_adapter
