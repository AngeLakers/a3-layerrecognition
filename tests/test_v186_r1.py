from dataclasses import asdict
from pathlib import Path
import json

import pytest
import torch

from a3_ultrasound.models.losses import head_training_loss, r1_supervision_weights
from a3_ultrasound.models.pcc import HeadStepOutput
from a3_ultrasound.schema import InterfaceBatch
from a3_ultrasound.training import ModelConfig, build_model, build_optimizer, TrainingConfig
from a3_ultrasound.v186 import digest
from a3_ultrasound.v186_r1 import load_frozen_encoder
from a3_ultrasound.v186_r1_replay import policies
from a3_ultrasound.v186_replay import policies as original_policies


def test_weighting_spans_the_batch_and_excludes_padding_and_current_target():
    labels = torch.tensor([[0, 0, -1], [1, 1, 0], [0, 1, 1], [1, 0, 0]])
    mask = labels >= 0
    z = torch.zeros(4, 3, requires_grad=True)
    target = labels.masked_fill(~mask, 0)
    e1 = r1_supervision_weights(z, target, mask, "easy_normal")
    e2 = r1_supervision_weights(z, target, mask, "single_upstream")
    assert not e1.requires_grad
    torch.testing.assert_close(e1, torch.tensor([[0.875, 0.875, 0.], [1., 1., 1.],
                                                [1., 1., 1.], [1., 1., 1.]]))
    torch.testing.assert_close(e2, torch.tensor([[1., 1., 0.], [1., 1.5, 1.],
                                                [1., 1., 1.5], [1., 1.5, 1.5]]))
    batch = InterfaceBatch(torch.zeros(4, 10), torch.ones(4, 10, dtype=torch.bool),
                           torch.zeros(4, 3, 5), mask, mask, labels)
    output = HeadStepOutput(logits=z, base_logodds=z)
    for mode, weights in (("easy_normal", e1), ("single_upstream", e2)):
        weighted = head_training_loss(output, batch, head_type="local_independent", v185_loss=mode)
        weighted.total.backward()
        w = weights / mask.sum(dim=1, keepdim=True)
        torch.testing.assert_close(z.grad, (0.5 - target) * w / w.sum())
        z.grad = None


@pytest.mark.parametrize("recovery_format", [False, True])
def test_frozen_encoder_rejects_an_overlapping_fold(tmp_path: Path, recovery_format):
    cfg = ModelConfig(architecture_version="a3-v1.8", backbone_norm="group8", canonical_per_trace=False,
                      head_type="local_independent", query_view="position2", common_feature_dim=128, token_dim=128,
                      relative_time_attention=True, readout_scope="full_waveform")
    model, source_model = build_model(cfg), build_model(cfg)
    fold = {"fold": "F1", "train": ["A"], "validation": ["B"], "test": ["C"]}
    path = tmp_path / "epoch_15.pt"
    resolved = {"fold": fold, "selection_entities": ["A"], "model": asdict(cfg),
                "diagnostic": False, "test_opened": False, "config_sha256": "abc"}
    checkpoint = {"epoch": 15, "model_state_dict": source_model.state_dict()}
    source = {"checkpoint": str(path)}
    if recovery_format:
        checkpoint["config_sha256"] = "abc"
        sidecar = path.with_name("resolved_config.json")
        sidecar.write_text(json.dumps(resolved), encoding="utf8")
        source["resolved_sha256"] = digest(sidecar)
    else:
        checkpoint["resolved_config"] = resolved
    torch.save(checkpoint, path)
    source["sha256"] = digest(path)
    with pytest.raises(ValueError, match="roles differ"):
        load_frozen_encoder(model, source, dict(fold, validation=["C"], test=["B"]), cfg)
    modules = load_frozen_encoder(model, source, fold, cfg)
    assert all(not module.training for module in modules)
    assert all(not p.requires_grad for module in modules for p in module.parameters())
    optimizer, counts = build_optimizer(model, TrainingConfig())
    assert "backbone" not in counts and "backbone_adapter" not in counts
    assert "local_readout" in counts and optimizer.param_groups
    for one, two in zip(model.backbone.parameters(), source_model.backbone.parameters()):
        torch.testing.assert_close(one, two)


def test_c3_controls_reuse_the_historical_points_and_counts():
    rows = [
        {"sample_id": f"B:{i}", "sample_weight": 1, "base_logodds": [0.01 * (i + 1), 3.0],
         "map_states": [1, 1], "y": [0, 1], "label_mask": [True, True]}
        for i in range(20)
    ]
    legacy = original_policies(rows)
    r1 = policies(rows, torch.arange(20).numpy())
    for arm in ("uncertain", "uniform"):
        assert {key: r1[arm][key] for key in legacy[arm]} == dict(legacy[arm])
    assert r1["uncertain"]["held"] == r1["deepest"]["held"] == 2
    assert r1["uncertain"]["review_errors"] == 0
    assert r1["deepest"]["review_errors"] == 2
    assert {one["review"] for one in r1.values()} == {2}





def test_r1_cannot_enter_tensor_fit_or_streaming_outer(tmp_path):
    from a3_ultrasound.stream_training import test_checkpoint
    from a3_ultrasound.training import run_from_config, run_training

    with pytest.raises(ValueError, match="streaming frozen-encoder"):
        run_training(ModelConfig(), TrainingConfig(v185_loss="easy_normal"), {}, tmp_path,
                     data_kind="synthetic", input_manifest={})
    tensor_config = tmp_path / "tensor.json"
    for control in ("nll", "weighted"):
        tensor_config.write_text(json.dumps({"schema_version": "a3-run-v1.4",
                                            "training": {"v185_loss": control},
                                            "r1_frozen_encoder": {"sha256": "marker"}}))
        with pytest.raises(ValueError, match="streaming frozen-encoder"):
            run_from_config(tensor_config, tmp_path)
    config = tmp_path / "outer.json"
    config.write_text(json.dumps({"outer_evaluation_permitted": True, "training": {"device": "cpu"}}))
    checkpoint = tmp_path / "development.pt"
    torch.save({"resolved_config": {"r1_frozen_encoder": {"sha256": "marker"}}}, checkpoint)
    with pytest.raises(ValueError, match="Outer evaluation is closed for an R1"):
        test_checkpoint(config, checkpoint, tmp_path)

def test_c3_fusion_priorities_do_not_depend_on_sampling_weights(monkeypatch):
    import numpy as np
    from a3_ultrasound import v186_r1_replay

    z = [[2, 7], [3, 6], [1, 4], [0, 5]]
    risk = np.array([3, 2, 1, 0])
    seen = []
    original = v186_r1_replay.budgeted_points

    def check_priority(rows, priority, budget):
        seen.append(np.asarray(priority).copy())
        return original(rows, priority, budget)

    monkeypatch.setattr(v186_r1_replay, "budgeted_points", check_priority)
    for weights in ([1, 1, 5, 5], [5, 5, 1, 1]):
        rows = [
            {"sample_id": f"B:{i}", "sample_weight": weights[i], "base_logodds": logits,
             "map_states": [0, 1], "y": [0, 0], "label_mask": [True, True]}
            for i, logits in enumerate(z)
        ]
        v186_r1_replay.policies(rows, reflection_risk=risk, budget=0.5)
    assert len(seen) == 4
    np.testing.assert_array_equal(seen[0], seen[2])
    np.testing.assert_array_equal(seen[1], seen[3])
    np.testing.assert_array_equal(v186_r1_replay.percentile_ranks(np.array([2, 2, 1])), [2/3, 2/3, 1/6])


def test_r1_frozen_config_and_checkpoint_rejected_in_v171(tmp_path, monkeypatch):
    from a3_ultrasound import stream_training, v171_run
    from a3_ultrasound.v171_protocol import PROTOCOL

    cfg = {"protocol": PROTOCOL, "r1_frozen_encoder": {"sha256": "marker"}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf8")
    with pytest.raises(ValueError, match="no-thin development protocol"):
        stream_training.train(config_path, tmp_path / "out", fold_name="F1",
                              head="local_independent", seed=23)

    # A consistent receipt must still be rejected before reading evaluation data.
    checkpoint = tmp_path / "best_guarded.pt"
    torch.save({"resolved_config": {"r1_frozen_encoder": cfg["r1_frozen_encoder"]}}, checkpoint)
    monkeypatch.setattr(v171_run, "cell_identity", lambda _: {"block": "B2"})
    monkeypatch.setattr(v171_run, "selection_artifacts", lambda _: {"fold": "F1"})
    receipt = {"cell": {"block": "B2"}, "selection": {"fold": "F1"},
               "config_sha256": digest(config_path), "weights_sha256": digest(checkpoint)}
    (tmp_path / "guarded_receipt.json").write_text(json.dumps(receipt), encoding="utf8")
    with pytest.raises(ValueError, match="R1 development checkpoints"):
        v171_run._guarded_checkpoint(cfg, config_path, tmp_path, device="cpu")
