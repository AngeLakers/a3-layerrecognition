from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a3_ultrasound import stream_training
from a3_ultrasound.recovery_diagnostics import layer_metrics


@pytest.mark.parametrize("passing", [True, False])
def test_recovery_keeps_last_and_fixed_weights_before_selecting(tmp_path, monkeypatch, passing):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"input_config": {}, "entities": [{"entity": e} for e in ("fit", "valid", "sealed")]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "catalog": str(catalog), "folds": [{"fold": "F4", "train": ["fit"], "validation": ["valid"], "test": ["sealed"]}],
        "heads": ["local_independent"], "seeds": [23], "model": {}, "formal_training_permitted": True,
        "epoch_selection_root": "synthetic", "recovery_diagnostic_board": "valid",
        "training": {"selection_rule": "guarded_joint", "epochs": 40, "min_epochs": 15, "patience": 10,
                     "normal_fpr_limit": .02, "lr_schedule": "cosine", "lr_min": 1e-5, "lr_warmup_epochs": 2},
    }))

    class Dataset:
        config = {}
        refs = np.asarray([[0, 0]])
        entities = ["fit"]

        def __len__(self):
            return 1

        def batches(self, *_args, **_kwargs):
            yield SimpleNamespace(x_raw=torch.zeros(1)), [0]

        def selection_summary(self):
            return {"fit": 1}

    def dataset(_cfg, split):
        assert split in ("train", "validation")
        return Dataset()

    calls = {}

    def evaluate(model, _data, *, phase, epoch, prediction_path=None, **_kwargs):
        calls[phase, epoch] = model.weight.item()
        if prediction_path:
            prediction_path.write_text("{}\n")
        score = 0.7 if epoch in (1, 16) else 0.6
        nll = 0.05 if epoch == 16 else 0.1
        return {"reliable": {"deepest_specimen_macro_f1": score, "positive_joint_map_accuracy": score,
                             "normal_joint_false_positive_rate": .01 if passing else .03, "joint_nll": nll}}

    monkeypatch.setattr(stream_training, "dataset_from_config", dataset)
    monkeypatch.setattr(stream_training, "use_epoch", lambda *_a, **_k: "synthetic")
    monkeypatch.setattr(stream_training, "build_model", lambda _cfg: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(stream_training, "build_optimizer", lambda model, _cfg, **_kw: (torch.optim.SGD(model.parameters(), lr=.001), {}))
    monkeypatch.setattr(stream_training, "configure_model", lambda *_a: None)
    monkeypatch.setattr(stream_training, "speed_context", lambda *_a, **_k: nullcontext())
    attempt = {"attempt_mask_sha256": None, "attempted": 0, "changed": 0, "eligible": 0}
    monkeypatch.setattr(stream_training, "training_conditional", lambda model, _batch, **_kw: (model.weight, attempt))
    monkeypatch.setattr(stream_training, "head_training_loss", lambda out, _batch, **_kw: SimpleNamespace(total=out.sum()))
    monkeypatch.setattr(stream_training, "evaluate_stream", evaluate)
    run = tmp_path / "run"
    result = stream_training.train(config, run, fold_name="F4", head="local_independent", seed=23)
    assert result["epochs"] == 25 and result["test_opened"] is False
    assert calls["validation_fixed_epoch15", 15] == calls["validation", 15]
    assert calls["validation_last_epoch", 25] == calls["validation", 25]
    assert calls["validation", 15] != calls["validation", 25]
    assert result["recovery_checkpoints"]["fixed_epoch15"]["guard_met"] == passing
    if passing:
        assert result["best_epoch"] == 16 and result["progress_anchor_epoch"] == 1
        assert calls["final_validation", 16] == calls["validation", 16]
    else:
        assert result["status"] == "NO_ADMISSIBLE_CHECKPOINT" and result["selected_checkpoint"] is None
        assert not (run / "predictions.jsonl").exists()


def test_weighted_layer_auroc_handles_tied_scores():
    result = layer_metrics([(0., 0, 0, 2), (1., 0, 1, 1), (1., 1, 1, 3)])
    assert result["weighted_auroc"] == pytest.approx(5 / 6)
    assert result["negative"]["reported_positive"] == pytest.approx(1 / 3)
    assert layer_metrics([(0., 1, 0, 2)])["weighted_auroc"] is None
