from dataclasses import replace

import pytest
import torch

from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.prefix_perturbation import training_conditional
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets
from a3_ultrasound.v171_protocol import FIXED_MODEL


def test_state_write_is_causal_waveform_dependent_and_exact():
    torch.manual_seed(23)
    cfg = ModelConfig(**(FIXED_MODEL | {"architecture_version": "a3-v1.8"}),
                      head_type="state_conditioned_gru", backbone="resnet1d")
    model = build_model(cfg).eval()
    batch = synthetic_datasets(23, 4)["train"].batch
    encoded = model.encode(batch)
    assert model.pre_readout is None and model.responsibility_projector is None
    assert encoded.responsibility_tokens.r_loc is encoded.local_readout.token
    first = model.conditional(encoded, batch.y)
    changed = batch.y.clone()
    changed[:, 1] = 1 - changed[:, 1]
    second = model.conditional(encoded, changed)
    torch.testing.assert_close(first.logits[:, :2], second.logits[:, :2])
    assert (first.prefix_hidden[:, 2] != second.prefix_hidden[:, 2]).any()
    erased = encoded.local_readout.token.clone()
    erased[:, 1] = 0
    erased = replace(encoded, responsibility_tokens=replace(encoded.responsibility_tokens, r_loc=erased))
    torch.testing.assert_close(model.conditional(erased, batch.y).logits, model.conditional(erased, changed).logits)

    exact = model.infer_encoded(encoded)
    assert exact.path_mask.sum(1).tolist() == [4, 8, 4, 8]
    torch.testing.assert_close(exact.raw_log_normalizer, torch.zeros_like(exact.raw_log_normalizer))
    without_labels = model.infer_exact(replace(batch, y=None, label_mask=None))
    torch.testing.assert_close(exact.raw_log_probabilities, without_labels.raw_log_probabilities)
    loss = head_training_loss(first, batch, head_type=cfg.head_type, local_aux_mode="clean_upstream")
    path_index = ((exact.path_states == batch.y[:, None]).all(-1) & exact.path_mask).long().argmax(1)
    log_p = exact.raw_log_probabilities.gather(1, path_index[:, None]).squeeze(1)
    torch.testing.assert_close(loss.chain, (-log_p / batch.interface_mask.sum(1)).mean())
    loss.total.backward()
    for projection in (model.head.write_zero, model.head.write_one):
        assert projection.bias is None and projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="true prefixes"):
        training_conditional(model, batch, probability=.2)


def test_synthetic_config_trains_and_reloads_identical_predictions(tmp_path):
    from pathlib import Path

    from a3_ultrasound.training import run_from_config

    config = Path(__file__).resolve().parents[1] / "configs/experiment/v1_8_gru_synthetic.yaml"
    trained, restored = tmp_path / "train", tmp_path / "reload"
    summary = run_from_config(config, trained)
    assert summary["data_kind"] == "synthetic" and summary["epochs_run"] == 2
    run_from_config(config, restored, checkpoint_path=trained / summary["selected_checkpoint"],
                    evaluate_split="validation")
    assert (restored / "validation_predictions.jsonl").read_bytes() == (trained / "validation_predictions.jsonl").read_bytes()


@pytest.mark.parametrize("head", ["local_pre_independent", "state_conditioned_gru", "waveform_only_gru"])
def test_patch_geometry_and_memory_ablation(head):
    cfg = ModelConfig(**(FIXED_MODEL | {"architecture_version": "a3-v1.8"}),
                      head_type=head, backbone="patchtst")
    model = build_model(cfg).eval()
    batch = synthetic_datasets(23, 4)["train"].batch
    encoded = model.encode(batch)
    centers = encoded.backbone.feature_centers
    torch.testing.assert_close(centers[0, :3], torch.tensor([7.5, 11.5, 15.5]))
    assert encoded.backbone.feature_mask.sum(1).tolist() == [11, 13, 11, 13]
    dirty = replace(batch, x_raw=batch.x_raw.masked_fill(~batch.length_mask, 1000))
    torch.testing.assert_close(model.encode(dirty).backbone.features, encoded.backbone.features)
    exact = model.infer_encoded(encoded)
    torch.testing.assert_close(exact.raw_log_normalizer, torch.zeros_like(exact.raw_log_normalizer))
    if head == "waveform_only_gru":
        assert model.head.write_one is None
        other = torch.where(batch.interface_mask, 1 - batch.y, batch.y)
        torch.testing.assert_close(model.conditional(encoded, batch.y).logits, model.conditional(encoded, other).logits)
    out = model.conditional(encoded, batch.y)
    head_training_loss(out, batch, head_type=head, local_aux_mode="clean_upstream").total.backward()
    assert model.backbone.encoder.W_P.weight.grad.abs().sum() > 0
