from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from a3_ultrasound.inference.enumerator import enumerate_binary_chain
from a3_ultrasound.models.baselines import LocalPreIndependentStep
from a3_ultrasound.models.capacity import head_baseline_parameter_counts
from a3_ultrasound.models.integrated import HEAD_TYPES
from a3_ultrasound.models.losses import head_training_loss, pcc_training_loss, validate_complete_training_batch
from a3_ultrasound.models.pcc import PCCConditionalStep, validate_prefix_states
from a3_ultrasound.training import (
    ModelConfig, TrainingConfig, TensorDataset, audit_splits, build_model, build_optimizer,
    evaluate, load_checkpoint, load_tensor_manifest, run_training, summarize_predictions,
    synthetic_datasets,
)


@pytest.fixture(autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_all_heads_forward_backward_optimizer_exact_and_label_isolation(head_type):
    torch.manual_seed(7)
    model = build_model(ModelConfig(head_type=head_type, backbone="tcn"))
    data = synthetic_datasets(19, 4)["train"]
    batch = data.batch
    optimizer, counts = build_optimizer(model, TrainingConfig())
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    assert "backbone_adapter" in counts and "pcc" in counts
    output = model.conditional(model.encode(batch), batch.y)
    loss = head_training_loss(output, batch, head_type=head_type)
    if head_type == "local_independent":
        torch.testing.assert_close(loss.total, loss.chain)
    else:
        torch.testing.assert_close(loss.total, loss.chain + 0.3 * loss.local)
    loss.total.backward()
    for part in (model.backbone, model.backbone_adapter, model.local_readout, model.head.base):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    original = model.infer_exact(batch)
    hook.remove()
    assert len(calls) == 1
    changed = replace(batch, y=torch.where(batch.interface_mask, 1 - batch.y, batch.y),
                      label_mask=torch.zeros_like(batch.interface_mask))
    blinded = model.infer_exact(changed)
    torch.testing.assert_close(original.probabilities, blinded.probabilities)
    torch.testing.assert_close(original.raw_log_normalizer, torch.zeros(4), atol=2e-6, rtol=0)
    assert original.path_mask.sum(1).tolist() == [4, 8, 4, 8]
    if "independent" in head_type:
        assert output.gate is None and output.correction is None and output.prefix_hidden is None
        assert not hasattr(model.head, "prefix_encoder")
        torch.testing.assert_close(original.conditional_probability_range, torch.zeros_like(original.marginals))


def test_local_pre_reads_context_but_not_query_gate_or_prefix():
    torch.manual_seed(3)
    head = LocalPreIndependentStep(local_dim=3, context_dim=4)
    loc = torch.randn(1, 3, 3, requires_grad=True)
    ctx = torch.randn(1, 3, 4, requires_grad=True)
    gate = torch.randn(1, 3, 5, requires_grad=True)
    query = torch.randn(1, 3, 2, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    output = head(loc, ctx, gate, query, torch.zeros(1, 3, dtype=torch.long), mask)
    gradients = torch.autograd.grad(output.logits.sum(), (loc, ctx, gate, query), allow_unused=True)
    assert gradients[0].abs().sum() > 0 and gradients[1].abs().sum() > 0
    assert gradients[2] is None and gradients[3] is None
    other = head(loc, ctx, gate + 100, query + 100, torch.ones(1, 3, dtype=torch.long), mask)
    torch.testing.assert_close(output.logits, other.logits)
    counts = head_baseline_parameter_counts(local_dim=3, context_dim=4, query_dim=2, prefix_dim=32, hidden_dim=64)
    assert counts.local_pre_independent_total == sum(p.numel() for p in head.parameters())


@pytest.mark.parametrize("states", [torch.tensor([[0, 2, -1]]), torch.tensor([[0, -1, -1]]),
                                  torch.tensor([[0.0, 0.5, 0.0]]), torch.tensor([[0.0, float("nan"), 0.0]])])
def test_invalid_prefix_is_not_clamped_or_truncated(states):
    mask = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="binary"):
        validate_prefix_states(states, mask)
    head = PCCConditionalStep(local_dim=1, context_dim=1, gate_dim=1, query_dim=1,
                              prefix_dim=1, hidden_dim=1)
    x = torch.ones(1, 3, 1)
    with pytest.raises(ValueError, match="binary"):
        head(x, x, x, x, states, mask)


def test_padding_is_safely_replaced_without_altering_real_zero():
    states = torch.tensor([[0, 1, -99]])
    assert validate_prefix_states(states, torch.tensor([[True, True, False]])).tolist() == [[0, 1, 0]]


def _dependent_step():
    step = PCCConditionalStep(local_dim=1, context_dim=1, gate_dim=1, query_dim=1,
                              prefix_dim=1, hidden_dim=1, rho=2, local_base_type="linear")
    with torch.no_grad():
        for p in step.parameters():
            p.zero_()
        step.base.network.bias.fill_(0.3)
        step.prefix_encoder.state_embedding.weight[:, 0] = torch.tensor([-1.0, 1.0])
        # GRU new-state gate consumes the state embedding (after the Query field).
        step.prefix_encoder.gru_cell.weight_ih[2, 1] = 1.0
        step.correction[0].weight[0, 1] = 2.0
        step.correction[-1].weight.fill_(1.0)
    return step


@pytest.mark.parametrize("k", [2, 3])
def test_nonzero_dependent_chain_raw_probabilities_marginals_map_and_training_nll(k):
    step = _dependent_step()
    x = torch.zeros(1, k, 1)
    mask = torch.ones(1, k, dtype=torch.bool)
    zero = torch.zeros(1, k, dtype=torch.long)
    one = zero.clone()
    one[:, :-1] = 1
    a, b = step(x, x, x, x, zero, mask), step(x, x, x, x, one, mask)
    assert (a.logits[:, 1:] - b.logits[:, 1:]).abs().min() > 0.1
    assert a.correction[:, 1:].abs().max() > 0
    torch.testing.assert_close(a.gate, b.gate)
    exact = enumerate_binary_chain(step, x, x, x, x, mask)
    paths = exact.path_states[0]
    raw = []
    for path in paths:
        output = step(x, x, x, x, path.unsqueeze(0), mask)
        # Probability product independent of enumerator's log-space implementation.
        probability = output.logits.sigmoid()[0]
        raw.append(torch.where(path.bool(), probability, 1 - probability).prod())
        loss = pcc_training_loss(output.logits, output.base_logodds, path.unsqueeze(0), mask, lambda_local=0)
        torch.testing.assert_close(k * loss.chain, -raw[-1].log(), atol=1e-6, rtol=1e-6)
    raw = torch.stack(raw)
    torch.testing.assert_close(raw.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(exact.raw_log_normalizer, torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(exact.raw_log_probabilities[0], raw.log())
    torch.testing.assert_close(exact.probabilities[0], raw)
    torch.testing.assert_close(exact.marginals[0], (raw[:, None] * paths).sum(0))
    assert exact.map_states[0].tolist() == paths[raw.argmax()].tolist()


@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_training_rejects_partial_labels_but_inference_accepts_them(head_type):
    model = build_model(ModelConfig(head_type=head_type, backbone="tcn"))
    batch = synthetic_datasets(7, 4)["train"].batch
    labels = batch.label_mask.clone()
    labels[0, 0] = False
    partial = replace(batch, label_mask=labels)
    with pytest.raises(ValueError, match="complete"):
        validate_complete_training_batch(partial)
    assert torch.isfinite(model.infer_exact(partial).marginals).all()
    assert torch.isfinite(model.infer_exact(replace(batch, y=None, label_mask=None)).marginals).all()


def test_explicit_ablation_variants_and_empty_pre_have_finite_gradients():
    batch = synthetic_datasets(8, 4)["train"].batch
    meta = batch.interface_meta.clone()
    meta[:, 0, 1] = meta[:, 0, 3]  # first Local window begins at sample zero: empty Pre
    batch = replace(batch, interface_meta=meta)
    model = build_model(ModelConfig(backbone="tcn", pre_query_conditioned=True))
    output = model(batch, batch.y)
    assert not output.pre_support[:, 0].any()
    loss = head_training_loss(output.conditional, batch, head_type="proposed", lambda_local=0,
                              disable_local_auxiliary=True)
    loss.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="explicit ablation"):
        head_training_loss(output.conditional, batch, head_type="proposed", lambda_local=0)
    full = build_model(ModelConfig(backbone="tcn", head_type="local_independent", readout_scope="full_waveform"))
    encoded = full.encode(batch)
    expected = encoded.backbone.feature_mask[:, None, :] & batch.interface_mask[:, :, None]
    assert torch.equal(encoded.local_support, expected)
    assert full.pre_readout.query_projection is None
    # No Gaussian weighting is passed in this variant.
    direct = full.local_readout(encoded.backbone.features, encoded.query, expected, batch.interface_mask)
    torch.testing.assert_close(encoded.local_readout.attention, direct.attention)
    with pytest.raises(ValueError, match="restricted"):
        build_model(ModelConfig(backbone="tcn", readout_scope="full_waveform"))


@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_common_training_checkpoint_and_exact_evaluation_roundtrip(tmp_path, head_type):
    data = synthetic_datasets(31, 4)
    config = TrainingConfig(epochs=1, batch_size=3, seed=31)
    output_dir = tmp_path / head_type
    summary = run_training(ModelConfig(backbone="tcn", head_type=head_type), config, data, output_dir,
                           data_kind="synthetic", input_manifest={"test_fixture": True})
    assert summary["status"] == "completed" and summary["performance_claims_released"] is False
    model, checkpoint = load_checkpoint(output_dir / "best.pt")
    metrics, rows = evaluate(model, data["test"], batch_size=3)
    assert metrics == summary["metrics"]["test"]
    saved = [json.loads(line) for line in (output_dir / "test_predictions.jsonl").read_text().splitlines()]
    assert saved == rows
    assert checkpoint["resolved_config"]["model"]["head_type"] == head_type
    assert checkpoint["resolved_config"]["effective_lambda_local"] == (0 if head_type == "local_independent" else 0.3)
    assert checkpoint["resolved_config"]["model"]["backbone_stage"] == "final_pre_pool"
    assert "conditional_probability_range" in rows[0]
    assert metrics["joint_complete_label_count"] == 4
    # Existing artifacts cannot be silently overwritten.
    with pytest.raises(FileExistsError):
        run_training(ModelConfig(backbone="tcn", head_type=head_type), config, data, output_dir,
                     data_kind="synthetic", input_manifest={})


def test_split_leakage_and_real_gate_fail_before_training(tmp_path):
    data = synthetic_datasets(1, 4)
    with pytest.raises(ValueError, match="GATED"):
        run_training(ModelConfig(), TrainingConfig(), data, tmp_path / "real", data_kind="real", input_manifest={})
    assert not (tmp_path / "real").exists()
    leaked = dict(data)
    leaked["validation"] = replace(data["validation"], specimen_ids=data["train"].specimen_ids)
    with pytest.raises(ValueError, match="specimen leakage"):
        audit_splits(leaked)


def test_npz_manifest_load_and_unlabeled_evaluation(tmp_path):
    data = synthetic_datasets(11, 4)
    for name, dataset in data.items():
        np.savez(tmp_path / f"{name}.npz", **{key: value.numpy() for key, value in vars(dataset.batch).items()},
                 sample_ids=np.array(dataset.sample_ids), specimen_ids=np.array(dataset.specimen_ids))
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("schema_version: a3-tensor-dataset-v1\ndata_kind: synthetic\npreprocessing_id: toy-v1\nsplit_id: toy-v1\nsplits:\n  train: train.npz\n  validation: validation.npz\n  test: test.npz\n")
    loaded, _ = load_tensor_manifest(manifest)
    torch.testing.assert_close(loaded["train"].batch.x_raw, data["train"].batch.x_raw)
    unlabeled = replace(loaded["test"], batch=replace(loaded["test"].batch, y=None, label_mask=None))
    metrics, rows = evaluate(build_model(ModelConfig(backbone="tcn")), unlabeled)
    assert metrics["joint_nll"] is None and metrics["labeled_interface_count"] == 0
    assert all(row["y"] is None for row in rows)


def test_runner_allows_partial_validation_and_unlabeled_test(tmp_path):
    data = synthetic_datasets(17, 4)
    val = data["validation"].batch
    reliable = val.label_mask.clone()
    reliable[:, 0] = False
    data["validation"] = replace(data["validation"], batch=replace(val, label_mask=reliable))
    data["test"] = replace(data["test"], batch=replace(data["test"].batch, y=None, label_mask=None))
    result = run_training(ModelConfig(backbone="tcn"), TrainingConfig(epochs=1, batch_size=4), data,
                          tmp_path / "unlabeled-test", data_kind="synthetic", input_manifest={})
    assert result["metrics"]["validation"]["deepest_specimen_macro_f1"] is not None
    assert result["metrics"]["validation"]["joint_nll"] is None
    assert result["metrics"]["test"]["labeled_interface_count"] == 0
    rows = (tmp_path / "unlabeled-test" / "test_predictions.jsonl").read_text().splitlines()
    assert len(rows) == 4 and all(json.loads(row)["y"] is None for row in rows)
