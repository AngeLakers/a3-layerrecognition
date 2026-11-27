from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from a3_ultrasound.training import ModelConfig, build_model, evaluate, synthetic_datasets
from a3_ultrasound.v171_protocol import FIXED_MODEL
from a3_ultrasound.v186_replay import feature_replay, map_from_delta, policies


@pytest.mark.parametrize("mode", ["score", "feature"])
def test_stage1_exact_path_initialization_gradient_and_export(mode):
    cfg = ModelConfig(**(FIXED_MODEL | dict(architecture_version="a3-v1.8", backbone="resnet1d",
        head_type="local_independent", relative_time_attention=True, readout_scope="full_waveform",
        common_feature_dim=128, token_dim=128)))
    data = synthetic_datasets(23, 4)["train"]
    batch = data.batch
    torch.manual_seed(23)
    original = build_model(cfg)
    torch.manual_seed(23)
    model = build_model(replace(cfg, stage1_donor=mode))
    for name, value in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)
    model.eval()
    original.eval()
    reference = original.infer_exact(batch)
    initial = model.infer_exact(batch)
    torch.testing.assert_close(reference.log_probabilities, initial.log_probabilities, rtol=0, atol=0)
    with torch.no_grad():
        model.head.score_scale.fill_(.3)
        if mode == "feature":
            model.head.projection.fill_(.2)
    encoded = model.encode(batch)
    encoded.local_readout.token.retain_grad()
    output = model.conditional(encoded, batch.y)
    flipped = model.conditional(encoded, torch.zeros_like(batch.y))
    torch.testing.assert_close(output.logits, flipped.logits)
    torch.testing.assert_close(output.logits[:, 0], output.base_logodds[:, 0])
    assert (output.logits[~batch.interface_mask] == 0).all()
    output.logits[:, 1].sum().backward()
    assert encoded.local_readout.token.grad[:, 0].abs().sum() > 0
    assert model.head.score_scale.grad is not None
    if mode == "feature":
        assert model.head.projection.grad.abs().sum() > 0
    _, rows = evaluate(model, data, batch_size=4)
    for row in rows:
        assert map_from_delta(row, [0.] * row["K"]) == row["map_states"]
        assert ("stage1_donor_projection" in row) == (mode == "feature")
    if mode == "feature":
        for row in rows:
            row["sample_weight"] = 1
        feature_replay(rows, {r["sample_id"]: r["base_probabilities"] for r in rows})


def test_c3_reviewed_errors_and_partial_point_fpr():
    normal = dict(sample_id="a", y=[0,0], label_mask=[True,True], base_logodds=[.001,4.],
                  map_states=[1,1], sample_weight=1)
    unknown = dict(sample_id="b", y=[-1,1], label_mask=[False,True], base_logodds=[.001,-3.],
                   map_states=[0,0], sample_weight=1)
    others = [dict(normal, sample_id=str(i), base_logodds=[4.,4.], map_states=[0,0]) for i in range(18)]
    arms = policies([normal, unknown, *others])
    assert all(c["review"] == 2 for c in arms.values())
    assert arms["uncertain"]["review_errors"] == 2
    assert arms["uniform"]["review_errors"] == 1.5
    assert arms["whole"]["fp"] == 0
    assert arms["uniform"]["fp"] == arms["uncertain"]["fp"] == 1
    assert arms["uniform"]["held"] == arms["uncertain"]["held"] == 2
    assert arms["uniform"]["pending"] == .5
    assert arms["uniform"]["review_known"] == 3


def test_permutation_only_changes_paired_feature_and_preserves_singletons():
    rows = []
    for name, p, probability in [("a", 2., .6), ("b", -2., .61), ("c", -2., .31)]:
        z = [1., p]
        scores = [a + b for a in (0.,z[0]) for b in (0.,z[1])]
        norm = math.log(sum(map(math.exp, scores)))
        rows.append(dict(sample_id=name, base_probabilities=[probability,.5], y=[1,1],
                         label_mask=[True,True], sample_weight=3, map_states=[1,int(p>0)],
                         stage1_logits=z, stage1_score_logits=[1.,0.], stage1_donor_projection=[0.,p],
                         log_probabilities=[v-norm for v in scores]))
    backup = deepcopy(rows)
    counts, coverage = feature_replay(rows, {r["sample_id"]: r["base_probabilities"] for r in rows})
    assert rows == backup
    assert counts["detected"] == 3
    assert coverage["positive_changed_weight"] == coverage["positive_swappable_weight"] == 6
    assert coverage["positive_targets_weight"] == 9

def test_identity_rejects_logits_inconsistent_with_lattice():
    row = dict(sample_id="a", y=[1,1], label_mask=[True,True], sample_weight=1,
               stage1_logits=[-2.,-2.], stage1_score_logits=[-2.,-2.],
               stage1_donor_projection=[0.,0.], map_states=[1,1], log_probabilities=[-4.,-2.,-2.,0.])
    with pytest.raises(ValueError, match="distribution"):
        feature_replay([row], {"a":[.12,.12]})
    row.update(stage1_logits=[0.,0.], stage1_score_logits=[0.,0.], map_states=[0,0],
               log_probabilities=[-math.log(4)]*4)
    feature_replay([row], {"a":[.5,.5]})


def test_checkpoint_swaps_fail_and_partial_pairs_survive(tmp_path, monkeypatch):
    import hashlib
    import json
    from a3_ultrasound import v186_replay as replay
    from a3_ultrasound.v186 import PROTOCOL, export_receipts

    jobs, baseline = tmp_path / "jobs", tmp_path / "baseline"
    row = dict(sample_id="a", specimen_id="b", y=[1,1], label_mask=[True,True], sample_weight=1,
               base_probabilities=[.5,.5], stage1_logits=[0.,0.], stage1_score_logits=[0.,0.],
               stage1_donor_projection=[0.,0.], map_states=[0,0], log_probabilities=[-math.log(4)]*4)
    raw = json.dumps(row).encode() + b"\n"
    old = baseline / "F1" / "predictions.jsonl"
    old.parent.mkdir(parents=True)
    old.write_bytes(raw)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    cohort = json.dumps([["a", "b", [1,1], [True,True], 1]], separators=(",", ":")).encode()
    monkeypatch.setattr(replay, "registry", lambda: {"F1":{"cohort":sha(cohort), "predictions":sha(raw)}})
    monkeypatch.setattr(replay, "validate_configuration", lambda *_: None)
    job = jobs / "F1_feature"
    run = job / "run"
    run.mkdir(parents=True)
    cfg = json.dumps({"source_commit":"abc"}).encode()
    (job / "config.json").write_bytes(cfg)
    for key, (pred, weight) in replay.EXPORTS.items():
        (run / pred).write_bytes(raw if key=="best" else raw+b" ")
        (run / weight).write_bytes(key.encode())
    receipt = dict(best_epoch=17, epochs=27, protocol=PROTOCOL, commit_hash="abc", config_sha256=sha(cfg),
                   test_opened=False, status="COMPLETED", selected_checkpoint="best_guarded.pt", guard_violated=False)
    receipt["stage1_exports"] = export_receipts(run, receipt)
    (run/"summary.json").write_text(json.dumps(receipt))
    result = replay.c1(jobs, baseline, "best")
    pairs = result["paired_feature_minus_control"]
    assert result["full_primary_comparison"] is False
    assert pairs["score"]["deep_missed"]["fold_boards"] == []
    assert pairs["permuted"]["deep_missed"]["fold_boards"] == [["F1","b"]]
    (run/"predictions.jsonl").write_bytes(raw.replace(b"0.5",b"0.6"))
    with pytest.raises(ValueError, match="training receipt"):
        replay.c1(jobs, baseline, "best")
