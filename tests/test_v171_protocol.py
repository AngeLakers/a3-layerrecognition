import json

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from a3_ultrasound.data.v171_cohort import file_sha, make_pair
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets
from a3_ultrasound.v171_b3 import (
    Panel, _validate, calibration_metrics, compare_frozen, fit_threshold, read_full_curves, score, verify_source,
)
from a3_ultrasound.v171_b4 import GROUPS, paired, raw_board
from a3_ultrasound.v171_protocol import LP, FOLDS, PROTOCOL, validate_config, matrix


def config_model(head, backbone="tcn", scope="indexed"):
    return ModelConfig(architecture_version="a3-v1.6", backbone=backbone,
                       backbone_norm="group8", query_view="position2", head_type=head,
                       physical_features=False, readout_scope=scope, rho=3)


def make_panel(logits, truth, *, technical=None):
    logits = np.asarray(logits)
    width = logits.shape[1]
    states = (np.arange(2**width)[:, None] >> np.arange(width-1, -1, -1)) & 1
    log = logits @ states.T
    log -= logsumexp(log, axis=1, keepdims=True)
    raw = log.argmax(1)
    padded = np.full((len(log), 8), np.nan)
    padded[:, :log.shape[1]] = log
    bad = np.zeros(len(log), bool) if technical is None else np.asarray(technical)
    padded[bad] = np.nan
    raw[bad] = -1
    y = np.zeros((len(log), 3), dtype=np.int64)
    y[:, :width] = states[truth]
    known = np.broadcast_to(np.arange(3) < width, y.shape)
    return _validate(Panel(np.repeat("Board", len(log)), np.repeat(width, len(log)), y, known,
                           np.ones(len(log), bool), bad, padded, raw))


def test_matrix_and_phase_authorization_precede_opening_data(tmp_path):
    rows = matrix()
    assert len({row.run_id for row in rows}) == 107
    assert {b: sum(row.block == b for row in rows) for b in ("B1", "B2", "B4")} == {"B1": 60, "B2": 15, "B4": 32}
    subset = [cell for cell in rows if cell.block == "B4"]
    assert {r.fold for r in subset} == set(FOLDS[:4])
    assert {r.seed for r in subset} == {23, 41}
    assert {r.arm for r in subset} == {"B", "C"}
    cell = rows[0]
    path = tmp_path / "conf.json"
    cfg = dict(protocol=PROTOCOL, cell=vars(cell), catalog="intentionally_missing")
    path.write_text(json.dumps(cfg))
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps(dict(protocol=PROTOCOL, phase="train", authorized=True,
                                    cells={cell.run_id: file_sha(path)})))
    with pytest.raises(PermissionError, match="exact outer"):
        validate_config(cfg, fold_name=cell.fold, seed=cell.seed, head=cell.head, phase="outer",
                        authorization=auth, config_path=path)


def test_b4_drop_is_equal_but_preserves_target_in_matched_control():
    y = np.zeros((20, 3), dtype=np.int8)
    y[:5] = [[1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]]
    y[19] = [1, 0, 1]
    mask = np.ones_like(y, dtype=bool)
    mask[19, 0] = False
    index = dict(y=y, eligible=mask.all(1), label_mask=mask, xy=np.column_stack((np.arange(20), np.arange(20))),
                 source_row=np.arange(20) * 2 + 7)
    board = dict(entity="example", K=3, ordinal=2)
    b, c, profile = make_pair(board, index, "101", 23)
    assert b == [[0, 7], [1, 9]]
    assert len(c) == len(b) == 2
    assert set(i for i, _ in c) <= {2, 3, 4}
    assert profile["B"]["states"].get("101", 0) == 0
    assert profile["C"]["states"]["101"] == 2
    assert profile["epoch_positives"] == 3 and profile["epoch_normals"] == 9


def test_full_waveform_mask_reports_actual_tokens_and_pre_has_no_output_gradient():
    data = synthetic_datasets(23, 3)["train"].batch
    model = build_model(config_model("local_independent", scope="full_waveform"), v171=True)
    encoded = model.encode(data)
    counts = encoded.local_token_diagnostics.counts
    expected = data.length_mask.sum(1)[:, None].expand_as(counts) * data.interface_mask
    assert torch.equal(counts, expected)
    conditional = model.conditional(encoded, data.y)
    conditional.logits.sum().backward()
    for name, p in model.named_parameters():
        if name.startswith(("pre_", "readout.pre")):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_b3_exact_log_map_order_and_common_curve_ties(tmp_path):
    panel = make_panel([[4., 4.], [5., 5.], [4., 4.]], [3, 3, 1])
    scores = score(panel, .1)
    assert scores[0] < scores[1] < 0
    assert scores[0] == scores[2]
    assert fit_threshold(panel, scores)["log_threshold"] == scores[1]
    path = tmp_path / "curves.npz"
    read_full_curves(panel, scores, path)
    with np.load(path, allow_pickle=False) as data:
        assert data["counts"][0, 0].tolist() == [0, 1, 3, 3]
        assert data["counts"][0, 1].tolist() == [0, 0, 1, 1]


def test_b3_rejects_wrong_archived_seed_and_unapproved_outer(tmp_path):
    cell = tmp_path / "F1_s23_local_pre_independent_none"
    cell.mkdir()
    path = cell / "raw.npz"
    np.savez(path, head="LP", fold="F1", seed=41, epoch=4)
    identity = dict(fold="F1", head=LP, seed=23, run_id=cell.name)
    with pytest.raises(ValueError, match="different model cell"):
        verify_source(path, phase="validation", **identity)
    with pytest.raises(PermissionError, match="requires authorization"):
        verify_source(path, phase="outer", **identity)


def test_b3_cross_backbone_results_cannot_enter_resnet_comparison(tmp_path):
    path = tmp_path / "b1_tcn.json"
    path.write_text(json.dumps(dict(fold="F1", seed=23, head=LP, run_id="v171_tcn_LP_F1_s23",
                                    outer={"specimens": []}, catalog_sha256="same")))
    with pytest.raises(ValueError, match="compatible ResNet"):
        compare_frozen([path])


def test_b3_reliability_uses_full_legal_state_space():
    panel = make_panel([[0., 0.], [0., 0.]], [0, 1])
    k2, k3 = calibration_metrics(panel, 1)
    assert k2["joint_nll"] == pytest.approx(np.log(4))
    assert k2["joint_brier"] == pytest.approx(.75)
    assert k2["map_ece"] == pytest.approx(.25)
    assert k3["records"] == 0


def test_b4_raw_destinations_conserve_and_common_seed_matching():
    panel = make_panel([[3., -3., -3.], [2., -2., 2.], [-3., -3., -3.]], [5, 5, 0])
    observed = raw_board(panel, "Board", 5)
    assert observed["endpoints"]["target_exact"] == [1, 2]
    assert observed["target_map_destinations"]["100"] == observed["target_map_destinations"]["101"] == 1
    summary = paired({("Board", 23): dict.fromkeys(GROUPS, observed),
                      ("Board", 41): {GROUPS[0]: observed}})
    target = summary["target_exact"]["difference_in_differences"]
    assert target["means"] == dict.fromkeys(("LP_B", "G1D_B", "LP_C", "G1D_C"), .5)
    assert target["missing"] == [{"board": "Board", "seed": 41}]
    lp_b, g1d_b = GROUPS[:2]
    b_only = paired({("Board", 23): {lp_b: observed, g1d_b: observed}})["target_exact"]
    assert b_only["LP_B_minus_G1D_B"]["estimate"] == 0
    assert b_only["difference_in_differences"]["estimate"] is None
