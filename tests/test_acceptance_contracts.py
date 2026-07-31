from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from a3_ultrasound.data import (
    require_train_only_statistics_source,
    validate_specimen_split_disjointness,
)
from a3_ultrasound.metrics import specimen_macro_f1_present_classes
from a3_ultrasound.models import (
    PCCConditionalStep,
    ResponsibilityTokenProjector,
    pcc_head_parameter_count,
    select_matched_joint_hidden_width,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATCHED_JOINT_SPEC = REPOSITORY_ROOT / "configs" / "model" / "matched_joint_spec.yaml"


def test_single_class_specimen_macro_f1_uses_present_classes_and_zero_division_zero() -> None:
    missing_prediction = specimen_macro_f1_present_classes(
        [1, 1, 1],
        [0, 0, 0],
    )
    assert missing_prediction.present_classes == (1,)
    assert missing_prediction.single_class_specimen
    assert missing_prediction.zero_division == 0
    assert missing_prediction.per_class_f1 == {1: 0.0}
    assert missing_prediction.macro_f1 == 0.0

    mixed = specimen_macro_f1_present_classes(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    )
    assert mixed.present_classes == (0, 1)
    assert not mixed.single_class_specimen
    assert mixed.macro_f1 == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_specimen_split_fixture_is_disjoint_and_statistics_are_train_only() -> None:
    audit = validate_specimen_split_disjointness(
        {
            "train": ["S01", "S02"],
            "validation": ["S03"],
            "calibration": ["S04"],
            "test": ["S05"],
        }
    )
    assert audit.counts == {
        "train": 2,
        "validation": 1,
        "calibration": 1,
        "test": 1,
    }
    assert len(set(audit.all_specimens)) == 5
    require_train_only_statistics_source("train")
    for forbidden_role in ("validation", "calibration", "test"):
        with pytest.raises(ValueError, match="Train only"):
            require_train_only_statistics_source(forbidden_role)

    with pytest.raises(ValueError, match="disjoint"):
        validate_specimen_split_disjointness(
            {
                "train": ["S01", "S02"],
                "validation": ["S03"],
                "calibration": ["S04"],
                "test": ["S01"],
            }
        )
    with pytest.raises(ValueError, match="outer whitespace"):
        validate_specimen_split_disjointness(
            {
                "train": ["S01 "],
                "validation": ["S03"],
                "calibration": ["S04"],
                "test": ["S05"],
            }
        )


def test_analytic_pcc_count_matches_implemented_modules() -> None:
    projector = ResponsibilityTokenProjector(
        local_dim=32,
        context_dim=32,
        query_dim=32,
        token_dim=32,
    )
    pcc = PCCConditionalStep(
        local_dim=32,
        context_dim=32,
        gate_dim=32,
        query_dim=32,
        prefix_dim=32,
        hidden_dim=64,
        max_interfaces=3,
    )
    actual = sum(parameter.numel() for parameter in projector.parameters()) + sum(
        parameter.numel() for parameter in pcc.parameters()
    )
    analytic = pcc_head_parameter_count(
        feature_dim=32,
        query_dim=32,
        token_dim=32,
        prefix_dim=32,
        hidden_dim=64,
        max_interfaces=3,
    )
    assert actual == analytic.total == 28_579


def test_matched_joint_width_is_label_free_and_within_ten_percent() -> None:
    match = select_matched_joint_hidden_width(
        feature_dim=32,
        query_dim=32,
        token_dim=32,
        prefix_dim=32,
        pcc_hidden_dim=64,
        fusion_dim=32,
        minimum=1,
        maximum=512,
        tolerance_relative=0.10,
        max_interfaces=3,
        supported_k=(2, 3),
    )
    assert match.joint_hidden_dim == 103
    assert match.pcc.total == 28_579
    assert match.matched_joint.total == 28_527
    assert match.absolute_difference == 52
    assert match.relative_difference == pytest.approx(52 / 28_579)
    assert match.within_tolerance
    with pytest.raises(ValueError, match="finite and nonnegative"):
        select_matched_joint_hidden_width(
            feature_dim=32,
            query_dim=32,
            token_dim=32,
            prefix_dim=32,
            pcc_hidden_dim=64,
            fusion_dim=32,
            tolerance_relative=float("nan"),
        )


def test_matched_joint_spec_freezes_fairness_and_seed_independence() -> None:
    spec = yaml.safe_load(MATCHED_JOINT_SPEC.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "a3-matched-joint-spec-v1"
    assert spec["state_space"]["supported_K"] == [2, 3]
    assert spec["decoder"]["interface_fusion"]["input_order"] == [
        "r_loc",
        "r_ctx",
        "r_gate",
    ]
    assert spec["decoder"]["joint_branches"]["K2"]["output_logits"] == 4
    assert spec["decoder"]["joint_branches"]["K3"]["output_logits"] == 8
    search = spec["capacity_match"]["joint_hidden_width_search"]
    assert search["reads_labels_or_validation_metrics"] is False
    initialization = spec["initialization_and_optimization"]
    assert initialization["shared_parameter_values_equal_at_initialization"] is True
    assert initialization["parameter_objects_independent"] is True
    assert initialization["post_initialization_weight_sharing"] == "forbidden"

    torch.manual_seed(211)
    pcc_shared = ResponsibilityTokenProjector(
        local_dim=4,
        context_dim=4,
        query_dim=3,
        token_dim=5,
    )
    torch.manual_seed(211)
    joint_shared = ResponsibilityTokenProjector(
        local_dim=4,
        context_dim=4,
        query_dim=3,
        token_dim=5,
    )
    joint_initial = {
        name: value.detach().clone() for name, value in joint_shared.state_dict().items()
    }
    for (pcc_name, pcc_value), (joint_name, joint_value) in zip(
        pcc_shared.state_dict().items(),
        joint_shared.state_dict().items(),
        strict=True,
    ):
        assert pcc_name == joint_name
        torch.testing.assert_close(pcc_value, joint_value)
    assert all(
        left is not right
        for left, right in zip(
            pcc_shared.parameters(),
            joint_shared.parameters(),
            strict=True,
        )
    )

    optimizer = torch.optim.SGD(pcc_shared.parameters(), lr=0.1)
    optimizer.zero_grad()
    pcc_shared.local_evidence_projection.weight.sum().backward()
    optimizer.step()
    for name, value in joint_shared.state_dict().items():
        torch.testing.assert_close(value, joint_initial[name])
    assert not torch.equal(
        pcc_shared.local_evidence_projection.weight,
        joint_shared.local_evidence_projection.weight,
    )
