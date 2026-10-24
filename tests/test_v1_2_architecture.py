from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml

from a3_ultrasound.inference import enumerate_binary_chain
from a3_ultrasound.models import (
    IndependentConditionalStep,
    IndexedReadout,
    VanillaClassifierChainStep,
    head_baseline_parameter_counts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _representations(k: int = 6):
    torch.manual_seed(71)
    return (
        torch.randn(1, k, 4),
        torch.randn(1, k, 5),
        torch.randn(1, k, 6),
        torch.randn(1, k, 3),
        torch.ones(1, k, dtype=torch.bool),
    )


def test_fresh_process_can_import_enumerator_without_models_import_order() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from a3_ultrasound.inference.enumerator import ExactInferenceOutput",
        ],
        # A fresh interpreter does not inherit pytest's configured src path.
        # Import the checkout directly without requiring an editable install.
        cwd=REPOSITORY_ROOT / "src",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_local_readout_has_no_dead_empty_context_but_pre_readout_does() -> None:
    local = IndexedReadout(feature_dim=8, query_dim=4, attention_dim=6)
    pre = IndexedReadout(feature_dim=8, query_dim=None, attention_dim=6)
    assert local.empty_context is None
    assert pre.empty_context is not None


def test_independent_and_vanilla_chain_baselines_are_k_independent() -> None:
    r_loc, r_ctx, r_gate, query, mask = _representations()
    states_zero = torch.zeros((1, 6), dtype=torch.long)
    states_one = torch.ones((1, 6), dtype=torch.long)

    independent = IndependentConditionalStep(local_dim=4)
    independent_zero = independent(
        r_loc, r_ctx, r_gate, query, states_zero, mask
    ).logits
    independent_one = independent(
        r_loc, r_ctx, r_gate, query, states_one, mask
    ).logits
    torch.testing.assert_close(independent_zero, independent_one)

    vanilla = VanillaClassifierChainStep(
        local_dim=4,
        context_dim=5,
        query_dim=3,
        prefix_dim=7,
        hidden_dim=9,
    )
    vanilla_zero_output = vanilla(
        r_loc, r_ctx, r_gate, query, states_zero, mask
    )
    vanilla_one_output = vanilla(
        r_loc, r_ctx, r_gate, query, states_one, mask
    )
    vanilla_zero = vanilla_zero_output.logits
    vanilla_one = vanilla_one_output.logits
    torch.testing.assert_close(vanilla_zero[:, 0], vanilla_one[:, 0])
    assert not torch.allclose(vanilla_zero[:, 1:], vanilla_one[:, 1:])
    assert vanilla_zero_output.base_logodds.shape == vanilla_zero.shape

    exact = enumerate_binary_chain(
        vanilla,
        r_loc,
        r_ctx,
        r_gate,
        query,
        mask,
    )
    assert exact.path_mask.sum().item() == 64


def test_vanilla_has_no_direct_query_or_gate_shortcut_and_exposes_local_base() -> None:
    r_loc, r_ctx, r_gate, query, mask = _representations(k=3)
    r_gate = r_gate.requires_grad_()
    query = query.requires_grad_()
    vanilla = VanillaClassifierChainStep(
        local_dim=4,
        context_dim=5,
        query_dim=3,
        prefix_dim=7,
        hidden_dim=9,
    )
    output = vanilla(
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.tensor([[0, 1, 0]], dtype=torch.long),
        mask,
    )

    first_query_gradient = torch.autograd.grad(
        output.logits[:, 0].sum(),
        query,
        retain_graph=True,
        allow_unused=True,
    )[0]
    gate_gradient = torch.autograd.grad(
        output.logits.sum(),
        r_gate,
        allow_unused=True,
    )[0]
    assert vanilla.classifier[0].in_features == 4 + 5 + 7
    assert first_query_gradient is not None
    assert torch.count_nonzero(first_query_gradient) == 0
    assert gate_gradient is None
    assert output.base_logodds.shape == output.logits.shape


def test_v1_2_baseline_capacity_table_matches_implemented_heads() -> None:
    capacities = head_baseline_parameter_counts(
        local_dim=32,
        context_dim=32,
        query_dim=32,
        prefix_dim=32,
        hidden_dim=64,
        local_base_hidden_dim=32,
    )
    independent = IndependentConditionalStep(local_dim=32, hidden_dim=32)
    vanilla = VanillaClassifierChainStep(
        local_dim=32,
        context_dim=32,
        query_dim=32,
        prefix_dim=32,
        hidden_dim=64,
        local_base_hidden_dim=32,
    )
    assert capacities.local_base == 1_089
    assert capacities.shared_prefix == 9_472
    assert capacities.vanilla_classifier == 6_273
    assert capacities.correction == 4_225
    assert capacities.gate == 2_177
    assert sum(parameter.numel() for parameter in independent.parameters()) == capacities.independent_total
    assert sum(parameter.numel() for parameter in vanilla.parameters()) == capacities.vanilla_total
    assert capacities.propagation_conditioned_total == 16_963


def test_v1_2_model_schema_records_user_approved_boundaries() -> None:
    schema = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "model" / "schema.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert schema["architecture_version"] == "a3-v1.3"
    assert schema["backbone"]["smoke_only"] == ["tcn"]
    assert schema["backbone"]["formal_training_candidates"] == [
        "resnet1d",
        "inceptiontime",
        "fcn1d",
    ]
    assert schema["structured_head"]["engineering_K_cap"] == 6
    assert schema["structured_head"]["trainable_parameters_depend_on_K"] is False
    assert schema["loss"]["equal_weight_for_each_supervised_sample"] is True
    assert schema["loss"]["local_auxiliary_loss"] == {
        "independent": "primary_local_base_only",
        "local_pre_independent": "required",
        "vanilla_classifier_chain": "required",
        "propagation_conditioned": "required",
        "lambda_validation_candidates": [0.1, 0.3, 1.0],
        "lambda_shared_for_vanilla_and_proposed": True,
        "zero_lambda": "explicit_disable_local_auxiliary_ablation_only",
    }
    assert schema["head_baselines"]["vanilla_classifier_chain"] == {
        "conditional_inputs": ["r_loc", "r_ctx", "prefix_hidden"],
        "direct_query_to_classifier": "forbidden",
        "gate_token_input": "forbidden",
    }
    assert schema["reference_dimensions"]["local_base"] == [32, 32, 1]
    assert schema["head_baselines"]["optional_small_K"] == ["flat_joint"]
