from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from a3_ultrasound.inference.enumerator import enumerate_pcc, lexicographic_binary_paths
from a3_ultrasound.inference.orders import physical_and_control_orders
from a3_ultrasound.models.pcc import PCCConditionalStep


def _step() -> PCCConditionalStep:
    torch.manual_seed(3)
    return PCCConditionalStep(
        local_dim=3,
        context_dim=4,
        gate_dim=5,
        query_dim=2,
        prefix_dim=3,
        hidden_dim=6,
    )


def _inputs(batch_size: int = 1, kmax: int = 3):
    torch.manual_seed(4)
    return (
        torch.randn((batch_size, kmax, 3), requires_grad=True),
        torch.randn((batch_size, kmax, 4), requires_grad=True),
        torch.randn((batch_size, kmax, 5), requires_grad=True),
        torch.randn((batch_size, kmax, 2), requires_grad=True),
    )


@pytest.mark.parametrize("rho", [0.0, -1.0, float("inf"), float("nan")])
def test_pcc_rejects_nonpositive_or_nonfinite_correction_bound(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        PCCConditionalStep(
            local_dim=3,
            context_dim=4,
            gate_dim=5,
            query_dim=2,
            prefix_dim=3,
            hidden_dim=6,
            rho=rho,
        )


def test_first_interface_is_exactly_local_base_with_nonzero_correction_head() -> None:
    step = _step()
    final_correction = step.correction[-1]
    assert isinstance(final_correction, torch.nn.Linear)
    with torch.no_grad():
        final_correction.weight.normal_(mean=0.0, std=0.2)
        final_correction.bias.fill_(0.4)
    r_loc, r_ctx, r_gate, query = _inputs()
    output = step(
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.zeros((1, 3), dtype=torch.long),
        torch.tensor([[True, True, True]]),
    )
    torch.testing.assert_close(output.logits[:, 0], output.base_logodds[:, 0])
    assert torch.count_nonzero(output.correction[:, 0]) == 0
    assert torch.count_nonzero(output.correction[:, 1:]) > 0


def test_correction_has_no_direct_autograd_path_from_local_representation() -> None:
    step = _step()
    r_loc, r_ctx, r_gate, query = _inputs()
    output = step(
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.zeros((1, 3), dtype=torch.long),
        torch.tensor([[True, True, True]]),
    )
    gradient = torch.autograd.grad(output.correction[:, 1:].sum(), r_loc, allow_unused=True)[0]
    assert gradient is None


def test_correction_functionally_reads_only_context_and_prefix_path() -> None:
    step = _step()
    final_correction = step.correction[-1]
    assert isinstance(final_correction, torch.nn.Linear)
    with torch.no_grad():
        final_correction.weight.normal_(mean=0.0, std=0.2)
        final_correction.bias.zero_()

    r_loc, r_ctx, r_gate, query = _inputs()
    states = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[True, True, True]])
    baseline = step(r_loc, r_ctx, r_gate, query, states, mask)
    unrelated_changed = step(
        r_loc + 100.0,
        r_ctx,
        r_gate - 100.0,
        query,
        states,
        mask,
    )
    context_changed = step(
        r_loc,
        r_ctx + 10.0,
        r_gate,
        query,
        states,
        mask,
    )

    torch.testing.assert_close(baseline.correction, unrelated_changed.correction)
    assert not torch.allclose(baseline.correction[:, 1:], context_changed.correction[:, 1:])


def test_prefix_mlp_is_ordered_masked_and_has_exact_zero_empty_prefix() -> None:
    step = _step()
    _, _, _, query = _inputs()
    mask = torch.tensor([[True, True, True]])
    states_01 = torch.tensor([[0, 1, 0]])
    states_10 = torch.tensor([[1, 0, 0]])
    history_01 = step.prefix_history(query, states_01, mask)
    history_10 = step.prefix_history(query, states_10, mask)

    assert torch.count_nonzero(history_01[:, 0]) == 0
    assert torch.count_nonzero(history_01[:, 1]) > 0
    assert not torch.allclose(history_01[:, 2], history_10[:, 2])

    future_changed = states_01.clone()
    future_changed[:, 1:] = 1 - future_changed[:, 1:]
    future_history = step.prefix_history(query, future_changed, mask)
    torch.testing.assert_close(history_01[:, 1], future_history[:, 1])


def test_prefix_mlp_distinguishes_real_good_state_from_padding_and_rejects_k_above_three() -> None:
    step = _step()
    _, _, _, query = _inputs()
    real_good = step.prefix_history(
        query,
        torch.zeros((1, 3), dtype=torch.long),
        torch.tensor([[True, True, False]]),
    )
    assert torch.count_nonzero(real_good[:, 1]) > 0

    too_long_query = torch.randn((1, 4, 2))
    with pytest.raises(ValueError, match="exceeds"):
        step.prefix_history(
            too_long_query,
            torch.zeros((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.bool),
        )


def test_gate_is_path_independent_while_correction_can_depend_on_prefix() -> None:
    step = _step()
    final_correction = step.correction[-1]
    assert isinstance(final_correction, torch.nn.Linear)
    with torch.no_grad():
        final_correction.weight.normal_(mean=0.0, std=0.2)
        final_correction.bias.zero_()

    r_loc, r_ctx, r_gate, query = _inputs()
    mask = torch.tensor([[True, True, True]])
    output_a = step(
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.tensor([[0, 0, 0]]),
        mask,
    )
    output_b = step(
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.tensor([[1, 1, 0]]),
        mask,
    )
    torch.testing.assert_close(output_a.gate, output_b.gate)
    assert not torch.allclose(output_a.correction[:, 1:], output_b.correction[:, 1:])


def test_lexicographic_paths_and_order_controls_are_complete() -> None:
    assert lexicographic_binary_paths(2, device=torch.device("cpu")).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert physical_and_control_orders(2) == ((0, 1), (1, 0))
    orders = physical_and_control_orders(3)
    assert orders[0] == (0, 1, 2)
    assert len(orders) == 6
    assert len(set(orders)) == 6


def test_exact_enumerator_normalizes_and_matches_bruteforce_marginals() -> None:
    step = _step()
    with torch.no_grad():
        step.base.weight.zero_()
        step.base.bias.fill_(0.7)
    r_loc, r_ctx, r_gate, query = _inputs(kmax=2)
    output = enumerate_pcc(step, r_loc, r_ctx, r_gate, query, torch.tensor([[True, True]]))
    probabilities = output.probabilities[0, output.path_mask[0]]
    paths = output.path_states[0, output.path_mask[0], :2].to(dtype=probabilities.dtype)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    expected_marginal = (probabilities.unsqueeze(1) * paths).sum(dim=0)
    torch.testing.assert_close(output.marginals[0, :2], expected_marginal)
    expected_single = torch.sigmoid(torch.tensor(0.7))
    torch.testing.assert_close(output.marginals[0, :2], expected_single.expand(2))


def test_k1_degenerates_to_a_plain_bernoulli_and_map_matches_argmax() -> None:
    step = _step()
    with torch.no_grad():
        step.base.weight.zero_()
        step.base.bias.fill_(-0.3)
    r_loc, r_ctx, r_gate, query = _inputs(kmax=1)
    output = enumerate_pcc(step, r_loc, r_ctx, r_gate, query, torch.tensor([[True]]))
    probabilities = output.probabilities[0, output.path_mask[0]]
    torch.testing.assert_close(probabilities, torch.tensor([torch.sigmoid(torch.tensor(0.3)), torch.sigmoid(torch.tensor(-0.3))]))
    best_path = output.path_states[0, output.path_mask[0]][probabilities.argmax()]
    assert output.map_states[0, 0].item() == best_path[0].item()


def test_log_space_probabilities_match_direct_bernoulli_product() -> None:
    step = _step()
    r_loc, r_ctx, r_gate, query = _inputs(kmax=2)
    output = enumerate_pcc(step, r_loc, r_ctx, r_gate, query, torch.tensor([[True, True]]))
    paths = output.path_states[0, output.path_mask[0], :2]
    direct = []
    for path in paths:
        logits = step(r_loc, r_ctx, r_gate, query, path.view(1, -1), torch.tensor([[True, True]])).logits[0]
        terms = torch.where(path.bool(), torch.sigmoid(logits), torch.sigmoid(-logits))
        direct.append(terms.prod())
    torch.testing.assert_close(output.probabilities[0, output.path_mask[0]], torch.stack(direct))


def test_mixed_k_masks_exclude_padding_from_paths_marginals_and_probabilities() -> None:
    step = _step()
    r_loc, r_ctx, r_gate, query = _inputs(batch_size=2, kmax=3)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = enumerate_pcc(step, r_loc, r_ctx, r_gate, query, mask)
    assert output.path_mask.sum(dim=1).tolist() == [4, 8]
    assert torch.all(output.path_states[0, output.path_mask[0], 2] == -1)
    assert output.map_states[0, 2].item() == -1
    torch.testing.assert_close(output.marginals[0, 2], torch.tensor(0.0))
    for index in range(2):
        torch.testing.assert_close(output.probabilities[index, output.path_mask[index]].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_padding_values_do_not_change_valid_path_distribution() -> None:
    step = _step()
    r_loc, r_ctx, r_gate, query = _inputs(kmax=3)
    mask = torch.tensor([[True, True, False]])
    baseline = enumerate_pcc(step, r_loc, r_ctx, r_gate, query, mask)
    r_loc_padded, r_ctx_padded, r_gate_padded, query_padded = (tensor.detach().clone() for tensor in (r_loc, r_ctx, r_gate, query))
    r_loc_padded[:, 2] = 1_000.0
    r_ctx_padded[:, 2] = -1_000.0
    r_gate_padded[:, 2] = 5_000.0
    query_padded[:, 2] = -5_000.0
    perturbed = enumerate_pcc(step, r_loc_padded, r_ctx_padded, r_gate_padded, query_padded, mask)
    torch.testing.assert_close(
        baseline.probabilities[0, baseline.path_mask[0]],
        perturbed.probabilities[0, perturbed.path_mask[0]],
    )


def test_exact_enumerator_is_explicitly_normalized_for_extreme_logits() -> None:
    step = _step()
    with torch.no_grad():
        step.base.weight.zero_()
        step.base.bias.fill_(1_000.0)
    r_loc, r_ctx, r_gate, query = _inputs(kmax=3)
    output = enumerate_pcc(
        step,
        r_loc,
        r_ctx,
        r_gate,
        query,
        torch.tensor([[True, True, True]]),
    )
    valid = output.path_mask[0]
    probabilities = output.probabilities[0, valid]
    assert torch.isfinite(output.log_probabilities[0, valid]).all()
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        torch.logsumexp(output.log_probabilities[0, valid], dim=0),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        probabilities.sum(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_exact_enumerator_rejects_noncontiguous_interface_mask() -> None:
    step = _step()
    r_loc, r_ctx, r_gate, query = _inputs(kmax=3)
    with pytest.raises(ValueError, match="contiguous ordered prefix"):
        enumerate_pcc(
            step,
            r_loc,
            r_ctx,
            r_gate,
            query,
            torch.tensor([[True, False, True]]),
        )
