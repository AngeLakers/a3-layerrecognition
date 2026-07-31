from __future__ import annotations

import pytest
import torch

from a3_ultrasound.models.responsibility_tokens import ResponsibilityTokenProjector


def test_responsibility_projector_rejects_degenerate_one_dimensional_layer_norm() -> None:
    with pytest.raises(ValueError, match=r"D_r.*> 1"):
        ResponsibilityTokenProjector(
            local_dim=3,
            context_dim=4,
            query_dim=2,
            token_dim=1,
        )


def test_responsibility_projector_uses_independent_source_projections_and_norms() -> None:
    projector = ResponsibilityTokenProjector(
        local_dim=3,
        context_dim=4,
        query_dim=2,
        token_dim=5,
    )
    projection_ids = {
        id(projector.local_evidence_projection.weight),
        id(projector.local_query_projection.weight),
        id(projector.context_projection.weight),
        id(projector.context_query_projection.weight),
        id(projector.gate_evidence_projection.weight),
        id(projector.gate_context_projection.weight),
        id(projector.gate_query_projection.weight),
    }
    assert len(projection_ids) == 7
    assert len({id(projector.local_norm), id(projector.context_norm), id(projector.gate_norm)}) == 3


def test_responsibility_projector_source_dependency_graph_is_exact() -> None:
    torch.manual_seed(17)
    projector = ResponsibilityTokenProjector(
        local_dim=3,
        context_dim=4,
        query_dim=2,
        token_dim=5,
    )
    e_loc = torch.randn((1, 2, 3), requires_grad=True)
    c_pre = torch.randn((1, 2, 4), requires_grad=True)
    query = torch.randn((1, 2, 2), requires_grad=True)
    tokens = projector(e_loc, c_pre, query, torch.ones((1, 2), dtype=torch.bool))

    local_dependencies = torch.autograd.grad(
        tokens.r_loc[..., 0].sum(),
        (e_loc, c_pre, query),
        allow_unused=True,
        retain_graph=True,
    )
    context_dependencies = torch.autograd.grad(
        tokens.r_ctx[..., 0].sum(),
        (e_loc, c_pre, query),
        allow_unused=True,
        retain_graph=True,
    )
    gate_dependencies = torch.autograd.grad(
        tokens.r_gate[..., 0].sum(),
        (e_loc, c_pre, query),
        allow_unused=True,
    )

    assert [gradient is not None for gradient in local_dependencies] == [True, False, True]
    assert [gradient is not None for gradient in context_dependencies] == [False, True, True]
    assert [gradient is not None for gradient in gate_dependencies] == [True, True, True]


def test_responsibility_projector_reapplies_interface_mask_after_layer_norm() -> None:
    torch.manual_seed(13)
    projector = ResponsibilityTokenProjector(
        local_dim=3,
        context_dim=4,
        query_dim=2,
        token_dim=5,
    )
    mask = torch.tensor([[True, False]])
    tokens = projector(
        torch.randn((1, 2, 3)),
        torch.randn((1, 2, 4)),
        torch.randn((1, 2, 2)),
        mask,
    )
    assert tokens.r_loc.shape == tokens.r_ctx.shape == tokens.r_gate.shape == (1, 2, 5)
    assert torch.count_nonzero(tokens.r_loc[:, 1]) == 0
    assert torch.count_nonzero(tokens.r_ctx[:, 1]) == 0
    assert torch.count_nonzero(tokens.r_gate[:, 1]) == 0
