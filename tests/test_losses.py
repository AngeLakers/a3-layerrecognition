from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from a3_ultrasound.models.losses import pcc_training_loss


def test_per_scan_ki_normalization_has_k2_k3_scale_parity() -> None:
    logits = torch.zeros((2, 3), requires_grad=True)
    base = torch.zeros((2, 3), requires_grad=True)
    y = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    output = pcc_training_loss(
        logits,
        base,
        y,
        mask,
        lambda_local=0.3,
    )

    expected = F.binary_cross_entropy_with_logits(
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        output.per_sample_chain,
        expected.expand(2),
    )
    torch.testing.assert_close(
        output.per_sample_local,
        expected.expand(2),
    )
    torch.testing.assert_close(output.total, expected * 1.3)


def test_partial_label_mask_keeps_each_supervised_sample_equally_weighted() -> None:
    logits = torch.zeros((2, 3), requires_grad=True)
    base = torch.zeros((2, 3), requires_grad=True)
    y = torch.tensor([[1, 0, 9], [0, 1, 0]], dtype=torch.int64)
    interface_mask = torch.tensor([[True, True, False], [True, True, True]])
    label_mask = torch.tensor([[True, False, False], [False, False, False]])

    output = pcc_training_loss(
        logits,
        base,
        y,
        interface_mask,
        label_mask=label_mask,
        lambda_local=0.0,
    )

    expected_single = F.binary_cross_entropy_with_logits(
        torch.tensor(0.0),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(output.chain, expected_single)
    assert output.supervised_sample_mask.tolist() == [True, False]
    output.total.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] != 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_loss_rejects_padding_supervision_and_dirty_labels() -> None:
    logits = torch.zeros((1, 2))
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="padded"):
        pcc_training_loss(
            logits,
            logits,
            torch.tensor([[0, 1]], dtype=torch.int64),
            mask,
            label_mask=torch.tensor([[True, True]]),
        )
    with pytest.raises(ValueError, match="good=0/abnormal=1"):
        pcc_training_loss(
            logits,
            logits,
            torch.tensor([[2, 0]], dtype=torch.int64),
            mask,
        )
