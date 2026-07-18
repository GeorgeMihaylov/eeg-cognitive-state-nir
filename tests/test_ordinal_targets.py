from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from model_zoo.DL.ordinal import (
    build_corn_targets_and_masks,
    build_cumulative_targets,
    coral_loss,
    coral_loss_parts,
    corn_loss,
    corn_loss_parts,
)


EXPECTED_TARGETS = torch.tensor(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=torch.float32,
)
EXPECTED_CORN_MASKS = torch.tensor(
    [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=torch.float32,
)


def test_cumulative_targets_cover_all_five_classes() -> None:
    labels = torch.arange(5, dtype=torch.int64)
    actual = build_cumulative_targets(labels, 5)
    torch.testing.assert_close(actual, EXPECTED_TARGETS)
    assert actual.shape == (5, 4)
    assert actual.dtype == torch.float32
    assert actual.device == labels.device


def test_corn_targets_and_masks_cover_all_five_classes() -> None:
    labels = torch.arange(5, dtype=torch.int64)
    targets, masks = build_corn_targets_and_masks(labels, 5)
    torch.testing.assert_close(targets, EXPECTED_TARGETS)
    torch.testing.assert_close(masks, EXPECTED_CORN_MASKS)


@pytest.mark.parametrize(
    "labels, message",
    [
        (torch.tensor([0.0, 1.0]), "integer tensor dtype"),
        (torch.tensor([-1, 0]), "must be in"),
        (torch.tensor([0, 5]), "must be in"),
        (torch.tensor([[0, 1]]), "shape"),
        (torch.tensor([], dtype=torch.int64), "cannot be empty"),
    ],
)
def test_invalid_ordinal_labels_are_rejected(
    labels: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_cumulative_targets(labels, 5)


def test_cumulative_targets_support_logits_dtype() -> None:
    targets = build_cumulative_targets(
        torch.tensor([0, 4]),
        5,
        dtype=torch.float64,
    )
    assert targets.dtype == torch.float64


def test_coral_loss_matches_manual_bce_mean() -> None:
    labels = torch.arange(5, dtype=torch.int64)
    logits = torch.zeros(5, 4, requires_grad=True)
    actual = coral_loss(logits, labels, 5)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        EXPECTED_TARGETS,
        reduction="mean",
    )
    torch.testing.assert_close(actual, expected)
    assert actual.item() == pytest.approx(math.log(2.0))
    parts = coral_loss_parts(logits, labels, 5)
    assert parts.denominator.item() == 20


def test_corn_loss_matches_manual_masked_bce() -> None:
    labels = torch.arange(5, dtype=torch.int64)
    logits = torch.zeros(5, 4, requires_grad=True)
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        EXPECTED_TARGETS,
        reduction="none",
    )
    expected = (elementwise * EXPECTED_CORN_MASKS).sum() / (
        EXPECTED_CORN_MASKS.sum()
    )
    actual = corn_loss(logits, labels, 5)
    torch.testing.assert_close(actual, expected)
    parts = corn_loss_parts(logits, labels, 5)
    assert parts.denominator.item() == 14


def test_empty_upper_corn_risk_sets_remain_finite() -> None:
    labels = torch.zeros(4, dtype=torch.int64)
    logits = torch.randn(4, 4, requires_grad=True)
    parts = corn_loss_parts(logits, labels, 5)
    assert parts.denominator.item() == 4
    assert torch.isfinite(parts.mean)
    parts.mean.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, 1:], torch.zeros_like(logits.grad[:, 1:]))


@pytest.mark.parametrize("loss_function", [coral_loss, corn_loss])
def test_ordinal_losses_produce_finite_gradients(loss_function) -> None:
    labels = torch.arange(10, dtype=torch.int64) % 5
    logits = torch.randn(10, 4, requires_grad=True)
    loss = loss_function(logits, labels, 5)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
