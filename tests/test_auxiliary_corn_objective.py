from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cogstate.model_zoo.DL.ordinal import (
    CategoricalCornObjectiveHandler,
    CategoricalCornOutput,
    corn_loss,
)


def _outputs() -> CategoricalCornOutput:
    torch.manual_seed(42)
    return CategoricalCornOutput(
        categorical_logits=torch.randn(10, 5, requires_grad=True),
        ordinal_logits=torch.randn(10, 4, requires_grad=True),
    )


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_composite_loss_matches_manual_formula(weight: float) -> None:
    labels = torch.arange(10) % 5
    outputs = _outputs()
    handler = CategoricalCornObjectiveHandler(5, weight)
    losses = handler.loss_components(outputs, labels)
    expected_ce = F.cross_entropy(outputs.categorical_logits, labels)
    expected_corn = corn_loss(outputs.ordinal_logits, labels, 5)
    torch.testing.assert_close(losses.categorical_loss, expected_ce)
    torch.testing.assert_close(losses.ordinal_loss, expected_corn)
    torch.testing.assert_close(
        losses.total_loss,
        expected_ce + weight * expected_corn,
    )


def test_zero_weight_has_zero_auxiliary_gradient_and_encoder_path_is_trainable() -> None:
    labels = torch.arange(10) % 5
    outputs = _outputs()
    handler = CategoricalCornObjectiveHandler(5, 0.0)
    loss = handler.compute_loss(outputs, labels)
    loss.backward()
    assert outputs.categorical_logits.grad is not None
    assert torch.isfinite(outputs.categorical_logits.grad).all()
    assert outputs.ordinal_logits.grad is not None
    assert torch.count_nonzero(outputs.ordinal_logits.grad) == 0


def test_positive_weight_produces_finite_gradients_for_both_heads() -> None:
    labels = torch.arange(10) % 5
    outputs = _outputs()
    handler = CategoricalCornObjectiveHandler(5, 0.5)
    handler.compute_loss(outputs, labels).backward()
    for tensor in (outputs.categorical_logits, outputs.ordinal_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


def test_decode_uses_categorical_primary_and_corn_auxiliary_semantics() -> None:
    outputs = CategoricalCornOutput(
        categorical_logits=torch.tensor([[0.0, 0.1, 3.0, 0.2, -1.0]]),
        ordinal_logits=torch.tensor([[4.0, 4.0, -4.0, -4.0]]),
    )
    decoded = CategoricalCornObjectiveHandler(5, 0.5).decode(outputs)
    assert decoded.y_pred.item() == 2
    assert decoded.aux_ordinal_prediction is not None
    assert decoded.aux_ordinal_prediction.item() == 2
    assert decoded.class_probabilities.shape == (1, 5)
    assert decoded.aux_class_probabilities is not None
    assert decoded.aux_class_probabilities.shape == (1, 5)
    torch.testing.assert_close(decoded.class_probabilities.sum(1), torch.ones(1))
    torch.testing.assert_close(decoded.aux_class_probabilities.sum(1), torch.ones(1))
