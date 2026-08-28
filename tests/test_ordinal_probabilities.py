from __future__ import annotations

import numpy as np
import pytest
import torch

from cogstate.model_zoo.DL.ordinal import (
    ClassificationObjectiveHandler,
    cumulative_to_class_probabilities,
    decode_ordinal_prediction,
    expected_rank,
    threshold_logits_to_cumulative_probabilities,
)


def test_coral_logits_convert_to_monotone_cumulative_probabilities() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    cumulative = threshold_logits_to_cumulative_probabilities(logits, "coral")
    torch.testing.assert_close(cumulative, torch.sigmoid(logits))
    assert torch.all(cumulative[:, :-1] >= cumulative[:, 1:])


def test_corn_logits_use_cumulative_product() -> None:
    logits = torch.tensor([[1.0, 0.5, -0.5, -1.0]])
    conditional = torch.sigmoid(logits)
    cumulative = threshold_logits_to_cumulative_probabilities(logits, "corn")
    torch.testing.assert_close(cumulative, torch.cumprod(conditional, dim=1))
    assert torch.all(cumulative[:, :-1] >= cumulative[:, 1:])


def test_cumulative_to_class_probabilities_is_exact() -> None:
    cumulative = torch.tensor([[0.8, 0.6, 0.3, 0.1]])
    actual = cumulative_to_class_probabilities(cumulative)
    expected = torch.tensor([[0.2, 0.2, 0.3, 0.2, 0.1]])
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 5)
    assert torch.all(actual >= 0)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(1))


def test_material_monotonicity_violation_raises() -> None:
    cumulative = torch.tensor([[0.8, 0.4, 0.7, 0.1]])
    with pytest.raises(ValueError, match="not monotone"):
        cumulative_to_class_probabilities(cumulative, tolerance=1e-6)


def test_roundoff_monotonicity_violation_is_corrected() -> None:
    cumulative = torch.tensor([[0.8, 0.6, 0.6000001, 0.1]])
    probabilities = cumulative_to_class_probabilities(
        cumulative,
        tolerance=1e-6,
    )
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(1),
        atol=1e-7,
        rtol=0,
    )


@pytest.mark.parametrize(
    "cumulative, expected",
    [
        ([0.49, 0.4, 0.3, 0.2], 0),
        ([0.5, 0.49, 0.4, 0.3], 1),
        ([0.9, 0.8, 0.7, 0.6], 4),
    ],
)
def test_threshold_decoding_uses_greater_than_or_equal_tie_rule(
    cumulative: list[float],
    expected: int,
) -> None:
    prediction = decode_ordinal_prediction(torch.tensor([cumulative]))
    assert prediction.tolist() == [expected]


def test_expected_rank_is_sum_and_bounded() -> None:
    cumulative = torch.tensor([[0.8, 0.6, 0.3, 0.1], [1.0, 1.0, 1.0, 1.0]])
    ranks = expected_rank(cumulative)
    torch.testing.assert_close(ranks, torch.tensor([1.8, 4.0]))
    assert torch.all((ranks >= 0) & (ranks <= 4))


def test_categorical_handler_preserves_softmax_and_argmax() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, 0.5]])
    decoded = ClassificationObjectiveHandler("categorical", 5).decode(logits)
    torch.testing.assert_close(decoded.class_probabilities, torch.softmax(logits, 1))
    assert decoded.y_pred.tolist() == [3]
    assert decoded.threshold_probabilities is None


@pytest.mark.parametrize("head_type", ["coral", "corn"])
def test_ordinal_handler_returns_unified_decoded_outputs(head_type: str) -> None:
    logits = torch.tensor([[1.0, 0.5, 0.0, -0.5], [2.0, 1.0, 0.0, -1.0]])
    if head_type == "coral":
        logits = logits.sort(dim=1, descending=True).values
    decoded = ClassificationObjectiveHandler(head_type, 5).decode(logits)
    assert decoded.class_probabilities.shape == (2, 5)
    assert decoded.threshold_probabilities is not None
    assert decoded.threshold_probabilities.shape == (2, 4)
    assert decoded.expected_rank is not None
    assert decoded.ordinal_argmax is not None
    assert torch.isfinite(decoded.class_probabilities).all()
    np.testing.assert_allclose(
        decoded.class_probabilities.sum(dim=1).numpy(),
        np.ones(2),
        atol=1e-6,
    )
    if head_type == "corn":
        assert decoded.conditional_probabilities is not None


def test_handler_rejects_unknown_head_and_bad_output_width() -> None:
    with pytest.raises(ValueError, match="Unsupported head_type"):
        ClassificationObjectiveHandler("future", 5)
    handler = ClassificationObjectiveHandler("coral", 5)
    with pytest.raises(ValueError, match="shape"):
        handler.decode(torch.zeros(3, 5))
