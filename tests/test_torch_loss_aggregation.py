from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from model_zoo import build_model
from model_zoo.DL.adapter import _aggregate_loss_component_values
from model_zoo.DL.ordinal import ClassificationObjectiveHandler
from model_zoo.DL.regression import RegressionObjectiveHandler


def _aggregate_batches(
    handler,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for start in range(0, len(outputs), batch_size):
        parts_by_name = handler.loss_component_parts(
            outputs[start:start + batch_size],
            targets[start:start + batch_size],
        )
        for name, parts in parts_by_name.items():
            numerators[name] = (
                numerators.get(name, 0.0)
                + float(parts.numerator.detach().item())
            )
            denominators[name] = (
                denominators.get(name, 0.0)
                + float(parts.denominator.detach().item())
            )
    values = _aggregate_loss_component_values(numerators, denominators)
    return values["objective"], denominators["objective"]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_multioutput_mse_matches_numpy_and_is_batch_size_independent(
    batch_size: int,
) -> None:
    targets = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=torch.float32,
    )
    outputs = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [3.0, 2.0]],
        dtype=torch.float32,
    )
    expected = float(
        np.mean((targets.numpy() - outputs.numpy()) ** 2)
    )

    actual, denominator = _aggregate_batches(
        RegressionObjectiveHandler(2, "mse"),
        outputs,
        targets,
        batch_size,
    )

    assert actual == pytest.approx(expected)
    assert denominator == 6


def test_scalar_and_seven_output_mse_use_element_count_denominator() -> None:
    scalar_targets = torch.tensor([0.0, 1.0, 2.0])
    scalar_outputs = torch.tensor([[1.0], [1.0], [0.0]])
    scalar_loss, scalar_denominator = _aggregate_batches(
        RegressionObjectiveHandler(1, "mse"),
        scalar_outputs,
        scalar_targets,
        batch_size=2,
    )
    assert scalar_loss == pytest.approx(
        float(np.mean((scalar_targets.numpy() - scalar_outputs[:, 0].numpy()) ** 2))
    )
    assert scalar_denominator == len(scalar_targets)

    targets = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 10
    outputs = targets + torch.linspace(-0.3, 0.3, 35).reshape(5, 7)
    loss, denominator = _aggregate_batches(
        RegressionObjectiveHandler(7, "mse"),
        outputs,
        targets,
        batch_size=2,
    )
    assert loss == pytest.approx(
        float(np.mean((targets.numpy() - outputs.numpy()) ** 2))
    )
    assert denominator == targets.numel()


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_smooth_l1_mean_is_batch_size_independent(batch_size: int) -> None:
    targets = torch.linspace(-1.0, 1.0, 15).reshape(5, 3)
    outputs = targets + torch.linspace(-2.0, 2.0, 15).reshape(5, 3)
    expected = float(
        F.smooth_l1_loss(outputs, targets, reduction="mean").item()
    )

    actual, denominator = _aggregate_batches(
        RegressionObjectiveHandler(3, "smooth_l1"),
        outputs,
        targets,
        batch_size,
    )

    assert actual == pytest.approx(expected)
    assert denominator == targets.numel()


@pytest.mark.parametrize("head_type", ["categorical", "coral", "corn"])
def test_classification_and_ordinal_component_aggregation_is_unchanged(
    head_type: str,
) -> None:
    labels = torch.tensor([0, 1, 2, 3, 1], dtype=torch.int64)
    width = 4 if head_type == "categorical" else 3
    outputs = torch.linspace(-1.5, 1.5, len(labels) * width).reshape(
        len(labels), width
    )
    handler = ClassificationObjectiveHandler(head_type, num_classes=4)
    expected = float(
        handler.loss_component_parts(outputs, labels)["objective"].mean.item()
    )

    actual, _ = _aggregate_batches(
        handler,
        outputs,
        labels,
        batch_size=2,
    )

    assert actual == pytest.approx(expected)


def test_regression_backward_uses_mean_loss_not_sum() -> None:
    targets = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    outputs = torch.tensor(
        [[1.0, 0.0], [4.0, 1.0]],
        requires_grad=True,
    )
    parts = RegressionObjectiveHandler(2, "mse").loss_component_parts(
        outputs, targets
    )["objective"]

    parts.mean.backward()

    expected_gradient = 2.0 * (outputs.detach() - targets) / outputs.numel()
    torch.testing.assert_close(outputs.grad, expected_gradient)


def test_best_validation_loss_matches_normalized_training_log() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(48, 4)).astype(np.float32)
    targets = rng.normal(size=(48, 2)).astype(np.float32)
    subjects = np.repeat([f"subject_{index}" for index in range(8)], 6)
    records = np.asarray([f"record_{index // 3}" for index in range(48)])
    model = build_model(
        "torch_mlp",
        "regression",
        input_shape=(4,),
        num_outputs=2,
        params={
            "hidden_dims": [8],
            "dropout": 0.0,
            "batch_size": 7,
            "max_epochs": 3,
            "learning_rate": 0.001,
            "early_stopping_patience": 3,
            "device": "cpu",
            "random_state": 42,
        },
    )
    model.set_validation_groups(
        subjects,
        subject_ids=subjects,
        record_ids=records,
        strategy="group_holdout",
        group_column="subject_id",
        validation_size=0.25,
        random_state=42,
    )

    model.fit(features, targets)

    logged_losses = [
        float(row["validation_loss"]) for row in model.training_log_
    ]
    assert model.best_validation_loss_ == pytest.approx(min(logged_losses))
    assert model.best_validation_loss_ < 10
