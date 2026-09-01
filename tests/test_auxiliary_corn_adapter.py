from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cogstate.model_zoo import build_model


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(5), 12).astype(np.int64)
    features = rng.normal(size=(len(labels), 8, 6)).astype(np.float32)
    features[:, :, 0] += labels[:, None] * 0.35
    return features, labels


def _adapter(weight: float):
    return build_model(
        "torch_transformer",
        "classification",
        input_shape=(8, 6),
        num_outputs=5,
        params={
            "head_type": "categorical_corn",
            "auxiliary_weight": weight,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 64,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )


@pytest.mark.parametrize("weight", [0.25, 0.5, 1.0])
def test_synthetic_fit_logs_components_and_returns_primary_and_auxiliary_predictions(
    weight: float,
) -> None:
    features, labels = _data()
    adapter = _adapter(weight).fit(features, labels)
    row = adapter.training_log_[0]
    assert row["train_total_loss"] == pytest.approx(
        row["train_categorical_loss"] + weight * row["train_ordinal_loss"]
    )
    assert row["validation_total_loss"] == pytest.approx(
        row["validation_categorical_loss"]
        + weight * row["validation_ordinal_loss"]
    )
    assert row["early_stopping_metric"] == "validation_categorical_loss"
    assert adapter.get_training_summary()["early_stopping_monitor"] == (
        "validation_categorical_loss"
    )

    detailed = adapter.predict_detailed(features[:15])
    assert detailed["class_probabilities"].shape == (15, 5)
    assert detailed["aux_threshold_probabilities"].shape == (15, 4)
    assert detailed["aux_class_probabilities"].shape == (15, 5)
    assert detailed["aux_expected_rank"].shape == (15,)
    assert detailed["aux_ordinal_prediction"].shape == (15,)
    np.testing.assert_array_equal(
        detailed["y_pred"], detailed["class_probabilities"].argmax(axis=1)
    )
    np.testing.assert_array_equal(
        detailed["aux_ordinal_prediction"],
        (detailed["aux_threshold_probabilities"] >= 0.5).sum(axis=1),
    )
    np.testing.assert_allclose(
        detailed["class_probabilities"].sum(axis=1), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        detailed["aux_class_probabilities"].sum(axis=1), 1.0, atol=1e-6
    )
    assert np.all(
        detailed["aux_threshold_probabilities"][:, :-1]
        >= detailed["aux_threshold_probabilities"][:, 1:]
    )


def test_joint_checkpoint_strict_reload_and_head_mismatches(tmp_path: Path) -> None:
    features, labels = _data()
    fitted = _adapter(0.5).fit(features, labels)
    before = fitted.predict_detailed(features[:10])
    checkpoint = tmp_path / "joint.pt"
    fitted.save(checkpoint)

    reloaded = _adapter(0.5).load(checkpoint)
    after = reloaded.predict_detailed(features[:10])
    for key in (
        "class_probabilities",
        "y_pred",
        "aux_threshold_probabilities",
        "aux_class_probabilities",
        "aux_expected_rank",
    ):
        np.testing.assert_allclose(before[key], after[key], atol=0, rtol=0)

    with pytest.raises(ValueError, match="auxiliary_weight"):
        _adapter(1.0).load(checkpoint)

    categorical = build_model(
        "torch_transformer",
        "classification",
        (8, 6),
        5,
        {"d_model": 8, "nhead": 2, "device": "cpu"},
    )
    with pytest.raises(ValueError, match="Checkpoint head_type"):
        categorical.load(checkpoint)


def test_joint_calibration_is_rejected_before_optimization() -> None:
    features, labels = _data()
    adapter = _adapter(0.5).fit(features, labels)
    with pytest.raises(NotImplementedError, match="Auxiliary CORN calibration"):
        adapter.fine_tune(features[:10], labels[:10])


def test_shared_encoder_receives_both_task_gradients() -> None:
    features, labels = _data()
    adapter = _adapter(0.5)
    model = adapter.model
    batch = torch.from_numpy(features[:10])
    targets = torch.from_numpy(labels[:10])

    outputs = model(batch)
    categorical_parts = adapter.objective_handler.loss_component_parts(
        outputs, targets
    )
    categorical_loss = categorical_parts["categorical"].mean
    categorical_loss.backward(retain_graph=True)
    categorical_grad = model.input_projection.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    outputs = model(batch)
    ordinal_parts = adapter.objective_handler.loss_component_parts(outputs, targets)
    ordinal_loss = ordinal_parts["ordinal"].mean
    ordinal_loss.backward()
    ordinal_grad = model.input_projection.weight.grad.detach().clone()

    assert torch.isfinite(categorical_grad).all()
    assert torch.isfinite(ordinal_grad).all()
    assert torch.count_nonzero(categorical_grad) > 0
    assert torch.count_nonzero(ordinal_grad) > 0
