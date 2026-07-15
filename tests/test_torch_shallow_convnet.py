import numpy as np
import torch

from model_zoo import build_model
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.DL.shallow_convnet import (
    SafeLog,
    SquareActivation,
    TorchShallowConvNetClassifier,
)


def _small_module() -> TorchShallowConvNetClassifier:
    return TorchShallowConvNetClassifier(
        n_channels=4,
        n_times=128,
        num_classes=5,
        n_filters=4,
        temporal_kernel_samples=9,
        pool_size=16,
        pool_stride=4,
        dropout=0.1,
    )


def test_shallow_convnet_forward_output_and_spatial_shape() -> None:
    module = _small_module()
    inputs = torch.zeros(3, 1, 4, 128)

    spatial = module.spatial(module.temporal(inputs))
    logits = module(inputs)

    assert spatial.shape[2] == 1
    assert logits.shape == (3, 5)


def test_square_and_safe_log_are_finite() -> None:
    values = torch.tensor([-1e8, -2.0, 0.0, 2.0, 1e8])
    transformed = SafeLog()(SquareActivation()(values))

    assert torch.isfinite(transformed).all()


def test_factory_builds_torch_shallow_convnet() -> None:
    adapter = build_model(
        model_name="torch_shallow_convnet",
        task_type="classification",
        input_shape=(1, 4, 128),
        num_outputs=5,
        params={
            "n_filters": 4,
            "temporal_kernel_samples": 9,
            "pool_size": 16,
            "pool_stride": 4,
            "device": "cpu",
            "random_state": 42,
        },
    )

    assert isinstance(adapter, TorchClassificationAdapter)
    assert isinstance(adapter.model, TorchShallowConvNetClassifier)
    assert adapter.input_shape == (1, 4, 128)
    assert adapter.num_classes == 5


def test_shallow_convnet_fit_predict_and_probabilities() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(40, 1, 4, 128)).astype(np.float32)
    labels = np.tile(np.arange(5), 8)
    adapter = build_model(
        model_name="torch_shallow_convnet",
        task_type="classification",
        input_shape=(1, 4, 128),
        num_outputs=5,
        params={
            "n_filters": 4,
            "temporal_kernel_samples": 9,
            "pool_size": 16,
            "pool_stride": 4,
            "dropout": 0.1,
            "batch_size": 8,
            "max_epochs": 1,
            "validation_size": 0.25,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )

    adapter.fit(features, labels)
    predictions = adapter.predict(features[:7])
    probabilities = adapter.predict_proba(features[:7])

    assert predictions.shape == (7,)
    assert probabilities.shape == (7, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
