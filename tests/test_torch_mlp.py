import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from cogstate.model_zoo import build_model
from cogstate.model_zoo.DL import TorchClassificationAdapter, TorchMLP


@pytest.fixture(scope="module")
def synthetic_classification_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=150,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int64)


@pytest.fixture(scope="module")
def fitted_torch_mlp(
    synthetic_classification_data: tuple[np.ndarray, np.ndarray],
) -> TorchClassificationAdapter:
    X, y = synthetic_classification_data
    model = build_model(
        model_name="torch_mlp",
        task_type="classification",
        input_shape=(X.shape[1],),
        num_outputs=3,
        params={
            "hidden_dims": [24, 12],
            "dropout": 0.1,
            "batch_size": 32,
            "max_epochs": 4,
            "learning_rate": 0.002,
            "validation_size": 0.2,
            "early_stopping_patience": 2,
            "device": "cpu",
            "random_state": 42,
        },
    )
    assert isinstance(model, TorchClassificationAdapter)
    return model.fit(X, y)


def test_torch_mlp_forward_pass() -> None:
    model = TorchMLP(
        input_dim=12,
        num_classes=5,
        hidden_dims=[16, 8],
        dropout=0.1,
        activation="relu",
    )

    output = model(torch.randn(7, 12))

    assert output.shape == (7, 5)


def test_torch_mlp_fit_predict(
    fitted_torch_mlp: TorchClassificationAdapter,
    synthetic_classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = synthetic_classification_data
    predictions = fitted_torch_mlp.predict(X[:20])

    assert predictions.shape == (20,)
    assert set(np.unique(predictions)).issubset({0, 1, 2})
    assert fitted_torch_mlp.n_epochs_trained_ >= 1
    assert fitted_torch_mlp.best_validation_loss_ is not None


def test_torch_mlp_predict_proba(
    fitted_torch_mlp: TorchClassificationAdapter,
    synthetic_classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = synthetic_classification_data
    probabilities = fitted_torch_mlp.predict_proba(X[:20])

    assert probabilities.shape == (20, 3)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_torch_mlp_save_checkpoint(
    fitted_torch_mlp: TorchClassificationAdapter,
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"

    fitted_torch_mlp.save(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.exists()
    assert "model_state_dict" in payload
    assert payload["input_shape"] == (12,)
    assert payload["num_classes"] == 3


def test_torch_mlp_rejects_wrong_shape() -> None:
    model = build_model(
        "torch_mlp",
        "classification",
        input_shape=(12,),
        num_outputs=3,
        params={"max_epochs": 1, "device": "cpu"},
    )

    with pytest.raises(ValueError, match="Expected input_shape"):
        model.fit(np.ones((30, 11), dtype=np.float32), np.arange(30) % 3)


def test_torch_mlp_rejects_non_finite_values() -> None:
    model = build_model(
        "torch_mlp",
        "classification",
        input_shape=(12,),
        num_outputs=3,
        params={"max_epochs": 1, "device": "cpu"},
    )
    X = np.ones((30, 12), dtype=np.float32)
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, np.arange(30) % 3)
