from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from cogstate.model_zoo import BaseModelAdapter, ModelLike, build_model
from cogstate.model_zoo.DL import TorchClassificationAdapter


def test_build_random_forest_classifier() -> None:
    model = build_model(
        model_name="random_forest",
        task_type="classification",
        input_shape=(20,),
        num_outputs=5,
        params={"n_estimators": 7, "random_state": 42},
    )

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 7
    assert isinstance(model, ModelLike)


def test_build_svm_classifier() -> None:
    model = build_model(
        model_name="svm",
        task_type="classification",
        input_shape=(4,),
        num_outputs=3,
        params={"C": 2.0},
    )

    assert isinstance(model, SVC)
    assert model.C == 2.0


def test_build_random_forest_regressor() -> None:
    model = build_model(
        model_name="random_forest",
        task_type="regression",
        input_shape=(8,),
        num_outputs=1,
        params={"n_estimators": 5, "random_state": 42},
    )

    assert isinstance(model, RandomForestRegressor)


def test_build_torch_mlp_classifier() -> None:
    model = build_model(
        model_name="torch_mlp",
        task_type="classification",
        input_shape=(12,),
        num_outputs=5,
        params={
            "hidden_dims": [16, 8],
            "max_epochs": 2,
            "device": "cpu",
        },
    )

    assert isinstance(model, TorchClassificationAdapter)
    assert model.input_shape == (12,)
    assert model.num_classes == 5


def test_build_torch_shallow_convnet_scalar_regressor() -> None:
    model = build_model(
        model_name="torch_shallow_convnet",
        task_type="regression",
        input_shape=(1, 4, 128),
        num_outputs=1,
        params={
            "n_filters": 4,
            "temporal_kernel_samples": 9,
            "pool_size": 16,
            "pool_stride": 4,
            "device": "cpu",
        },
    )

    assert isinstance(model, TorchClassificationAdapter)
    assert model.task_type == "regression"
    assert model.num_outputs == 1


def test_reject_incompatible_model_and_task() -> None:
    with pytest.raises(ValueError, match="not available for regression"):
        build_model(
            model_name="logistic_regression",
            task_type="regression",
            input_shape=(8,),
            num_outputs=1,
            params={},
        )


def test_reject_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(
            model_name="unknown",
            task_type="classification",
            input_shape=(8,),
            num_outputs=5,
            params={},
        )


class DummyAdapter(BaseModelAdapter):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyAdapter":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def save(self, path: str | Path) -> None:
        return None


def test_future_adapter_satisfies_runner_protocol() -> None:
    assert isinstance(DummyAdapter(), ModelLike)
