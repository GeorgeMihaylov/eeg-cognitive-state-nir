"""Compatibility facade for canonical sklearn model builders."""

from model_zoo.ML.sklearn_models import (
    CLASSIFICATION_MODEL_NAMES,
    REGRESSION_MODEL_NAMES,
    SKLEARN_MODEL_NAMES,
    build_sklearn_model,
)

__all__ = [
    "CLASSIFICATION_MODEL_NAMES",
    "REGRESSION_MODEL_NAMES",
    "SKLEARN_MODEL_NAMES",
    "build_sklearn_model",
]
