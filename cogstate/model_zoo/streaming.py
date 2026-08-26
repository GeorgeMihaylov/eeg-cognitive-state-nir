"""Adapters that connect model-zoo estimators to ``cogstate.streaming``."""
from __future__ import annotations

import numpy as np


class StreamingModelAdapter:
    """Expose model-zoo ``predict_proba`` output as named probabilities."""
    def __init__(self, estimator, class_names=None, version: str = "model-zoo"):
        self.estimator = estimator
        self.class_names = list(class_names) if class_names is not None else None
        self.version = version

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        values = np.asarray(features)
        expected = getattr(self.estimator, "input_shape", None)
        if expected is not None and tuple(values.shape) == tuple(expected): values = values[None, ...]
        elif values.ndim == 1: values = values[None, :]
        probabilities = np.asarray(self.estimator.predict_proba(values))[0]
        classes = getattr(self.estimator, "classes_", None)
        names = self.class_names or ([str(value) for value in classes] if classes is not None else [str(i) for i in range(len(probabilities))])
        return dict(zip(names, map(float, probabilities)))


class StreamingPMMultiTaskAdapter:
    """Expose seven independent PM heads to the streaming inference service."""
    def __init__(self, estimator, version: str = "model-zoo"):
        self.estimator, self.version = estimator, version

    def predict_pm_proba(self, features: np.ndarray) -> dict[str, dict[str, float]]:
        values = np.asarray(features)
        expected = getattr(self.estimator, "input_shape", None)
        if expected is not None and tuple(values.shape) == tuple(expected): values = values[None, ...]
        elif values.ndim == 1: values = values[None, :]
        return {
            metric: dict(zip(("low", "medium", "high"), map(float, probabilities[0])))
            for metric, probabilities in self.estimator.predict_proba(values).items()
        }
