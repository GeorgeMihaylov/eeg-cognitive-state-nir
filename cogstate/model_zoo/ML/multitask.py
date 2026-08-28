"""Seven-target PM classification built from the repository model factory."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cogstate.protocol import N_PM_CLASSES, PM_METRICS
from ..factory import build_model


class PMMultiTaskClassifier:
    """One leakage-safe classifier per PM target, with a uniform API."""

    def __init__(
        self,
        model_name: str,
        *,
        input_shape: Sequence[int] | None = None,
        params: Mapping[str, Any] | None = None,
        metric_names: Sequence[str] = PM_METRICS,
    ) -> None:
        self.model_name = model_name
        self.input_shape = tuple(input_shape) if input_shape else None
        self.params = dict(params or {})
        self.metric_names = tuple(metric_names)
        self.models_: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PMMultiTaskClassifier":
        labels = np.asarray(y)
        if labels.ndim != 2 or labels.shape[1] != len(self.metric_names):
            raise ValueError("y must be [windows, seven PM targets]")
        self.models_.clear()
        for index, metric in enumerate(self.metric_names):
            valid = labels[:, index] >= 0
            if valid.sum() == 0:
                continue
            model = build_model(
                self.model_name,
                "classification",
                self.input_shape,
                N_PM_CLASSES,
                self.params,
            )
            model.fit(np.asarray(X)[valid], labels[valid, index])
            self.models_[metric] = model
        if not self.models_:
            raise ValueError("No PM target has valid labels")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = np.full((len(X), len(self.metric_names)), -1, dtype=np.int8)
        for index, metric in enumerate(self.metric_names):
            if metric in self.models_:
                output[:, index] = self.models_[metric].predict(X)
        return output

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {
            metric: self.models_[metric].predict_proba(X)
            for metric in self.metric_names
            if metric in self.models_
        }
