"""Leakage-safe target transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TargetTransform(ABC):
    """Minimal fit/transform protocol for target-only transformations."""

    @abstractmethod
    def fit(self, y_outer_train: np.ndarray) -> "TargetTransform":
        """Fit using outer-train targets only."""

    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform a train, validation, or test partition."""

    @abstractmethod
    def manifest(self) -> dict[str, Any]:
        """Return deterministic transform provenance."""


class IdentityTargetTransform(TargetTransform):
    def __init__(self) -> None:
        self._fitted = False

    def fit(self, y_outer_train: np.ndarray) -> "IdentityTargetTransform":
        values = _as_single_output_float(y_outer_train)
        if not np.isfinite(values).any():
            raise ValueError("Cannot fit target transform without finite targets")
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Target transform must be fit before transform")
        return np.asarray(y).copy()

    def manifest(self) -> dict[str, Any]:
        return {"transform": "identity", "fitted": self._fitted}


class FoldLocalQuantileTargetTransform(TargetTransform):
    """Discretize a continuous target with outer-train-only quantiles."""

    def __init__(self, q: int, *, duplicates: str = "drop") -> None:
        if q not in {3, 5}:
            raise ValueError("Fold-local quantile transform supports q=3 or q=5")
        if duplicates not in {"drop", "raise"}:
            raise ValueError("duplicates must be 'drop' or 'raise'")
        self.q = int(q)
        self.duplicates = duplicates
        self._boundaries: np.ndarray | None = None
        self._fit_n = 0

    def fit(self, y_outer_train: np.ndarray) -> "FoldLocalQuantileTargetTransform":
        values = _as_single_output_float(y_outer_train)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Cannot fit quantile boundaries without finite targets")
        probabilities = np.linspace(0.0, 1.0, self.q + 1)
        try:
            boundaries = np.quantile(finite, probabilities, method="linear")
        except TypeError:  # NumPy < 1.22
            boundaries = np.quantile(finite, probabilities, interpolation="linear")
        unique_boundaries = np.unique(boundaries)
        if self.duplicates == "raise" and len(unique_boundaries) != len(boundaries):
            raise ValueError(
                "Quantile boundaries are not unique; choose duplicates='drop' "
                "or change the target"
            )
        self._boundaries = (
            unique_boundaries if self.duplicates == "drop" else boundaries
        ).astype(float, copy=False)
        self._fit_n = int(finite.size)
        return self

    @property
    def actual_class_count(self) -> int:
        if self._boundaries is None:
            raise RuntimeError("Target transform has not been fit")
        return max(1, len(self._boundaries) - 1)

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self._boundaries is None:
            raise RuntimeError("Target transform must be fit before transform")
        values = _as_single_output_float(y)
        result = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        # np.digitize with right=True matches right-closed qcut intervals.
        result[valid] = np.digitize(
            values[valid], self._boundaries[1:-1], right=True
        ).astype(float)
        return result

    def manifest(self) -> dict[str, Any]:
        if self._boundaries is None:
            raise RuntimeError("Target transform has not been fit")
        return {
            "transform": "fold_local_quantile",
            "fit_scope": "outer_train_only",
            "requested_quantiles": self.q,
            "duplicates": self.duplicates,
            "actual_class_count": self.actual_class_count,
            "boundaries": self._boundaries.tolist(),
            "fit_sample_count": self._fit_n,
        }


def _as_single_output_float(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"Fold-local target transform expects shape [n_samples], got {values.shape}"
        )
    return values
