"""Train-only feature selection with explicit classification/regression semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


SUPPORTED_TASK_TYPES = frozenset({"classification", "regression"})
SUPPORTED_METHODS = frozenset({"tree_importance", "mutual_info"})


@dataclass(frozen=True)
class SelectionConfig:
    task_type: str = "classification"
    method: str = "tree_importance"
    top_k: int | None = 50
    importance_threshold: float = 0.0
    redundancy_corr_threshold: float | None = 0.95
    random_state: int = 42
    n_estimators: int = 300

    def __post_init__(self) -> None:
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(SUPPORTED_TASK_TYPES)}, "
                f"got {self.task_type!r}"
            )
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {sorted(SUPPORTED_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.top_k is not None and int(self.top_k) <= 0:
            raise ValueError("top_k must be positive or None")
        if not np.isfinite(self.importance_threshold):
            raise ValueError("importance_threshold must be finite")
        if self.redundancy_corr_threshold is not None and not (
            0.0 < float(self.redundancy_corr_threshold) <= 1.0
        ):
            raise ValueError("redundancy_corr_threshold must be in (0, 1] or None")
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive")


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    selected_feature_names: tuple[str, ...]
    dropped_feature_names: tuple[str, ...]
    importance_scores: tuple[float, ...]

    @property
    def selected_names(self) -> list[str]:
        """Compatibility alias for the source implementation."""
        return list(self.selected_feature_names)

    @property
    def importances(self) -> np.ndarray:
        return np.asarray(self.importance_scores, dtype=float)

    @property
    def dropped_redundant(self) -> list[str]:
        return list(self.dropped_feature_names)

    def to_manifest(self) -> dict[str, object]:
        return {
            "selected_indices": list(self.selected_indices),
            "selected_feature_names": list(self.selected_feature_names),
            "dropped_feature_names": list(self.dropped_feature_names),
            "importance_scores": list(self.importance_scores),
        }


def _validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y)
    names = tuple(str(name) for name in feature_names)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("X must have shape [n_samples, n_features]")
    if not np.isfinite(features).all():
        raise ValueError("X contains NaN or Inf")
    if targets.ndim != 1 or len(targets) != len(features):
        raise ValueError("y must be one-dimensional and aligned with X")
    if len(names) != features.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and aligned with X columns")
    if task_type == "regression":
        targets = np.asarray(targets, dtype=float)
        if not np.isfinite(targets).all():
            raise ValueError("regression y contains NaN or Inf")
    elif targets.dtype.kind in "fc" and not np.isfinite(targets).all():
        raise ValueError("classification y contains NaN or Inf")
    elif targets.dtype.kind == "O" and any(value is None for value in targets):
        raise ValueError("classification y contains missing values")
    return features, targets, names


def _redundancy_keep_indices(
    X: np.ndarray,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if threshold is None or X.shape[1] == 1:
        keep = np.arange(X.shape[1], dtype=np.int64)
        return keep, np.empty(0, dtype=np.int64)
    centered = X - np.mean(X, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(np.square(centered), axis=0))
    normalized = np.zeros_like(centered)
    valid = norms > np.finfo(float).eps
    normalized[:, valid] = centered[:, valid] / norms[valid]
    correlation = normalized.T @ normalized
    dropped: set[int] = set()
    for first in range(X.shape[1]):
        if first in dropped:
            continue
        for second in range(first + 1, X.shape[1]):
            if second not in dropped and abs(correlation[first, second]) > threshold:
                dropped.add(second)
    dropped_indices = np.asarray(sorted(dropped), dtype=np.int64)
    keep_indices = np.asarray(
        [index for index in range(X.shape[1]) if index not in dropped],
        dtype=np.int64,
    )
    return keep_indices, dropped_indices


def _importance_scores(
    X: np.ndarray,
    y: np.ndarray,
    config: SelectionConfig,
) -> np.ndarray:
    if config.method == "tree_importance":
        estimator_class = (
            RandomForestClassifier
            if config.task_type == "classification"
            else RandomForestRegressor
        )
        estimator = estimator_class(
            n_estimators=int(config.n_estimators),
            random_state=int(config.random_state),
            n_jobs=1,
        )
        estimator.fit(X, y)
        return np.asarray(estimator.feature_importances_, dtype=float)
    if config.task_type == "classification":
        return np.asarray(
            mutual_info_classif(X, y, random_state=config.random_state),
            dtype=float,
        )
    return np.asarray(
        mutual_info_regression(X, y, random_state=config.random_state),
        dtype=float,
    )


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    config: SelectionConfig,
) -> SelectionResult:
    """Fit selection on the supplied train partition only.

    This reusable function deliberately knows nothing about benchmark folds.
    Callers are responsible for passing only the authorized training partition.
    """
    features, targets, names = _validate_training_data(
        X, y, feature_names, config.task_type
    )
    keep, dropped = _redundancy_keep_indices(
        features, config.redundancy_corr_threshold
    )
    filtered_importance = _importance_scores(features[:, keep], targets, config)
    order = np.lexsort((keep, -filtered_importance))
    if config.top_k is not None:
        order = order[: min(int(config.top_k), len(order))]
    else:
        order = order[filtered_importance[order] > config.importance_threshold]
    if len(order) == 0:
        raise ValueError("feature selection removed every feature")
    selected = keep[order]
    full_importance = np.zeros(features.shape[1], dtype=float)
    full_importance[keep] = filtered_importance
    return SelectionResult(
        selected_indices=tuple(int(index) for index in selected),
        selected_feature_names=tuple(names[index] for index in selected),
        dropped_feature_names=tuple(names[index] for index in dropped),
        importance_scores=tuple(float(value) for value in full_importance),
    )


class FeatureSelector:
    """Sklearn-like reusable selector; ``fit`` must receive train data only."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.result_: SelectionResult | None = None

    @property
    def result(self) -> SelectionResult | None:
        return self.result_

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return self._require_result().selected_indices

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return self._require_result().selected_feature_names

    @property
    def dropped_feature_names(self) -> tuple[str, ...]:
        return self._require_result().dropped_feature_names

    @property
    def importance_scores(self) -> tuple[float, ...]:
        return self._require_result().importance_scores

    def _require_result(self) -> SelectionResult:
        if self.result_ is None:
            raise RuntimeError("FeatureSelector must be fit before transform")
        return self.result_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> "FeatureSelector":
        self.result_ = select_features(X, y, feature_names, self.config)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = self._require_result()
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(result.importance_scores):
            raise ValueError("X shape is incompatible with the fitted selector")
        if not np.isfinite(features).all():
            raise ValueError("X contains NaN or Inf")
        return np.ascontiguousarray(features[:, result.selected_indices])

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)
