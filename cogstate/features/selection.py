""" Отбор
наиболее значимых выполняется автоматически через оценку важности
признаков по обученному ансамблевому классификатору (LightGBM /
Random Forest — те же модели, что используются в 10.2.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


@dataclass
class SelectionConfig:
    method: str = "tree_importance"     # "tree_importance" | "mutual_info"
    top_k: Optional[int] = 50           # None -> отбор по порогу, а не по числу
    importance_threshold: float = 0.0   # минимальная важность признака, если top_k не задан
    redundancy_corr_threshold: float = 0.95  # порог для отбраковки дублирующих признаков
    random_state: int = 42


@dataclass
class SelectionResult:
    selected_indices: List[int]
    selected_names: List[str]
    importances: np.ndarray             # важности всех исходных признаков (для отчётности)
    dropped_redundant: List[str]        # признаки, отброшенные как избыточные (до отбора по важности)


def _drop_redundant_features(
    X: np.ndarray, feature_names: List[str], threshold: float
) -> tuple[np.ndarray, List[str], List[str]]:
    corr_matrix = np.corrcoef(X, rowvar=False)
    n_features = X.shape[1]
    to_drop = set()

    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
            if j in to_drop:
                continue
            if np.abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    keep = [i for i in range(n_features) if i not in to_drop]
    dropped_names = [feature_names[i] for i in sorted(to_drop)]
    return X[:, keep], [feature_names[i] for i in keep], dropped_names


def _compute_tree_importance(X: np.ndarray, y: np.ndarray, config: SelectionConfig) -> np.ndarray:
    """
    Важность признаков через RandomForest (mean decrease in impurity).
    Может (и должен) быть заменён на предобученную модель — интерфейс функции не меняется.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model.feature_importances_


def _compute_mutual_info_importance(X: np.ndarray, y: np.ndarray, config: SelectionConfig) -> np.ndarray:
    """Альтернатива важности по ансамблю — оценка взаимной информации между признаком и меткой."""
    return mutual_info_classif(X, y, random_state=config.random_state)


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: Optional[SelectionConfig] = None,
) -> SelectionResult:
    config = config or SelectionConfig()

    X_filtered, names_filtered, dropped = _drop_redundant_features(
        X, feature_names, config.redundancy_corr_threshold
    )

    if config.method == "tree_importance":
        importances = _compute_tree_importance(X_filtered, y, config)
    elif config.method == "mutual_info":
        importances = _compute_mutual_info_importance(X_filtered, y, config)
    else:
        raise ValueError(f"Неизвестный метод отбора признаков: {config.method}")

    order = np.argsort(importances)[::-1]

    if config.top_k is not None:
        selected_order = order[: config.top_k]
    else:
        selected_order = order[importances[order] > config.importance_threshold]

    selected_names = [names_filtered[i] for i in selected_order]
    selected_indices = [feature_names.index(name) for name in selected_names]

    full_importances = np.zeros(len(feature_names))
    for name, importance in zip(names_filtered, importances):
        full_importances[feature_names.index(name)] = importance

    return SelectionResult(
        selected_indices=selected_indices,
        selected_names=selected_names,
        importances=full_importances,
        dropped_redundant=dropped,
    )


class FeatureSelector:
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.result: Optional[SelectionResult] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> "FeatureSelector":
        self.result = select_features(X, y, feature_names, self.config)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("FeatureSelector не обучен — вызовите fit() перед transform()")
        return X[:, self.result.selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)
