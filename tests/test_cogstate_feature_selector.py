from __future__ import annotations

import numpy as np
import pytest

from cogstate.features.selection import FeatureSelector, SelectionConfig


def _data(task_type: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=120)
    X = np.column_stack(
        [signal, signal, rng.normal(size=120), rng.normal(size=120), rng.normal(size=120)]
    )
    if task_type == "classification":
        y = (signal > 0).astype(np.int64)
    else:
        y = 2.0 * signal + rng.normal(scale=0.05, size=len(signal))
    return X, y, [f"feature_{index}" for index in range(X.shape[1])]


@pytest.mark.parametrize("task_type", ["classification", "regression"])
@pytest.mark.parametrize("method", ["tree_importance", "mutual_info"])
def test_feature_selector_supports_task_and_method(
    task_type: str, method: str
) -> None:
    X, y, names = _data(task_type)
    config = SelectionConfig(
        task_type=task_type,
        method=method,
        top_k=2,
        n_estimators=25,
        random_state=42,
    )
    first = FeatureSelector(config).fit(X, y, names)
    second = FeatureSelector(config).fit(X, y, names)

    assert first.transform(X).shape == (len(X), 2)
    assert first.selected_indices == second.selected_indices
    assert first.selected_feature_names == second.selected_feature_names
    assert "feature_1" in first.dropped_feature_names
    assert len(first.importance_scores) == X.shape[1]
    assert first.result.to_manifest()["selected_indices"] == list(first.selected_indices)


def test_feature_selector_transform_reuses_frozen_indices() -> None:
    X, y, names = _data("classification")
    selector = FeatureSelector(
        SelectionConfig(task_type="classification", top_k=2, n_estimators=20)
    )
    train_transformed = selector.fit_transform(X, y, names)
    test = X[::-1].copy()
    expected = test[:, selector.selected_indices]
    np.testing.assert_array_equal(selector.transform(test), expected)
    assert train_transformed.shape == expected.shape


def test_feature_selector_threshold_mode_and_names() -> None:
    X, y, names = _data("regression")
    selector = FeatureSelector(
        SelectionConfig(
            task_type="regression",
            top_k=None,
            importance_threshold=0.01,
            n_estimators=20,
        )
    ).fit(X, y, names)
    assert selector.selected_feature_names
    assert all(name in names for name in selector.selected_feature_names)


def test_feature_selector_requires_fit() -> None:
    selector = FeatureSelector(SelectionConfig(task_type="classification"))
    with pytest.raises(RuntimeError, match="must be fit"):
        selector.transform(np.zeros((2, 3)))


def test_feature_selector_rejects_invalid_task_type() -> None:
    with pytest.raises(ValueError, match="task_type"):
        SelectionConfig(task_type="auto")


def test_feature_selector_rejects_nonfinite_data() -> None:
    X, y, names = _data("regression")
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        FeatureSelector(SelectionConfig(task_type="regression")).fit(X, y, names)
