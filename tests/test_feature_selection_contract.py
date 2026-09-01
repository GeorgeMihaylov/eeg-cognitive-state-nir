from __future__ import annotations

import numpy as np
import pytest

from cogstate.features.selection import FeatureSelector, SelectionConfig


@pytest.mark.parametrize("task_type", ["classification", "regression"])
@pytest.mark.parametrize("method", ["tree_importance", "mutual_info"])
def test_selector_supports_explicit_task_types_deterministically(
    task_type: str, method: str
) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 12))
    signal = X[:, 0] - 0.5 * X[:, 1]
    y = (signal > 0).astype(int) if task_type == "classification" else signal
    names = [f"EEG.f{i}" for i in range(X.shape[1])]
    config = SelectionConfig(
        method=method, top_k=5, random_state=42, task_type=task_type
    )
    first = FeatureSelector(config).fit(X, y, names)
    second = FeatureSelector(config).fit(X, y, names)
    assert first.result.selected_names == second.result.selected_names
    assert first.result.selected_indices == second.result.selected_indices
    assert len(first.result.selected_names) == 5
    np.testing.assert_allclose(first.transform(X), X[:, first.result.selected_indices])


def test_selector_default_remains_classification_and_transform_never_refits() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 8))
    y = (X[:, 0] > 0).astype(int)
    names = [f"f{i}" for i in range(8)]
    selector = FeatureSelector(SelectionConfig(top_k=4)).fit(X, y, names)
    selected = list(selector.result.selected_indices)
    selector.transform(np.full((3, 8), 1000.0))
    assert selector.config.task_type == "classification"
    assert list(selector.result.selected_indices) == selected


def test_selector_rejects_implicit_or_invalid_task_type() -> None:
    X = np.ones((10, 3))
    with pytest.raises(ValueError, match="task_type"):
        FeatureSelector(SelectionConfig(task_type="guess")).fit(
            X, np.arange(10), ["a", "b", "c"]
        )
