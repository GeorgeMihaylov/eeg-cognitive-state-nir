"""Contracts for the personalized AutoML namespace after package migration."""

from __future__ import annotations

import numpy as np

from bench.automl import personalized, scientific
from bench.automl.personalized.bindings import build_eeg_registry
from bench.automl.personalized.splits import build_inner_split


def test_scientific_and_personalized_automl_are_distinct_namespaces() -> None:
    assert scientific.__name__ == "bench.automl.scientific"
    assert personalized.__name__ == "bench.automl.personalized"
    assert scientific is not personalized


def test_personalized_registry_uses_canonical_project_bindings() -> None:
    registry = build_eeg_registry()
    names = {candidate.name for candidate in registry}
    assert {"random_forest", "lstm_head_only", "transformer_head_only"} <= names


def test_personalized_inner_split_remains_chronological_and_disjoint() -> None:
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.arange(8)
    split = build_inner_split(X, y, inner_train_frac=0.75)
    np.testing.assert_array_equal(split.inner_train_y, y[:6])
    np.testing.assert_array_equal(split.inner_val_y, y[6:])
    assert not set(split.inner_train_y) & set(split.inner_val_y)
