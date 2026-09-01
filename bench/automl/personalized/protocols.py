"""Framework-neutral protocols for application candidate adapters."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal contract any fitted-or-fittable model must satisfy.

    sklearn-style estimators already satisfy this. Torch-based models
    need a thin wrapper (see `examples/eeg_project_bindings.py`) that
    also implements `freeze_all_but_head` / `unfreeze_all` if they want
    to support HEAD_ONLY adaptation.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ModelAdapter": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


ModelBuilder = Callable[[Sequence[int], int, Mapping[str, Any]], ModelAdapter]
"""(input_shape, num_outputs, params) -> a fresh or pretrained ModelAdapter.

`input_shape` and `num_outputs` are opaque to this package — they are
whatever your builders expect (a flat feature count, a
(sequence_length, n_features) tuple, a raw window shape, etc.).
"""

MetricFn = Callable[[np.ndarray, np.ndarray], float]
"""(y_true, y_pred) -> scalar score. Any task-appropriate metric works:
macro F1, balanced accuracy, R², Spearman, MAE, ..."""

ConditionFn = Callable[[Any], bool]
"""Arbitrary gating predicate over a SubjectMetaFeatures instance.

Typed as Callable[[Any], bool] rather than importing SubjectMetaFeatures
to avoid a circular import; in practice this always receives a
`meta_features.SubjectMetaFeatures`.
"""

ShadowDiagnosticRunner = Callable[[Any, Any], Mapping[str, Any]]
"""(CandidateSpec, arbitrary payload) -> diagnostic metrics dict.

Used for methods that are logged but never selectable — e.g. a domain
adaptation technique still under confirmatory review. The AutoML core
never imports a concrete implementation of this; callers register one
per candidate name via `PersonalizedAutoML(shadow_runners=...)`.
"""
