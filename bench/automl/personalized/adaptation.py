"""Application-only adaptation helpers for an already selected candidate."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .protocols import MetricFn, ModelAdapter
from .registry import AdaptationMethod, CandidateSpec
from .splits import InnerSplit


def adapt_candidate(
    candidate: CandidateSpec,
    *,
    pretrained_global_adapter: ModelAdapter,
    input_shape: Sequence[int],
    num_outputs: int,
    inner_split: Optional[InnerSplit],
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> ModelAdapter:

    fit_kwargs = dict(fit_kwargs or {})
    method = candidate.adaptation_method

    if method is AdaptationMethod.REFIT:
        estimator = candidate.build(input_shape, num_outputs)
        if inner_split is not None:
            estimator.fit(inner_split.inner_train_X, inner_split.inner_train_y, **fit_kwargs)
        return estimator

    if method is AdaptationMethod.ZERO_SHOT:
        return copy.deepcopy(pretrained_global_adapter)

    adapter = copy.deepcopy(pretrained_global_adapter)
    if inner_split is None:
        return adapter

    if method is AdaptationMethod.HEAD_ONLY:
        _require_freeze_support(adapter)
        adapter.freeze_all_but_head()
        adapter.fit(inner_split.inner_train_X, inner_split.inner_train_y, **fit_kwargs)
        adapter.unfreeze_all()
        return adapter

    if method is AdaptationMethod.FULL_FINETUNE:
        if hasattr(adapter, "unfreeze_all"):
            adapter.unfreeze_all()
        adapter.fit(inner_split.inner_train_X, inner_split.inner_train_y, **fit_kwargs)
        return adapter

    if method is AdaptationMethod.SHADOW_ONLY:
        raise RuntimeError(
            f"Candidate {candidate.name!r} is SHADOW_ONLY and must be run "
            "through a registered ShadowDiagnosticRunner "
            "(see PersonalizedAutoML(shadow_runners=...)), not adapt_candidate()."
        )

    raise ValueError(f"Unhandled adaptation method: {method}")


def _require_freeze_support(adapter: ModelAdapter) -> None:
    if not (hasattr(adapter, "freeze_all_but_head") and hasattr(adapter, "unfreeze_all")):
        raise ValueError(
            f"{type(adapter).__name__} does not implement "
            "freeze_all_but_head()/unfreeze_all(); HEAD_ONLY adaptation "
            "requires an adapter that declares its own output head "
            "(see examples/eeg_project_bindings.py for a torch reference)."
        )


def score_candidate(
    fitted: ModelAdapter,
    val_X: np.ndarray,
    val_y: np.ndarray,
    metric_fn: MetricFn,
) -> float:
    predictions = fitted.predict(val_X)
    return float(metric_fn(val_y, predictions))
