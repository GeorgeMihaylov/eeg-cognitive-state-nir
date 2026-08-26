"""Staged application search over an explicit inner split only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .adaptation import adapt_candidate, score_candidate
from .protocols import MetricFn, ModelAdapter
from .registry import AdaptationMethod, CandidateSpec
from .splits import InnerSplit


@dataclass(frozen=True)
class SearchResult:
    candidate: CandidateSpec
    inner_metric: float
    fitted: ModelAdapter


def staged_search(
    shortlist: Sequence[CandidateSpec],
    *,
    pretrained_adapters: Mapping[str, ModelAdapter],
    input_shape: Sequence[int],
    num_outputs: int,
    inner_split: InnerSplit,
    metric_fn: MetricFn,
    higher_is_better: bool = True,
    probe_fit_kwargs: Optional[Mapping[str, object]] = None,
    full_fit_kwargs: Optional[Mapping[str, object]] = None,
) -> SearchResult:
    if not shortlist:
        raise ValueError("shortlist must be non-empty")
    probe_kwargs = dict(probe_fit_kwargs or {})
    full_kwargs = dict(full_fit_kwargs or {})

    def _run(candidate: CandidateSpec, budget_kwargs: Mapping[str, object]):
        kwargs = (
            budget_kwargs
            if candidate.adaptation_method
            in (AdaptationMethod.HEAD_ONLY, AdaptationMethod.FULL_FINETUNE)
            else {}
        )
        fitted = adapt_candidate(
            candidate,
            pretrained_global_adapter=pretrained_adapters[candidate.name],
            input_shape=input_shape,
            num_outputs=num_outputs,
            inner_split=inner_split,
            fit_kwargs=kwargs,
        )
        metric = score_candidate(
            fitted, inner_split.inner_val_X, inner_split.inner_val_y, metric_fn
        )
        return metric, fitted

    probe_results = [(c, *_run(c, probe_kwargs)) for c in shortlist]
    probe_results.sort(key=lambda item: item[1], reverse=higher_is_better)

    survivors = probe_results[: max(1, (len(probe_results) + 1) // 2)]
    final_results = [(c, *_run(c, full_kwargs)) for c, _, _ in survivors]
    final_results.sort(key=lambda item: item[1], reverse=higher_is_better)

    best_candidate, best_metric, best_fitted = final_results[0]
    return SearchResult(candidate=best_candidate, inner_metric=best_metric, fitted=best_fitted)
