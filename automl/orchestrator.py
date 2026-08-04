from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .adaptation import adapt_candidate, score_candidate
from .meta_features import SubjectMetaFeatures
from .portfolio import PortfolioRecord, PortfolioStore, new_run_id, now
from .protocols import MetricFn, ModelAdapter, ShadowDiagnosticRunner
from .registry import CandidateRegistry
from .search import staged_search
from .splits import InnerSplit, build_inner_split


class PersonalizedAutoML:
    """Selects and adapts a model for one subject at a time.

    `run_for_subject` is the entry point:
      1. build a shortlist (+ any eligible shadow candidates) from the
         registry, using target-free meta-features,
      2. if there's more than one selectable candidate and enough
         calibration data, run a leakage-safe inner (successive-halving)
         search to pick a winner; otherwise use the sole/default one,
      3. re-adapt the winner on the full calibration data,
      4. score it once on the untouched held-out data,
      5. run any registered shadow diagnostics (never selectable) and
         log everything to the portfolio.
    """

    def __init__(
        self,
        registry: CandidateRegistry,
        metric_fn: MetricFn,
        *,
        portfolio: Optional[PortfolioStore] = None,
        higher_is_better: bool = True,
        shadow_runners: Optional[Mapping[str, ShadowDiagnosticRunner]] = None,
        min_inner_search_samples: int = 4,
    ) -> None:
        self.registry = registry
        self.metric_fn = metric_fn
        self.portfolio = portfolio or PortfolioStore()
        self.higher_is_better = higher_is_better
        self.shadow_runners = dict(shadow_runners or {})
        self.min_inner_search_samples = min_inner_search_samples

    def run_for_subject(
        self,
        *,
        meta: SubjectMetaFeatures,
        calibration_X: np.ndarray,
        calibration_y: np.ndarray,
        holdout_X: np.ndarray,
        holdout_y: np.ndarray,
        pretrained_adapters: Mapping[str, ModelAdapter],
        input_shape: Sequence[int],
        num_outputs: int,
        shadow_payload: Any = None,
    ) -> PortfolioRecord:
        selectable, shadow = self.registry.shortlist(meta)

        shadow_metrics: Dict[str, Any] = {}
        for candidate in shadow:
            runner = self.shadow_runners.get(candidate.name)
            if runner is None:
                continue
            try:
                shadow_metrics[candidate.name] = dict(runner(candidate, shadow_payload))
            except Exception as exc:  # diagnostics must never break selection
                shadow_metrics[candidate.name] = {"error": str(exc)}

        if len(selectable) == 1 or calibration_X.shape[0] < self.min_inner_search_samples:
            winner = selectable[0]
            inner_metric = float("nan")
        else:
            inner_split = build_inner_split(
                calibration_X, calibration_y, min_samples=self.min_inner_search_samples
            )
            result = staged_search(
                selectable,
                pretrained_adapters=pretrained_adapters,
                input_shape=input_shape,
                num_outputs=num_outputs,
                inner_split=inner_split,
                metric_fn=self.metric_fn,
                higher_is_better=self.higher_is_better,
            )
            winner = result.candidate
            inner_metric = result.inner_metric

        final_split = InnerSplit(
            inner_train_X=calibration_X,
            inner_train_y=calibration_y,
            inner_val_X=holdout_X,
            inner_val_y=holdout_y,
        )
        fitted = adapt_candidate(
            winner,
            pretrained_global_adapter=pretrained_adapters[winner.name],
            input_shape=input_shape,
            num_outputs=num_outputs,
            inner_split=final_split,
        )
        holdout_metric = score_candidate(fitted, holdout_X, holdout_y, self.metric_fn)

        record = PortfolioRecord(
            run_id=new_run_id(),
            subject_id=meta.subject_id,
            meta=meta,
            selected_candidate=winner.name,
            selected_status=winner.status.value,
            inner_metric=inner_metric,
            holdout_metric=holdout_metric,
            shadow_metrics=shadow_metrics,
            timestamp=now(),
        )
        self.portfolio.append(record)
        return record
