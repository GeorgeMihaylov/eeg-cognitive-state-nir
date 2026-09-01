"""Canonical benchmark objective for leakage-safe inner model selection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bench.bench_runner import BenchmarkRunner, CompletedBenchmarkRun

from .search_space import SearchSpaceSpec
from .trial_resolver import resolve_automl_trial_config


@dataclass(frozen=True)
class AutoMLTrialResult:
    study_name: str
    trial_number: int
    trial_parameters: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    resolved_config_hash: str
    outer_fold: int
    inner_split: str
    objective_value: float | None
    secondary_metrics: Mapping[str, float] = field(default_factory=dict)
    state: str = "COMPLETE"
    runtime_seconds: float = 0.0
    benchmark_run_reference: Mapping[str, Any] | None = None
    failure_reason: str | None = None
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "trial_number": int(self.trial_number),
            "trial_parameters": dict(self.trial_parameters),
            "resolved_config": dict(self.resolved_config),
            "resolved_config_hash": self.resolved_config_hash,
            "outer_fold": int(self.outer_fold),
            "inner_split": self.inner_split,
            "objective_value": self.objective_value,
            "secondary_metrics": dict(self.secondary_metrics),
            "state": self.state,
            "runtime_seconds": float(self.runtime_seconds),
            "benchmark_run_reference": (
                None
                if self.benchmark_run_reference is None
                else dict(self.benchmark_run_reference)
            ),
            "failure_reason": self.failure_reason,
            "reused": bool(self.reused),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AutoMLTrialResult":
        return cls(**dict(values))


def extract_group_result(
    results: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    dataset_name = next(iter(config["datasets"]))
    task_name = str(config["tasks"][0])
    model_name = next(iter(config["models"]))
    try:
        return results[dataset_name]["models"][task_name][model_name][
            "group_kfold_subject"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Standard benchmark result is missing the configured GroupKFold result"
        ) from exc


def _load_completed_metrics(completed: CompletedBenchmarkRun) -> Mapping[str, Any]:
    metrics_path = completed.run_directory / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


class NestedBenchmarkObjective:
    """Resolve and execute one trial without access to outer-test labels."""

    def __init__(
        self,
        *,
        study_name: str,
        base_config: Mapping[str, Any],
        search_space: SearchSpaceSpec,
        outer_fold: int,
        outer_train_subjects: Sequence[str],
        outer_test_subjects: Sequence[str],
        inner_splits: int,
        random_state: int,
        benchmark_runs_root: str | Path,
        max_epochs: int | None = None,
        max_windows: int | None = None,
        runner_factory: Callable[[dict[str, Any]], Any] = BenchmarkRunner,
        completed_run_finder: Callable[..., CompletedBenchmarkRun | None] = (
            BenchmarkRunner.find_completed_run
        ),
    ) -> None:
        self.study_name = study_name
        self.base_config = dict(base_config)
        self.search_space = search_space
        self.outer_fold = int(outer_fold)
        self.outer_train_subjects = tuple(str(v) for v in outer_train_subjects)
        self.outer_test_subjects = tuple(str(v) for v in outer_test_subjects)
        self.inner_splits = int(inner_splits)
        self.random_state = int(random_state)
        self.benchmark_runs_root = Path(benchmark_runs_root)
        self.max_epochs = max_epochs
        self.max_windows = max_windows
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder

    def resolve(
        self,
        trial_parameters: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return resolve_automl_trial_config(
            base_config=self.base_config,
            trial_parameters=trial_parameters,
            search_space=self.search_space,
            outer_fold=self.outer_fold,
            outer_train_subjects=self.outer_train_subjects,
            outer_test_subjects=self.outer_test_subjects,
            inner_splits=self.inner_splits,
            random_state=self.random_state,
            benchmark_runs_root=self.benchmark_runs_root,
            max_epochs=self.max_epochs,
            max_windows=self.max_windows,
        )

    def execute(
        self,
        trial_number: int,
        trial_parameters: Mapping[str, Any],
    ) -> AutoMLTrialResult:
        started = time.perf_counter()
        resolved: dict[str, Any] = {}
        config_hash = ""
        try:
            resolved, config_hash = self.resolve(trial_parameters)
            dataset = next(iter(resolved["datasets"].values()))
            included = set(map(str, dataset["include_subject_ids"]))
            forbidden = included.intersection(self.outer_test_subjects)
            if forbidden:
                raise RuntimeError(
                    f"Outer-test subjects leaked into an inner trial: {sorted(forbidden)}"
                )
            completed = self.completed_run_finder(
                resolved,
                search_directories=[resolved["output_dir"]],
            )
            reused = completed is not None
            if completed is None:
                runner = self.runner_factory(resolved)
                runner.run()
                completed = runner.completed_run()
                results = runner.results
            else:
                results = _load_completed_metrics(completed)
            group = extract_group_result(results, resolved)
            aggregated = group.get("aggregated", {})
            objective_value = aggregated.get("balanced_accuracy_mean")
            if objective_value is None:
                raise ValueError(
                    "Standard benchmark result has no balanced_accuracy_mean"
                )
            secondary = {
                "macro_f1": float(aggregated["macro_f1_mean"]),
            }
            if "balanced_accuracy_std" in aggregated:
                secondary["balanced_accuracy_std"] = float(
                    aggregated["balanced_accuracy_std"]
                )
            if "macro_f1_std" in aggregated:
                secondary["macro_f1_std"] = float(
                    aggregated["macro_f1_std"]
                )
            fold_names = sorted(group.get("folds", {}))
            return AutoMLTrialResult(
                study_name=self.study_name,
                trial_number=trial_number,
                trial_parameters=dict(trial_parameters),
                resolved_config=resolved,
                resolved_config_hash=config_hash,
                outer_fold=self.outer_fold,
                inner_split=",".join(fold_names),
                objective_value=float(objective_value),
                secondary_metrics=secondary,
                state="COMPLETE",
                runtime_seconds=time.perf_counter() - started,
                benchmark_run_reference=completed.to_dict(),
                reused=reused,
            )
        except Exception as exc:
            return AutoMLTrialResult(
                study_name=self.study_name,
                trial_number=trial_number,
                trial_parameters=dict(trial_parameters),
                resolved_config=resolved,
                resolved_config_hash=config_hash,
                outer_fold=self.outer_fold,
                inner_split=f"aggregate_{self.inner_splits}fold",
                objective_value=None,
                state="FAIL",
                runtime_seconds=time.perf_counter() - started,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
