"""Resolve backend-neutral AutoML trials through the shared config resolver."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.bench_runner import benchmark_config_hash
from bench.experiments.preprocessing_ablation import resolve_trial_config

from .search_space import SearchSpaceSpec


def _single_dataset(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or len(datasets) != 1:
        raise ValueError("Initial AutoML track requires exactly one dataset")
    dataset = next(iter(datasets.values()))
    if not isinstance(dataset, dict):
        raise ValueError("The configured dataset must be a mapping")
    return dataset


def resolve_automl_trial_config(
    *,
    base_config: Mapping[str, Any],
    trial_parameters: Mapping[str, Any],
    search_space: SearchSpaceSpec,
    outer_fold: int,
    outer_train_subjects: Sequence[str],
    outer_test_subjects: Sequence[str],
    inner_splits: int,
    random_state: int,
    benchmark_runs_root: str | Path,
    max_epochs: int | None = None,
    max_windows: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Build one leakage-safe inner-search benchmark configuration."""
    search_space.validate_parameters(trial_parameters)
    if outer_fold < 1:
        raise ValueError("outer_fold must be positive")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2")
    train_subjects = sorted({str(value) for value in outer_train_subjects})
    test_subjects = sorted({str(value) for value in outer_test_subjects})
    if not train_subjects:
        raise ValueError("outer_train_subjects must not be empty")
    overlap = sorted(set(train_subjects).intersection(test_subjects))
    if overlap:
        raise ValueError(f"Outer train/test subjects overlap: {overlap}")

    protocol_config = deepcopy(dict(base_config))
    dataset = _single_dataset(protocol_config)
    dataset["include_subject_ids"] = train_subjects
    evaluation = protocol_config.setdefault("evaluation", {})
    evaluation.update({
        "protocol": "group_kfold_subject",
        "group_column": "subject_id",
        "n_splits": int(inner_splits),
        "random_state": int(random_state),
        "role": "inner_search",
        "outer_fold": int(outer_fold),
        "inner_split": "all",
    })
    evaluation.pop("folds", None)
    evaluation.pop("precomputed_fold_column", None)

    neutral_parameters = {
        "model.name": "torch_transformer",
        **dict(trial_parameters),
        "training.random_state": int(random_state),
    }
    if max_epochs is not None:
        neutral_parameters["training.max_epochs"] = int(max_epochs)
    if max_windows is not None:
        neutral_parameters["dataset.max_windows"] = int(max_windows)
    resolved = resolve_trial_config(protocol_config, neutral_parameters)
    config_hash = benchmark_config_hash(resolved)
    resolved["output_dir"] = str(
        Path(benchmark_runs_root) / config_hash[:20]
    )
    return resolved, config_hash


def resolve_outer_evaluation_config(
    *,
    base_config: Mapping[str, Any],
    trial_parameters: Mapping[str, Any],
    search_space: SearchSpaceSpec,
    outer_fold: int,
    random_state: int,
    benchmark_runs_root: str | Path,
    max_epochs: int | None = None,
    max_windows: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve the selected trial for one untouched outer-fold evaluation."""
    search_space.validate_parameters(trial_parameters)
    protocol_config = deepcopy(dict(base_config))
    dataset = _single_dataset(protocol_config)
    dataset.pop("include_subject_ids", None)
    evaluation = protocol_config.setdefault("evaluation", {})
    evaluation.update({
        "protocol": "group_kfold_subject",
        "group_column": "subject_id",
        "folds": [int(outer_fold)],
        "random_state": int(random_state),
        "role": "outer_evaluation",
        "outer_fold": int(outer_fold),
        "selected_on": "inner_group_kfold_subject",
    })
    neutral_parameters = {
        "model.name": "torch_transformer",
        **dict(trial_parameters),
        "training.random_state": int(random_state),
    }
    if max_epochs is not None:
        neutral_parameters["training.max_epochs"] = int(max_epochs)
    if max_windows is not None:
        neutral_parameters["dataset.max_windows"] = int(max_windows)
    resolved = resolve_trial_config(protocol_config, neutral_parameters)
    config_hash = benchmark_config_hash(resolved)
    resolved["output_dir"] = str(
        Path(benchmark_runs_root) / config_hash[:20]
    )
    return resolved, config_hash
