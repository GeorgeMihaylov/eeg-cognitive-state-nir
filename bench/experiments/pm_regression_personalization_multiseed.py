"""Three-seed orchestration for multi-output PM personalization.

This module only resolves seed-specific configurations, invokes the existing
single-seed experiment, validates compatibility, and aggregates artifacts.
Neural fitting remains in BenchmarkRunner and TorchClassificationAdapter.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.experiments.pm_regression_personalization import (
    CANONICAL_TARGETS,
    PMRegressionPersonalizationExperiment,
    _bootstrap_interval,
    _canonical_hash,
    _file_sha256,
    _repo_path,
    _write_json,
    metric_gain,
)


SCHEMA_VERSION = "pm-regression-personalization-multiseed-v1"
METHODS = ("zero_shot", "head_only", "full_model")
METRICS = (
    "macro_mae",
    "macro_rmse",
    "macro_r2",
    "macro_spearman",
    "macro_abs_bias",
)
SPLIT_HASH_COLUMNS = (
    "outer_train_subject_hash",
    "inner_train_subject_hash",
    "inner_validation_subject_hash",
    "calibration_sample_hash",
    "adaptation_train_sample_hash",
    "adaptation_validation_sample_hash",
    "evaluation_sample_hash",
    "preprocessor_hash",
)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("pm_regression_personalization.py"),
        _repo_path("model_zoo/DL/adapter.py"),
        _repo_path("model_zoo/DL/mlp.py"),
        _repo_path("bench/validation/metrics.py"),
    ):
        digest.update(str(path.relative_to(_repo_path("."))).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_pm_multiseed_spec(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Multiseed PM config not found: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    missing = sorted(
        {"experiment", "base_template", "targets", "calibration"} - set(document)
    )
    if missing:
        raise ValueError(f"Multiseed PM config is missing: {missing}")
    experiment = document["experiment"]
    if experiment.get("type") != "pm_regression_personalization_multiseed":
        raise ValueError(
            "experiment.type must be "
            "'pm_regression_personalization_multiseed'"
        )
    split_seed = int(experiment.get("split_seed", -1))
    model_seeds = tuple(int(value) for value in experiment.get("model_seeds", ()))
    if split_seed != 42:
        raise ValueError("split_seed must remain 42")
    if model_seeds not in {(7, 42, 2026), (7, 2026)}:
        raise ValueError(
            "model_seeds must be [7, 42, 2026], or [7, 2026] for smoke"
        )
    if tuple(document["targets"]) != CANONICAL_TARGETS:
        raise ValueError("Canonical seven-target order changed")
    calibration = document["calibration"]
    if float(calibration.get("maximum_calibration_fraction", -1)) != 0.20:
        raise ValueError("Only the 20% calibration budget is allowed")
    if tuple(calibration.get("methods", ())) != METHODS:
        raise ValueError(f"calibration.methods must be {list(METHODS)}")
    if not bool(experiment.get("require_cuda", True)):
        raise ValueError("Multiseed PM personalization requires CUDA")
    return document


def resolve_seed_base_config(
    template: Mapping[str, Any],
    *,
    model_seed: int,
    split_seed: int,
    output_dir: str | Path,
    maximum_epochs: Optional[int] = None,
    outer_folds: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Change only model RNG while fixing every split RNG."""
    config = deepcopy(dict(template))
    config["output_dir"] = str(output_dir)
    for section in ("validation", "evaluation", "task_config"):
        config.setdefault(section, {})["random_state"] = int(split_seed)
    if outer_folds is not None:
        config.setdefault("evaluation", {})["folds"] = [
            int(value) for value in outer_folds
        ]
    for model_config in config["models"].values():
        params = model_config.setdefault("params", {})
        params["random_state"] = int(model_seed)
        params["device"] = "cuda"
        if maximum_epochs is not None:
            params["max_epochs"] = int(maximum_epochs)
            params["early_stopping_patience"] = min(
                int(params.get("early_stopping_patience", maximum_epochs)),
                int(maximum_epochs),
            )
    return config


def resolve_seed_personalization_config(
    document: Mapping[str, Any],
    *,
    model_seed: int,
    split_seed: int,
    base_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": (
                f"{document['experiment']['name']}_seed_{int(model_seed)}"
            ),
            "type": "pm_regression_personalization",
            "output_dir": str(output_dir),
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "require_cuda": True,
            "resume": True,
        },
        "base_run": {
            "config_path": str(base_config_path),
            "dataset": str(
                document["base_template"].get(
                    "dataset", "emotiv_pm_regression"
                )
            ),
            "task": "performance_metrics_regression",
            "model": str(
                document["base_template"].get("model", "torch_mlp")
            ),
            "train_if_missing": True,
        },
        "targets": list(CANONICAL_TARGETS),
        "calibration": deepcopy(dict(document["calibration"])),
        "statistics": deepcopy(dict(document.get("statistics", {}))),
    }


def _metric_summary(
    frame: pd.DataFrame,
    *,
    grouping: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(grouping), sort=True, dropna=False)
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouping, keys))
        for metric in METRICS:
            after = pd.to_numeric(group[f"{metric}_after"], errors="coerce")
            gains = pd.to_numeric(group[f"{metric}_gain"], errors="coerce")
            after = after[np.isfinite(after)]
            gains = gains[np.isfinite(gains)]
            low, high = _bootstrap_interval(
                gains,
                samples=bootstrap_samples,
                random_state=bootstrap_seed,
            )
            rows.append({
                **identity,
                "metric": metric,
                "n_subjects": int(group["subject_id"].nunique()),
                "defined_subjects": int(len(after)),
                "mean": float(after.mean()) if len(after) else np.nan,
                "median": float(after.median()) if len(after) else np.nan,
                "std": float(after.std(ddof=1)) if len(after) > 1 else np.nan,
                "q25": float(after.quantile(0.25)) if len(after) else np.nan,
                "q75": float(after.quantile(0.75)) if len(after) else np.nan,
                "min": float(after.min()) if len(after) else np.nan,
                "max": float(after.max()) if len(after) else np.nan,
                "mean_gain": float(gains.mean()) if len(gains) else np.nan,
                "median_gain": float(gains.median()) if len(gains) else np.nan,
                "positive_fraction": (
                    float(np.mean(gains > 0)) if len(gains) else np.nan
                ),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_resamples": int(bootstrap_samples),
            })
    return pd.DataFrame(rows)


def build_multiseed_aggregates(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average seeds within subject before any multiseed inference."""
    expected = set(map(int, model_seeds))
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ].copy()
    per_seed = _metric_summary(
        completed,
        grouping=("model_seed", "method"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    counts = completed.groupby(["subject_id", "method"])[
        "model_seed"
    ].agg(lambda values: set(map(int, values)))
    complete_keys = set(counts.loc[counts.map(lambda value: value == expected)].index)
    complete = completed.loc[[
        (str(row.subject_id), str(row.method)) in complete_keys
        for row in completed.itertuples()
    ]].copy()
    metric_columns = [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("before", "after", "gain")
    ]
    target_columns = [
        f"{target}_{metric}_{suffix}"
        for target in CANONICAL_TARGETS
        for metric in (
            "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
        )
        for suffix in ("before", "after", "gain")
    ]
    subject_seed_means = (
        complete.groupby(
            ["subject_id", "source", "outer_fold", "method"],
            as_index=False,
            sort=True,
        )[metric_columns + target_columns + [
            "targets_mae_improved_count",
            "targets_r2_improved_count",
            "targets_spearman_improved_count",
        ]]
        .mean()
    )
    multiseed = _metric_summary(
        subject_seed_means,
        grouping=("method",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ).rename(columns={
        "mean": "mean_subject_metric",
        "median": "median_subject_metric",
        "std": "std_subject_metric",
        "mean_gain": "mean_subject_gain",
        "median_gain": "median_subject_gain",
        "positive_fraction": "positive_subject_fraction",
    })
    source = _metric_summary(
        subject_seed_means,
        grouping=("source", "method"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return per_seed, multiseed, source, subject_seed_means


def build_stability_summary(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
) -> pd.DataFrame:
    seeds = tuple(map(int, model_seeds))
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ]
    rows: list[dict[str, Any]] = []
    for (subject, source, method), group in completed.groupby(
        ["subject_id", "source", "method"], sort=True
    ):
        if set(group["model_seed"].astype(int)) != set(seeds):
            continue
        for metric in METRICS:
            gains = {
                int(row.model_seed): float(getattr(row, f"{metric}_gain"))
                for row in group.itertuples()
            }
            values = np.asarray([gains[seed] for seed in seeds], dtype=float)
            positive = int(np.sum(values > 0))
            rows.append({
                "record_type": "subject",
                "subject_id": str(subject),
                "source": str(source),
                "method": str(method),
                "metric": metric,
                **{f"gain_seed_{seed}": gains[seed] for seed in seeds},
                "mean_gain": float(np.mean(values)),
                "std_gain": float(np.std(values, ddof=1)),
                "minimum_gain": float(np.min(values)),
                "maximum_gain": float(np.max(values)),
                "positive_seeds_count": positive,
                "improved_in_at_least_2_of_3": positive >= 2,
                "improved_in_all_3": positive == len(seeds),
            })
    subjects = pd.DataFrame(rows)
    aggregates: list[dict[str, Any]] = []
    for (method, metric), group in subjects.groupby(
        ["method", "metric"], sort=True
    ):
        aggregates.append({
            "record_type": "aggregate",
            "subject_id": None,
            "source": "all",
            "method": method,
            "metric": metric,
            "n_subjects": int(len(group)),
            "subjects_improved_at_least_2_of_3": int(
                group["improved_in_at_least_2_of_3"].sum()
            ),
            "fraction_improved_at_least_2_of_3": float(
                group["improved_in_at_least_2_of_3"].mean()
            ),
            "subjects_improved_all_3": int(
                group["improved_in_all_3"].sum()
            ),
            "fraction_improved_all_3": float(
                group["improved_in_all_3"].mean()
            ),
            "mean_subject_gain": float(group["mean_gain"].mean()),
            "median_subject_gain": float(group["mean_gain"].median()),
            "mean_seed_std": float(group["std_gain"].mean()),
        })
    return pd.concat(
        [subjects, pd.DataFrame(aggregates)], ignore_index=True, sort=False
    )


def _paired_comparisons(
    frame: pd.DataFrame,
    *,
    grouping: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    comparisons = (
        ("head_only", "zero_shot"),
        ("full_model", "zero_shot"),
        ("full_model", "head_only"),
    )
    rows: list[dict[str, Any]] = []
    grouped = (
        [((), frame)]
        if not grouping
        else frame.groupby(list(grouping), sort=True, dropna=False)
    )
    for key, scoped in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouping, keys))
        for method, reference in comparisons:
            for metric in METRICS:
                pivot = scoped.pivot_table(
                    index="subject_id",
                    columns="method",
                    values=f"{metric}_after",
                    aggfunc="first",
                )
                if method not in pivot or reference not in pivot:
                    continue
                differences = pd.Series([
                    metric_gain(metric, reference_value, method_value)
                    for method_value, reference_value in zip(
                        pivot[method], pivot[reference]
                    )
                    if np.isfinite([method_value, reference_value]).all()
                ])
                low, high = _bootstrap_interval(
                    differences,
                    samples=bootstrap_samples,
                    random_state=bootstrap_seed,
                )
                rows.append({
                    **identity,
                    "method": method,
                    "reference_method": reference,
                    "metric": f"{metric}_gain",
                    "n_subjects": int(len(differences)),
                    "mean_difference": (
                        float(differences.mean())
                        if len(differences) else np.nan
                    ),
                    "median_difference": (
                        float(differences.median())
                        if len(differences) else np.nan
                    ),
                    "positive_fraction": (
                        float(np.mean(differences > 0))
                        if len(differences) else np.nan
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                })
    return pd.DataFrame(rows)


def build_target_summary(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    seeds = set(map(int, model_seeds))
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ]
    rows: list[dict[str, Any]] = []
    for method, method_group in completed.groupby("method", sort=True):
        valid_subjects = method_group.groupby("subject_id")[
            "model_seed"
        ].agg(lambda values: set(map(int, values)))
        keep = set(valid_subjects.loc[valid_subjects.map(lambda x: x == seeds)].index)
        method_group = method_group.loc[
            method_group["subject_id"].astype(str).isin(keep)
        ]
        for target in CANONICAL_TARGETS:
            columns = [
                f"{target}_{metric}_{suffix}"
                for metric in (
                    "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
                )
                for suffix in ("before", "after", "gain")
            ]
            per_subject = method_group.groupby("subject_id")[columns].mean()
            seed_gain = method_group.pivot_table(
                index="subject_id",
                columns="model_seed",
                values=f"{target}_mae_gain",
                aggfunc="first",
            )
            mae_gain = per_subject[f"{target}_mae_gain"]
            low, high = _bootstrap_interval(
                mae_gain,
                samples=bootstrap_samples,
                random_state=bootstrap_seed,
            )
            positive_counts = (seed_gain > 0).sum(axis=1)
            row: dict[str, Any] = {
                "target_name": target,
                "method": method,
                "n_subjects": int(len(per_subject)),
                "mae_before": float(
                    per_subject[f"{target}_mae_before"].mean()
                ),
                "mae_after": float(
                    per_subject[f"{target}_mae_after"].mean()
                ),
                "mae_gain": float(mae_gain.mean()),
                "mae_gain_ci_low": low,
                "mae_gain_ci_high": high,
                "fraction_subjects_improved": float(np.mean(mae_gain > 0)),
                "fraction_improved_at_least_2_seeds": float(
                    np.mean(positive_counts >= 2)
                ),
                "fraction_improved_all_3_seeds": float(
                    np.mean(positive_counts == len(seeds))
                ),
            }
            for metric in ("rmse", "r2", "pearson", "spearman", "abs_bias"):
                row[f"{metric}_gain"] = float(
                    per_subject[f"{target}_{metric}_gain"].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


class PMRegressionPersonalizationMultiseedExperiment:
    """Compose three canonical single-seed PM personalization runs."""

    def __init__(self, config_path: str | Path):
        self.config_path = _repo_path(config_path)
        self.document = load_pm_multiseed_spec(self.config_path)
        self.experiment = self.document["experiment"]
        self.split_seed = int(self.experiment["split_seed"])
        self.model_seeds = tuple(map(int, self.experiment["model_seeds"]))
        self.base_template_path = _repo_path(
            self.document["base_template"]["config_path"]
        )
        self.base_template = yaml.safe_load(
            self.base_template_path.read_text(encoding="utf-8")
        ) or {}

    def _seed42_compatibility(self) -> dict[str, Any]:
        source_root = _repo_path(
            self.experiment.get(
                "seed_42_source_root",
                "benchmark_results/pm_regression_personalization_20pct",
            )
        )
        manifests = [
            path for path in source_root.rglob("run_manifest.json")
            if json.loads(path.read_text(encoding="utf-8")).get("status")
            == "completed"
        ]
        direct = source_root / "run_manifest.json"
        if direct.is_file() and direct not in manifests:
            payload = json.loads(direct.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                manifests.append(direct)
        if not manifests:
            return {
                "eligible": False,
                "source_run": None,
                "reason": "No completed seed-42 manifest found",
                "checks": {},
            }
        manifest_path = max(manifests, key=lambda path: path.stat().st_mtime)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        predictions_path = manifest_path.parent / "predictions.parquet"
        expected_schema = {
            "subject_id", "source", "sample_id", "record_id", "outer_fold",
            "method", "target_name", "target_index", "y_true",
            "y_pred_before", "y_pred_after",
        }
        prediction_schema_match = (
            predictions_path.is_file()
            and expected_schema.issubset(
                set(pd.read_parquet(predictions_path).columns)
            )
        )
        current_data_path = _repo_path(
            self.base_template["datasets"][
                self.document["base_template"].get(
                    "dataset", "emotiv_pm_regression"
                )
            ]["data_path"]
        )
        checks = {
            "dataset_hash_match": (
                manifest.get("dataset_sha256")
                == _file_sha256(current_data_path)
            ),
            "config_hash_match": False,
            "implementation_hash_match": (
                manifest.get("implementation_hash")
                == _implementation_hash()
            ),
            "split_hash_match": False,
            "prediction_schema_match": bool(prediction_schema_match),
            "target_order_match": tuple(
                manifest.get("target_order", ())
            ) == CANONICAL_TARGETS,
            "completed_conditions_match": int(
                manifest.get("completed_conditions", -1)
            ) == 265,
        }
        eligible = all(checks.values())
        return {
            "eligible": eligible,
            "source_run": str(manifest_path.parent),
            "source_manifest_hash": _file_sha256(manifest_path),
            "reason": (
                "Compatible"
                if eligible
                else (
                    "Strict reuse rejected: implementation/config changed "
                    "and the historical split hashes are incomplete"
                )
            ),
            "checks": checks,
        }

    @staticmethod
    def _seed_frames(
        manifest: Mapping[str, Any],
        *,
        model_seed: int,
        split_seed: int,
    ) -> dict[str, pd.DataFrame]:
        root = Path(manifest["output_dir"])
        subjects = pd.read_csv(
            root / "personalization_subject_metrics.csv"
        )
        subjects["model_seed"] = int(model_seed)
        subjects["split_seed"] = int(split_seed)
        predictions = pd.read_parquet(root / "predictions.parquet")
        predictions["model_seed"] = int(model_seed)
        predictions["split_seed"] = int(split_seed)
        splits = pd.read_csv(root / "calibration_split_audit.csv")
        checkpoints = pd.read_csv(root / "checkpoint_audit.csv")
        global_folds = pd.read_csv(root / "global_fold_summary.csv")
        for frame in (splits, checkpoints, global_folds):
            frame["model_seed"] = int(model_seed)
            frame["split_seed"] = int(split_seed)
        failures = _safe_read_csv(root / "failures.csv")
        if not failures.empty:
            failures["model_seed"] = int(model_seed)
            failures["split_seed"] = int(split_seed)
        return {
            "subjects": subjects,
            "predictions": predictions,
            "splits": splits,
            "checkpoints": checkpoints,
            "global_folds": global_folds,
            "failures": failures,
        }

    def _split_consistency(
        self,
        frames: Mapping[int, Mapping[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for seed, seed_frames in frames.items():
            split = seed_frames["splits"].copy()
            preferred = split.loc[split["method"] == "full_model"].copy()
            preferred["model_seed"] = int(seed)
            rows.append(preferred)
        frame = pd.concat(rows, ignore_index=True)
        grouping = frame.groupby(["outer_fold", "subject_id"], sort=True)
        for column in SPLIT_HASH_COLUMNS:
            frame[f"{column}_consistent"] = grouping[column].transform(
                "nunique"
            ).eq(1)
        consistency_columns = [
            f"{column}_consistent" for column in SPLIT_HASH_COLUMNS
        ]
        frame["all_split_hashes_consistent"] = frame[
            consistency_columns
        ].all(axis=1)
        if not frame["all_split_hashes_consistent"].all():
            raise RuntimeError("Split/preprocessing hashes differ between seeds")
        overlap_columns = (
            "calibration_evaluation_overlap",
            "adaptation_train_validation_overlap",
            "adaptation_evaluation_overlap",
            "duplicate_sample_ids",
            "target_in_global_inner_train",
            "target_in_global_inner_validation",
        )
        if int(frame[list(overlap_columns)].to_numpy(dtype=int).sum()) != 0:
            raise RuntimeError("Non-zero leakage/overlap in split audit")
        return frame

    def execute(
        self,
        *,
        fold_limit: Optional[int] = None,
        subject_limit: Optional[int] = None,
        max_epochs: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is disabled")
        implementation_hash = _implementation_hash()
        resolved = {
            "schema_version": SCHEMA_VERSION,
            "document": self.document,
            "fold_limit": fold_limit,
            "subject_limit": subject_limit,
            "max_epochs": max_epochs,
            "implementation_hash": implementation_hash,
        }
        config_hash = _canonical_hash(resolved)
        root = _repo_path(output_dir or self.experiment["output_dir"])
        resume_enabled = bool(resume or self.experiment.get("resume", False))
        run_dir: Optional[Path] = None
        if resume_enabled and root.is_dir():
            for candidate in sorted(
                (path for path in root.iterdir() if path.is_dir()),
                reverse=True,
            ):
                progress_path = candidate / "progress.json"
                if not progress_path.is_file():
                    continue
                progress = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
                if progress.get("config_hash") != config_hash:
                    continue
                if progress.get("implementation_hash") != implementation_hash:
                    raise RuntimeError("Multiseed implementation hash mismatch")
                run_dir = candidate
                manifest_path = run_dir / "run_manifest.json"
                if (
                    progress.get("status") == "completed"
                    and manifest_path.is_file()
                ):
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["resumed"] = True
                    manifest["resume_skipped_model_seeds"] = len(
                        progress.get("completed_model_seeds", ())
                    )
                    return manifest
                break
        if run_dir is None:
            run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=False)
            _write_json(run_dir / "progress.json", {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "completed_model_seeds": [],
            })
            with (run_dir / "resolved_multiseed.yaml").open(
                "w", encoding="utf-8"
            ) as output:
                yaml.safe_dump(resolved, output, sort_keys=False)

        compatibility = self._seed42_compatibility()
        completed: dict[int, dict[str, Any]] = {}
        for seed in self.model_seeds:
            manifest_path = (
                run_dir / f"seed_{seed}" / "personalization"
                / "run_manifest.json"
            )
            if manifest_path.is_file():
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if payload.get("status") == "completed":
                    completed[seed] = payload

        started = time.perf_counter()
        provenance_rows: list[dict[str, Any]] = []
        new_global_trainings = 0
        for seed in self.model_seeds:
            was_completed = seed in completed
            seed_root = run_dir / f"seed_{seed}"
            base_path = seed_root / "resolved_base.yaml"
            single_path = seed_root / "resolved_personalization.yaml"
            base_output = seed_root / "global_base"
            personalization_output = seed_root / "personalization"
            seed_root.mkdir(parents=True, exist_ok=True)
            base_config = resolve_seed_base_config(
                self.base_template,
                model_seed=seed,
                split_seed=self.split_seed,
                output_dir=base_output,
                maximum_epochs=(
                    None
                    if self.experiment.get("global_max_epochs") is None
                    else int(self.experiment["global_max_epochs"])
                ),
                outer_folds=self.experiment.get("outer_folds"),
            )
            single_config = resolve_seed_personalization_config(
                self.document,
                model_seed=seed,
                split_seed=self.split_seed,
                base_config_path=base_path,
                output_dir=personalization_output,
            )
            base_path.write_text(
                yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
            )
            single_path.write_text(
                yaml.safe_dump(single_config, sort_keys=False), encoding="utf-8"
            )
            if seed not in completed:
                experiment = PMRegressionPersonalizationExperiment(single_path)
                manifest = experiment.execute(
                    fold_limit=fold_limit,
                    subject_limit=subject_limit,
                    methods=METHODS,
                    max_epochs=max_epochs,
                    resume=resume_enabled,
                )
                completed[seed] = manifest
                new_global_trainings += int(manifest["global_trainings"])
                del experiment
                gc.collect()
                torch.cuda.empty_cache()
            checks = compatibility.get("checks", {})
            provenance_rows.append({
                "model_seed": seed,
                "source_run": compatibility.get("source_run"),
                "reused": False,
                "dataset_hash_match": checks.get("dataset_hash_match", False),
                "config_hash_match": checks.get("config_hash_match", False),
                "implementation_hash_match": checks.get(
                    "implementation_hash_match", False
                ),
                "split_hash_match": checks.get("split_hash_match", False),
                "prediction_schema_match": checks.get(
                    "prediction_schema_match", False
                ),
                "compatibility_status": (
                    "compatible" if compatibility["eligible"] else "rerun"
                ),
                "notes": compatibility["reason"],
                "seed_run": completed[seed]["output_dir"],
                "outcome": "resumed" if was_completed else "completed",
            })
            _write_json(run_dir / "progress.json", {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "completed_model_seeds": sorted(completed),
            })

        seed_frames = {
            seed: self._seed_frames(
                completed[seed],
                model_seed=seed,
                split_seed=self.split_seed,
            )
            for seed in self.model_seeds
        }
        subjects = pd.concat(
            [frame["subjects"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        predictions = pd.concat(
            [frame["predictions"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        checkpoints = pd.concat(
            [frame["checkpoints"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        global_folds = pd.concat(
            [frame["global_folds"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        failure_frames = [
            frame["failures"] for frame in seed_frames.values()
            if not frame["failures"].empty
        ]
        failures = (
            pd.concat(failure_frames, ignore_index=True)
            if failure_frames else pd.DataFrame()
        )
        split_consistency = self._split_consistency(seed_frames)

        if tuple(sorted(subjects["model_seed"].unique())) != self.model_seeds:
            raise RuntimeError("Model seed coverage is incomplete")
        subject_folds = subjects.groupby("subject_id")["outer_fold"].nunique()
        if not subject_folds.eq(1).all():
            raise RuntimeError("Outer fold assignments differ between seeds")
        if not checkpoints["initial_matches_global"].astype(bool).all():
            raise RuntimeError("Fine-tuning initial checkpoint mismatch")
        if not checkpoints["global_state_unchanged"].astype(bool).all():
            raise RuntimeError("A condition changed the shared global model")
        head = checkpoints.loc[checkpoints["method"] == "head_only"]
        if not head["frozen_parameters_unchanged"].astype(bool).all():
            raise RuntimeError("Head-only changed frozen backbone")
        zero = checkpoints.loc[checkpoints["method"] == "zero_shot"]
        if not zero["fine_tune_initial_hash"].eq(
            zero["fine_tune_final_hash"]
        ).all():
            raise RuntimeError("Zero-shot changed model state")
        for fold, group in global_folds.groupby("outer_fold", sort=True):
            if group.groupby("model_seed")[
                "global_checkpoint_hash"
            ].first().nunique() != len(self.model_seeds):
                raise RuntimeError(
                    f"Global checkpoint hashes do not differ in {fold}"
                )

        prediction_key = [
            "model_seed", "outer_fold", "subject_id", "method",
            "sample_id", "target_name",
        ]
        if predictions.duplicated(prediction_key).any():
            raise RuntimeError("Duplicate multiseed prediction keys")
        values = predictions[
            ["y_true", "y_pred_before", "y_pred_after"]
        ].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite multiseed predictions")
        if set(predictions["target_name"]) != set(CANONICAL_TARGETS):
            raise RuntimeError("Target order/coverage changed")
        truth_variants = predictions.groupby(
            ["outer_fold", "subject_id", "sample_id", "target_name"]
        )["y_true"].nunique()
        if int(truth_variants.max()) != 1:
            raise RuntimeError("y_true differs between methods or seeds")
        evaluation_sets = predictions.groupby(
            ["subject_id", "model_seed", "method"]
        )["sample_id"].agg(lambda values: frozenset(map(str, values)))
        for subject in predictions["subject_id"].astype(str).unique():
            if len(set(evaluation_sets.loc[subject])) != 1:
                raise RuntimeError(
                    f"Final evaluation differs for subject {subject}"
                )
        zero_predictions = predictions.loc[
            predictions["method"] == "zero_shot"
        ]
        if not np.array_equal(
            zero_predictions["y_pred_before"].to_numpy(),
            zero_predictions["y_pred_after"].to_numpy(),
        ):
            raise RuntimeError("Zero-shot before/after predictions differ")

        bootstrap_samples = int(
            self.document.get("statistics", {}).get(
                "bootstrap_samples", 1000
            )
        )
        per_seed, multiseed, source_summary, subject_means = (
            build_multiseed_aggregates(
                subjects,
                model_seeds=self.model_seeds,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=self.split_seed,
            )
        )
        stability = build_stability_summary(
            subjects, model_seeds=self.model_seeds
        )
        target_summary = build_target_summary(
            subjects,
            model_seeds=self.model_seeds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=self.split_seed,
        )
        paired_by_seed = _paired_comparisons(
            subjects.loc[subjects["status"] == "completed"],
            grouping=("model_seed",),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=self.split_seed,
        )
        paired_multiseed = _paired_comparisons(
            subject_means,
            grouping=(),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=self.split_seed,
        )

        paths = {
            "seed_provenance": run_dir / "seed_provenance.csv",
            "global_fold_summary": run_dir / "global_fold_summary.csv",
            "multiseed_subject_metrics": (
                run_dir / "multiseed_subject_metrics.csv"
            ),
            "per_seed_aggregate": run_dir / "per_seed_aggregate.csv",
            "multiseed_aggregate": run_dir / "multiseed_aggregate.csv",
            "multiseed_target_summary": (
                run_dir / "multiseed_target_summary.csv"
            ),
            "multiseed_source_summary": (
                run_dir / "multiseed_source_summary.csv"
            ),
            "stability_summary": run_dir / "stability_summary.csv",
            "paired_comparisons_by_seed": (
                run_dir / "paired_comparisons_by_seed.csv"
            ),
            "paired_comparisons_multiseed": (
                run_dir / "paired_comparisons_multiseed.csv"
            ),
            "split_consistency_audit": (
                run_dir / "split_consistency_audit.csv"
            ),
            "checkpoint_audit": run_dir / "checkpoint_audit.csv",
            "predictions": run_dir / "predictions.parquet",
            "failures": run_dir / "failures.csv",
        }
        pd.DataFrame(provenance_rows).to_csv(
            paths["seed_provenance"], index=False
        )
        global_folds.to_csv(paths["global_fold_summary"], index=False)
        subjects.to_csv(paths["multiseed_subject_metrics"], index=False)
        per_seed.to_csv(paths["per_seed_aggregate"], index=False)
        multiseed.to_csv(paths["multiseed_aggregate"], index=False)
        target_summary.to_csv(
            paths["multiseed_target_summary"], index=False
        )
        source_summary.to_csv(
            paths["multiseed_source_summary"], index=False
        )
        stability.to_csv(paths["stability_summary"], index=False)
        paired_by_seed.to_csv(
            paths["paired_comparisons_by_seed"], index=False
        )
        paired_multiseed.to_csv(
            paths["paired_comparisons_multiseed"], index=False
        )
        split_consistency.to_csv(
            paths["split_consistency_audit"], index=False
        )
        checkpoints.to_csv(paths["checkpoint_audit"], index=False)
        predictions.to_parquet(paths["predictions"], index=False)
        failures.to_csv(paths["failures"], index=False)

        elapsed = time.perf_counter() - started
        completed_conditions = int(
            (subjects["status"] == "completed").sum()
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "run_directory": str(run_dir),
            "split_seed": self.split_seed,
            "model_seeds": list(self.model_seeds),
            "seed_42_reused": False,
            "seed_42_compatibility": compatibility,
            "new_global_trainings": int(new_global_trainings),
            "global_fold_rows": int(len(global_folds)),
            "subjects": int(subjects["subject_id"].nunique()),
            "condition_rows": int(len(subjects)),
            "completed_conditions": completed_conditions,
            "incomplete_conditions": int(len(subjects) - completed_conditions),
            "failed_conditions": int(len(failures)),
            "prediction_rows": int(len(predictions)),
            "elapsed_seconds": float(elapsed),
            "global_training_time_seconds": float(
                global_folds["global_training_time_seconds"].sum()
            ),
            "fine_tuning_time_seconds": float(
                subjects["training_time_seconds"].sum()
            ),
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_gpu_memory_bytes": int(max(
                global_folds["peak_gpu_memory_bytes"].max(),
                subjects["peak_gpu_memory_bytes"].max(),
            )),
            "target_order": list(CANONICAL_TARGETS),
            "artifacts": {key: str(path) for key, path in paths.items()},
            "seed_runs": {
                str(seed): completed[seed]["output_dir"]
                for seed in self.model_seeds
            },
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(run_dir / "progress.json", {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "completed_model_seeds": list(self.model_seeds),
        })
        return manifest


__all__ = [
    "METRICS",
    "METHODS",
    "PMRegressionPersonalizationMultiseedExperiment",
    "build_multiseed_aggregates",
    "build_stability_summary",
    "build_target_summary",
    "load_pm_multiseed_spec",
    "resolve_seed_base_config",
    "resolve_seed_personalization_config",
]
