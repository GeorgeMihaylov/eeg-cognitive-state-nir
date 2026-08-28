"""Leakage-safe personalization of canonical multi-output PM regression.

The experiment deliberately composes the standard :class:`BenchmarkRunner`
for global model fitting and the shared torch adapter for fine-tuning.  It
contains no independent neural-network training loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import BenchmarkRunner, benchmark_config_hash
from bench.experiments.user_calibration import (
    CalibrationSpec,
    _canonical_hash,
    _checkpoint_payload,
    _implementation_hash as _classification_implementation_hash,
    _parameter_audit,
    _parameter_digest,
    _repo_path,
    _safe_component,
    _state_digest,
    _write_json,
    chronological_window_partition,
)
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter
from cogstate.model_zoo.factory import build_model
from cogstate.adaptation.regression_calibration import (
    AffineCalibration,
    apply_affine_calibration,
    apply_bias_correction,
    fit_affine_calibration,
    fit_bias_correction,
)


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "pm-regression-personalization-v1"
CANONICAL_TARGETS = (
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
)
METHODS = (
    "zero_shot",
    "bias_correction",
    "affine_calibration",
    "head_only",
    "full_model",
)
COMPLETED_STATUS = "completed"
ALLOWED_STATUSES = frozenset({
    COMPLETED_STATUS,
    "insufficient_calibration_samples",
    "insufficient_adaptation_train",
    "insufficient_evaluation_samples",
    "constant_target",
    "non_finite_predictions",
    "training_failed",
})
ERROR_METRICS = frozenset({"mae", "rmse", "abs_bias"})
HIGHER_IS_BETTER_METRICS = frozenset({"r2", "pearson", "spearman"})


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    digest.update(_classification_implementation_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ordered_hash(values: Sequence[Any]) -> str:
    return _canonical_hash([str(value) for value in values])


def _validated_regression_arrays(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError(
            "Regression arrays must have identical [samples, targets] shape, "
            f"got {truth.shape} and {prediction.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression arrays must be finite")
    return truth, prediction


def regression_personalization_metrics(
    y_true: Any,
    y_pred: Any,
    target_names: Sequence[str] = CANONICAL_TARGETS,
) -> dict[str, Any]:
    """Return the requested target-first metric naming contract."""
    truth, prediction = _validated_regression_arrays(y_true, y_pred)
    raw = MetricsCalculator.calculate_regression_metrics(
        truth, prediction, target_names=list(target_names)
    )
    result: dict[str, Any] = {
        "n_samples": int(len(truth)),
        "n_outputs": int(truth.shape[1]),
    }
    metric_names = (
        "mae", "rmse", "r2", "pearson", "spearman",
        "mean_error", "abs_bias",
    )
    for target_name in target_names:
        key = MetricsCalculator.normalize_target_name(target_name)
        for metric in metric_names:
            result[f"{target_name}_{metric}"] = raw[f"{metric}_{key}"]
    for metric in ("mae", "rmse", "r2", "pearson", "spearman", "abs_bias"):
        result[f"macro_{metric}"] = raw[f"{metric}_macro"]
    result["defined_pearson_targets"] = raw["pearson_valid_targets"]
    result["defined_spearman_targets"] = raw["spearman_valid_targets"]
    return result


def metric_gain(metric: str, before: float, after: float) -> float:
    normalized = str(metric).removeprefix("macro_")
    for target_name in CANONICAL_TARGETS:
        normalized = normalized.removeprefix(f"{target_name}_")
    if normalized in ERROR_METRICS:
        return float(before - after)
    if normalized in HIGHER_IS_BETTER_METRICS:
        return float(after - before)
    raise ValueError(f"Unknown metric gain direction for {metric!r}")


def _metric_bundle(
    truth: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before_metrics = regression_personalization_metrics(truth, before)
    after_metrics = regression_personalization_metrics(truth, after)
    gains: dict[str, Any] = {}
    for name in (
        "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
    ):
        before_value = before_metrics[f"macro_{name}"]
        after_value = after_metrics[f"macro_{name}"]
        gains[f"macro_{name}_gain"] = (
            np.nan
            if not np.isfinite([before_value, after_value]).all()
            else metric_gain(name, before_value, after_value)
        )
    for target_name in CANONICAL_TARGETS:
        for name in (
            "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
        ):
            before_value = before_metrics[f"{target_name}_{name}"]
            after_value = after_metrics[f"{target_name}_{name}"]
            gains[f"{target_name}_{name}_gain"] = (
                np.nan
                if not np.isfinite([before_value, after_value]).all()
                else metric_gain(name, before_value, after_value)
            )
    return before_metrics, after_metrics, gains


def _prediction_frame(
    *,
    fold_name: str,
    subject_id: str,
    source: str,
    method: str,
    metadata: pd.DataFrame,
    truth: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> pd.DataFrame:
    truth, before = _validated_regression_arrays(truth, before)
    _, after = _validated_regression_arrays(truth, after)
    rows: list[dict[str, Any]] = []
    for sample_index, sample in metadata.reset_index(drop=True).iterrows():
        for target_index, target_name in enumerate(CANONICAL_TARGETS):
            rows.append({
                "subject_id": str(subject_id),
                "source": str(source),
                "sample_id": str(sample["sample_id"]),
                "record_id": str(sample["record_id"]),
                "outer_fold": str(fold_name),
                "method": str(method),
                "target_name": target_name,
                "target_index": target_index,
                "y_true": float(truth[sample_index, target_index]),
                "y_pred_before": float(before[sample_index, target_index]),
                "y_pred_after": float(after[sample_index, target_index]),
            })
    frame = pd.DataFrame(rows)
    key = ["subject_id", "sample_id", "outer_fold", "method", "target_name"]
    if frame.duplicated(key).any():
        raise RuntimeError("Regression personalization prediction keys are not unique")
    if not np.isfinite(
        frame[["y_true", "y_pred_before", "y_pred_after"]].to_numpy()
    ).all():
        raise RuntimeError("Regression personalization predictions are non-finite")
    return frame


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    random_state: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    means = np.asarray([
        np.mean(rng.choice(array, size=len(array), replace=True))
        for _ in range(samples)
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate_outputs(
    subject_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completed = subject_metrics.loc[
        subject_metrics["status"] == COMPLETED_STATUS
    ].copy()
    aggregate_rows: list[dict[str, Any]] = []
    for (source, method), group in completed.groupby(
        ["source", "method"], sort=True
    ):
        for metric in (
            "macro_mae", "macro_rmse", "macro_r2",
            "macro_pearson", "macro_spearman", "macro_abs_bias",
        ):
            values = pd.to_numeric(group[f"{metric}_after"], errors="coerce")
            finite = values[np.isfinite(values)]
            aggregate_rows.append({
                "scope": "overall" if source == "all" else "source_subset",
                "source": source,
                "method": method,
                "metric": metric,
                "n_subjects": int(len(group)),
                "defined_subjects": int(len(finite)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            })

    paired_subjects = completed.loc[completed["source"] == "all"].copy()
    comparisons = (
        ("bias_correction", "zero_shot"),
        ("affine_calibration", "zero_shot"),
        ("head_only", "zero_shot"),
        ("full_model", "zero_shot"),
        ("affine_calibration", "bias_correction"),
        ("head_only", "bias_correction"),
        ("full_model", "bias_correction"),
        ("full_model", "head_only"),
    )
    paired_rows: list[dict[str, Any]] = []
    for method, reference in comparisons:
        left = paired_subjects.loc[
            paired_subjects["method"] == method
        ].set_index(["outer_fold", "subject_id"])
        right = paired_subjects.loc[
            paired_subjects["method"] == reference
        ].set_index(["outer_fold", "subject_id"])
        common = left.index.intersection(right.index)
        for metric in (
            "macro_mae", "macro_rmse", "macro_r2",
            "macro_spearman", "macro_abs_bias",
        ):
            values: list[float] = []
            for key in common:
                method_value = float(left.loc[key, f"{metric}_after"])
                reference_value = float(right.loc[key, f"{metric}_after"])
                if np.isfinite([method_value, reference_value]).all():
                    values.append(metric_gain(
                        metric, reference_value, method_value
                    ))
            low, high = _bootstrap_interval(
                values,
                samples=bootstrap_samples,
                random_state=bootstrap_seed,
            )
            array = np.asarray(values, dtype=float)
            paired_rows.append({
                "method": method,
                "reference_method": reference,
                "metric": f"{metric}_gain",
                "n_subjects": int(len(array)),
                "mean_difference": (
                    float(np.mean(array)) if len(array) else np.nan
                ),
                "median_difference": (
                    float(np.median(array)) if len(array) else np.nan
                ),
                "positive_fraction": (
                    float(np.mean(array > 0)) if len(array) else np.nan
                ),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            })

    target_rows: list[dict[str, Any]] = []
    for method, group in paired_subjects.groupby("method", sort=True):
        for target in CANONICAL_TARGETS:
            row: dict[str, Any] = {
                "target_name": target,
                "method": method,
                "n_subjects": int(len(group)),
            }
            for metric in (
                "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
            ):
                before = pd.to_numeric(
                    group[f"{target}_{metric}_before"], errors="coerce"
                )
                after = pd.to_numeric(
                    group[f"{target}_{metric}_after"], errors="coerce"
                )
                gain = pd.to_numeric(
                    group[f"{target}_{metric}_gain"], errors="coerce"
                )
                finite_gain = gain[np.isfinite(gain)]
                low, high = _bootstrap_interval(
                    finite_gain,
                    samples=bootstrap_samples,
                    random_state=bootstrap_seed,
                )
                row[f"baseline_{metric}"] = (
                    float(before[np.isfinite(before)].mean())
                    if np.isfinite(before).any() else np.nan
                )
                row[f"personalized_{metric}"] = (
                    float(after[np.isfinite(after)].mean())
                    if np.isfinite(after).any() else np.nan
                )
                row[f"{metric}_gain"] = (
                    float(finite_gain.mean()) if len(finite_gain) else np.nan
                )
                row[f"{metric}_gain_ci_low"] = low
                row[f"{metric}_gain_ci_high"] = high
                row[f"{metric}_fraction_subjects_improved"] = (
                    float(np.mean(finite_gain > 0))
                    if len(finite_gain) else np.nan
                )
            target_rows.append(row)
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(target_rows),
    )


class PMRegressionPersonalizationExperiment:
    """Run one-seed, 20%-budget PM personalization over outer-test users."""

    def __init__(self, config_path: str | Path):
        self.config_path = _repo_path(config_path)
        self.document = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        ) or {}
        experiment = self.document.get("experiment", {})
        if experiment.get("type") != "pm_regression_personalization":
            raise ValueError(
                "experiment.type must be 'pm_regression_personalization'"
            )
        targets = tuple(self.document.get("targets", ()))
        if targets != CANONICAL_TARGETS:
            raise ValueError(
                "targets must use the canonical seven-output PM order"
            )
        configured_methods = tuple(
            self.document.get("calibration", {}).get("methods", ())
        )
        if not configured_methods or not set(configured_methods).issubset(METHODS):
            raise ValueError(f"Unknown or empty methods: {configured_methods}")
        self.base_config_path = _repo_path(
            self.document["base_run"]["config_path"]
        )
        self.base_config = yaml.safe_load(
            self.base_config_path.read_text(encoding="utf-8")
        ) or {}
        dataset_name, task_name, model_name = self._identities()
        dataset_config = self.base_config["datasets"][dataset_name]
        if tuple(dataset_config.get("target_cols", ())) != CANONICAL_TARGETS:
            raise ValueError("Base run target order is not canonical")
        if task_name != "performance_metrics_regression":
            raise ValueError("Base task must be performance_metrics_regression")
        if self.base_config["models"][model_name].get("task_type") != "regression":
            raise ValueError("Base model must use regression task_type")

    def _identities(self) -> tuple[str, str, str]:
        base = self.document["base_run"]
        return (
            str(base.get("dataset", next(iter(self.base_config["datasets"])))),
            str(base.get("task", self.base_config["tasks"][0])),
            str(base.get("model", next(iter(self.base_config["models"])))),
        )

    def _ensure_base_run(self):
        completed = BenchmarkRunner.find_completed_run(self.base_config)
        if completed is None:
            if not bool(self.document["base_run"].get("train_if_missing", False)):
                raise FileNotFoundError(
                    f"No completed base run matches {self.base_config_path}"
                )
            LOGGER.info("Training missing canonical global regression run")
            runner = BenchmarkRunner(deepcopy(self.base_config))
            runner.run()
            completed = runner.completed_run()
        return completed

    def _load_fold_adapter(
        self,
        checkpoint: Path,
        model_name: str,
    ) -> TorchClassificationAdapter:
        payload = _checkpoint_payload(checkpoint)
        if payload.get("task_type") != "regression":
            raise ValueError("Expected a regression checkpoint")
        if tuple(payload.get("input_shape", ())) != (448,):
            raise ValueError(
                f"Expected checkpoint input_shape=(448,), got "
                f"{payload.get('input_shape')}"
            )
        if int(payload.get("num_outputs", -1)) != len(CANONICAL_TARGETS):
            raise ValueError("Regression checkpoint must contain seven outputs")
        adapter = build_model(
            model_name=str(self.base_config["models"][model_name]["type"]),
            task_type="regression",
            input_shape=tuple(int(value) for value in payload["input_shape"]),
            num_outputs=int(payload["num_outputs"]),
            params=deepcopy(self.base_config["models"][model_name]["params"]),
        )
        if not isinstance(adapter, TorchClassificationAdapter):
            raise TypeError("PM personalization requires the shared torch adapter")
        adapter.load(checkpoint)
        return adapter

    def execute(
        self,
        *,
        fold_limit: Optional[int] = None,
        subject_limit: Optional[int] = None,
        methods: Optional[Sequence[str]] = None,
        max_epochs: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        experiment_config = self.document["experiment"]
        calibration = self.document["calibration"]
        require_cuda = bool(experiment_config.get("require_cuda", True))
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing a CPU fallback")
        selected_methods = tuple(
            calibration["methods"] if methods is None else methods
        )
        unknown_methods = sorted(set(selected_methods) - set(METHODS))
        if unknown_methods:
            raise ValueError(f"Unknown methods: {unknown_methods}")
        budget = float(calibration.get("maximum_calibration_fraction", 0.20))
        if budget != 0.20:
            raise ValueError("This experiment fixes the calibration budget at 20%")
        model_seed = int(experiment_config.get("model_seed", 42))
        split_seed = int(experiment_config.get("split_seed", 42))
        dataset_name, task_name, model_name = self._identities()
        if model_seed < 0 or split_seed != 42:
            raise ValueError(
                "model_seed must be non-negative and split_seed must remain 42"
            )
        configured_model_seed = int(
            self.base_config["models"][model_name]["params"].get(
                "random_state", -1
            )
        )
        if configured_model_seed != model_seed:
            raise ValueError(
                "Base model random_state must equal experiment.model_seed: "
                f"{configured_model_seed} != {model_seed}"
            )
        for section in ("validation", "evaluation", "task_config"):
            configured_split_seed = int(
                self.base_config.get(section, {}).get("random_state", -1)
            )
            if configured_split_seed != split_seed:
                raise ValueError(
                    f"{section}.random_state must equal split_seed: "
                    f"{configured_split_seed} != {split_seed}"
                )

        base_run = self._ensure_base_run()
        base_run_dir = base_run.run_directory
        base_results = json.loads(
            (base_run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        dataset_path = _repo_path(
            self.base_config["datasets"][dataset_name]["data_path"]
        )
        dataset_hash = _file_sha256(dataset_path)
        code_hash = _implementation_hash()
        resolved = {
            "schema_version": SCHEMA_VERSION,
            "config": self.document,
            "base_config_hash": benchmark_config_hash(self.base_config),
            "dataset_sha256": dataset_hash,
            "implementation_hash": code_hash,
            "fold_limit": fold_limit,
            "subject_limit": subject_limit,
            "methods": list(selected_methods),
            "max_epochs_override": max_epochs,
        }
        config_hash = _canonical_hash(resolved)
        root = _repo_path(
            output_dir
            if output_dir is not None
            else experiment_config["output_dir"]
        )
        root.mkdir(parents=True, exist_ok=True)
        progress_path = root / "progress.json"
        resume_enabled = bool(resume or experiment_config.get("resume", False))
        completed_keys: set[str] = set()
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("config_hash") != config_hash:
                if resume_enabled:
                    raise RuntimeError(
                        "Resume config hash does not match the existing run"
                    )
                raise FileExistsError(
                    f"Output directory already contains another run: {root}"
                )
            if progress.get("dataset_sha256") != dataset_hash:
                raise RuntimeError("Resume dataset fingerprint mismatch")
            if progress.get("implementation_hash") != code_hash:
                raise RuntimeError("Resume implementation hash mismatch")
            completed_keys = set(progress.get("condition_keys", ()))
            if (
                resume_enabled
                and progress.get("status") == COMPLETED_STATUS
                and (root / "run_manifest.json").is_file()
            ):
                manifest = json.loads(
                    (root / "run_manifest.json").read_text(encoding="utf-8")
                )
                manifest["resumed"] = True
                manifest["resume_skipped_completed_conditions"] = len(
                    completed_keys
                )
                return manifest
        elif resume_enabled or not progress_path.exists():
            _write_json(progress_path, {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "config_hash": config_hash,
                "dataset_sha256": dataset_hash,
                "implementation_hash": code_hash,
                "condition_keys": [],
                "completed_conditions": 0,
                "failed_conditions": 0,
            })

        runner = BenchmarkRunner(deepcopy(self.base_config))
        data = runner.load_dataset(dataset_name)
        if (
            data.data.shape != (43174, 448)
            or data.labels.shape != (43174, 7)
            or len(np.unique(data.subject_ids)) != 53
        ):
            raise RuntimeError(
                "Canonical PM complete-case contract changed: "
                f"X={data.data.shape}, y={data.labels.shape}, "
                f"subjects={len(np.unique(data.subject_ids))}"
            )
        task = get_task(
            task_name, data, self.base_config.get("task_config", {})
        )
        evaluation = self.base_config["evaluation"]
        folds = CrossValidator(task).run_group_kfold(
            group_column=evaluation["group_column"],
            n_splits=int(evaluation.get("n_splits", 5)),
            random_state=int(evaluation.get("random_state", 42)),
            precomputed_fold_column=evaluation.get("precomputed_fold_column"),
        )
        configured_folds = evaluation.get("folds")
        if configured_folds:
            allowed = {f"fold_{int(value):02d}" for value in configured_folds}
            folds = {key: value for key, value in folds.items() if key in allowed}
        if fold_limit is not None:
            folds = dict(list(folds.items())[: int(fold_limit)])

        subject_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        failure_rows: list[dict[str, Any]] = []
        for condition_path in root.rglob("condition_result.json"):
            result = json.loads(condition_path.read_text(encoding="utf-8"))
            key = str(result["condition_key"])
            if result.get("status") == COMPLETED_STATUS:
                completed_keys.add(key)
                subject_rows.append(dict(result["subject_metrics"]))
                split_rows.append(dict(result["split_audit"]))
                checkpoint_rows.append(dict(result["checkpoint_audit"]))
                calibration_rows.extend(result.get("calibration_parameters", ()))
                prediction_frames.append(pd.read_parquet(
                    condition_path.parent / "predictions.parquet"
                ))
        failures_path = root / "failures.csv"
        if failures_path.is_file():
            failure_rows = pd.read_csv(failures_path).to_dict("records")

        global_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        personalization_seconds = 0.0

        def persist(status: str = "running") -> None:
            subjects = pd.DataFrame(subject_rows)
            subjects.to_csv(
                root / "personalization_subject_metrics.csv", index=False
            )
            pd.DataFrame(split_rows).to_csv(
                root / "calibration_split_audit.csv", index=False
            )
            pd.DataFrame(checkpoint_rows).to_csv(
                root / "checkpoint_audit.csv", index=False
            )
            pd.DataFrame(calibration_rows).to_csv(
                root / "calibration_parameters.csv", index=False
            )
            pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
            if prediction_frames:
                predictions = pd.concat(
                    prediction_frames, ignore_index=True
                )
                key = [
                    "subject_id", "sample_id", "outer_fold",
                    "method", "target_name",
                ]
                if predictions.duplicated(key).any():
                    raise RuntimeError("Duplicate unified prediction keys")
                predictions.to_parquet(root / "predictions.parquet", index=False)
            _write_json(progress_path, {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "config_hash": config_hash,
                "dataset_sha256": dataset_hash,
                "implementation_hash": code_hash,
                "condition_keys": sorted(completed_keys),
                "completed_conditions": len(completed_keys),
                "failed_conditions": len(failure_rows),
                "updated_at": datetime.now().isoformat(),
            })

        persist()
        for fold_name, split in folds.items():
            if require_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            if split.metadata.get("subject_overlap"):
                raise RuntimeError(f"Outer subject overlap in {fold_name}")
            fold_result = base_results[dataset_name]["models"][task_name][
                model_name
            ]["group_kfold_subject"]["folds"][fold_name]
            checkpoint = _repo_path(fold_result["artifacts"]["model"])
            checkpoint_sha = _file_sha256(checkpoint)
            base_adapter = self._load_fold_adapter(checkpoint, model_name)
            if require_cuda and base_adapter.device_.type != "cuda":
                raise RuntimeError(
                    f"{fold_name} loaded on {base_adapter.device_}, expected CUDA"
                )
            global_hash = _state_digest(base_adapter)
            validation_split = base_adapter.validation_split_ or {}
            inner_train_subjects = set(
                map(str, validation_split.get("inner_train_subject_ids", ()))
            )
            inner_validation_subjects = set(
                map(
                    str,
                    validation_split.get("inner_validation_subject_ids", ()),
                )
            )
            outer_test_subjects = set(
                np.unique(split.subject_test).astype(str).tolist()
            )
            if inner_train_subjects & outer_test_subjects:
                raise RuntimeError("Inner train contains outer-test subjects")
            if inner_validation_subjects & outer_test_subjects:
                raise RuntimeError("Inner validation contains outer-test subjects")
            preprocessing_state = base_adapter.get_feature_preprocessing_state()
            preprocessing_hash = _canonical_hash(
                preprocessing_state or {"strategy": "legacy"}
            )
            global_training = fold_result.get("training", {})
            global_rows.append({
                "outer_fold": fold_name,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "checkpoint": str(checkpoint),
                "global_checkpoint_hash": global_hash,
                "global_checkpoint_file_hash": checkpoint_sha,
                "global_model_state_hash": global_hash,
                "n_outer_train_subjects": int(len(np.unique(split.subject_train))),
                "n_outer_test_subjects": int(len(outer_test_subjects)),
                "outer_train_test_overlap": int(
                    len(set(split.subject_train) & set(split.subject_test))
                ),
                "inner_train_outer_test_overlap": int(
                    len(inner_train_subjects & outer_test_subjects)
                ),
                "inner_validation_outer_test_overlap": int(
                    len(inner_validation_subjects & outer_test_subjects)
                ),
                "preprocessor_hash": preprocessing_hash,
                "preprocessor_fit_subject_hash": hashlib.sha256(
                    _json_text(sorted(inner_train_subjects)).encode("utf-8")
                ).hexdigest(),
                "outer_test_subject_hash": hashlib.sha256(
                    _json_text(sorted(outer_test_subjects)).encode("utf-8")
                ).hexdigest(),
                "fit_target_overlap": int(
                    len(inner_train_subjects & outer_test_subjects)
                ),
                "epochs_trained": global_training.get("epochs_trained"),
                "best_epoch": global_training.get("best_epoch"),
                "best_validation_loss": global_training.get(
                    "best_validation_loss"
                ),
                "global_training_time_seconds": fold_result.get(
                    "training_time", 0.0
                ),
                "device_type": global_training.get("device"),
                "device_name": global_training.get("device_name"),
                "peak_gpu_memory_bytes": global_training.get(
                    "peak_gpu_memory_bytes", 0
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            })
            test_metadata = runner._partition_sequence_metadata(split, "test")
            subjects = sorted(outer_test_subjects)
            configured_subjects = set(map(
                str, calibration.get("target_subjects", ())
            ))
            if configured_subjects:
                subjects = [value for value in subjects if value in configured_subjects]
            if subject_limit is not None:
                subjects = subjects[: int(subject_limit)]

            for subject_id in subjects:
                subject_mask = (
                    np.asarray(split.subject_test).astype(str) == subject_id
                )
                subject_X = np.asarray(split.X_test)[subject_mask]
                subject_y = np.asarray(split.y_test)[subject_mask]
                metadata = test_metadata.loc[subject_mask].reset_index(drop=True)
                sources = sorted(metadata["source"].astype(str).unique())
                source = sources[0] if len(sources) == 1 else "both"
                split_spec = CalibrationSpec(
                    method="zero_shot",
                    budget_seconds=None,
                    budget_fraction=budget,
                    fraction_allocation="global_prefix",
                    purge_windows=0,
                    minimum_calibration_samples=int(
                        calibration.get("minimum_calibration_samples", 5)
                    ),
                    minimum_evaluation_samples=int(
                        calibration.get("minimum_evaluation_samples", 20)
                    ),
                )
                partition = chronological_window_partition(
                    subject_X,
                    subject_y,
                    metadata,
                    split_spec,
                    window_seconds=float(calibration.get("window_seconds", 10.0)),
                    max_gap_seconds=float(
                        calibration.get("max_gap_seconds", 10.5)
                    ),
                )
                adaptation_spec = CalibrationSpec(
                    method="zero_shot",
                    budget_seconds=None,
                    budget_fraction=1.0 - float(
                        calibration.get("adaptation_validation_fraction", 0.20)
                    ),
                    fraction_allocation="global_prefix",
                    purge_windows=0,
                    minimum_calibration_samples=1,
                    minimum_evaluation_samples=1,
                )
                adaptation = chronological_window_partition(
                    partition.calibration_X,
                    partition.calibration_y,
                    partition.calibration_metadata,
                    adaptation_spec,
                    window_seconds=float(calibration.get("window_seconds", 10.0)),
                    max_gap_seconds=float(
                        calibration.get("max_gap_seconds", 10.5)
                    ),
                )
                calibration_ids = set(
                    partition.calibration_metadata["sample_id"].astype(str)
                )
                evaluation_ids = set(
                    partition.evaluation_metadata["sample_id"].astype(str)
                )
                adaptation_train_ids = set(
                    adaptation.calibration_metadata["sample_id"].astype(str)
                )
                adaptation_validation_ids = set(
                    adaptation.evaluation_metadata["sample_id"].astype(str)
                )
                split_audit = {
                    "outer_fold": fold_name,
                    "subject_id": subject_id,
                    "source": source,
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "total_target_samples": int(len(subject_X)),
                    "calibration_pool_samples": int(len(partition.calibration_X)),
                    "adaptation_train_samples": int(
                        len(adaptation.calibration_X)
                    ),
                    "adaptation_validation_samples": int(
                        len(adaptation.evaluation_X)
                    ),
                    "evaluation_samples": int(len(partition.evaluation_X)),
                    "calibration_evaluation_overlap": int(
                        len(calibration_ids & evaluation_ids)
                    ),
                    "adaptation_train_validation_overlap": int(
                        len(adaptation_train_ids & adaptation_validation_ids)
                    ),
                    "adaptation_evaluation_overlap": int(
                        len(
                            (adaptation_train_ids | adaptation_validation_ids)
                            & evaluation_ids
                        )
                    ),
                    "target_in_global_inner_train": (
                        subject_id in inner_train_subjects
                    ),
                    "target_in_global_inner_validation": (
                        subject_id in inner_validation_subjects
                    ),
                    "duplicate_sample_ids": int(
                        metadata["sample_id"].astype(str).duplicated().sum()
                    ),
                    "outer_train_subject_hash": _ordered_hash(
                        sorted(np.unique(split.subject_train).astype(str))
                    ),
                    "inner_train_subject_hash": _ordered_hash(
                        sorted(inner_train_subjects)
                    ),
                    "inner_validation_subject_hash": _ordered_hash(
                        sorted(inner_validation_subjects)
                    ),
                    "calibration_sample_hash": _ordered_hash(
                        sorted(calibration_ids)
                    ),
                    "adaptation_train_sample_hash": _ordered_hash(
                        sorted(adaptation_train_ids)
                    ),
                    "adaptation_validation_sample_hash": _ordered_hash(
                        sorted(adaptation_validation_ids)
                    ),
                    "evaluation_sample_hash": _ordered_hash(
                        sorted(evaluation_ids)
                    ),
                    "preprocessor_hash": preprocessing_hash,
                    "sort_order": "source,record_id,t_start,sample_id",
                }
                base_adaptation_predictions = base_adapter.predict(
                    adaptation.calibration_X
                )
                base_evaluation_predictions = base_adapter.predict(
                    partition.evaluation_X
                )
                if (
                    base_adaptation_predictions.shape
                    != adaptation.calibration_y.shape
                    or base_evaluation_predictions.shape
                    != partition.evaluation_y.shape
                ):
                    raise RuntimeError("Regression output shape is not (n, 7)")

                for method in selected_methods:
                    condition_key = "|".join([
                        fold_name, subject_id, method,
                        f"seed={model_seed}", f"budget={budget:.4f}",
                    ])
                    if condition_key in completed_keys:
                        LOGGER.info("Resume skip %s", condition_key)
                        continue
                    condition_dir = (
                        root / fold_name / _safe_component(subject_id) / method
                    )
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    condition_started = time.perf_counter()
                    status = COMPLETED_STATUS
                    if len(partition.calibration_X) < int(
                        calibration.get("minimum_calibration_samples", 5)
                    ):
                        status = "insufficient_calibration_samples"
                    elif len(adaptation.calibration_X) < int(
                        calibration.get("minimum_adaptation_train_samples", 4)
                    ):
                        status = "insufficient_adaptation_train"
                    elif len(partition.evaluation_X) < int(
                        calibration.get("minimum_evaluation_samples", 20)
                    ):
                        status = "insufficient_evaluation_samples"
                    adapted = base_adapter.clone()
                    initial_hash = _state_digest(adapted)
                    trainable, frozen, trainable_count, frozen_count = (
                        _parameter_audit(adapted, method)
                    )
                    frozen_hash_before = _parameter_digest(adapted, frozen)
                    training_log = pd.DataFrame()
                    parameters: list[dict[str, Any]] = []
                    predictions_after = base_evaluation_predictions.copy()
                    try:
                        if status == COMPLETED_STATUS:
                            if method == "bias_correction":
                                bias = fit_bias_correction(
                                    adaptation.calibration_y,
                                    base_adaptation_predictions,
                                )
                                predictions_after = apply_bias_correction(
                                    base_evaluation_predictions, bias
                                )
                                parameters = [{
                                    "outer_fold": fold_name,
                                    "subject_id": subject_id,
                                    "method": method,
                                    "target_name": target,
                                    "bias": float(bias[index]),
                                    "n_fit_samples": int(
                                        len(adaptation.calibration_y)
                                    ),
                                } for index, target in enumerate(CANONICAL_TARGETS)]
                            elif method == "affine_calibration":
                                affine = fit_affine_calibration(
                                    adaptation.calibration_y,
                                    base_adaptation_predictions,
                                    alpha=float(
                                        calibration.get("affine_alpha", 1.0)
                                    ),
                                )
                                predictions_after = apply_affine_calibration(
                                    base_evaluation_predictions, affine
                                )
                                parameters = [{
                                    "outer_fold": fold_name,
                                    "subject_id": subject_id,
                                    "method": method,
                                    **value,
                                } for value in affine.parameters]
                            elif method in {"head_only", "full_model"}:
                                method_config = dict(
                                    calibration.get("method_params", {}).get(
                                        method, {}
                                    )
                                )
                                adapted.fine_tune(
                                    adaptation.calibration_X,
                                    adaptation.calibration_y,
                                    mode=method,
                                    X_validation=adaptation.evaluation_X,
                                    y_validation=adaptation.evaluation_y,
                                    max_epochs=(
                                        int(max_epochs)
                                        if max_epochs is not None
                                        else int(method_config["max_epochs"])
                                    ),
                                    learning_rate=float(
                                        method_config["learning_rate"]
                                    ),
                                    weight_decay=float(
                                        method_config.get(
                                            "weight_decay", 0.0001
                                        )
                                    ),
                                    early_stopping_patience=int(
                                        method_config.get(
                                            "early_stopping_patience", 3
                                        )
                                    ),
                                    random_state=model_seed,
                                )
                                predictions_after = adapted.predict(
                                    partition.evaluation_X
                                )
                                training_log = pd.DataFrame(
                                    adapted.training_log_
                                )
                            elif method != "zero_shot":
                                raise ValueError(f"Unknown method {method}")
                            if not np.isfinite(predictions_after).all():
                                status = "non_finite_predictions"
                    except (ValueError, RuntimeError) as exc:
                        status = (
                            "non_finite_predictions"
                            if "finite" in str(exc).lower()
                            else "training_failed"
                        )
                        failure_rows.append({
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "method": method,
                            "model_seed": model_seed,
                            "budget": budget,
                            "status": status,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        })

                    final_hash = _state_digest(adapted)
                    frozen_hash_after = _parameter_digest(adapted, frozen)
                    checkpoint_audit = {
                        "outer_fold": fold_name,
                        "subject_id": subject_id,
                        "method": method,
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "global_checkpoint_hash": global_hash,
                        "global_checkpoint_file_hash": checkpoint_sha,
                        "global_model_state_hash": global_hash,
                        "fine_tune_initial_hash": initial_hash,
                        "fine_tune_final_hash": final_hash,
                        "initial_matches_global": initial_hash == global_hash,
                        "global_state_unchanged": (
                            _state_digest(base_adapter) == global_hash
                        ),
                        "frozen_parameters_unchanged": (
                            frozen_hash_before == frozen_hash_after
                        ),
                        "trainable_parameter_count": trainable_count,
                        "frozen_parameter_count": frozen_count,
                        "preprocessor_hash": preprocessing_hash,
                        "peak_gpu_memory_bytes": int(
                            adapted.peak_gpu_memory_bytes_
                            if method in {"head_only", "full_model"}
                            else 0
                        ),
                    }
                    if status == COMPLETED_STATUS:
                        before_metrics, after_metrics, gains = _metric_bundle(
                            partition.evaluation_y,
                            base_evaluation_predictions,
                            predictions_after,
                        )
                        improvement_counts = {}
                        for metric in ("mae", "rmse", "r2", "spearman"):
                            values = np.asarray([
                                gains[f"{target}_{metric}_gain"]
                                for target in CANONICAL_TARGETS
                            ], dtype=float)
                            improvement_counts[
                                f"targets_{metric}_improved_count"
                            ] = int(np.sum(values > 0))
                        condition_training_time = (
                            time.perf_counter() - condition_started
                        )
                        subject_metrics = {
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "source": source,
                            "method": method,
                            "split_seed": split_seed,
                            "model_seed": model_seed,
                            "budget": budget,
                            "status": status,
                            "n_total_target_samples": int(len(subject_X)),
                            "n_calibration": int(len(partition.calibration_X)),
                            "n_adaptation_train": int(
                                len(adaptation.calibration_X)
                            ),
                            "n_adaptation_validation": int(
                                len(adaptation.evaluation_X)
                            ),
                            "n_final_evaluation": int(
                                len(partition.evaluation_X)
                            ),
                            "calibration_sample_count": int(
                                len(partition.calibration_X)
                            ),
                            "adaptation_train_sample_count": int(
                                len(adaptation.calibration_X)
                            ),
                            "adaptation_validation_sample_count": int(
                                len(adaptation.evaluation_X)
                            ),
                            "evaluation_sample_count": int(
                                len(partition.evaluation_X)
                            ),
                            "evaluation_sample_hash": split_audit[
                                "evaluation_sample_hash"
                            ],
                            "calibration_sample_hash": split_audit[
                                "calibration_sample_hash"
                            ],
                            "adaptation_train_sample_hash": split_audit[
                                "adaptation_train_sample_hash"
                            ],
                            "adaptation_validation_sample_hash": split_audit[
                                "adaptation_validation_sample_hash"
                            ],
                            "preprocessor_hash": preprocessing_hash,
                            "global_checkpoint_hash": global_hash,
                            "fine_tune_initial_hash": initial_hash,
                            "fine_tune_final_hash": final_hash,
                            "training_time_seconds": condition_training_time,
                            "peak_gpu_memory_bytes": checkpoint_audit[
                                "peak_gpu_memory_bytes"
                            ],
                            **{
                                f"{key}_before": value
                                for key, value in before_metrics.items()
                                if key not in {"n_samples", "n_outputs"}
                            },
                            **{
                                f"{key}_after": value
                                for key, value in after_metrics.items()
                                if key not in {"n_samples", "n_outputs"}
                            },
                            **gains,
                            **improvement_counts,
                        }
                        subject_metrics["all_targets_mae_improved"] = (
                            improvement_counts[
                                "targets_mae_improved_count"
                            ] == len(CANONICAL_TARGETS)
                        )
                        subject_metrics["majority_targets_mae_improved"] = (
                            improvement_counts[
                                "targets_mae_improved_count"
                            ] >= 4
                        )
                        predictions_frame = _prediction_frame(
                            fold_name=fold_name,
                            subject_id=subject_id,
                            source=source,
                            method=method,
                            metadata=partition.evaluation_metadata,
                            truth=partition.evaluation_y,
                            before=base_evaluation_predictions,
                            after=predictions_after,
                        )
                        adapted.save(condition_dir / "model.pt")
                        predictions_frame.to_parquet(
                            condition_dir / "predictions.parquet", index=False
                        )
                        training_log.to_csv(
                            condition_dir / "training_log.csv", index=False
                        )
                        _write_json(
                            condition_dir / "metrics.json",
                            {
                                "before": before_metrics,
                                "after": after_metrics,
                                "gains": gains,
                            },
                        )
                        _write_json(
                            condition_dir / "calibration_parameters.json",
                            parameters,
                        )
                        _write_json(
                            condition_dir / "split_audit.json", split_audit
                        )
                        _write_json(
                            condition_dir / "checkpoint_audit.json",
                            checkpoint_audit,
                        )
                        condition_result = {
                            "condition_key": condition_key,
                            "status": status,
                            "subject_metrics": subject_metrics,
                            "split_audit": {**split_audit, "method": method},
                            "checkpoint_audit": checkpoint_audit,
                            "calibration_parameters": parameters,
                        }
                        _write_json(
                            condition_dir / "condition_result.json",
                            condition_result,
                        )
                        completed_keys.add(condition_key)
                        subject_rows.append(subject_metrics)
                        split_rows.append({**split_audit, "method": method})
                        checkpoint_rows.append(checkpoint_audit)
                        calibration_rows.extend(parameters)
                        prediction_frames.append(predictions_frame)
                    else:
                        _write_json(
                            condition_dir / "condition_result.json",
                            {
                                "condition_key": condition_key,
                                "status": status,
                            },
                        )
                    elapsed = time.perf_counter() - condition_started
                    personalization_seconds += elapsed
                    LOGGER.info(
                        "%s %s %s: %s (%.2fs)",
                        fold_name, subject_id, method, status, elapsed
                    )
                    persist()

        global_frame = pd.DataFrame(global_rows).drop_duplicates("outer_fold")
        global_frame.to_csv(root / "global_fold_summary.csv", index=False)
        subject_frame = pd.DataFrame(subject_rows)
        all_sources = subject_frame.copy()
        if not all_sources.empty:
            all_sources["source"] = "all"
            aggregation_input = pd.concat(
                [subject_frame, all_sources], ignore_index=True
            )
        else:
            aggregation_input = subject_frame
        aggregate, paired, target_summary = _aggregate_outputs(
            aggregation_input,
            bootstrap_samples=int(
                self.document.get("statistics", {}).get(
                    "bootstrap_samples", 1000
                )
            ),
            bootstrap_seed=int(
                self.document.get("statistics", {}).get(
                    "bootstrap_seed", 42
                )
            ),
        )
        aggregate.to_csv(root / "aggregate_metrics.csv", index=False)
        paired.to_csv(root / "paired_comparisons.csv", index=False)
        target_summary.to_csv(root / "target_metric_summary.csv", index=False)
        total_seconds = time.perf_counter() - started
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": COMPLETED_STATUS,
            "config_path": str(self.config_path),
            "config_hash": config_hash,
            "implementation_hash": code_hash,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "dataset_shape": [int(data.data.shape[0]), int(data.data.shape[1])],
            "target_shape": [
                int(data.labels.shape[0]), int(data.labels.shape[1])
            ],
            "target_order": list(CANONICAL_TARGETS),
            "n_subjects": int(len(np.unique(data.subject_ids))),
            "base_benchmark_run": str(base_run_dir),
            "base_config_hash": benchmark_config_hash(self.base_config),
            "global_trainings": int(len(global_frame)),
            "completed_conditions": int(len(completed_keys)),
            "failed_conditions": int(len(failure_rows)),
            "methods": list(selected_methods),
            "budget": budget,
            "model_seed": model_seed,
            "split_seed": split_seed,
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU"
            ),
            "peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available() else 0
            ),
            "global_training_time_seconds": float(
                pd.to_numeric(
                    global_frame["global_training_time_seconds"],
                    errors="coerce",
                ).sum()
                if not global_frame.empty else 0.0
            ),
            "personalization_time_seconds": personalization_seconds,
            "orchestration_time_seconds": total_seconds,
            "output_dir": str(root),
            "artifacts": {
                name: str(root / name)
                for name in (
                    "run_manifest.json",
                    "progress.json",
                    "failures.csv",
                    "global_fold_summary.csv",
                    "personalization_subject_metrics.csv",
                    "target_metric_summary.csv",
                    "aggregate_metrics.csv",
                    "paired_comparisons.csv",
                    "calibration_parameters.csv",
                    "calibration_split_audit.csv",
                    "checkpoint_audit.csv",
                    "predictions.parquet",
                )
            },
        }
        _write_json(root / "run_manifest.json", manifest)
        persist(COMPLETED_STATUS)
        return manifest
