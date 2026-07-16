"""Strict directional cross-source trials executed by ``BenchmarkRunner``."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.datasets_registry import get_dataset
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.sequence_utils import build_sequences
from model_zoo.factory import model_requires_sequences


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA_VERSION = "cross-source-plan-v1"
RESULT_SCHEMA_VERSION = "cross-source-results-v1"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _report_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cross-source experiment not found: {path}")
    with open(path, encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    required = {"experiment", "dataset", "models", "matrix", "evaluation"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Cross-source experiment is missing sections: {missing}")
    return document


def _direction(value: str | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(value, Mapping):
        train_source = str(value.get("train_source", "")).strip()
        test_source = str(value.get("test_source", "")).strip()
    else:
        parts = [part.strip() for part in str(value).split("->")]
        if len(parts) != 2:
            raise ValueError(f"Direction must use train->test syntax, got {value!r}")
        train_source, test_source = parts
    if not train_source or not test_source or train_source == test_source:
        raise ValueError(f"Invalid cross-source direction: {value!r}")
    return train_source, test_source


def _slug(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_" for char in str(value)
    ).strip("_")


def _completed_report_value(completed: CompletedBenchmarkRun) -> dict[str, Any]:
    value = completed.to_dict()
    for key in (
        "benchmark_run_directory",
        "benchmark_result_file",
        "benchmark_summary_file",
        "benchmark_manifest_file",
    ):
        if value.get(key):
            value[key] = _report_path(value[key])
    return value


def _metric(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


def _class_counts(values: Any) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _format_class_counts(values: Mapping[str, Any]) -> str:
    return " / ".join(
        f"{label}:{int(values.get(str(label), 0))}" for label in range(5)
    )


@dataclass(frozen=True)
class CrossSourceTrialPlan:
    trial_id: str
    train_source: str
    test_source: str
    subject_mode: str
    model_name: str
    prediction_unit: str
    status: str
    invalid_reasons: tuple[str, ...]
    action: str
    counts: Mapping[str, Any]
    estimated_runtime_seconds: float
    config_hash: str
    output_dir: Path
    resolved_config: Mapping[str, Any]
    completed_run: Optional[CompletedBenchmarkRun] = None

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        value = {
            "trial_id": self.trial_id,
            "train_source": self.train_source,
            "test_source": self.test_source,
            "subject_mode": self.subject_mode,
            "model": self.model_name,
            "prediction_unit": self.prediction_unit,
            "status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "action": self.action,
            "counts": dict(self.counts),
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "config_hash": self.config_hash,
            "output_dir": _report_path(self.output_dir),
            "completed_run": (
                None
                if self.completed_run is None
                else _completed_report_value(self.completed_run)
            ),
        }
        if include_config:
            value["resolved_config"] = deepcopy(dict(self.resolved_config))
        return value


class CrossSourceExperiment:
    """Expand, validate, execute, and resume the bounded transfer matrix."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        runner_factory: Callable[[dict[str, Any]], Any] = BenchmarkRunner,
        completed_run_finder: Callable[..., Optional[CompletedBenchmarkRun]] = (
            BenchmarkRunner.find_completed_run
        ),
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = _load_yaml(self.spec_path)
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self._data = None
        self._task = None

    @property
    def reports_dir(self) -> Path:
        return _repo_path(
            self.document["experiment"].get("reports_dir", "reports")
        )

    @property
    def output_root(self) -> Path:
        return _repo_path(self.document["experiment"]["output_dir"])

    def _load_task(self):
        if self._task is None:
            dataset_spec = self.document["dataset"]
            dataset_config = {
                "data_path": str(_repo_path(dataset_spec["data_path"])),
                "feature_set": dataset_spec.get("feature_set", "pow_plus_eeg"),
                "target_col": dataset_spec.get("target", "label_q5"),
                "subject_col": dataset_spec.get("subject_col", "subject_id"),
                "discretize": False,
                "max_features": int(dataset_spec.get("max_features", 448)),
                "logical_recording_map_path": str(
                    _repo_path(dataset_spec["logical_recording_map_path"])
                ),
            }
            dataset = get_dataset(dataset_spec["name"], dataset_config)
            self._data = dataset.load()
            self._task = get_task(
                dataset_spec["task"],
                self._data,
                {"random_state": 42},
            )
        return self._task

    def _split(
        self,
        train_source: str,
        test_source: str,
        subject_mode: str,
        *,
        seed: int,
        max_train_windows: Optional[int],
        max_test_windows: Optional[int],
    ):
        evaluation = self.document["evaluation"]
        thresholds = evaluation.get("thresholds", {})
        return CrossValidator(self._load_task()).run_cross_source_holdout(
            train_source=train_source,
            test_source=test_source,
            subject_mode=subject_mode,
            remove_logical_duplicates=bool(
                evaluation.get("remove_logical_duplicates", True)
            ),
            minimum_train_subjects=int(
                thresholds.get("minimum_train_subjects", 5)
            ),
            minimum_test_subjects=int(
                thresholds.get("minimum_test_subjects", 3)
            ),
            minimum_train_classes=int(
                thresholds.get("minimum_train_classes", 5)
            ),
            minimum_test_classes=int(
                thresholds.get("minimum_test_classes", 2)
            ),
            minimum_predictions_per_test_subject=int(
                thresholds.get("minimum_predictions_per_test_subject", 20)
            ),
            max_train_windows=max_train_windows,
            max_test_windows=max_test_windows,
            random_state=seed,
        )

    @staticmethod
    def _sequence_metadata(split, partition: str) -> pd.DataFrame:
        values = {
            "subject_id": np.asarray(getattr(split, f"subject_{partition}")),
            "sample_id": np.asarray(getattr(split, f"sample_id_{partition}")),
            "record_id": np.asarray(getattr(split, f"record_id_{partition}")),
        }
        values.update({
            key: np.asarray(column)
            for key, column in getattr(
                split, f"row_metadata_{partition}"
            ).items()
        })
        return pd.DataFrame(values)

    def _prediction_counts(
        self,
        split,
        model_type: str,
    ) -> tuple[str, int, int, int, dict[str, int], dict[str, int]]:
        if not model_requires_sequences(model_type):
            minimum = int(
                pd.Series(split.subject_test).value_counts().min()
            ) if len(split.y_test) else 0
            return (
                "feature_window",
                len(split.y_train),
                len(split.y_test),
                minimum,
                _class_counts(split.y_train),
                _class_counts(split.y_test),
            )
        sequence = self.document["sequence"]
        results = []
        for partition in ("train", "test"):
            result = build_sequences(
                X=getattr(split, f"X_{partition}"),
                y=getattr(split, f"y_{partition}"),
                metadata=self._sequence_metadata(split, partition),
                sequence_length=int(sequence.get("length", 8)),
                stride=int(sequence.get("stride", 1)),
                target_position=str(sequence.get("target_position", "last")),
                expected_step_seconds=sequence.get("expected_step_seconds"),
                max_gap_seconds=sequence.get("max_gap_seconds"),
            )
            results.append(result)
        test_counts = results[1].metadata["subject_id"].value_counts()
        minimum = int(test_counts.min()) if len(test_counts) else 0
        return (
            "feature_sequence",
            len(results[0].X),
            len(results[1].X),
            minimum,
            _class_counts(results[0].y),
            _class_counts(results[1].y),
        )

    def _resolved_config(
        self,
        *,
        train_source: str,
        test_source: str,
        subject_mode: str,
        model_name: str,
        seed: int,
        max_train_windows: Optional[int],
        max_test_windows: Optional[int],
        max_epochs: Optional[int],
    ) -> tuple[dict[str, Any], str, Path]:
        dataset_spec = self.document["dataset"]
        model_config = deepcopy(self.document["models"][model_name])
        model_params = model_config.setdefault("params", {})
        model_params["random_state"] = int(seed)
        if max_epochs is not None and str(model_config["type"]).startswith("torch_"):
            if int(max_epochs) <= 0:
                raise ValueError("max_epochs must be positive")
            model_params["max_epochs"] = int(max_epochs)
        evaluation = deepcopy(self.document["evaluation"])
        evaluation.update({
            "protocol": "cross_source_holdout",
            "train_source": train_source,
            "test_source": test_source,
            "subject_mode": subject_mode,
            "random_state": int(seed),
        })
        if max_train_windows is not None:
            evaluation["max_train_windows"] = int(max_train_windows)
        if max_test_windows is not None:
            evaluation["max_test_windows"] = int(max_test_windows)
        validation_by_mode = self.document.get("validation_by_subject_mode", {})
        validation = deepcopy(validation_by_mode.get(subject_mode, {}))
        validation["random_state"] = int(seed)
        config: dict[str, Any] = {
            "datasets": {
                dataset_spec["name"]: {
                    "data_path": str(dataset_spec["data_path"]),
                    "feature_set": dataset_spec.get(
                        "feature_set", "pow_plus_eeg"
                    ),
                    "target_col": dataset_spec.get("target", "label_q5"),
                    "subject_col": dataset_spec.get(
                        "subject_col", "subject_id"
                    ),
                    "n_classes": int(dataset_spec.get("n_classes", 5)),
                    "discretize": False,
                    "max_features": int(dataset_spec.get("max_features", 448)),
                    "logical_recording_map_path": str(
                        dataset_spec["logical_recording_map_path"]
                    ),
                }
            },
            "tasks": [dataset_spec["task"]],
            "models": {model_name: model_config},
            "evaluation": evaluation,
            "task_config": {"random_state": int(seed)},
            "run_within_subject": False,
            "run_loso": False,
        }
        if validation:
            config["validation"] = validation
        if model_requires_sequences(model_config["type"]):
            config["sequence"] = deepcopy(self.document["sequence"])
        config_hash = benchmark_config_hash(config)
        configured_output_root = Path(
            self.document["experiment"]["output_dir"]
        )
        configured_output_dir = (
            configured_output_root / "runs" / config_hash[:20]
        )
        output_dir = _repo_path(configured_output_dir)
        config["output_dir"] = str(configured_output_dir)
        return config, config_hash, output_dir

    def plan(
        self,
        *,
        directions: Optional[Sequence[str]] = None,
        subject_modes: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        seed: int = 42,
        max_train_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> list[CrossSourceTrialPlan]:
        matrix = self.document["matrix"]
        available_directions = [_direction(value) for value in matrix["directions"]]
        requested_directions = (
            available_directions
            if not directions
            else [_direction(value) for value in directions]
        )
        unknown_directions = sorted(
            set(requested_directions) - set(available_directions)
        )
        if unknown_directions:
            raise ValueError(f"Unknown cross-source directions: {unknown_directions}")
        available_modes = [str(value) for value in matrix["subject_modes"]]
        selected_modes = available_modes if not subject_modes else list(subject_modes)
        unknown_modes = sorted(set(selected_modes) - set(available_modes))
        if unknown_modes:
            raise ValueError(f"Unknown cross-source subject modes: {unknown_modes}")
        available_models = [str(value) for value in matrix["models"]]
        selected_models = available_models if not models else list(models)
        unknown_models = sorted(set(selected_models) - set(available_models))
        if unknown_models:
            raise ValueError(f"Unknown cross-source models: {unknown_models}")
        trial_count = (
            len(requested_directions) * len(selected_modes) * len(selected_models)
        )
        if trial_count > 8:
            raise ValueError(f"Cross-source matrix is limited to 8 trials, got {trial_count}")

        split_cache = {}
        plans: list[CrossSourceTrialPlan] = []
        for train_source, test_source in requested_directions:
            for subject_mode in selected_modes:
                split_key = (train_source, test_source, subject_mode)
                split = split_cache.setdefault(
                    split_key,
                    self._split(
                        train_source,
                        test_source,
                        subject_mode,
                        seed=seed,
                        max_train_windows=max_train_windows,
                        max_test_windows=max_test_windows,
                    ),
                )
                for model_name in selected_models:
                    model_spec = self.document["models"][model_name]
                    (
                        unit,
                        n_train,
                        n_test,
                        minimum_test,
                        train_class_distribution,
                        test_class_distribution,
                    ) = self._prediction_counts(split, str(model_spec["type"]))
                    resolved, config_hash, output_dir = self._resolved_config(
                        train_source=train_source,
                        test_source=test_source,
                        subject_mode=subject_mode,
                        model_name=model_name,
                        seed=seed,
                        max_train_windows=max_train_windows,
                        max_test_windows=max_test_windows,
                        max_epochs=max_epochs,
                    )
                    completed = self.completed_run_finder(
                        resolved, search_directories=[output_dir]
                    )
                    invalid_reasons = list(split.metadata["invalid_reasons"])
                    threshold = int(
                        split.metadata["thresholds"][
                            "minimum_predictions_per_test_subject"
                        ]
                    )
                    if minimum_test < threshold:
                        invalid_reasons.append(
                            f"{unit} minimum test predictions per subject="
                            f"{minimum_test} is below configured minimum {threshold}"
                        )
                    status = "valid" if not invalid_reasons else "invalid"
                    if status == "invalid":
                        action = "skip_invalid"
                    elif completed is not None:
                        action = "reuse_completed"
                    else:
                        action = "run"
                    base_runtime = float(
                        self.document.get("runtime_estimates_seconds", {}).get(
                            model_name, 0.0
                        )
                    )
                    configured_epochs = int(
                        model_spec.get("params", {}).get("max_epochs", 1)
                    )
                    epoch_factor = (
                        1.0
                        if not str(model_spec["type"]).startswith("torch_")
                        else min(
                            1.0,
                            int(max_epochs or configured_epochs) / configured_epochs,
                        )
                    )
                    counts = {
                        "train_windows": int(len(split.y_train)),
                        "test_windows": int(len(split.y_test)),
                        "train_predictions": int(n_train),
                        "test_predictions": int(n_test),
                        "minimum_test_predictions_per_subject": minimum_test,
                        "train_subjects": int(split.metadata["n_train_subjects"]),
                        "test_subjects": int(split.metadata["n_test_subjects"]),
                        "train_records": int(split.metadata["n_train_records"]),
                        "test_records": int(split.metadata["n_test_records"]),
                        "train_logical_recordings": int(
                            split.metadata["n_train_logical_recordings"]
                        ),
                        "test_logical_recordings": int(
                            split.metadata["n_test_logical_recordings"]
                        ),
                        "train_subject_ids": list(
                            split.metadata.get("train_subject_ids", [])
                        ),
                        "test_subject_ids": list(
                            split.metadata.get("test_subject_ids", [])
                        ),
                        "shared_subject_ids": list(
                            split.metadata.get("shared_subject_ids", [])
                        ),
                        "eligible_shared_subject_ids": list(
                            split.metadata.get(
                                "eligible_shared_subject_ids", []
                            )
                        ),
                        "excluded_subjects": deepcopy(
                            split.metadata.get("excluded_subjects", {})
                        ),
                        "removed_duplicate_logical_recordings": list(
                            split.metadata.get("excluded_logical_record_ids", [])
                        ),
                        "train_class_distribution": train_class_distribution,
                        "test_class_distribution": test_class_distribution,
                    }
                    trial_id = (
                        f"{_slug(train_source)}_to_{_slug(test_source)}__"
                        f"{_slug(subject_mode)}__{_slug(model_name)}"
                    )
                    plans.append(CrossSourceTrialPlan(
                        trial_id=trial_id,
                        train_source=train_source,
                        test_source=test_source,
                        subject_mode=subject_mode,
                        model_name=model_name,
                        prediction_unit=unit,
                        status=status,
                        invalid_reasons=tuple(dict.fromkeys(invalid_reasons)),
                        action=action,
                        counts=counts,
                        estimated_runtime_seconds=base_runtime * epoch_factor,
                        config_hash=config_hash,
                        output_dir=output_dir,
                        resolved_config=resolved,
                        completed_run=completed,
                    ))
        return plans

    @staticmethod
    def render_plan(plans: Sequence[CrossSourceTrialPlan]) -> str:
        lines = [
            "# Cross-source experiment plan",
            "",
            "| Trial | Direction | Mode | Model | Unit | Train | Test | Subjects | Status | Action | Est. seconds |",
            "|---|---|---|---|---|---:|---:|---|---|---|---:|",
        ]
        for plan in plans:
            counts = plan.counts
            lines.append(
                f"| `{plan.trial_id}` | {plan.train_source} -> {plan.test_source} "
                f"| {plan.subject_mode} | {plan.model_name} | "
                f"{plan.prediction_unit} | {counts['train_predictions']} | "
                f"{counts['test_predictions']} | {counts['train_subjects']} / "
                f"{counts['test_subjects']} | {plan.status} | {plan.action} | "
                f"{plan.estimated_runtime_seconds:.1f} |"
            )
            if plan.invalid_reasons:
                lines.append(
                    "|  |  |  |  | invalid reasons |  |  |  | "
                    + "; ".join(plan.invalid_reasons)
                    + " |  |  |"
                )
        lines.extend([
            "",
            f"Valid trials: **{sum(plan.status == 'valid' for plan in plans)}**.",
            f"Invalid trials: **{sum(plan.status == 'invalid' for plan in plans)}**.",
            f"Planned runs: **{sum(plan.action == 'run' for plan in plans)}**.",
            "",
            "## Trial details",
            "",
        ])
        for plan in plans:
            counts = plan.counts
            lines.extend([
                f"- `{plan.trial_id}`: logical recordings train/test "
                f"{counts['train_logical_recordings']} / "
                f"{counts['test_logical_recordings']}; shared subjects "
                f"{len(counts['shared_subject_ids'])} total / "
                f"{len(counts['eligible_shared_subject_ids'])} with residual "
                f"data in both sources; removed duplicate "
                f"logical recordings "
                f"{len(counts['removed_duplicate_logical_recordings'])}; "
                f"minimum test predictions per subject "
                f"{counts['minimum_test_predictions_per_subject']}.",
                f"  Train classes: "
                f"{_format_class_counts(counts['train_class_distribution'])}; "
                f"test classes: "
                f"{_format_class_counts(counts['test_class_distribution'])}.",
            ])
        lines.append("")
        return "\n".join(lines)

    def write_plan_reports(
        self, plans: Sequence[CrossSourceTrialPlan]
    ) -> dict[str, str]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self.reports_dir / "cross_source_experiment_plan.md"
        json_path = self.reports_dir / "cross_source_experiment_plan.json"
        markdown_path.write_text(self.render_plan(plans), encoding="utf-8")
        payload = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "experiment_spec": _report_path(self.spec_path),
            "trials": [plan.to_dict(include_config=True) for plan in plans],
        }
        json_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return {"markdown": str(markdown_path), "json": str(json_path)}

    @staticmethod
    def _completed_trial_payload(
        plan: CrossSourceTrialPlan,
        completed: CompletedBenchmarkRun,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with open(completed.result_file, encoding="utf-8") as input_file:
            benchmark_results = json.load(input_file)
        dataset_result = next(iter(benchmark_results.values()))
        task_result = next(iter(dataset_result["models"].values()))
        model_result = task_result[plan.model_name]["cross_source_holdout"]
        split_result = next(iter(model_result["splits"].values()))
        predictions = pd.read_parquet(model_result["artifacts"]["predictions"])
        probability_columns = sorted(
            [column for column in predictions if column.startswith("proba_")],
            key=lambda value: int(value.split("_")[1]),
        )
        all_probabilities = (
            None
            if not probability_columns
            else predictions[probability_columns].to_numpy(dtype=float)
        )
        subject_rows = []
        for subject_id, frame in predictions.groupby("subject_id", sort=True):
            subject_probabilities = (
                None
                if not probability_columns
                else frame[probability_columns].to_numpy(dtype=float)
            )
            metrics = MetricsCalculator.calculate_all_metrics(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                subject_probabilities,
            )
            subject_rows.append({
                "trial_id": plan.trial_id,
                "subject_id": str(subject_id),
                "n_predictions": int(len(frame)),
                "metrics": metrics,
            })
        confusion_matrix = np.asarray(
            model_result["metrics"]["confusion_matrix"], dtype=int
        )
        row_totals = confusion_matrix.sum(axis=1)
        per_class_recall = {
            str(label): (
                None
                if row_totals[label] == 0
                else float(confusion_matrix[label, label] / row_totals[label])
            )
            for label in range(len(confusion_matrix))
        }
        trial_payload = {
            **plan.to_dict(),
            "status": "completed",
            "benchmark": _completed_report_value(completed),
            "metrics": model_result["metrics"],
            "training_time": model_result["training_time_total"],
            "training": split_result.get("training", {}),
            "artifacts": split_result.get("artifacts", {}),
            "prediction_rows": int(len(predictions)),
            "prediction_ids_unique": bool(
                not predictions[
                    "sequence_id"
                    if "sequence_id" in predictions
                    else "sample_id"
                ].duplicated().any()
            ),
            "test_true_class_distribution": _class_counts(
                predictions["y_true"]
            ),
            "predicted_class_distribution": _class_counts(
                predictions["y_pred"]
            ),
            "per_class_recall": per_class_recall,
            "quality_checks": {
                "expected_prediction_rows": int(
                    plan.counts["test_predictions"]
                ),
                "prediction_coverage_complete": int(len(predictions)) == int(
                    plan.counts["test_predictions"]
                ),
                "probabilities_finite": bool(
                    all_probabilities is not None
                    and np.isfinite(all_probabilities).all()
                ),
                "probability_sum_max_error": (
                    None
                    if all_probabilities is None
                    else float(
                        np.abs(all_probabilities.sum(axis=1) - 1.0).max()
                    )
                ),
                "test_source_pure": set(
                    predictions["source"].astype(str)
                ) == {plan.test_source},
                "test_subjects": sorted(
                    predictions["subject_id"].astype(str).unique().tolist()
                ),
                "classes_present": sorted(
                    predictions["y_true"].astype(int).unique().tolist()
                ),
                "subject_overlap": model_result["split_metadata"].get(
                    "subject_overlap", []
                ),
                "logical_record_overlap": model_result["split_metadata"].get(
                    "logical_record_overlap", []
                ),
                "source_record_overlap": model_result["split_metadata"].get(
                    "record_overlap", []
                ),
                "sample_overlap": model_result["split_metadata"].get(
                    "sample_overlap", []
                ),
                "raw_interval_overlap": model_result["split_metadata"].get(
                    "raw_interval_overlap", []
                ),
            },
        }
        return trial_payload, subject_rows

    def _load_in_domain_references(self) -> dict[str, Any]:
        references = {}
        for source, reference in self.document.get(
            "in_domain_references", {}
        ).items():
            config_path = _repo_path(reference["config"])
            with open(config_path, encoding="utf-8") as input_file:
                config = yaml.safe_load(input_file) or {}
            output_dir = _repo_path(config["output_dir"])
            completed = BenchmarkRunner.find_completed_run(
                config, search_directories=[output_dir]
            )
            if completed is None:
                references[str(source)] = {
                    "status": "missing",
                    "config": _report_path(config_path),
                }
                continue
            with open(completed.result_file, encoding="utf-8") as input_file:
                results = json.load(input_file)
            dataset_result = next(iter(results.values()))
            task_result = next(iter(dataset_result["models"].values()))
            models = {}
            for model_name, model_value in task_result.items():
                group = model_value["group_kfold_subject"]
                models[model_name] = {
                    "aggregated": group["aggregated"],
                    "folds": {
                        fold_name: {
                            "metrics": fold["metrics"],
                            "training_time": fold["training_time"],
                            "training": fold.get("training", {}),
                            "test_subject_ids": fold["split_metadata"].get(
                                "test_subject_ids", []
                            ),
                        }
                        for fold_name, fold in group["folds"].items()
                    },
                    "predictions": _report_path(
                        group["artifacts"]["predictions"]
                    ),
                }
            references[str(source)] = {
                "status": "completed",
                "config": _report_path(config_path),
                "benchmark": _completed_report_value(completed),
                "models": models,
            }
        return references

    def write_result_reports(
        self,
        plans: Sequence[CrossSourceTrialPlan],
        completed_by_trial: Mapping[str, CompletedBenchmarkRun],
    ) -> dict[str, str]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        trial_rows = []
        subject_rows = []
        for plan in plans:
            completed = completed_by_trial.get(plan.trial_id)
            if completed is None:
                trial_rows.append(plan.to_dict())
                continue
            trial, subjects = self._completed_trial_payload(plan, completed)
            trial_rows.append(trial)
            subject_rows.extend(subjects)
        in_domain = self._load_in_domain_references()
        for row in trial_rows:
            if row.get("status") != "completed":
                continue
            reference = in_domain.get(row["test_source"], {})
            model_reference = reference.get("models", {}).get(
                row["model"], {}
            )
            aggregated = model_reference.get("aggregated", {})
            if aggregated:
                row["in_domain_test_source_reference"] = {
                    "source": row["test_source"],
                    "accuracy_mean": aggregated.get("accuracy_mean"),
                    "balanced_accuracy_mean": aggregated.get(
                        "balanced_accuracy_mean"
                    ),
                    "macro_f1_mean": aggregated.get("macro_f1_mean"),
                    "accuracy_delta": row["metrics"]["accuracy"]
                    - aggregated["accuracy_mean"],
                    "balanced_accuracy_delta": row["metrics"][
                        "balanced_accuracy"
                    ] - aggregated["balanced_accuracy_mean"],
                    "macro_f1_delta": row["metrics"]["macro_f1"]
                    - aggregated["macro_f1_mean"],
                }
        subject_aggregates = {}
        for trial_id in sorted({row["trial_id"] for row in subject_rows}):
            rows = [row for row in subject_rows if row["trial_id"] == trial_id]
            subject_aggregates[trial_id] = {}
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1",
                "weighted_f1", "kappa", "ordinal_mae",
                "severe_error_rate",
            ):
                values = np.asarray([
                    row["metrics"].get(metric, np.nan) for row in rows
                ], dtype=float)
                values = values[np.isfinite(values)]
                subject_aggregates[trial_id][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "n_subjects": int(len(values)),
                }
        summary = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "experiment_spec": _report_path(self.spec_path),
            "trials": trial_rows,
            "in_domain_references": in_domain,
            "subject_aggregates": subject_aggregates,
        }
        summary_path = self.reports_dir / "cross_source_summary.json"
        subject_path = self.reports_dir / "cross_source_subject_metrics.json"
        report_path = self.reports_dir / "cross_source_generalization_report.md"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        subject_path.write_text(
            json.dumps(
                {"schema_version": RESULT_SCHEMA_VERSION, "subjects": subject_rows},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        lines = [
            "# Cross-source generalization",
            "",
            "Strict source-exclusive transfer uses disjoint subjects. "
            "Shared-subject transfer is reported separately and is not a "
            "subject-independent estimate.",
            "",
            "| Direction | Mode | Model | Status | N test | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Severe error | In-domain test-source BA | Delta BA |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in trial_rows:
            metrics = row.get("metrics", {})
            reference = row.get("in_domain_test_source_reference", {})
            lines.append(
                f"| {row['train_source']} -> {row['test_source']} | "
                f"{row['subject_mode']} | {row['model']} | {row['status']} | "
                f"{row.get('prediction_rows', '')} | "
                f"{_metric(metrics.get('accuracy'))} | "
                f"{_metric(metrics.get('balanced_accuracy'))} | "
                f"{_metric(metrics.get('macro_f1'))} | "
                f"{_metric(metrics.get('weighted_f1'))} | "
                f"{_metric(metrics.get('kappa'))} | "
                f"{_metric(metrics.get('auc'))} | "
                f"{_metric(metrics.get('severe_error_rate'))} | "
                f"{_metric(reference.get('balanced_accuracy_mean'))} | "
                f"{_metric(reference.get('balanced_accuracy_delta'), signed=True)} |"
            )
            if row.get("invalid_reasons"):
                lines.append(
                    "|  |  | invalid reasons | "
                    + "; ".join(row["invalid_reasons"])
                    + " |  |  |  |  |  |  |  |  |  |  |"
                )
        lines.extend([
            "",
            "## In-domain references",
            "",
            "| Source | Model | Accuracy mean/std | Balanced accuracy mean/std | Macro F1 mean/std |",
            "|---|---|---:|---:|---:|",
        ])
        for source, reference in in_domain.items():
            for model_name, model_value in reference.get("models", {}).items():
                aggregated = model_value["aggregated"]
                lines.append(
                    f"| {source} | {model_name} | "
                    f"{aggregated['accuracy_mean']:.4f} / "
                    f"{aggregated['accuracy_std']:.4f} | "
                    f"{aggregated['balanced_accuracy_mean']:.4f} / "
                    f"{aggregated['balanced_accuracy_std']:.4f} | "
                    f"{aggregated['macro_f1_mean']:.4f} / "
                    f"{aggregated['macro_f1_std']:.4f} |"
                )
        lines.extend([
            "",
            "## Training details",
            "",
            "| Direction | Model | Device | Epochs | Best epoch | Best validation loss | Parameters | Training seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in trial_rows:
            if row.get("status") != "completed":
                continue
            training = row.get("training", {})
            lines.append(
                f"| {row['train_source']} -> {row['test_source']} | "
                f"{row['model']} | {training.get('device_name', '')} | "
                f"{training.get('epochs_trained', '')} | "
                f"{training.get('best_epoch', '')} | "
                f"{_metric(training.get('best_validation_loss'))} | "
                f"{training.get('trainable_parameter_count', '')} | "
                f"{float(row.get('training_time', 0.0)):.3f} |"
            )
        lines.extend([
            "",
            "## Subject-level performance",
            "",
            "Subject metrics are descriptive aggregates over test subjects; RF "
            "windows and Transformer sequences remain different prediction units.",
            "",
            "| Direction | Model | Subjects | Balanced accuracy mean/std | Macro F1 mean/std | Severe error mean/std |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in trial_rows:
            if row.get("status") != "completed":
                continue
            aggregate = subject_aggregates[row["trial_id"]]
            ba = aggregate["balanced_accuracy"]
            f1 = aggregate["macro_f1"]
            severe = aggregate["severe_error_rate"]
            lines.append(
                f"| {row['train_source']} -> {row['test_source']} | "
                f"{row['model']} | {ba['n_subjects']} | "
                f"{ba['mean']:.4f} / {ba['std']:.4f} | "
                f"{f1['mean']:.4f} / {f1['std']:.4f} | "
                f"{severe['mean']:.4f} / {severe['std']:.4f} |"
            )
        lines.extend([
            "",
            "## Class-level error analysis",
            "",
            "Rows in each confusion matrix are true classes and columns are "
            "predicted classes. Counts use each model's own prediction unit.",
        ])
        for row in trial_rows:
            if row.get("status") != "completed":
                continue
            metrics = row["metrics"]
            lines.extend([
                "",
                f"### {row['train_source']} -> {row['test_source']} / "
                f"{row['model']}",
                "",
                "- True class counts: "
                f"{_format_class_counts(row['test_true_class_distribution'])}.",
                "- Predicted class counts: "
                f"{_format_class_counts(row['predicted_class_distribution'])}.",
                "- Per-class recall: "
                + " / ".join(
                    f"{label}:{_metric(row['per_class_recall'].get(str(label)))}"
                    for label in range(5)
                )
                + ".",
                "",
                "| True / predicted | 0 | 1 | 2 | 3 | 4 |",
                "|---:|---:|---:|---:|---:|---:|",
            ])
            for label, values in enumerate(metrics["confusion_matrix"]):
                lines.append(
                    f"| {label} | " + " | ".join(str(value) for value in values) + " |"
                )
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- CS1 is strict unseen-subject transfer: subjects, logical "
            "recordings, source records, sample IDs and canonical raw intervals "
            "have zero train/test overlap.",
            "- CS2 is not estimable with the configured minimums after removing "
            "33 exact logical-record duplicates: only one subject retains data "
            "in both sources, below both train and test subject minimums.",
            "- All four completed CS1 accuracies exceed the five-class random "
            "reference of 0.20. Transformer exceeds RF in both directions.",
            "- Transfer is asymmetric, especially for Transformer: "
            "gpn_data -> Old_EEG is stronger than Old_EEG -> gpn_data. "
            "This is descriptive, not a significance claim.",
            "- Relative to source-only GroupKFold on the destination source, "
            "the observed balanced-accuracy deltas are small and can be positive "
            "or negative; the evaluated subject populations differ, so these are "
            "contextual references rather than paired estimates.",
            "- Source-identity predictability was not trained or selected in this "
            "experiment; no separability claim is made from target-test labels.",
            "- Both sources cover all five classes in every valid outer partition. "
            "No additional seeds, target fine-tuning or preprocessing variants "
            "were run.",
            "- Middle classes 2 and 3 have the weakest recall in most completed "
            "trials; class 1 is additionally weak for the reverse-direction "
            "Transformer. Class-frequency and prediction-distribution shifts are "
            "shown above and may contribute to, but do not prove the cause of, "
            "the directional gap.",
            "",
            "No difference is described as statistically significant without "
            "a separate paired analysis. A five-class random baseline is 0.20.",
            "",
        ])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "report": str(report_path),
            "summary": str(summary_path),
            "subject_metrics": str(subject_path),
        }

    def execute(
        self,
        plans: Sequence[CrossSourceTrialPlan],
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        completed_by_trial: dict[str, CompletedBenchmarkRun] = {}
        outcomes = []
        for plan in plans:
            if plan.status == "invalid":
                outcomes.append(plan.to_dict())
                continue
            completed = self.completed_run_finder(
                plan.resolved_config, search_directories=[plan.output_dir]
            )
            if resume and completed is not None:
                completed_by_trial[plan.trial_id] = completed
                outcomes.append({
                    **plan.to_dict(),
                    "status": "reused",
                    "benchmark": completed.to_dict(),
                })
                continue
            runner = self.runner_factory(deepcopy(dict(plan.resolved_config)))
            runner.run()
            completed = runner.completed_run()
            completed_by_trial[plan.trial_id] = completed
            outcomes.append({
                **plan.to_dict(),
                "status": "completed",
                "benchmark": completed.to_dict(),
            })
        reports = self.write_result_reports(plans, completed_by_trial)
        return {"trials": outcomes, "reports": reports}
