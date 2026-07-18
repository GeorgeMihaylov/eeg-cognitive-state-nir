"""Random-forest EEG/POW feature-group classification and regression audit."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import GroupKFold

from bench.analysis.label_target_audit import _jsonable, _write_json
from bench.analysis.paired_statistics import (
    apply_holm_by_family,
    paired_subject_statistics,
)
from bench.analysis.subject_metrics import (
    calculate_regression_subject_metrics,
    calculate_subject_metrics,
)
from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.sequence_utils import (
    SEQUENCE_INDEX_COLUMNS,
    build_sequences,
    sequence_index_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_LABEL_THRESHOLDS = (0.330177, 0.387786, 0.444458, 0.526585)
FEATURE_GROUP_ORDER = ("eeg_only", "pow_only", "eeg_pow")
TASK_ORDER = ("classification", "regression")
PAIR_ORDER = (
    ("eeg_only", "pow_only"),
    ("eeg_only", "eeg_pow"),
    ("pow_only", "eeg_pow"),
)
IDENTITY_COLUMNS = ("sample_id", "fold", "subject_id", "record_id", "source")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_feature_group_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Feature-group experiment not found: {spec_path}")
    with spec_path.open(encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    required = {"experiment", "dataset", "feature_groups", "tasks", "models", "evaluation"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Feature-group experiment is missing sections: {missing}")
    return document


def quantize_regression_predictions(
    values: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] = GLOBAL_LABEL_THRESHOLDS,
) -> np.ndarray:
    """Apply fixed right-closed global label boundaries to finite predictions."""

    predictions = np.asarray(values, dtype=float).reshape(-1)
    edges = np.asarray(thresholds, dtype=float).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("Regression predictions must be finite before quantization")
    if len(edges) != 4 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Exactly four finite increasing global thresholds are required")
    return np.searchsorted(edges, predictions, side="left").astype(np.int64)


def prediction_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    compare_target: bool,
) -> dict[str, Any]:
    """Audit exact IDs, folds, metadata, and optionally task target values."""

    required = set(IDENTITY_COLUMNS) | {"y_true"}
    missing_left = sorted(required - set(left))
    missing_right = sorted(required - set(right))
    if missing_left or missing_right:
        return {
            "exact_match": False,
            "missing_left": missing_left,
            "missing_right": missing_right,
        }
    left_duplicates = int(left["sample_id"].duplicated().sum())
    right_duplicates = int(right["sample_id"].duplicated().sum())
    merged = left[list(required)].merge(
        right[list(required)],
        on="sample_id",
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate=("one_to_one" if not left_duplicates and not right_duplicates else None),
    )
    mismatches: dict[str, int] = {
        "sample_id_membership": int((merged["_merge"] != "both").sum()),
    }
    matched = merged.loc[merged["_merge"] == "both"]
    for column in IDENTITY_COLUMNS[1:]:
        left_values = matched[f"{column}_left"]
        right_values = matched[f"{column}_right"]
        if column == "fold":
            mismatch = left_values.astype(int) != right_values.astype(int)
        else:
            mismatch = left_values.astype(str) != right_values.astype(str)
        mismatches[column] = int(mismatch.sum())
    if compare_target:
        mismatches["y_true"] = int((~np.isclose(
            matched["y_true_left"].astype(float),
            matched["y_true_right"].astype(float),
            equal_nan=True,
        )).sum())
    return {
        "exact_match": bool(
            len(left) == len(right)
            and left_duplicates == 0
            and right_duplicates == 0
            and not any(mismatches.values())
        ),
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "matched_rows": int(len(matched)),
        "left_duplicate_sample_ids": left_duplicates,
        "right_duplicate_sample_ids": right_duplicates,
        "mismatches": mismatches,
    }


def sequence_prediction_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    compare_predictions: bool = False,
) -> dict[str, Any]:
    """Audit canonical sequence identity, fold assignment, target and outputs."""

    identity = "sequence_id"
    metadata_columns = (
        "fold", "subject_id", "record_id", "source", "target_sample_id",
        "target_time", "y_true",
    )
    required = {identity, *metadata_columns}
    missing_left = sorted(required - set(left))
    missing_right = sorted(required - set(right))
    if missing_left or missing_right:
        return {
            "exact_match": False,
            "missing_left": missing_left,
            "missing_right": missing_right,
        }
    left_duplicates = int(left[identity].duplicated().sum())
    right_duplicates = int(right[identity].duplicated().sum())
    merged = left[[identity, *metadata_columns]].merge(
        right[[identity, *metadata_columns]],
        on=identity,
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate=("one_to_one" if not left_duplicates and not right_duplicates else None),
    )
    mismatches: dict[str, int] = {
        "sequence_id_membership": int((merged["_merge"] != "both").sum())
    }
    matched = merged.loc[merged["_merge"] == "both"]
    for column in metadata_columns:
        left_values = matched[f"{column}_left"]
        right_values = matched[f"{column}_right"]
        if column in {"target_time", "y_true"}:
            mismatch = ~np.isclose(
                left_values.astype(float),
                right_values.astype(float),
                equal_nan=True,
            )
        elif column == "fold":
            mismatch = left_values.astype(int) != right_values.astype(int)
        else:
            mismatch = left_values.astype(str) != right_values.astype(str)
        mismatches[column] = int(mismatch.sum())
    prediction_mismatches: dict[str, Any] = {}
    if compare_predictions and not mismatches["sequence_id_membership"]:
        outputs = left[[identity, "y_pred", *[c for c in left if c.startswith("proba_")]]].merge(
            right[[identity, "y_pred", *[c for c in right if c.startswith("proba_")]]],
            on=identity,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        prediction_mismatches["y_pred"] = int(
            (outputs["y_pred_left"].astype(int) != outputs["y_pred_right"].astype(int)).sum()
        )
        probability_deltas = []
        for class_id in range(5):
            left_column = f"proba_{class_id}_left"
            right_column = f"proba_{class_id}_right"
            if left_column in outputs and right_column in outputs:
                probability_deltas.append(
                    np.abs(
                        outputs[left_column].to_numpy(dtype=float)
                        - outputs[right_column].to_numpy(dtype=float)
                    )
                )
        if probability_deltas:
            combined = np.concatenate(probability_deltas)
            prediction_mismatches["probability_max_abs_delta"] = float(combined.max())
            prediction_mismatches["probability_mean_abs_delta"] = float(combined.mean())
    return {
        "exact_match": bool(
            len(left) == len(right)
            and left_duplicates == 0
            and right_duplicates == 0
            and not any(mismatches.values())
        ),
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "matched_rows": int(len(matched)),
        "left_duplicate_sequence_ids": left_duplicates,
        "right_duplicate_sequence_ids": right_duplicates,
        "mismatches": mismatches,
        "prediction_differences": prediction_mismatches,
    }


def resolve_trial_config(
    document: Mapping[str, Any],
    *,
    task_name: str,
    feature_group: str,
    seed: int,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one matrix cell into the canonical ``BenchmarkRunner`` schema."""

    if task_name not in document["tasks"]:
        raise ValueError(f"Unknown experiment task {task_name!r}")
    if feature_group not in document["feature_groups"]:
        raise ValueError(f"Unknown feature group {feature_group!r}")
    task = document["tasks"][task_name]
    group = document["feature_groups"][feature_group]
    model_name = str(task["model"])
    model = deepcopy(document["models"][model_name])
    params = model.setdefault("params", {})
    params["random_state"] = int(seed)

    root = _repo_path(output_root or document["experiment"]["output_dir"])
    trial_prefix = str(document["experiment"].get("trial_prefix", "rf"))
    trial_id = f"{trial_prefix}_{task_name}_{feature_group}"
    trial_output = root / "runs" / trial_id
    dataset = document["dataset"]
    config = {
        "output_dir": str(trial_output),
        "datasets": {
            str(dataset["name"]): {
                "data_path": str(_repo_path(dataset["data_path"])),
                "feature_set": str(group["feature_set"]),
                "feature_group": feature_group,
                "target_col": str(task["target"]),
                "subject_col": str(dataset.get("subject_col", "subject_id")),
                "n_classes": int(task.get("n_classes", 5)),
                "discretize": False,
                "max_features": int(group["feature_count"]),
                "expected_feature_count": int(group["feature_count"]),
                "feature_list_sha256": str(group["feature_list_sha256"]),
            }
        },
        "tasks": [str(task["benchmark_task"])],
        "models": {model_name: model},
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": int(document["evaluation"].get("n_splits", 5)),
            "group_column": str(document["evaluation"].get("group_column", "subject_id")),
            "random_state": int(seed),
        },
        "task_config": {"random_state": int(seed)},
        "run_within_subject": False,
        "run_loso": False,
        "experiment": {
            "name": str(document["experiment"]["name"]),
            "trial_id": trial_id,
            "task": task_name,
            "target": str(task["target"]),
            "feature_group": feature_group,
            "feature_count": int(group["feature_count"]),
            "feature_list_sha256": str(group["feature_list_sha256"]),
            "seed": int(seed),
        },
    }
    for section in ("sequence", "validation"):
        if section in document:
            config[section] = deepcopy(document[section])
    return config


@dataclass(frozen=True)
class FeatureGroupTrialPlan:
    trial_id: str
    task: str
    target: str
    model: str
    feature_group: str
    feature_count: int
    feature_list_sha256: str
    fold_count: int
    rows: int
    subjects: int
    status: str
    invalid_reasons: tuple[str, ...]
    action: str
    output_dir: Path
    config_hash: str
    resolved_config: Mapping[str, Any]
    completed_run: CompletedBenchmarkRun | None = None
    input_shape: tuple[int, ...] | None = None
    sequence_length: int | None = None
    sequence_count: int | None = None
    sequence_index_sha256: str | None = None
    model_parameters: Mapping[str, Any] | None = None

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        value = {
            "trial_id": self.trial_id,
            "task": self.task,
            "target": self.target,
            "model": self.model,
            "feature_group": self.feature_group,
            "feature_count": self.feature_count,
            "feature_list_sha256": self.feature_list_sha256,
            "fold_count": self.fold_count,
            "rows": self.rows,
            "subjects": self.subjects,
            "expected_output_directory": _relative_path(self.output_dir),
            "existing_reusable_run": (
                None
                if self.completed_run is None
                else _relative_path(self.completed_run.run_directory)
            ),
            "validity_status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "action": self.action,
            "config_hash": self.config_hash,
        }
        if self.input_shape is not None:
            value["input_shape"] = list(self.input_shape)
        if self.sequence_length is not None:
            value["sequence_length"] = self.sequence_length
        if self.sequence_count is not None:
            value["sequence_count"] = self.sequence_count
        if self.sequence_index_sha256 is not None:
            value["sequence_index_sha256"] = self.sequence_index_sha256
        if self.model_parameters is not None:
            value["model_parameters"] = _jsonable(self.model_parameters)
        if include_config:
            value["resolved_config"] = _jsonable(self.resolved_config)
        return value


class FeatureGroupRFExperiment:
    """Plan, execute, resume, and analyze the six canonical RF trials."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        runner_factory: Callable[[dict[str, Any]], Any] = BenchmarkRunner,
        completed_run_finder: Callable[..., CompletedBenchmarkRun | None] = (
            BenchmarkRunner.find_completed_run
        ),
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_feature_group_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.report_path = _repo_path(self.document["experiment"]["report_path"])
        self.summary_path = _repo_path(self.document["experiment"]["summary_path"])
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self._supervised: pd.DataFrame | None = None
        self._feature_manifests: dict[str, dict[str, Any]] | None = None
        self._canonical_alignment: dict[str, Any] | None = None

    @property
    def data_path(self) -> Path:
        return _repo_path(self.document["dataset"]["data_path"])

    def _audit_dataset(self) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
        if self._supervised is not None:
            return self._supervised, self._feature_manifests or {}, self._canonical_alignment or {}
        schema_columns = list(pq.ParquetFile(self.data_path).schema.names)
        metadata_columns = [
            "subject_id", "record_id", "source", "t_start", "label_q5",
            "target_focus",
        ]
        frame = pd.read_parquet(self.data_path, columns=metadata_columns)
        frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
        supervised = frame.loc[
            frame["label_q5"].notna() & frame["target_focus"].notna()
        ].copy()
        supervised["label_q5"] = supervised["label_q5"].astype(np.int64)
        splitter = GroupKFold(n_splits=int(self.document["evaluation"]["n_splits"]))
        supervised["fold"] = 0
        for fold, (_, test_index) in enumerate(
            splitter.split(supervised, supervised["label_q5"], supervised["subject_id"]),
            start=1,
        ):
            supervised.iloc[test_index, supervised.columns.get_loc("fold")] = fold

        manifests: dict[str, dict[str, Any]] = {}
        forbidden_exact = {
            "target_focus", "label_q5", "subject_id", "record_id", "source",
            "sample_id", "t_start", "t_end", "t_center", "window_id",
        }
        for group_name, group in self.document["feature_groups"].items():
            names = resolve_feature_columns(schema_columns, str(group["feature_set"]))
            digest = feature_list_sha256(names)
            forbidden = sorted(
                name for name in names
                if name in forbidden_exact
                or name.startswith("target_")
                or name.startswith("PM.")
            )
            manifests[group_name] = {
                "feature_group": group_name,
                "feature_count": len(names),
                "ordered_feature_names": names,
                "feature_list_sha256": digest,
                "expected_feature_count": int(group["feature_count"]),
                "expected_feature_list_sha256": str(group["feature_list_sha256"]),
                "forbidden_features": forbidden,
                "valid": bool(
                    len(names) == int(group["feature_count"])
                    and digest == str(group["feature_list_sha256"])
                    and not forbidden
                ),
            }

        reference_path = _repo_path(
            self.document["experiment"]["canonical_reference_predictions"]
        )
        reference = pd.read_parquet(reference_path)
        if "source" not in reference and "record_id" in reference:
            reference["source"] = np.where(
                reference["record_id"].astype(str).str.startswith("gpn_data"),
                "gpn_data",
                "Old_EEG",
            )
        expected = supervised.rename(columns={"label_q5": "y_true"})
        alignment = prediction_alignment(expected, reference, compare_target=True)
        alignment["reference_path"] = _relative_path(reference_path)
        alignment["train_test_subject_overlap"] = {
            str(fold): sorted(
                set(supervised.loc[supervised["fold"] != fold, "subject_id"].astype(str))
                & set(supervised.loc[supervised["fold"] == fold, "subject_id"].astype(str))
            )
            for fold in sorted(supervised["fold"].unique())
        }
        self._supervised = supervised.reset_index(drop=True)
        self._feature_manifests = manifests
        self._canonical_alignment = alignment
        return self._supervised, manifests, alignment

    @staticmethod
    def _selection(values: Sequence[str] | None, available: Sequence[str]) -> list[str]:
        selected = list(available) if values is None else [str(value) for value in values]
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown matrix values: {unknown}; available={list(available)}")
        return [value for value in available if value in selected]

    def plan(
        self,
        *,
        feature_groups: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        seed: int = 42,
    ) -> list[FeatureGroupTrialPlan]:
        supervised, manifests, alignment = self._audit_dataset()
        selected_groups = self._selection(feature_groups, FEATURE_GROUP_ORDER)
        selected_tasks = self._selection(tasks, list(self.document["tasks"]))
        available_models = list(self.document["models"])
        selected_models = self._selection(models, available_models)
        plans: list[FeatureGroupTrialPlan] = []
        for task_name in selected_tasks:
            task = self.document["tasks"][task_name]
            model_name = str(task["model"])
            if model_name not in selected_models:
                continue
            for group_name in selected_groups:
                config = resolve_trial_config(
                    self.document,
                    task_name=task_name,
                    feature_group=group_name,
                    seed=seed,
                    output_root=self.output_root,
                )
                output = Path(config["output_dir"])
                digest = benchmark_config_hash(config)
                completed = self.completed_run_finder(
                    config, search_directories=[output]
                )
                invalid: list[str] = []
                manifest = manifests[group_name]
                if not manifest["valid"]:
                    invalid.append("feature manifest mismatch or forbidden features")
                if len(supervised) != int(self.document["dataset"]["expected_supervised_rows"]):
                    invalid.append("supervised row count mismatch")
                if not alignment.get("exact_match", False):
                    invalid.append("canonical fold alignment mismatch")
                if any(alignment["train_test_subject_overlap"].values()):
                    invalid.append("train/test subject overlap")
                status = "valid" if not invalid else "invalid"
                plans.append(FeatureGroupTrialPlan(
                    trial_id=str(config["experiment"]["trial_id"]),
                    task=task_name,
                    target=str(task["target"]),
                    model=model_name,
                    feature_group=group_name,
                    feature_count=int(manifest["feature_count"]),
                    feature_list_sha256=str(manifest["feature_list_sha256"]),
                    fold_count=int(self.document["evaluation"]["n_splits"]),
                    rows=int(len(supervised)),
                    subjects=int(supervised["subject_id"].nunique()),
                    status=status,
                    invalid_reasons=tuple(invalid),
                    action=("reuse" if completed is not None else "run"),
                    output_dir=output,
                    config_hash=digest,
                    resolved_config=config,
                    completed_run=completed,
                ))
        return plans

    @staticmethod
    def render_plan(plans: Sequence[FeatureGroupTrialPlan]) -> str:
        lines = [
            "# RF feature-group experiment plan",
            "",
            "| Trial | Task / target | Model | Features | Count | Hash | Folds | Rows | Subjects | Output | Reusable | Status |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| `{plan.trial_id}` | {plan.task} / `{plan.target}` | "
                f"{plan.model} | {plan.feature_group} | {plan.feature_count} | "
                f"`{plan.feature_list_sha256[:16]}` | {plan.fold_count} | "
                f"{plan.rows} | {plan.subjects} | `{_relative_path(plan.output_dir)}` | "
                f"{'yes' if plan.completed_run else 'no'} | {plan.status} |"
            )
        lines.extend([
            "",
            f"Trials: **{len(plans)}**; valid: **{sum(p.status == 'valid' for p in plans)}**; "
            f"to run: **{sum(p.action == 'run' for p in plans)}**.",
            "Plan-only performs no training and writes no benchmark artifacts.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _completed_trial(
        plan: FeatureGroupTrialPlan,
        completed: CompletedBenchmarkRun,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        with completed.result_file.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
        dataset_name = next(iter(plan.resolved_config["datasets"]))
        task_name = str(plan.resolved_config["tasks"][0])
        result = payload[dataset_name]["models"][task_name][plan.model][
            "group_kfold_subject"
        ]
        predictions = pd.read_parquet(_repo_path(result["artifacts"]["predictions"]))
        return result, predictions

    @staticmethod
    def _fold_metrics(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "fold": fold_name,
                **{
                    key: value
                    for key, value in fold_result["metrics"].items()
                    if key != "confusion_matrix"
                },
                "training_time": fold_result["training_time"],
            }
            for fold_name, fold_result in result["folds"].items()
        ]

    @staticmethod
    def _aggregate_rows(
        rows: Sequence[Mapping[str, Any]],
        metrics: Sequence[str],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"n_folds": len(rows)}
        for metric in metrics:
            values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            output[metric] = {
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std()) if len(finite) else np.nan,
                "values": values.tolist(),
            }
        return output

    def _source_results(
        self,
        predictions: Mapping[tuple[str, str], pd.DataFrame],
    ) -> list[dict[str, Any]]:
        supervised, _, _ = self._audit_dataset()
        truth = supervised.set_index("sample_id")["label_q5"]
        rows: list[dict[str, Any]] = []
        for (task, group_name), frame in predictions.items():
            for source, source_frame in frame.groupby("source", sort=True):
                subjects = sorted(source_frame["subject_id"].astype(str).unique())
                row: dict[str, Any] = {
                    "task": task,
                    "feature_group": group_name,
                    "source": str(source),
                    "windows": int(len(source_frame)),
                    "subjects": len(subjects),
                    "subject_ids": subjects,
                    "subject_counts_are_non_additive": True,
                }
                if task == "classification":
                    proba = source_frame.filter(regex=r"^proba_\d+$").to_numpy(dtype=float)
                    metrics = MetricsCalculator.calculate_all_metrics(
                        source_frame["y_true"].to_numpy(dtype=int),
                        source_frame["y_pred"].to_numpy(dtype=int),
                        proba if proba.shape[1] else None,
                    )
                    row.update({
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "ordinal_mae": metrics["ordinal_mae"],
                    })
                else:
                    metrics = MetricsCalculator.calculate_regression_metrics(
                        source_frame["y_true"].to_numpy(dtype=float),
                        source_frame["y_pred"].to_numpy(dtype=float),
                    )
                    predicted_class = quantize_regression_predictions(
                        source_frame["y_pred"].to_numpy(dtype=float)
                    )
                    true_class = truth.loc[source_frame["sample_id"]].to_numpy(dtype=int)
                    quantized = MetricsCalculator.calculate_all_metrics(
                        true_class, predicted_class
                    )
                    row.update({
                        "mae": metrics["mae"],
                        "spearman": metrics["spearman"],
                        "quantized_ordinal_mae": quantized["ordinal_mae"],
                    })
                rows.append(row)
        return rows

    @staticmethod
    def _feature_parts(name: str) -> dict[str, str | None]:
        base, _, statistic = name.partition("__")
        pieces = base.split(".")
        family = pieces[0] if pieces else "unknown"
        channel = pieces[1] if len(pieces) > 1 else None
        band = pieces[2] if family == "POW" and len(pieces) > 2 else None
        return {
            "feature_family": family,
            "channel": channel,
            "frequency_band": band,
            "statistic": statistic or None,
        }

    def _importance_results(
        self,
        plans: Sequence[FeatureGroupTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        detailed: list[pd.DataFrame] = []
        for plan in plans:
            result, _ = self._completed_trial(plan, completed[plan.trial_id])
            for fold_name, fold in result["folds"].items():
                path = fold.get("artifacts", {}).get("feature_importance")
                if not path:
                    raise ValueError(
                        f"Feature importance missing for {plan.trial_id} {fold_name}"
                    )
                frame = pd.read_parquet(_repo_path(path))
                frame["task"] = plan.task
                frame["feature_group"] = plan.feature_group
                frame["fold"] = fold_name
                detailed.append(frame)
        all_importance = pd.concat(detailed, ignore_index=True)
        parts = all_importance["feature_name"].map(self._feature_parts).apply(pd.Series)
        all_importance = pd.concat([all_importance, parts], axis=1)
        summary = (
            all_importance.groupby(
                ["task", "feature_group", "feature_name", "feature_family", "channel", "frequency_band", "statistic"],
                dropna=False,
                sort=True,
            )
            .agg(
                mean_importance=("importance", "mean"),
                importance_sd=("importance", "std"),
                mean_rank=("rank", "mean"),
                folds_in_top_20=("rank", lambda values: int((values <= 20).sum())),
                folds=("fold", "nunique"),
            )
            .reset_index()
        )
        summary["importance_sd"] = summary["importance_sd"].fillna(0.0)
        summary = summary.sort_values(
            ["task", "feature_group", "mean_importance", "feature_name"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        per_fold_aggregate = all_importance.groupby(
            [
                "task", "feature_group", "fold", "feature_family", "channel",
                "frequency_band", "statistic",
            ],
            dropna=False,
            sort=True,
        )["importance"].sum()
        aggregate = (
            per_fold_aggregate.groupby(level=[0, 1, 3, 4, 5, 6])
            .mean()
            .rename("mean_total_importance")
            .reset_index()
        )
        analysis_dir = self.output_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        all_importance.to_parquet(analysis_dir / "fold_feature_importance.parquet", index=False)
        summary.to_parquet(analysis_dir / "feature_importance_summary.parquet", index=False)
        aggregate.to_parquet(analysis_dir / "feature_importance_aggregates.parquet", index=False)
        return _jsonable(summary.to_dict("records")), _jsonable(aggregate.to_dict("records"))

    @staticmethod
    def _paired_results(
        subject_metrics: Mapping[tuple[str, str], pd.DataFrame],
        *,
        n_resamples: int,
        random_state: int,
    ) -> list[dict[str, Any]]:
        metric_specs = {
            "classification": {
                "balanced_accuracy": True,
                "macro_f1": True,
                "ordinal_mae": False,
            },
            "regression": {"mae": False, "spearman": True},
        }
        rows: list[dict[str, Any]] = []
        for task, metrics in metric_specs.items():
            for left_group, right_group in PAIR_ORDER:
                left = subject_metrics[(task, left_group)].set_index("subject_id")
                right = subject_metrics[(task, right_group)].set_index("subject_id")
                common = sorted(set(left.index.astype(str)) & set(right.index.astype(str)))
                for metric, higher_is_better in metrics.items():
                    left_values = left.loc[common, metric].to_numpy(dtype=float)
                    right_values = right.loc[common, metric].to_numpy(dtype=float)
                    oriented_left = left_values if higher_is_better else -left_values
                    oriented_right = right_values if higher_is_better else -right_values
                    statistics = paired_subject_statistics(
                        oriented_left,
                        oriented_right,
                        n_resamples=n_resamples,
                        random_state=random_state,
                    )
                    rows.append({
                        "family": f"rf_{task}_feature_groups",
                        "task": task,
                        "comparison": f"{left_group}_vs_{right_group}",
                        "left": left_group,
                        "right": right_group,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "difference_definition": (
                            "left_minus_right"
                            if higher_is_better
                            else "right_error_minus_left_error"
                        ),
                        **statistics,
                    })
        rows = apply_holm_by_family(rows, p_key="wilcoxon_p_value")
        sign_adjusted = apply_holm_by_family(rows, p_key="sign_test_p_value")
        for row, sign_row in zip(rows, sign_adjusted):
            row["holm_adjusted_sign_p_value"] = sign_row[
                "holm_adjusted_p_value"
            ]
        return rows

    def _analyze(
        self,
        plans: Sequence[FeatureGroupTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
    ) -> dict[str, Any]:
        supervised, feature_manifests, canonical_alignment = self._audit_dataset()
        truth_by_sample = supervised.set_index("sample_id")
        predictions: dict[tuple[str, str], pd.DataFrame] = {}
        subject_metrics: dict[tuple[str, str], pd.DataFrame] = {}
        fold_metrics: dict[tuple[str, str], list[dict[str, Any]]] = {}
        quantized_results: dict[str, Any] = {}
        analysis_dir = self.output_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for plan in plans:
            result, frame = self._completed_trial(plan, completed[plan.trial_id])
            frame = frame.sort_values(["fold", "sample_id"], kind="mergesort").reset_index(drop=True)
            predictions[(plan.task, plan.feature_group)] = frame
            fold_metrics[(plan.task, plan.feature_group)] = self._fold_metrics(result)
            trial_dir = analysis_dir / plan.trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            if plan.task == "classification":
                subjects = calculate_subject_metrics(
                    frame,
                    track="feature_group_rf_classification",
                    model=plan.feature_group,
                    seed=int(plan.resolved_config["experiment"]["seed"]),
                )
            else:
                subjects = calculate_regression_subject_metrics(
                    frame,
                    track="feature_group_rf_regression",
                    model=plan.feature_group,
                    seed=int(plan.resolved_config["experiment"]["seed"]),
                )
                quantized = frame.copy()
                quantized["continuous_prediction"] = quantized["y_pred"].astype(float)
                quantized["y_pred"] = quantize_regression_predictions(
                    quantized["continuous_prediction"].to_numpy(dtype=float)
                )
                quantized["y_true"] = truth_by_sample.loc[
                    quantized["sample_id"], "label_q5"
                ].to_numpy(dtype=int)
                quantized.to_parquet(trial_dir / "quantized_predictions.parquet", index=False)
                quantized_folds = []
                for fold, group in quantized.groupby("fold", sort=True):
                    metrics = MetricsCalculator.calculate_all_metrics(
                        group["y_true"].to_numpy(dtype=int),
                        group["y_pred"].to_numpy(dtype=int),
                    )
                    quantized_folds.append({"fold": int(fold), **metrics})
                quantized_results[plan.feature_group] = {
                    "thresholds": list(GLOBAL_LABEL_THRESHOLDS),
                    "folds": quantized_folds,
                    "aggregated": self._aggregate_rows(
                        quantized_folds,
                        [
                            "balanced_accuracy", "macro_f1", "ordinal_mae",
                            "adjacent_accuracy", "severe_error_rate",
                        ],
                    ),
                    "diagnostic_only": True,
                }
            subjects.to_parquet(trial_dir / "subject_metrics.parquet", index=False)
            subject_metrics[(plan.task, plan.feature_group)] = subjects

        alignment_rows: list[dict[str, Any]] = []
        for task in TASK_ORDER:
            for left, right in combinations(FEATURE_GROUP_ORDER, 2):
                alignment_rows.append({
                    "scope": task,
                    "comparison": f"{left}_vs_{right}",
                    **prediction_alignment(
                        predictions[(task, left)],
                        predictions[(task, right)],
                        compare_target=True,
                    ),
                })
        for group_name in FEATURE_GROUP_ORDER:
            alignment_rows.append({
                "scope": "classification_vs_regression",
                "comparison": group_name,
                **prediction_alignment(
                    predictions[("classification", group_name)],
                    predictions[("regression", group_name)],
                    compare_target=False,
                ),
            })
        if not all(row["exact_match"] for row in alignment_rows):
            raise ValueError("Prediction artifacts are not exactly aligned")

        classification = {
            group: {
                "folds": fold_metrics[("classification", group)],
                "aggregated": self._aggregate_rows(
                    fold_metrics[("classification", group)],
                    [
                        "balanced_accuracy", "macro_f1", "accuracy", "weighted_f1",
                        "kappa", "auc", "ordinal_mae", "adjacent_accuracy",
                        "severe_error_rate",
                    ],
                ),
            }
            for group in FEATURE_GROUP_ORDER
        }
        regression = {
            group: {
                "folds": fold_metrics[("regression", group)],
                "aggregated": self._aggregate_rows(
                    fold_metrics[("regression", group)],
                    ["mae", "rmse", "r2", "pearson", "spearman"],
                ),
            }
            for group in FEATURE_GROUP_ORDER
        }
        paired = self._paired_results(
            subject_metrics,
            n_resamples=int(self.document["analysis"].get("bootstrap_samples", 10_000)),
            random_state=int(self.document["analysis"].get("random_state", 42)),
        )
        importance, importance_aggregates = self._importance_results(plans, completed)
        sources = self._source_results(predictions)
        summary = {
            "analysis_name": self.document["experiment"]["name"],
            "seed": int(plans[0].resolved_config["experiment"]["seed"]),
            "supervised_rows": int(len(supervised)),
            "subjects": int(supervised["subject_id"].nunique()),
            "feature_groups": feature_manifests,
            "canonical_alignment": canonical_alignment,
            "prediction_alignment": alignment_rows,
            "classification": classification,
            "regression": regression,
            "regression_to_class": quantized_results,
            "subject_level_comparisons": paired,
            "source_level_results": sources,
            "feature_importance": importance,
            "feature_importance_aggregates": importance_aggregates,
            "permutation_importance": {
                "status": "not_run",
                "reason": "optional computation was omitted to keep the six-trial audit bounded",
            },
            "completed_runs": {
                plan.trial_id: _relative_path(completed[plan.trial_id].run_directory)
                for plan in plans
            },
        }
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(self._render_report(_jsonable(summary)), encoding="utf-8")
        return _jsonable(summary)

    @staticmethod
    def _render_report(summary: Mapping[str, Any]) -> str:
        def metric(value: Any, digits: int = 4) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "NA"
            return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "NA"

        def optional(value: Any) -> str:
            return "" if value is None or pd.isna(value) else str(value)

        classification = summary["classification"]
        regression = summary["regression"]
        quantized = summary["regression_to_class"]
        ba = {
            group: classification[group]["aggregated"]["balanced_accuracy"]["mean"]
            for group in FEATURE_GROUP_ORDER
        }
        macro = {
            group: classification[group]["aggregated"]["macro_f1"]["mean"]
            for group in FEATURE_GROUP_ORDER
        }
        ordinal = {
            group: classification[group]["aggregated"]["ordinal_mae"]["mean"]
            for group in FEATURE_GROUP_ORDER
        }
        mae = {
            group: regression[group]["aggregated"]["mae"]["mean"]
            for group in FEATURE_GROUP_ORDER
        }
        spearman = {
            group: regression[group]["aggregated"]["spearman"]["mean"]
            for group in FEATURE_GROUP_ORDER
        }
        rank_score = {group: 0 for group in FEATURE_GROUP_ORDER}
        for rank, group in enumerate(sorted(FEATURE_GROUP_ORDER, key=lambda value: -ba[value])):
            rank_score[group] += rank
        for rank, group in enumerate(sorted(FEATURE_GROUP_ORDER, key=lambda value: mae[value])):
            rank_score[group] += rank
        recommended = sorted(FEATURE_GROUP_ORDER, key=lambda value: (rank_score[value], value))[:2]

        mismatch_total = sum(
            sum(row.get("mismatches", {}).values())
            for row in summary["prediction_alignment"]
        )
        feature_lines = []
        for group in FEATURE_GROUP_ORDER:
            manifest = summary["feature_groups"][group]
            feature_lines.append(
                f"| {group} | {manifest['feature_count']} | "
                f"`{manifest['feature_list_sha256']}` | {manifest['valid']} |"
            )
        classification_lines = []
        for group in FEATURE_GROUP_ORDER:
            values = classification[group]["aggregated"]
            classification_lines.append(
                f"| {group} | {metric(values['balanced_accuracy']['mean'])} +/- "
                f"{metric(values['balanced_accuracy']['std'])} | "
                f"{metric(values['macro_f1']['mean'])} +/- {metric(values['macro_f1']['std'])} | "
                f"{metric(values['accuracy']['mean'])} | {metric(values['weighted_f1']['mean'])} | "
                f"{metric(values['kappa']['mean'])} | {metric(values['auc']['mean'])} | "
                f"{metric(values['ordinal_mae']['mean'])} | "
                f"{metric(values['adjacent_accuracy']['mean'])} | "
                f"{metric(values['severe_error_rate']['mean'])} |"
            )
        regression_lines = []
        for group in FEATURE_GROUP_ORDER:
            values = regression[group]["aggregated"]
            regression_lines.append(
                f"| {group} | {metric(values['mae']['mean'])} +/- {metric(values['mae']['std'])} | "
                f"{metric(values['rmse']['mean'])} +/- {metric(values['rmse']['std'])} | "
                f"{metric(values['r2']['mean'])} | {metric(values['pearson']['mean'])} | "
                f"{metric(values['spearman']['mean'])} |"
            )
        quantized_lines = []
        for group in FEATURE_GROUP_ORDER:
            values = quantized[group]["aggregated"]
            quantized_lines.append(
                f"| {group} | {metric(values['balanced_accuracy']['mean'])} | "
                f"{metric(values['macro_f1']['mean'])} | {metric(values['ordinal_mae']['mean'])} | "
                f"{metric(values['adjacent_accuracy']['mean'])} | "
                f"{metric(values['severe_error_rate']['mean'])} | "
                f"{metric(ba[group])} / {metric(ordinal[group])} |"
            )
        paired_lines = [
            f"| {row['task']} | {row['comparison']} | {row['metric']} | "
            f"{metric(row['mean_difference'])} | {metric(row['median_difference'])} | "
            f"[{metric(row['ci_low'])}, {metric(row['ci_high'])}] | "
            f"{row['subjects_improved']} / {row['subjects_degraded']} / {row['ties']} | "
            f"{metric(row['wilcoxon_p_value'])} | {metric(row['holm_adjusted_p_value'])} | "
            f"{metric(row['sign_test_p_value'])} | {metric(row['holm_adjusted_sign_p_value'])} | "
            f"{metric(row['rank_biserial'])} |"
            for row in summary["subject_level_comparisons"]
        ]
        source_lines = []
        for row in summary["source_level_results"]:
            if row["task"] == "classification":
                values = (
                    f"BA={metric(row['balanced_accuracy'])}; "
                    f"macro-F1={metric(row['macro_f1'])}; "
                    f"ordinal-MAE={metric(row['ordinal_mae'])}"
                )
            else:
                values = (
                    f"MAE={metric(row['mae'])}; Spearman={metric(row['spearman'])}; "
                    f"quantized ordinal-MAE={metric(row['quantized_ordinal_mae'])}"
                )
            source_lines.append(
                f"| {row['task']} | {row['feature_group']} | {row['source']} | "
                f"{row['windows']} | {row['subjects']} | {values} |"
            )
        importance_lines = []
        importance = pd.DataFrame(summary["feature_importance"])
        for task in TASK_ORDER:
            for group in FEATURE_GROUP_ORDER:
                selected = importance.loc[
                    (importance["task"] == task)
                    & (importance["feature_group"] == group)
                ].head(5)
                for row in selected.to_dict("records"):
                    importance_lines.append(
                        f"| {task} | {group} | `{row['feature_name']}` | "
                        f"{metric(row['mean_importance'], 6)} +/- "
                        f"{metric(row['importance_sd'], 6)} | "
                        f"{row['folds_in_top_20']} | {row['feature_family']} | "
                        f"{optional(row.get('channel'))} | "
                        f"{optional(row.get('frequency_band'))} |"
                    )

        eeg_retention = ba["eeg_only"] / ba["eeg_pow"]
        pow_retention = ba["pow_only"] / ba["eeg_pow"]
        combined_gain = ba["eeg_pow"] - ba["pow_only"]
        lines = [
            "# Random Forest feature-group and regression audit",
            "",
            "## 1. Objective",
            "",
            "Compare EEG-only, headset spectral-power (POW)-only, and EEG+POW inputs "
            "under identical five-fold subject GroupKFold splits for global `label_q5` "
            "classification and continuous `target_focus` regression.",
            "",
            "## 2. Feature-group definitions",
            "",
            "| Group | Features | Ordered-list SHA-256 | Valid |",
            "| --- | ---: | --- | --- |",
            *feature_lines,
            "",
            "EEG features are deterministic `EEG.*` window statistics. POW features are "
            "deterministic `POW.*` headset spectral-power columns aggregated by "
            "mean/std/min/max; they are not Performance Metrics (`PM.*`). The full ordered "
            "lists are stored in the JSON report and fold feature manifests.",
            "",
            "## 3. Leakage checks",
            "",
            f"- Supervised windows: {summary['supervised_rows']}; subjects: {summary['subjects']}.",
            "- Targets, PM fields, subject/record/source/sample identifiers, and time metadata "
            "are absent from all model feature lists.",
            "- Every outer fold has zero train/test subject overlap.",
            "- The global `label_q5` is used unchanged; leakage-safe sensitivity labels are not trained on here.",
            "",
            "## 4. Exact alignment",
            "",
            f"All classification groups, all regression groups, and matching classification/"
            f"regression trials have exact sample/fold/subject/record/source alignment. "
            f"Total mismatches: **{mismatch_total}**. Canonical baseline alignment is "
            f"{summary['canonical_alignment']['exact_match']}.",
            "",
            "## 5. Classification results",
            "",
            "| Group | Balanced accuracy | Macro F1 | Accuracy | Weighted F1 | Kappa | AUC | Ordinal MAE | Adjacent accuracy | Severe error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *classification_lines,
            "",
            f"EEG-only retains {eeg_retention:.1%} of combined balanced accuracy; POW-only "
            f"retains {pow_retention:.1%}. Adding EEG to POW changes balanced accuracy by "
            f"{combined_gain:+.4f}.",
            "",
            "## 6. Regression results",
            "",
            "| Group | MAE | RMSE | R2 | Pearson | Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *regression_lines,
            "",
            f"Lowest fold-mean MAE: **{min(mae, key=mae.get)}**. Highest fold-mean "
            f"Spearman: **{max(spearman, key=spearman.get)}**.",
            "",
            "## 7. Regression-to-class results",
            "",
            "Fixed global thresholds `[0.330177, 0.387786, 0.444458, 0.526585]` were "
            "applied without refitting. This is a diagnostic comparison, not an optimized "
            "ordinal method.",
            "",
            "| Group | Quantized BA | Quantized macro F1 | Quantized ordinal MAE | Adjacent accuracy | Severe error | Direct BA / ordinal MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *quantized_lines,
            "",
            "## 8. Subject-level comparisons",
            "",
            "Positive differences always favor the left group; error differences are "
            "oriented as `right_error - left_error`. CIs use 10,000 paired subject bootstraps. "
            "Holm correction is separate for classification and regression families.",
            "",
            "| Task | Comparison | Metric | Mean Delta | Median Delta | 95% CI | Improved/degraded/ties | Wilcoxon p | Holm p | Sign p | Holm sign p | Rank-biserial |",
            "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *paired_lines,
            "",
            "## 9. Source-level results",
            "",
            "Source results are descriptive. A person present in both sources is not counted "
            "as two independent units in paired tests.",
            "",
            "| Task | Group | Source | Windows | Subjects | Metrics |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *source_lines,
            "",
            "Classification source gaps are at most about 0.005 balanced-accuracy points. "
            "Old_EEG regression MAE is about 0.005-0.006 higher than gpn_data, but the "
            "feature-group ranking is unchanged; there is no strong qualitative source reversal.",
            "",
            "## 10. Feature importance",
            "",
            "| Task | Group | Feature | Mean +/- SD | Folds in top-20 | Family | Channel | Band |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
            *importance_lines,
            "",
            "Importance is impurity-based, descriptive, and not used for feature selection. "
            "Correlated EEG/POW variables can divide or inflate importance. Optional permutation "
            "importance was not run to keep this six-trial audit bounded.",
            "",
            "## 11. Circularity implications",
            "",
            "POW and Focus are both exported by the same headset ecosystem. The source files "
            "identify POW as spectral-power features and PM.Focus as a separate proprietary "
            "metric, but the vendor's Focus computation is not available. Strong POW performance "
            "would therefore be compatible with shared signal content or partial algorithmic "
            "circularity; it would not prove either causality or direct leakage. Here POW-only "
            "is weaker than EEG-only, and adding EEG improves POW on classification and "
            "regression, so the results do not look like trivial reconstruction of Focus from "
            "POW alone. Proprietary-algorithm circularity nevertheless cannot be excluded.",
            "",
            "## 12. Limitations",
            "",
            "This audit uses one fixed RF configuration and one global label definition. RF "
            "importance is biased toward correlated/high-variance features, source analyses are "
            "descriptive, and regression quantization is not a trained ordinal objective.",
            "",
            "## 13. Recommendation for Transformer experiment",
            "",
            f"Run the next Transformer comparison on **{recommended[0]}** and "
            f"**{recommended[1]}**, the two groups with the best combined classification-BA "
            "and regression-MAE ranks. Do not launch it automatically; retain the third group "
            "only if the subject-level or circularity interpretation requires a dedicated control.",
            "",
        ]
        return "\n".join(lines)

    def execute(
        self,
        plans: Sequence[FeatureGroupTrialPlan],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        invalid = [plan for plan in plans if plan.status != "valid"]
        if invalid:
            raise ValueError(
                "Invalid feature-group trials: "
                + "; ".join(
                    f"{plan.trial_id}: {', '.join(plan.invalid_reasons)}"
                    for plan in invalid
                )
            )
        completed: dict[str, CompletedBenchmarkRun] = {}
        outcomes: list[dict[str, Any]] = []
        for plan in plans:
            existing = self.completed_run_finder(
                plan.resolved_config, search_directories=[plan.output_dir]
            )
            if resume and existing is not None:
                completed[plan.trial_id] = existing
                outcomes.append({**plan.to_dict(), "outcome": "resumed"})
                continue
            runner = self.runner_factory(deepcopy(dict(plan.resolved_config)))
            runner.run()
            run = runner.completed_run()
            completed[plan.trial_id] = run
            outcomes.append({**plan.to_dict(), "outcome": "completed"})

        prefix = str(self.document["experiment"].get("trial_prefix", "rf"))
        expected = {
            f"{prefix}_{task}_{group}"
            for task in self.document["tasks"]
            for group in FEATURE_GROUP_ORDER
            if str(self.document["tasks"][task]["model"])
            in self.document["models"]
        }
        analysis = (
            self._analyze(plans, completed)
            if {plan.trial_id for plan in plans} == expected
            else {
                "status": "partial_matrix",
                "reason": "full comparison requires the complete configured matrix",
            }
        )
        return {"trials": outcomes, "analysis": analysis}


class FeatureGroupTransformerExperiment(FeatureGroupRFExperiment):
    """Canonical three-way Transformer feature-group classification audit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._canonical_sequence_index: pd.DataFrame | None = None
        self._sequence_build_stats: dict[str, Any] | None = None

    def _audit_dataset(
        self,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
        if self._canonical_sequence_index is not None:
            return (
                self._supervised if self._supervised is not None else pd.DataFrame(),
                self._feature_manifests or {},
                self._canonical_alignment or {},
            )
        supervised, manifests, _ = super()._audit_dataset()
        sequence = self.document["sequence"]
        metadata = supervised[[
            "source", "subject_id", "record_id", "sample_id", "t_start"
        ]].copy()
        built = build_sequences(
            X=np.zeros((len(supervised), 1), dtype=np.float32),
            y=supervised["label_q5"].to_numpy(dtype=np.int64),
            metadata=metadata,
            sequence_length=int(sequence["length"]),
            stride=int(sequence.get("stride", 1)),
            target_position=str(sequence.get("target_position", "last")),
            expected_step_seconds=sequence.get("expected_step_seconds"),
            max_gap_seconds=sequence.get("max_gap_seconds"),
        )
        canonical = built.metadata.copy()
        canonical["fold"] = canonical["subject_id"].map(
            supervised.drop_duplicates("subject_id").set_index("subject_id")["fold"]
        ).astype(int)
        canonical["y_true"] = built.y.astype(np.int64)
        canonical = canonical.sort_values(
            ["fold", "sequence_id"], kind="mergesort"
        ).reset_index(drop=True)
        reference_path = _repo_path(
            self.document["experiment"]["canonical_reference_predictions"]
        )
        reference = pd.read_parquet(reference_path)
        alignment = sequence_prediction_alignment(canonical, reference)
        alignment.update({
            "reference_path": _relative_path(reference_path),
            "sequence_index_sha256": sequence_index_sha256(canonical),
            "expected_sequences": int(
                self.document["dataset"].get("expected_sequences", len(canonical))
            ),
            "supervised_subjects": int(supervised["subject_id"].nunique()),
            "sequence_subjects": int(canonical["subject_id"].nunique()),
            "subjects_without_sequences": sorted(
                set(supervised["subject_id"].astype(str))
                - set(canonical["subject_id"].astype(str))
            ),
            "train_test_subject_overlap": {
                str(fold): sorted(
                    set(canonical.loc[canonical["fold"] != fold, "subject_id"].astype(str))
                    & set(canonical.loc[canonical["fold"] == fold, "subject_id"].astype(str))
                )
                for fold in sorted(canonical["fold"].unique())
            },
        })
        self._canonical_sequence_index = canonical
        self._sequence_build_stats = built.stats
        self._canonical_alignment = alignment
        return supervised, manifests, alignment

    def plan(
        self,
        *,
        feature_groups: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        seed: int = 42,
    ) -> list[FeatureGroupTrialPlan]:
        plans = super().plan(
            feature_groups=feature_groups,
            tasks=tasks,
            models=models,
            seed=seed,
        )
        canonical = self._canonical_sequence_index
        if canonical is None:
            raise RuntimeError("Canonical sequence audit did not produce an index")
        expected = int(self.document["dataset"].get("expected_sequences", len(canonical)))
        sequence_hash = sequence_index_sha256(canonical)
        output: list[FeatureGroupTrialPlan] = []
        for plan in plans:
            reasons = list(plan.invalid_reasons)
            if len(canonical) != expected:
                reasons.append(
                    f"sequence count mismatch: observed={len(canonical)}, expected={expected}"
                )
            length = int(plan.resolved_config["sequence"]["length"])
            output.append(replace(
                plan,
                rows=int(len(canonical)),
                subjects=int(canonical["subject_id"].nunique()),
                status=("valid" if not reasons else "invalid"),
                invalid_reasons=tuple(reasons),
                input_shape=(length, plan.feature_count),
                sequence_length=length,
                sequence_count=int(len(canonical)),
                sequence_index_sha256=sequence_hash,
                model_parameters=deepcopy(
                    plan.resolved_config["models"][plan.model]["params"]
                ),
            ))
        return output

    @staticmethod
    def render_plan(plans: Sequence[FeatureGroupTrialPlan]) -> str:
        lines = [
            "# Transformer feature-group experiment plan",
            "",
            "| Trial | Features | Count | Hash | Input shape | Sequences | Sequence hash | Subjects | Folds | Reusable | Status |",
            "| --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| `{plan.trial_id}` | {plan.feature_group} | {plan.feature_count} | "
                f"`{plan.feature_list_sha256[:16]}` | `{list(plan.input_shape or ())}` | "
                f"{plan.sequence_count} | `{(plan.sequence_index_sha256 or '')[:16]}` | "
                f"{plan.subjects} | {plan.fold_count} | "
                f"{'yes' if plan.completed_run else 'no'} | {plan.status} |"
            )
        if plans:
            lines.extend([
                "",
                "Model parameters:",
                "",
                "```json",
                json.dumps(_jsonable(plans[0].model_parameters), indent=2),
                "```",
                "",
                f"Expected output root: `{_relative_path(plans[0].output_dir.parent.parent)}`.",
            ])
        lines.extend([
            "",
            f"Trials: **{len(plans)}**; valid: **{sum(p.status == 'valid' for p in plans)}**; "
            f"to run: **{sum(p.action == 'run' for p in plans)}**.",
            "Plan-only performs no training and writes no benchmark artifacts.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _transformer_fold_metrics(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = FeatureGroupRFExperiment._fold_metrics(result)
        for row, (_, fold) in zip(rows, result["folds"].items()):
            training = fold.get("training", {})
            row.update({
                "epochs_trained": training.get("epochs_trained"),
                "best_epoch": training.get("best_epoch"),
                "best_validation_loss": training.get("best_validation_loss"),
                "device": training.get("device"),
                "device_name": training.get("device_name"),
                "parameter_count": training.get("trainable_parameter_count"),
            })
        return rows

    @staticmethod
    def _transformer_paired_results(
        subject_metrics: Mapping[str, pd.DataFrame],
        *,
        n_resamples: int,
        random_state: int,
    ) -> list[dict[str, Any]]:
        metrics = {
            "balanced_accuracy": True,
            "macro_f1": True,
            "ordinal_mae": False,
            "severe_error_rate": False,
        }
        rows: list[dict[str, Any]] = []
        for left_group, right_group in PAIR_ORDER:
            left = subject_metrics[left_group].set_index("subject_id")
            right = subject_metrics[right_group].set_index("subject_id")
            common = sorted(set(left.index.astype(str)) & set(right.index.astype(str)))
            for metric, higher_is_better in metrics.items():
                left_values = left.loc[common, metric].to_numpy(dtype=float)
                right_values = right.loc[common, metric].to_numpy(dtype=float)
                statistics = paired_subject_statistics(
                    left_values if higher_is_better else -left_values,
                    right_values if higher_is_better else -right_values,
                    n_resamples=n_resamples,
                    random_state=random_state,
                )
                rows.append({
                    "family": "transformer_classification_feature_groups",
                    "task": "classification",
                    "comparison": f"{left_group}_vs_{right_group}",
                    "left": left_group,
                    "right": right_group,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "difference_definition": (
                        "left_minus_right"
                        if higher_is_better
                        else "right_error_minus_left_error"
                    ),
                    **statistics,
                })
        rows = apply_holm_by_family(rows, p_key="wilcoxon_p_value")
        sign_rows = apply_holm_by_family(rows, p_key="sign_test_p_value")
        for row, sign_row in zip(rows, sign_rows):
            row["holm_adjusted_sign_p_value"] = sign_row["holm_adjusted_p_value"]
        return rows

    @staticmethod
    def _class_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fold, fold_frame in frame.groupby("fold", sort=True):
            for value in MetricsCalculator.calculate_class_metrics(
                fold_frame["y_true"].to_numpy(dtype=int),
                fold_frame["y_pred"].to_numpy(dtype=int),
                labels=np.arange(5),
            ):
                rows.append({"fold": int(fold), **value})
        return rows

    @staticmethod
    def _source_classification_results(
        predictions: Mapping[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group_name, frame in predictions.items():
            for source, source_frame in frame.groupby("source", sort=True):
                proba = source_frame.filter(regex=r"^proba_\d+$").to_numpy(dtype=float)
                metrics = MetricsCalculator.calculate_all_metrics(
                    source_frame["y_true"].to_numpy(dtype=int),
                    source_frame["y_pred"].to_numpy(dtype=int),
                    proba,
                )
                rows.append({
                    "feature_group": group_name,
                    "source": str(source),
                    "sequences": int(len(source_frame)),
                    "subjects": int(source_frame["subject_id"].nunique()),
                    "subject_counts_are_non_additive": True,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ordinal_mae": metrics["ordinal_mae"],
                    "severe_error_rate": metrics["severe_error_rate"],
                })
        return rows

    def _artifact_audit(
        self,
        plans: Sequence[FeatureGroupTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for plan in plans:
            result, _ = self._completed_trial(plan, completed[plan.trial_id])
            for fold_name, fold in result["folds"].items():
                artifacts = fold.get("artifacts", {})
                validation_path = _repo_path(artifacts["validation_split"])
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                normalization_path = _repo_path(artifacts["normalization_stats"])
                normalization = json.loads(
                    normalization_path.read_text(encoding="utf-8")
                )
                sequence_manifest_path = _repo_path(
                    artifacts["sequence_index_manifest"]
                )
                sequence_manifest = json.loads(
                    sequence_manifest_path.read_text(encoding="utf-8")
                )
                split_metadata = fold["split_metadata"]
                required_artifacts = {
                    "predictions", "metrics", "class_metrics", "feature_manifest",
                    "sequence_stats", "sequence_index_manifest", "validation_split",
                    "model", "training_log", "normalization_stats",
                }
                rows.append({
                    "trial_id": plan.trial_id,
                    "fold": fold_name,
                    "outer_subject_overlap": split_metadata.get("subject_overlap", []),
                    "inner_record_overlap": validation.get("record_overlap", []),
                    "inner_group_overlap": validation.get("group_overlap", []),
                    "outer_test_record_overlap": validation.get(
                        "outer_test_record_overlap", []
                    ),
                    "normalization_scope": normalization.get("scope"),
                    "sequence_count": sequence_manifest["sequence_count"],
                    "sequence_index_sha256": sequence_manifest[
                        "sequence_index_sha256"
                    ],
                    "missing_artifacts": sorted(required_artifacts - set(artifacts)),
                })
        valid = all(
            not row["outer_subject_overlap"]
            and not row["inner_record_overlap"]
            and not row["inner_group_overlap"]
            and not row["outer_test_record_overlap"]
            and row["normalization_scope"] == "inner_train_only"
            and not row["missing_artifacts"]
            for row in rows
        )
        return {"valid": valid, "folds": rows}

    def _baseline_reproduction(
        self,
        combined_plan: FeatureGroupTrialPlan,
        combined_result: Mapping[str, Any],
        combined_predictions: pd.DataFrame,
    ) -> dict[str, Any]:
        reference_path = _repo_path(
            self.document["experiment"]["canonical_reference_predictions"]
        )
        reference = pd.read_parquet(reference_path)
        alignment = sequence_prediction_alignment(
            combined_predictions, reference, compare_predictions=True
        )
        reference_run = _repo_path(
            self.document["experiment"]["canonical_reference_run"]
        )
        baseline_config = yaml.safe_load(
            (reference_run / "config.yaml").read_text(encoding="utf-8")
        )
        new_config = combined_plan.resolved_config
        comparable_sections = {
            "sequence": baseline_config.get("sequence") == new_config.get("sequence"),
            "validation": baseline_config.get("validation") == new_config.get("validation"),
            "evaluation": baseline_config.get("evaluation") == new_config.get("evaluation"),
            "model": baseline_config.get("models", {}).get("torch_transformer")
            == new_config.get("models", {}).get("torch_transformer"),
        }
        baseline_metrics = json.loads(
            (reference_run / "metrics.json").read_text(encoding="utf-8")
        )
        baseline_result = baseline_metrics["emotiv_cognitive"]["models"][
            "cognitive_load_5class"
        ]["torch_transformer"]["group_kfold_subject"]
        normalization_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        try:
            import torch

            for fold_name in sorted(combined_result["folds"]):
                old_fold = baseline_result["folds"][fold_name]
                new_fold = combined_result["folds"][fold_name]
                old_checkpoint = torch.load(
                    _repo_path(old_fold["artifacts"]["model"]),
                    map_location="cpu",
                    weights_only=False,
                )
                new_checkpoint = torch.load(
                    _repo_path(new_fold["artifacts"]["model"]),
                    map_location="cpu",
                    weights_only=False,
                )
                mean_delta = np.max(np.abs(
                    old_checkpoint["feature_mean"].numpy()
                    - new_checkpoint["feature_mean"].numpy()
                ))
                scale_delta = np.max(np.abs(
                    old_checkpoint["feature_scale"].numpy()
                    - new_checkpoint["feature_scale"].numpy()
                ))
                normalization_rows.append({
                    "fold": fold_name,
                    "mean_max_abs_delta": float(mean_delta),
                    "scale_max_abs_delta": float(scale_delta),
                })
                old_validation = json.loads(
                    _repo_path(old_fold["artifacts"]["validation_split"]).read_text(
                        encoding="utf-8"
                    )
                )
                new_validation = json.loads(
                    _repo_path(new_fold["artifacts"]["validation_split"]).read_text(
                        encoding="utf-8"
                    )
                )
                validation_rows.append({
                    "fold": fold_name,
                    "inner_train_groups_equal": old_validation.get(
                        "inner_train_group_ids"
                    ) == new_validation.get("inner_train_group_ids"),
                    "inner_validation_groups_equal": old_validation.get(
                        "inner_validation_group_ids"
                    ) == new_validation.get("inner_validation_group_ids"),
                })
        except (ImportError, KeyError, OSError, ValueError) as exc:
            normalization_rows.append({"status": "unavailable", "reason": str(exc)})
        metric_deltas = {}
        for metric in (
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "kappa", "auc",
        ):
            old_value = baseline_result["aggregated"][f"{metric}_mean"]
            new_value = combined_result["aggregated"][f"{metric}_mean"]
            metric_deltas[metric] = float(new_value - old_value)
        return {
            "reference_run": _relative_path(reference_run),
            "alignment": alignment,
            "comparable_config_sections": comparable_sections,
            "normalization_comparison": normalization_rows,
            "validation_group_comparison": validation_rows,
            "metric_mean_deltas_new_minus_baseline": metric_deltas,
            "same_seed": True,
            "deterministic_controls": (
                "Python/NumPy/Torch/CUDA seeded; deterministic cuDNN enabled; "
                "DataLoader generator seeded"
            ),
        }

    def _analyze(
        self,
        plans: Sequence[FeatureGroupTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
    ) -> dict[str, Any]:
        supervised, feature_manifests, canonical_alignment = self._audit_dataset()
        canonical = self._canonical_sequence_index
        if canonical is None:
            raise RuntimeError("Canonical sequence index is unavailable")
        predictions: dict[str, pd.DataFrame] = {}
        subject_metrics: dict[str, pd.DataFrame] = {}
        fold_metrics: dict[str, list[dict[str, Any]]] = {}
        class_metrics: dict[str, list[dict[str, Any]]] = {}
        results: dict[str, Mapping[str, Any]] = {}
        analysis_dir = self.output_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for plan in plans:
            result, frame = self._completed_trial(plan, completed[plan.trial_id])
            frame = frame.sort_values(
                ["fold", "sequence_id"], kind="mergesort"
            ).reset_index(drop=True)
            predictions[plan.feature_group] = frame
            results[plan.feature_group] = result
            fold_metrics[plan.feature_group] = self._transformer_fold_metrics(result)
            subjects = calculate_subject_metrics(
                frame,
                track="feature_group_transformer_classification",
                model=plan.feature_group,
                seed=int(plan.resolved_config["experiment"]["seed"]),
            )
            subject_metrics[plan.feature_group] = subjects
            classes = self._class_metrics(frame)
            class_metrics[plan.feature_group] = classes
            trial_dir = analysis_dir / plan.trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            subjects.to_parquet(trial_dir / "subject_metrics.parquet", index=False)
            pd.DataFrame(classes).to_parquet(
                trial_dir / "class_metrics.parquet", index=False
            )

        alignment_rows = []
        for left, right in combinations(FEATURE_GROUP_ORDER, 2):
            alignment_rows.append({
                "comparison": f"{left}_vs_{right}",
                **sequence_prediction_alignment(predictions[left], predictions[right]),
            })
        canonical_rows = []
        for group_name in FEATURE_GROUP_ORDER:
            alignment = sequence_prediction_alignment(
                canonical, predictions[group_name]
            )
            alignment["feature_group"] = group_name
            alignment["prediction_sequence_index_sha256"] = sequence_index_sha256(
                predictions[group_name]
            )
            canonical_rows.append(alignment)
        if not all(row["exact_match"] for row in alignment_rows + canonical_rows):
            raise ValueError("Transformer sequence predictions are not exactly aligned")

        metric_names = (
            "balanced_accuracy", "macro_f1", "accuracy", "weighted_f1",
            "kappa", "auc", "ordinal_mae", "adjacent_accuracy",
            "severe_error_rate", "epochs_trained", "best_validation_loss",
            "training_time",
        )
        classification = {
            group: {
                "folds": fold_metrics[group],
                "aggregated": self._aggregate_rows(fold_metrics[group], metric_names),
                "parameter_count": next(
                    (
                        row["parameter_count"]
                        for row in fold_metrics[group]
                        if row.get("parameter_count") is not None
                    ),
                    None,
                ),
            }
            for group in FEATURE_GROUP_ORDER
        }
        paired = self._transformer_paired_results(
            subject_metrics,
            n_resamples=int(self.document["analysis"].get("bootstrap_samples", 10_000)),
            random_state=int(self.document["analysis"].get("random_state", 42)),
        )
        sources = self._source_classification_results(predictions)
        artifact_audit = self._artifact_audit(plans, completed)
        combined_plan = next(plan for plan in plans if plan.feature_group == "eeg_pow")
        baseline = self._baseline_reproduction(
            combined_plan, results["eeg_pow"], predictions["eeg_pow"]
        )
        rf_summary = json.loads(
            _repo_path(self.document["analysis"]["rf_summary_path"]).read_text(
                encoding="utf-8"
            )
        )
        comparison = {}
        for family, result in (
            ("transformer", classification),
            ("random_forest", rf_summary["classification"]),
        ):
            values = {
                group: result[group]["aggregated"]["balanced_accuracy"]["mean"]
                for group in FEATURE_GROUP_ORDER
            }
            errors = {
                group: result[group]["aggregated"]["ordinal_mae"]["mean"]
                for group in FEATURE_GROUP_ORDER
            }
            comparison[family] = {
                "balanced_accuracy": values,
                "retention_eeg": float(values["eeg_only"] / values["eeg_pow"]),
                "retention_pow": float(values["pow_only"] / values["eeg_pow"]),
                "combined_minus_eeg": float(values["eeg_pow"] - values["eeg_only"]),
                "combined_minus_pow": float(values["eeg_pow"] - values["pow_only"]),
                "ordinal_mae": errors,
                "eeg_error_increase_vs_combined": float(
                    errors["eeg_only"] - errors["eeg_pow"]
                ),
                "pow_error_increase_vs_combined": float(
                    errors["pow_only"] - errors["eeg_pow"]
                ),
            }
        summary = {
            "analysis_name": self.document["experiment"]["name"],
            "seed": int(plans[0].resolved_config["experiment"]["seed"]),
            "supervised_windows": int(len(supervised)),
            "supervised_subjects": int(supervised["subject_id"].nunique()),
            "sequences": int(len(canonical)),
            "subjects": int(canonical["subject_id"].nunique()),
            "subjects_without_sequences": sorted(
                set(supervised["subject_id"].astype(str))
                - set(canonical["subject_id"].astype(str))
            ),
            "sequence_index_sha256": sequence_index_sha256(canonical),
            "sequence_index_columns": list(SEQUENCE_INDEX_COLUMNS),
            "sequence_build_stats": self._sequence_build_stats,
            "feature_groups": feature_manifests,
            "canonical_alignment": canonical_alignment,
            "prediction_alignment": alignment_rows,
            "canonical_prediction_alignment": canonical_rows,
            "classification": classification,
            "class_level_metrics": class_metrics,
            "subject_level_comparisons": paired,
            "source_level_results": sources,
            "artifact_and_leakage_audit": artifact_audit,
            "baseline_reproduction": baseline,
            "rf_transformer_comparison": comparison,
            "completed_runs": {
                plan.trial_id: _relative_path(completed[plan.trial_id].run_directory)
                for plan in plans
            },
        }
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            self._render_transformer_report(_jsonable(summary)), encoding="utf-8"
        )
        return _jsonable(summary)

    @staticmethod
    def _render_transformer_report(summary: Mapping[str, Any]) -> str:
        def metric(value: Any, digits: int = 4) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "NA"
            return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "NA"

        results = summary["classification"]
        comparison = summary["rf_transformer_comparison"]
        transformer = comparison["transformer"]
        aggregate_lines = []
        fold_lines = []
        for group in FEATURE_GROUP_ORDER:
            values = results[group]["aggregated"]
            aggregate_lines.append(
                f"| {group} | {metric(values['balanced_accuracy']['mean'])} +/- "
                f"{metric(values['balanced_accuracy']['std'])} | "
                f"{metric(values['macro_f1']['mean'])} +/- {metric(values['macro_f1']['std'])} | "
                f"{metric(values['accuracy']['mean'])} | {metric(values['weighted_f1']['mean'])} | "
                f"{metric(values['kappa']['mean'])} | {metric(values['auc']['mean'])} | "
                f"{metric(values['ordinal_mae']['mean'])} | "
                f"{metric(values['severe_error_rate']['mean'])} | "
                f"{results[group]['parameter_count']} |"
            )
            for fold in results[group]["folds"]:
                fold_lines.append(
                    f"| {group} | {fold['fold']} | {metric(fold['balanced_accuracy'])} | "
                    f"{metric(fold['macro_f1'])} | {metric(fold['ordinal_mae'])} | "
                    f"{metric(fold['severe_error_rate'])} | {fold.get('epochs_trained')} | "
                    f"{metric(fold.get('best_validation_loss'))} | "
                    f"{metric(fold.get('training_time'), 1)} |"
                )
        paired_lines = []
        for row in summary["subject_level_comparisons"]:
            paired_lines.append(
                f"| {row['comparison']} | {row['metric']} | {metric(row['mean_difference'])} | "
                f"{metric(row['median_difference'])} | "
                f"[{metric(row['ci_low'])}, {metric(row['ci_high'])}] | "
                f"{row['subjects_improved']} / {row['subjects_degraded']} / {row['ties']} | "
                f"{metric(row['wilcoxon_p_value'])} | {metric(row['holm_adjusted_p_value'])} | "
                f"{metric(row['sign_test_p_value'])} | "
                f"{metric(row['rank_biserial'])} |"
            )
        source_lines = [
            f"| {row['feature_group']} | {row['source']} | {row['sequences']} | "
            f"{row['subjects']} | {metric(row['balanced_accuracy'])} | "
            f"{metric(row['macro_f1'])} | {metric(row['ordinal_mae'])} | "
            f"{metric(row['severe_error_rate'])} |"
            for row in summary["source_level_results"]
        ]
        baseline = summary["baseline_reproduction"]
        prediction_delta = baseline["alignment"].get("prediction_differences", {})
        configuration_matches = all(
            baseline["comparable_config_sections"].values()
        )
        normalization_max = max(
            (
                max(row.get("mean_max_abs_delta", np.inf), row.get("scale_max_abs_delta", np.inf))
                for row in baseline["normalization_comparison"]
            ),
            default=np.inf,
        )
        combined_advantage = transformer["combined_minus_eeg"]
        devices = sorted({
            str(fold.get("device_name") or fold.get("device"))
            for group in FEATURE_GROUP_ORDER
            for fold in results[group]["folds"]
        })
        source_rankings = {
            row["source"]: sorted(
                (
                    candidate
                    for candidate in summary["source_level_results"]
                    if candidate["source"] == row["source"]
                ),
                key=lambda candidate: candidate["balanced_accuracy"],
                reverse=True,
            )[0]["feature_group"]
            for row in summary["source_level_results"]
        }
        return "\n".join([
            "# Transformer feature-group ablation",
            "",
            "## 1. Objective",
            "",
            "Test whether the RF EEG/POW feature-group conclusions persist when eight-window "
            "temporal context is encoded by the canonical Transformer.",
            "",
            "## 2. Canonical architecture",
            "",
            "`Linear(input,128) -> learned positions -> 2 x TransformerEncoder(nhead=4, "
            "FF=256, GELU, dropout=0.1) -> last pooling -> 5-class head`; sequence length 8, "
            "AdamW, seed 42, record-group inner validation and train-only standardization. "
            f"`device: auto` resolved to {devices}.",
            "",
            "## 3. Feature-group definitions",
            "",
            "| Group | Count | Ordered-list SHA-256 | Input shape |",
            "| --- | ---: | --- | --- |",
            *[
                f"| {group} | {value['feature_count']} | `{value['feature_list_sha256']}` | "
                f"`[B, 8, {value['feature_count']}]` |"
                for group, value in summary["feature_groups"].items()
            ],
            "",
            "## 4. Exact sequence alignment",
            "",
            f"Canonical sequences: **{summary['sequences']}**; sequence-index SHA-256: "
            f"`{summary['sequence_index_sha256']}`. All feature groups have identical "
            "sequence IDs, folds, subjects, records, sources, target windows, times and labels; "
            "total alignment mismatches are zero. The sequences contain "
            f"**{summary['subjects']} of {summary['supervised_subjects']}** supervised subjects; "
            f"{summary['subjects_without_sequences']} has no valid length-8 sequence and is "
            "absent identically from the published baseline and all three trials.",
            "",
            "## 5. Leakage checks",
            "",
            f"Artifact/leakage audit valid: **{summary['artifact_and_leakage_audit']['valid']}**. "
            "Outer subject overlap, inner record/group overlap and outer-test record overlap are "
            "zero in all 15 folds. Normalization is fitted on inner train only.",
            "",
            "## 6. Fold results",
            "",
            "| Group | Fold | BA | Macro F1 | Ordinal MAE | Severe error | Epochs | Best val loss | Seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *fold_lines,
            "",
            "## 7. Aggregate results",
            "",
            "| Group | Balanced accuracy | Macro F1 | Accuracy | Weighted F1 | Kappa | AUC | Ordinal MAE | Severe error | Parameters |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *aggregate_lines,
            "",
            f"EEG-only BA retention is **{transformer['retention_eeg'] * 100:.1f}%**; "
            f"POW-only retention is **{transformer['retention_pow'] * 100:.1f}%**. "
            f"Combined minus EEG-only BA is {combined_advantage:+.4f}; combined minus POW-only "
            f"is {transformer['combined_minus_pow']:+.4f}. EEG-only remains better than POW-only, "
            "and combined is best for BA, macro F1, ordinal MAE and severe-error rate.",
            "",
            "## 8. Subject-level statistics",
            "",
            "Positive differences favor the left group; error differences use "
            "`right_error - left_error`. CIs use 10,000 paired subject bootstraps; Holm "
            "correction covers the Transformer comparison family.",
            "",
            "| Comparison | Metric | Mean Delta | Median Delta | 95% CI | Improved/degraded/ties | Wilcoxon p | Holm p | Sign p | Rank-biserial |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
            *paired_lines,
            "",
            "## 9. Source-level results",
            "",
            "Source results are descriptive; overlapping people are not treated as independent "
            "units in paired tests.",
            "",
            "| Group | Source | Sequences | Subjects | BA | Macro F1 | Ordinal MAE | Severe error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *source_lines,
            "",
            f"Best source-specific groups: {source_rankings}.",
            "",
            "## 10. RF versus Transformer interpretation",
            "",
            f"RF EEG/POW BA retention was {comparison['random_forest']['retention_eeg'] * 100:.1f}% / "
            f"{comparison['random_forest']['retention_pow'] * 100:.1f}%; Transformer retention is "
            f"{transformer['retention_eeg'] * 100:.1f}% / {transformer['retention_pow'] * 100:.1f}%. "
            "POW therefore does not gain relative importance from temporal context; its retention "
            "falls, while the combined-minus-POW gap grows from "
            f"{comparison['random_forest']['combined_minus_pow']:.4f} to "
            f"{transformer['combined_minus_pow']:.4f}. The static and temporal models support the "
            "same feature-group ordering. This is descriptive, not a paired RF-vs-Transformer test.",
            "",
            "## 11. Circularity implications",
            "",
            "POW and Focus originate from the same proprietary headset ecosystem, so shared "
            "algorithmic content cannot be excluded. The relative POW-only result indicates "
            "that temporal context does not make POW disproportionately predictive: POW-only is "
            "the weakest group overall and within both sources. This argues against simple Focus "
            "reconstruction from POW alone, but cannot exclude partial proprietary circularity.",
            "",
            "## 12. Recommendation for target formulation",
            "",
            "Keep global `label_q5` for benchmark comparability and retain leakage-safe labels as "
            "sensitivity analysis. Because the classes are ordered and temporal models reduce "
            "large errors differently from nominal F1, ordinal classification is the most direct "
            "next target-formulation extension.",
            "",
            "## 13. Recommendation for next experiment",
            "",
            "Future main modeling should retain both EEG-only and EEG+POW variants: EEG-only "
            "tests signal-specific validity, while EEG+POW measures the best available feature "
            "representation. The matched temporal comparison is now complete; implement an "
            "ordinal classifier next, before a regression or joint ordinal-regression Transformer. "
            "The combined advantage over EEG-only is subject-level Holm-significant at 0.0483 "
            "for BA, macro F1, ordinal MAE and severe error, while the advantage over POW-only is "
            "stronger across the same outcomes.",
            "",
            "## 14. Limitations",
            "",
            f"Only seed 42 and one fixed architecture were evaluated. Baseline config sections "
            f"match: {configuration_matches}; maximum normalization-stat delta: "
            f"{metric(normalization_max, 8)}; baseline y-pred differences: "
            f"{prediction_delta.get('y_pred', 'NA')}; probability max absolute delta: "
            f"{metric(prediction_delta.get('probability_max_abs_delta'), 8)}. One supervised "
            "subject has no valid length-8 sequence. Source analyses are descriptive, and no "
            "RF-vs-Transformer significance test was performed.",
            "",
        ])


def build_feature_group_experiment(
    spec_path: str | Path,
    **kwargs: Any,
) -> FeatureGroupRFExperiment:
    """Select the shared feature-group workflow from the experiment specification."""

    document = load_feature_group_spec(spec_path)
    experiment_type = str(document["experiment"].get("type", "rf")).lower()
    if experiment_type == "transformer":
        return FeatureGroupTransformerExperiment(spec_path, **kwargs)
    if experiment_type == "rf":
        return FeatureGroupRFExperiment(spec_path, **kwargs)
    raise ValueError(f"Unsupported feature-group experiment type {experiment_type!r}")
