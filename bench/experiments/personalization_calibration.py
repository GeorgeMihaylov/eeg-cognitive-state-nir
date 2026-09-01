"""Leakage-safe planning for seven-PM participant personalization.

This module deliberately stops at protocol materialization.  Model fitting is
delegated to the existing benchmark runner and ``TorchClassificationAdapter``;
no second training loop is introduced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from copy import deepcopy

import numpy as np
import pandas as pd

from bench.datasets.datasets_registry import get_dataset
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from bench.tasks.target_transforms import (
    build_fold_local_target_transform,
    build_target_transform_manifest,
    validate_target_transform_manifest,
)
from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter
from cogstate.model_zoo.factory import build_model

from .user_calibration import (
    CalibrationSpec,
    _use_reference_evaluation,
    chronological_window_partition,
)


SCHEMA_VERSION = "personalization-calibration-v1"
SUPPORTED_TASK_TYPES = ("classification", "regression")
SUPPORTED_MODES = ("zero_shot", "head_only", "full_model", "margin_head")
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
)
REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson", "spearman")
PLAN_FILES = (
    "resolved_protocol.json",
    "run_matrix.csv",
    "participant_calibration_plan.csv",
    "model_compatibility.csv",
    "protocol_manifest.json",
    "README.md",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible protocol state."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash(sorted(str(value) for value in values))


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _target_id(pm: str, task_type: str) -> str:
    if task_type == "classification":
        return f"pm_{pm}_q3_fold_local"
    if task_type == "regression":
        return f"pm_{pm}_regression"
    raise ValueError(f"Unknown task_type {task_type!r}")


@dataclass(frozen=True)
class PlanFilters:
    """Optional deterministic subset for a later diagnostic execution."""

    outer_fold: int | None = None
    pm: str | None = None
    task_type: str | None = None
    calibration_mode: str | None = None
    model: str | None = None
    budget_fraction: float | None = None

    def validate(self, n_splits: int) -> None:
        if self.outer_fold is not None and not 1 <= self.outer_fold <= n_splits:
            raise ValueError(f"outer_fold must be in [1, {n_splits}]")
        if self.pm is not None and self.pm not in PM_METRICS:
            raise ValueError(f"Unknown PM {self.pm!r}; expected {list(PM_METRICS)}")
        if self.task_type is not None and self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {list(SUPPORTED_TASK_TYPES)}"
            )
        if (
            self.calibration_mode is not None
            and self.calibration_mode not in SUPPORTED_MODES
        ):
            raise ValueError(
                f"calibration_mode must be one of {list(SUPPORTED_MODES)}"
            )
        if self.model is not None and self.model not in {
            "torch_shallow_convnet", "torch_eegnet", "torch_mlp", "xgboost"
        }:
            raise ValueError(f"Unknown personalization model {self.model!r}")
        if self.budget_fraction is not None and float(self.budget_fraction) not in {
            0.0, 0.01, 0.05, 0.1, 0.2
        }:
            raise ValueError("budget_fraction must be one of 0, .01, .05, .1, .2")


def validate_protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the immutable scientific protocol."""
    resolved = json.loads(json.dumps(dict(config)))
    if resolved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    experiment = resolved.get("experiment", {})
    if experiment.get("type") != "personalization_calibration":
        raise ValueError("experiment.type must be 'personalization_calibration'")
    if not str(experiment.get("experiment_id", "")).strip():
        raise ValueError("experiment.experiment_id is required")

    pms = tuple(resolved.get("pms", ()))
    if pms != tuple(PM_METRICS):
        raise ValueError("pms must contain all seven canonical PMs in registry order")
    tasks = tuple(resolved.get("task_types", ()))
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in SUPPORTED_TASK_TYPES for task in tasks)
        or tasks != tuple(
            task for task in SUPPORTED_TASK_TYPES if task in set(tasks)
        )
    ):
        raise ValueError(
            "task_types must be a non-empty, unique subset of "
            "['classification', 'regression'] in registry order"
        )
    modes = tuple(resolved.get("calibration", {}).get("modes", ()))
    canonical_modes = tuple(
        mode for mode in SUPPORTED_MODES if mode in set(modes)
    )
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(mode not in SUPPORTED_MODES for mode in modes)
        or modes != canonical_modes
        or modes[0] != "zero_shot"
    ):
        raise ValueError(
            "calibration.modes must be a unique ordered subset of "
            "zero_shot, head_only, full_model, margin_head and must "
            "start with zero_shot"
        )
    budgets = tuple(
        float(value)
        for value in resolved.get("calibration", {}).get(
            "budgets_fraction", ()
        )
    )
    if not budgets or budgets[0] != 0.0:
        raise ValueError("budgets_fraction must start with the zero-shot budget 0")
    if tuple(sorted(set(budgets))) != budgets or budgets[-1] >= 1.0:
        raise ValueError("budgets_fraction must be unique, increasing, and below 1")
    resolved["calibration"]["budgets_fraction"] = list(budgets)

    protocol = resolved.get("protocol", {})
    if int(protocol.get("n_outer_folds", 0)) != 5:
        raise ValueError("protocol.n_outer_folds must remain 5")
    if protocol.get("outer_group_column") != "subject_id":
        raise ValueError("protocol.outer_group_column must be subject_id")
    if protocol.get("fixed_outer_fold_column") != "outer_fold":
        raise ValueError("protocol.fixed_outer_fold_column must be outer_fold")
    if protocol.get("split_strategy") != "chronological_prefix":
        raise ValueError("Only chronological_prefix calibration is allowed")
    if protocol.get("fraction_allocation") != "global_prefix":
        raise ValueError("Only strict participant-level global_prefix is allowed in v1")
    if protocol.get("q3_fit_scope") != "outer_train_only":
        raise ValueError("Q3 fit scope must remain outer_train_only")
    analysis = resolved.get("analysis", {})
    if float(analysis.get("formal_accuracy_threshold", -1.0)) != 0.75:
        raise ValueError("analysis.formal_accuracy_threshold must remain 0.75")
    if analysis.get("aggregation") != "participant_macro":
        raise ValueError("analysis.aggregation must remain participant_macro")
    if analysis.get("threshold_role") != "report_only_not_for_selection":
        raise ValueError(
            "The 75% accuracy threshold must be report-only, not a selection rule"
        )
    if not resolved.get("models"):
        raise ValueError("At least one model is required")
    execution = resolved.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution configuration is required")
    model_ids = {str(item["model_id"]) for item in resolved["models"]}
    configured_model_ids = set(execution.get("model_params", {}))
    if configured_model_ids != model_ids:
        raise ValueError(
            "execution.model_params must define exactly the planned models"
        )
    base_training = execution.get("base_training", {})
    required_training = {
        "max_epochs", "learning_rate", "weight_decay", "validation_size",
        "early_stopping_patience", "device", "random_state",
    }
    missing_training = sorted(required_training - set(base_training))
    if missing_training:
        raise ValueError(
            f"execution.base_training is missing: {missing_training}"
        )
    return resolved


def build_model_compatibility(
    models: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Probe supported personalization modes for each model/task pair."""
    rows: list[dict[str, Any]] = []

    for model_config in models:
        model_name = str(model_config["model_id"])
        family = str(model_config["input_family"])
        input_shape = (1, 14, 2560) if family == "raw" else (448,)

        for task_type in SUPPORTED_TASK_TYPES:
            supported = False
            reason = ""
            head_only = False
            full_model = False
            margin_head = False
            adapter_name = ""

            try:
                if model_name == "xgboost":
                    if family != "features":
                        reason = (
                            "xgboost personalization requires feature input"
                        )
                    elif task_type != "classification":
                        reason = (
                            "xgboost margin_head personalization is "
                            "classification-only"
                        )
                    else:
                        estimator = build_model(
                            model_name="xgboost",
                            task_type="classification",
                            input_shape=None,
                            num_outputs=3,
                            params={
                                "n_estimators": 1,
                                "max_depth": 1,
                                "learning_rate": 0.1,
                                "objective": "multi:softprob",
                                "num_class": 3,
                                "random_state": 42,
                                "n_jobs": 1,
                            },
                        )
                        adapter_name = type(estimator).__name__
                        supported = True
                        margin_head = True

                else:
                    num_outputs = (
                        3 if task_type == "classification" else 1
                    )
                    adapter = build_model(
                        model_name=model_name,
                        task_type=task_type,
                        input_shape=input_shape,
                        num_outputs=num_outputs,
                        params={
                            "device": "cpu",
                            "batch_size": 2,
                            "max_epochs": 1,
                            "early_stopping_patience": 1,
                            "random_state": 42,
                            "num_workers": 0,
                        },
                    )
                    adapter_name = type(adapter).__name__

                    if not isinstance(
                        adapter, TorchClassificationAdapter
                    ):
                        reason = (
                            "factory result does not use "
                            "TorchClassificationAdapter"
                        )
                    else:
                        supported = True
                        full_model = callable(
                            getattr(adapter, "fine_tune", None)
                        )
                        prefixes = getattr(
                            adapter.model,
                            "output_head_parameter_prefixes",
                            None,
                        )
                        head_only = (
                            callable(prefixes)
                            and bool(tuple(prefixes()))
                        )

                        if not full_model:
                            supported = False
                            reason = (
                                "shared adapter fine_tune() is unavailable"
                            )

            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "model": model_name,
                    "input_family": family,
                    "task_type": task_type,
                    "factory_supported": supported,
                    "zero_shot_supported": supported,
                    "head_only_supported": (
                        supported and head_only
                    ),
                    "full_model_supported": (
                        supported and full_model
                    ),
                    "margin_head_supported": (
                        supported and margin_head
                    ),
                    "adapter": adapter_name,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["model", "task_type"],
        kind="mergesort",
    ).reset_index(drop=True)


def fit_outer_train_q3(
    *,
    pm: str,
    outer_fold: int,
    outer_train_sample_ids: np.ndarray,
    outer_train_targets: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Fit one frozen Q3 transform using only explicitly supplied train rows."""
    spec = get_target_spec(_target_id(pm, "classification"))
    transform = build_fold_local_target_transform(spec)
    transform.fit(np.asarray(outer_train_targets))
    manifest = build_target_transform_manifest(
        spec,
        transform,
        outer_fold=outer_fold,
        outer_train_sample_ids=np.asarray(outer_train_sample_ids),
        outer_train_targets=np.asarray(outer_train_targets),
    )
    validate_target_transform_manifest(manifest)
    if manifest["actual_class_count"] != 3:
        raise ValueError(
            f"{pm} fold {outer_fold} produced "
            f"{manifest['actual_class_count']} Q3 classes"
        )
    return transform, manifest


def _participant_partition(
    frame: pd.DataFrame,
    *,
    budget: float,
    reference_budget: float,
    protocol: Mapping[str, Any],
) -> tuple[Any, Any]:
    if frame["subject_id"].astype(str).nunique() != 1:
        raise ValueError("A participant partition cannot mix subject_id values")
    ordered = frame.sort_values(
        ["absolute_t_start", "source", "record_id", "t_start", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True).copy()
    if not np.isfinite(ordered["absolute_t_start"].to_numpy(dtype=float)).all():
        raise ValueError("Participant absolute_t_start values must be finite")
    # Reuse the existing global-prefix splitter while making its ordering
    # explicitly participant-global. Original physical identifiers and times are
    # retained for audits and restored below.
    for column in ("source", "record_id", "t_start", "t_end"):
        ordered[f"physical_{column}"] = ordered[column]
    step = float(protocol.get("window_seconds", 10.0))
    ordered["source"] = "participant_timeline"
    ordered["record_id"] = (
        ordered["subject_id"].astype(str) + "|participant_timeline"
    )
    ordered["t_start"] = np.arange(len(ordered), dtype=float) * step
    ordered["t_end"] = ordered["t_start"] + step
    dummy_X = np.zeros((len(ordered), 1), dtype=np.float32)
    y = ordered["target_value"].to_numpy(dtype=np.float32)

    def split(fraction: float) -> Any:
        spec = CalibrationSpec(
            method="zero_shot",
            budget_seconds=None,
            budget_fraction=float(fraction),
            split_strategy="chronological_prefix",
            fraction_allocation="global_prefix",
            purge_windows=int(protocol.get("purge_windows", 0)),
            minimum_calibration_samples=int(
                protocol.get("minimum_calibration_samples", 5)
            ),
            minimum_evaluation_samples=int(
                protocol.get("minimum_evaluation_samples", 20)
            ),
        )
        return chronological_window_partition(
            dummy_X,
            y,
            ordered,
            spec,
            window_seconds=float(protocol.get("window_seconds", 10.0)),
            max_gap_seconds=float(protocol.get("max_gap_seconds", 10.5)),
        )

    def restore(partition: Any) -> Any:
        for name in (
            "calibration_metadata", "evaluation_metadata",
            "purged_metadata", "reserved_metadata",
        ):
            metadata = getattr(partition, name)
            if metadata.empty:
                continue
            for column in ("source", "record_id", "t_start", "t_end"):
                metadata[column] = metadata[f"physical_{column}"]
            setattr(partition, name, metadata)
        return partition

    reference = restore(split(reference_budget))
    selected = restore(split(budget))
    return _use_reference_evaluation(selected, reference), reference


def validate_temporal_partition(partition: Any) -> dict[str, Any]:
    """Enforce disjoint IDs and calibration-before-evaluation per recording."""
    calibration = partition.calibration_metadata
    evaluation = partition.evaluation_metadata
    calibration_ids = set(calibration["sample_id"].astype(str))
    evaluation_ids = set(evaluation["sample_id"].astype(str))
    overlap = calibration_ids & evaluation_ids
    if overlap:
        raise RuntimeError(f"Calibration/evaluation overlap: {sorted(overlap)[:5]}")
    subjects = set(calibration.get("subject_id", pd.Series(dtype=str)).astype(str))
    subjects |= set(evaluation.get("subject_id", pd.Series(dtype=str)).astype(str))
    if len(subjects) > 1:
        raise RuntimeError(f"Cross-participant contamination: {sorted(subjects)}")

    if not calibration.empty and not evaluation.empty:
        if float(calibration["absolute_t_start"].max()) >= float(
            evaluation["absolute_t_start"].min()
        ):
            raise RuntimeError(
                "Participant calibration is not globally earlier than evaluation"
            )

    checked_records = 0
    for record_id in sorted(
        set(calibration.get("record_id", pd.Series(dtype=str)).astype(str))
        & set(evaluation.get("record_id", pd.Series(dtype=str)).astype(str))
    ):
        cal = calibration.loc[calibration["record_id"].astype(str) == record_id]
        ev = evaluation.loc[evaluation["record_id"].astype(str) == record_id]
        if float(cal["t_start"].max()) >= float(ev["t_start"].min()):
            raise RuntimeError(
                f"Calibration is not earlier than evaluation in {record_id}"
            )
        checked_records += 1
    return {
        "calibration_evaluation_overlap": 0,
        "subject_count": len(subjects),
        "records_with_temporal_check": checked_records,
        "calibration_before_evaluation": True,
    }


def build_participant_calibration_plan(
    frame: pd.DataFrame,
    *,
    pm: str,
    budgets: Sequence[float],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Build paired, fixed-evaluation participant plans for one PM."""
    required = {
        "sample_id",
        "source",
        "subject_id",
        "record_id",
        "record_group_id",
        "t_start",
        "t_end",
        "absolute_t_start",
        "outer_fold",
        "target_value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Target frame is missing columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Target frame sample_id values must be unique")
    if frame.groupby("subject_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("Each participant must belong to exactly one outer fold")
    budgets = tuple(float(value) for value in budgets)
    reference_budget = max(budgets)
    rows: list[dict[str, Any]] = []
    transform_manifests: dict[int, dict[str, Any]] = {}

    for outer_fold in range(1, int(protocol["n_outer_folds"]) + 1):
        train = frame.loc[frame["outer_fold"] != outer_fold]
        test = frame.loc[frame["outer_fold"] == outer_fold]
        train_subjects = set(train["subject_id"].astype(str))
        test_subjects = set(test["subject_id"].astype(str))
        subject_overlap = train_subjects & test_subjects
        if subject_overlap:
            raise RuntimeError(
                f"Outer subject leakage in fold {outer_fold}: {sorted(subject_overlap)}"
            )
        if train["record_group_id"].isna().any() or test[
            "record_group_id"
        ].isna().any():
            raise ValueError("record_group_id must be present for every outer-fold row")
        train_record_groups = set(train["record_group_id"].astype(str))
        test_record_groups = set(test["record_group_id"].astype(str))
        record_group_overlap = train_record_groups & test_record_groups
        if record_group_overlap:
            raise RuntimeError(
                "Outer logical-record leakage in fold "
                f"{outer_fold}: {sorted(record_group_overlap)[:20]}"
            )
        _, manifest = fit_outer_train_q3(
            pm=pm,
            outer_fold=outer_fold,
            outer_train_sample_ids=train["sample_id"].to_numpy(),
            outer_train_targets=train["target_value"].to_numpy(),
        )
        transform_manifests[outer_fold] = manifest

        for subject_id, subject in test.groupby("subject_id", sort=True):
            subject = subject.sort_values(
                ["source", "record_id", "t_start", "sample_id"],
                kind="mergesort",
            ).reset_index(drop=True)
            for budget in budgets:
                partition, reference = _participant_partition(
                    subject,
                    budget=budget,
                    reference_budget=reference_budget,
                    protocol=protocol,
                )
                audit = validate_temporal_partition(partition)
                n_calibration = len(partition.calibration_metadata)
                n_evaluation = len(partition.evaluation_metadata)
                reasons: list[str] = []
                if budget > 0 and n_calibration < int(
                    protocol.get("minimum_calibration_samples", 5)
                ):
                    reasons.append("insufficient_calibration_samples")
                if n_evaluation < int(
                    protocol.get("minimum_evaluation_samples", 20)
                ):
                    reasons.append("insufficient_evaluation_samples")
                rows.append(
                    {
                        "pm": pm,
                        "outer_fold": outer_fold,
                        "subject_id": str(subject_id),
                        "sources": "|".join(
                            sorted(subject["source"].astype(str).unique())
                        ),
                        "records": int(subject["record_id"].nunique()),
                        "record_groups": int(
                            subject["record_group_id"].nunique()
                        ),
                        "budget_fraction": budget,
                        "budget_windows": n_calibration,
                        "budget_seconds_actual": float(
                            partition.actual_seconds
                        ),
                        "total_available_windows": len(subject),
                        "evaluation_windows": n_evaluation,
                        "reserved_windows": len(partition.reserved_metadata),
                        "purged_windows": len(partition.purged_metadata),
                        "calibration_sample_hash": _sample_hash(
                            partition.calibration_metadata["sample_id"]
                        ),
                        "evaluation_sample_hash": _sample_hash(
                            partition.evaluation_metadata["sample_id"]
                        ),
                        "reference_evaluation_sample_hash": _sample_hash(
                            reference.evaluation_metadata["sample_id"]
                        ),
                        "q3_transform_hash": manifest["transform_hash"],
                        "outer_train_subject_overlap": len(subject_overlap),
                        "outer_train_record_group_overlap": len(
                            record_group_overlap
                        ),
                        **audit,
                        "status": "insufficient_data" if reasons else "planned",
                        "reason": "|".join(reasons),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["pm", "outer_fold", "subject_id", "budget_fraction"],
        kind="mergesort",
    ).reset_index(drop=True)
    duplicated = result.duplicated(
        ["pm", "outer_fold", "subject_id", "budget_fraction"]
    )
    if duplicated.any():
        raise RuntimeError("Participant plan contains duplicate conditions")
    return result, transform_manifests


def build_run_matrix(
    *,
    config: Mapping[str, Any],
    compatibility: pd.DataFrame,
    participant_plan: pd.DataFrame,
    filters: PlanFilters | None = None,
) -> pd.DataFrame:
    """Expand the deterministic fold/PM/task/model/mode/budget matrix."""
    filters = filters or PlanFilters()
    n_splits = int(config["protocol"]["n_outer_folds"])
    filters.validate(n_splits)
    budgets = [float(value) for value in config["calibration"]["budgets_fraction"]]
    rows: list[dict[str, Any]] = []
    compatibility_index = compatibility.set_index(["model", "task_type"])
    for pm in config["pms"]:
        for task_type in config["task_types"]:
            target_id = _target_id(pm, task_type)
            for model_config in config["models"]:
                model = str(model_config["model_id"])
                compatible = compatibility_index.loc[(model, task_type)]
                for outer_fold in range(1, n_splits + 1):
                    for mode in config["calibration"]["modes"]:
                        mode_budgets = [0.0] if mode == "zero_shot" else budgets[1:]
                        for budget in mode_budgets:
                            if (
                                filters.budget_fraction is not None
                                and budget != float(filters.budget_fraction)
                            ):
                                continue
                            if filters.outer_fold is not None and outer_fold != filters.outer_fold:
                                continue
                            if filters.pm is not None and pm != filters.pm:
                                continue
                            if filters.task_type is not None and task_type != filters.task_type:
                                continue
                            if filters.model is not None and model != filters.model:
                                continue
                            if (
                                filters.calibration_mode is not None
                                and mode != filters.calibration_mode
                            ):
                                continue
                            mode_supported = bool(
                                compatible[f"{mode}_supported"]
                            )
                            participants = participant_plan.loc[
                                participant_plan["pm"].eq(pm)
                                & participant_plan["outer_fold"].eq(outer_fold)
                                & participant_plan["budget_fraction"].eq(budget)
                            ]
                            sufficient = int(
                                participants["status"].eq("planned").sum()
                            )
                            condition_key = {
                                "pm": pm,
                                "task_type": task_type,
                                "target_id": target_id,
                                "model": model,
                                "outer_fold": outer_fold,
                                "mode": mode,
                                "budget_fraction": budget,
                                "seed": int(config["experiment"]["random_state"]),
                            }
                            reason = ""
                            status = "planned"
                            if not mode_supported:
                                status = "unsupported"
                                reason = str(compatible["reason"] or f"{mode} unavailable")
                            elif sufficient == 0:
                                status = "insufficient_data"
                                reason = "no participant satisfies minimum cohort sizes"
                            rows.append(
                                {
                                    **condition_key,
                                    "condition_id": stable_hash(condition_key)[:20],
                                    "input_family": model_config["input_family"],
                                    "participants_total": int(len(participants)),
                                    "participants_sufficient": sufficient,
                                    "participants_insufficient": int(
                                        len(participants) - sufficient
                                    ),
                                    "participant_execution_count": (
                                        sufficient if mode_supported else 0
                                    ),
                                    "q3_transform_hash": (
                                        ""
                                        if task_type == "regression" or participants.empty
                                        else str(
                                            participants["q3_transform_hash"].iloc[0]
                                        )
                                    ),
                                    "status": status,
                                    "reason": reason,
                                }
                            )
    result = pd.DataFrame(rows).sort_values(
        [
            "pm",
            "task_type",
            "model",
            "outer_fold",
            "mode",
            "budget_fraction",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    if result["condition_id"].duplicated().any():
        raise RuntimeError("Run matrix condition_id values are not unique")
    return result


def aggregate_participant_metrics(
    participant_metrics: pd.DataFrame,
    *,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute equal-participant macro means and deltas versus zero-shot."""
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unknown task_type {task_type!r}")
    metrics = (
        CLASSIFICATION_METRICS
        if task_type == "classification"
        else REGRESSION_METRICS
    )
    keys = ["pm", "model", "outer_fold", "subject_id"]
    required = set(keys) | {"mode", "budget_fraction", *metrics}
    missing = sorted(required - set(participant_metrics.columns))
    if missing:
        raise ValueError(f"Participant metrics are missing columns: {missing}")
    frame = participant_metrics.copy()
    baseline = frame.loc[frame["mode"].eq("zero_shot")].copy()
    if baseline.duplicated(keys).any():
        raise ValueError("Zero-shot baseline must be unique per participant")
    baseline = baseline.set_index(keys)
    for metric in metrics:
        lookup = baseline[metric]
        frame[f"zero_shot_{metric}"] = [
            lookup.get(tuple(row[key] for key in keys), np.nan)
            for _, row in frame.iterrows()
        ]
        frame[f"delta_{metric}_vs_zero_shot"] = (
            pd.to_numeric(frame[metric], errors="coerce")
            - pd.to_numeric(frame[f"zero_shot_{metric}"], errors="coerce")
        )
    group_keys = ["pm", "model", "mode", "budget_fraction"]
    aggregations: dict[str, tuple[str, str]] = {
        "participants": ("subject_id", "nunique")
    }
    for metric in metrics:
        aggregations[f"{metric}_participant_macro"] = (metric, "mean")
        aggregations[f"delta_{metric}_vs_zero_shot_participant_macro"] = (
            f"delta_{metric}_vs_zero_shot",
            "mean",
        )
    summary = frame.groupby(group_keys, sort=True).agg(**aggregations).reset_index()
    return frame, summary


class PersonalizationCalibrationPlanner:
    """Materialize the full protocol without reading raw tensors or training."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        data_root: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        document = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.config = validate_protocol_config(document)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_root = (
            self.repo_root if data_root is None else Path(data_root).resolve()
        )
        configured_output = self.config["experiment"]["output_dir"]
        self.output_dir = (
            _resolve_path(self.repo_root, configured_output)
            if output_dir is None
            else Path(output_dir).resolve()
        )

    def _load_target_frame(self, pm: str) -> pd.DataFrame:
        data_config = self.config["data"]
        dataset_config = {
            "data_path": str(_resolve_path(self.data_root, data_config["raw_manifest"])),
            "cache_path_root": str(self.data_root),
            "target_data_path": str(
                _resolve_path(self.data_root, data_config["processed_targets"])
            ),
            "target_id": _target_id(pm, "regression"),
            "dataset_mode": data_config["dataset_mode"],
            "logical_recording_map_path": str(
                _resolve_path(self.data_root, data_config["logical_recording_map"])
            ),
            "raw_preprocessing": data_config["raw_preprocessing"],
        }
        dataset = get_dataset("emotiv_raw_eeg", dataset_config)
        data = dataset.load()
        frame = pd.DataFrame(
            {
                "sample_id": data.sample_ids,
                "source": data.get_row_values("source"),
                "subject_id": data.subject_ids,
                "record_id": data.record_ids,
                "record_group_id": data.get_row_values("record_group_id"),
                "t_start": data.get_row_values("t_start"),
                "t_end": data.get_row_values("t_end"),
                "outer_fold": data.get_row_values("outer_fold"),
                "target_value": data.labels,
            }
        )
        raw_manifest = getattr(data.data, "manifest", None)
        if raw_manifest is None or "absolute_t_start" not in raw_manifest:
            raise RuntimeError(
                "Canonical raw view must expose absolute_t_start for temporal planning"
            )
        manifest_ids = raw_manifest["sample_id"].astype(str).to_numpy()
        if not np.array_equal(manifest_ids, frame["sample_id"].astype(str).to_numpy()):
            raise RuntimeError("Raw view manifest order does not match EEGData sample IDs")
        frame["absolute_t_start"] = raw_manifest["absolute_t_start"].to_numpy()
        if not np.isfinite(frame["target_value"].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite target values remain for PM {pm}")
        return frame

    @property
    def protocol_hash(self) -> str:
        # Execution placement/hyperparameters are independently bound into each
        # base checkpoint identity. Keeping them outside the already frozen
        # split/target protocol hash preserves the audited v1 protocol identity.
        protocol = deepcopy(self.config)
        protocol.pop("execution", None)
        return stable_hash(protocol)

    def materialize_tables(
        self,
        *,
        filters: PlanFilters | None = None,
    ) -> dict[str, Any]:
        """Resolve deterministic tables used by plan, dry-run and execution."""
        filters = filters or PlanFilters()
        filters.validate(int(self.config["protocol"]["n_outer_folds"]))
        compatibility = build_model_compatibility(self.config["models"])
        participant_frames: list[pd.DataFrame] = []
        transforms: dict[str, dict[int, dict[str, Any]]] = {}
        cohorts: dict[str, pd.DataFrame] = {}
        for pm in self.config["pms"]:
            target_frame = self._load_target_frame(pm)
            cohorts[pm] = target_frame.copy()
            participant_plan, manifests = build_participant_calibration_plan(
                target_frame,
                pm=pm,
                budgets=self.config["calibration"]["budgets_fraction"],
                protocol=self.config["protocol"],
            )
            participant_frames.append(participant_plan)
            transforms[pm] = manifests
        participants = pd.concat(participant_frames, ignore_index=True)
        run_matrix = build_run_matrix(
            config=self.config,
            compatibility=compatibility,
            participant_plan=participants,
            filters=filters,
        )
        return {
            "filters": filters,
            "compatibility": compatibility,
            "participants": participants,
            "run_matrix": run_matrix,
            "transforms": transforms,
            "cohorts": cohorts,
        }

    def plan(
        self,
        *,
        filters: PlanFilters | None = None,
        resume: bool = False,
        write_artifacts: bool = True,
    ) -> dict[str, Any]:
        tables = self.materialize_tables(filters=filters)
        filters = tables["filters"]
        compatibility = tables["compatibility"]
        participants = tables["participants"]
        run_matrix = tables["run_matrix"]
        transforms = tables["transforms"]
        filter_payload = {
            key: value for key, value in asdict(filters).items()
            if key not in {"model", "budget_fraction"} or value is not None
        }
        plan_hash = stable_hash({
            "protocol_hash": self.protocol_hash,
            "filters": filter_payload,
        })
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment"]["experiment_id"],
            "result_status": "planned",
            "training_executed": False,
            "protocol_hash": self.protocol_hash,
            "plan_hash": plan_hash,
            "filters": filter_payload,
            "config_path": str(self.config_path),
            "n_outer_folds": int(self.config["protocol"]["n_outer_folds"]),
            "pms": list(self.config["pms"]),
            "task_types": list(self.config["task_types"]),
            "models": [item["model_id"] for item in self.config["models"]],
            "modes": list(self.config["calibration"]["modes"]),
            "budgets_fraction": list(
                self.config["calibration"]["budgets_fraction"]
            ),
            "run_conditions": int(len(run_matrix)),
            "planned_conditions": int(run_matrix["status"].eq("planned").sum()),
            "unsupported_conditions": int(
                run_matrix["status"].eq("unsupported").sum()
            ),
            "insufficient_conditions": int(
                run_matrix["status"].eq("insufficient_data").sum()
            ),
            "participant_execution_count": int(
                run_matrix["participant_execution_count"].sum()
            ),
            "participant_plan_rows": int(len(participants)),
            "unique_participants": int(participants["subject_id"].nunique()),
            "leakage_checks": {
                "outer_subject_overlap_max": int(
                    participants["outer_train_subject_overlap"].max()
                ),
                "outer_record_group_overlap_max": int(
                    participants["outer_train_record_group_overlap"].max()
                ),
                "calibration_evaluation_overlap_max": int(
                    participants["calibration_evaluation_overlap"].max()
                ),
                "all_calibration_before_evaluation": bool(
                    participants["calibration_before_evaluation"].all()
                ),
                "q3_fit_scope": "outer_train_only",
                "fixed_evaluation_across_budgets": bool(
                    participants.groupby(["pm", "outer_fold", "subject_id"])[
                        "evaluation_sample_hash"
                    ].nunique().max()
                    == 1
                ),
            },
            "formal_criteria": {
                "classification_accuracy_threshold": float(
                    self.config["analysis"]["formal_accuracy_threshold"]
                ),
                "aggregation": self.config["analysis"]["aggregation"],
                "threshold_role": self.config["analysis"]["threshold_role"],
            },
            "resume_contract": {
                "key": "condition_id",
                "required_protocol_hash": self.protocol_hash,
                "required_plan_hash": plan_hash,
                "completed_condition_is_immutable": True,
                "incompatible_hash_fails": True,
            },
        }
        if write_artifacts:
            self._write_plan(
                compatibility=compatibility,
                participants=participants,
                run_matrix=run_matrix,
                transforms=transforms,
                cohorts=tables["cohorts"],
                manifest=manifest,
                resume=resume,
            )
        return manifest

    def _write_plan(
        self,
        *,
        compatibility: pd.DataFrame,
        participants: pd.DataFrame,
        run_matrix: pd.DataFrame,
        transforms: Mapping[str, Mapping[int, Mapping[str, Any]]],
        cohorts: Mapping[str, pd.DataFrame],
        manifest: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "protocol_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("protocol_hash") != self.protocol_hash:
                raise RuntimeError("Resume protocol hash mismatch")
            if (
                existing.get("plan_hash") is not None
                and existing.get("plan_hash") != manifest.get("plan_hash")
            ):
                raise RuntimeError("Resume filter/plan hash mismatch")
            if not resume:
                raise FileExistsError(
                    "Plan artifacts already exist; pass --resume for an idempotent "
                    "same-protocol rewrite"
                )
        (self.output_dir / "resolved_protocol.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        compatibility.to_csv(self.output_dir / "model_compatibility.csv", index=False)
        participants.to_csv(
            self.output_dir / "participant_calibration_plan.csv", index=False
        )
        cohort_root = self.output_dir / "cohorts"
        cohort_root.mkdir(parents=True, exist_ok=True)
        for pm, cohort in cohorts.items():
            cohort.loc[:, [
                "sample_id", "source", "subject_id", "record_id",
                "record_group_id", "t_start", "t_end", "absolute_t_start",
                "outer_fold",
            ]].sort_values("sample_id", kind="mergesort").to_parquet(
                cohort_root / f"{pm}.parquet", index=False
            )
        run_matrix.to_csv(self.output_dir / "run_matrix.csv", index=False)
        transform_root = self.output_dir / "target_transforms"
        for pm, fold_manifests in transforms.items():
            pm_root = transform_root / pm
            pm_root.mkdir(parents=True, exist_ok=True)
            for fold, transform_manifest in fold_manifests.items():
                (pm_root / f"fold_{fold:02d}.json").write_text(
                    json.dumps(transform_manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
        manifest_path.write_text(
            json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "README.md").write_text(
            "# Personalization calibration v1 (plan only)\n\n"
            "This directory contains a deterministic protocol plan. No model "
            "training or checkpoint creation occurred. Q3 boundaries are fitted "
            "only on each outer-train fold. Every participant uses one strict "
            "absolute-time chronological prefix and one fixed late evaluation "
            "suffix shared by all budgets and modes.\n\n"
            "Runtime output is not a scientific result until the complete planned "
            "execution and participant-level aggregation are finished.\n",
            encoding="utf-8",
        )
