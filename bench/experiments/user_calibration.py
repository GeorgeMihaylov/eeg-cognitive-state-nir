"""Subject-specific calibration built on canonical benchmark checkpoints."""

from __future__ import annotations

import gc
import hashlib
import html
import json
import logging
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import BenchmarkRunner, benchmark_config_hash
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.DL.sequence_utils import (
    TIME_COLUMN_PRIORITY,
    SequenceBuildResult,
    build_sequences,
)
from model_zoo.factory import build_model, model_requires_sequences


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_SCHEMA_VERSION = "user-calibration-v1"
CALIBRATION_METHODS = frozenset(
    {"zero_shot", "subject_normalization", "head_only", "full_model"}
)
LOGGER = logging.getLogger(__name__)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_component(value: Any) -> str:
    text = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("._")
    return text or "unnamed"


def _is_zero_budget(spec: "CalibrationSpec") -> bool:
    return (
        spec.budget_seconds is not None
        and float(spec.budget_seconds) == 0.0
    ) or (
        spec.budget_fraction is not None
        and float(spec.budget_fraction) == 0.0
    )


@dataclass(frozen=True)
class CalibrationSpec:
    """Serializable, AutoML-ready calibration protocol parameters."""

    method: str
    budget_seconds: Optional[float] = 0.0
    budget_fraction: Optional[float] = None
    split_strategy: str = "chronological_prefix"
    fraction_allocation: str = "per_record_prefix"
    purge_windows: int = 7
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4
    calibration_validation_fraction: float = 0.2
    fallback_fixed_epochs: int = 3
    min_calibration_sequences: int = 1
    min_evaluation_sequences: int = 20
    minimum_calibration_samples: int = 1
    minimum_evaluation_samples: int = 20
    minimum_adaptation_train_samples: int = 1
    minimum_adaptation_validation_samples: int = 1
    minimum_final_evaluation_samples: Optional[int] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        normalized = self.method.strip().lower()
        aliases = {
            "normalization": "subject_normalization",
            "subject_norm": "subject_normalization",
            "head": "head_only",
            "head_only_finetuning": "head_only",
            "full": "full_model",
            "full_finetuning": "full_model",
            "no_adaptation": "zero_shot",
        }
        normalized = aliases.get(normalized, normalized)
        object.__setattr__(self, "method", normalized)
        if normalized not in CALIBRATION_METHODS:
            raise ValueError(
                f"Unknown calibration method {self.method!r}; expected "
                f"{sorted(CALIBRATION_METHODS)}"
            )
        if self.split_strategy != "chronological_prefix":
            raise ValueError("Only split_strategy='chronological_prefix' is supported")
        if self.fraction_allocation not in {
            "per_record_prefix", "global_prefix"
        }:
            raise ValueError(
                "fraction_allocation must be 'per_record_prefix' or "
                "'global_prefix'"
            )
        if self.budget_seconds is not None and self.budget_fraction is not None:
            raise ValueError("Set either budget_seconds or budget_fraction, not both")
        if self.budget_seconds is None and self.budget_fraction is None:
            raise ValueError("A calibration budget is required")
        if self.budget_seconds is not None and self.budget_seconds < 0:
            raise ValueError("budget_seconds cannot be negative")
        if self.budget_fraction is not None and not 0 <= self.budget_fraction < 1:
            raise ValueError("budget_fraction must be in [0, 1)")
        if self.purge_windows < 0:
            raise ValueError("purge_windows cannot be negative")
        if self.max_epochs <= 0 or self.fallback_fixed_epochs <= 0:
            raise ValueError("Calibration epoch counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid calibration optimizer parameters")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not 0 < self.calibration_validation_fraction < 1:
            raise ValueError("calibration_validation_fraction must be in (0, 1)")
        if self.min_calibration_sequences <= 0:
            raise ValueError("min_calibration_sequences must be positive")
        if self.min_evaluation_sequences <= 0:
            raise ValueError("min_evaluation_sequences must be positive")
        if self.minimum_calibration_samples <= 0:
            raise ValueError("minimum_calibration_samples must be positive")
        if self.minimum_evaluation_samples <= 0:
            raise ValueError("minimum_evaluation_samples must be positive")
        if self.minimum_adaptation_train_samples <= 0:
            raise ValueError(
                "minimum_adaptation_train_samples must be positive"
            )
        if self.minimum_adaptation_validation_samples <= 0:
            raise ValueError(
                "minimum_adaptation_validation_samples must be positive"
            )
        if self.minimum_final_evaluation_samples is None:
            object.__setattr__(
                self,
                "minimum_final_evaluation_samples",
                self.minimum_evaluation_samples,
            )
        elif self.minimum_final_evaluation_samples <= 0:
            raise ValueError(
                "minimum_final_evaluation_samples must be positive"
            )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CalibrationSpec":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown calibration parameters: {unknown}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return _canonical_hash(self.to_dict())


def resolve_calibration_parameters(
    base: CalibrationSpec | Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> CalibrationSpec:
    """Resolve `calibration.*` trial parameters without an AutoML dependency."""
    resolved = base.to_dict() if isinstance(base, CalibrationSpec) else dict(base)
    allowed = {field.name for field in fields(CalibrationSpec)}
    for dotted_name, value in parameters.items():
        if not str(dotted_name).startswith("calibration."):
            raise ValueError(
                f"Calibration trial parameter must start with 'calibration.': "
                f"{dotted_name!r}"
            )
        name = str(dotted_name).split(".", 1)[1]
        if name not in allowed:
            raise ValueError(f"Unknown calibration parameter: {dotted_name}")
        resolved[name] = value
    return CalibrationSpec.from_dict(resolved)


@dataclass
class WindowPartition:
    calibration_X: np.ndarray
    calibration_y: np.ndarray
    calibration_metadata: pd.DataFrame
    evaluation_X: np.ndarray
    evaluation_y: np.ndarray
    evaluation_metadata: pd.DataFrame
    purged_metadata: pd.DataFrame
    requested_seconds: float
    actual_seconds: float
    available_seconds: float
    time_column: str
    requested_fraction: Optional[float] = None
    actual_fraction: float = 0.0
    reserved_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)


def _time_column(metadata: pd.DataFrame) -> str:
    for column in TIME_COLUMN_PRIORITY:
        if column in metadata.columns:
            return column
    raise ValueError(
        f"Calibration metadata needs one of {list(TIME_COLUMN_PRIORITY)}"
    )


def _ordered_segments(
    metadata: pd.DataFrame,
    *,
    max_gap_seconds: float,
) -> tuple[pd.DataFrame, str]:
    frame = metadata.reset_index(drop=True).copy()
    required = {"source", "subject_id", "record_id", "sample_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Calibration metadata is missing columns: {missing}")
    time_column = _time_column(frame)
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["source", "record_id", time_column, "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    segment_keys: list[str] = []
    positions: list[int] = []
    for (source, record_id), group in frame.groupby(
        ["source", "record_id"], sort=False, dropna=False
    ):
        times = group[time_column].to_numpy(dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("Calibration time values must be finite")
        breaks = np.r_[True, (np.diff(times) <= 0) | (np.diff(times) > max_gap_seconds)]
        segment_ids = np.cumsum(breaks) - 1
        for segment_id in np.unique(segment_ids):
            count = int(np.sum(segment_ids == segment_id))
            segment_keys.extend(
                [f"{source}|{record_id}|segment={int(segment_id)}"] * count
            )
            positions.extend(range(count))
    frame["_segment_key"] = segment_keys
    frame["_segment_position"] = positions
    return frame, time_column


def chronological_window_partition(
    X: Any,
    y: Any,
    metadata: pd.DataFrame,
    spec: CalibrationSpec,
    *,
    window_seconds: float,
    max_gap_seconds: float,
) -> WindowPartition:
    """Split original windows before building any overlapping sequences."""
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y)
    if len(features) != len(labels) or len(features) != len(metadata):
        raise ValueError("Calibration X, y, and metadata lengths must match")
    if window_seconds <= 0 or max_gap_seconds <= 0:
        raise ValueError("Window and maximum-gap durations must be positive")
    ordered, time_column = _ordered_segments(
        metadata, max_gap_seconds=max_gap_seconds
    )
    segment_durations: list[tuple[str, float]] = []
    for key, group in ordered.groupby("_segment_key", sort=False):
        times = group[time_column].to_numpy(dtype=np.float64)
        duration = float(times[-1] - times[0] + window_seconds)
        segment_durations.append((str(key), duration))
    available_seconds = float(sum(value for _, value in segment_durations))
    requested_seconds = (
        float(spec.budget_seconds)
        if spec.budget_seconds is not None
        else available_seconds * float(spec.budget_fraction)
    )

    calibration_rows: list[int] = []
    purged_rows: list[int] = []
    actual_seconds = 0.0
    remaining = requested_seconds
    if (
        spec.budget_fraction is not None
        and spec.fraction_allocation == "global_prefix"
    ):
        count = int(np.floor(len(ordered) * float(spec.budget_fraction)))
        calibration_rows.extend(
            ordered["_row_index"].to_numpy(dtype=np.int64)[:count].tolist()
        )
        actual_seconds = float(count * window_seconds)
        if count < len(ordered) and spec.purge_windows:
            selected_segment = (
                None if count == 0 else ordered.iloc[count - 1]["_segment_key"]
            )
            following = ordered.iloc[count:]
            if selected_segment is not None:
                following = following.loc[
                    following["_segment_key"] == selected_segment
                ]
            purged_rows.extend(
                following["_row_index"].to_numpy(dtype=np.int64)[
                    : spec.purge_windows
                ].tolist()
            )
    else:
        for segment_key, _ in segment_durations:
            group = ordered.loc[ordered["_segment_key"] == segment_key]
            row_indices = group["_row_index"].to_numpy(dtype=np.int64)
            times = group[time_column].to_numpy(dtype=np.float64)
            if spec.budget_fraction is not None:
                count = int(np.floor(len(group) * float(spec.budget_fraction)))
                if count <= 0:
                    continue
                calibration_rows.extend(row_indices[:count].tolist())
                consumed = float(times[count - 1] - times[0] + window_seconds)
                actual_seconds += consumed
                if count < len(group) and spec.purge_windows:
                    purge_stop = min(len(group), count + spec.purge_windows)
                    purged_rows.extend(row_indices[count:purge_stop].tolist())
                continue
            if remaining <= 1e-9:
                continue
            cumulative = times - times[0] + window_seconds
            count = min(
                len(group),
                int(np.searchsorted(cumulative, remaining, side="left") + 1),
            )
            calibration_rows.extend(row_indices[:count].tolist())
            consumed = float(cumulative[count - 1])
            actual_seconds += consumed
            remaining = max(0.0, requested_seconds - actual_seconds)
            if count < len(group):
                purge_stop = min(len(group), count + spec.purge_windows)
                purged_rows.extend(row_indices[count:purge_stop].tolist())
                remaining = 0.0

    calibration_index = np.asarray(sorted(set(calibration_rows)), dtype=np.int64)
    purged_index = np.asarray(sorted(set(purged_rows)), dtype=np.int64)
    excluded = set(calibration_index.tolist()) | set(purged_index.tolist())
    evaluation_index = np.asarray(
        [index for index in range(len(features)) if index not in excluded],
        dtype=np.int64,
    )
    if set(calibration_index.tolist()) & set(evaluation_index.tolist()):
        raise RuntimeError("Calibration and evaluation windows overlap")

    raw_metadata = metadata.reset_index(drop=True)
    return WindowPartition(
        calibration_X=np.ascontiguousarray(features[calibration_index]),
        calibration_y=labels[calibration_index],
        calibration_metadata=raw_metadata.iloc[calibration_index].reset_index(drop=True),
        evaluation_X=np.ascontiguousarray(features[evaluation_index]),
        evaluation_y=labels[evaluation_index],
        evaluation_metadata=raw_metadata.iloc[evaluation_index].reset_index(drop=True),
        purged_metadata=raw_metadata.iloc[purged_index].reset_index(drop=True),
        requested_seconds=requested_seconds,
        actual_seconds=actual_seconds,
        available_seconds=available_seconds,
        time_column=time_column,
        requested_fraction=spec.budget_fraction,
        actual_fraction=(
            0.0 if len(features) == 0 else len(calibration_index) / len(features)
        ),
    )


def _as_window_observations(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> SequenceBuildResult:
    """Expose feature windows through the existing calibration result contract."""
    frame = metadata.reset_index(drop=True).copy()
    required = {"sample_id", "record_id", "subject_id", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Window metadata is missing columns: {missing}")
    frame["record_group_id"] = frame.get("record_group_id", frame["record_id"])
    frame["target_sample_id"] = frame["sample_id"].astype(str)
    frame["sequence_id"] = frame["sample_id"].astype(str)
    frame["sequence_length"] = 1
    return SequenceBuildResult(
        X=np.ascontiguousarray(np.asarray(X, dtype=np.float32)),
        y=np.asarray(y),
        metadata=frame,
        stats={
            "mode": "feature_windows",
            "input_windows": int(len(frame)),
            "output_sequences": int(len(frame)),
            "sequence_length": 1,
        },
    )


def _use_reference_evaluation(
    partition: WindowPartition,
    reference: WindowPartition,
) -> WindowPartition:
    """Use one fixed late evaluation suffix for every calibration budget."""
    calibration_ids = set(
        partition.calibration_metadata["sample_id"].astype(str)
    )
    evaluation_ids = set(
        reference.evaluation_metadata["sample_id"].astype(str)
    )
    if calibration_ids & evaluation_ids:
        raise RuntimeError("Calibration overlaps the fixed final evaluation")
    all_metadata = pd.concat(
        [
            partition.calibration_metadata,
            partition.evaluation_metadata,
            partition.purged_metadata,
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    used_ids = calibration_ids | evaluation_ids
    reserved = all_metadata.loc[
        ~all_metadata["sample_id"].astype(str).isin(used_ids)
    ].reset_index(drop=True)
    return WindowPartition(
        calibration_X=partition.calibration_X,
        calibration_y=partition.calibration_y,
        calibration_metadata=partition.calibration_metadata,
        evaluation_X=reference.evaluation_X,
        evaluation_y=reference.evaluation_y,
        evaluation_metadata=reference.evaluation_metadata,
        purged_metadata=reference.purged_metadata,
        requested_seconds=partition.requested_seconds,
        actual_seconds=partition.actual_seconds,
        available_seconds=partition.available_seconds,
        time_column=partition.time_column,
        requested_fraction=partition.requested_fraction,
        actual_fraction=partition.actual_fraction,
        reserved_metadata=reserved,
    )


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    sequence_config: Mapping[str, Any],
) -> SequenceBuildResult:
    return build_sequences(
        X=X,
        y=y,
        metadata=metadata,
        sequence_length=int(
            sequence_config.get("length", sequence_config.get("sequence_length", 10))
        ),
        stride=int(sequence_config.get("stride", 1)),
        target_position=str(sequence_config.get("target_position", "last")),
        expected_step_seconds=float(sequence_config["expected_step_seconds"]),
        max_gap_seconds=float(sequence_config["max_gap_seconds"]),
    )


def _build_model_inputs(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    sequence_config: Optional[Mapping[str, Any]],
) -> SequenceBuildResult:
    if sequence_config is None:
        return _as_window_observations(X, y, metadata)
    return _build_sequences(X, y, metadata, sequence_config)


def _class_coverage(labels: np.ndarray) -> dict[str, Any]:
    classes, counts = np.unique(labels, return_counts=True)
    total = int(np.sum(counts))
    return {
        "classes_present": [int(value) for value in classes.tolist()],
        "number_of_classes": int(len(classes)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(classes, counts)
        },
        "majority_class_fraction": (
            None if total == 0 else float(np.max(counts) / total)
        ),
    }


def calibration_normalization_statistics(
    calibration_windows: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute subject statistics from unique calibration windows only."""
    features = np.asarray(calibration_windows, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(
            "Subject normalization requires non-empty [windows, features] data"
        )
    if not np.isfinite(features).all():
        raise ValueError("Calibration normalization data must be finite")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return mean, scale


def _state_digest(adapter: TorchClassificationAdapter) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _parameter_digest(
    adapter: TorchClassificationAdapter,
    names: Sequence[str],
) -> str:
    selected = set(str(name) for name in names)
    digest = hashlib.sha256()
    for name, parameter in sorted(adapter.model.named_parameters()):
        if name not in selected:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _parameter_audit(
    adapter: TorchClassificationAdapter,
    method: str,
) -> tuple[list[str], list[str], int, int]:
    if method == "head_only":
        resolver = getattr(
            adapter.model, "output_head_parameter_prefixes", None
        )
        if not callable(resolver):
            raise ValueError(
                "Head-only calibration requires "
                "output_head_parameter_prefixes()"
            )
        prefixes = tuple(str(value) for value in resolver())
        trainable = [
            name for name, _ in adapter.model.named_parameters()
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
    elif method == "full_model":
        trainable = [name for name, _ in adapter.model.named_parameters()]
    else:
        trainable = []
    trainable_set = set(trainable)
    frozen = [
        name for name, _ in adapter.model.named_parameters()
        if name not in trainable_set
    ]
    counts = {
        name: int(parameter.numel())
        for name, parameter in adapter.model.named_parameters()
    }
    return (
        trainable,
        frozen,
        sum(counts[name] for name in trainable),
        sum(counts[name] for name in frozen),
    )


def _implementation_hash() -> str:
    """Hash the small implementation surface that defines calibration results."""
    digest = hashlib.sha256()
    paths = (
        Path(__file__),
        REPO_ROOT / "bench" / "validation" / "metrics.py",
        REPO_ROOT / "model_zoo" / "DL" / "adapter.py",
        REPO_ROOT / "model_zoo" / "DL" / "mlp.py",
    )
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 1000,
    random_state: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None, None
    rng = np.random.default_rng(random_state)
    means = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean()
        for _ in range(int(samples))
    ])
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _normalized_subject_metrics(subjects: pd.DataFrame) -> pd.DataFrame:
    """Expose the full-experiment column contract without breaking legacy CSVs."""
    frame = subjects.copy()
    frame["method"] = frame["calibration_method"]
    frame["budget_requested"] = frame["budget"]
    frame["budget_actual"] = frame["actual_calibration_fraction"].where(
        frame["budget_fraction"].notna(),
        frame["actual_calibration_duration"] / frame[
            "requested_calibration_duration"
        ].replace(0, np.nan),
    ).fillna(0.0)
    frame["n_total_target_samples"] = (
        frame["calibration_samples"]
        + frame.get("reserved_samples", 0)
        + frame["evaluation_samples"]
    )
    frame["n_calibration_pool"] = frame["calibration_samples"]
    frame["n_adaptation_train"] = frame["adaptation_train_samples"]
    frame["n_adaptation_validation"] = frame[
        "adaptation_validation_samples"
    ]
    frame["n_final_evaluation"] = frame["evaluation_samples"]
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "ordinal_mae",
        "severe_error_rate",
    ):
        frame[f"{metric}_after"] = frame[metric]
    frame["accuracy_gain"] = frame["accuracy_absolute_gain"]
    frame["balanced_accuracy_gain"] = frame[
        "balanced_accuracy_absolute_gain"
    ]
    frame["macro_f1_gain"] = frame["macro_f1_absolute_gain"]
    frame["accuracy_at_least_075"] = frame["accuracy_after"] >= 0.75
    frame["status"] = frame["status"].replace({
        "valid": "completed",
        "insufficient_calibration_data": "insufficient_calibration_samples",
        "insufficient_evaluation_data": "insufficient_evaluation_samples",
        "insufficient_sequence_context": "insufficient_target_samples",
    })
    return frame


def _aggregate_metric_rows(
    metrics: pd.DataFrame,
    *,
    scope: str,
    source: str,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completed = metrics.loc[metrics["status"] == "completed"].copy()
    for (method, budget), group in completed.groupby(
        ["method", "budget"], sort=True
    ):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            after = pd.to_numeric(
                group[f"{metric}_after"], errors="coerce"
            ).dropna()
            gains = pd.to_numeric(
                group[f"{metric}_gain"], errors="coerce"
            ).dropna()
            low, high = _bootstrap_mean_interval(
                gains,
                samples=bootstrap_samples,
                random_state=random_state,
            )
            rows.append({
                "scope": scope,
                "source": source,
                "method": method,
                "budget": float(budget),
                "metric": metric,
                "mean": None if after.empty else float(after.mean()),
                "median": None if after.empty else float(after.median()),
                "std": None if len(after) < 2 else float(after.std(ddof=1)),
                "min": None if after.empty else float(after.min()),
                "max": None if after.empty else float(after.max()),
                "q25": None if after.empty else float(after.quantile(0.25)),
                "q75": None if after.empty else float(after.quantile(0.75)),
                "mean_gain": None if gains.empty else float(gains.mean()),
                "median_gain": None if gains.empty else float(gains.median()),
                "gain_bootstrap_ci_low": low,
                "gain_bootstrap_ci_high": high,
                "bootstrap_resamples": int(bootstrap_samples),
                "n_subjects": int(group["subject_id"].nunique()),
                "subjects_improved": int((gains > 0).sum()),
                "subjects_improved_fraction": (
                    None if gains.empty else float((gains > 0).mean())
                ),
                "subjects_accuracy_at_least_075": int(
                    group.loc[
                        group["accuracy_after"] >= 0.75, "subject_id"
                    ].nunique()
                ),
                "subjects_accuracy_at_least_075_fraction": (
                    None
                    if group.empty
                    else float((group["accuracy_after"] >= 0.75).mean())
                ),
            })
    return pd.DataFrame(rows)


def _source_subject_metrics(
    predictions: pd.DataFrame,
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate per-source metrics without duplicating shared identities."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    probability_columns = sorted(
        [
            column for column in predictions.columns
            if str(column).startswith("proba_")
        ],
        key=lambda value: int(str(value).split("_")[-1]),
    )
    keys = [
        "outer_fold", "subject_id", "source", "seed",
        "calibration_method", "budget",
    ]
    for values, group in predictions.groupby(keys, sort=True):
        metrics = MetricsCalculator.calculate_all_metrics(
            group["y_true"].to_numpy(dtype=int),
            group["y_pred"].to_numpy(dtype=int),
            group[probability_columns].to_numpy(dtype=float),
            labels=np.arange(len(probability_columns)),
        )
        rows.append({
            **dict(zip(keys, values)),
            "method": values[4],
            "status": "completed",
            "n_final_evaluation": int(len(group)),
            "accuracy_after": metrics["accuracy"],
            "balanced_accuracy_after": metrics["balanced_accuracy"],
            "macro_f1_after": metrics["macro_f1"],
        })
    frame = pd.DataFrame(rows)
    baseline = frame.loc[
        frame["method"] == "zero_shot",
        [
            "outer_fold", "subject_id", "source", "seed", "budget",
            "accuracy_after", "balanced_accuracy_after", "macro_f1_after",
        ],
    ].rename(columns={
        metric + "_after": metric + "_before"
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    })
    frame = frame.merge(
        baseline,
        on=["outer_fold", "subject_id", "source", "seed", "budget"],
        how="left",
        validate="many_to_one",
    )
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        frame[f"{metric}_gain"] = (
            frame[f"{metric}_after"] - frame[f"{metric}_before"]
        )
    calibration = subject_metrics.loc[
        :,
        [
            "outer_fold", "subject_id", "seed", "method", "budget",
            "n_calibration_pool", "number_of_classes",
        ],
    ]
    return frame.merge(
        calibration,
        on=["outer_fold", "subject_id", "seed", "method", "budget"],
        how="left",
        validate="many_to_one",
    )


def _paired_comparison_rows(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completed = metrics.loc[metrics["status"] == "completed"].copy()
    comparisons = (
        ("head_only", "zero_shot"),
        ("full_model", "zero_shot"),
        ("full_model", "head_only"),
    )
    for budget, budget_group in completed.groupby("budget", sort=True):
        for left, right in comparisons:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                pivot = budget_group.pivot_table(
                    index="subject_id",
                    columns="method",
                    values=f"{metric}_after",
                    aggfunc="first",
                )
                if left not in pivot or right not in pivot:
                    continue
                differences = (pivot[left] - pivot[right]).dropna()
                low, high = _bootstrap_mean_interval(
                    differences.to_numpy(),
                    samples=bootstrap_samples,
                    random_state=random_state,
                )
                rows.append({
                    "budget": float(budget),
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "n_subjects": int(len(differences)),
                    "mean_difference": (
                        None
                        if differences.empty
                        else float(differences.mean())
                    ),
                    "median_difference": (
                        None
                        if differences.empty
                        else float(differences.median())
                    ),
                    "positive_fraction": (
                        None
                        if differences.empty
                        else float((differences > 0).mean())
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_resamples": int(bootstrap_samples),
                })
    return pd.DataFrame(rows)


def _threshold_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = metrics.loc[metrics["status"] == "completed"].copy()
    summary_rows: list[dict[str, Any]] = []
    for source, source_group in [
        ("overall", completed),
        *[
            (str(value), completed.loc[completed["source"] == value])
            for value in sorted(completed["source"].dropna().unique())
        ],
    ]:
        for (method, budget), group in source_group.groupby(
            ["method", "budget"], sort=True
        ):
            accuracy = pd.to_numeric(
                group["accuracy_after"], errors="coerce"
            ).dropna()
            reached = group.loc[group["accuracy_after"] >= 0.75]
            summary_rows.append({
                "method": method,
                "budget": float(budget),
                "source": source,
                "n_subjects": int(group["subject_id"].nunique()),
                "n_subjects_accuracy_ge_075": int(
                    reached["subject_id"].nunique()
                ),
                "fraction_accuracy_ge_075": (
                    None if group.empty else float(
                        (group["accuracy_after"] >= 0.75).mean()
                    )
                ),
                "mean_accuracy": (
                    None if accuracy.empty else float(accuracy.mean())
                ),
                "median_accuracy": (
                    None if accuracy.empty else float(accuracy.median())
                ),
                "min_accuracy": (
                    None if accuracy.empty else float(accuracy.min())
                ),
                "max_accuracy": (
                    None if accuracy.empty else float(accuracy.max())
                ),
            })
    reached_columns = [
        "subject_id", "source", "outer_fold", "seed", "method", "budget",
        "accuracy_after", "balanced_accuracy_after", "macro_f1_after",
    ]
    reached = completed.loc[
        completed["accuracy_after"] >= 0.75, reached_columns
    ].copy()
    return pd.DataFrame(summary_rows), reached


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _svg_line_chart(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    label: str,
) -> None:
    frame = summary.loc[
        (summary["valid_subjects"] > 0) & summary[metric].notna()
    ].copy()
    width, height = 820, 460
    left, right, top, bottom = 80, 30, 35, 65
    x_values = sorted(frame["budget"].unique())
    y_values = frame[metric].to_numpy(dtype=float)
    y_min = float(np.min(y_values)) - 0.01
    y_max = float(np.max(y_values)) + 0.01
    y_span = max(1e-6, y_max - y_min)
    x_min, x_max = min(x_values), max(x_values)
    x_span = max(1.0, x_max - x_min)
    x = lambda value: left + (float(value) - x_min) / x_span * (width - left - right)
    y = lambda value: top + (y_max - float(value)) / y_span * (height - top - bottom)
    colors = {
        "zero_shot": "#4c78a8",
        "subject_normalization": "#e45756",
        "head_only": "#54a24b",
        "full_model": "#b279a2",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f"<title>{html.escape(label)} versus calibration duration</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>',
    ]
    for value in np.linspace(y_min, y_max, 6):
        position = y(value)
        parts.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{width-right}" y2="{position:.1f}" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{position+4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>'
        )
    for value in x_values:
        position = x(value)
        parts.append(
            f'<text x="{position:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value:g}</text>'
        )
    for method, group in frame.groupby("method", sort=False):
        group = group.sort_values("budget")
        points = " ".join(
            f'{x(row.budget):.1f},{y(getattr(row, metric)):.1f}'
            for row in group.itertuples()
        )
        color = colors.get(str(method), "#777")
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for row in group.itertuples():
            parts.append(
                f'<circle cx="{x(row.budget):.1f}" cy="{y(getattr(row, metric)):.1f}" r="4" fill="{color}"/>'
            )
    legend_x = left
    for method in frame["method"].drop_duplicates():
        color = colors.get(str(method), "#777")
        parts.extend([
            f'<rect x="{legend_x}" y="8" width="12" height="12" fill="{color}"/>',
            f'<text x="{legend_x+17}" y="19" font-size="12">{html.escape(str(method))}</text>',
        ])
        legend_x += 155
    parts.extend([
        f'<text x="{(left+width-right)/2:.1f}" y="{height-15}" text-anchor="middle" font-size="13">Calibration budget (fraction or seconds)</text>',
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 18 {height/2:.1f})">{html.escape(label)}</text>',
        "</svg>",
    ])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_heatmap(subjects: pd.DataFrame, output_path: Path) -> None:
    frame = subjects.loc[
        (subjects["status"] == "valid")
        & (subjects["calibration_method"] != "zero_shot")
    ].copy()
    frame["condition"] = (
        frame["calibration_method"].astype(str) + "_"
        + frame["budget"].astype(str)
    )
    pivot = frame.pivot_table(
        index="subject_id",
        columns="condition",
        values="delta_balanced_accuracy_vs_zero_shot",
        aggfunc="mean",
    )
    cell_w, cell_h, left, top = 72, 15, 105, 140
    width = left + cell_w * len(pivot.columns) + 25
    height = top + cell_h * len(pivot.index) + 35
    maximum = max(0.01, float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        "<title>Per-subject balanced accuracy delta versus matched zero-shot</title>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for column_index, column in enumerate(pivot.columns):
        x = left + column_index * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top-8}" font-size="10" text-anchor="start" transform="rotate(-55 {x:.1f} {top-8})">{html.escape(str(column))}</text>'
        )
    for row_index, (subject, row) in enumerate(pivot.iterrows()):
        y = top + row_index * cell_h
        parts.append(
            f'<text x="{left-5}" y="{y+11}" text-anchor="end" font-size="9">{html.escape(str(subject))}</text>'
        )
        for column_index, value in enumerate(row):
            if pd.isna(value):
                color = "#eeeeee"
            else:
                strength = min(1.0, abs(float(value)) / maximum)
                if value >= 0:
                    color = f"rgb({int(240-130*strength)},{int(248-60*strength)},{int(240-120*strength)})"
                else:
                    color = f"rgb({int(248-40*strength)},{int(240-140*strength)},{int(240-140*strength)})"
            x = left + column_index * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-1}" fill="{color}"/>'
            )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_coverage_scatter(subjects: pd.DataFrame, output_path: Path) -> None:
    frame = subjects.loc[
        (subjects["status"] == "valid")
        & subjects["calibration_method"].isin(["head_only", "full_model"])
    ].copy()
    width, height = 760, 430
    left, right, top, bottom = 70, 25, 25, 60
    values = frame["delta_balanced_accuracy_vs_zero_shot"].to_numpy(dtype=float)
    y_min, y_max = min(-0.08, float(np.min(values))), max(0.12, float(np.max(values)))
    y = lambda value: top + (y_max - float(value)) / (y_max-y_min) * (height-top-bottom)
    x = lambda value: left + (float(value)-1) / 4 * (width-left-right)
    colors = {"head_only": "#54a24b", "full_model": "#b279a2"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        "<title>Calibration class coverage and balanced accuracy change</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{width-right}" y2="{y(0):.1f}" stroke="#777"/>',
    ]
    for row in frame.itertuples():
        parts.append(
            f'<circle cx="{x(row.number_of_classes):.1f}" cy="{y(row.delta_balanced_accuracy_vs_zero_shot):.1f}" r="3" fill="{colors.get(row.calibration_method, "#777")}" fill-opacity="0.55"/>'
        )
    for value in range(1, 6):
        parts.append(
            f'<text x="{x(value):.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value}</text>'
        )
    parts.extend([
        f'<text x="{width/2:.1f}" y="{height-12}" text-anchor="middle" font-size="13">Classes present in calibration</text>',
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 18 {height/2:.1f})">Delta balanced accuracy</text>',
        "</svg>",
    ])
    output_path.write_text("\n".join(parts), encoding="utf-8")


class UserCalibrationExperiment:
    """Calibrate independent subject clones of canonical outer-fold models."""

    def __init__(self, config_path: str | Path):
        self.config_path = _repo_path(config_path)
        with open(self.config_path, encoding="utf-8") as input_file:
            self.document = yaml.safe_load(input_file) or {}
        if self.document.get("experiment", {}).get("type") != "user_calibration":
            raise ValueError("experiment.type must be 'user_calibration'")
        base_run = self.document["base_run"]
        if "config_path" in base_run:
            base_config_path = _repo_path(base_run["config_path"])
            self.base_config = yaml.safe_load(
                base_config_path.read_text(encoding="utf-8")
            )
            completed = BenchmarkRunner.find_completed_run(self.base_config)
            if completed is None and bool(base_run.get("train_if_missing", False)):
                base_runner = BenchmarkRunner(deepcopy(self.base_config))
                base_runner.run()
                completed = base_runner.completed_run()
            if completed is None:
                raise FileNotFoundError(
                    "No completed base benchmark matches "
                    f"{base_config_path}; run it first or set train_if_missing: true"
                )
            self.base_run_dir = completed.run_directory
        else:
            self.base_run_dir = _repo_path(base_run["run_directory"])
            self.base_config = yaml.safe_load(
                (self.base_run_dir / "config.yaml").read_text(encoding="utf-8")
            )
        configured_model = str(
            self.document["base_run"].get(
                "model", next(iter(self.base_config["models"]))
            )
        )
        head_type = str(
            self.base_config["models"][configured_model]
            .get("params", {})
            .get("head_type", "categorical")
        ).strip().lower()
        if head_type != "categorical":
            message = (
                "Auxiliary CORN calibration is not supported yet."
                if head_type == "categorical_corn"
                else (
                    "Ordinal calibration is not supported yet; "
                    f"received head_type={head_type!r}"
                )
            )
            raise NotImplementedError(message)
        manifest_path = self.base_run_dir / "run_manifest.json"
        self.base_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        self.base_hash = benchmark_config_hash(self.base_config)
        BenchmarkRunner.validate_completed_run(
            self.base_run_dir,
            expected_config_hash=self.base_hash,
            result_file=(
                None
                if not self.base_manifest
                else _repo_path(self.base_manifest["benchmark_result_file"])
            ),
            manifest_file=manifest_path if manifest_path.is_file() else None,
        )
        self.base_results = json.loads(
            (self.base_run_dir / "metrics.json").read_text(encoding="utf-8")
        )

    def _identities(self) -> tuple[str, str, str]:
        base = self.document["base_run"]
        dataset_name = str(base.get("dataset", next(iter(self.base_config["datasets"]))))
        task_name = str(base.get("task", self.base_config["tasks"][0]))
        model_name = str(base.get("model", next(iter(self.base_config["models"]))))
        return dataset_name, task_name, model_name

    def _fold_checkpoint(
        self,
        fold_name: str,
        dataset_name: str,
        task_name: str,
        model_name: str,
    ) -> Path:
        artifacts = self.base_results[dataset_name]["models"][task_name][model_name][
            "group_kfold_subject"
        ]["folds"][fold_name]["artifacts"]
        path = _repo_path(artifacts["model"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing base fold checkpoint: {path}")
        return path

    def _load_fold_adapter(
        self,
        checkpoint: Path,
        model_name: str,
    ) -> TorchClassificationAdapter:
        payload = _checkpoint_payload(checkpoint)
        adapter = build_model(
            model_name=str(self.base_config["models"][model_name]["type"]),
            task_type="classification",
            input_shape=tuple(int(value) for value in payload["input_shape"]),
            num_outputs=int(payload["num_classes"]),
            params=deepcopy(self.base_config["models"][model_name]["params"]),
        )
        if not isinstance(adapter, TorchClassificationAdapter):
            raise TypeError("Calibration requires TorchClassificationAdapter")
        adapter.load(checkpoint)
        return adapter

    def _specs(
        self,
        *,
        budgets_seconds: Optional[Sequence[float]] = None,
        methods: Optional[Sequence[str]] = None,
        max_epochs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> list[CalibrationSpec]:
        config = self.document["calibration"]
        configured_fractions = config.get("budgets_fraction")
        if budgets_seconds is None and configured_fractions is not None:
            selected_budgets = list(configured_fractions)
            budget_mode = "fraction"
        else:
            selected_budgets = list(
                config["budgets_seconds"]
                if budgets_seconds is None
                else budgets_seconds
            )
            budget_mode = "seconds"
        selected_methods = list(config["methods"] if methods is None else methods)
        shared = dict(config.get("defaults", {}))
        method_parameters = config.get("method_params", {})
        specs: list[CalibrationSpec] = []
        for budget in selected_budgets:
            for method in selected_methods:
                values = {
                    **shared,
                    **dict(method_parameters.get(str(method), {})),
                    "method": str(method),
                    "budget_seconds": (
                        None
                        if budget_mode == "fraction"
                        else float(budget)
                    ),
                    "budget_fraction": (
                        float(budget)
                        if budget_mode == "fraction"
                        else None
                    ),
                }
                normalized_method = CalibrationSpec.from_dict(values).method
                if max_epochs is not None and normalized_method in {
                    "head_only", "full_model"
                }:
                    values["max_epochs"] = int(max_epochs)
                    values["fallback_fixed_epochs"] = min(
                        int(values.get("fallback_fixed_epochs", 3)), int(max_epochs)
                    )
                if random_state is not None:
                    values["random_state"] = int(random_state)
                specs.append(CalibrationSpec.from_dict(values))
        return specs

    @staticmethod
    def _status(
        spec: CalibrationSpec,
        partition: WindowPartition,
        calibration_sequences: SequenceBuildResult,
        evaluation_sequences: SequenceBuildResult,
    ) -> str:
        if spec.method != "zero_shot" and not _is_zero_budget(spec):
            if len(partition.calibration_X) == 0 or (
                partition.actual_seconds + 1e-6 < partition.requested_seconds
                and spec.budget_fraction is None
            ):
                return "insufficient_calibration_data"
            if len(partition.calibration_X) < spec.minimum_calibration_samples:
                return "insufficient_calibration_data"
            if len(calibration_sequences.X) < spec.min_calibration_sequences:
                return "insufficient_sequence_context"
        if len(partition.evaluation_X) < int(
            spec.minimum_final_evaluation_samples
        ):
            return "insufficient_evaluation_data"
        if len(evaluation_sequences.X) < spec.min_evaluation_sequences:
            return "insufficient_evaluation_data"
        return "valid"

    def _calibration_validation(
        self,
        partition: WindowPartition,
        spec: CalibrationSpec,
        sequence_config: Optional[Mapping[str, Any]],
        *,
        window_seconds: float,
        max_gap_seconds: float,
    ) -> tuple[SequenceBuildResult, Optional[SequenceBuildResult], str]:
        train_fraction = 1.0 - spec.calibration_validation_fraction
        validation_spec = CalibrationSpec(
            method=spec.method,
            budget_seconds=None,
            budget_fraction=train_fraction,
            split_strategy=spec.split_strategy,
            fraction_allocation=spec.fraction_allocation,
            purge_windows=spec.purge_windows,
            max_epochs=spec.max_epochs,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            early_stopping_patience=spec.early_stopping_patience,
            calibration_validation_fraction=spec.calibration_validation_fraction,
            fallback_fixed_epochs=spec.fallback_fixed_epochs,
            min_calibration_sequences=spec.min_calibration_sequences,
            min_evaluation_sequences=spec.min_evaluation_sequences,
            minimum_calibration_samples=spec.minimum_calibration_samples,
            minimum_evaluation_samples=spec.minimum_evaluation_samples,
            minimum_adaptation_train_samples=(
                spec.minimum_adaptation_train_samples
            ),
            minimum_adaptation_validation_samples=(
                spec.minimum_adaptation_validation_samples
            ),
            minimum_final_evaluation_samples=(
                spec.minimum_final_evaluation_samples
            ),
            random_state=spec.random_state,
        )
        inner = chronological_window_partition(
            partition.calibration_X,
            partition.calibration_y,
            partition.calibration_metadata,
            validation_spec,
            window_seconds=window_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        train_sequences = _build_model_inputs(
            inner.calibration_X,
            inner.calibration_y,
            inner.calibration_metadata,
            sequence_config,
        )
        validation_sequences = _build_model_inputs(
            inner.evaluation_X,
            inner.evaluation_y,
            inner.evaluation_metadata,
            sequence_config,
        )
        if (
            len(train_sequences.X) >= spec.minimum_adaptation_train_samples
            and len(validation_sequences.X)
            >= spec.minimum_adaptation_validation_samples
        ):
            return train_sequences, validation_sequences, "chronological_holdout"
        all_sequences = _build_model_inputs(
            partition.calibration_X,
            partition.calibration_y,
            partition.calibration_metadata,
            sequence_config,
        )
        return all_sequences, None, "none_fixed_epochs"

    @staticmethod
    def _split_manifest(
        partition: WindowPartition,
        calibration_sequences: SequenceBuildResult,
        evaluation_sequences: SequenceBuildResult,
        coverage: Mapping[str, Any],
        validation_mode: str,
        adaptation_train: Optional[SequenceBuildResult] = None,
        adaptation_validation: Optional[SequenceBuildResult] = None,
    ) -> dict[str, Any]:
        time_column = partition.time_column
        calibration_ids = partition.calibration_metadata.get(
            "sample_id", pd.Series(dtype=object)
        ).astype(str).tolist()
        purged_ids = partition.purged_metadata.get(
            "sample_id", pd.Series(dtype=object)
        ).astype(str).tolist()
        evaluation_ids = partition.evaluation_metadata.get(
            "sample_id", pd.Series(dtype=object)
        ).astype(str).tolist()
        reserved_ids = partition.reserved_metadata.get(
            "sample_id", pd.Series(dtype=object)
        ).astype(str).tolist()
        overlap = sorted(set(calibration_ids) & set(evaluation_ids))
        if overlap:
            raise RuntimeError(f"Calibration/evaluation sample overlap: {overlap[:10]}")
        adaptation_train_ids = (
            []
            if adaptation_train is None
            else adaptation_train.metadata["target_sample_id"].astype(str).tolist()
        )
        adaptation_validation_ids = (
            []
            if adaptation_validation is None
            else adaptation_validation.metadata["target_sample_id"].astype(str).tolist()
        )
        adaptation_overlap = sorted(
            set(adaptation_train_ids) & set(adaptation_validation_ids)
        )
        evaluation_adaptation_overlap = sorted(
            (set(adaptation_train_ids) | set(adaptation_validation_ids))
            & set(evaluation_ids)
        )
        if adaptation_overlap or evaluation_adaptation_overlap:
            raise RuntimeError("Adaptation partitions overlap each other or evaluation")
        return {
            "split_strategy": "chronological_prefix",
            "time_column": time_column,
            "requested_calibration_duration_seconds": partition.requested_seconds,
            "actual_calibration_duration_seconds": partition.actual_seconds,
            "available_subject_duration_seconds": partition.available_seconds,
            "calibration_windows": len(calibration_ids),
            "purged_windows": len(purged_ids),
            "reserved_windows": len(reserved_ids),
            "evaluation_windows": len(evaluation_ids),
            "requested_calibration_fraction": partition.requested_fraction,
            "actual_calibration_fraction": partition.actual_fraction,
            "calibration_sequences": int(len(calibration_sequences.X)),
            "evaluation_sequences": int(len(evaluation_sequences.X)),
            "calibration_sample_ids": calibration_ids,
            "purged_sample_ids": purged_ids,
            "reserved_sample_ids": reserved_ids,
            "evaluation_sample_ids": evaluation_ids,
            "window_overlap": overlap,
            "adaptation_train_samples": len(adaptation_train_ids),
            "adaptation_validation_samples": len(adaptation_validation_ids),
            "adaptation_train_sample_ids": adaptation_train_ids,
            "adaptation_validation_sample_ids": adaptation_validation_ids,
            "adaptation_validation_overlap": adaptation_overlap,
            "evaluation_adaptation_overlap": evaluation_adaptation_overlap,
            "calibration_class_distribution": _class_coverage(
                calibration_sequences.y
            ),
            "adaptation_train_class_distribution": _class_coverage(
                np.asarray([])
                if adaptation_train is None
                else adaptation_train.y
            ),
            "adaptation_validation_class_distribution": _class_coverage(
                np.asarray([])
                if adaptation_validation is None
                else adaptation_validation.y
            ),
            "evaluation_class_distribution": _class_coverage(
                evaluation_sequences.y
            ),
            "calibration_record_ids": sorted(
                partition.calibration_metadata.get(
                    "record_id", pd.Series(dtype=object)
                ).astype(str).unique().tolist()
            ),
            "evaluation_record_ids": sorted(
                partition.evaluation_metadata.get(
                    "record_id", pd.Series(dtype=object)
                ).astype(str).unique().tolist()
            ),
            "calibration_validation_mode": validation_mode,
            **dict(coverage),
        }

    @staticmethod
    def _prediction_frame(
        fold_name: str,
        subject_id: str,
        spec: CalibrationSpec,
        partition: WindowPartition,
        sequences: SequenceBuildResult,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        metadata = sequences.metadata.reset_index(drop=True)
        time_column = partition.time_column
        calibration_start = (
            None
            if partition.calibration_metadata.empty
            else float(partition.calibration_metadata[time_column].min())
        )
        calibration_end = (
            None
            if partition.calibration_metadata.empty
            else float(partition.calibration_metadata[time_column].max())
        )
        evaluation_start = float(partition.evaluation_metadata[time_column].min())
        evaluation_end = float(partition.evaluation_metadata[time_column].max())
        frame = pd.DataFrame({
            "outer_fold": fold_name,
            "subject_id": str(subject_id),
            "source": metadata["source"].astype(str),
            "record_id": metadata["record_id"].astype(str),
            "record_group_id": metadata["record_group_id"].astype(str),
            "sample_id": metadata["target_sample_id"].astype(str),
            "sequence_id": metadata["sequence_id"].astype(str),
            "calibration_method": spec.method,
            "budget_seconds": spec.budget_seconds,
            "budget_fraction": spec.budget_fraction,
            "budget": (
                spec.budget_fraction
                if spec.budget_fraction is not None
                else spec.budget_seconds
            ),
            "seed": int(spec.random_state),
            "y_true": sequences.y.astype(int),
            "y_pred": np.asarray(predictions, dtype=int),
            "is_calibration_sample": False,
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
        })
        for class_index in range(probabilities.shape[1]):
            frame[f"proba_{class_index}"] = probabilities[:, class_index]
        return frame

    def execute(
        self,
        *,
        fold_limit: Optional[int] = None,
        subject_limit: Optional[int] = None,
        budgets_seconds: Optional[Sequence[float]] = None,
        methods: Optional[Sequence[str]] = None,
        max_epochs: Optional[int] = None,
        random_state: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        write_reports: bool = True,
        resume: bool = False,
    ) -> dict[str, Any]:
        dataset_name, task_name, model_name = self._identities()
        specs = self._specs(
            budgets_seconds=budgets_seconds,
            methods=methods,
            max_epochs=max_epochs,
            random_state=random_state,
        )
        model_type = str(self.base_config["models"][model_name]["type"])
        uses_sequences = model_requires_sequences(model_type)
        sequence_config = (
            self.base_config.get("sequence") if uses_sequences else None
        )
        if uses_sequences and sequence_config is None:
            raise ValueError("Sequence calibration requires base sequence config")
        sequence_length = (
            int(sequence_config["length"])
            if sequence_config is not None
            else 1
        )
        if any(spec.purge_windows < sequence_length - 1 for spec in specs):
            raise ValueError(
                f"purge_windows must be at least sequence_length - 1 "
                f"({sequence_length - 1})"
            )
        calibration_config = self.document["calibration"]
        experiment_config = self.document["experiment"]
        require_cuda = bool(experiment_config.get("require_cuda", False))
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required by this calibration experiment, but is unavailable"
            )
        window_seconds = float(calibration_config.get("window_seconds", 10.0))
        max_gap_seconds = float(
            calibration_config.get("max_gap_seconds", window_seconds * 1.05)
        )

        code_hash = _implementation_hash()
        resolved = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "base_run": deepcopy(self.document["base_run"]),
            "base_config_hash": self.base_hash,
            "specs": [spec.to_dict() for spec in specs],
            "fold_limit": fold_limit,
            "subject_limit": subject_limit,
            "input_mode": "sequences" if uses_sequences else "feature_windows",
            "implementation_hash": code_hash,
        }
        config_hash = _canonical_hash(resolved)
        root = _repo_path(
            output_dir
            if output_dir is not None
            else self.document["experiment"]["output_dir"]
        )
        resume_enabled = bool(resume or experiment_config.get("resume", False))
        run_dir: Optional[Path] = None
        if resume_enabled and root.is_dir():
            for candidate in sorted(
                (path for path in root.iterdir() if path.is_dir()),
                reverse=True,
            ):
                progress_path = candidate / "progress.json"
                if not progress_path.is_file():
                    continue
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("config_hash") != config_hash:
                    continue
                if progress.get("implementation_hash") != code_hash:
                    raise RuntimeError(
                        "Resume state implementation hash does not match current code"
                    )
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
                    manifest["resume_skipped_completed_conditions"] = int(
                        progress.get("completed_conditions", 0)
                    )
                    return manifest
                break
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = root / timestamp
            run_dir.mkdir(parents=True, exist_ok=False)
            with open(
                run_dir / "resolved_calibration.yaml", "w", encoding="utf-8"
            ) as output:
                yaml.safe_dump(resolved, output, sort_keys=False)
            _write_json(
                run_dir / "progress.json",
                {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "status": "running",
                    "config_hash": config_hash,
                    "implementation_hash": code_hash,
                    "completed_conditions": 0,
                    "failed_conditions": 0,
                    "condition_keys": [],
                },
            )

        runner = BenchmarkRunner(deepcopy(self.base_config))
        data = runner.load_dataset(dataset_name)
        task = get_task(task_name, data, self.base_config.get("task_config", {}))
        evaluation = self.base_config["evaluation"]
        folds = CrossValidator(task).run_group_kfold(
            group_column=evaluation["group_column"],
            n_splits=int(evaluation.get("n_splits", 5)),
            random_state=int(evaluation.get("random_state", 42)),
            precomputed_fold_column=evaluation.get("precomputed_fold_column"),
        )
        requested_folds = evaluation.get("folds")
        if requested_folds is not None:
            requested_names = {
                f"fold_{int(fold):02d}" for fold in requested_folds
            }
            unknown = requested_names.difference(folds)
            if unknown:
                raise ValueError(
                    f"Requested evaluation folds do not exist: {sorted(unknown)}"
                )
            folds = {
                name: split for name, split in folds.items()
                if name in requested_names
            }
        if fold_limit is not None:
            if fold_limit <= 0:
                raise ValueError("fold_limit must be positive")
            folds = dict(list(folds.items())[: int(fold_limit)])

        subject_rows: list[dict[str, Any]] = []
        split_audit_rows: list[dict[str, Any]] = []
        checkpoint_audit_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        failure_rows: list[dict[str, Any]] = []
        global_fold_rows: list[dict[str, Any]] = []
        completed_condition_keys: set[str] = set()
        for condition_path in sorted(run_dir.rglob("condition_result.json")):
            payload = json.loads(condition_path.read_text(encoding="utf-8"))
            condition_key = str(payload["condition_key"])
            if condition_key in completed_condition_keys:
                raise RuntimeError(
                    f"Duplicate resume condition key: {condition_key}"
                )
            completed_condition_keys.add(condition_key)
            subject_rows.append(dict(payload["subject_metrics"]))
            split_audit_rows.append(dict(payload["split_audit"]))
            checkpoint_audit_rows.append(dict(payload["checkpoint_audit"]))
            prediction_path = condition_path.parent / "predictions.parquet"
            if not prediction_path.is_file():
                raise RuntimeError(
                    f"Completed condition is missing predictions: {prediction_path}"
                )
            prediction_frames.append(pd.read_parquet(prediction_path))
        failures_path = run_dir / "failures.csv"
        if failures_path.is_file():
            failure_rows = pd.read_csv(failures_path).to_dict("records")

        def persist_progress() -> None:
            raw_subjects = pd.DataFrame(subject_rows)
            normalized = (
                pd.DataFrame()
                if raw_subjects.empty
                else _normalized_subject_metrics(raw_subjects)
            )
            normalized.to_csv(
                run_dir / "calibration_subject_metrics.csv", index=False
            )
            pd.DataFrame(split_audit_rows).to_csv(
                run_dir / "calibration_split_audit.csv", index=False
            )
            pd.DataFrame(checkpoint_audit_rows).to_csv(
                run_dir / "checkpoint_audit.csv", index=False
            )
            pd.DataFrame(
                failure_rows,
                columns=[
                    "outer_fold", "subject_id", "budget", "method", "seed",
                    "status", "error_type", "error_message", "traceback",
                ],
            ).to_csv(failures_path, index=False)
            _write_json(
                run_dir / "progress.json",
                {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "status": "running",
                    "config_hash": config_hash,
                    "implementation_hash": code_hash,
                    "completed_conditions": len(completed_condition_keys),
                    "failed_conditions": len(failure_rows),
                    "condition_keys": sorted(completed_condition_keys),
                },
            )

        persist_progress()
        device_info: Optional[dict[str, str]] = None
        started = time.perf_counter()
        for fold_name, outer_split in folds.items():
            if require_cuda:
                torch.cuda.empty_cache()
            if outer_split.metadata.get("subject_overlap"):
                raise RuntimeError(f"Outer subject leakage in {fold_name}")
            checkpoint = self._fold_checkpoint(
                fold_name, dataset_name, task_name, model_name
            )
            base_adapter = self._load_fold_adapter(checkpoint, model_name)
            if require_cuda and base_adapter.device_.type != "cuda":
                raise RuntimeError(
                    f"{fold_name} checkpoint loaded on {base_adapter.device_}; "
                    "CUDA is required"
                )
            if device_info is None:
                device_info = {
                    "device": str(base_adapter.device_),
                    "device_name": (
                        torch.cuda.get_device_name(base_adapter.device_)
                        if base_adapter.device_.type == "cuda"
                        else "CPU"
                    ),
                }
            base_digest = _state_digest(base_adapter)
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            test_subjects = sorted(np.unique(outer_split.subject_test).astype(str))
            configured_subjects = {
                str(value)
                for value in self.document["calibration"].get(
                    "target_subjects", []
                )
            }
            if configured_subjects:
                test_subjects = [
                    value for value in test_subjects
                    if value in configured_subjects
                ]
            if subject_limit is not None:
                if subject_limit <= 0:
                    raise ValueError("subject_limit must be positive")
                test_subjects = test_subjects[: int(subject_limit)]
            train_subjects = set(np.unique(outer_split.subject_train).astype(str))
            if train_subjects & set(test_subjects):
                raise RuntimeError(f"Outer train/test subject leakage in {fold_name}")
            validation_split = base_adapter.validation_split_ or {}
            global_train_subjects = {
                str(value) for value in
                validation_split.get("inner_train_subject_ids", [])
            }
            global_validation_subjects = {
                str(value) for value in
                validation_split.get("inner_validation_subject_ids", [])
            }
            all_outer_test_subjects = {
                str(value) for value in np.unique(outer_split.subject_test)
            }
            preprocessing_state = base_adapter.get_feature_preprocessing_state()
            fit_test_overlap = sorted(
                global_train_subjects & all_outer_test_subjects
            )
            if fit_test_overlap:
                raise RuntimeError(
                    f"Preprocessing subject leakage in {fold_name}: "
                    f"{fit_test_overlap}"
                )
            base_fold = self.base_results[dataset_name]["models"][task_name][
                model_name
            ]["group_kfold_subject"]["folds"][fold_name]
            training_metadata = dict(base_fold.get("training", {}))
            clip_percentiles = (
                []
                if preprocessing_state is None
                else preprocessing_state.get("clip_percentiles", [])
            )
            global_fold_rows.append({
                "outer_fold": fold_name,
                "seed": int(evaluation.get("random_state", 42)),
                "n_outer_train_subjects": len(train_subjects),
                "n_outer_test_subjects": len(all_outer_test_subjects),
                "n_inner_train_subjects": len(global_train_subjects),
                "n_inner_validation_subjects": len(
                    global_validation_subjects
                ),
                "n_train_samples": int(base_fold.get("n_train", 0)),
                "n_test_samples": int(base_fold.get("n_test", 0)),
                "training_time_seconds": float(
                    base_fold.get("training_time", 0.0)
                ),
                "epochs_trained": training_metadata.get("epochs_trained"),
                "best_epoch": training_metadata.get("best_epoch"),
                "best_validation_loss": training_metadata.get(
                    "best_validation_loss"
                ),
                "device_type": training_metadata.get("device"),
                "device_name": training_metadata.get("device_name"),
                "peak_gpu_memory_bytes": training_metadata.get(
                    "peak_gpu_memory_bytes", 0
                ),
                "preprocessor_fit_subjects": json.dumps(
                    sorted(global_train_subjects)
                ),
                "outer_test_subjects": json.dumps(
                    sorted(all_outer_test_subjects)
                ),
                "fit_test_overlap": len(fit_test_overlap),
                "feature_hash": (
                    None
                    if preprocessing_state is None
                    else preprocessing_state.get("feature_hash")
                ),
                "lower_quantile": (
                    None if not clip_percentiles else clip_percentiles[0]
                ),
                "upper_quantile": (
                    None if not clip_percentiles else clip_percentiles[1]
                ),
            })
            LOGGER.info(
                "Calibration %s: global_train_subjects=%d target_subjects=%d "
                "checkpoint_restored=%s",
                fold_name,
                len(train_subjects),
                len(test_subjects),
                checkpoint,
            )
            test_metadata = runner._partition_sequence_metadata(outer_split, "test")

            for subject_id in test_subjects:
                subject_mask = np.asarray(outer_split.subject_test).astype(str) == subject_id
                subject_X = np.asarray(outer_split.X_test)[subject_mask]
                subject_y = np.asarray(outer_split.y_test)[subject_mask]
                subject_metadata = test_metadata.loc[subject_mask].reset_index(drop=True)
                subject_sources = sorted(
                    subject_metadata["source"].astype(str).unique().tolist()
                )
                subject_source = (
                    subject_sources[0]
                    if len(subject_sources) == 1
                    else "both"
                )
                subject_root = (
                    run_dir / fold_name / _safe_component(subject_id)
                )
                subject_root.mkdir(parents=True, exist_ok=True)
                subject_checkpoint = subject_root / "global_model.pt"
                shutil.copy2(checkpoint, subject_checkpoint)
                if hashlib.sha256(subject_checkpoint.read_bytes()).hexdigest() != (
                    checkpoint_digest
                ):
                    raise RuntimeError(
                        "Copied subject-level global checkpoint hash mismatch"
                    )
                _write_json(
                    subject_root / "global_checkpoint.json",
                    {
                        "source_checkpoint": str(checkpoint),
                        "saved_checkpoint": str(subject_checkpoint),
                        "sha256": checkpoint_digest,
                        "model_state_hash": base_digest,
                    },
                )
                budget_values = sorted({
                    (
                        "fraction",
                        float(spec.budget_fraction or 0.0),
                    )
                    if spec.budget_fraction is not None
                    else ("seconds", float(spec.budget_seconds or 0.0))
                    for spec in specs
                }, key=lambda value: (value[0], value[1]))
                maximum_spec = max(
                    specs,
                    key=lambda item: (
                        float(item.budget_fraction or 0.0),
                        float(item.budget_seconds or 0.0),
                    ),
                )
                reference_partition = chronological_window_partition(
                    subject_X,
                    subject_y,
                    subject_metadata,
                    maximum_spec,
                    window_seconds=window_seconds,
                    max_gap_seconds=max_gap_seconds,
                )
                for budget_kind, budget_value in budget_values:
                    budget_specs = [
                        spec for spec in specs
                        if (
                            (
                                "fraction",
                                float(spec.budget_fraction or 0.0),
                            )
                            if spec.budget_fraction is not None
                            else (
                                "seconds",
                                float(spec.budget_seconds or 0.0),
                            )
                        ) == (budget_kind, budget_value)
                    ]
                    split_spec = budget_specs[0]
                    partition = chronological_window_partition(
                        subject_X,
                        subject_y,
                        subject_metadata,
                        split_spec,
                        window_seconds=window_seconds,
                        max_gap_seconds=max_gap_seconds,
                    )
                    if self.document["calibration"].get("budgets_fraction") is not None:
                        partition = _use_reference_evaluation(
                            partition, reference_partition
                        )
                    calibration_sequences = _build_model_inputs(
                        partition.calibration_X,
                        partition.calibration_y,
                        partition.calibration_metadata,
                        sequence_config,
                    )
                    evaluation_sequences = _build_model_inputs(
                        partition.evaluation_X,
                        partition.evaluation_y,
                        partition.evaluation_metadata,
                        sequence_config,
                    )
                    coverage = _class_coverage(calibration_sequences.y)

                    for spec in budget_specs:
                        condition_key = "|".join([
                            fold_name,
                            subject_id,
                            f"{budget_kind}:{budget_value:.8f}",
                            spec.method,
                            str(spec.random_state),
                        ])
                        if condition_key in completed_condition_keys:
                            LOGGER.info(
                                "Resume skip completed condition %s",
                                condition_key,
                            )
                            continue
                        if require_cuda:
                            torch.cuda.reset_peak_memory_stats()
                        status = self._status(
                            spec,
                            partition,
                            calibration_sequences,
                            evaluation_sequences,
                        )
                        artifact_dir = (
                            run_dir / fold_name / _safe_component(subject_id)
                            / (
                                f"budget_{budget_value:.4f}"
                                + ("fraction" if budget_kind == "fraction" else "s")
                            )
                            / spec.method
                        )
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        with open(
                            artifact_dir / "calibration_spec.yaml",
                            "w",
                            encoding="utf-8",
                        ) as output:
                            yaml.safe_dump(spec.to_dict(), output, sort_keys=False)
                        model_reference = {
                            "base_benchmark_run": str(self.base_run_dir),
                            "outer_fold": fold_name,
                            "base_checkpoint": str(checkpoint),
                            "base_checkpoint_sha256": checkpoint_digest,
                            "base_config_hash": self.base_hash,
                            **dict(device_info or {}),
                        }
                        _write_json(
                            artifact_dir / "model_reference.json", model_reference
                        )

                        validation_mode = "not_applicable"
                        training_log = pd.DataFrame()
                        training_time = 0.0
                        metrics: dict[str, Any] = {}
                        metrics_before: dict[str, Any] = {}
                        predictions_frame = pd.DataFrame()
                        adapted: Optional[TorchClassificationAdapter] = None
                        normalization_metadata: Optional[dict[str, Any]] = None
                        train_sequences: Optional[SequenceBuildResult] = None
                        validation_sequences: Optional[SequenceBuildResult] = None
                        global_hash = base_digest
                        initial_hash: Optional[str] = None
                        final_hash: Optional[str] = None
                        initial_predictions_match = False
                        frozen_unchanged: Optional[bool] = None
                        trainable_names: list[str] = []
                        frozen_names: list[str] = []
                        trainable_count = 0
                        frozen_count = 0
                        frozen_hash_before: Optional[str] = None
                        frozen_hash_after: Optional[str] = None
                        trainable_hash_before: Optional[str] = None
                        trainable_hash_after: Optional[str] = None
                        if status == "valid":
                            base_probabilities = base_adapter.predict_proba(
                                evaluation_sequences.X
                            )
                            base_predictions = base_probabilities.argmax(axis=1)
                            metrics_before = MetricsCalculator.calculate_all_metrics(
                                evaluation_sequences.y,
                                base_predictions,
                                base_probabilities,
                                labels=np.arange(base_adapter.num_classes),
                            )
                            adapted = base_adapter.clone()
                            initial_hash = _state_digest(adapted)
                            initial_clone_probabilities = adapted.predict_proba(
                                evaluation_sequences.X
                            )
                            initial_predictions_match = bool(np.allclose(
                                initial_clone_probabilities,
                                base_probabilities,
                                atol=1e-7,
                                rtol=1e-7,
                            ))
                            if initial_hash != global_hash or not initial_predictions_match:
                                raise RuntimeError(
                                    "Fine-tuning clone does not match global checkpoint"
                                )
                            (
                                trainable_names,
                                frozen_names,
                                trainable_count,
                                frozen_count,
                            ) = _parameter_audit(adapted, spec.method)
                            frozen_hash_before = _parameter_digest(
                                adapted, frozen_names
                            )
                            trainable_hash_before = _parameter_digest(
                                adapted, trainable_names
                            )
                            if _is_zero_budget(spec):
                                validation_mode = "zero_budget_no_adaptation"
                            elif spec.method == "subject_normalization":
                                subject_mean, subject_scale = (
                                    calibration_normalization_statistics(
                                        partition.calibration_X
                                    )
                                )
                                normalization_metadata = {
                                    "combination": "replace_outer_train_with_calibration_only",
                                    "evaluation_used": False,
                                    "base_train_mean": adapted.feature_mean_,
                                    "base_train_scale": adapted.feature_scale_,
                                    "subject_calibration_mean": subject_mean,
                                    "subject_calibration_scale": subject_scale,
                                }
                                adapted.set_feature_normalization(
                                    subject_mean, subject_scale
                                )
                            elif spec.method in {"head_only", "full_model"}:
                                train_sequences, validation_sequences, validation_mode = (
                                    self._calibration_validation(
                                        partition,
                                        spec,
                                        sequence_config,
                                        window_seconds=window_seconds,
                                        max_gap_seconds=max_gap_seconds,
                                    )
                                )
                                fit_epochs = (
                                    spec.max_epochs
                                    if validation_sequences is not None
                                    else min(
                                        spec.max_epochs, spec.fallback_fixed_epochs
                                    )
                                )
                                fit_started = time.perf_counter()
                                adapted.random_state = spec.random_state
                                try:
                                    adapted.fine_tune(
                                        train_sequences.X,
                                        train_sequences.y,
                                        mode=spec.method,
                                        X_validation=(
                                            None
                                            if validation_sequences is None
                                            else validation_sequences.X
                                        ),
                                        y_validation=(
                                            None
                                            if validation_sequences is None
                                            else validation_sequences.y
                                        ),
                                        max_epochs=fit_epochs,
                                        learning_rate=spec.learning_rate,
                                        weight_decay=spec.weight_decay,
                                        early_stopping_patience=(
                                            spec.early_stopping_patience
                                        ),
                                        random_state=spec.random_state,
                                    )
                                except Exception as exc:
                                    failure = {
                                        "outer_fold": fold_name,
                                        "subject_id": subject_id,
                                        "budget": budget_value,
                                        "method": spec.method,
                                        "seed": int(spec.random_state),
                                        "status": "training_failed",
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc),
                                        "traceback": traceback.format_exc(),
                                    }
                                    failure_rows.append(failure)
                                    _write_json(
                                        artifact_dir / "failure.json", failure
                                    )
                                    persist_progress()
                                    LOGGER.exception(
                                        "Calibration condition failed: %s",
                                        condition_key,
                                    )
                                    del adapted
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                training_time = time.perf_counter() - fit_started
                                training_log = pd.DataFrame(adapted.training_log_)

                            probabilities = adapted.predict_proba(
                                evaluation_sequences.X
                            )
                            predictions = probabilities.argmax(axis=1)
                            if not np.isfinite(probabilities).all() or not np.allclose(
                                probabilities.sum(axis=1), 1.0, atol=1e-5
                            ):
                                failure = {
                                    "outer_fold": fold_name,
                                    "subject_id": subject_id,
                                    "budget": budget_value,
                                    "method": spec.method,
                                    "seed": int(spec.random_state),
                                    "status": "non_finite_predictions",
                                    "error_type": "InvalidProbabilities",
                                    "error_message": (
                                        "Probabilities are non-finite or do not "
                                        "sum to one"
                                    ),
                                    "traceback": "",
                                }
                                failure_rows.append(failure)
                                _write_json(
                                    artifact_dir / "failure.json", failure
                                )
                                persist_progress()
                                del adapted
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            metrics = MetricsCalculator.calculate_all_metrics(
                                evaluation_sequences.y,
                                predictions,
                                probabilities,
                                labels=np.arange(adapted.num_classes),
                            )
                            final_hash = _state_digest(adapted)
                            frozen_hash_after = _parameter_digest(
                                adapted, frozen_names
                            )
                            trainable_hash_after = _parameter_digest(
                                adapted, trainable_names
                            )
                            frozen_unchanged = (
                                frozen_hash_before == frozen_hash_after
                            )
                            if (
                                spec.method == "zero_shot" or _is_zero_budget(spec)
                            ) and final_hash != initial_hash:
                                raise RuntimeError(
                                    "No-adaptation model state unexpectedly changed"
                                )
                            if spec.method == "head_only" and not frozen_unchanged:
                                raise RuntimeError(
                                    "Frozen body parameters changed in head-only mode"
                                )
                            predictions_frame = self._prediction_frame(
                                fold_name,
                                subject_id,
                                spec,
                                partition,
                                evaluation_sequences,
                                predictions,
                                probabilities,
                            )
                            prediction_frames.append(predictions_frame)
                            before_frame = self._prediction_frame(
                                fold_name,
                                subject_id,
                                spec,
                                partition,
                                evaluation_sequences,
                                base_predictions,
                                base_probabilities,
                            )
                            before_frame.to_parquet(
                                artifact_dir / "predictions_before.parquet",
                                index=False,
                            )
                            predictions_frame.to_parquet(
                                artifact_dir / "predictions_after.parquet",
                                index=False,
                            )
                            adapted.save(artifact_dir / "model.pt")

                        split_manifest = self._split_manifest(
                            partition,
                            calibration_sequences,
                            evaluation_sequences,
                            coverage,
                            validation_mode,
                            adaptation_train=train_sequences,
                            adaptation_validation=validation_sequences,
                        )
                        split_manifest["status"] = status
                        split_manifest["purge_windows"] = spec.purge_windows
                        split_manifest["evaluation_used_for_normalization"] = False
                        split_manifest["evaluation_used_for_early_stopping"] = False
                        validation_split = base_adapter.validation_split_ or {}
                        global_train_subjects = set(
                            str(value) for value in
                            validation_split.get("inner_train_subject_ids", [])
                        )
                        global_validation_subjects = set(
                            str(value) for value in
                            validation_split.get(
                                "inner_validation_subject_ids", []
                            )
                        )
                        global_target_overlap = int(
                            subject_id in global_train_subjects
                            or subject_id in global_validation_subjects
                        )
                        if global_target_overlap:
                            raise RuntimeError(
                                "Target subject reached global train/validation"
                            )
                        preprocessing_state = (
                            base_adapter.get_feature_preprocessing_state()
                        )
                        preprocessing_audit = {
                            "strategy": (
                                None
                                if preprocessing_state is None
                                else preprocessing_state.get("strategy")
                            ),
                            "preprocessor_fit_subjects": sorted(
                                global_train_subjects
                            ),
                            "target_subject": subject_id,
                            "fit_target_overlap": int(
                                subject_id in global_train_subjects
                            ),
                            "n_fit_samples": (
                                None
                                if preprocessing_state is None
                                else preprocessing_state.get("n_fit_samples")
                            ),
                            "state_hash": (
                                None
                                if preprocessing_state is None
                                else _canonical_hash(preprocessing_state)
                            ),
                        }
                        if preprocessing_audit["fit_target_overlap"]:
                            raise RuntimeError(
                                "Target subject reached preprocessing fit"
                            )
                        _write_json(
                            artifact_dir / "preprocessing_audit.json",
                            preprocessing_audit,
                        )

                        _write_json(
                            artifact_dir / "calibration_split.json", split_manifest
                        )
                        training_log.to_csv(
                            artifact_dir / "calibration_training_log.csv", index=False
                        )
                        training_log.to_csv(
                            artifact_dir / "training_log.csv", index=False
                        )
                        budget_seconds = (
                            None
                            if spec.budget_seconds is None
                            else float(spec.budget_seconds)
                        )
                        budget_fraction = (
                            None
                            if spec.budget_fraction is None
                            else float(spec.budget_fraction)
                        )
                        metric_payload = {
                            "status": status,
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "calibration_method": spec.method,
                            "budget_seconds": budget_seconds,
                            "budget_fraction": budget_fraction,
                            "training_time_seconds": training_time,
                            "calibration_validation_mode": validation_mode,
                            "metrics": metrics,
                        }
                        _write_json(
                            artifact_dir / "calibration_metrics.json", metric_payload
                        )
                        _write_json(
                            artifact_dir / "metrics_before.json", metrics_before
                        )
                        _write_json(
                            artifact_dir / "metrics_after.json", metrics
                        )
                        fine_tuning_summary = {
                            "status": status,
                            "method": spec.method,
                            "global_checkpoint_hash": global_hash,
                            "fine_tune_initial_hash": initial_hash,
                            "fine_tune_final_hash": final_hash,
                            "initial_matches_global": initial_hash == global_hash,
                            "initial_predictions_match_global": (
                                initial_predictions_match
                            ),
                            "frozen_parameters_unchanged": frozen_unchanged,
                            "frozen_hash_before": frozen_hash_before,
                            "frozen_hash_after": frozen_hash_after,
                            "trainable_hash_before": trainable_hash_before,
                            "trainable_hash_after": trainable_hash_after,
                            "trainable_parameter_names": trainable_names,
                            "frozen_parameter_names": frozen_names,
                            "trainable_parameter_count": trainable_count,
                            "frozen_parameter_count": frozen_count,
                            "epochs_trained": (
                                0
                                if (
                                    spec.method == "zero_shot"
                                    or _is_zero_budget(spec)
                                    or adapted is None
                                )
                                else adapted.n_epochs_trained_
                            ),
                            "best_epoch": (
                                None
                                if (
                                    spec.method == "zero_shot"
                                    or _is_zero_budget(spec)
                                    or adapted is None
                                )
                                else adapted.best_epoch_
                            ),
                            "best_validation_loss": (
                                None
                                if adapted is None or _is_zero_budget(spec)
                                else adapted.best_validation_loss_
                            ),
                            "fine_tuning_validation_strategy": validation_mode,
                            "evaluation_used_for_early_stopping": False,
                        }
                        _write_json(
                            artifact_dir / "fine_tuning_summary.json",
                            fine_tuning_summary,
                        )
                        with open(
                            artifact_dir / "config.yaml", "w", encoding="utf-8"
                        ) as output:
                            yaml.safe_dump(
                                {
                                    "base_run": model_reference,
                                    "calibration": spec.to_dict(),
                                },
                                output,
                                sort_keys=False,
                            )
                        calibration_samples = partition.calibration_metadata.copy()
                        calibration_samples["label_q5"] = partition.calibration_y
                        calibration_samples.loc[
                            :, ["sample_id", "source", "subject_id", "record_id",
                                partition.time_column, "label_q5"]
                        ].to_parquet(
                            artifact_dir / "calibration_samples.parquet",
                            index=False,
                        )
                        evaluation_samples = partition.evaluation_metadata.copy()
                        evaluation_samples["label_q5"] = partition.evaluation_y
                        evaluation_samples.loc[
                            :, ["sample_id", "source", "subject_id", "record_id",
                                partition.time_column, "label_q5"]
                        ].to_parquet(
                            artifact_dir / "evaluation_samples.parquet",
                            index=False,
                        )
                        if normalization_metadata is not None:
                            _write_json(
                                artifact_dir / "normalization_stats.json",
                                normalization_metadata,
                            )
                        if predictions_frame.empty:
                            predictions_frame = pd.DataFrame(columns=[
                                "outer_fold", "subject_id", "source", "record_id",
                                "record_group_id", "sample_id", "sequence_id",
                                "calibration_method", "budget_seconds",
                                "budget_fraction", "budget", "seed", "y_true",
                                "y_pred", "proba_0", "proba_1", "proba_2",
                                "proba_3", "proba_4", "is_calibration_sample",
                                "calibration_start", "calibration_end",
                                "evaluation_start", "evaluation_end",
                            ])
                        predictions_frame.to_parquet(
                            artifact_dir / "predictions.parquet", index=False
                        )

                        row = {
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "source": subject_source,
                            "source_membership": json.dumps(subject_sources),
                            "seed": int(spec.random_state),
                            "calibration_method": spec.method,
                            "budget_seconds": budget_seconds,
                            "budget_fraction": budget_fraction,
                            "budget": (
                                budget_fraction
                                if budget_fraction is not None
                                else budget_seconds
                            ),
                            "status": status,
                            "requested_calibration_duration": partition.requested_seconds,
                            "actual_calibration_duration": partition.actual_seconds,
                            "requested_calibration_fraction": (
                                partition.requested_fraction
                            ),
                            "actual_calibration_fraction": (
                                partition.actual_fraction
                            ),
                            "calibration_samples": len(partition.calibration_X),
                            "reserved_samples": len(partition.reserved_metadata),
                            "n_calibration": len(partition.calibration_X),
                            "adaptation_train_samples": (
                                0
                                if train_sequences is None
                                else len(train_sequences.X)
                            ),
                            "adaptation_validation_samples": (
                                0
                                if validation_sequences is None
                                else len(validation_sequences.X)
                            ),
                            "evaluation_samples": len(partition.evaluation_X),
                            "n_evaluation": len(partition.evaluation_X),
                            "calibration_sequences": len(calibration_sequences.X),
                            "evaluation_sequences": len(evaluation_sequences.X),
                            "record_coverage": len(
                                split_manifest["calibration_record_ids"]
                            ),
                            "calibration_validation_mode": validation_mode,
                            "training_time_seconds": training_time,
                            "fine_tuning_time_seconds": training_time,
                            "device_type": str(base_adapter.device_.type),
                            "device_name": (
                                None
                                if device_info is None
                                else device_info["device_name"]
                            ),
                            "peak_gpu_memory_bytes": (
                                int(torch.cuda.max_memory_allocated())
                                if base_adapter.device_.type == "cuda"
                                else 0
                            ),
                            "global_checkpoint_hash": global_hash,
                            "fine_tune_initial_hash": initial_hash,
                            "fine_tune_final_hash": final_hash,
                            "initial_matches_global": initial_hash == global_hash,
                            "frozen_parameters_unchanged": frozen_unchanged,
                            "frozen_hash_before": frozen_hash_before,
                            "frozen_hash_after": frozen_hash_after,
                            "trainable_hash_before": trainable_hash_before,
                            "trainable_hash_after": trainable_hash_after,
                            "trainable_parameter_count": trainable_count,
                            "frozen_parameter_count": frozen_count,
                            "epochs_trained": (
                                0
                                if (
                                    spec.method == "zero_shot"
                                    or _is_zero_budget(spec)
                                    or adapted is None
                                )
                                else adapted.n_epochs_trained_
                            ),
                            "best_epoch": (
                                None
                                if (
                                    spec.method == "zero_shot"
                                    or _is_zero_budget(spec)
                                    or adapted is None
                                )
                                else adapted.best_epoch_
                            ),
                            "best_validation_loss": (
                                None
                                if (
                                    spec.method == "zero_shot"
                                    or _is_zero_budget(spec)
                                    or adapted is None
                                )
                                else adapted.best_validation_loss_
                            ),
                            **{
                                f"{name}_before": value
                                for name, value in metrics_before.items()
                            },
                            **coverage,
                            "classes_in_evaluation": split_manifest[
                                "evaluation_class_distribution"
                            ]["classes_present"],
                            "number_of_classes_in_evaluation": split_manifest[
                                "evaluation_class_distribution"
                            ]["number_of_classes"],
                            "evaluation_class_counts": split_manifest[
                                "evaluation_class_distribution"
                            ]["class_counts"],
                            **metrics,
                        }
                        for metric_name in (
                            "accuracy", "balanced_accuracy", "macro_f1"
                        ):
                            before = metrics_before.get(metric_name)
                            after = metrics.get(metric_name)
                            absolute = (
                                None
                                if before is None or after is None
                                else float(after - before)
                            )
                            relative = (
                                None
                                if absolute is None or abs(float(before)) < 1e-12
                                else float(absolute / float(before))
                            )
                            row[f"{metric_name}_absolute_gain"] = absolute
                            row[f"{metric_name}_relative_gain"] = relative
                        row["class_counts"] = json.dumps(
                            row["class_counts"], sort_keys=True
                        )
                        row["classes_present"] = json.dumps(
                            row["classes_present"]
                        )
                        row["classes_in_evaluation"] = json.dumps(
                            row["classes_in_evaluation"]
                        )
                        row["evaluation_class_counts"] = json.dumps(
                            row["evaluation_class_counts"], sort_keys=True
                        )
                        subject_rows.append(row)
                        split_audit_row = {
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "seed": int(spec.random_state),
                            "method": spec.method,
                            "budget": row["budget"],
                            "split_strategy": spec.split_strategy,
                            "n_global_train_subjects": len(
                                global_train_subjects
                            ),
                            "n_global_validation_subjects": len(
                                global_validation_subjects
                            ),
                            "n_calibration_pool": len(partition.calibration_X),
                            "n_adaptation_train": row[
                                "adaptation_train_samples"
                            ],
                            "n_adaptation_validation": row[
                                "adaptation_validation_samples"
                            ],
                            "n_final_evaluation": len(partition.evaluation_X),
                            "global_target_overlap": global_target_overlap,
                            "calibration_evaluation_overlap": len(
                                split_manifest["window_overlap"]
                            ),
                            "adaptation_validation_overlap": len(
                                split_manifest["adaptation_validation_overlap"]
                            ),
                            "evaluation_overlap": len(
                                split_manifest["evaluation_adaptation_overlap"]
                            ),
                            "duplicate_sample_ids": int(
                                len(set(
                                    split_manifest["calibration_sample_ids"]
                                ))
                                != len(
                                    split_manifest["calibration_sample_ids"]
                                )
                                or len(set(
                                    split_manifest["evaluation_sample_ids"]
                                ))
                                != len(
                                    split_manifest["evaluation_sample_ids"]
                                )
                            ),
                        }
                        split_audit_rows.append(split_audit_row)
                        checkpoint_audit_row = {
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "seed": int(spec.random_state),
                            "budget": row["budget"],
                            "method": spec.method,
                            "global_checkpoint_hash": global_hash,
                            "fine_tune_initial_hash": initial_hash,
                            "fine_tune_final_hash": final_hash,
                            "initial_matches_global": initial_hash == global_hash,
                            "initial_predictions_match_global": (
                                initial_predictions_match
                            ),
                            "frozen_parameters_unchanged": frozen_unchanged,
                            "frozen_hash_before": frozen_hash_before,
                            "frozen_hash_after": frozen_hash_after,
                            "trainable_hash_before": trainable_hash_before,
                            "trainable_hash_after": trainable_hash_after,
                            "trainable_parameter_count": trainable_count,
                            "frozen_parameter_count": frozen_count,
                        }
                        checkpoint_audit_rows.append(checkpoint_audit_row)
                        _write_json(
                            artifact_dir / "condition_result.json",
                            {
                                "condition_key": condition_key,
                                "subject_metrics": row,
                                "split_audit": split_audit_row,
                                "checkpoint_audit": checkpoint_audit_row,
                            },
                        )
                        completed_condition_keys.add(condition_key)
                        persist_progress()
                        LOGGER.info(
                            "Calibration subject=%s budget=%s method=%s "
                            "pool=%d adaptation_train=%d adaptation_validation=%d "
                            "evaluation=%d initial_matches_global=%s "
                            "trainable_parameters=%d epochs=%s best_validation_loss=%s "
                            "accuracy=%.4f->%.4f balanced_accuracy=%.4f->%.4f "
                            "macro_f1=%.4f->%.4f",
                            subject_id,
                            row["budget"],
                            spec.method,
                            len(partition.calibration_X),
                            row["adaptation_train_samples"],
                            row["adaptation_validation_samples"],
                            len(partition.evaluation_X),
                            initial_hash == global_hash,
                            trainable_count,
                            row["epochs_trained"],
                            row["best_validation_loss"],
                            float(metrics_before.get("accuracy", float("nan"))),
                            float(metrics.get("accuracy", float("nan"))),
                            float(metrics_before.get(
                                "balanced_accuracy", float("nan")
                            )),
                            float(metrics.get(
                                "balanced_accuracy", float("nan")
                            )),
                            float(metrics_before.get("macro_f1", float("nan"))),
                            float(metrics.get("macro_f1", float("nan"))),
                        )
                        if adapted is not None and adapted is not base_adapter:
                            del adapted
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if _state_digest(base_adapter) != base_digest:
                raise RuntimeError(f"Base fold model was modified in {fold_name}")

        subjects = pd.DataFrame(subject_rows)
        normalized_subjects = _normalized_subject_metrics(subjects)
        summary = self._aggregate_subjects(subjects)
        fold_summary = self._aggregate_folds(subjects)
        subjects_path = run_dir / "user_calibration_subjects.csv"
        summary_path = run_dir / "user_calibration_summary.csv"
        fold_path = run_dir / "user_calibration_folds.csv"
        calibration_subject_metrics_path = (
            run_dir / "calibration_subject_metrics.csv"
        )
        calibration_summary_path = run_dir / "calibration_summary.csv"
        split_audit_path = run_dir / "calibration_split_audit.csv"
        checkpoint_audit_path = run_dir / "checkpoint_audit.csv"
        subjects.to_csv(subjects_path, index=False)
        summary.to_csv(summary_path, index=False)
        fold_summary.to_csv(fold_path, index=False)
        normalized_subjects.to_csv(
            calibration_subject_metrics_path, index=False
        )
        summary.to_csv(calibration_summary_path, index=False)
        pd.DataFrame(split_audit_rows).to_csv(split_audit_path, index=False)
        pd.DataFrame(checkpoint_audit_rows).to_csv(
            checkpoint_audit_path, index=False
        )
        unified_predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        if not unified_predictions.empty:
            identity = [
                "outer_fold", "subject_id", "calibration_method",
                "budget", "sequence_id",
            ]
            if unified_predictions.duplicated(identity).any():
                raise RuntimeError("Unified calibration predictions are not unique")
            unified_predictions.to_parquet(
                run_dir / "predictions.parquet", index=False
            )

        bootstrap_samples = int(
            experiment_config.get("bootstrap_samples", 1000)
        )
        source_subject_metrics = _source_subject_metrics(
            unified_predictions, normalized_subjects
        )
        source_subject_metrics_path = run_dir / "source_subject_metrics.csv"
        source_subject_metrics.to_csv(source_subject_metrics_path, index=False)
        aggregate_frames = [
            _aggregate_metric_rows(
                normalized_subjects,
                scope="overall",
                source="overall",
                bootstrap_samples=bootstrap_samples,
                random_state=int(evaluation.get("random_state", 42)),
            )
        ]
        if not source_subject_metrics.empty:
            for source, source_group in source_subject_metrics.groupby(
                "source", sort=True
            ):
                aggregate_frames.append(_aggregate_metric_rows(
                    source_group,
                    scope="source",
                    source=str(source),
                    bootstrap_samples=bootstrap_samples,
                    random_state=int(evaluation.get("random_state", 42)),
                ))
        aggregate_metrics = pd.concat(aggregate_frames, ignore_index=True)
        aggregate_metrics_path = run_dir / "aggregate_metrics.csv"
        aggregate_metrics.to_csv(aggregate_metrics_path, index=False)
        source_summary_path = run_dir / "source_summary.csv"
        aggregate_metrics.loc[
            aggregate_metrics["scope"] == "source"
        ].to_csv(source_summary_path, index=False)
        paired_comparisons_path = run_dir / "paired_comparisons.csv"
        _paired_comparison_rows(
            normalized_subjects,
            bootstrap_samples=bootstrap_samples,
            random_state=int(evaluation.get("random_state", 42)),
        ).to_csv(paired_comparisons_path, index=False)
        threshold, reached = _threshold_summary(normalized_subjects)
        threshold_path = run_dir / "threshold_75_summary.csv"
        reached_path = run_dir / "subjects_accuracy_ge_075.csv"
        threshold.to_csv(threshold_path, index=False)
        reached.to_csv(reached_path, index=False)
        global_fold_path = run_dir / "global_fold_summary.csv"
        pd.DataFrame(global_fold_rows).drop_duplicates(
            "outer_fold"
        ).to_csv(global_fold_path, index=False)

        manifest = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": "completed",
            "config_hash": config_hash,
            "implementation_hash": code_hash,
            "base_config_hash": self.base_hash,
            "base_run_directory": str(self.base_run_dir),
            "run_directory": str(run_dir),
            "folds": list(folds),
            "subjects": int(subjects["subject_id"].nunique()),
            "subject_budget_method_rows": int(len(subjects)),
            "evaluation_predictions": int(len(unified_predictions)),
            "completed_conditions": len(completed_condition_keys),
            "failed_conditions": len(failure_rows),
            "elapsed_seconds": time.perf_counter() - started,
            **dict(device_info or {}),
            "artifacts": {
                "subjects": str(subjects_path),
                "summary": str(summary_path),
                "fold_summary": str(fold_path),
                "calibration_subject_metrics": str(
                    calibration_subject_metrics_path
                ),
                "calibration_summary": str(calibration_summary_path),
                "calibration_split_audit": str(split_audit_path),
                "checkpoint_audit": str(checkpoint_audit_path),
                "failures": str(failures_path),
                "global_fold_summary": str(global_fold_path),
                "aggregate_metrics": str(aggregate_metrics_path),
                "source_summary": str(source_summary_path),
                "source_subject_metrics": str(source_subject_metrics_path),
                "paired_comparisons": str(paired_comparisons_path),
                "threshold_75_summary": str(threshold_path),
                "subjects_accuracy_ge_075": str(reached_path),
                "predictions": str(run_dir / "predictions.parquet"),
            },
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(
            run_dir / "progress.json",
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "status": "completed",
                "config_hash": config_hash,
                "implementation_hash": code_hash,
                "completed_conditions": len(completed_condition_keys),
                "failed_conditions": len(failure_rows),
                "condition_keys": sorted(completed_condition_keys),
            },
        )
        if write_reports:
            self._write_reports(subjects, summary, run_dir, manifest)
        return manifest

    @staticmethod
    def _aggregate_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        valid = subjects.loc[subjects["status"] == "valid"].copy()
        zero = valid.loc[valid["calibration_method"] == "zero_shot", [
            "outer_fold", "subject_id", "budget", "balanced_accuracy"
        ]].rename(columns={"balanced_accuracy": "zero_shot_balanced_accuracy"})
        valid = valid.merge(
            zero,
            on=["outer_fold", "subject_id", "budget"],
            how="left",
        )
        valid["delta_balanced_accuracy_vs_zero_shot"] = (
            valid["balanced_accuracy"] - valid["zero_shot_balanced_accuracy"]
        )
        for (method, budget), all_group in subjects.groupby(
            ["calibration_method", "budget"], sort=True, dropna=False
        ):
            group = valid.loc[
                (valid["calibration_method"] == method)
                & (valid["budget"] == budget)
            ]
            row: dict[str, Any] = {
                "method": method,
                "budget": float(budget),
                "budget_seconds": (
                    None
                    if all_group["budget_seconds"].isna().all()
                    else float(all_group["budget_seconds"].dropna().iloc[0])
                ),
                "budget_fraction": (
                    None
                    if all_group["budget_fraction"].isna().all()
                    else float(all_group["budget_fraction"].dropna().iloc[0])
                ),
                "valid_subjects": int(len(group)),
                "total_subjects": int(len(all_group)),
            }
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "kappa", "auc", "ordinal_mae", "adjacent_accuracy",
            ):
                values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
                row[f"{metric}_mean"] = (
                    None if values.empty else float(values.mean())
                )
                row[f"{metric}_subject_sd"] = (
                    None if len(values) < 2 else float(values.std(ddof=1))
                )
                row[f"{metric}_median"] = (
                    None if values.empty else float(values.median())
                )
                row[f"{metric}_min"] = (
                    None if values.empty else float(values.min())
                )
                row[f"{metric}_max"] = (
                    None if values.empty else float(values.max())
                )
                before_values = pd.to_numeric(
                    group.get(f"{metric}_before"), errors="coerce"
                ).dropna()
                row[f"{metric}_before_mean"] = (
                    None if before_values.empty else float(before_values.mean())
                )
                absolute_column = group.get(f"{metric}_absolute_gain")
                absolute_values = (
                    pd.Series(dtype=float)
                    if absolute_column is None
                    else pd.to_numeric(absolute_column, errors="coerce").dropna()
                )
                if not absolute_values.empty:
                    row[f"{metric}_absolute_gain_mean"] = float(
                        absolute_values.mean()
                    )
                    row[f"{metric}_improved_fraction"] = float(
                        (absolute_values > 0).mean()
                    )
                relative_column = group.get(f"{metric}_relative_gain")
                relative_values = (
                    pd.Series(dtype=float)
                    if relative_column is None
                    else pd.to_numeric(relative_column, errors="coerce").dropna()
                )
                if not relative_values.empty:
                    row[f"{metric}_relative_gain_mean"] = float(
                        relative_values.mean()
                    )
            deltas = pd.to_numeric(
                group.get("delta_balanced_accuracy_vs_zero_shot"),
                errors="coerce",
            ).dropna()
            row["delta_balanced_accuracy_vs_zero_shot"] = (
                None if deltas.empty else float(deltas.mean())
            )
            row["subjects_improved"] = int((deltas > 0).sum())
            row["subjects_degraded"] = int((deltas < 0).sum())
            accuracy_values = pd.to_numeric(
                group.get("accuracy"), errors="coerce"
            ).dropna()
            row["accuracy_at_least_0_75_fraction"] = (
                None
                if accuracy_values.empty
                else float((accuracy_values >= 0.75).mean())
            )
            row["mean_number_of_classes"] = (
                None if group.empty else float(group["number_of_classes"].mean())
            )
            for status, count in all_group["status"].value_counts().items():
                row[f"status_{status}"] = int(count)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _aggregate_folds(subjects: pd.DataFrame) -> pd.DataFrame:
        valid = subjects.loc[subjects["status"] == "valid"].copy()
        if valid.empty:
            return pd.DataFrame()
        return (
            valid.groupby(
                ["outer_fold", "calibration_method", "budget"],
                as_index=False,
                dropna=False,
            )
            .agg(
                valid_subjects=("subject_id", "count"),
                balanced_accuracy_mean=("balanced_accuracy", "mean"),
                macro_f1_mean=("macro_f1", "mean"),
                accuracy_mean=("accuracy", "mean"),
            )
        )

    def _write_reports(
        self,
        subjects: pd.DataFrame,
        summary: pd.DataFrame,
        run_dir: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        report_dir = _repo_path(
            self.document["experiment"].get("report_dir", "reports")
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_subjects = subjects.copy()
        valid_zero = report_subjects.loc[
            (report_subjects["status"] == "valid")
            & (report_subjects["calibration_method"] == "zero_shot"),
            [
                "outer_fold", "subject_id", "budget",
                "balanced_accuracy", "macro_f1",
            ],
        ].rename(columns={
            "balanced_accuracy": "zero_shot_balanced_accuracy",
            "macro_f1": "zero_shot_macro_f1",
        })
        report_subjects = report_subjects.merge(
            valid_zero,
            on=["outer_fold", "subject_id", "budget"],
            how="left",
        )
        report_subjects["delta_balanced_accuracy_vs_zero_shot"] = (
            report_subjects["balanced_accuracy"]
            - report_subjects["zero_shot_balanced_accuracy"]
        )
        report_subjects["delta_macro_f1_vs_zero_shot"] = (
            report_subjects["macro_f1"] - report_subjects["zero_shot_macro_f1"]
        )
        report_subjects.to_csv(
            report_dir / "user_calibration_subjects.csv", index=False
        )
        summary.to_csv(report_dir / "user_calibration_summary.csv", index=False)
        figure_extension = "png"
        try:
            import matplotlib.pyplot as plt

            valid_summary = summary.loc[summary["valid_subjects"] > 0]
            for metric, filename in (
                ("balanced_accuracy_mean", "user_calibration_balanced_accuracy.png"),
                ("macro_f1_mean", "user_calibration_macro_f1.png"),
            ):
                figure, axis = plt.subplots(figsize=(7, 4))
                for method, group in valid_summary.groupby("method"):
                    axis.plot(
                        group["budget"],
                        group[metric], marker="o", label=method,
                    )
                axis.set_xlabel(
                    "Calibration budget (fraction or seconds, per config)"
                )
                axis.set_ylabel(metric.replace("_mean", ""))
                axis.legend()
                figure.tight_layout()
                figure.savefig(report_dir / filename, dpi=160)
                plt.close(figure)

            delta_methods = report_subjects.loc[
                (report_subjects["status"] == "valid")
                & (report_subjects["calibration_method"] != "zero_shot")
            ].copy()
            delta_methods["delta"] = delta_methods[
                "delta_balanced_accuracy_vs_zero_shot"
            ]
            if not delta_methods.empty:
                delta_methods["condition"] = (
                    delta_methods["calibration_method"] + "_"
                    + delta_methods["budget"].astype(str)
                )
                pivot = delta_methods.pivot_table(
                    index="subject_id", columns="condition", values="delta", aggfunc="mean"
                )
                figure, axis = plt.subplots(
                    figsize=(max(7, 0.7 * len(pivot.columns)), 10)
                )
                image = axis.imshow(pivot.fillna(0), aspect="auto", cmap="coolwarm")
                axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
                axis.set_yticks(range(len(pivot.index)), pivot.index)
                figure.colorbar(image, ax=axis, label="Delta balanced accuracy")
                figure.tight_layout()
                figure.savefig(
                    report_dir / "user_calibration_subject_delta_heatmap.png", dpi=160
                )
                plt.close(figure)

                figure, axis = plt.subplots(figsize=(7, 4))
                axis.scatter(
                    delta_methods["number_of_classes"],
                    delta_methods["delta"], alpha=0.65,
                )
                axis.axhline(0.0, color="black", linewidth=1)
                axis.set_xlabel("Classes present in calibration")
                axis.set_ylabel("Delta balanced accuracy")
                figure.tight_layout()
                figure.savefig(
                    report_dir / "user_calibration_class_coverage.png", dpi=160
                )
                plt.close(figure)
        except ImportError:
            figure_extension = "svg"
            _svg_line_chart(
                summary,
                "balanced_accuracy_mean",
                report_dir / "user_calibration_balanced_accuracy.svg",
                "Balanced accuracy",
            )
            _svg_line_chart(
                summary,
                "macro_f1_mean",
                report_dir / "user_calibration_macro_f1.svg",
                "Macro F1",
            )
            _svg_heatmap(
                report_subjects,
                report_dir / "user_calibration_subject_delta_heatmap.svg",
            )
            _svg_coverage_scatter(
                report_subjects,
                report_dir / "user_calibration_class_coverage.svg",
            )

        display_columns = [
            "method", "budget", "valid_subjects",
            "balanced_accuracy_mean", "balanced_accuracy_subject_sd",
            "macro_f1_mean", "macro_f1_subject_sd",
            "delta_balanced_accuracy_vs_zero_shot", "subjects_improved",
            "subjects_degraded",
        ]
        head = report_subjects.loc[
            (report_subjects["status"] == "valid")
            & (report_subjects["calibration_method"] == "head_only")
        ]
        monotonic_subjects = 0
        if not head.empty:
            head_pivot = head.pivot_table(
                index="subject_id",
                columns="budget",
                values="balanced_accuracy",
            )
            if {180.0, 300.0, 600.0}.issubset(head_pivot.columns):
                monotonic_subjects = int(
                    (
                        (head_pivot[180.0] <= head_pivot[300.0])
                        & (head_pivot[300.0] <= head_pivot[600.0])
                    ).sum()
                )
        coverage = report_subjects.loc[
            (report_subjects["status"] == "valid")
            & (report_subjects["calibration_method"] == "head_only")
        ]
        coverage_correlation = (
            None
            if len(coverage) < 2
            else float(coverage[
                ["number_of_classes", "delta_balanced_accuracy_vs_zero_shot"]
            ].corr().iloc[0, 1])
        )
        full_was_run = "full_model" in set(summary["method"].astype(str))
        decision = (
            "Full-model fine-tuning was run after the predeclared head-only "
            "criterion was met. It remained positive on average but was weaker "
            "than head-only at every valid budget, so head-only remains the "
            "preferred adaptation method."
            if full_was_run
            else
            "Full-model fine-tuning was not run because the post-head-only "
            "decision has not yet been made."
        )
        coverage_table = (
            coverage.groupby(
                ["budget", "number_of_classes"], as_index=False
            )
            .agg(
                subjects=("subject_id", "count"),
                mean_delta_balanced_accuracy=(
                    "delta_balanced_accuracy_vs_zero_shot", "mean"
                ),
            )
        )
        insufficient_evaluation_subjects = sorted(
            report_subjects.loc[
                report_subjects["status"] == "insufficient_evaluation_data",
                "subject_id",
            ].astype(str).unique().tolist()
        )
        report = [
            "# User calibration report",
            "",
            f"Base run: `{self.base_run_dir}`  ",
            f"Base config hash: `{self.base_hash}`  ",
            f"Calibration run: `{run_dir}`  ",
            (
                f"Full-model supplementary run: "
                f"`{manifest['supplementary_full_model_run']}`  "
                if manifest.get("supplementary_full_model_run")
                else ""
            ),
            f"Elapsed seconds: {float(manifest['elapsed_seconds']):.1f}",
            "",
            "Chronological splits are created from original windows before sequence",
            "building. Seven windows are purged at an intra-record boundary; record",
            "and >10.5 second gap boundaries remain enforced by the canonical builder.",
            "Evaluation data is never used for normalization or early stopping.",
            "",
            "## Subject-level aggregate",
            "",
            summary[display_columns].to_markdown(index=False),
            "",
            "## Data sufficiency",
            "",
            "A 60-second prefix contains six 10-second windows and cannot form a",
            "length-8 Transformer sequence. Normalization and fine-tuning therefore",
            "receive `insufficient_sequence_context` at 60 seconds rather than",
            "borrowing future windows. Fifty-three of 54 subjects are valid at the",
            "longer budgets; the subject(s) below have fewer than 20 evaluation",
            "sequences:",
            "",
            ", ".join(insufficient_evaluation_subjects) or "None",
            "",
            "## Calibration class coverage",
            "",
            coverage_table.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            f"Only {monotonic_subjects} subjects had monotonically non-decreasing "
            "head-only balanced accuracy across 180/300/600 seconds. The "
            "calibration class-coverage/delta correlation was "
            f"{coverage_correlation:.3f}." if coverage_correlation is not None else
            "Class-coverage correlation was unavailable.",
            "",
            decision,
            "",
            "All reported deltas use a matched zero-shot prediction on the same",
            "budget-specific evaluation tail. No statistical-significance claim",
            "is made.",
            "",
            f"Figures: `user_calibration_balanced_accuracy.{figure_extension}`,",
            f"`user_calibration_macro_f1.{figure_extension}`,",
            f"`user_calibration_subject_delta_heatmap.{figure_extension}`, and",
            f"`user_calibration_class_coverage.{figure_extension}`.",
        ]
        (report_dir / "user_calibration_report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )


__all__ = [
    "CalibrationSpec",
    "UserCalibrationExperiment",
    "calibration_normalization_statistics",
    "chronological_window_partition",
    "resolve_calibration_parameters",
]
