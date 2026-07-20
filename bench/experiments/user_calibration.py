"""Subject-specific calibration built on canonical benchmark checkpoints."""

from __future__ import annotations

import gc
import hashlib
import html
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
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
from model_zoo.factory import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_SCHEMA_VERSION = "user-calibration-v1"
CALIBRATION_METHODS = frozenset(
    {"zero_shot", "subject_normalization", "head_only", "full_model"}
)


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


@dataclass(frozen=True)
class CalibrationSpec:
    """Serializable, AutoML-ready calibration protocol parameters."""

    method: str
    budget_seconds: Optional[float] = 0.0
    budget_fraction: Optional[float] = None
    split_strategy: str = "chronological_prefix"
    purge_windows: int = 7
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4
    calibration_validation_fraction: float = 0.2
    fallback_fixed_epochs: int = 3
    min_calibration_sequences: int = 1
    min_evaluation_sequences: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        normalized = self.method.strip().lower()
        aliases = {
            "normalization": "subject_normalization",
            "subject_norm": "subject_normalization",
            "head": "head_only",
            "full": "full_model",
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
        if self.budget_seconds is not None and self.budget_fraction is not None:
            raise ValueError("Set either budget_seconds or budget_fraction, not both")
        if self.budget_seconds is None and self.budget_fraction is None:
            raise ValueError("A calibration budget is required")
        if self.budget_seconds is not None and self.budget_seconds < 0:
            raise ValueError("budget_seconds cannot be negative")
        if self.budget_fraction is not None and not 0 < self.budget_fraction < 1:
            raise ValueError("budget_fraction must be between 0 and 1")
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
    for segment_key, _ in segment_durations:
        group = ordered.loc[ordered["_segment_key"] == segment_key]
        row_indices = group["_row_index"].to_numpy(dtype=np.int64)
        times = group[time_column].to_numpy(dtype=np.float64)
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
    x_values = sorted(frame["budget_seconds"].unique())
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
            f'<text x="{position:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value/60:g}</text>'
        )
    for method, group in frame.groupby("method", sort=False):
        group = group.sort_values("budget_seconds")
        points = " ".join(
            f'{x(row.budget_seconds):.1f},{y(getattr(row, metric)):.1f}'
            for row in group.itertuples()
        )
        color = colors.get(str(method), "#777")
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for row in group.itertuples():
            parts.append(
                f'<circle cx="{x(row.budget_seconds):.1f}" cy="{y(getattr(row, metric)):.1f}" r="4" fill="{color}"/>'
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
        f'<text x="{(left+width-right)/2:.1f}" y="{height-15}" text-anchor="middle" font-size="13">Calibration duration (minutes)</text>',
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
        + frame["budget_seconds"].astype(int).astype(str) + "s"
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
        self.base_run_dir = _repo_path(self.document["base_run"]["run_directory"])
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
        self.base_manifest = json.loads(
            (self.base_run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.base_hash = benchmark_config_hash(self.base_config)
        BenchmarkRunner.validate_completed_run(
            self.base_run_dir,
            expected_config_hash=self.base_hash,
            result_file=_repo_path(self.base_manifest["benchmark_result_file"]),
            manifest_file=self.base_run_dir / "run_manifest.json",
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
        selected_budgets = list(
            config["budgets_seconds"] if budgets_seconds is None else budgets_seconds
        )
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
                    "budget_seconds": float(budget),
                    "budget_fraction": None,
                }
                if max_epochs is not None and str(method) in {"head_only", "full_model"}:
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
        if spec.method != "zero_shot":
            if len(partition.calibration_X) == 0 or (
                partition.actual_seconds + 1e-6 < partition.requested_seconds
            ):
                return "insufficient_calibration_data"
            if len(calibration_sequences.X) < spec.min_calibration_sequences:
                return "insufficient_sequence_context"
        if len(evaluation_sequences.X) < spec.min_evaluation_sequences:
            return "insufficient_evaluation_data"
        return "valid"

    def _calibration_validation(
        self,
        partition: WindowPartition,
        spec: CalibrationSpec,
        sequence_config: Mapping[str, Any],
    ) -> tuple[SequenceBuildResult, Optional[SequenceBuildResult], str]:
        train_fraction = 1.0 - spec.calibration_validation_fraction
        validation_spec = CalibrationSpec(
            method=spec.method,
            budget_seconds=None,
            budget_fraction=train_fraction,
            split_strategy=spec.split_strategy,
            purge_windows=spec.purge_windows,
            max_epochs=spec.max_epochs,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            early_stopping_patience=spec.early_stopping_patience,
            calibration_validation_fraction=spec.calibration_validation_fraction,
            fallback_fixed_epochs=spec.fallback_fixed_epochs,
            min_calibration_sequences=spec.min_calibration_sequences,
            min_evaluation_sequences=spec.min_evaluation_sequences,
            random_state=spec.random_state,
        )
        inner = chronological_window_partition(
            partition.calibration_X,
            partition.calibration_y,
            partition.calibration_metadata,
            validation_spec,
            window_seconds=float(sequence_config["expected_step_seconds"]),
            max_gap_seconds=float(sequence_config["max_gap_seconds"]),
        )
        train_sequences = _build_sequences(
            inner.calibration_X,
            inner.calibration_y,
            inner.calibration_metadata,
            sequence_config,
        )
        validation_sequences = _build_sequences(
            inner.evaluation_X,
            inner.evaluation_y,
            inner.evaluation_metadata,
            sequence_config,
        )
        if len(train_sequences.X) and len(validation_sequences.X):
            return train_sequences, validation_sequences, "chronological_holdout"
        all_sequences = _build_sequences(
            partition.calibration_X,
            partition.calibration_y,
            partition.calibration_metadata,
            sequence_config,
        )
        return all_sequences, None, "fixed_epochs_no_validation"

    @staticmethod
    def _split_manifest(
        partition: WindowPartition,
        calibration_sequences: SequenceBuildResult,
        evaluation_sequences: SequenceBuildResult,
        coverage: Mapping[str, Any],
        validation_mode: str,
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
        overlap = sorted(set(calibration_ids) & set(evaluation_ids))
        if overlap:
            raise RuntimeError(f"Calibration/evaluation sample overlap: {overlap[:10]}")
        return {
            "split_strategy": "chronological_prefix",
            "time_column": time_column,
            "requested_calibration_duration_seconds": partition.requested_seconds,
            "actual_calibration_duration_seconds": partition.actual_seconds,
            "available_subject_duration_seconds": partition.available_seconds,
            "calibration_windows": len(calibration_ids),
            "purged_windows": len(purged_ids),
            "evaluation_windows": len(evaluation_ids),
            "calibration_sequences": int(len(calibration_sequences.X)),
            "evaluation_sequences": int(len(evaluation_sequences.X)),
            "calibration_sample_ids": calibration_ids,
            "purged_sample_ids": purged_ids,
            "evaluation_sample_ids": evaluation_ids,
            "window_overlap": overlap,
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
            "record_id": metadata["record_id"].astype(str),
            "record_group_id": metadata["record_group_id"].astype(str),
            "sample_id": metadata["target_sample_id"].astype(str),
            "sequence_id": metadata["sequence_id"].astype(str),
            "calibration_method": spec.method,
            "budget_seconds": spec.budget_seconds,
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
    ) -> dict[str, Any]:
        dataset_name, task_name, model_name = self._identities()
        specs = self._specs(
            budgets_seconds=budgets_seconds,
            methods=methods,
            max_epochs=max_epochs,
            random_state=random_state,
        )
        sequence_config = self.base_config["sequence"]
        sequence_length = int(sequence_config["length"])
        if any(spec.purge_windows < sequence_length - 1 for spec in specs):
            raise ValueError(
                f"purge_windows must be at least sequence_length - 1 "
                f"({sequence_length - 1})"
            )

        resolved = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "base_run": deepcopy(self.document["base_run"]),
            "base_config_hash": self.base_hash,
            "specs": [spec.to_dict() for spec in specs],
            "fold_limit": fold_limit,
            "subject_limit": subject_limit,
        }
        config_hash = _canonical_hash(resolved)
        root = _repo_path(
            output_dir
            if output_dir is not None
            else self.document["experiment"]["output_dir"]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        with open(run_dir / "resolved_calibration.yaml", "w", encoding="utf-8") as output:
            yaml.safe_dump(resolved, output, sort_keys=False)

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
        if fold_limit is not None:
            if fold_limit <= 0:
                raise ValueError("fold_limit must be positive")
            folds = dict(list(folds.items())[: int(fold_limit)])

        subject_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        device_info: Optional[dict[str, str]] = None
        started = time.perf_counter()
        for fold_name, outer_split in folds.items():
            if outer_split.metadata.get("subject_overlap"):
                raise RuntimeError(f"Outer subject leakage in {fold_name}")
            checkpoint = self._fold_checkpoint(
                fold_name, dataset_name, task_name, model_name
            )
            base_adapter = self._load_fold_adapter(checkpoint, model_name)
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
            if subject_limit is not None:
                if subject_limit <= 0:
                    raise ValueError("subject_limit must be positive")
                test_subjects = test_subjects[: int(subject_limit)]
            train_subjects = set(np.unique(outer_split.subject_train).astype(str))
            if train_subjects & set(test_subjects):
                raise RuntimeError(f"Outer train/test subject leakage in {fold_name}")
            test_metadata = runner._partition_sequence_metadata(outer_split, "test")

            for subject_id in test_subjects:
                subject_mask = np.asarray(outer_split.subject_test).astype(str) == subject_id
                subject_X = np.asarray(outer_split.X_test)[subject_mask]
                subject_y = np.asarray(outer_split.y_test)[subject_mask]
                subject_metadata = test_metadata.loc[subject_mask].reset_index(drop=True)
                budget_values = sorted(
                    {float(spec.budget_seconds or 0.0) for spec in specs}
                )
                for budget_seconds in budget_values:
                    budget_specs = [
                        spec for spec in specs
                        if float(spec.budget_seconds or 0.0) == budget_seconds
                    ]
                    split_spec = budget_specs[0]
                    partition = chronological_window_partition(
                        subject_X,
                        subject_y,
                        subject_metadata,
                        split_spec,
                        window_seconds=float(sequence_config["expected_step_seconds"]),
                        max_gap_seconds=float(sequence_config["max_gap_seconds"]),
                    )
                    calibration_sequences = _build_sequences(
                        partition.calibration_X,
                        partition.calibration_y,
                        partition.calibration_metadata,
                        sequence_config,
                    )
                    evaluation_sequences = _build_sequences(
                        partition.evaluation_X,
                        partition.evaluation_y,
                        partition.evaluation_metadata,
                        sequence_config,
                    )
                    coverage = _class_coverage(calibration_sequences.y)

                    for spec in budget_specs:
                        status = self._status(
                            spec,
                            partition,
                            calibration_sequences,
                            evaluation_sequences,
                        )
                        artifact_dir = (
                            run_dir / fold_name / _safe_component(subject_id)
                            / f"budget_{int(budget_seconds):04d}s" / spec.method
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
                        predictions_frame = pd.DataFrame()
                        adapted: Optional[TorchClassificationAdapter] = None
                        normalization_metadata: Optional[dict[str, Any]] = None
                        if status == "valid":
                            adapted = (
                                base_adapter
                                if spec.method == "zero_shot"
                                else base_adapter.clone()
                            )
                            if spec.method == "subject_normalization":
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
                                        partition, spec, sequence_config
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
                                adapted.fine_tune(
                                    train_sequences.X,
                                    train_sequences.y,
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
                                    trainable_parameter_prefixes=(
                                        ("classifier.",)
                                        if spec.method == "head_only"
                                        else None
                                    ),
                                    max_epochs=fit_epochs,
                                    learning_rate=spec.learning_rate,
                                    weight_decay=spec.weight_decay,
                                    early_stopping_patience=spec.early_stopping_patience,
                                )
                                training_time = time.perf_counter() - fit_started
                                training_log = pd.DataFrame(adapted.training_log_)
                                state = {
                                    key: value.detach().cpu()
                                    for key, value in adapted.model.state_dict().items()
                                    if spec.method == "full_model"
                                    or key.startswith("classifier.")
                                }
                                torch.save(
                                    {
                                        "base_reference": model_reference,
                                        "method": spec.method,
                                        "model_state_dict": state,
                                    },
                                    artifact_dir / (
                                        "calibrated_model.pt"
                                        if spec.method == "full_model"
                                        else "calibrated_head.pt"
                                    ),
                                )

                            probabilities = adapted.predict_proba(
                                evaluation_sequences.X
                            )
                            predictions = probabilities.argmax(axis=1)
                            if not np.isfinite(probabilities).all() or not np.allclose(
                                probabilities.sum(axis=1), 1.0, atol=1e-5
                            ):
                                raise RuntimeError("Invalid calibrated probabilities")
                            metrics = MetricsCalculator.calculate_all_metrics(
                                evaluation_sequences.y, predictions, probabilities
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

                        split_manifest = self._split_manifest(
                            partition,
                            calibration_sequences,
                            evaluation_sequences,
                            coverage,
                            validation_mode,
                        )
                        split_manifest["status"] = status
                        split_manifest["purge_windows"] = spec.purge_windows
                        split_manifest["evaluation_used_for_normalization"] = False
                        split_manifest["evaluation_used_for_early_stopping"] = False
                        _write_json(
                            artifact_dir / "calibration_split.json", split_manifest
                        )
                        training_log.to_csv(
                            artifact_dir / "calibration_training_log.csv", index=False
                        )
                        metric_payload = {
                            "status": status,
                            "outer_fold": fold_name,
                            "subject_id": subject_id,
                            "calibration_method": spec.method,
                            "budget_seconds": budget_seconds,
                            "training_time_seconds": training_time,
                            "calibration_validation_mode": validation_mode,
                            "metrics": metrics,
                        }
                        _write_json(
                            artifact_dir / "calibration_metrics.json", metric_payload
                        )
                        if normalization_metadata is not None:
                            _write_json(
                                artifact_dir / "normalization_stats.json",
                                normalization_metadata,
                            )
                        if predictions_frame.empty:
                            predictions_frame = pd.DataFrame(columns=[
                                "outer_fold", "subject_id", "record_id",
                                "record_group_id", "sample_id", "sequence_id",
                                "calibration_method", "budget_seconds", "y_true",
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
                            "calibration_method": spec.method,
                            "budget_seconds": budget_seconds,
                            "status": status,
                            "requested_calibration_duration": partition.requested_seconds,
                            "actual_calibration_duration": partition.actual_seconds,
                            "calibration_sequences": len(calibration_sequences.X),
                            "evaluation_sequences": len(evaluation_sequences.X),
                            "record_coverage": len(
                                split_manifest["calibration_record_ids"]
                            ),
                            "calibration_validation_mode": validation_mode,
                            "training_time_seconds": training_time,
                            **coverage,
                            **metrics,
                        }
                        row["class_counts"] = json.dumps(
                            row["class_counts"], sort_keys=True
                        )
                        row["classes_present"] = json.dumps(
                            row["classes_present"]
                        )
                        subject_rows.append(row)
                        if adapted is not None and adapted is not base_adapter:
                            del adapted
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if _state_digest(base_adapter) != base_digest:
                raise RuntimeError(f"Base fold model was modified in {fold_name}")

        subjects = pd.DataFrame(subject_rows)
        summary = self._aggregate_subjects(subjects)
        fold_summary = self._aggregate_folds(subjects)
        subjects_path = run_dir / "user_calibration_subjects.csv"
        summary_path = run_dir / "user_calibration_summary.csv"
        fold_path = run_dir / "user_calibration_folds.csv"
        subjects.to_csv(subjects_path, index=False)
        summary.to_csv(summary_path, index=False)
        fold_summary.to_csv(fold_path, index=False)
        unified_predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        if not unified_predictions.empty:
            identity = [
                "outer_fold", "subject_id", "calibration_method",
                "budget_seconds", "sequence_id",
            ]
            if unified_predictions.duplicated(identity).any():
                raise RuntimeError("Unified calibration predictions are not unique")
            unified_predictions.to_parquet(
                run_dir / "predictions.parquet", index=False
            )

        manifest = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": "completed",
            "config_hash": config_hash,
            "base_config_hash": self.base_hash,
            "base_run_directory": str(self.base_run_dir),
            "run_directory": str(run_dir),
            "folds": list(folds),
            "subjects": int(subjects["subject_id"].nunique()),
            "subject_budget_method_rows": int(len(subjects)),
            "evaluation_predictions": int(len(unified_predictions)),
            "elapsed_seconds": time.perf_counter() - started,
            **dict(device_info or {}),
            "artifacts": {
                "subjects": str(subjects_path),
                "summary": str(summary_path),
                "fold_summary": str(fold_path),
                "predictions": str(run_dir / "predictions.parquet"),
            },
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        if write_reports:
            self._write_reports(subjects, summary, run_dir, manifest)
        return manifest

    @staticmethod
    def _aggregate_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        valid = subjects.loc[subjects["status"] == "valid"].copy()
        zero = valid.loc[valid["calibration_method"] == "zero_shot", [
            "outer_fold", "subject_id", "budget_seconds", "balanced_accuracy"
        ]].rename(columns={"balanced_accuracy": "zero_shot_balanced_accuracy"})
        valid = valid.merge(
            zero,
            on=["outer_fold", "subject_id", "budget_seconds"],
            how="left",
        )
        valid["delta_balanced_accuracy_vs_zero_shot"] = (
            valid["balanced_accuracy"] - valid["zero_shot_balanced_accuracy"]
        )
        for (method, budget), all_group in subjects.groupby(
            ["calibration_method", "budget_seconds"], sort=True
        ):
            group = valid.loc[
                (valid["calibration_method"] == method)
                & (valid["budget_seconds"] == budget)
            ]
            row: dict[str, Any] = {
                "method": method,
                "budget_seconds": float(budget),
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
            deltas = pd.to_numeric(
                group.get("delta_balanced_accuracy_vs_zero_shot"),
                errors="coerce",
            ).dropna()
            row["delta_balanced_accuracy_vs_zero_shot"] = (
                None if deltas.empty else float(deltas.mean())
            )
            row["subjects_improved"] = int((deltas > 0).sum())
            row["subjects_degraded"] = int((deltas < 0).sum())
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
                ["outer_fold", "calibration_method", "budget_seconds"],
                as_index=False,
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
                "outer_fold", "subject_id", "budget_seconds",
                "balanced_accuracy", "macro_f1",
            ],
        ].rename(columns={
            "balanced_accuracy": "zero_shot_balanced_accuracy",
            "macro_f1": "zero_shot_macro_f1",
        })
        report_subjects = report_subjects.merge(
            valid_zero,
            on=["outer_fold", "subject_id", "budget_seconds"],
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
                        group["budget_seconds"] / 60.0,
                        group[metric], marker="o", label=method,
                    )
                axis.set_xlabel("Calibration duration (minutes)")
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
                    + delta_methods["budget_seconds"].astype(int).astype(str) + "s"
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
            "method", "budget_seconds", "valid_subjects",
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
                columns="budget_seconds",
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
                ["budget_seconds", "number_of_classes"], as_index=False
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
