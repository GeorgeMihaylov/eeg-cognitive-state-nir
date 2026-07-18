"""Temporal target diagnostics that never use EEG or POW features."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import yaml

from bench.analysis.label_target_audit import (
    _jsonable,
    _repo_path,
    _sha256_file,
    _write_json,
)
from bench.validation.metrics import MetricsCalculator


RECORD_COLUMNS = ["source", "subject_id", "record_id"]
ORDER_COLUMNS = [*RECORD_COLUMNS, "t_start", "sample_id"]
REQUIRED_COLUMNS = [
    "source",
    "subject_id",
    "record_id",
    "t_start",
    "t_end",
    "target_focus",
    "label_q5",
]
DIAGNOSTIC_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ordinal_mae",
    "adjacent_accuracy",
    "severe_error_rate",
)


def prepare_temporal_frame(
    frame: pd.DataFrame,
    *,
    target_col: str = "target_focus",
    label_col: str = "label_q5",
    n_classes: int = 5,
) -> pd.DataFrame:
    """Create stable record-local order and leakage-safe position covariates.

    Position is calculated on every source row before missing targets are removed.
    The synthetic ``sample_id`` exactly follows the benchmark loader convention for
    a Parquet without a stored ID: the original zero-based row position.
    """

    required = set(REQUIRED_COLUMNS) | {target_col, label_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal audit input is missing columns: {missing}")
    work = frame.copy()
    if "sample_id" not in work:
        work["sample_id"] = np.arange(len(work), dtype=np.int64)
    if work["sample_id"].isna().any() or not work["sample_id"].is_unique:
        raise ValueError("sample_id must be complete and unique before sequencing")
    for column in ("t_start", "t_end", target_col, label_col):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in RECORD_COLUMNS:
        if work[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")
        work[column] = work[column].astype(str)
    if work["t_start"].isna().any() or work["t_end"].isna().any():
        raise ValueError("t_start and t_end must be numeric and complete")

    work = work.sort_values(ORDER_COLUMNS, kind="mergesort").reset_index(drop=True)
    grouped = work.groupby(RECORD_COLUMNS, sort=False, observed=True)
    work["absolute_window_index"] = grouped.cumcount().astype(np.int64)
    record_windows = grouped["sample_id"].transform("size").astype(np.int64)
    denominator = (record_windows - 1).replace(0, 1)
    work["normalized_record_progress"] = (
        work["absolute_window_index"] / denominator
    ).astype(float)
    work["record_duration"] = (
        grouped["t_end"].transform("max")
        - grouped["t_start"].transform("min")
    ).astype(float)
    work["record_windows"] = record_windows

    target_missing = work[target_col].isna()
    label_missing = work[label_col].isna()
    if not np.array_equal(target_missing.to_numpy(), label_missing.to_numpy()):
        raise ValueError("target_focus and label_q5 missing-value masks differ")
    supervised = work.loc[~target_missing].copy()
    labels = supervised[label_col].to_numpy(dtype=float)
    if not np.all(labels == np.floor(labels)):
        raise ValueError("label_q5 contains non-integer values")
    observed = sorted(np.unique(labels.astype(int)).tolist())
    if observed != list(range(n_classes)):
        raise ValueError(
            f"Expected label classes {list(range(n_classes))}, observed {observed}"
        )
    supervised[label_col] = labels.astype(np.int64)
    return supervised.reset_index(drop=True)


def make_lag_pairs(
    frame: pd.DataFrame,
    *,
    value_col: str,
    lag: int,
) -> pd.DataFrame:
    """Return record-local past/current pairs; a positive lag never sees future rows."""

    if lag <= 0:
        raise ValueError("lag must be positive")
    ordered = frame.sort_values(ORDER_COLUMNS, kind="mergesort").copy()
    grouped = ordered.groupby(RECORD_COLUMNS, sort=False, observed=True)
    ordered["previous_value"] = grouped[value_col].shift(lag)
    ordered["previous_sample_id"] = grouped["sample_id"].shift(lag)
    ordered["previous_window_index"] = grouped["absolute_window_index"].shift(lag)
    mask = (
        ordered["previous_value"].notna()
        & (
            ordered["absolute_window_index"]
            - ordered["previous_window_index"]
            == lag
        )
    )
    pairs = ordered.loc[mask].copy()
    pairs["current_value"] = pairs[value_col]
    pairs["lag"] = int(lag)
    if not pairs.empty:
        current_keys = pairs[RECORD_COLUMNS].astype(str).agg("\x1f".join, axis=1)
        previous_lookup = ordered.set_index("sample_id")[RECORD_COLUMNS]
        previous_keys = previous_lookup.loc[
            pairs["previous_sample_id"].to_numpy()
        ].astype(str).agg("\x1f".join, axis=1).to_numpy()
        if not np.array_equal(current_keys.to_numpy(), previous_keys):
            raise RuntimeError("Lag pairs crossed a source, subject, or record boundary")
    return pairs.reset_index(drop=True)


def _correlation(pairs: pd.DataFrame) -> float | None:
    if len(pairs) < 2:
        return None
    previous = pairs["previous_value"].to_numpy(dtype=float)
    current = pairs["current_value"].to_numpy(dtype=float)
    if np.std(previous) == 0.0 or np.std(current) == 0.0:
        return None
    value = float(np.corrcoef(previous, current)[0, 1])
    return value if np.isfinite(value) else None


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _selected_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if not len(y_true):
        return {"n_samples": 0, **{key: None for key in DIAGNOSTIC_METRICS}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
    return {
        "n_samples": int(len(y_true)),
        "n_classes_present": int(len(np.unique(y_true))),
        **{key: float(metrics[key]) for key in DIAGNOSTIC_METRICS},
    }


def metrics_by_group(
    predictions: pd.DataFrame,
    group_columns: str | Sequence[str],
) -> list[dict[str, Any]]:
    columns = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    rows: list[dict[str, Any]] = []
    grouping: str | list[str] = columns[0] if len(columns) == 1 else columns
    for keys, group in predictions.groupby(grouping, sort=True, observed=True):
        key_values = (keys,) if len(columns) == 1 else keys
        row = {column: str(value) for column, value in zip(columns, key_values)}
        row.update(
            _selected_metrics(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
            )
        )
        rows.append(row)
    return rows


def previous_label_predictions(
    frame: pd.DataFrame,
    *,
    label_col: str = "label_q5",
) -> pd.DataFrame:
    """Diagnostic ``prediction(t) = true_label(t-1)`` within each record."""

    pairs = make_lag_pairs(frame, value_col=label_col, lag=1)
    output = pairs[
        [
            "sample_id",
            "previous_sample_id",
            "source",
            "subject_id",
            "record_id",
            "t_start",
            "absolute_window_index",
        ]
    ].copy()
    output["y_true"] = pairs["current_value"].astype(int)
    output["y_pred"] = pairs["previous_value"].astype(int)
    output["protocol"] = "within_record_lag1"
    output["diagnostic_set"] = "previous_label"
    output["model"] = "previous_label_rule"
    output["fold"] = pairs["outer_fold"].astype(int) if "outer_fold" in pairs else pd.NA
    output["prediction_id"] = [
        f"previous_label:{sample_id}" for sample_id in output["sample_id"]
    ]
    if not output["prediction_id"].is_unique:
        raise RuntimeError("Previous-label diagnostic prediction IDs are not unique")
    return output


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        "overall": _selected_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_pred"].to_numpy(dtype=int),
        ),
        "by_source": metrics_by_group(predictions, "source"),
        "by_subject": metrics_by_group(predictions, "subject_id"),
        "by_record": metrics_by_group(
            predictions, ["source", "subject_id", "record_id"]
        ),
    }


def calculate_runs(
    frame: pd.DataFrame,
    *,
    label_col: str = "label_q5",
) -> pd.DataFrame:
    ordered = frame.sort_values(ORDER_COLUMNS, kind="mergesort").copy()
    grouped = ordered.groupby(RECORD_COLUMNS, sort=False, observed=True)
    previous_label = grouped[label_col].shift(1)
    previous_index = grouped["absolute_window_index"].shift(1)
    boundary = (
        previous_label.isna()
        | (ordered[label_col] != previous_label)
        | ((ordered["absolute_window_index"] - previous_index) != 1)
    )
    ordered["run_id"] = boundary.groupby(
        [ordered[column] for column in RECORD_COLUMNS], sort=False
    ).cumsum().astype(np.int64)
    rows: list[dict[str, Any]] = []
    for keys, group in ordered.groupby(
        [*RECORD_COLUMNS, "run_id"], sort=True, observed=True
    ):
        source, subject_id, record_id, run_id = keys
        rows.append(
            {
                "source": str(source),
                "subject_id": str(subject_id),
                "record_id": str(record_id),
                "run_id": int(run_id),
                "class_id": int(group[label_col].iloc[0]),
                "start_sample_id": group["sample_id"].iloc[0],
                "end_sample_id": group["sample_id"].iloc[-1],
                "length_windows": int(len(group)),
                "duration_seconds": float(group["t_end"].iloc[-1] - group["t_start"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> dict[str, Any]:
    def row(group: pd.DataFrame) -> dict[str, Any]:
        return {
            "runs": int(len(group)),
            "length_windows": _distribution(group["length_windows"]),
            "duration_seconds": _distribution(group["duration_seconds"]),
        }

    return {
        "overall": row(runs),
        "by_source": [
            {"source": str(source), **row(group)}
            for source, group in runs.groupby("source", sort=True, observed=True)
        ],
        "by_subject": [
            {"subject_id": str(subject), **row(group)}
            for subject, group in runs.groupby("subject_id", sort=True, observed=True)
        ],
        "by_class": [
            {"class_id": int(class_id), **row(group)}
            for class_id, group in runs.groupby("class_id", sort=True, observed=True)
        ],
    }


def calculate_temporal_statistics(
    frame: pd.DataFrame,
    *,
    lags: Sequence[int] = (1, 2, 3, 5, 10, 20),
    target_col: str = "target_focus",
    label_col: str = "label_q5",
    n_classes: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    autocorrelation: dict[str, Any] = {}
    source_autocorrelation: dict[str, Any] = {}
    record_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for lag in lags:
        pairs = make_lag_pairs(frame, value_col=target_col, lag=int(lag))
        autocorrelation[str(lag)] = {
            "pairs": int(len(pairs)),
            "pooled_pearson": _correlation(pairs),
        }
        source_autocorrelation[str(lag)] = {
            str(source): {
                "pairs": int(len(group)),
                "pooled_pearson": _correlation(group),
            }
            for source, group in pairs.groupby("source", sort=True, observed=True)
        }
        for keys, group in pairs.groupby(RECORD_COLUMNS, sort=True, observed=True):
            key = tuple(str(value) for value in keys)
            record_rows.setdefault(
                key,
                {**dict(zip(RECORD_COLUMNS, key)), "windows": int(
                    frame.loc[
                        (frame["source"] == key[0])
                        & (frame["subject_id"] == key[1])
                        & (frame["record_id"] == key[2])
                    ].shape[0]
                )},
            )[f"autocorrelation_lag_{lag}"] = _correlation(group)

    adjacent = make_lag_pairs(frame, value_col=target_col, lag=1)
    adjacent["change"] = adjacent["current_value"] - adjacent["previous_value"]
    adjacent["absolute_change"] = adjacent["change"].abs()
    change_by_source = []
    for source, group in adjacent.groupby("source", sort=True, observed=True):
        change_by_source.append(
            {
                "source": str(source),
                "signed_change": _distribution(group["change"]),
                "absolute_change": _distribution(group["absolute_change"]),
            }
        )
    for keys, group in adjacent.groupby(RECORD_COLUMNS, sort=True, observed=True):
        key = tuple(str(value) for value in keys)
        record_rows.setdefault(key, dict(zip(RECORD_COLUMNS, key))).update(
            {
                "adjacent_pairs": int(len(group)),
                "mean_absolute_change": float(group["absolute_change"].mean()),
                "median_absolute_change": float(group["absolute_change"].median()),
            }
        )

    transitions = make_lag_pairs(frame, value_col=label_col, lag=1)
    transitions["previous_class"] = transitions["previous_value"].astype(int)
    transitions["current_class"] = transitions["current_value"].astype(int)
    transitions["class_distance"] = (
        transitions["current_class"] - transitions["previous_class"]
    ).abs()
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(
        counts,
        (
            transitions["previous_class"].to_numpy(dtype=int),
            transitions["current_class"].to_numpy(dtype=int),
        ),
        1,
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )

    def transition_row(group: pd.DataFrame) -> dict[str, Any]:
        distance = group["class_distance"].to_numpy(dtype=int)
        return {
            "pairs": int(len(group)),
            "same_class_probability": float(np.mean(distance == 0)),
            "adjacent_class_probability": float(np.mean(distance == 1)),
            "two_or_more_classes_probability": float(np.mean(distance >= 2)),
        }

    transition_by_source = [
        {"source": str(source), **transition_row(group)}
        for source, group in transitions.groupby("source", sort=True, observed=True)
    ]
    transition_by_subject = [
        {"subject_id": str(subject), **transition_row(group)}
        for subject, group in transitions.groupby("subject_id", sort=True, observed=True)
    ]
    runs = calculate_runs(frame, label_col=label_col)
    statistics = {
        "sequence_definition": {
            "group_columns": RECORD_COLUMNS,
            "order_columns": ["t_start", "sample_id"],
            "future_values_used": False,
            "records_crossed": False,
        },
        "target_focus": {
            "autocorrelation": autocorrelation,
            "autocorrelation_by_source": source_autocorrelation,
            "adjacent_change": {
                "signed": _distribution(adjacent["change"]),
                "absolute": _distribution(adjacent["absolute_change"]),
                "by_source": change_by_source,
                "by_record": list(record_rows.values()),
            },
        },
        "label_q5": {
            "transitions": transition_row(transitions),
            "transition_counts": counts.tolist(),
            "transition_probabilities_by_previous_class": probabilities.tolist(),
            "transitions_by_source": transition_by_source,
            "transitions_by_subject": transition_by_subject,
            "runs": summarize_runs(runs),
        },
    }
    return statistics, runs


def blocked_time_predictions(
    frame: pd.DataFrame,
    *,
    early_end: float = 0.4,
    late_start: float = 0.6,
    label_col: str = "label_q5",
) -> pd.DataFrame:
    """Previous-label diagnostics within early/late blocks and across a gap."""

    if not 0.0 < early_end < late_start < 1.0:
        raise ValueError("blocked-time bounds must satisfy 0 < early_end < late_start < 1")
    outputs: list[pd.DataFrame] = []
    for protocol, mask in (
        ("blocked_time_early_adjacent", frame["normalized_record_progress"] <= early_end),
        ("blocked_time_late_adjacent", frame["normalized_record_progress"] >= late_start),
    ):
        pairs = make_lag_pairs(frame.loc[mask].copy(), value_col=label_col, lag=1)
        if pairs.empty:
            continue
        block = pairs[
            [
                "sample_id", "previous_sample_id", "source", "subject_id",
                "record_id", "t_start", "absolute_window_index",
            ]
        ].copy()
        block["y_true"] = pairs["current_value"].astype(int)
        block["y_pred"] = pairs["previous_value"].astype(int)
        block["protocol"] = protocol
        block["gap_windows"] = 1
        block["gap_seconds"] = (
            pairs["t_start"].to_numpy(dtype=float)
            - frame.set_index("sample_id").loc[
                pairs["previous_sample_id"].to_numpy(), "t_start"
            ].to_numpy(dtype=float)
        )
        block["fold"] = pairs["outer_fold"].astype(int) if "outer_fold" in pairs else pd.NA
        outputs.append(block)

    bridge_rows: list[dict[str, Any]] = []
    for _, group in frame.groupby(RECORD_COLUMNS, sort=True, observed=True):
        ordered = group.sort_values(["t_start", "sample_id"], kind="mergesort")
        early = ordered.loc[ordered["normalized_record_progress"] <= early_end]
        late = ordered.loc[ordered["normalized_record_progress"] >= late_start]
        if early.empty or late.empty:
            continue
        previous = early.iloc[-1]
        current = late.iloc[0]
        if int(current["absolute_window_index"]) <= int(previous["absolute_window_index"]):
            continue
        bridge_rows.append(
            {
                "sample_id": current["sample_id"],
                "previous_sample_id": previous["sample_id"],
                "source": current["source"],
                "subject_id": current["subject_id"],
                "record_id": current["record_id"],
                "t_start": float(current["t_start"]),
                "absolute_window_index": int(current["absolute_window_index"]),
                "y_true": int(current[label_col]),
                "y_pred": int(previous[label_col]),
                "protocol": "blocked_time_cross_gap",
                "gap_windows": int(
                    current["absolute_window_index"] - previous["absolute_window_index"]
                ),
                "gap_seconds": float(current["t_start"] - previous["t_start"]),
                "fold": int(current["outer_fold"]) if "outer_fold" in current else pd.NA,
            }
        )
    if bridge_rows:
        outputs.append(pd.DataFrame(bridge_rows))
    if not outputs:
        return pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True, sort=False)
    result["diagnostic_set"] = "blocked_time"
    result["model"] = "previous_label_rule"
    result["prediction_id"] = [
        f"{protocol}:{sample_id}"
        for protocol, sample_id in zip(result["protocol"], result["sample_id"])
    ]
    if not result["prediction_id"].is_unique:
        raise RuntimeError("Blocked-time prediction IDs are not unique")
    return result


def summarize_blocked_time(predictions: pd.DataFrame) -> dict[str, Any]:
    protocols: dict[str, Any] = {}
    for protocol, group in predictions.groupby("protocol", sort=True, observed=True):
        protocols[str(protocol)] = {
            "overall": _selected_metrics(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
            ),
            "gap_windows": _distribution(group["gap_windows"]),
            "gap_seconds": _distribution(group["gap_seconds"]),
            "by_source": metrics_by_group(group, "source"),
        }
    return {
        "definition": {
            "early": "normalized_record_progress <= early_end",
            "late": "normalized_record_progress >= late_start",
            "cross_gap": "last early label predicts first late label",
            "scientific_role": "structure diagnostic, not a benchmark model",
        },
        "protocols": protocols,
    }


def load_temporal_audit_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    with spec_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or not isinstance(spec.get("audit"), dict):
        raise ValueError("Temporal audit YAML must contain an 'audit' mapping")
    required = {"data_path", "output_dir", "temporal_report", "diagnostic_report", "summary_path"}
    missing = sorted(required - set(spec["audit"]))
    if missing:
        raise ValueError(f"Temporal audit config is missing keys: {missing}")
    return spec


def _markdown_metrics(metrics: Mapping[str, Any]) -> str:
    columns = ["n_samples", *DIAGNOSTIC_METRICS]
    return " | ".join(
        str(metrics.get(column))
        if not isinstance(metrics.get(column), float)
        else f"{metrics[column]:.6f}"
        for column in columns
    )


@dataclass
class TemporalTargetAudit:
    spec_path: Path
    spec: dict[str, Any]
    data_path: Path
    output_dir: Path
    temporal_report: Path
    diagnostic_report: Path
    summary_path: Path

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.spec = load_temporal_audit_spec(self.spec_path)
        audit = self.spec["audit"]
        self.data_path = _repo_path(audit["data_path"])
        self.output_dir = _repo_path(output_dir or audit["output_dir"])
        self.temporal_report = _repo_path(audit["temporal_report"])
        self.diagnostic_report = _repo_path(audit["diagnostic_report"])
        self.summary_path = _repo_path(audit["summary_path"])

    def plan(self) -> dict[str, Any]:
        baseline = self.spec.get("diagnostic_baselines", {})
        return {
            "analysis_name": self.spec["audit"].get("name", "temporal_target_audit"),
            "spec_path": self.spec_path,
            "data_path": self.data_path,
            "group_columns": RECORD_COLUMNS,
            "order_columns": ["t_start", "sample_id"],
            "lags": self.spec["audit"].get("lags", [1, 2, 3, 5, 10, 20]),
            "outer_protocol": "GroupKFold by subject_id",
            "n_splits": int(baseline.get("n_splits", 5)),
            "diagnostic_sets": ["D0", "D1", "D2", "D3"],
            "models": ["majority_outer_train", "logistic_regression", "random_forest"],
            "forbidden_inputs": ["EEG.*", "POW.*", "subject_id", "record_id", "future labels"],
            "output_dir": self.output_dir,
            "models_trained": "simple sklearn diagnostics only",
            "deep_models_trained": 0,
            "writes_performed": False,
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        return "\n".join(
            [
                "# Temporal target audit plan",
                "",
                f"- Input: `{_jsonable(plan['data_path'])}`",
                f"- Sequences: {' + '.join(plan['group_columns'])}",
                f"- Order: {', '.join(plan['order_columns'])}",
                f"- Target lags: {plan['lags']}",
                f"- Outer protocol: {plan['outer_protocol']} ({plan['n_splits']} folds)",
                f"- Diagnostic sets: {', '.join(plan['diagnostic_sets'])}",
                f"- Models: {', '.join(plan['models'])}",
                "- EEG/POW features: none",
                "- Deep models trained: 0",
                "- Writes performed: no",
            ]
        )

    def _render_temporal_report(self, summary: Mapping[str, Any]) -> str:
        temporal = summary["temporal_statistics"]
        transitions = temporal["label_q5"]["transitions"]
        runs = temporal["label_q5"]["runs"]["overall"]["length_windows"]
        run_seconds = temporal["label_q5"]["runs"]["overall"]["duration_seconds"]
        transition_probabilities = temporal["label_q5"][
            "transition_probabilities_by_previous_class"
        ]
        lines = [
            "# Temporal label structure",
            "",
            "Sequences are formed strictly within `source + subject_id + record_id` and "
            "ordered by `t_start, sample_id`. Positive lags use only earlier windows.",
            "",
            "## Target autocorrelation",
            "",
            "| Lag | Pairs | Pooled Pearson autocorrelation |",
            "| --- | ---: | ---: |",
        ]
        for lag, values in temporal["target_focus"]["autocorrelation"].items():
            lines.append(
                f"| {lag} | {values['pairs']} | {values['pooled_pearson']:.6f} |"
            )
        change = temporal["target_focus"]["adjacent_change"]["absolute"]
        transition_sources = temporal["label_q5"]["transitions_by_source"]
        lines.extend(
            [
                "",
                "## Adjacent target change",
                "",
                f"Mean absolute change is `{change['mean']:.6f}`; median "
                f"`{change['median']:.6f}`, 95th percentile `{change['q95']:.6f}`.",
                "",
                "## Class stability",
                "",
                f"- Same next class: {transitions['same_class_probability']:.4%}",
                f"- Adjacent-class transition: {transitions['adjacent_class_probability']:.4%}",
                f"- Two-or-more-class transition: "
                f"{transitions['two_or_more_classes_probability']:.4%}",
                f"- Mean run length: {runs['mean']:.3f} windows",
                f"- Median run length: {runs['median']:.3f} windows",
                f"- 95th percentile run length: {runs['q95']:.3f} windows",
                f"- Mean / median duration: {run_seconds['mean']:.3f} / "
                f"{run_seconds['median']:.3f} seconds",
                "",
                "| Source | Pairs | Same class | Adjacent class | Two or more classes |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['source']} | {row['pairs']} | "
                    f"{row['same_class_probability']:.4%} | "
                    f"{row['adjacent_class_probability']:.4%} | "
                    f"{row['two_or_more_classes_probability']:.4%} |"
                    for row in transition_sources
                ],
                "",
                "Transition probabilities, with rows as previous class and columns as next class:",
                "",
                "| Previous \\ Next | 0 | 1 | 2 | 3 | 4 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *[
                    f"| {class_id} | "
                    + " | ".join(f"{value:.4f}" for value in row)
                    + " |"
                    for class_id, row in enumerate(transition_probabilities)
                ],
                "",
                "## Previous-label diagnostic",
                "",
                "This is a structural diagnostic using the true preceding label, not a "
                "deployable model. First windows of every record and pairs crossing missing "
                "target windows are excluded.",
                "",
                "`n_samples | accuracy | balanced_accuracy | macro_f1 | ordinal_mae | "
                "adjacent_accuracy | severe_error_rate`",
                "",
                _markdown_metrics(summary["previous_label"]["overall"]),
                "",
                "## Blocked-time check",
                "",
            ]
        )
        for protocol, values in summary["blocked_time"]["protocols"].items():
            lines.append(f"- `{protocol}`: {_markdown_metrics(values['overall'])}")
        lines.extend(
            [
                "",
                "Close-neighbor early/late results and the cross-gap bridge are descriptive "
                "checks. Every subject/record is an outer-test observation in exactly one "
                "canonical fold. The cross-gap rule uses the last early true label to predict "
                "only the first late label of each eligible record.",
                "",
                "## Interpretation risk",
                "",
                "The high adjacent-window autocorrelation and class persistence mean that a "
                "sequence model can obtain apparent benefit from local smoothness, record "
                "position, or access to correlated neighbouring windows. The previous-label "
                "result is an upper diagnostic that uses unavailable true history, not a fair "
                "competitor. Subject GroupKFold prevents subject identity leakage, but claims "
                "about temporal decoding should additionally report blocked or forward-time "
                "checks and must not attribute all sequential gain to EEG physiology.",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_diagnostic_report(self, summary: Mapping[str, Any]) -> str:
        overall = summary["diagnostic_baselines"]["overall"]
        overall_map = {
            (row["diagnostic_set"], row["model"]): row["metrics"]
            for row in overall
        }
        d0 = overall_map[("D0", "majority_outer_train")]
        d2 = overall_map[("D2", "logistic_regression")]
        lines = [
            "# Diagnostic baselines without EEG",
            "",
            "All results use the canonical five subject GroupKFold partitions. Model fitting, "
            "one-hot encoding, and numeric scaling are confined to each outer-train partition. "
            "No EEG, POW, subject ID, record ID, future label, or test statistic is an input.",
            "",
            "| Diagnostic | Model | Features | Accuracy | Balanced accuracy | Macro F1 | "
            "Ordinal MAE | Adjacent accuracy | Severe error rate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in overall:
            metrics = row["metrics"]
            lines.append(
                f"| {row['diagnostic_set']} | {row['model']} | "
                f"{', '.join(row['feature_columns']) or 'outer-train majority'} | "
                f"{metrics['accuracy']:.6f} | {metrics['balanced_accuracy']:.6f} | "
                f"{metrics['macro_f1']:.6f} | {metrics['ordinal_mae']:.6f} | "
                f"{metrics['adjacent_accuracy']:.6f} | "
                f"{metrics['severe_error_rate']:.6f} |"
            )
        lines.extend(
            [
                "",
                "D0 is the class mode from outer-train only. D1 uses source; D2 uses "
                "normalized record progress, zero-based absolute window index, and record "
                "duration; D3 combines D1 and D2. Record progress and duration use complete "
                "record metadata and are therefore retrospective covariates, not necessarily "
                "available in an online setting.",
                "",
                f"The strongest pooled accuracy is the D2 logistic control at "
                f"`{d2['accuracy']:.6f}`, a delta of "
                f"`{d2['accuracy'] - d0['accuracy']:+.6f}` from D0. Its balanced-accuracy "
                f"delta from D0 is "
                f"`{d2['balanced_accuracy'] - d0['balanced_accuracy']:+.6f}`. Adding source "
                "in D3 does not improve that result, so the modest signal is primarily "
                "record-position/duration structure rather than acquisition source.",
                "",
                "Source-stratified and subject-stratified metrics are retained in "
                "`diagnostic_metrics.json`; fold means and standard deviations are retained "
                "in the summary JSON.",
                "",
                "These controls quantify target structure and acquisition context. They are "
                "not evidence that a cognitive state is decoded from EEG.",
                "",
            ]
        )
        return "\n".join(lines)

    def execute(self) -> dict[str, Any]:
        from bench.analysis.diagnostic_baselines import (
            align_with_canonical_predictions,
            assign_subject_folds,
            run_diagnostic_baselines,
        )

        audit = self.spec["audit"]
        baseline_spec = self.spec.get("diagnostic_baselines", {})
        target_col = str(audit.get("target_column", "target_focus"))
        label_col = str(audit.get("label_column", "label_q5"))
        n_classes = int(audit.get("n_classes", 5))
        before_hash = _sha256_file(self.data_path)
        before_size = self.data_path.stat().st_size
        frame = pd.read_parquet(
            self.data_path,
            columns=list(dict.fromkeys([*REQUIRED_COLUMNS, target_col, label_col])),
        )
        prepared = prepare_temporal_frame(
            frame,
            target_col=target_col,
            label_col=label_col,
            n_classes=n_classes,
        )
        expected_rows = audit.get("expected_supervised_rows")
        if expected_rows is not None and len(prepared) != int(expected_rows):
            raise ValueError(
                f"Expected {expected_rows} supervised rows, observed {len(prepared)}"
            )
        folded, fold_metadata = assign_subject_folds(
            prepared,
            n_splits=int(baseline_spec.get("n_splits", 5)),
        )
        reference_path = audit.get("canonical_reference_predictions")
        alignment = None
        if reference_path:
            alignment = align_with_canonical_predictions(
                folded, _repo_path(reference_path), label_col=label_col
            )

        temporal, runs = calculate_temporal_statistics(
            folded,
            lags=[int(value) for value in audit.get("lags", [1, 2, 3, 5, 10, 20])],
            target_col=target_col,
            label_col=label_col,
            n_classes=n_classes,
        )
        persistence_predictions = previous_label_predictions(folded, label_col=label_col)
        persistence_metrics = summarize_predictions(persistence_predictions)
        blocked_spec = self.spec.get("blocked_time", {})
        blocked_predictions = blocked_time_predictions(
            folded,
            early_end=float(blocked_spec.get("early_end", 0.4)),
            late_start=float(blocked_spec.get("late_start", 0.6)),
            label_col=label_col,
        )
        blocked_metrics = summarize_blocked_time(blocked_predictions)
        baseline_predictions, baseline_metrics = run_diagnostic_baselines(
            folded,
            label_col=label_col,
            n_classes=n_classes,
            spec=baseline_spec,
        )
        predictions = pd.concat(
            [persistence_predictions, blocked_predictions, baseline_predictions],
            ignore_index=True,
            sort=False,
        )
        if predictions["prediction_id"].isna().any() or not predictions["prediction_id"].is_unique:
            raise RuntimeError("Unified diagnostic prediction IDs must be complete and unique")

        temporal_output = {
            "supervised_rows": int(len(folded)),
            "subjects": int(folded["subject_id"].nunique()),
            "records": int(folded[RECORD_COLUMNS].drop_duplicates().shape[0]),
            "folds": fold_metadata,
            "canonical_alignment": alignment,
            "temporal_statistics": temporal,
            "run_rows": runs.to_dict(orient="records"),
            "previous_label": persistence_metrics,
            "blocked_time": blocked_metrics,
        }
        diagnostic_output = {
            "folds": fold_metadata,
            "canonical_alignment": alignment,
            "previous_label": persistence_metrics,
            "blocked_time": blocked_metrics,
            "baselines": baseline_metrics,
        }
        summary = {
            "analysis_name": audit.get("name", "temporal_target_audit"),
            "analysis_only": True,
            "deep_models_trained": 0,
            "eeg_or_pow_features_used": False,
            "data_path": self.data_path,
            "input_sha256": before_hash,
            "input_size_bytes": before_size,
            "supervised_rows": int(len(folded)),
            "subjects": int(folded["subject_id"].nunique()),
            "records": int(folded[RECORD_COLUMNS].drop_duplicates().shape[0]),
            "canonical_alignment": alignment,
            "temporal_statistics": {
                "target_focus": {
                    "autocorrelation": temporal["target_focus"]["autocorrelation"],
                    "autocorrelation_by_source": temporal["target_focus"][
                        "autocorrelation_by_source"
                    ],
                    "adjacent_change": {
                        "signed": temporal["target_focus"]["adjacent_change"]["signed"],
                        "absolute": temporal["target_focus"]["adjacent_change"]["absolute"],
                        "by_source": temporal["target_focus"]["adjacent_change"][
                            "by_source"
                        ],
                    },
                },
                "label_q5": {
                    "transitions": temporal["label_q5"]["transitions"],
                    "transitions_by_source": temporal["label_q5"][
                        "transitions_by_source"
                    ],
                    "transition_counts": temporal["label_q5"]["transition_counts"],
                    "transition_probabilities_by_previous_class": temporal["label_q5"][
                        "transition_probabilities_by_previous_class"
                    ],
                    "runs": {
                        "overall": temporal["label_q5"]["runs"]["overall"],
                        "by_source": temporal["label_q5"]["runs"]["by_source"],
                        "by_class": temporal["label_q5"]["runs"]["by_class"],
                    },
                },
            },
            "previous_label": {
                "overall": persistence_metrics["overall"],
                "by_source": persistence_metrics["by_source"],
            },
            "blocked_time": blocked_metrics,
            "diagnostic_baselines": {
                "overall": baseline_metrics["overall"],
                "fold_summary": baseline_metrics["fold_summary"],
                "by_source": baseline_metrics["by_source"],
                "feature_policy": baseline_metrics["feature_policy"],
            },
            "artifacts": {
                "temporal_statistics": self.output_dir / "temporal_statistics.json",
                "diagnostic_predictions": self.output_dir / "diagnostic_predictions.parquet",
                "diagnostic_metrics": self.output_dir / "diagnostic_metrics.json",
                "temporal_report": self.temporal_report,
                "diagnostic_report": self.diagnostic_report,
                "summary": self.summary_path,
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "temporal_statistics.json", temporal_output)
        predictions.to_parquet(
            self.output_dir / "diagnostic_predictions.parquet", index=False
        )
        _write_json(self.output_dir / "diagnostic_metrics.json", diagnostic_output)
        after_hash = _sha256_file(self.data_path)
        if after_hash != before_hash or self.data_path.stat().st_size != before_size:
            raise RuntimeError("Input Parquet changed during temporal audit")
        summary["input_sha256_after"] = after_hash
        summary["input_modified"] = False
        _write_json(self.summary_path, summary)
        self.temporal_report.parent.mkdir(parents=True, exist_ok=True)
        self.temporal_report.write_text(
            self._render_temporal_report(_jsonable(summary)), encoding="utf-8"
        )
        self.diagnostic_report.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostic_report.write_text(
            self._render_diagnostic_report(_jsonable(summary)), encoding="utf-8"
        )
        return _jsonable(summary)


__all__ = [
    "TemporalTargetAudit",
    "blocked_time_predictions",
    "calculate_runs",
    "calculate_temporal_statistics",
    "make_lag_pairs",
    "prepare_temporal_frame",
    "previous_label_predictions",
    "summarize_blocked_time",
    "summarize_predictions",
]
