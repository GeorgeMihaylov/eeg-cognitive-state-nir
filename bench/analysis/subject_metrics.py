"""Subject-level aggregation for benchmark prediction artifacts."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bench.validation.metrics import MetricsCalculator


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("proba_")]
    return sorted(columns, key=lambda column: int(column.split("_", 1)[1]))


def evaluation_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return evaluation rows, excluding calibration inputs when marked."""

    if "is_calibration_sample" not in frame:
        return frame.copy()
    marker = frame["is_calibration_sample"].fillna(False).astype(bool)
    return frame.loc[~marker].copy()


def _fold_value(group: pd.DataFrame) -> str | None:
    column = "fold" if "fold" in group else "outer_fold" if "outer_fold" in group else None
    if column is None:
        return None
    values = group[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(
            "A subject must belong to one outer fold; "
            f"observed={sorted(values.tolist())}"
        )
    value = values[0]
    if value.startswith("fold_"):
        return value
    try:
        return f"fold_{int(float(value)):02d}"
    except ValueError:
        return value


def _source_value(group: pd.DataFrame) -> str:
    if "source" in group:
        values = sorted(set(group["source"].dropna().astype(str)))
    elif "record_id" in group:
        values = sorted({
            "gpn_data" if value.startswith("gpn_data") else "Old_EEG"
            for value in group["record_id"].dropna().astype(str)
        })
    else:
        values = []
    return "+".join(values) if values else "unknown"


def _subject_row(
    subject_id: str,
    group: pd.DataFrame,
    *,
    track: str,
    model: str,
    seed: int,
    budget_seconds: float | None,
) -> dict[str, Any]:
    y_true = group["y_true"].to_numpy(dtype=int)
    y_pred = group["y_pred"].to_numpy(dtype=int)
    proba_columns = probability_columns(group)
    n_probability_classes = len(proba_columns)
    classes_present = sorted(np.unique(y_true).astype(int).tolist())
    probability_classes = list(range(n_probability_classes))
    class_distribution = {
        str(label): int((y_true == label).sum()) for label in classes_present
    }
    y_proba: np.ndarray | None = None
    auc_status: str
    if not proba_columns:
        auc_status = "not_available"
    elif not np.isfinite(group[proba_columns].to_numpy(dtype=float)).all():
        auc_status = "undefined_nonfinite_probabilities"
    elif classes_present != probability_classes:
        auc_status = "undefined_missing_true_classes"
    else:
        y_proba = group[proba_columns].to_numpy(dtype=float)
        auc_status = "defined"

    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_proba)
    auc = float(metrics.get("auc", np.nan))
    if not np.isfinite(auc):
        if auc_status == "defined":
            auc_status = "undefined_metric_error"
        auc = np.nan
    records = int(group["record_id"].nunique()) if "record_id" in group else 0
    row: dict[str, Any] = {
        "track": track,
        "model": model,
        "seed": int(seed),
        "subject_id": str(subject_id),
        "outer_fold": _fold_value(group),
        "source": _source_value(group),
        "records": records,
        "n_samples": int(len(group)),
        "classes_present": json.dumps(classes_present),
        "class_distribution": json.dumps(class_distribution, sort_keys=True),
        "n_classes_present": len(classes_present),
        "class_policy": "balanced_accuracy_and_macro_f1_over_true_classes_present",
        "auc_status": auc_status,
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "kappa": float(metrics["kappa"]),
        "auc": auc,
        "ordinal_mae": float(metrics["ordinal_mae"]),
        "adjacent_accuracy": float(metrics["adjacent_accuracy"]),
        "severe_error_rate": float(metrics["severe_error_rate"]),
    }
    if budget_seconds is not None:
        row["budget_seconds"] = float(budget_seconds)
    return row


def calculate_subject_metrics(
    predictions: pd.DataFrame,
    *,
    track: str,
    model: str,
    seed: int,
    budget_seconds: float | None = None,
    subject_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate one independent metrics row per anonymized subject.

    Balanced accuracy and macro F1 use scikit-learn's explicit present-class
    policy. Multiclass AUC is reported only when every probability class is
    represented in that subject's evaluation labels; undefined AUC remains NaN.
    """

    required = {"subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    frame = evaluation_predictions(predictions)
    if subject_ids is not None:
        allowed = {str(value) for value in subject_ids}
        frame = frame.loc[frame["subject_id"].astype(str).isin(allowed)]
    rows = [
        _subject_row(
            str(subject_id),
            group,
            track=track,
            model=model,
            seed=seed,
            budget_seconds=budget_seconds,
        )
        for subject_id, group in frame.groupby("subject_id", sort=True)
    ]
    return pd.DataFrame(rows)


def calculate_regression_subject_metrics(
    predictions: pd.DataFrame,
    *,
    track: str,
    model: str,
    seed: int,
) -> pd.DataFrame:
    """Calculate continuous-target metrics without filling undefined values."""

    required = {"subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for subject_id, group in evaluation_predictions(predictions).groupby(
        "subject_id", sort=True
    ):
        metrics = MetricsCalculator.calculate_regression_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        rows.append({
            "track": track,
            "model": model,
            "seed": int(seed),
            "subject_id": str(subject_id),
            "outer_fold": _fold_value(group),
            "source": _source_value(group),
            "records": int(group["record_id"].nunique())
            if "record_id" in group
            else 0,
            "n_samples": int(metrics["n_samples"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "pearson": float(metrics["pearson"]),
            "spearman": float(metrics["spearman"]),
        })
    return pd.DataFrame(rows)
