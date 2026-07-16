"""Class, ordinal, and source-aware descriptive error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from bench.validation.metrics import MetricsCalculator


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray | None:
    columns = sorted(
        (column for column in frame if column.startswith("proba_")),
        key=lambda column: int(column.split("_", 1)[1]),
    )
    if not columns:
        return None
    values = frame[columns].to_numpy(dtype=float)
    return values if np.isfinite(values).all() else None


def calculate_error_analysis(
    predictions: pd.DataFrame,
    *,
    labels: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Calculate classification and ordinal errors from immutable predictions."""

    required = {"y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    frame = predictions
    if "is_calibration_sample" in frame:
        frame = frame.loc[~frame["is_calibration_sample"].fillna(False).astype(bool)]
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    distance = np.abs(y_pred.astype(float) - y_true.astype(float))
    predicted_counts = np.bincount(y_pred, minlength=len(labels))
    true_counts = np.bincount(y_true, minlength=len(labels))
    central_predictions = np.isin(y_pred, [1, 2, 3])
    extreme_truth = np.isin(y_true, [0, 4])
    return {
        "n_samples": int(len(frame)),
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": normalized.tolist(),
        "per_class": [
            {
                "class": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
        "true_class_distribution": {
            str(label): int(true_counts[index]) for index, label in enumerate(labels)
        },
        "predicted_class_distribution": {
            str(label): int(predicted_counts[index]) for index, label in enumerate(labels)
        },
        "ordinal_mae": float(distance.mean()),
        "adjacent_accuracy": float((distance <= 1).mean()),
        "severe_error_rate": float((distance >= 2).mean()),
        "adjacent_error_rate": float((distance == 1).mean()),
        "exact_accuracy": float((distance == 0).mean()),
        "extreme_class_recall_0": float(recall[labels.index(0)]),
        "extreme_class_recall_4": float(recall[labels.index(4)]),
        "extreme_truth_predicted_centrally": (
            float((central_predictions & extreme_truth).sum() / extreme_truth.sum())
            if extreme_truth.any()
            else np.nan
        ),
    }


def _ensure_source(frame: pd.DataFrame) -> pd.DataFrame:
    if "source" in frame:
        return frame
    if "record_id" not in frame:
        raise ValueError("Source analysis requires source or record_id")
    output = frame.copy()
    output["source"] = np.where(
        output["record_id"].astype(str).str.startswith("gpn_data"),
        "gpn_data",
        "Old_EEG",
    )
    return output


def summarize_by_source(
    predictions: pd.DataFrame,
    *,
    model: str | None = None,
) -> pd.DataFrame:
    """Return descriptive source metrics without treating overlaps as new subjects.

    The returned DataFrame records source-specific subject sets. Its attributes
    contain the overall unique-subject count and explicitly mark source subject
    counts as non-additive.
    """

    frame = _ensure_source(predictions)
    if "is_calibration_sample" in frame:
        frame = frame.loc[~frame["is_calibration_sample"].fillna(False).astype(bool)]
    rows: list[dict[str, Any]] = []
    for source, group in frame.groupby("source", sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        probability = _probability_matrix(group)
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, probability)
        subject_ids = sorted(set(group["subject_id"].astype(str)))
        rows.append({
            "model": model or (
                str(group["model"].iloc[0]) if "model" in group else "unknown"
            ),
            "source": str(source),
            "subjects": len(subject_ids),
            "subject_ids": ",".join(subject_ids),
            "samples": int(len(group)),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "ordinal_mae": float(metrics["ordinal_mae"]),
        })
    result = pd.DataFrame(rows)
    result.attrs["unique_subjects_overall"] = int(frame["subject_id"].nunique())
    result.attrs["source_subject_counts_are_additive"] = False
    overlap: set[str] = set()
    source_sets = [set(value.split(",")) for value in result.get("subject_ids", [])]
    for index, left in enumerate(source_sets):
        for right in source_sets[index + 1:]:
            overlap.update(left & right)
    result.attrs["subjects_in_multiple_sources"] = sorted(overlap - {""})
    return result
