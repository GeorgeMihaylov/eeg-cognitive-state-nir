"""Metrics and evaluation helpers for seven independently labelled PM tasks."""
from __future__ import annotations
import numpy as np
from .metrics import classification_metrics
from cogstate.protocol import PM_METRICS


def evaluate_pm_tasks(y_true, y_pred, metric_names=PM_METRICS):
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if actual.shape != predicted.shape or actual.ndim != 2: raise ValueError("Expected equally shaped [windows, targets] arrays")
    per_target = {}
    for index, name in enumerate(metric_names):
        valid = actual[:, index] >= 0
        per_target[name] = classification_metrics(actual[valid, index], predicted[valid, index]) if valid.any() else None
    available = [value for value in per_target.values() if value is not None]
    return {"per_target": per_target, "macro_f1_mean": float(np.mean([value["f1_macro"] for value in available])) if available else float("nan")}
