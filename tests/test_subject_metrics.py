from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analysis.subject_metrics import calculate_subject_metrics


def _probabilities(labels: list[int], n_classes: int = 5) -> dict[str, list[float]]:
    matrix = np.full((len(labels), n_classes), 0.025, dtype=float)
    for row, label in enumerate(labels):
        matrix[row, label] = 0.9
    return {f"proba_{index}": matrix[:, index].tolist() for index in range(n_classes)}


def test_subject_metrics_and_explicit_auc_policy() -> None:
    full_labels = [0, 1, 2, 3, 4]
    missing_labels = [0, 0, 1, 1]
    frame = pd.concat([
        pd.DataFrame({
            "subject_id": "S1",
            "fold": 1,
            "record_id": "gpn_data__r1",
            "y_true": full_labels,
            "y_pred": full_labels,
            **_probabilities(full_labels),
        }),
        pd.DataFrame({
            "subject_id": "S2",
            "fold": 2,
            "record_id": "Old_EEG__r2",
            "y_true": missing_labels,
            "y_pred": [0, 1, 1, 2],
            **_probabilities(missing_labels),
        }),
    ], ignore_index=True)
    metrics = calculate_subject_metrics(
        frame,
        track="feature_window",
        model="model",
        seed=42,
    ).set_index("subject_id")
    assert metrics.loc["S1", "accuracy"] == 1.0
    assert metrics.loc["S1", "auc_status"] == "defined"
    assert metrics.loc["S1", "auc"] == 1.0
    assert metrics.loc["S2", "auc_status"] == "undefined_missing_true_classes"
    assert np.isnan(metrics.loc["S2", "auc"])
    assert metrics.loc["S2", "classes_present"] == "[0, 1]"
    assert metrics.loc["S2", "class_policy"].startswith("balanced_accuracy")


def test_calibration_inputs_are_excluded_from_subject_metrics() -> None:
    frame = pd.DataFrame({
        "subject_id": ["S1"] * 4,
        "outer_fold": ["fold_01"] * 4,
        "record_id": ["Old_EEG__r"] * 4,
        "y_true": [4, 0, 1, 2],
        "y_pred": [0, 0, 1, 1],
        "is_calibration_sample": [True, False, False, False],
    })
    metrics = calculate_subject_metrics(
        frame,
        track="calibration",
        model="head_only",
        seed=42,
        budget_seconds=180,
    ).iloc[0]
    assert metrics["n_samples"] == 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["ordinal_mae"] == 1 / 3
    assert metrics["budget_seconds"] == 180
