from __future__ import annotations

import numpy as np

from bench.validation.metrics import MetricsCalculator


TARGET_NAMES = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]


def test_multioutput_metrics_match_per_target_means() -> None:
    truth = np.arange(56, dtype=float).reshape(8, 7) / 10.0
    prediction = truth + np.arange(7, dtype=float)[None, :] / 100.0

    metrics = MetricsCalculator.calculate_regression_metrics(
        truth,
        prediction,
        target_names=TARGET_NAMES,
    )

    assert metrics["n_outputs"] == 7
    assert len(metrics["per_target"]) == 7
    expected_mae = np.mean([
        metrics[f"mae_{name.removeprefix('target_')}"]
        for name in TARGET_NAMES
    ])
    assert np.isclose(metrics["mae_macro"], expected_mae)
    assert metrics["pearson_valid_targets"] == 7
    assert "accuracy" not in metrics


def test_constant_target_is_reported_as_undefined_correlation() -> None:
    truth = np.arange(35, dtype=float).reshape(5, 7)
    truth[:, 0] = 1.0
    prediction = truth.copy()

    metrics = MetricsCalculator.calculate_regression_metrics(
        truth,
        prediction,
        target_names=TARGET_NAMES,
    )

    assert np.isnan(metrics["pearson_attention"])
    assert np.isnan(metrics["spearman_attention"])
    assert metrics["pearson_valid_targets"] == 6
    assert metrics["spearman_valid_targets"] == 6


def test_subject_level_aggregation_uses_subject_means() -> None:
    truth = np.arange(42, dtype=float).reshape(6, 7)
    prediction = truth + 0.5
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"])

    metrics, table = MetricsCalculator.calculate_subject_regression_metrics(
        truth,
        prediction,
        subjects,
        TARGET_NAMES,
        fold=1,
    )

    assert len(table) == 3 * 7
    assert table["n_windows"].eq(2).all()
    assert metrics["subject_n_subjects"] == 3
    assert np.isclose(metrics["subject_mae_macro"], 0.5)
