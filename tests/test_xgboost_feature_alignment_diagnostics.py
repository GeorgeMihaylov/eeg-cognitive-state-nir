from __future__ import annotations

import numpy as np
import pytest

from bench.analysis.xgboost_feature_alignment_diagnostics import (
    apply_location_only,
    apply_shrinkage,
    displacement_table,
    estimate_location_scale,
    prediction_diagnostics,
    safe_correlations,
    subject_macro_reference_diagnostics,
    temporal_drift_summary,
)


def test_location_only_and_shrinkage_have_exact_endpoint_semantics() -> None:
    original = np.asarray([[10.0, -5.0], [12.0, -1.0]])
    reference_center = np.asarray([2.0, 4.0])
    calibration_center = np.asarray([11.0, -3.0])
    location_only = apply_location_only(
        original,
        reference_center=reference_center,
        calibration_center=calibration_center,
    )

    np.testing.assert_allclose(
        location_only,
        original + reference_center - calibration_center,
    )
    np.testing.assert_allclose(apply_shrinkage(original, location_only, 0.0), original)
    np.testing.assert_allclose(
        apply_shrinkage(original, location_only, 1.0), location_only
    )
    np.testing.assert_allclose(
        apply_shrinkage(original, location_only, 0.25),
        original + 0.25 * (location_only - original),
    )


def test_diagnostic_statistics_match_standard_and_robust_contracts() -> None:
    X = np.asarray([[0.0, 1.0], [2.0, 3.0], [100.0, 8.0]])
    mean, std = estimate_location_scale(X, "standard_location_scale")
    median, iqr = estimate_location_scale(X, "robust_location_scale")

    np.testing.assert_allclose(mean, np.mean(X, axis=0))
    np.testing.assert_allclose(std, np.std(X, axis=0, ddof=0))
    np.testing.assert_allclose(median, np.median(X, axis=0))
    np.testing.assert_allclose(
        iqr,
        np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0),
    )


def test_displacement_uses_real_feature_names_and_reference_scale() -> None:
    original = np.zeros((4, 3), dtype=float)
    transformed = np.tile(np.asarray([1.0, 2.0, 8.0]), (4, 1))
    table, summary = displacement_table(
        original,
        transformed,
        reference_scale=np.asarray([2.0, 2.0, 4.0]),
        feature_names=["EEG.AF3.mean", "POW.AF3.alpha", "EEG.F7.std"],
        scale_epsilon=1e-12,
    )

    assert table["feature_name"].tolist() == [
        "EEG.AF3.mean", "POW.AF3.alpha", "EEG.F7.std"
    ]
    np.testing.assert_allclose(table["normalized_displacement"], [0.5, 1.0, 2.0])
    assert summary["normalized_displacement_p50"] == pytest.approx(1.0)
    assert summary["fraction_features_normalized_gt_0_5"] == pytest.approx(2 / 3)


def test_temporal_drift_is_normalized_only_by_outer_reference() -> None:
    calibration = np.asarray([[0.0, 0.0], [2.0, 4.0]])
    evaluation = np.asarray([[2.0, 4.0], [4.0, 8.0]])
    drift = temporal_drift_summary(
        calibration,
        evaluation,
        reference_scale=np.asarray([2.0, 4.0]),
        method="standard_location_scale",
        scale_epsilon=1e-12,
    )

    assert drift["center_drift_p50"] == pytest.approx(1.0)
    assert drift["scale_drift_p50"] == pytest.approx(0.0)


def test_prediction_shift_reports_class_metrics_confusion_and_collapse() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    baseline_pred = y_true.copy()
    baseline_proba = np.eye(3)[baseline_pred] * 0.8 + 0.2 / 3
    collapsed_pred = np.zeros_like(y_true)
    collapsed_proba = np.tile(np.asarray([0.9, 0.05, 0.05]), (len(y_true), 1))

    summary, classes = prediction_diagnostics(
        y_true,
        collapsed_pred,
        collapsed_proba,
        baseline_pred=baseline_pred,
        baseline_proba=baseline_proba,
    )

    assert summary["class_collapse"] is True
    assert summary["predicted_class_count"] == 1
    assert summary["prediction_flip_rate"] == pytest.approx(4 / 6)
    assert len(classes) == 3
    assert classes[0]["recall"] == pytest.approx(1.0)
    assert classes[1]["recall"] == pytest.approx(0.0)
    assert classes[2]["confusion_pred_0"] == pytest.approx(1.0)


def test_safe_correlations_and_subject_macro_bias() -> None:
    correlations = safe_correlations([1, 2, 3, 4], [2, 4, 6, 8])
    assert correlations["pearson"] == pytest.approx(1.0)
    assert correlations["spearman"] == pytest.approx(1.0)
    assert safe_correlations([1, 1, 1], [1, 2, 3])["pearson"] is None

    X = np.asarray([[0.0], [0.0], [0.0], [0.0], [10.0]])
    subjects = np.asarray(["many", "many", "many", "many", "few"])
    bias = subject_macro_reference_diagnostics(X, subjects, ["EEG.AF3.mean"])
    assert bias["available"] is True
    assert bias["n_subjects"] == 2
    assert (
        bias["methods"]["standard_location_scale"]
        ["normalized_center_difference_max"]
        > 0
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
def test_shrinkage_rejects_invalid_alpha(alpha: float) -> None:
    X = np.ones((2, 2))
    with pytest.raises(ValueError, match="alpha"):
        apply_shrinkage(X, X, alpha)
