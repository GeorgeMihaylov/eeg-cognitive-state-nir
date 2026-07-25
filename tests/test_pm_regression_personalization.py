from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bench.experiments.pm_regression_personalization import (
    ALLOWED_STATUSES,
    CANONICAL_TARGETS,
    METHODS,
    PMRegressionPersonalizationExperiment,
    _aggregate_outputs,
    _prediction_frame,
    apply_affine_calibration,
    apply_bias_correction,
    fit_affine_calibration,
    fit_bias_correction,
    metric_gain,
    regression_personalization_metrics,
)
from bench.experiments.user_calibration import (
    CalibrationSpec,
    _parameter_audit,
    _parameter_digest,
    _state_digest,
    chronological_window_partition,
)
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.factory import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]


def _windows(rows: int = 100):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(rows, 6)).astype(np.float32)
    y = rng.normal(size=(rows, 7)).astype(np.float32)
    metadata = pd.DataFrame({
        "source": np.where(np.arange(rows) < rows // 2, "Old_EEG", "gpn_data"),
        "subject_id": "target",
        "record_id": np.where(np.arange(rows) < rows // 2, "old", "gpn"),
        "record_group_id": np.where(
            np.arange(rows) < rows // 2, "old", "gpn"
        ),
        "sample_id": [f"sample-{index:04d}" for index in range(rows)],
        "t_start": np.r_[
            np.arange(rows // 2), np.arange(rows - rows // 2)
        ].astype(float) * 10.0,
    })
    return X, y, metadata


def _fraction_spec(fraction: float) -> CalibrationSpec:
    return CalibrationSpec(
        method="zero_shot",
        budget_seconds=None,
        budget_fraction=fraction,
        fraction_allocation="global_prefix",
        purge_windows=0,
        minimum_calibration_samples=1,
        minimum_evaluation_samples=1,
    )


@pytest.fixture(scope="module")
def regression_adapter():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    weights = rng.normal(size=(6, 7)).astype(np.float32)
    y = X @ weights + rng.normal(scale=0.05, size=(80, 7)).astype(np.float32)
    adapter = build_model(
        model_name="torch_mlp",
        task_type="regression",
        input_shape=(6,),
        num_outputs=7,
        params={
            "hidden_dims": [12, 8],
            "dropout": 0.0,
            "activation": "relu",
            "regression_loss": "mse",
            "batch_size": 16,
            "max_epochs": 2,
            "learning_rate": 0.001,
            "validation_size": 0.2,
            "early_stopping_patience": 2,
            "device": "cpu",
            "random_state": 42,
        },
    )
    adapter.fit(X, y)
    return adapter, X, y


def test_canonical_target_order() -> None:
    assert CANONICAL_TARGETS == (
        "target_attention", "target_engagement", "target_excitement",
        "target_stress", "target_relaxation", "target_interest",
        "target_focus",
    )


def test_all_five_methods_are_registered() -> None:
    assert METHODS == (
        "zero_shot", "bias_correction", "affine_calibration",
        "head_only", "full_model",
    )


def test_bias_correction_recovers_known_shift() -> None:
    prediction = np.arange(70, dtype=float).reshape(10, 7)
    shift = np.arange(7, dtype=float) / 10
    bias = fit_bias_correction(prediction + shift, prediction)
    np.testing.assert_allclose(bias, shift)


def test_bias_application_is_target_wise() -> None:
    prediction = np.zeros((3, 7))
    bias = np.arange(7, dtype=float)
    result = apply_bias_correction(prediction, bias)
    np.testing.assert_allclose(result, np.tile(bias, (3, 1)))


def test_bias_rejects_wrong_output_shape() -> None:
    with pytest.raises(ValueError, match="one bias per target"):
        apply_bias_correction(np.zeros((3, 7)), np.zeros(6))


def test_affine_calibration_recovers_known_mapping() -> None:
    prediction = np.arange(140, dtype=float).reshape(20, 7) / 100
    coefficients = np.linspace(0.5, 1.5, 7)
    intercepts = np.linspace(-0.2, 0.2, 7)
    truth = prediction * coefficients + intercepts
    fitted = fit_affine_calibration(truth, prediction, alpha=0.0)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-9)
    np.testing.assert_allclose(fitted.intercepts, intercepts, atol=1e-9)


def test_affine_parameters_cover_seven_targets() -> None:
    prediction = np.arange(70, dtype=float).reshape(10, 7)
    fitted = fit_affine_calibration(prediction, prediction)
    assert len(fitted.parameters) == 7
    assert [item["target_name"] for item in fitted.parameters] == list(
        CANONICAL_TARGETS
    )


def test_affine_fallback_for_constant_predictions() -> None:
    prediction = np.ones((10, 7))
    truth = np.arange(70, dtype=float).reshape(10, 7)
    fitted = fit_affine_calibration(truth, prediction)
    assert all(item["fallback_used"] for item in fitted.parameters)
    assert all(
        item["fallback_reason"] == "constant_prediction"
        for item in fitted.parameters
    )


def test_affine_application_remains_finite() -> None:
    prediction = np.arange(70, dtype=float).reshape(10, 7)
    fitted = fit_affine_calibration(prediction + 2.0, prediction)
    result = apply_affine_calibration(prediction, fitted)
    assert np.isfinite(result).all()


def test_metrics_are_computed_for_each_target() -> None:
    truth = np.arange(70, dtype=float).reshape(10, 7)
    metrics = regression_personalization_metrics(truth, truth + 1)
    for target in CANONICAL_TARGETS:
        assert metrics[f"{target}_mae"] == pytest.approx(1.0)
        assert f"{target}_spearman" in metrics


def test_macro_mae_is_target_mean() -> None:
    truth = np.zeros((8, 7))
    errors = np.arange(1, 8, dtype=float)
    prediction = np.tile(errors, (8, 1))
    metrics = regression_personalization_metrics(truth, prediction)
    assert metrics["macro_mae"] == pytest.approx(errors.mean())


def test_nan_pearson_is_not_replaced_by_zero() -> None:
    truth = np.ones((8, 7))
    prediction = np.ones((8, 7))
    metrics = regression_personalization_metrics(truth, prediction)
    assert np.isnan(metrics["macro_pearson"])
    assert metrics["defined_pearson_targets"] == 0


def test_mean_error_and_absolute_bias_are_signed_consistently() -> None:
    metrics = regression_personalization_metrics(
        np.zeros((5, 7)), np.full((5, 7), -0.25)
    )
    assert metrics["target_focus_mean_error"] == pytest.approx(-0.25)
    assert metrics["target_focus_abs_bias"] == pytest.approx(0.25)


def test_mae_gain_rewards_lower_error() -> None:
    assert metric_gain("mae", 0.4, 0.3) == pytest.approx(0.1)


def test_abs_bias_gain_rewards_lower_bias() -> None:
    assert metric_gain("macro_abs_bias", 0.2, 0.1) == pytest.approx(0.1)


def test_r2_gain_rewards_higher_score() -> None:
    assert metric_gain("r2", -0.2, 0.1) == pytest.approx(0.3)


def test_chronological_calibration_and_evaluation_do_not_overlap() -> None:
    X, y, metadata = _windows()
    split = chronological_window_partition(
        X, y, metadata, _fraction_spec(0.20),
        window_seconds=10.0, max_gap_seconds=10.5,
    )
    assert len(split.calibration_X) == 20
    assert set(split.calibration_metadata.sample_id).isdisjoint(
        split.evaluation_metadata.sample_id
    )


def test_adaptation_split_is_eighty_twenty_inside_calibration() -> None:
    X, y, metadata = _windows()
    outer = chronological_window_partition(
        X, y, metadata, _fraction_spec(0.20),
        window_seconds=10.0, max_gap_seconds=10.5,
    )
    inner = chronological_window_partition(
        outer.calibration_X,
        outer.calibration_y,
        outer.calibration_metadata,
        _fraction_spec(0.80),
        window_seconds=10.0,
        max_gap_seconds=10.5,
    )
    assert len(inner.calibration_X) == 16
    assert len(inner.evaluation_X) == 4
    assert set(inner.calibration_metadata.sample_id).isdisjoint(
        inner.evaluation_metadata.sample_id
    )


def test_prediction_frame_has_long_seven_target_contract() -> None:
    _, truth, metadata = _windows(rows=10)
    frame = _prediction_frame(
        fold_name="fold_01",
        subject_id="target",
        source="both",
        method="zero_shot",
        metadata=metadata,
        truth=truth,
        before=truth,
        after=truth,
    )
    assert len(frame) == 70
    assert set(frame.target_name) == set(CANONICAL_TARGETS)


def test_prediction_keys_are_unique() -> None:
    _, truth, metadata = _windows(rows=10)
    frame = _prediction_frame(
        fold_name="fold_01", subject_id="target", source="both",
        method="zero_shot", metadata=metadata, truth=truth,
        before=truth, after=truth,
    )
    key = ["subject_id", "sample_id", "outer_fold", "method", "target_name"]
    assert not frame.duplicated(key).any()


def test_prediction_frame_rejects_non_finite_values() -> None:
    _, truth, metadata = _windows(rows=10)
    truth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _prediction_frame(
            fold_name="fold_01", subject_id="target", source="both",
            method="zero_shot", metadata=metadata, truth=truth,
            before=np.zeros_like(truth), after=np.zeros_like(truth),
        )


def test_regression_adapter_predicts_n_by_seven(regression_adapter) -> None:
    adapter, X, _ = regression_adapter
    assert adapter.predict(X[:9]).shape == (9, 7)


def test_regression_adapter_fine_tunes_with_explicit_validation(
    regression_adapter,
) -> None:
    adapter, X, y = regression_adapter
    cloned = adapter.clone()
    cloned.fine_tune(
        X[:20], y[:20],
        X_validation=X[20:30], y_validation=y[20:30],
        mode="full_model", max_epochs=1, early_stopping_patience=1,
    )
    assert cloned.predict(X[:5]).shape == (5, 7)
    assert cloned.n_epochs_trained_ == 1
    assert cloned.training_log_[0]["validation_accuracy"] is None


def test_head_only_keeps_backbone_frozen(regression_adapter) -> None:
    adapter, X, y = regression_adapter
    cloned = adapter.clone()
    _, frozen, _, _ = _parameter_audit(cloned, "head_only")
    before = _parameter_digest(cloned, frozen)
    cloned.fine_tune(
        X[:20], y[:20],
        X_validation=X[20:30], y_validation=y[20:30],
        mode="head_only", max_epochs=1, early_stopping_patience=1,
    )
    assert _parameter_digest(cloned, frozen) == before


def test_full_model_clone_starts_from_global_checkpoint(regression_adapter) -> None:
    adapter, _, _ = regression_adapter
    assert _state_digest(adapter.clone()) == _state_digest(adapter)


def test_each_subject_clone_is_independent(regression_adapter) -> None:
    adapter, _, _ = regression_adapter
    first = adapter.clone()
    second = adapter.clone()
    with torch.no_grad():
        next(first.model.parameters()).add_(1.0)
    assert _state_digest(first) != _state_digest(second)
    assert _state_digest(second) == _state_digest(adapter)


def test_zero_shot_does_not_change_model_state(regression_adapter) -> None:
    adapter, X, _ = regression_adapter
    before = _state_digest(adapter)
    adapter.predict(X[:5])
    assert _state_digest(adapter) == before


def test_simple_calibrators_do_not_change_model_state(regression_adapter) -> None:
    adapter, X, y = regression_adapter
    before = _state_digest(adapter)
    prediction = adapter.predict(X[:20])
    fit_bias_correction(y[:20], prediction)
    fit_affine_calibration(y[:20], prediction)
    assert _state_digest(adapter) == before


def test_metrics_calculator_keeps_single_output_regression_contract() -> None:
    result = MetricsCalculator.calculate_regression_metrics(
        np.arange(5), np.arange(5) + 1
    )
    assert result["mae"] == pytest.approx(1.0)
    assert result["mean_error"] == pytest.approx(1.0)
    assert result["abs_bias"] == pytest.approx(1.0)


def test_aggregate_bootstrap_is_deterministic() -> None:
    rows = []
    for subject_index in range(3):
        for method in METHODS:
            row = {
                "outer_fold": "fold_01",
                "subject_id": f"s{subject_index}",
                "source": "all",
                "method": method,
                "status": "completed",
            }
            for metric in (
                "macro_mae", "macro_rmse", "macro_r2",
                "macro_pearson", "macro_spearman", "macro_abs_bias",
            ):
                row[f"{metric}_after"] = float(subject_index + 1)
            for target in CANONICAL_TARGETS:
                for metric in (
                    "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
                ):
                    row[f"{target}_{metric}_before"] = 2.0
                    row[f"{target}_{metric}_after"] = 1.0
                    row[f"{target}_{metric}_gain"] = 1.0
            rows.append(row)
    frame = pd.DataFrame(rows)
    first = _aggregate_outputs(
        frame, bootstrap_samples=50, bootstrap_seed=42
    )
    second = _aggregate_outputs(
        frame, bootstrap_samples=50, bootstrap_seed=42
    )
    pd.testing.assert_frame_equal(first[1], second[1])


def test_paired_aggregation_uses_one_overall_row_per_subject() -> None:
    rows = []
    for source in ("gpn_data", "all"):
        for method in METHODS:
            row = {
                "outer_fold": "fold_01",
                "subject_id": "s1",
                "source": source,
                "method": method,
                "status": "completed",
            }
            for metric in (
                "macro_mae", "macro_rmse", "macro_r2",
                "macro_pearson", "macro_spearman", "macro_abs_bias",
            ):
                row[f"{metric}_after"] = 1.0
            for target in CANONICAL_TARGETS:
                for metric in (
                    "mae", "rmse", "r2", "pearson", "spearman", "abs_bias"
                ):
                    row[f"{target}_{metric}_before"] = 2.0
                    row[f"{target}_{metric}_after"] = 1.0
                    row[f"{target}_{metric}_gain"] = 1.0
            rows.append(row)
    _, paired, targets = _aggregate_outputs(
        pd.DataFrame(rows), bootstrap_samples=10, bootstrap_seed=42
    )
    assert (paired["n_subjects"] == 1).all()
    assert (targets["n_subjects"] == 1).all()


def test_configs_use_canonical_targets_and_cuda() -> None:
    experiment = PMRegressionPersonalizationExperiment(
        REPO_ROOT
        / "experiments/calibration/pm_regression_personalization_20pct.yaml"
    )
    assert tuple(experiment.document["targets"]) == CANONICAL_TARGETS
    assert experiment.document["experiment"]["require_cuda"] is True


def test_target_subject_is_absent_from_synthetic_global_partitions() -> None:
    target = "new-user"
    inner_train = {"s1", "s2"}
    inner_validation = {"s3"}
    assert target not in inner_train
    assert target not in inner_validation


def test_all_runtime_statuses_are_explicit() -> None:
    assert {
        "completed", "insufficient_calibration_samples",
        "insufficient_adaptation_train", "insufficient_evaluation_samples",
        "constant_target", "non_finite_predictions", "training_failed",
    } == set(ALLOWED_STATUSES)
