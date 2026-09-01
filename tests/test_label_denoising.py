"""Tests for causal and advanced PM label denoising."""

import numpy as np
import pytest

from cogstate.ingestion.label_denoising import (
    HuberTrendConfig,
    RobustKalmanConfig,
    denoise_labels,
    denoise_pm_by_record,
    huber_trend_pm,
    robust_kalman_pm,
)
from cogstate.protocol import PM_METRICS


N_METRICS = len(PM_METRICS)


def pm_matrix(first_column):
    values = np.full((len(first_column), N_METRICS), 0.5, dtype=float)
    values[:, 0] = first_column
    return values


def test_simple_causal_smoothing_remains_available_after_merge():
    values = np.array([0.2, 0.4, np.nan, 0.8])

    median = denoise_labels(values, mode="causal_median", window=3)
    exponential = denoise_labels(
        values, mode="causal_exponential_smoothing", alpha=0.5
    )

    np.testing.assert_allclose(median[[0, 1, 3]], [0.2, 0.3, 0.8])
    np.testing.assert_allclose(exponential[[0, 1, 3]], [0.2, 0.3, 0.8])
    assert np.isnan(median[2]) and np.isnan(exponential[2])


def test_robust_kalman_suppresses_and_marks_impulse():
    source = pm_matrix([0.5] * 8 + [1.0] + [0.5] * 8)
    config = RobustKalmanConfig(
        process_variance=1e-4,
        observation_variance=1e-3,
        outlier_weight_threshold=0.5,
    )
    result = robust_kalman_pm(source, config)

    assert result.converged
    assert result.anomaly_mask[8, 0]
    assert result.values[8, 0] < 0.6
    assert result.confidence[8, 0] < result.confidence[7, 0]


def test_robust_kalman_preserves_missing_gap_and_restarts():
    source = pm_matrix([0.2, 0.2, np.nan, 0.8, 0.8])
    result = robust_kalman_pm(source)

    assert np.isnan(result.values[2, 0])
    assert result.values[1, 0] == pytest.approx(0.2)
    assert result.values[3, 0] == pytest.approx(0.8)


def test_huber_trend_suppresses_impulse_but_preserves_level_shift():
    clean = np.r_[np.full(15, 0.2), np.full(15, 0.8)]
    noisy = clean.copy()
    noisy[7] = 1.0
    config = HuberTrendConfig(
        huber_delta=0.05,
        first_order_penalty=0.02,
        second_order_penalty=0.1,
        max_iterations=1000,
    )
    result = huber_trend_pm(pm_matrix(noisy), config)

    assert result.converged
    assert result.anomaly_mask[7, 0]
    assert result.values[7, 0] < 0.35
    assert np.mean(result.values[10:15, 0]) < 0.35
    assert np.mean(result.values[15:20, 0]) > 0.65


def test_huber_trend_constant_series_is_unchanged():
    source = pm_matrix(np.full(12, 0.4))
    result = huber_trend_pm(source)

    np.testing.assert_allclose(result.values[:, 0], 0.4, atol=1e-5)
    assert not result.anomaly_mask[:, 0].any()


@pytest.mark.parametrize("method", ["robust_kalman", "huber_trend"])
def test_advanced_methods_do_not_share_state_between_records(method):
    source = pm_matrix([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
    result = denoise_pm_by_record(
        source, ["a", "a", "a", "b", "b", "b"], method=method
    )

    assert result.values[2, 0] < 0.3
    assert result.values[3, 0] > 0.7


def test_invalid_values_are_reported_and_not_imputed():
    source = pm_matrix([0.5, -0.1, 1.1, np.inf])
    for cleaner in (robust_kalman_pm, huber_trend_pm):
        result = cleaner(source)
        assert result.invalid_mask[:, 0].tolist() == [False, True, True, True]
        assert np.isnan(result.values[1:, 0]).all()


def test_wrong_advanced_config_type_is_rejected():
    with pytest.raises(TypeError):
        denoise_pm_by_record(
            pm_matrix([0.5]),
            ["a"],
            method="robust_kalman",
            config=HuberTrendConfig(),
        )


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (RobustKalmanConfig, {"process_variance": 0}),
        (RobustKalmanConfig, {"degrees_of_freedom": 0}),
        (HuberTrendConfig, {"huber_delta": 0}),
        (HuberTrendConfig, {"first_order_penalty": 0, "second_order_penalty": 0}),
    ],
)
def test_invalid_advanced_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
