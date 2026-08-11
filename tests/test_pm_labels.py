import numpy as np
import pytest

from cogstate.ingestion.pm_labels import (
    PMCleaningConfig,
    TertileDiscretizer,
    clean_pm,
    clean_pm_by_record,
)
from cogstate.protocol import PM_METRICS


N_METRICS = len(PM_METRICS)


def pm_matrix(first_column):
    values = np.full((len(first_column), N_METRICS), 0.5, dtype=float)
    values[:, 0] = first_column
    return values


def test_default_is_non_destructive_but_flags_impulse():
    source = pm_matrix([0.5, 0.5, 0.5, 0.5, 0.9])
    result = clean_pm(source)

    np.testing.assert_array_equal(result.values, source)
    assert result.anomaly_mask[4, 0]
    assert result.summary() == {
        "observations": source.size,
        "invalid": 0,
        "anomalies": 1,
        "retained": source.size,
    }


def test_invalid_and_inactive_values_remain_missing():
    source = pm_matrix([0.5, -0.1, 1.1, np.nan, np.inf])
    result = clean_pm(source)

    assert result.invalid_mask[:, 0].tolist() == [False, True, True, True, True]
    assert np.isnan(result.values[1:, 0]).all()


@pytest.mark.parametrize("policy", ["nan", "local_median"])
def test_outlier_policy_is_explicit(policy):
    source = pm_matrix([0.5, 0.5, 0.5, 0.5, 0.9])
    config = PMCleaningConfig(outlier_policy=policy)
    result = clean_pm(source, config)

    assert result.anomaly_mask[4, 0]
    if policy == "nan":
        assert np.isnan(result.values[4, 0])
    else:
        assert result.values[4, 0] == pytest.approx(0.5)


def test_causal_median_does_not_use_future_values():
    config = PMCleaningConfig(
        mode="causal_median", median_window=3, min_absolute_deviation=1.0
    )
    first = clean_pm(pm_matrix([0.1, 0.2, 0.3, 0.9]), config).values[:, 0]
    changed_future = clean_pm(pm_matrix([0.1, 0.2, 0.3, 0.0]), config).values[:, 0]

    np.testing.assert_allclose(first[:3], [0.1, 0.15, 0.2])
    np.testing.assert_allclose(first[:3], changed_future[:3])


def test_gap_resets_detector_and_smoother_by_default():
    config = PMCleaningConfig(mode="causal_exponential_smoothing", exponential_alpha=0.5)
    result = clean_pm(pm_matrix([0.2, 0.4, np.nan, 0.8]), config)

    np.testing.assert_allclose(result.values[[0, 1, 3], 0], [0.2, 0.3, 0.8])
    assert np.isnan(result.values[2, 0])
    assert not result.anomaly_mask[3, 0]


def test_warmup_disables_outlier_detection_during_adaptive_scaling():
    config = PMCleaningConfig(warmup_samples=5)
    result = clean_pm(pm_matrix([0.5, 0.5, 0.5, 0.9, 0.5, 0.9]), config)

    assert not result.anomaly_mask[3, 0]
    assert result.anomaly_mask[5, 0]


def test_records_never_share_history_even_when_interleaved():
    source = pm_matrix([0.5, 0.1, 0.5, 0.1, 0.5, 0.9])
    result = clean_pm_by_record(source, ["a", "b", "a", "b", "a", "b"])

    assert not result.anomaly_mask[:, 0].any()


def test_tertiles_are_fitted_only_from_training_values():
    train = np.tile(np.linspace(0.0, 1.0, 9)[:, None], (1, N_METRICS))
    discretizer = TertileDiscretizer().fit(train)
    before = discretizer.thresholds_.copy()
    labels = discretizer.transform(np.full((1, N_METRICS), 100.0))

    np.testing.assert_array_equal(discretizer.thresholds_, before)
    np.testing.assert_array_equal(labels, np.full((1, N_METRICS), 2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"outlier_policy": "unknown"},
        {"outlier_window": 0},
        {"min_outlier_history": 8},
        {"exponential_alpha": 0.0},
        {"valid_min": 1.0, "valid_max": 1.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PMCleaningConfig(**kwargs)
