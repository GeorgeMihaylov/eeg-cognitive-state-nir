from __future__ import annotations

import numpy as np
import pytest

from cogstate.preprocessing import (
    EOGRegression,
    OfflinePreprocessingConfig,
    OfflinePreprocessingPipeline,
    WaveletDenoisingConfig,
    apply_causal,
    baseline_correct_epochs,
    common_average_reference,
    detrend_signal,
    regress_eog,
    robust_average_reference,
    wavelet_denoise,
)


def test_reference_detrend_and_baseline_operations_are_finite_and_non_mutating() -> None:
    signal = np.arange(60, dtype=float).reshape(20, 3)
    before = signal.copy()
    referenced = common_average_reference(signal)
    np.testing.assert_allclose(referenced.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_array_equal(signal, before)

    time = np.linspace(-1.0, 1.0, 500)
    detrended = detrend_signal(np.column_stack((3.0 * time + np.sin(20 * time), -2.0 * time)))
    np.testing.assert_allclose(
        [np.polyfit(time, detrended[:, channel], 1)[0] for channel in range(2)],
        0.0,
        atol=1e-12,
    )
    epochs = np.random.default_rng(7).normal(size=(5, 100, 4)) + 12.0
    corrected = baseline_correct_epochs(epochs, baseline=slice(0, 20))
    np.testing.assert_allclose(corrected[:, :20].mean(axis=1), 0.0, atol=1e-12)


def test_robust_reference_excludes_extreme_channel() -> None:
    rng = np.random.default_rng(4)
    good = rng.normal(scale=0.1, size=(2048, 15))
    signal = np.column_stack((good, 20.0 * rng.normal(size=2048)))
    referenced, report = robust_average_reference(signal, z_threshold=2.5)
    assert report.excluded_channels == (15,)
    np.testing.assert_allclose(referenced[:, :15].mean(axis=1), 0.0, atol=1e-12)


def test_wavelet_and_eog_regression_are_optional_deterministic_operations() -> None:
    rng = np.random.default_rng(8)
    time = np.arange(2048) / 256.0
    clean = np.sin(2 * np.pi * 10 * time)
    noisy = clean + 0.5 * rng.normal(size=len(time))
    first, report = wavelet_denoise(
        noisy[:, None], WaveletDenoisingConfig(threshold_method="bayes")
    )
    second, _ = wavelet_denoise(
        noisy[:, None], WaveletDenoisingConfig(threshold_method="bayes")
    )
    np.testing.assert_array_equal(first, second)
    assert report.level > 0
    assert np.sqrt(np.mean((first[:, 0] - clean) ** 2)) < np.sqrt(np.mean((noisy - clean) ** 2))

    eog = rng.normal(size=(3000, 2))
    brain = rng.normal(scale=0.2, size=(3000, 4))
    eeg = brain + eog @ np.array([[1.0, 0.5, -0.4, 0.2], [0.2, -0.8, 0.3, 0.7]])
    cleaned, model, eog_report = regress_eog(eeg, eog, ridge_alpha=1e-4)
    assert isinstance(model, EOGRegression)
    assert cleaned.shape == eeg.shape
    assert eog_report.mean_absolute_correlation_after < 0.1 * eog_report.mean_absolute_correlation_before


def test_offline_pipeline_stage_order_and_causal_equivalence() -> None:
    signal = np.random.default_rng(6).normal(size=(1024, 4))
    config = OfflinePreprocessingConfig(
        sample_rate=256.0,
        apply_filter=True,
        filter_mode="causal",
        detrend_order=None,
        reference_method="none",
        detect_and_interpolate_bad_channels=False,
    )
    result = OfflinePreprocessingPipeline(config).transform(signal)
    np.testing.assert_allclose(result.values, apply_causal(signal, config.filter_config), atol=1e-12)
    assert result.report.stages == ("filter:causal", "reference:none")


def test_offline_pipeline_requires_explicit_fit_for_eog_and_keeps_optional_absence_safe() -> None:
    eeg = np.ones((100, 4))
    eog = np.ones((100, 1))
    pipeline = OfflinePreprocessingPipeline(
        OfflinePreprocessingConfig(sample_rate=256.0, apply_filter=False, use_eog_regression=True)
    )
    with pytest.raises(RuntimeError, match="Call fit"):
        pipeline.transform(eeg, eog=eog)

    ordinary = OfflinePreprocessingPipeline(
        OfflinePreprocessingConfig(
            sample_rate=256.0,
            apply_filter=False,
            detrend_order=None,
            reference_method="none",
            detect_and_interpolate_bad_channels=False,
        )
    ).transform(eeg)
    assert ordinary.values.shape == eeg.shape
