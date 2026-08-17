import numpy as np
import pytest

from cogstate.preprocessing import (
    EOGRegression,
    OfflinePreprocessingConfig,
    OfflinePreprocessingPipeline,
    WaveletDenoisingConfig,
    baseline_correct_epochs,
    common_average_reference,
    detrend_signal,
    regress_eog,
    robust_average_reference,
    wavelet_denoise,
)


def test_common_average_reference_has_zero_channel_mean() -> None:
    signal = np.arange(60, dtype=float).reshape(20, 3)

    referenced = common_average_reference(signal)

    np.testing.assert_allclose(referenced.mean(axis=1), 0.0, atol=1e-12)


def test_robust_reference_excludes_extreme_channel() -> None:
    rng = np.random.default_rng(4)
    samples = 2048
    time = np.arange(samples) / 256.0
    good = np.column_stack(
        [
            np.sin(2 * np.pi * 10 * time) + 0.05 * rng.normal(size=samples)
            for _ in range(15)
        ]
    )
    signal = np.column_stack((good, 20.0 * rng.normal(size=samples)))

    referenced, report = robust_average_reference(signal, z_threshold=2.5)

    assert report.excluded_channels == (15,)
    np.testing.assert_allclose(referenced[:, :15].mean(axis=1), 0.0, atol=1e-12)


def test_detrend_removes_linear_drift() -> None:
    time = np.linspace(-1.0, 1.0, 500)
    signal = np.column_stack((3.0 * time + np.sin(20 * time), -2.0 * time))

    cleaned = detrend_signal(signal, order=1)

    slopes = [np.polyfit(time, cleaned[:, channel], 1)[0] for channel in range(2)]
    np.testing.assert_allclose(slopes, 0.0, atol=1e-12)


def test_baseline_correction_zeroes_selected_interval() -> None:
    rng = np.random.default_rng(7)
    epochs = rng.normal(size=(5, 100, 4)) + 12.0

    corrected = baseline_correct_epochs(epochs, baseline=slice(0, 20))

    np.testing.assert_allclose(corrected[:, :20].mean(axis=1), 0.0, atol=1e-12)


def test_wavelet_denoising_reduces_seeded_noise_rmse() -> None:
    rng = np.random.default_rng(8)
    samples = 2048
    time = np.arange(samples) / 256.0
    clean = np.sin(2 * np.pi * 10 * time)
    noisy = clean + 0.5 * rng.normal(size=samples)

    denoised, report = wavelet_denoise(
        noisy[:, None], WaveletDenoisingConfig(threshold_method="bayes")
    )

    noisy_rmse = np.sqrt(np.mean((noisy - clean) ** 2))
    denoised_rmse = np.sqrt(np.mean((denoised[:, 0] - clean) ** 2))
    assert denoised_rmse < noisy_rmse
    assert report.level > 0
    assert denoised.shape == (samples, 1)


def test_eog_regression_reduces_ocular_correlation() -> None:
    rng = np.random.default_rng(12)
    samples = 3000
    eog = rng.normal(size=(samples, 2))
    brain = rng.normal(scale=0.2, size=(samples, 4))
    propagation = np.array([[1.0, 0.5, -0.4, 0.2], [0.2, -0.8, 0.3, 0.7]])
    eeg = brain + eog @ propagation

    cleaned, model, report = regress_eog(eeg, eog, ridge_alpha=1e-4)

    assert isinstance(model, EOGRegression)
    assert cleaned.shape == eeg.shape
    assert report.mean_absolute_correlation_after < 0.1 * report.mean_absolute_correlation_before


def test_offline_pipeline_preserves_alignment_and_reports_steps() -> None:
    rng = np.random.default_rng(21)
    samples = 1024
    time = np.arange(samples) / 256.0
    eeg = np.column_stack(
        [
            np.sin(2 * np.pi * frequency * time) + 0.1 * rng.normal(size=samples)
            for frequency in (8, 10, 12, 15)
        ]
    )
    config = OfflinePreprocessingConfig(
        sample_rate=256.0,
        reference_method="common_average",
        detect_and_interpolate_bad_channels=False,
        wavelet_config=WaveletDenoisingConfig(level=3),
    )

    result = OfflinePreprocessingPipeline(config).transform(eeg)

    assert result.values.shape == eeg.shape
    assert np.isfinite(result.values).all()
    assert result.report.input_shape == eeg.shape
    assert result.report.output_shape == eeg.shape
    assert result.report.reference.method == "common_average"
    assert result.report.wavelet is not None


def test_pipeline_requires_fit_before_eog_transform() -> None:
    config = OfflinePreprocessingConfig(
        sample_rate=256.0,
        apply_filter=False,
        use_eog_regression=True,
    )
    pipeline = OfflinePreprocessingPipeline(config)
    eeg = np.ones((100, 4))
    eog = np.ones((100, 1))

    with pytest.raises(RuntimeError, match="Call fit"):
        pipeline.transform(eeg, eog=eog)

