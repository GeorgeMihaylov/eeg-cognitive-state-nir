import numpy as np
import pytest

from cogstate.preprocessing import (
    FasterConfig,
    detect_bad_channel_epoch_pairs,
    detect_bad_channels,
    interpolate_channels,
    run_faster,
)


def test_global_bad_channel_is_detected_by_variance() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(4096, 16))
    signal[:, 15] *= 30.0

    bad = detect_bad_channels(signal, FasterConfig(z_threshold=2.5, run_ica=False))

    assert 15 in bad


def test_channel_in_epoch_detection_is_per_epoch_across_channels() -> None:
    rng = np.random.default_rng(2)
    epochs = rng.normal(size=(5, 512, 16))
    epochs[3, :, 12] *= 25.0

    pairs = detect_bad_channel_epoch_pairs(
        epochs, FasterConfig(z_threshold=2.5, run_ica=False)
    )

    assert (3, 12) in pairs


def test_spherical_interpolation_preserves_good_channels() -> None:
    rng = np.random.default_rng(3)
    samples, channels = 200, 12
    angles = np.linspace(0, 2 * np.pi, channels, endpoint=False)
    positions = np.column_stack(
        (np.cos(angles), np.sin(angles), np.full(channels, 0.5))
    )
    signal = rng.normal(size=(samples, channels))
    signal[:, 4] = 1000.0
    config = FasterConfig(interpolation_method="spherical", run_ica=False)

    cleaned = interpolate_channels(signal, [4], positions, config=config)

    np.testing.assert_allclose(cleaned[:, np.arange(channels) != 4], signal[:, np.arange(channels) != 4])
    assert np.isfinite(cleaned[:, 4]).all()
    assert not np.allclose(cleaned[:, 4], signal[:, 4])


def test_complete_faster_runs_all_four_stages_and_aligns_labels() -> None:
    rng = np.random.default_rng(10)
    sources = rng.laplace(size=(24 * 128, 12))
    mixing = rng.normal(size=(12, 12))
    epochs = (sources @ mixing.T).reshape(24, 128, 12)
    epochs[23] *= 20.0
    config = FasterConfig(
        z_threshold=2.8,
        run_ica=True,
        ica_n_components=8,
        ica_max_iter=1000,
    )

    cleaned, report = run_faster(epochs, config, sample_rate=128.0)

    assert report.ica_fitted
    assert 23 in report.bad_epochs
    assert cleaned.shape == (len(report.kept_epoch_indices), 128, 12)
    assert report.kept_epoch_mask.shape == (24,)
    assert report.kept_epoch_mask.sum() == len(cleaned)
    np.testing.assert_allclose(cleaned.mean(axis=2), 0.0, atol=1e-12)


def test_full_faster_requires_sample_rate_for_ica() -> None:
    epochs = np.ones((4, 100, 4))

    with pytest.raises(ValueError, match="sample_rate"):
        run_faster(epochs, FasterConfig(run_ica=True))
