from __future__ import annotations

import numpy as np
import pytest

from cogstate.preprocessing import ArtifactICA, FullFasterConfig, IcaConfig, run_faster_full
from cogstate.preprocessing.full_faster import (
    detect_bad_channel_epoch_pairs,
    detect_bad_channels,
    interpolate_channels,
)


def test_global_bad_channel_and_channel_epoch_are_detected() -> None:
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(4096, 16))
    signal[:, 15] *= 30.0
    assert 15 in detect_bad_channels(
        signal, FullFasterConfig(z_threshold=2.5, run_ica=False)
    )

    epochs = rng.normal(size=(5, 512, 16))
    epochs[3, :, 12] *= 25.0
    pairs = detect_bad_channel_epoch_pairs(
        epochs, FullFasterConfig(z_threshold=2.5, run_ica=False)
    )
    assert (3, 12) in pairs


def test_spherical_interpolation_preserves_good_channels_and_is_explicit() -> None:
    rng = np.random.default_rng(3)
    channels = 12
    angles = np.linspace(0, 2 * np.pi, channels, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles), np.full(channels, 0.5)))
    signal = rng.normal(size=(200, channels))
    signal[:, 4] = 1000.0
    config = FullFasterConfig(interpolation_method="spherical", run_ica=False)
    cleaned = interpolate_channels(signal, [4], positions, config=config)
    np.testing.assert_allclose(cleaned[:, np.arange(channels) != 4], signal[:, np.arange(channels) != 4])
    assert np.isfinite(cleaned[:, 4]).all()
    with pytest.raises(ValueError, match="channel_positions"):
        interpolate_channels(signal, [4], config=config)


def test_full_faster_maps_retained_and_original_epoch_indices() -> None:
    rng = np.random.default_rng(10)
    epochs = rng.normal(size=(24, 128, 12))
    epochs[23] *= 20.0
    config = FullFasterConfig(z_threshold=2.8, run_ica=False)
    cleaned, report = run_faster_full(epochs, config)
    assert 23 in report.bad_epochs
    assert cleaned.shape[0] == len(report.kept_epoch_indices)
    assert report.kept_epoch_mask.shape == (24,)
    assert report.kept_epoch_mask.sum() == len(cleaned)
    assert all(pair[0] in report.kept_epoch_indices for pair in report.bad_channel_epoch_pairs_original)


def test_full_faster_ica_reports_rank_components_and_convergence() -> None:
    rng = np.random.default_rng(11)
    sources = rng.laplace(size=(16 * 96, 6))
    mixing = rng.normal(size=(6, 8))
    epochs = (sources @ mixing).reshape(16, 96, 8)
    config = FullFasterConfig(
        z_threshold=3.5, run_ica=True, ica_n_components=8, ica_max_iter=1000
    )
    cleaned, report = run_faster_full(epochs, config, sample_rate=128.0)
    assert cleaned.shape[2] == 8
    assert report.ica_fitted
    assert report.ica_input_rank == 6
    assert report.ica_n_components == 6
    assert isinstance(report.ica_converged, bool)
    assert set(report.component_bads_by_metric) >= {
        "spatial_kurtosis", "hurst", "power_gradient", "median_gradient"
    }


def test_full_faster_is_deterministic_and_requires_sample_rate_for_ica() -> None:
    epochs = np.random.default_rng(12).normal(size=(10, 80, 5))
    config = FullFasterConfig(run_ica=False)
    first, first_report = run_faster_full(epochs, config)
    second, second_report = run_faster_full(epochs, config)
    np.testing.assert_array_equal(first, second)
    assert first_report == second_report
    with pytest.raises(ValueError, match="sample_rate"):
        run_faster_full(epochs, FullFasterConfig(run_ica=True))


def test_single_rank_safe_ica_can_use_full_faster_component_metrics() -> None:
    rng = np.random.default_rng(13)
    signal = rng.laplace(size=(2400, 6)) @ rng.normal(size=(6, 6))
    config = IcaConfig(
        n_components=6,
        max_iter=1000,
        faster_config=FullFasterConfig(run_ica=False),
        component_metric_profile="full_faster",
    )
    ica = ArtifactICA(config).fit(signal, sample_rate=128.0)
    assert ica.input_rank == 6
    assert ica.n_components == 6
    assert isinstance(ica.converged, bool)
    assert np.isfinite(ica.transform(signal[:100])).all()
