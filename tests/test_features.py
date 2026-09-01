import numpy as np

from cogstate.features.connectivity import (
    ConnectivityConfig,
    compute_coherence_matrix,
    compute_plv_matrix,
    extract_connectivity_features,
    feature_names as connectivity_feature_names,
    summarize_connectivity_matrix,
)
from cogstate.features.pipeline import build_default_pipeline
from cogstate.features.spectral import (
    SpectralConfig,
    compute_band_ratios,
    compute_spectral_edge_frequency,
    extract_spectral_features,
)
from cogstate.features.statistical import StatisticalConfig, extract_statistical_features
from cogstate.streaming.buffer import Window


def test_engagement_index_uses_beta_over_alpha_plus_theta() -> None:
    powers = {
        "theta": np.array([2.0]),
        "alpha": np.array([3.0]),
        "beta": np.array([10.0]),
    }

    ratios = compute_band_ratios(powers)

    np.testing.assert_allclose(ratios["engagement_index"], [2.0])


def test_spectral_edge_ignores_power_outside_configured_bands() -> None:
    sample_rate = 256.0
    time = np.arange(2048) / sample_rate
    window = (
        np.sin(2 * np.pi * 10 * time)
        + 20.0 * np.sin(2 * np.pi * 60 * time)
    )[:, None]
    config = SpectralConfig(
        sample_rate=sample_rate,
        spectral_edge_band_hz=(1.0, 45.0),
    )

    edge = compute_spectral_edge_frequency(window, config)

    assert 8.0 <= edge[0] <= 45.0


def test_uncomputed_connectivity_pairs_do_not_bias_summary() -> None:
    matrix = np.full((4, 4), np.nan)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = 0.75

    summary = summarize_connectivity_matrix(matrix)

    assert summary == {"mean": 0.75, "std": 0.0, "max": 0.75}


def test_pair_budget_marks_only_measured_pairs_as_finite() -> None:
    rng = np.random.default_rng(1)
    window = rng.normal(size=(512, 8))
    config = ConnectivityConfig(sample_rate=128.0, max_channel_pairs=5)

    matrix = compute_coherence_matrix(window, config, config.bands["alpha"])
    upper = matrix[np.triu_indices(8, k=1)]

    assert np.isfinite(upper).sum() == 5


def test_plv_is_computed_separately_for_frequency_bands() -> None:
    rng = np.random.default_rng(2)
    sample_rate = 256.0
    time = np.arange(2048) / sample_rate
    shared_alpha = np.sin(2 * np.pi * 10 * time)
    window = np.column_stack(
        (
            shared_alpha + 0.8 * np.sin(2 * np.pi * 20 * time),
            shared_alpha + 0.8 * np.sin(2 * np.pi * 23 * time + rng.uniform()),
        )
    )
    config = ConnectivityConfig(sample_rate=sample_rate, max_channel_pairs=None)

    alpha = compute_plv_matrix(window, config, config.bands["alpha"])[0, 1]
    beta = compute_plv_matrix(window, config, config.bands["beta"])[0, 1]

    assert alpha > 0.95
    assert alpha > beta


def test_full_feature_vector_matches_declared_schema() -> None:
    rng = np.random.default_rng(4)
    signal = rng.normal(size=(256, 4))
    pipeline = build_default_pipeline(128.0)
    window = Window(0.0, 2.0, {"eeg": signal}, {"eeg": np.arange(256) / 128.0})

    values = pipeline(signal, window)
    names = pipeline.feature_names(4)
    connectivity = extract_connectivity_features(
        signal, ConnectivityConfig(sample_rate=128.0)
    )

    assert len(values) == len(names)
    assert list(connectivity) == connectivity_feature_names(
        ConnectivityConfig(sample_rate=128.0)
    )
    assert np.isfinite(values).all()


def test_spectral_extractor_returns_finite_values_for_constant_signal() -> None:
    signal = np.ones((512, 3))

    features = extract_spectral_features(signal, SpectralConfig(sample_rate=128.0))

    assert all(np.isfinite(values).all() for values in features.values())


def test_statistical_extractor_returns_finite_values_for_constant_signal() -> None:
    features = extract_statistical_features(
        np.ones((512, 3)), StatisticalConfig()
    )

    assert all(np.isfinite(values).all() for values in features.values())
