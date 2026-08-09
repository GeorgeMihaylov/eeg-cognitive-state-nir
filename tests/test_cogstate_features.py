from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from cogstate.features import (
    FEATURE_SCHEMA_VERSION,
    ConnectivityConfig,
    EntropyConfig,
    FeaturePipeline,
    FeaturePipelineConfig,
    SpectralConfig,
    StatisticalConfig,
    channel_pairs,
)
from cogstate.features import connectivity, entropy, spectral, statistical


def _sine(frequency: float, *, samples: int = 2560, channels: int = 2) -> np.ndarray:
    time = np.arange(samples, dtype=float) / 256.0
    return np.column_stack(
        [np.sin(2.0 * np.pi * frequency * time + channel * 0.1) for channel in range(channels)]
    )


def _pipeline(**groups: bool) -> FeaturePipeline:
    defaults = {
        "include_spectral": False,
        "include_statistical": False,
        "include_entropy": False,
        "include_connectivity": False,
    }
    defaults.update(groups)
    return FeaturePipeline(FeaturePipelineConfig(sample_rate=256.0, **defaults))


def test_spectral_sine_bands_relative_power_and_determinism() -> None:
    config = SpectralConfig(sample_rate=256.0)
    alpha = spectral.extract_spectral_features(_sine(10.0), config)
    beta = spectral.extract_spectral_features(_sine(20.0), config)

    assert np.all(alpha["power_alpha"] > alpha["power_theta"])
    assert np.all(alpha["power_alpha"] > alpha["power_beta"])
    assert np.all(beta["power_beta"] > beta["power_alpha"])
    assert np.all(beta["power_beta"] > beta["power_theta"])
    relative = np.stack(
        [alpha[f"relpower_{name}"] for name in ("delta", "theta", "alpha", "beta", "gamma")]
    )
    np.testing.assert_allclose(relative.sum(axis=0), 1.0, atol=1e-10)
    assert np.isfinite(relative).all()
    assert np.all((alpha["spectral_edge_frequency"] >= 0.0))
    assert np.all((alpha["spectral_edge_frequency"] <= 128.0))
    repeated = spectral.extract_spectral_features(_sine(10.0), config)
    for name in alpha:
        np.testing.assert_array_equal(alpha[name], repeated[name])


@pytest.mark.parametrize(
    "window",
    [
        np.ones((512, 3)),
        np.ones((512, 3)) + np.linspace(0.0, 1e-10, 512)[:, None],
        np.random.default_rng(42).normal(size=(512, 3)),
    ],
)
def test_statistical_features_are_finite_for_stable_edge_cases(window: np.ndarray) -> None:
    features = statistical.extract_statistical_features(window, StatisticalConfig())
    assert list(features) == statistical.feature_names(StatisticalConfig())
    assert all(value.shape == (3,) for value in features.values())
    assert all(np.isfinite(value).all() for value in features.values())


def test_constant_signal_has_zero_hjorth_activity_without_nan() -> None:
    features = statistical.extract_statistical_features(
        np.ones((512, 2)), StatisticalConfig()
    )
    np.testing.assert_allclose(features["hjorth_activity"], 0.0, atol=1e-15)
    np.testing.assert_allclose(features["hjorth_mobility"], 0.0, atol=1e-15)
    np.testing.assert_allclose(features["hjorth_complexity"], 0.0, atol=1e-15)


def test_entropy_noise_exceeds_periodic_for_spectral_and_permutation() -> None:
    periodic = _sine(10.0, samples=1024, channels=1)
    noise = np.random.default_rng(42).normal(size=(1024, 1))
    config = EntropyConfig(sample_rate=256.0, include_sample_entropy=False)
    periodic_features = entropy.extract_entropy_features(periodic, config)
    noise_features = entropy.extract_entropy_features(noise, config)

    assert noise_features["spectral_entropy"][0] > periodic_features["spectral_entropy"][0]
    assert noise_features["permutation_entropy"][0] > periodic_features["permutation_entropy"][0]
    assert all(np.isfinite(value).all() for value in noise_features.values())


def test_sample_entropy_toggle_changes_only_sample_entropy_block() -> None:
    window = np.random.default_rng(42).normal(size=(384, 2))
    off = EntropyConfig(sample_rate=256.0, include_sample_entropy=False)
    on = EntropyConfig(sample_rate=256.0, include_sample_entropy=True)
    off_features = entropy.extract_entropy_features(window, off)
    on_features = entropy.extract_entropy_features(window, on)

    assert entropy.feature_names(off) == ["spectral_entropy", "permutation_entropy"]
    assert entropy.feature_names(on) == [
        "spectral_entropy",
        "permutation_entropy",
        "sample_entropy",
    ]
    for name in off_features:
        np.testing.assert_array_equal(off_features[name], on_features[name])
    assert np.isfinite(on_features["sample_entropy"]).all()


def test_connectivity_uses_all_ninety_one_pairs_in_deterministic_order() -> None:
    pairs = channel_pairs(14)
    assert len(pairs) == 91
    assert pairs[:4] == ((0, 1), (0, 2), (0, 3), (0, 4))
    assert pairs[-1] == (12, 13)
    assert pairs == channel_pairs(14)


def test_connectivity_identical_signals_have_high_correlation_and_plv() -> None:
    base = _sine(10.0, samples=1024, channels=1)[:, 0]
    window = np.column_stack([base, base])
    config = ConnectivityConfig(sample_rate=256.0)
    features = connectivity.extract_connectivity_features(window, config)
    assert features["correlation_mean"][0] == pytest.approx(1.0, abs=1e-12)
    assert features["plv_mean"][0] == pytest.approx(1.0, abs=1e-12)


def test_independent_signals_have_lower_connectivity_than_synchronized() -> None:
    rng = np.random.default_rng(42)
    independent = rng.normal(size=(2048, 2))
    synchronized = np.column_stack([independent[:, 0], independent[:, 0]])
    config = ConnectivityConfig(sample_rate=256.0, metrics=("correlation", "plv"))
    first = connectivity.extract_connectivity_features(independent, config)
    second = connectivity.extract_connectivity_features(synchronized, config)
    assert abs(first["correlation_mean"][0]) < second["correlation_mean"][0]
    assert first["plv_mean"][0] < second["plv_mean"][0]


def test_connectivity_constant_signal_remains_finite() -> None:
    features = connectivity.extract_connectivity_features(
        np.ones((512, 3)), ConnectivityConfig(sample_rate=256.0)
    )
    assert all(np.isfinite(value).all() for value in features.values())


def test_connectivity_summary_ignores_uncomputed_matrix_entries() -> None:
    matrix = np.full((4, 4), np.nan)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = 0.75
    matrix[0, 2] = matrix[2, 0] = 0.25

    summary = connectivity.summarize_connectivity_matrix(
        matrix, computed_pairs=((0, 1), (0, 2))
    )
    assert summary == pytest.approx({"mean": 0.5, "std": 0.25, "max": 0.75})


def test_coherence_is_computed_once_per_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = connectivity.signal.coherence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(connectivity.signal, "coherence", counted)
    window = np.random.default_rng(42).normal(size=(512, 4))
    features = connectivity.extract_connectivity_features(
        window, ConnectivityConfig(sample_rate=256.0, metrics=("coherence",))
    )
    assert calls == len(channel_pairs(4)) == 6
    assert len(features) == 5 * 3
    assert all(np.isfinite(value).all() for value in features.values())


def test_plv_computes_hilbert_once_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = connectivity.signal.hilbert

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(connectivity.signal, "hilbert", counted)
    connectivity.extract_connectivity_features(
        np.random.default_rng(42).normal(size=(512, 4)),
        ConnectivityConfig(sample_rate=256.0, metrics=("plv",)),
    )
    assert calls == 1


@pytest.mark.parametrize(
    "group,expected_prefix",
    [
        ("spectral", "spectral__"),
        ("statistical", "statistical__"),
        ("entropy", "entropy__"),
        ("connectivity", "connectivity__"),
    ],
)
def test_pipeline_supports_each_group_independently(
    group: str, expected_prefix: str
) -> None:
    pipeline = _pipeline(**{f"include_{group}": True})
    window = np.random.default_rng(42).normal(size=(512, 3))
    vector = pipeline.transform_window(window)
    names = pipeline.feature_names(3)
    assert len(vector) == len(names) > 0
    assert all(name.startswith(expected_prefix) for name in names)
    assert np.isfinite(vector).all()


def test_pipeline_all_groups_profile_dimension_schema_and_hash() -> None:
    profile = yaml.safe_load(
        Path("experiments/features/preliminary_model_zoo_features_v1.json").read_text(
            encoding="utf-8"
        )
    )
    pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(profile))
    names = pipeline.feature_names()
    specification = pipeline.feature_specification()

    assert profile["schema_version"] == FEATURE_SCHEMA_VERSION
    assert len(names) == specification["n_features"] == 371
    assert not any("engagement_index" in name for name in names)
    assert len(specification["connectivity"]["channel_pairs"]) == 91
    assert specification["connectivity"]["pair_policy"] == "all_unique_unordered"
    assert pipeline.feature_hash() == pipeline.feature_hash()


def test_feature_hash_is_independent_of_input_mapping_order() -> None:
    first = SpectralConfig(sample_rate=256.0)
    second = SpectralConfig(
        sample_rate=256.0,
        bands=dict(reversed(list(first.bands.items()))),
    )
    first_pipeline = FeaturePipeline(
        FeaturePipelineConfig(
            sample_rate=256.0,
            channel_names=("A", "B"),
            include_statistical=False,
            include_entropy=False,
            include_connectivity=False,
            spectral_config=first,
        )
    )
    second_pipeline = FeaturePipeline(
        FeaturePipelineConfig(
            sample_rate=256.0,
            channel_names=("A", "B"),
            include_statistical=False,
            include_entropy=False,
            include_connectivity=False,
            spectral_config=second,
        )
    )
    assert first_pipeline.feature_names() == second_pipeline.feature_names()
    assert first_pipeline.feature_hash() == second_pipeline.feature_hash()


def test_sample_entropy_adds_one_feature_per_channel() -> None:
    off = _pipeline(include_entropy=True)
    on = FeaturePipeline(
        FeaturePipelineConfig(
            sample_rate=256.0,
            include_spectral=False,
            include_statistical=False,
            include_entropy=True,
            include_connectivity=False,
            entropy_config=EntropyConfig(
                sample_rate=256.0, include_sample_entropy=True
            ),
        )
    )
    assert len(on.feature_names(14)) - len(off.feature_names(14)) == 14


def test_batch_transform_matches_stacked_single_windows() -> None:
    windows = np.random.default_rng(42).normal(size=(3, 384, 3))
    pipeline = _pipeline(include_spectral=True, include_statistical=True)
    batch = pipeline.transform_batch(windows, chunk_size=2)
    expected = np.stack([pipeline.transform_window(window) for window in windows])
    np.testing.assert_allclose(batch, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        (np.zeros(64), ValueError, "samples, channels"),
        (np.zeros((4, 2)), ValueError, "at least 8 samples"),
        (np.full((16, 2), np.nan), ValueError, "NaN or Inf"),
        (np.full((16, 2), np.inf), ValueError, "NaN or Inf"),
    ],
)
def test_pipeline_rejects_invalid_input(invalid, exception, message) -> None:
    with pytest.raises(exception, match=message):
        _pipeline(include_statistical=True).transform_window(invalid)


def test_band_above_nyquist_is_rejected() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SpectralConfig(sample_rate=64.0)


def test_cogstate_features_does_not_import_bench_or_model_zoo() -> None:
    for path in Path("cogstate/features").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert roots.isdisjoint({"bench", "model_zoo"}), path


def test_cogstate_package_does_not_import_bench() -> None:
    for path in Path("cogstate").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert "bench" not in roots, path
