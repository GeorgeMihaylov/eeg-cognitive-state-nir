"""Low-latency feature pipeline for primary real-time inference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cogstate.streaming.buffer import Window

from . import connectivity, entropy, spectral, statistical


def _extract_streaming_spectral(
    signal: np.ndarray,
    config: spectral.SpectralConfig,
) -> dict[str, np.ndarray]:
    """Preserve the deployed streaming-v1 schema on validated primitives."""
    features = dict(spectral.extract_spectral_features(signal, config))
    edge = features.pop("spectral_edge_frequency")
    epsilon = np.finfo(float).eps
    features["engagement_index"] = (
        features["power_theta"] + features["power_alpha"]
    ) / (features["power_alpha"] + features["power_beta"] + epsilon)
    features["spectral_edge_freq"] = edge
    return features


def _streaming_spectral_names(config: spectral.SpectralConfig) -> list[str]:
    names = spectral.feature_names(config)
    edge_index = names.index("spectral_edge_frequency")
    names[edge_index:edge_index + 1] = [
        "engagement_index",
        "spectral_edge_freq",
    ]
    return names


@dataclass
class LightweightFeatureConfig:
    sample_rate: float
    spectral_config: spectral.SpectralConfig = field(init=False)
    statistical_config: statistical.StatisticalConfig = field(
        default_factory=statistical.StatisticalConfig
    )

    def __post_init__(self) -> None:
        self.spectral_config = spectral.SpectralConfig(sample_rate=self.sample_rate)


class LightweightFeaturePipeline:
    """Spectral and statistical features without expensive entropy/connectivity."""

    def __init__(self, config: LightweightFeatureConfig) -> None:
        self.config = config

    def __call__(self, clean_signal: np.ndarray, window: Window) -> np.ndarray:
        spectral_features = _extract_streaming_spectral(
            clean_signal, self.config.spectral_config
        )
        statistical_features = statistical.extract_statistical_features(
            clean_signal, self.config.statistical_config
        )
        return np.concatenate(
            [
                *spectral_features.values(),
                *statistical_features.values(),
            ]
        )

    def feature_names(self, n_channels: int) -> list[str]:
        names: list[str] = []
        for base_name in _streaming_spectral_names(self.config.spectral_config):
            names.extend(f"{base_name}_ch{channel}" for channel in range(n_channels))
        for base_name in statistical.feature_names(self.config.statistical_config):
            names.extend(f"{base_name}_ch{channel}" for channel in range(n_channels))
        return names


def build_lightweight_pipeline(sample_rate: float) -> LightweightFeaturePipeline:
    return LightweightFeaturePipeline(LightweightFeatureConfig(sample_rate))


@dataclass
class FullStreamingFeatureConfig:
    sample_rate: float
    spectral_config: spectral.SpectralConfig = field(init=False)
    statistical_config: statistical.StatisticalConfig = field(
        default_factory=statistical.StatisticalConfig
    )
    entropy_config: entropy.EntropyConfig = field(init=False)
    connectivity_config: connectivity.ConnectivityConfig = field(init=False)

    def __post_init__(self) -> None:
        self.spectral_config = spectral.SpectralConfig(sample_rate=self.sample_rate)
        self.entropy_config = entropy.EntropyConfig(
            sample_rate=self.sample_rate,
            include_sample_entropy=True,
        )
        self.connectivity_config = connectivity.ConnectivityConfig(
            sample_rate=self.sample_rate,
            max_channel_pairs=50,
        )


class FullStreamingFeaturePipeline:
    """Compatibility profile for the immutable 399-feature streaming bundle."""

    def __init__(self, config: FullStreamingFeatureConfig) -> None:
        self.config = config

    def __call__(self, clean_signal: np.ndarray, window: Window) -> np.ndarray:
        del window
        groups = (
            _extract_streaming_spectral(clean_signal, self.config.spectral_config),
            statistical.extract_statistical_features(
                clean_signal, self.config.statistical_config
            ),
            entropy.extract_entropy_features(clean_signal, self.config.entropy_config),
            connectivity.extract_connectivity_features(
                clean_signal, self.config.connectivity_config
            ),
        )
        return np.concatenate(
            [np.asarray(value, dtype=float) for group in groups for value in group.values()]
        )

    def feature_names(self, n_channels: int) -> list[str]:
        names: list[str] = []
        per_channel_groups = (
            _streaming_spectral_names(self.config.spectral_config),
            statistical.feature_names(self.config.statistical_config),
            entropy.feature_names(self.config.entropy_config),
        )
        for group_names in per_channel_groups:
            for base_name in group_names:
                names.extend(
                    f"{base_name}_ch{channel}" for channel in range(n_channels)
                )
        names.extend(connectivity.feature_names(self.config.connectivity_config))
        return names


def build_streaming_full_pipeline(sample_rate: float) -> FullStreamingFeaturePipeline:
    return FullStreamingFeaturePipeline(FullStreamingFeatureConfig(sample_rate))
