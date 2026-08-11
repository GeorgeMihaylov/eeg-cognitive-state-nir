"""Low-latency feature pipeline for primary real-time inference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cogstate.streaming.buffer import Window

from . import spectral, statistical


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
        spectral_features = spectral.extract_spectral_features(
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
        for base_name in spectral.feature_names(self.config.spectral_config):
            names.extend(f"{base_name}_ch{channel}" for channel in range(n_channels))
        for base_name in statistical.feature_names(self.config.statistical_config):
            names.extend(f"{base_name}_ch{channel}" for channel in range(n_channels))
        return names


def build_lightweight_pipeline(sample_rate: float) -> LightweightFeaturePipeline:
    return LightweightFeaturePipeline(LightweightFeatureConfig(sample_rate))
