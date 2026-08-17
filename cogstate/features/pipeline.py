"""
pipeline.py — сборка всех групп признаков в единый вектор (10.2.3).

Реализует интерфейс FeatureExtractor из streaming/stream_processor.py
(вызывается как extract_features(clean_signal, window) -> np.ndarray).

Порядок конкатенации фиксирован в feature_names() — тот же порядок
должен использоваться при обучении моделей (10.2.4) и при отборе
признаков (selection.py), иначе индексы FeatureSelector разъедутся
с реальными столбцами вектора.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from cogstate.streaming.buffer import Window

from . import connectivity, entropy, spectral, statistical


@dataclass
class FeaturePipelineConfig:
    sample_rate: float
    spectral_config: spectral.SpectralConfig = field(init=False)
    statistical_config: statistical.StatisticalConfig = field(default_factory=statistical.StatisticalConfig)
    entropy_config: entropy.EntropyConfig = field(init=False)
    connectivity_config: connectivity.ConnectivityConfig = field(init=False)

    def __post_init__(self):
        self.spectral_config = spectral.SpectralConfig(sample_rate=self.sample_rate)
        self.entropy_config = entropy.EntropyConfig(sample_rate=self.sample_rate)
        self.connectivity_config = connectivity.ConnectivityConfig(sample_rate=self.sample_rate)


def _flatten_per_channel(features: Dict[str, np.ndarray]) -> np.ndarray:
    """{имя: [n_channels]} -> плоский вектор [n_features * n_channels], порядок = порядок ключей словаря."""
    return np.concatenate([values for values in features.values()])


def _flatten_scalar(features: Dict[str, np.ndarray]) -> np.ndarray:
    """{имя: [1]} -> плоский вектор [n_features] (связностные признаки — уже сводки уровня окна)."""
    return np.concatenate([values for values in features.values()])


class FeaturePipeline:
    """
    Объект, реализующий интерфейс FeatureExtractor из
    stream_processor.py (вызывается как pipeline(clean_signal, window)).
    """

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config

    def __call__(self, clean_signal: np.ndarray, window: Window) -> np.ndarray:
        power_spectrum = spectral.compute_power_spectrum(
            clean_signal, self.config.spectral_config
        )
        spectral_features = spectral.extract_spectral_features(
            clean_signal,
            self.config.spectral_config,
            spectrum=power_spectrum,
        )
        statistical_features = statistical.extract_statistical_features(clean_signal, self.config.statistical_config)
        entropy_features = entropy.extract_entropy_features(
            clean_signal,
            self.config.entropy_config,
            spectrum=power_spectrum,
        )
        connectivity_features = connectivity.extract_connectivity_features(clean_signal, self.config.connectivity_config)

        return np.concatenate([
            _flatten_per_channel(spectral_features),
            _flatten_per_channel(statistical_features),
            _flatten_per_channel(entropy_features),
            _flatten_scalar(connectivity_features),
        ])

    def feature_names(self, n_channels: int) -> List[str]:
        names: List[str] = []
        for base_name in spectral.feature_names(self.config.spectral_config):
            names += [f"{base_name}_ch{ch}" for ch in range(n_channels)]
        for base_name in statistical.feature_names(self.config.statistical_config):
            names += [f"{base_name}_ch{ch}" for ch in range(n_channels)]
        for base_name in entropy.feature_names(self.config.entropy_config):
            names += [f"{base_name}_ch{ch}" for ch in range(n_channels)]
        names += connectivity.feature_names(self.config.connectivity_config)
        return names


def build_default_pipeline(sample_rate: float) -> FeaturePipeline:
    return FeaturePipeline(FeaturePipelineConfig(sample_rate=sample_rate))
