from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, zscore
from sklearn.decomposition import FastICA


@dataclass
class FasterConfig:
    z_threshold: float = 3.0                 # порог z-score для всех 4 уровней
    interpolate_bad_channels: bool = True
    interpolate_bad_channel_epoch: bool = True
    hurst_max_lag: int = 100                 # максимальный лаг для оценки показателя Хёрста
    spectral_slope_band_hz: Tuple[float, float] = (8.0, 45.0)  # диапазон для оценки наклона спектра


def hurst_exponent(x: np.ndarray, max_lag: int = 100) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return 0.5

    max_lag = min(max_lag, n // 2)
    lags = np.arange(2, max_lag)
    if len(lags) < 2:
        return 0.5

    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 2:
        return 0.5

    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(slope)  # ~H, чем ближе к 0.5 — тем "нормальнее" для EEG-подобного сигнала


def spectral_slope(x: np.ndarray, sample_rate: float, band_hz: Tuple[float, float]) -> float:
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 256))
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1]) & (psd > 0)
    if mask.sum() < 2:
        return 0.0
    slope, _ = np.polyfit(np.log(freqs[mask]), np.log(psd[mask]), 1)
    return float(slope)


def _safe_zscore(features: np.ndarray) -> np.ndarray:
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    return (features - np.mean(features, axis=0)) / std


def compute_channel_stats(signal: np.ndarray, config: FasterConfig) -> np.ndarray:
    variance = np.var(signal, axis=0)
    channel_hurst = np.array([
        hurst_exponent(signal[:, ch], config.hurst_max_lag)
        for ch in range(signal.shape[1])
    ])

    mean_other = (np.sum(signal, axis=1, keepdims=True) - signal) / max(signal.shape[1] - 1, 1)
    correlations = np.array([
        np.corrcoef(signal[:, ch], mean_other[:, ch])[0, 1]
        for ch in range(signal.shape[1])
    ])
    correlations = np.nan_to_num(correlations, nan=0.0)

    return np.stack([variance, correlations, channel_hurst], axis=1)


def detect_bad_channels(signal: np.ndarray, config: FasterConfig) -> List[int]:
    features = compute_channel_stats(signal, config)
    z = _safe_zscore(features)
    bad_mask = np.any(np.abs(z) > config.z_threshold, axis=1)
    return list(np.where(bad_mask)[0])


def interpolate_channels(signal: np.ndarray, bad_channels: List[int]) -> np.ndarray:
    """
    Интерполяция плохих каналов средним по соседним "хорошим" каналам.
    В оригинальном FASTER используется интерполяция по сферическим
    сплайнам с учётом расположения электродов (spherical spline
    interpolation, требует монтажа/coordinates) — здесь упрощённый
    вариант без геометрии электродов, достаточный для потокового
    прототипа, где полный монтаж не всегда доступен (10.3 требует
    переносимости между разными гарнитурами с разной топологией).
    """
    if not bad_channels:
        return signal

    good_channels = [c for c in range(signal.shape[1]) if c not in bad_channels]
    if not good_channels:
        return signal  # нечем интерполировать — все каналы помечены как плохие

    result = signal.copy()
    reference = np.mean(signal[:, good_channels], axis=1)
    for ch in bad_channels:
        result[:, ch] = reference
    return result


def compute_epoch_stats(epochs: np.ndarray) -> np.ndarray:
    amplitude_range = np.ptp(epochs, axis=1).mean(axis=1)
    variance = np.var(epochs, axis=1).mean(axis=1)

    channel_means = epochs.mean(axis=1)                      # [n_epochs, n_channels]
    overall_channel_mean = channel_means.mean(axis=0)         # [n_channels]
    deviation = np.abs(channel_means - overall_channel_mean).mean(axis=1)

    return np.stack([amplitude_range, variance, deviation], axis=1)


def detect_bad_epochs(epochs: np.ndarray, config: FasterConfig) -> List[int]:
    features = compute_epoch_stats(epochs)
    z = _safe_zscore(features)
    bad_mask = np.any(np.abs(z) > config.z_threshold, axis=1)
    return list(np.where(bad_mask)[0])


def compute_component_stats(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_components = sources.shape[1]

    spatial_kurtosis = kurtosis(mixing_matrix, axis=0, fisher=True)
    component_hurst = np.array([
        hurst_exponent(sources[:, i], config.hurst_max_lag) for i in range(n_components)
    ])
    component_slope = np.array([
        spectral_slope(sources[:, i], sample_rate, config.spectral_slope_band_hz)
        for i in range(n_components)
    ])

    if eog_signal is not None:
        eog_corr = np.array([
            abs(np.corrcoef(sources[:, i], eog_signal)[0, 1]) for i in range(n_components)
        ])
        eog_corr = np.nan_to_num(eog_corr, nan=0.0)
    else:
        eog_corr = np.zeros(n_components)  # без EOG-канала признак не участвует в отбраковке

    return np.stack([spatial_kurtosis, component_hurst, component_slope, eog_corr], axis=1)


def detect_bad_components(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> List[int]:
    features = compute_component_stats(sources, mixing_matrix, sample_rate, config, eog_signal)
    z = _safe_zscore(features)
    bad_mask = np.any(np.abs(z) > config.z_threshold, axis=1)
    return list(np.where(bad_mask)[0])


def compute_channel_epoch_stats(epochs: np.ndarray) -> np.ndarray:
    variance = np.var(epochs, axis=1)                                  # [n_epochs, n_channels]
    median_gradient = np.median(np.abs(np.diff(epochs, axis=1)), axis=1)
    amplitude_range = np.ptp(epochs, axis=1)

    channel_mean_amplitude = epochs.mean(axis=1)                       # [n_epochs, n_channels]
    overall_channel_mean = channel_mean_amplitude.mean(axis=0)          # [n_channels]
    deviation = np.abs(channel_mean_amplitude - overall_channel_mean)

    return np.stack([variance, median_gradient, amplitude_range, deviation], axis=2)


def detect_bad_channel_epoch_pairs(
    epochs: np.ndarray, config: FasterConfig
) -> List[Tuple[int, int]]:
    features = compute_channel_epoch_stats(epochs)   # [n_epochs, n_channels, 4]
    n_epochs, n_channels, n_features = features.shape

    bad_pairs: List[Tuple[int, int]] = []
    for feat_idx in range(n_features):
        column = features[:, :, feat_idx]            # [n_epochs, n_channels]
        z = _safe_zscore(column)
        epoch_idx, channel_idx = np.where(np.abs(z) > config.z_threshold)
        bad_pairs.extend(zip(epoch_idx.tolist(), channel_idx.tolist()))

    return sorted(set(bad_pairs))


def interpolate_channel_epoch_pairs(
    epochs: np.ndarray, bad_pairs: List[Tuple[int, int]]
) -> np.ndarray:
    if not bad_pairs:
        return epochs

    result = epochs.copy()
    bad_by_epoch: dict[int, List[int]] = {}
    for epoch_idx, channel_idx in bad_pairs:
        bad_by_epoch.setdefault(epoch_idx, []).append(channel_idx)

    for epoch_idx, bad_channels in bad_by_epoch.items():
        result[epoch_idx] = interpolate_channels(result[epoch_idx], bad_channels)

    return result


@dataclass
class FasterReport:
    bad_channels: List[int] = field(default_factory=list)
    bad_epochs: List[int] = field(default_factory=list)
    bad_components: List[int] = field(default_factory=list)
    bad_channel_epoch_pairs: List[Tuple[int, int]] = field(default_factory=list)


def run_faster(
    epochs: np.ndarray,
    config: Optional[FasterConfig] = None,
) -> Tuple[np.ndarray, FasterReport]:
    config = config or FasterConfig()
    report = FasterReport()

    continuous = epochs.reshape(-1, epochs.shape[2])
    report.bad_channels = detect_bad_channels(continuous, config)
    if report.bad_channels and config.interpolate_bad_channels:
        epochs = np.stack([
            interpolate_channels(epoch, report.bad_channels) for epoch in epochs
        ])

    report.bad_epochs = detect_bad_epochs(epochs, config)
    good_epoch_mask = np.ones(epochs.shape[0], dtype=bool)
    good_epoch_mask[report.bad_epochs] = False
    clean_epochs = epochs[good_epoch_mask]

    if clean_epochs.shape[0] > 0:
        report.bad_channel_epoch_pairs = detect_bad_channel_epoch_pairs(clean_epochs, config)
        if report.bad_channel_epoch_pairs and config.interpolate_bad_channel_epoch:
            clean_epochs = interpolate_channel_epoch_pairs(clean_epochs, report.bad_channel_epoch_pairs)

    return clean_epochs, report


def apply_faster(signal: np.ndarray, config: Optional[FasterConfig] = None) -> np.ndarray:
    """
    Облегчённая версия для потокового режима: применяется к одному
    окну (10.2.6), где нет доступа к другим эпохам записи
    """
    config = config or FasterConfig()
    bad_channels = detect_bad_channels(signal, config)

    if bad_channels and config.interpolate_bad_channels:
        signal = interpolate_channels(signal, bad_channels)

    return signal


@dataclass
class IcaConfig:
    n_components: Optional[int] = None      # по умолчанию = n_channels
    max_iter: int = 500
    random_state: int = 42
    faster_config: FasterConfig = field(default_factory=FasterConfig)


class ArtifactICA:
    """
    Для потокового режима ICA обучается вычисляется одним пакетным проходом по уже накопленному массиву калибровочной
    записи пользователя и затем применяется как фиксированное
    линейное преобразование — полное переобучение ICA на каждое
    окно слишком дорого для реального времени.
    """

    def __init__(self, config: Optional[IcaConfig] = None):
        self.config = config or IcaConfig()
        self._ica: Optional[FastICA] = None
        self._artifact_components: List[int] = []

    def fit(
        self,
        calibration_signal: np.ndarray,
        sample_rate: float,
        eog_signal: Optional[np.ndarray] = None,
    ) -> "ArtifactICA":
        n_components = self.config.n_components or calibration_signal.shape[1]
        self._ica = FastICA(
            n_components=n_components,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        sources = self._ica.fit_transform(calibration_signal)
        mixing_matrix = self._ica.mixing_  # [n_channels, n_components]

        self._artifact_components = detect_bad_components(
            sources, mixing_matrix, sample_rate, self.config.faster_config, eog_signal
        )
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self._ica is None:
            raise RuntimeError("ArtifactICA не обучен — вызовите fit() на калибровочных данных")

        sources = self._ica.transform(signal)
        sources[:, self._artifact_components] = 0.0
        return self._ica.inverse_transform(sources)

    @property
    def n_artifact_components(self) -> int:
        return len(self._artifact_components)

