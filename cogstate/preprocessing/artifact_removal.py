from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, zscore
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings


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
    return np.where(bad_mask)[0].astype(int).tolist()


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
    return np.where(bad_mask)[0].astype(int).tolist()


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
    return np.where(bad_mask)[0].astype(int).tolist()


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

    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(
            "run_faster expects [n_epochs, n_samples, n_channels]"
        )
    if epochs.shape[0] == 0:
        raise ValueError("run_faster received zero epochs")
    if epochs.shape[1] == 0 or epochs.shape[2] == 0:
        raise ValueError("run_faster received an empty dimension")
    if not np.isfinite(epochs).all():
        raise ValueError("run_faster input contains NaN or Inf")
    report = FasterReport()

    continuous = epochs.reshape(-1, epochs.shape[2])
    report.bad_channels = detect_bad_channels(continuous, config)
    if report.bad_channels and config.interpolate_bad_channels:
        epochs = np.stack([
            interpolate_channels(epoch, report.bad_channels) for epoch in epochs
        ])

    report.bad_epochs = detect_bad_epochs(epochs, config)

    original_epoch_indices = np.arange(epochs.shape[0])
    good_epoch_mask = np.ones(epochs.shape[0], dtype=bool)
    good_epoch_mask[report.bad_epochs] = False

    clean_epochs = epochs[good_epoch_mask]
    clean_to_original = original_epoch_indices[good_epoch_mask]

    if clean_epochs.shape[0] > 0:
        clean_bad_pairs = detect_bad_channel_epoch_pairs(
            clean_epochs, config
        )

        report.bad_channel_epoch_pairs = [
            (int(clean_to_original[epoch_idx]), int(channel_idx))
            for epoch_idx, channel_idx in clean_bad_pairs
        ]

        if clean_bad_pairs and config.interpolate_bad_channel_epoch:
            clean_epochs = interpolate_channel_epoch_pairs(
                clean_epochs, clean_bad_pairs
            )

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
    n_components: Optional[int] = None
    max_iter: int = 500
    random_state: int = 42
    faster_config: FasterConfig = field(default_factory=FasterConfig)


class ArtifactICA:
    """
    ICA is fitted once on allowed training/calibration data and then
    applied as a fixed transform during inference.
    """

    def __init__(self, config: Optional[IcaConfig] = None):
        self.config = config or IcaConfig()
        self._ica: Optional[FastICA] = None
        self._artifact_components: List[int] = []

        self._input_rank: Optional[int] = None
        self._input_n_channels: Optional[int] = None
        self._n_components: Optional[int] = None
        self._n_iter: Optional[int] = None
        self._converged: Optional[bool] = None

    def fit(
        self,
        calibration_signal: np.ndarray,
        sample_rate: float,
        eog_signal: Optional[np.ndarray] = None,
    ) -> "ArtifactICA":
        signal = np.asarray(calibration_signal, dtype=float)

        if signal.ndim != 2:
            raise ValueError(
                "ArtifactICA.fit expects [n_samples, n_channels]"
            )
        if signal.shape[0] < 2:
            raise ValueError("ArtifactICA.fit requires at least two samples")
        if signal.shape[1] < 2:
            raise ValueError("ArtifactICA.fit requires at least two channels")
        if not np.isfinite(signal).all():
            raise ValueError("ArtifactICA.fit input contains NaN or Inf")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")

        if eog_signal is not None:
            eog_signal = np.asarray(eog_signal, dtype=float)
            if eog_signal.ndim != 1:
                raise ValueError("eog_signal must be one-dimensional")
            if eog_signal.shape[0] != signal.shape[0]:
                raise ValueError(
                    "eog_signal length must match calibration_signal"
                )
            if not np.isfinite(eog_signal).all():
                raise ValueError("eog_signal contains NaN or Inf")

        n_channels = int(signal.shape[1])
        rank = int(np.linalg.matrix_rank(signal))

        if rank < 2:
            raise ValueError(
                f"Calibration signal rank is too low for ICA: {rank}"
            )

        requested_components = (
            int(self.config.n_components)
            if self.config.n_components is not None
            else n_channels
        )

        if requested_components < 2:
            raise ValueError("ICA n_components must be at least 2")
        if requested_components > n_channels:
            raise ValueError(
                "ICA n_components cannot exceed the number of channels"
            )

        effective_components = min(requested_components, rank)

        if effective_components < requested_components:
            warnings.warn(
                f"ICA n_components reduced from {requested_components} "
                f"to signal rank {effective_components}",
                RuntimeWarning,
                stacklevel=2,
            )

        self._input_rank = rank
        self._input_n_channels = n_channels
        self._n_components = effective_components

        self._ica = FastICA(
            n_components=effective_components,
            max_iter=int(self.config.max_iter),
            random_state=int(self.config.random_state),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            sources = self._ica.fit_transform(signal)

        convergence_warnings = [
            item for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]

        self._n_iter = int(self._ica.n_iter_)
        self._converged = len(convergence_warnings) == 0

        if not self._converged:
            warnings.warn(
                "FastICA did not converge; result should not be treated "
                "as validated preprocessing.",
                ConvergenceWarning,
                stacklevel=2,
            )

        mixing_matrix = self._ica.mixing_
        self._artifact_components = detect_bad_components(
            sources,
            mixing_matrix,
            sample_rate,
            self.config.faster_config,
            eog_signal,
        )

        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self._ica is None:
            raise RuntimeError(
                "ArtifactICA is not fitted; call fit() on calibration data"
            )

        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 2:
            raise ValueError(
                "ArtifactICA.transform expects [n_samples, n_channels]"
            )
        if not np.isfinite(signal).all():
            raise ValueError("ArtifactICA.transform input contains NaN or Inf")

        if (
            self._input_n_channels is not None
            and signal.shape[1] != self._input_n_channels
        ):
            raise ValueError(
                f"ArtifactICA was fitted on {self._input_n_channels} channels, "
                f"got {signal.shape[1]}"
            )

        sources = self._ica.transform(signal)

        if self._artifact_components:
            sources[:, self._artifact_components] = 0.0

        cleaned = self._ica.inverse_transform(sources)

        if not np.isfinite(cleaned).all():
            raise RuntimeError(
                "ArtifactICA produced NaN or Inf during transform"
            )

        return cleaned

    @property
    def n_artifact_components(self) -> int:
        return len(self._artifact_components)

    @property
    def artifact_components(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in self._artifact_components)

    @property
    def input_rank(self) -> Optional[int]:
        return self._input_rank

    @property
    def n_components(self) -> Optional[int]:
        return self._n_components

    @property
    def n_iter(self) -> Optional[int]:
        return self._n_iter

    @property
    def converged(self) -> Optional[bool]:
        return self._converged
