"""Connectivity features with explicit pair masks and band-limited phase."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, coherence, hilbert, sosfiltfilt

from .spectral import DEFAULT_BANDS


@dataclass
class ConnectivityConfig:
    sample_rate: float
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )
    nperseg: int = 128
    max_channel_pairs: int | None = 50
    phase_filter_order: int = 4

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.nperseg < 2:
            raise ValueError("sample_rate must be positive and nperseg >= 2")
        if self.max_channel_pairs is not None and self.max_channel_pairs < 1:
            raise ValueError("max_channel_pairs must be positive or None")
        if self.phase_filter_order < 1:
            raise ValueError("phase_filter_order must be positive")
        nyquist = self.sample_rate / 2.0
        for name, (low, high) in self.bands.items():
            if not 0 < low < high < nyquist:
                raise ValueError(
                    f"Connectivity band {name!r} must lie strictly below Nyquist"
                )


def _window_matrix(window: object) -> np.ndarray:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Window must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("Window contains non-finite values")
    return values


def _selected_pairs(n_channels: int, maximum: int | None) -> list[tuple[int, int]]:
    pairs = list(combinations(range(n_channels), 2))
    if maximum is None or maximum >= len(pairs):
        return pairs
    # Even coverage avoids always favoring low-index channels when a latency
    # budget prevents computing every pair.
    indices = np.linspace(0, len(pairs) - 1, num=maximum, dtype=int)
    return [pairs[index] for index in np.unique(indices)]


def _empty_connectivity_matrix(n_channels: int) -> np.ndarray:
    matrix = np.full((n_channels, n_channels), np.nan)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_correlation_matrix(window: np.ndarray) -> np.ndarray:
    values = _window_matrix(window)
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    matrix = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((values.shape[1], values.shape[1])),
        where=denominator > np.finfo(float).eps,
    )
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def compute_coherence_matrix(
    window: np.ndarray,
    config: ConnectivityConfig,
    band: Tuple[float, float],
) -> np.ndarray:
    values = _window_matrix(window)
    matrix = _empty_connectivity_matrix(values.shape[1])
    nperseg = min(config.nperseg, len(values))
    for first, second in _selected_pairs(
        values.shape[1], config.max_channel_pairs
    ):
        if np.std(values[:, first]) <= np.finfo(float).eps or np.std(
            values[:, second]
        ) <= np.finfo(float).eps:
            value = 0.0
        else:
            frequencies, magnitude = coherence(
                values[:, first],
                values[:, second],
                fs=config.sample_rate,
                nperseg=nperseg,
            )
            mask = (frequencies >= band[0]) & (frequencies < band[1])
            value = float(np.nanmean(magnitude[mask])) if np.any(mask) else 0.0
            if not np.isfinite(value):
                value = 0.0
        matrix[first, second] = matrix[second, first] = value
    return matrix


def compute_plv_matrix(
    window: np.ndarray,
    config: ConnectivityConfig,
    band: Tuple[float, float] | None = None,
) -> np.ndarray:
    values = _window_matrix(window)
    if band is not None:
        sos = butter(
            config.phase_filter_order,
            band,
            btype="bandpass",
            fs=config.sample_rate,
            output="sos",
        )
        values = sosfiltfilt(sos, values, axis=0)
    phases = np.angle(hilbert(values, axis=0))
    matrix = _empty_connectivity_matrix(values.shape[1])
    for first, second in _selected_pairs(
        values.shape[1], config.max_channel_pairs
    ):
        phase_difference = phases[:, first] - phases[:, second]
        value = float(np.abs(np.mean(np.exp(1j * phase_difference))))
        matrix[first, second] = matrix[second, first] = value
    return matrix


def summarize_connectivity_matrix(matrix: np.ndarray) -> Dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Connectivity matrix must be square")
    upper = values[np.triu_indices(len(values), k=1)]
    measured = upper[np.isfinite(upper)]
    if not len(measured):
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(measured)),
        "std": float(np.std(measured)),
        "max": float(np.max(measured)),
    }


def extract_connectivity_features(
    window: np.ndarray, config: ConnectivityConfig
) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    correlation = summarize_connectivity_matrix(compute_correlation_matrix(window))
    for statistic, value in correlation.items():
        features[f"correlation_{statistic}"] = np.array([value])

    for band_name, band in config.bands.items():
        coherence_summary = summarize_connectivity_matrix(
            compute_coherence_matrix(window, config, band)
        )
        plv_summary = summarize_connectivity_matrix(
            compute_plv_matrix(window, config, band)
        )
        for statistic, value in coherence_summary.items():
            features[f"coherence_{band_name}_{statistic}"] = np.array([value])
        for statistic, value in plv_summary.items():
            features[f"plv_{band_name}_{statistic}"] = np.array([value])
    return features


def feature_names(config: ConnectivityConfig) -> List[str]:
    names = [f"correlation_{statistic}" for statistic in ("mean", "std", "max")]
    for band_name in config.bands:
        names += [
            f"coherence_{band_name}_{statistic}"
            for statistic in ("mean", "std", "max")
        ]
        names += [
            f"plv_{band_name}_{statistic}"
            for statistic in ("mean", "std", "max")
        ]
    return names
