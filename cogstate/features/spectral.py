"""Deterministic spectral EEG features for ``[samples, channels]`` windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from ._validation import ordered_bands, validate_sample_rate, validate_window


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class PowerSpectrum:
    """One shared Welch estimate reusable by all spectral feature helpers."""

    frequencies: np.ndarray
    psd: np.ndarray


@dataclass(frozen=True)
class SpectralConfig:
    sample_rate: float
    bands: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )
    nperseg: int = 256
    noverlap: int | None = None
    window: str = "hann"
    detrend: str = "constant"
    scaling: str = "density"
    average: str = "mean"
    spectral_edge: float = 0.95
    spectral_edge_band_hz: tuple[float, float] | None = None
    include_engagement_index: bool = False

    def __post_init__(self) -> None:
        validate_sample_rate(self.sample_rate)
        ordered_bands(self.bands, sample_rate=self.sample_rate)
        if int(self.nperseg) < 2:
            raise ValueError("nperseg must be at least 2")
        if self.noverlap is not None and int(self.noverlap) < 0:
            raise ValueError("noverlap must be non-negative or None")
        if not 0.0 < float(self.spectral_edge) <= 1.0:
            raise ValueError("spectral_edge must be in (0, 1]")
        if self.spectral_edge_band_hz is not None:
            low, high = self.spectral_edge_band_hz
            if not (0.0 <= float(low) < float(high) <= self.sample_rate / 2.0):
                raise ValueError("spectral_edge_band_hz must lie inside Nyquist")

    @property
    def ordered_bands(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        return ordered_bands(self.bands, sample_rate=self.sample_rate)


def compute_power_spectrum(
    window: np.ndarray, config: SpectralConfig
) -> PowerSpectrum:
    signal = validate_window(window)
    nperseg = min(int(config.nperseg), signal.shape[0])
    noverlap = config.noverlap
    if noverlap is not None and int(noverlap) >= nperseg:
        raise ValueError("noverlap must be smaller than the effective nperseg")
    frequencies, psd = welch(
        signal,
        fs=float(config.sample_rate),
        window=config.window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=config.detrend,
        scaling=config.scaling,
        average=config.average,
        axis=0,
    )
    return PowerSpectrum(frequencies=frequencies, psd=psd)


def _welch(window: np.ndarray, config: SpectralConfig) -> tuple[np.ndarray, np.ndarray]:
    estimate = compute_power_spectrum(window, config)
    return estimate.frequencies, estimate.psd


def _band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    if np.count_nonzero(mask) < 2:
        return np.zeros(psd.shape[1], dtype=np.float64)
    return np.asarray(trapezoid(psd[mask], frequencies[mask], axis=0), dtype=float)


def _band_powers_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    config: SpectralConfig,
) -> dict[str, np.ndarray]:
    return {
        name: _band_power(frequencies, psd, band)
        for name, band in config.ordered_bands
    }


def compute_band_powers(
    window: np.ndarray,
    config: SpectralConfig,
) -> dict[str, np.ndarray]:
    frequencies, psd = _welch(window, config)
    return _band_powers_from_psd(frequencies, psd, config)


def compute_relative_band_powers(
    band_powers: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not band_powers:
        raise ValueError("band_powers must be non-empty")
    arrays = [np.asarray(value, dtype=float) for value in band_powers.values()]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("all band power arrays must have the same shape")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("band power arrays contain NaN or Inf")
    total = np.sum(np.stack(arrays, axis=0), axis=0)
    denominator = np.where(total > np.finfo(float).eps, total, 1.0)
    return {
        str(name): np.asarray(value, dtype=float) / denominator
        for name, value in band_powers.items()
    }


def compute_band_ratios(
    band_powers: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    required = {"theta", "alpha", "beta"}
    missing = sorted(required - set(band_powers))
    if missing:
        raise ValueError(f"band powers are missing required bands: {missing}")
    eps = np.finfo(float).eps
    theta = np.asarray(band_powers["theta"], dtype=float)
    alpha = np.asarray(band_powers["alpha"], dtype=float)
    beta = np.asarray(band_powers["beta"], dtype=float)
    if not np.isfinite(np.stack([theta, alpha, beta])).all():
        raise ValueError("band power arrays contain NaN or Inf")
    ratios = {
        "theta_beta_ratio": theta / np.maximum(beta, eps),
        "alpha_theta_ratio": alpha / np.maximum(theta, eps),
    }
    if required.issubset(band_powers):
        total = np.sum(
            np.stack([np.asarray(value, dtype=float) for value in band_powers.values()]),
            axis=0,
        )
        floor = np.maximum(total * 1e-12, eps)
        ratios["engagement_index"] = beta / np.maximum(alpha + theta, floor)
    return ratios


def _spectral_edge_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    edge: float,
) -> np.ndarray:
    cumulative = np.cumsum(psd, axis=0)
    total = cumulative[-1]
    thresholds = float(edge) * total
    result = np.zeros(psd.shape[1], dtype=float)
    for channel in range(psd.shape[1]):
        if total[channel] <= np.finfo(float).eps:
            result[channel] = 0.0
            continue
        index = int(np.searchsorted(cumulative[:, channel], thresholds[channel]))
        result[channel] = frequencies[min(index, len(frequencies) - 1)]
    return result


def _spectral_edge_inputs(
    frequencies: np.ndarray,
    psd: np.ndarray,
    config: SpectralConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.spectral_edge_band_hz is None:
        return frequencies, psd
    low, high = config.spectral_edge_band_hz
    mask = (frequencies >= low) & (frequencies <= high)
    return frequencies[mask], psd[mask]


def compute_spectral_edge_frequency(
    window: np.ndarray,
    config: SpectralConfig,
) -> np.ndarray:
    frequencies, psd = _welch(window, config)
    frequencies, psd = _spectral_edge_inputs(frequencies, psd, config)
    if frequencies.size == 0:
        return np.zeros(psd.shape[1], dtype=float)
    return _spectral_edge_from_psd(frequencies, psd, config.spectral_edge)


def extract_spectral_features(
    window: np.ndarray,
    config: SpectralConfig,
) -> dict[str, np.ndarray]:
    """Extract one deterministic spectral feature block per channel."""
    frequencies, psd = _welch(window, config)
    band_powers = _band_powers_from_psd(frequencies, psd, config)
    relative = compute_relative_band_powers(band_powers)
    ratios = compute_band_ratios(band_powers)
    features: dict[str, np.ndarray] = {}
    for name, _ in config.ordered_bands:
        features[f"power_{name}"] = band_powers[name]
    for name, _ in config.ordered_bands:
        features[f"relpower_{name}"] = relative[name]
    features.update(
        {
            name: value
            for name, value in ratios.items()
            if name != "engagement_index" or config.include_engagement_index
        }
    )
    edge_frequencies, edge_psd = _spectral_edge_inputs(frequencies, psd, config)
    features["spectral_edge_frequency"] = (
        np.zeros(psd.shape[1], dtype=float)
        if edge_frequencies.size == 0
        else _spectral_edge_from_psd(
            edge_frequencies, edge_psd, config.spectral_edge
        )
    )
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("spectral feature extraction produced NaN or Inf")
    return features


def feature_names(config: SpectralConfig) -> list[str]:
    bands = [name for name, _ in config.ordered_bands]
    names = [
        *[f"power_{name}" for name in bands],
        *[f"relpower_{name}" for name in bands],
        "theta_beta_ratio",
        "alpha_theta_ratio",
    ]
    if config.include_engagement_index:
        names.append("engagement_index")
    names.append("spectral_edge_frequency")
    return names
