"""Spectral EEG features computed from one shared Welch estimate per window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import welch


DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class PowerSpectrum:
    frequencies: np.ndarray
    psd: np.ndarray


@dataclass
class SpectralConfig:
    sample_rate: float
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )
    nperseg: int = 256
    spectral_edge: float = 0.95
    spectral_edge_band_hz: Tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.nperseg < 2:
            raise ValueError("sample_rate must be positive and nperseg >= 2")
        nyquist = self.sample_rate / 2.0
        if not self.bands:
            raise ValueError("At least one spectral band is required")
        for name, (low, high) in self.bands.items():
            if not 0 <= low < high <= nyquist:
                raise ValueError(
                    f"Band {name!r} must lie inside [0, Nyquist={nyquist:g}]"
                )
        if not 0 < self.spectral_edge < 1:
            raise ValueError("spectral_edge must lie in (0, 1)")
        if self.spectral_edge_band_hz is not None:
            low, high = self.spectral_edge_band_hz
            if not 0 <= low < high <= nyquist:
                raise ValueError("spectral_edge_band_hz must lie inside Nyquist")


def _window_matrix(window: object) -> np.ndarray:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Window must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("Window contains non-finite values")
    return values


def compute_power_spectrum(
    window: np.ndarray, config: SpectralConfig
) -> PowerSpectrum:
    values = _window_matrix(window)
    frequencies, psd = welch(
        values,
        fs=config.sample_rate,
        nperseg=min(config.nperseg, len(values)),
        axis=0,
    )
    return PowerSpectrum(frequencies=frequencies, psd=psd)


def _band_power(
    frequencies: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
) -> np.ndarray:
    # Adjacent canonical bands share a boundary. Half-open masks prevent the
    # boundary Welch bin from being counted twice; the last bin is immaterial
    # for integration but may be included safely at Nyquist.
    mask = (frequencies >= band[0]) & (frequencies < band[1])
    if not np.any(mask):
        return np.zeros(psd.shape[1])
    return np.trapezoid(psd[mask], frequencies[mask], axis=0)


def compute_band_powers(
    window: np.ndarray,
    config: SpectralConfig,
    *,
    spectrum: PowerSpectrum | None = None,
) -> Dict[str, np.ndarray]:
    estimate = spectrum or compute_power_spectrum(window, config)
    return {
        name: _band_power(estimate.frequencies, estimate.psd, band)
        for name, band in config.bands.items()
    }


def compute_relative_band_powers(
    band_powers: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    total = np.sum(np.stack(tuple(band_powers.values())), axis=0)
    total = np.maximum(total, np.finfo(float).tiny)
    return {name: power / total for name, power in band_powers.items()}


def compute_band_ratios(
    band_powers: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    required = {"theta", "alpha", "beta"}
    missing = required.difference(band_powers)
    if missing:
        raise ValueError(f"Band ratios require bands: {sorted(missing)}")
    theta = band_powers["theta"]
    alpha = band_powers["alpha"]
    beta = band_powers["beta"]
    total = np.sum(np.stack(tuple(band_powers.values())), axis=0)
    denominator_floor = np.maximum(total * 1e-12, np.finfo(float).eps)
    return {
        "theta_beta_ratio": theta / np.maximum(beta, denominator_floor),
        "alpha_theta_ratio": alpha / np.maximum(theta, denominator_floor),
        # Common EEG engagement index: beta / (alpha + theta).
        "engagement_index": beta / np.maximum(
            alpha + theta, denominator_floor
        ),
    }


def compute_spectral_edge_frequency(
    window: np.ndarray,
    config: SpectralConfig,
    edge: float | None = None,
    *,
    spectrum: PowerSpectrum | None = None,
) -> np.ndarray:
    estimate = spectrum or compute_power_spectrum(window, config)
    edge_value = config.spectral_edge if edge is None else edge
    if not 0 < edge_value < 1:
        raise ValueError("edge must lie in (0, 1)")
    if config.spectral_edge_band_hz is None:
        band = (
            min(low for low, _ in config.bands.values()),
            max(high for _, high in config.bands.values()),
        )
    else:
        band = config.spectral_edge_band_hz
    mask = (estimate.frequencies >= band[0]) & (estimate.frequencies <= band[1])
    frequencies = estimate.frequencies[mask]
    psd = estimate.psd[mask]
    if not len(frequencies):
        return np.zeros(estimate.psd.shape[1])
    cumulative = np.cumsum(psd, axis=0)
    thresholds = edge_value * cumulative[-1]
    output = np.empty(psd.shape[1])
    for channel in range(psd.shape[1]):
        index = int(np.searchsorted(cumulative[:, channel], thresholds[channel]))
        output[channel] = frequencies[min(index, len(frequencies) - 1)]
    return output


def extract_spectral_features(
    window: np.ndarray,
    config: SpectralConfig,
    *,
    spectrum: PowerSpectrum | None = None,
) -> Dict[str, np.ndarray]:
    estimate = spectrum or compute_power_spectrum(window, config)
    band_powers = compute_band_powers(window, config, spectrum=estimate)
    relative_powers = compute_relative_band_powers(band_powers)
    features: Dict[str, np.ndarray] = {}
    features.update({f"power_{name}": value for name, value in band_powers.items()})
    features.update(
        {f"relpower_{name}": value for name, value in relative_powers.items()}
    )
    features.update(compute_band_ratios(band_powers))
    features["spectral_edge_freq"] = compute_spectral_edge_frequency(
        window, config, spectrum=estimate
    )
    return features


def feature_names(config: SpectralConfig) -> List[str]:
    names = [f"power_{band}" for band in config.bands]
    names += [f"relpower_{band}" for band in config.bands]
    names += [
        "theta_beta_ratio",
        "alpha_theta_ratio",
        "engagement_index",
        "spectral_edge_freq",
    ]
    return names
