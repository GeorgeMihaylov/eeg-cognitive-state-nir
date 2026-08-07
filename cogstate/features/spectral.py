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


@dataclass
class SpectralConfig:
    sample_rate: float
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    nperseg: int = 256


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.zeros(psd.shape[1])
    return np.trapz(psd[mask], freqs[mask], axis=0)


def compute_band_powers(window: np.ndarray, config: SpectralConfig) -> Dict[str, np.ndarray]:
    nperseg = min(config.nperseg, window.shape[0])
    freqs, psd = welch(window, fs=config.sample_rate, nperseg=nperseg, axis=0)

    return {name: _band_power(freqs, psd, band) for name, band in config.bands.items()}


def compute_relative_band_powers(band_powers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    total = sum(band_powers.values())
    total = np.where(total == 0, 1e-12, total)
    return {name: power / total for name, power in band_powers.items()}


def compute_band_ratios(band_powers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = 1e-12
    theta, beta = band_powers["theta"], band_powers["beta"] + eps
    alpha = band_powers["alpha"]

    return {
        "theta_beta_ratio": theta / beta,
        "alpha_theta_ratio": alpha / (theta + eps),
        "engagement_index": (theta + alpha) / (band_powers["alpha"] + band_powers["beta"] + eps),
    }


def compute_spectral_edge_frequency(window: np.ndarray, config: SpectralConfig, edge: float = 0.95) -> np.ndarray:
    nperseg = min(config.nperseg, window.shape[0])
    freqs, psd = welch(window, fs=config.sample_rate, nperseg=nperseg, axis=0)

    cumulative = np.cumsum(psd, axis=0)
    total = cumulative[-1]
    total = np.where(total == 0, 1e-12, total)
    threshold = edge * total

    sef = np.zeros(psd.shape[1])
    for ch in range(psd.shape[1]):
        idx = np.searchsorted(cumulative[:, ch], threshold[ch])
        idx = min(idx, len(freqs) - 1)
        sef[ch] = freqs[idx]
    return sef


def extract_spectral_features(window: np.ndarray, config: SpectralConfig) -> Dict[str, np.ndarray]:
    band_powers = compute_band_powers(window, config)
    relative_powers = compute_relative_band_powers(band_powers)
    ratios = compute_band_ratios(band_powers)
    sef = compute_spectral_edge_frequency(window, config)

    features: Dict[str, np.ndarray] = {}
    features.update({f"power_{name}": values for name, values in band_powers.items()})
    features.update({f"relpower_{name}": values for name, values in relative_powers.items()})
    features.update(ratios)
    features["spectral_edge_freq"] = sef
    return features


def feature_names(config: SpectralConfig) -> List[str]:
    names = [f"power_{b}" for b in config.bands] + [f"relpower_{b}" for b in config.bands]
    names += ["theta_beta_ratio", "alpha_theta_ratio", "engagement_index", "spectral_edge_freq"]
    return names
