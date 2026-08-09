"""Entropy features: spectral, permutation and optional sample entropy."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.signal import welch
from scipy.spatial import cKDTree

from ._validation import validate_sample_rate, validate_window


@dataclass(frozen=True)
class EntropyConfig:
    sample_rate: float
    include_spectral_entropy: bool = True
    include_permutation_entropy: bool = True
    include_sample_entropy: bool = False
    spectral_nperseg: int = 256
    spectral_noverlap: int | None = None
    spectral_window: str = "hann"
    permutation_order: int = 3
    permutation_delay: int = 1
    sample_entropy_m: int = 2
    sample_entropy_r_ratio: float = 0.2

    def __post_init__(self) -> None:
        validate_sample_rate(self.sample_rate)
        if not any(
            (
                self.include_spectral_entropy,
                self.include_permutation_entropy,
                self.include_sample_entropy,
            )
        ):
            raise ValueError("at least one entropy measure must be enabled")
        if int(self.spectral_nperseg) < 2:
            raise ValueError("spectral_nperseg must be at least 2")
        if self.spectral_noverlap is not None and int(self.spectral_noverlap) < 0:
            raise ValueError("spectral_noverlap must be non-negative or None")
        if int(self.permutation_order) < 2:
            raise ValueError("permutation_order must be at least 2")
        if int(self.permutation_delay) < 1:
            raise ValueError("permutation_delay must be positive")
        if int(self.sample_entropy_m) < 1:
            raise ValueError("sample_entropy_m must be positive")
        if (
            not np.isfinite(self.sample_entropy_r_ratio)
            or self.sample_entropy_r_ratio <= 0
        ):
            raise ValueError("sample_entropy_r_ratio must be finite and positive")


def spectral_entropy_1d(
    values: np.ndarray,
    sample_rate: float,
    *,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> float:
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("spectral entropy expects a finite one-dimensional signal")
    rate = validate_sample_rate(sample_rate)
    effective_nperseg = min(int(nperseg), len(signal))
    if noverlap is not None and int(noverlap) >= effective_nperseg:
        raise ValueError("spectral_noverlap must be smaller than effective nperseg")
    _, psd = welch(
        signal,
        fs=rate,
        window=window,
        nperseg=effective_nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    total = float(np.sum(psd))
    if total <= np.finfo(float).eps:
        return 0.0
    probabilities = psd / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.size <= 1:
        return 0.0
    value = -np.sum(probabilities * np.log2(probabilities)) / np.log2(
        probabilities.size
    )
    return float(value)


def permutation_entropy_1d(
    values: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> float:
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("permutation entropy expects a finite one-dimensional signal")
    order = int(order)
    delay = int(delay)
    if order < 2 or delay < 1:
        raise ValueError("permutation order must be >=2 and delay must be positive")
    count = len(signal) - (order - 1) * delay
    if count <= 0:
        return 0.0
    offsets = np.arange(order) * delay
    embedded = signal[np.arange(count)[:, None] + offsets[None, :]]
    patterns = np.argsort(embedded, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    denominator = np.log2(factorial(order))
    return 0.0 if denominator == 0 else float(entropy / denominator)


def sample_entropy_1d(
    values: np.ndarray,
    m: int = 2,
    r_ratio: float = 0.2,
) -> float:
    """Compute exact sample entropy using Chebyshev-neighbour KD trees.

    The mathematical definition matches the previous pair-count implementation,
    while avoiding allocation of a full quadratic distance matrix. The feature is
    still explicitly optional because its cost is materially higher than the
    other entropy measures on 2560-sample windows.
    """
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("sample entropy expects a finite one-dimensional signal")
    m = int(m)
    if m < 1 or not np.isfinite(r_ratio) or float(r_ratio) <= 0:
        raise ValueError("sample entropy requires m>=1 and positive r_ratio")
    if len(signal) < m + 2:
        return 0.0
    tolerance = float(r_ratio) * float(np.std(signal))
    if tolerance <= np.finfo(float).eps:
        return 0.0

    def count_pairs(length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(signal, length)
        tree = cKDTree(np.ascontiguousarray(templates))
        return int(len(tree.query_pairs(tolerance, p=np.inf, output_type="ndarray")))

    matches_m = count_pairs(m)
    matches_m1 = count_pairs(m + 1)
    if matches_m == 0 or matches_m1 == 0:
        return 0.0
    return float(-np.log(matches_m1 / matches_m))


def extract_entropy_features(
    window: np.ndarray,
    config: EntropyConfig,
) -> dict[str, np.ndarray]:
    signal = validate_window(window)
    features: dict[str, np.ndarray] = {}
    if config.include_spectral_entropy:
        features["spectral_entropy"] = np.asarray(
            [
                spectral_entropy_1d(
                    signal[:, channel],
                    config.sample_rate,
                    nperseg=config.spectral_nperseg,
                    noverlap=config.spectral_noverlap,
                    window=config.spectral_window,
                )
                for channel in range(signal.shape[1])
            ],
            dtype=float,
        )
    if config.include_permutation_entropy:
        features["permutation_entropy"] = np.asarray(
            [
                permutation_entropy_1d(
                    signal[:, channel],
                    config.permutation_order,
                    config.permutation_delay,
                )
                for channel in range(signal.shape[1])
            ],
            dtype=float,
        )
    if config.include_sample_entropy:
        features["sample_entropy"] = np.asarray(
            [
                sample_entropy_1d(
                    signal[:, channel],
                    config.sample_entropy_m,
                    config.sample_entropy_r_ratio,
                )
                for channel in range(signal.shape[1])
            ],
            dtype=float,
        )
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("entropy feature extraction produced NaN or Inf")
    return features


def feature_names(config: EntropyConfig) -> list[str]:
    names: list[str] = []
    if config.include_spectral_entropy:
        names.append("spectral_entropy")
    if config.include_permutation_entropy:
        names.append("permutation_entropy")
    if config.include_sample_entropy:
        names.append("sample_entropy")
    return names
