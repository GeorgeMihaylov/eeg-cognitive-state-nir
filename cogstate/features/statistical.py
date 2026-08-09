"""Numerically stable statistical EEG features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import validate_window


@dataclass(frozen=True)
class StatisticalConfig:
    include_hjorth: bool = True
    include_higher_moments: bool = True
    variance_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.variance_epsilon) or self.variance_epsilon <= 0:
            raise ValueError("variance_epsilon must be finite and positive")


def compute_hjorth_parameters(
    window: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    signal = validate_window(window)
    activity = np.var(signal, axis=0)
    first = np.diff(signal, axis=0)
    second = np.diff(first, axis=0)
    variance_first = np.var(first, axis=0)
    variance_second = np.var(second, axis=0)
    mobility = np.zeros_like(activity)
    valid_activity = activity > epsilon
    mobility[valid_activity] = np.sqrt(
        variance_first[valid_activity] / activity[valid_activity]
    )
    derivative_mobility = np.zeros_like(activity)
    valid_first = variance_first > epsilon
    derivative_mobility[valid_first] = np.sqrt(
        variance_second[valid_first] / variance_first[valid_first]
    )
    complexity = np.zeros_like(activity)
    valid_mobility = mobility > epsilon
    complexity[valid_mobility] = (
        derivative_mobility[valid_mobility] / mobility[valid_mobility]
    )
    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def compute_zero_crossing_rate(window: np.ndarray) -> np.ndarray:
    signal = validate_window(window)
    signs = np.signbit(signal)
    return np.mean(signs[1:] != signs[:-1], axis=0)


def compute_basic_moments(
    window: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    signal = validate_window(window)
    mean = np.mean(signal, axis=0)
    centered = signal - mean
    variance = np.mean(np.square(centered), axis=0)
    standard_deviation = np.sqrt(variance)
    skewness = np.zeros_like(mean)
    excess_kurtosis = np.zeros_like(mean)
    stable = variance > epsilon
    skewness[stable] = (
        np.mean(centered[:, stable] ** 3, axis=0)
        / standard_deviation[stable] ** 3
    )
    excess_kurtosis[stable] = (
        np.mean(centered[:, stable] ** 4, axis=0) / variance[stable] ** 2 - 3.0
    )
    return {
        "mean": mean,
        "std": standard_deviation,
        "skewness": skewness,
        "kurtosis": excess_kurtosis,
        "peak_to_peak": np.ptp(signal, axis=0),
        "rms": np.sqrt(np.mean(np.square(signal), axis=0)),
    }


def extract_statistical_features(
    window: np.ndarray,
    config: StatisticalConfig,
) -> dict[str, np.ndarray]:
    signal = validate_window(window)
    moments = compute_basic_moments(signal, epsilon=config.variance_epsilon)
    if config.include_higher_moments:
        features = dict(moments)
    else:
        features = {name: moments[name] for name in ("mean", "std")}
    features["zero_crossing_rate"] = compute_zero_crossing_rate(signal)
    if config.include_hjorth:
        features.update(
            compute_hjorth_parameters(signal, epsilon=config.variance_epsilon)
        )
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("statistical feature extraction produced NaN or Inf")
    return features


def feature_names(config: StatisticalConfig) -> list[str]:
    names = ["mean", "std"]
    if config.include_higher_moments:
        names.extend(["skewness", "kurtosis", "peak_to_peak", "rms"])
    names.append("zero_crossing_rate")
    if config.include_hjorth:
        names.extend(
            ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
        )
    return names
