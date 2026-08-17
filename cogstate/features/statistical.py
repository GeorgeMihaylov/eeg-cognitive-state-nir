from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import kurtosis, skew


@dataclass
class StatisticalConfig:
    include_hjorth: bool = True
    include_higher_moments: bool = True


def compute_hjorth_parameters(window: np.ndarray) -> Dict[str, np.ndarray]:
    activity = np.var(window, axis=0)

    d1 = np.diff(window, axis=0)
    d2 = np.diff(d1, axis=0)

    var_d1 = np.var(d1, axis=0)
    var_d2 = np.var(d2, axis=0)

    eps = 1e-12
    mobility = np.sqrt(var_d1 / (activity + eps))
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + eps))
    complexity = mobility_d1 / (mobility + eps)

    return {"hjorth_activity": activity, "hjorth_mobility": mobility, "hjorth_complexity": complexity}


def compute_zero_crossing_rate(window: np.ndarray) -> np.ndarray:
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=0)) > 0
    return crossings.mean(axis=0)


def compute_basic_moments(window: np.ndarray) -> Dict[str, np.ndarray]:
    standard_deviation = np.std(window, axis=0)
    variable = standard_deviation > np.finfo(float).eps
    skewness = np.zeros(window.shape[1])
    excess_kurtosis = np.zeros(window.shape[1])
    if np.any(variable):
        skewness[variable] = skew(window[:, variable], axis=0)
        excess_kurtosis[variable] = kurtosis(
            window[:, variable], axis=0, fisher=True
        )
    return {
        "mean": np.mean(window, axis=0),
        "std": standard_deviation,
        "skewness": np.nan_to_num(skewness, nan=0.0),
        "kurtosis": np.nan_to_num(excess_kurtosis, nan=0.0),
        "peak_to_peak": np.ptp(window, axis=0),
        "rms": np.sqrt(np.mean(window ** 2, axis=0)),
    }


def extract_statistical_features(window: np.ndarray, config: StatisticalConfig) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}

    if config.include_higher_moments:
        features.update(compute_basic_moments(window))
    else:
        features["mean"] = np.mean(window, axis=0)
        features["std"] = np.std(window, axis=0)

    features["zero_crossing_rate"] = compute_zero_crossing_rate(window)

    if config.include_hjorth:
        features.update(compute_hjorth_parameters(window))

    return features


def feature_names(config: StatisticalConfig) -> List[str]:
    names = ["mean", "std"]
    if config.include_higher_moments:
        names += ["skewness", "kurtosis", "peak_to_peak", "rms"]
    names.append("zero_crossing_rate")
    if config.include_hjorth:
        names += ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    return names
