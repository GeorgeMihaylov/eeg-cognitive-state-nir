"""Shared validation helpers for target-free EEG feature extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


MIN_WINDOW_SAMPLES = 8


def validate_sample_rate(sample_rate: float) -> float:
    """Return a finite positive sampling rate."""
    value = float(sample_rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sample_rate must be finite and positive")
    return value


def validate_window(window: np.ndarray, *, min_samples: int = MIN_WINDOW_SAMPLES) -> np.ndarray:
    """Validate the canonical ``[samples, channels]`` EEG layout."""
    if not isinstance(window, np.ndarray):
        raise TypeError("EEG window must be a numpy.ndarray")
    if window.ndim != 2:
        raise ValueError(
            "EEG window must have shape [samples, channels], "
            f"got {window.shape}"
        )
    if window.shape[0] < int(min_samples):
        raise ValueError(
            f"EEG window needs at least {int(min_samples)} samples, "
            f"got {window.shape[0]}"
        )
    if window.shape[1] < 1:
        raise ValueError("EEG window must contain at least one channel")
    if not np.issubdtype(window.dtype, np.number):
        raise TypeError("EEG window must contain numeric values")
    if not np.isfinite(window).all():
        raise ValueError("EEG window contains NaN or Inf")
    return np.asarray(window, dtype=np.float64)


def ordered_bands(
    bands: Mapping[str, tuple[float, float]],
    *,
    sample_rate: float,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """Validate and deterministically order named frequency bands."""
    rate = validate_sample_rate(sample_rate)
    if not isinstance(bands, Mapping) or not bands:
        raise ValueError("bands must be a non-empty mapping")
    normalized: list[tuple[str, tuple[float, float]]] = []
    for raw_name, raw_limits in bands.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("band names must be non-empty")
        if len(raw_limits) != 2:
            raise ValueError(f"Band {name!r} must contain low/high frequencies")
        low, high = (float(raw_limits[0]), float(raw_limits[1]))
        if not np.isfinite([low, high]).all() or low < 0 or high <= low:
            raise ValueError(f"Band {name!r} has invalid limits {(low, high)}")
        if high > rate / 2.0:
            raise ValueError(
                f"Band {name!r} upper frequency {high} exceeds Nyquist "
                f"frequency {rate / 2.0}"
            )
        normalized.append((name, (low, high)))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("band names must be unique")
    return tuple(sorted(normalized, key=lambda item: (item[1][0], item[1][1], item[0])))


def json_safe(value: Any) -> Any:
    """Convert dataclass configuration values to deterministic JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
