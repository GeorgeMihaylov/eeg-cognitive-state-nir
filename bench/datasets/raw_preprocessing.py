"""Validated, reproducible preprocessing for timestamp-aligned raw EEG."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


RAW_PREPROCESSING_VERSION = "raw-preprocessing-v1"

DEFAULT_RAW_PREPROCESSING: dict[str, Any] = {
    "resample_hz": 256.0,
    "bandpass": {
        "enabled": False,
        "low_hz": 1.0,
        "high_hz": 45.0,
    },
    "notch": {
        "enabled": False,
        "frequency_hz": 50.0,
        "quality_factor": 30.0,
    },
    "rereference": {"mode": "none"},
    "artifact_rejection": {
        "enabled": False,
        "max_abs_amplitude": None,
        "max_flat_fraction": None,
    },
}


def normalize_raw_preprocessing(
    config: Mapping[str, Any] | None,
    *,
    default_resample_hz: float = 256.0,
) -> dict[str, Any]:
    """Merge and validate the public raw-preprocessing schema."""
    normalized = deepcopy(DEFAULT_RAW_PREPROCESSING)
    normalized["resample_hz"] = float(default_resample_hz)
    if config is not None:
        unknown = sorted(set(config) - set(normalized))
        if unknown:
            raise ValueError(f"Unknown raw_preprocessing keys: {unknown}")
        for key, value in config.items():
            if isinstance(normalized[key], dict):
                if not isinstance(value, Mapping):
                    raise ValueError(f"raw_preprocessing.{key} must be a mapping")
                unknown_nested = sorted(set(value) - set(normalized[key]))
                if unknown_nested:
                    raise ValueError(
                        f"Unknown raw_preprocessing.{key} keys: {unknown_nested}"
                    )
                normalized[key].update(value)
            else:
                normalized[key] = value

    resample_hz = float(normalized["resample_hz"])
    if not np.isfinite(resample_hz) or resample_hz <= 0:
        raise ValueError("raw_preprocessing.resample_hz must be positive and finite")
    normalized["resample_hz"] = resample_hz
    nyquist = resample_hz / 2.0

    bandpass = normalized["bandpass"]
    bandpass["enabled"] = bool(bandpass["enabled"])
    bandpass["low_hz"] = float(bandpass["low_hz"])
    bandpass["high_hz"] = float(bandpass["high_hz"])
    if bandpass["enabled"] and not (
        0 < bandpass["low_hz"] < bandpass["high_hz"] < nyquist
    ):
        raise ValueError(
            "Enabled bandpass requires 0 < low_hz < high_hz < Nyquist"
        )

    notch = normalized["notch"]
    notch["enabled"] = bool(notch["enabled"])
    notch["frequency_hz"] = float(notch["frequency_hz"])
    notch["quality_factor"] = float(notch["quality_factor"])
    if notch["enabled"] and not 0 < notch["frequency_hz"] < nyquist:
        raise ValueError("Enabled notch frequency_hz must be below Nyquist")
    if notch["enabled"] and notch["quality_factor"] <= 0:
        raise ValueError("Enabled notch quality_factor must be positive")

    rereference = normalized["rereference"]
    mode = str(rereference["mode"]).strip().lower()
    if mode == "car":
        mode = "common_average"
    if mode not in {"none", "common_average"}:
        raise ValueError(
            "raw_preprocessing.rereference.mode must be 'none' or "
            "'common_average'"
        )
    rereference["mode"] = mode

    rejection = normalized["artifact_rejection"]
    rejection["enabled"] = bool(rejection["enabled"])
    for key in ("max_abs_amplitude", "max_flat_fraction"):
        value = rejection[key]
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"raw_preprocessing.artifact_rejection.{key} must be "
                    "non-negative and finite"
                )
        rejection[key] = value
    flat_limit = rejection["max_flat_fraction"]
    if flat_limit is not None and flat_limit > 1:
        raise ValueError("max_flat_fraction must be in [0, 1]")
    return normalized


def raw_preprocessing_hash(
    config: Mapping[str, Any] | None,
    *,
    channels: Sequence[str],
    default_resample_hz: float = 256.0,
) -> str:
    """Hash preprocessing plus channel order for cache namespace isolation."""
    payload = {
        "version": RAW_PREPROCESSING_VERSION,
        "channels": [str(channel) for channel in channels],
        "raw_preprocessing": normalize_raw_preprocessing(
            config, default_resample_hz=default_resample_hz
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_variant_name(config: Mapping[str, Any]) -> str:
    """Return a short human-readable cache namespace prefix."""
    parts = ["raw"]
    if config["bandpass"]["enabled"]:
        parts.append("bp")
    if config["notch"]["enabled"]:
        parts.append("notch")
    if config["rereference"]["mode"] == "common_average":
        parts.append("car")
    if config["artifact_rejection"]["enabled"]:
        parts.append("artifact-qc")
    return "-".join(parts)


def apply_raw_preprocessing(
    signals: np.ndarray,
    *,
    sampling_rate: float,
    config: Mapping[str, Any] | None,
) -> np.ndarray:
    """Apply zero-phase filters and rereferencing to ``[channel, time]``."""
    normalized = normalize_raw_preprocessing(
        config, default_resample_hz=sampling_rate
    )
    array = np.asarray(signals, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Raw preprocessing expects [channels, time], got {array.shape}"
        )
    if array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"Raw preprocessing input is too short: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Raw preprocessing input contains NaN or Inf")

    result = array.astype(np.float64, copy=True)
    bandpass = normalized["bandpass"]
    if bandpass["enabled"]:
        sos = butter(
            4,
            [bandpass["low_hz"], bandpass["high_hz"]],
            btype="bandpass",
            fs=float(sampling_rate),
            output="sos",
        )
        try:
            result = sosfiltfilt(sos, result, axis=1)
        except ValueError as exc:
            raise ValueError(
                "Raw EEG interval is too short for zero-phase bandpass filtering"
            ) from exc

    notch = normalized["notch"]
    if notch["enabled"]:
        numerator, denominator = iirnotch(
            notch["frequency_hz"],
            notch["quality_factor"],
            fs=float(sampling_rate),
        )
        try:
            result = filtfilt(numerator, denominator, result, axis=1)
        except ValueError as exc:
            raise ValueError(
                "Raw EEG interval is too short for zero-phase notch filtering"
            ) from exc

    if normalized["rereference"]["mode"] == "common_average":
        result = result - result.mean(axis=0, keepdims=True)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Raw preprocessing produced NaN or Inf")
    return result


def raw_window_artifact_metrics(signals: np.ndarray) -> dict[str, Any]:
    """Compute threshold-independent per-window diagnostics by channel."""
    array = np.asarray(signals, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [channels, time], got {array.shape}")
    finite = np.isfinite(array)
    differences = np.diff(array, axis=1)
    return {
        "max_abs_amplitude": float(np.nanmax(np.abs(array))),
        "max_flat_fraction": float(
            np.max(np.mean(np.abs(differences) <= 1e-6, axis=1))
        ),
        "channel_peak_to_peak": np.ptp(array, axis=1).astype(float).tolist(),
        "channel_variance": np.var(array, axis=1).astype(float).tolist(),
        "channel_flat_fraction": np.mean(
            np.abs(differences) <= 1e-6, axis=1
        ).astype(float).tolist(),
        "non_finite_fraction": float(1.0 - finite.mean()),
    }
