"""Canonical record model and PM aggregation for gpn_data / Old_EEG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from cogstate.protocol import EEG_CHANNELS, PM_METRICS, WINDOW_SECONDS, WINDOW_SAMPLES


@dataclass(frozen=True)
class EEGWindow:
    eeg: np.ndarray
    subject_id: str
    record_id: str
    start_time: float
    end_time: float
    pm: Mapping[str, float]

    def __post_init__(self):
        signal = np.asarray(self.eeg)
        if signal.shape != (len(EEG_CHANNELS), WINDOW_SAMPLES):
            raise ValueError(f"Canonical EEG window must be [{len(EEG_CHANNELS)}, {WINDOW_SAMPLES}], got {signal.shape}")
        if self.end_time - self.start_time != WINDOW_SECONDS:
            raise ValueError("Canonical EEG windows must be exactly 10 seconds")
        unknown = set(self.pm) - set(PM_METRICS)
        if unknown:
            raise ValueError(f"Unknown PM metrics: {sorted(unknown)}")


def aggregate_pm_by_window(timestamps, values, *, window_start: float, n_windows: int, window_seconds: float = WINDOW_SECONDS):
    """Mean PM per absolute 10-second interval; NaN is retained when absent.

    Call this separately for each physical record.  It intentionally never
    carries values over a record boundary or interpolates missing intervals.
    """
    times = np.asarray(timestamps, dtype=float)
    pm = np.asarray(values, dtype=float)
    if pm.ndim != 2 or pm.shape[1] != len(PM_METRICS) or len(times) != len(pm):
        raise ValueError(f"values must have shape [samples, {len(PM_METRICS)}] and match timestamps")
    result = np.full((n_windows, len(PM_METRICS)), np.nan, dtype=float)
    # Use absolute buckets to match the canonical ``floor(Timestamp / 10)``
    # protocol even when the first sample is not exactly at a window boundary.
    first_bucket = int(np.floor(window_start / window_seconds))
    indices = np.floor(times / window_seconds).astype(int) - first_bucket
    for index in range(n_windows):
        mask = indices == index
        if mask.any():
            result[index] = np.nanmean(pm[mask], axis=0)
    return result


def aggregate_pm_statistics_by_window(timestamps, values, *, window_start: float, n_windows: int, window_seconds: float = WINDOW_SECONDS):
    """Return mean/std/min/max/last PM summaries for audit and ablations."""
    times, pm = np.asarray(timestamps, float), np.asarray(values, float)
    if pm.ndim != 2 or pm.shape[1] != len(PM_METRICS) or len(times) != len(pm):
        raise ValueError(f"values must have shape [samples, {len(PM_METRICS)}] and match timestamps")
    summaries = {name: np.full((n_windows, len(PM_METRICS)), np.nan) for name in ("mean", "std", "min", "max", "last")}
    first_bucket = int(np.floor(window_start / window_seconds))
    indices = np.floor(times / window_seconds).astype(int) - first_bucket
    for index in range(n_windows):
        selected = pm[indices == index]
        if len(selected):
            summaries["mean"][index] = np.nanmean(selected, axis=0)
            summaries["std"][index] = np.nanstd(selected, axis=0)
            summaries["min"][index] = np.nanmin(selected, axis=0)
            summaries["max"][index] = np.nanmax(selected, axis=0)
            summaries["last"][index] = selected[-1]
    return summaries
