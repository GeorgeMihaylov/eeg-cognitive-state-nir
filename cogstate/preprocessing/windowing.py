from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WindowingConfig:
    sample_rate: float
    window_size_s: float = 2.0
    step_size_s: float = 0.5
    drop_last_incomplete: bool = True

    @property
    def window_size_samples(self) -> int:
        return int(round(self.window_size_s * self.sample_rate))

    @property
    def step_size_samples(self) -> int:
        return int(round(self.step_size_s * self.sample_rate))

    @property
    def overlap_ratio(self) -> float:
        return 1.0 - (self.step_size_s / self.window_size_s)


def segment_signal(signal: np.ndarray, config: WindowingConfig) -> np.ndarray:
    window_len = config.window_size_samples
    step = config.step_size_samples

    if window_len <= 0 or step <= 0:
        raise ValueError("window_size_s и step_size_s должны давать положительное число отсчётов")
    if signal.shape[0] < window_len:
        return np.empty((0, window_len, signal.shape[1]), dtype=signal.dtype)

    n_windows = (signal.shape[0] - window_len) // step + 1
    windows = np.stack([
        signal[i * step: i * step + window_len]
        for i in range(n_windows)
    ])
    return windows


def segment_with_timestamps(
    signal: np.ndarray,
    timestamps: np.ndarray,
    config: WindowingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    windows = segment_signal(signal, config)
    step = config.step_size_samples
    n_windows = windows.shape[0]
    window_start_times = np.array([timestamps[i * step] for i in range(n_windows)])
    return windows, window_start_times


def align_labels_to_windows(
    window_start_times: np.ndarray,
    label_timestamps: np.ndarray,
    label_values: np.ndarray,
    tolerance_s: float = 0.25,
) -> np.ndarray:
    aligned = np.full(window_start_times.shape[0], np.nan, dtype=float)

    for i, t in enumerate(window_start_times):
        idx = np.argmin(np.abs(label_timestamps - t))
        if np.abs(label_timestamps[idx] - t) <= tolerance_s:
            aligned[i] = label_values[idx]

    return aligned
