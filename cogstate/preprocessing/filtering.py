from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, lfilter_zi


@dataclass
class FilterConfig:
    sample_rate: float
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 45.0
    bandpass_order: int = 4
    notch_freq_hz: float = 50.0
    notch_quality_factor: float = 30.0


def design_bandpass(config: FilterConfig):
    nyquist = config.sample_rate / 2.0
    low = config.bandpass_low_hz / nyquist
    high = config.bandpass_high_hz / nyquist
    b, a = butter(config.bandpass_order, [low, high], btype="band")
    return b, a


def design_notch(config: FilterConfig):
    b, a = iirnotch(config.notch_freq_hz, config.notch_quality_factor, config.sample_rate)
    return b, a


def apply_offline(signal: np.ndarray, config: FilterConfig) -> np.ndarray:
    b_band, a_band = design_bandpass(config)
    b_notch, a_notch = design_notch(config)

    filtered = filtfilt(b_band, a_band, signal, axis=0)
    filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
    return filtered


class StreamingFilter:

    def __init__(self, config: FilterConfig, n_channels: int):
        self.config = config
        self._b_band, self._a_band = design_bandpass(config)
        self._b_notch, self._a_notch = design_notch(config)

        zi_band = lfilter_zi(self._b_band, self._a_band)
        zi_notch = lfilter_zi(self._b_notch, self._a_notch)
        self._zi_band = np.tile(zi_band, (n_channels, 1)).T
        self._zi_notch = np.tile(zi_notch, (n_channels, 1)).T

    def process(self, chunk: np.ndarray) -> np.ndarray:
        filtered, self._zi_band = lfilter(
            self._b_band, self._a_band, chunk, axis=0, zi=self._zi_band
        )
        filtered, self._zi_notch = lfilter(
            self._b_notch, self._a_notch, filtered, axis=0, zi=self._zi_notch
        )
        return filtered

    def reset(self) -> None:
        zi_band = lfilter_zi(self._b_band, self._a_band)
        zi_notch = lfilter_zi(self._b_notch, self._a_notch)
        n_channels = self._zi_band.shape[1]
        self._zi_band = np.tile(zi_band, (n_channels, 1)).T
        self._zi_notch = np.tile(zi_notch, (n_channels, 1)).T
