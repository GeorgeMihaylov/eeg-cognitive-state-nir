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


    def __post_init__(self) -> None:
        values = [
            self.sample_rate,
            self.bandpass_low_hz,
            self.bandpass_high_hz,
            self.notch_freq_hz,
            self.notch_quality_factor,
        ]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("FilterConfig values must be finite")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        nyquist = self.sample_rate / 2.0
        if not (0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist):
            raise ValueError(
                "band-pass frequencies must satisfy "
                "0 < low < high < Nyquist"
            )
        if int(self.bandpass_order) != self.bandpass_order or self.bandpass_order <= 0:
            raise ValueError("bandpass_order must be a positive integer")
        if not (0 < self.notch_freq_hz < nyquist):
            raise ValueError("notch_freq_hz must satisfy 0 < notch < Nyquist")
        if self.notch_quality_factor <= 0:
            raise ValueError("notch_quality_factor must be positive")


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
        if int(n_channels) != n_channels or n_channels <= 0:
            raise ValueError("n_channels must be a positive integer")

        self.config = config
        self._n_channels = int(n_channels)
        self._b_band, self._a_band = design_bandpass(config)
        self._b_notch, self._a_notch = design_notch(config)

        self._zi_band_template = lfilter_zi(
            self._b_band, self._a_band
        )
        self._zi_notch_template = lfilter_zi(
            self._b_notch, self._a_notch
        )

        self._zi_band = None
        self._zi_notch = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2:
            raise ValueError(
                "StreamingFilter.process expects [n_samples, n_channels]"
            )
        if chunk.shape[0] == 0:
            raise ValueError("StreamingFilter.process received zero samples")
        if chunk.shape[1] != self._n_channels:
            raise ValueError(
                f"StreamingFilter was configured for {self._n_channels} channels, "
                f"got {chunk.shape[1]}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("StreamingFilter input contains NaN or Inf")

        if self._zi_band is None:
            self._zi_band = (
                self._zi_band_template[:, None]
                * chunk[0][None, :]
            )

        filtered, self._zi_band = lfilter(
            self._b_band,
            self._a_band,
            chunk,
            axis=0,
            zi=self._zi_band,
        )

        if self._zi_notch is None:
            self._zi_notch = (
                self._zi_notch_template[:, None]
                * filtered[0][None, :]
            )

        filtered, self._zi_notch = lfilter(
            self._b_notch,
            self._a_notch,
            filtered,
            axis=0,
            zi=self._zi_notch,
        )

        return filtered

    def reset(self) -> None:
        self._zi_band = None
        self._zi_notch = None
