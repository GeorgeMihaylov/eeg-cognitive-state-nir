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
    notch_freq_hz: float | None = 50.0
    notch_quality_factor: float = 30.0

    def __post_init__(self) -> None:
        nyquist = self.sample_rate / 2.0
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist:
            raise ValueError("Bandpass frequencies must lie inside Nyquist")
        if self.bandpass_order < 1:
            raise ValueError("bandpass_order must be positive")
        if self.notch_freq_hz is not None and not 0 < self.notch_freq_hz < nyquist:
            raise ValueError("notch_freq_hz must lie inside Nyquist or be None")
        if self.notch_quality_factor <= 0:
            raise ValueError("notch_quality_factor must be positive")


def design_bandpass(config: FilterConfig):
    nyquist = config.sample_rate / 2.0
    low = config.bandpass_low_hz / nyquist
    high = config.bandpass_high_hz / nyquist
    b, a = butter(config.bandpass_order, [low, high], btype="band")
    return b, a


def design_notch(config: FilterConfig):
    if config.notch_freq_hz is None:
        raise ValueError("Cannot design a disabled notch filter")
    b, a = iirnotch(config.notch_freq_hz, config.notch_quality_factor, config.sample_rate)
    return b, a


def apply_offline(signal: np.ndarray, config: FilterConfig) -> np.ndarray:
    values = _validate_signal(signal)
    b_band, a_band = design_bandpass(config)
    filtered = filtfilt(b_band, a_band, values, axis=0)
    if config.notch_freq_hz is not None:
        b_notch, a_notch = design_notch(config)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
    return filtered


def _validate_signal(signal: object, n_channels: int | None = None) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Signal must be a non-empty [samples, channels] matrix")
    if n_channels is not None and values.shape[1] != n_channels:
        raise ValueError(f"Expected {n_channels} channels, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    return values


class StreamingFilter:

    def __init__(self, config: FilterConfig, n_channels: int):
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.config = config
        self.n_channels = n_channels
        self._b_band, self._a_band = design_bandpass(config)
        if config.notch_freq_hz is not None:
            self._b_notch, self._a_notch = design_notch(config)
        else:
            self._b_notch = self._a_notch = None

        self._base_zi_band = lfilter_zi(self._b_band, self._a_band)
        self._base_zi_notch = (
            lfilter_zi(self._b_notch, self._a_notch)
            if self._b_notch is not None and self._a_notch is not None
            else None
        )
        self.reset()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        values = _validate_signal(chunk, self.n_channels)
        if not self._initialized:
            self._zi_band = self._base_zi_band[:, None] * values[0][None, :]
            self._initialized = True
        filtered, self._zi_band = lfilter(
            self._b_band, self._a_band, values, axis=0, zi=self._zi_band
        )
        if self._b_notch is not None and self._a_notch is not None:
            if self._zi_notch is None:
                assert self._base_zi_notch is not None
                self._zi_notch = self._base_zi_notch[:, None] * filtered[0][None, :]
            filtered, self._zi_notch = lfilter(
                self._b_notch, self._a_notch, filtered, axis=0, zi=self._zi_notch
            )
        return filtered

    def reset(self) -> None:
        self._zi_band = np.zeros((len(self._base_zi_band), self.n_channels))
        self._zi_notch: np.ndarray | None = None
        self._initialized = False


def apply_causal(signal: np.ndarray, config: FilterConfig) -> np.ndarray:
    """Filter one complete recording exactly as the streaming filter would."""
    values = _validate_signal(signal)
    return StreamingFilter(config, values.shape[1]).process(values)
