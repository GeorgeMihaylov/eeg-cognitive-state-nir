"""Offline detrending and wavelet shrinkage for EEG matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pywt


ThresholdMethod = Literal["universal", "bayes"]
ThresholdMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class WaveletDenoisingConfig:
    wavelet: str = "sym8"
    level: int | None = None
    threshold_method: ThresholdMethod = "bayes"
    threshold_mode: ThresholdMode = "soft"
    threshold_scale: float = 1.0

    def __post_init__(self) -> None:
        try:
            pywt.Wavelet(self.wavelet)
        except ValueError as exc:
            raise ValueError(f"Unknown wavelet: {self.wavelet!r}") from exc
        if self.level is not None and self.level < 1:
            raise ValueError("Wavelet level must be positive")
        if self.threshold_method not in {"universal", "bayes"}:
            raise ValueError("threshold_method must be 'universal' or 'bayes'")
        if self.threshold_mode not in {"soft", "hard"}:
            raise ValueError("threshold_mode must be 'soft' or 'hard'")
        if self.threshold_scale <= 0:
            raise ValueError("threshold_scale must be positive")


@dataclass(frozen=True)
class WaveletDenoisingReport:
    wavelet: str
    level: int
    noise_sigma: tuple[float, ...]
    thresholds: tuple[tuple[float, ...], ...]


def detrend_signal(signal: object, *, order: int = 1) -> np.ndarray:
    """Remove a polynomial trend independently from every channel."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("Signal must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    if order < 0 or order >= len(values):
        raise ValueError("Detrend order must be in [0, n_samples)")
    time = np.linspace(-1.0, 1.0, len(values))
    design = np.vander(time, N=order + 1, increasing=True)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def baseline_correct_epochs(
    epochs: object, *, baseline: slice
) -> np.ndarray:
    """Subtract each epoch/channel mean over an explicit baseline slice."""
    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3 or not values.shape[0]:
        raise ValueError("Epochs must be [epochs, samples, channels]")
    selected = values[:, baseline, :]
    if not selected.shape[1]:
        raise ValueError("Baseline slice selects no samples")
    return values - np.mean(selected, axis=1, keepdims=True)


def _threshold(detail: np.ndarray, sigma: float, config: WaveletDenoisingConfig) -> float:
    if config.threshold_method == "universal":
        value = sigma * np.sqrt(2.0 * np.log(max(detail.size, 2)))
    else:
        detail_variance = float(np.mean(detail * detail))
        signal_sigma = np.sqrt(max(detail_variance - sigma * sigma, 0.0))
        value = (
            sigma * sigma / signal_sigma
            if signal_sigma > np.finfo(float).eps
            else float(np.max(np.abs(detail)))
        )
    return float(config.threshold_scale * value)


def wavelet_denoise(
    signal: object,
    config: WaveletDenoisingConfig = WaveletDenoisingConfig(),
) -> tuple[np.ndarray, WaveletDenoisingReport]:
    """Denoise each channel with DWT detail-coefficient shrinkage."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("Signal must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    wavelet = pywt.Wavelet(config.wavelet)
    maximum_level = pywt.dwt_max_level(len(values), wavelet.dec_len)
    if maximum_level < 1:
        raise ValueError("Signal is too short for the selected wavelet")
    level = min(config.level or maximum_level, maximum_level)
    output = np.empty_like(values)
    sigmas: list[float] = []
    all_thresholds: list[tuple[float, ...]] = []

    for channel in range(values.shape[1]):
        coefficients = pywt.wavedec(
            values[:, channel], wavelet, level=level, mode="symmetric"
        )
        finest = coefficients[-1]
        sigma = float(
            np.median(np.abs(finest - np.median(finest))) / 0.6744897501960817
        )
        thresholds = tuple(
            _threshold(detail, sigma, config) for detail in coefficients[1:]
        )
        cleaned = [coefficients[0]] + [
            pywt.threshold(detail, threshold, mode=config.threshold_mode)
            for detail, threshold in zip(coefficients[1:], thresholds)
        ]
        output[:, channel] = pywt.waverec(
            cleaned, wavelet, mode="symmetric"
        )[: len(values)]
        sigmas.append(sigma)
        all_thresholds.append(thresholds)

    return output, WaveletDenoisingReport(
        wavelet=config.wavelet,
        level=level,
        noise_sigma=tuple(sigmas),
        thresholds=tuple(all_thresholds),
    )
