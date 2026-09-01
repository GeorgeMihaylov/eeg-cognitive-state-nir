"""Pairwise EEG connectivity summaries with explicit computed-pair semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import signal

from ._validation import ordered_bands, validate_sample_rate, validate_window
from .spectral import DEFAULT_BANDS


CONNECTIVITY_METRICS = ("correlation", "coherence", "plv")
SUMMARY_STATISTICS = ("mean", "std", "max")


@dataclass(frozen=True)
class ConnectivityConfig:
    sample_rate: float
    bands: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )
    metrics: tuple[str, ...] = CONNECTIVITY_METRICS
    summary_statistics: tuple[str, ...] = SUMMARY_STATISTICS
    nperseg: int = 128
    noverlap: int | None = None
    window: str = "hann"
    detrend: str = "constant"
    max_channel_pairs: int | None = None
    plv_mode: Literal["broadband", "band"] = "broadband"
    phase_filter_order: int = 4

    def __post_init__(self) -> None:
        validate_sample_rate(self.sample_rate)
        ordered_bands(self.bands, sample_rate=self.sample_rate)
        unknown_metrics = set(self.metrics) - set(CONNECTIVITY_METRICS)
        if unknown_metrics or not self.metrics:
            raise ValueError(
                f"unsupported connectivity metrics: {sorted(unknown_metrics)}"
            )
        unknown_statistics = set(self.summary_statistics) - set(SUMMARY_STATISTICS)
        if unknown_statistics or not self.summary_statistics:
            raise ValueError(
                f"unsupported connectivity summaries: {sorted(unknown_statistics)}"
            )
        if int(self.nperseg) < 2:
            raise ValueError("nperseg must be at least 2")
        if self.noverlap is not None and int(self.noverlap) < 0:
            raise ValueError("noverlap must be non-negative or None")
        if self.max_channel_pairs is not None and int(self.max_channel_pairs) <= 0:
            raise ValueError("max_channel_pairs must be positive or None")
        if self.plv_mode not in {"broadband", "band"}:
            raise ValueError("plv_mode must be 'broadband' or 'band'")
        if int(self.phase_filter_order) < 1:
            raise ValueError("phase_filter_order must be positive")

    @property
    def ordered_bands(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        return ordered_bands(self.bands, sample_rate=self.sample_rate)


def channel_pairs(
    n_channels: int,
    max_channel_pairs: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic unordered pairs in lexicographic channel order."""
    n_channels = int(n_channels)
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    pairs = tuple(combinations(range(n_channels), 2))
    if max_channel_pairs is not None:
        pairs = pairs[: int(max_channel_pairs)]
    return pairs


def _pairwise_correlation_values(
    window: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    centered = window - np.mean(window, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(np.square(centered), axis=0))
    values = np.zeros(len(pairs), dtype=float)
    for index, (first, second) in enumerate(pairs):
        denominator = norms[first] * norms[second]
        if denominator > np.finfo(float).eps:
            values[index] = np.dot(centered[:, first], centered[:, second]) / denominator
    return np.clip(values, -1.0, 1.0)


def _coherence_band_values(
    window: np.ndarray,
    config: ConnectivityConfig,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    values = {
        name: np.zeros(len(pairs), dtype=float)
        for name, _ in config.ordered_bands
    }
    nperseg = min(int(config.nperseg), window.shape[0])
    if config.noverlap is not None and int(config.noverlap) >= nperseg:
        raise ValueError("noverlap must be smaller than effective nperseg")
    for pair_index, (first, second) in enumerate(pairs):
        with np.errstate(divide="ignore", invalid="ignore"):
            frequencies, spectrum = signal.coherence(
                window[:, first],
                window[:, second],
                fs=float(config.sample_rate),
                window=config.window,
                nperseg=nperseg,
                noverlap=config.noverlap,
                detrend=config.detrend,
            )
        spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
        for name, band in config.ordered_bands:
            mask = (frequencies >= band[0]) & (frequencies <= band[1])
            if np.any(mask):
                values[name][pair_index] = float(np.mean(spectrum[mask]))
    return values


def _plv_values(
    window: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    phases = np.angle(signal.hilbert(window, axis=0))
    channel_variance = np.var(window, axis=0)
    values = np.zeros(len(pairs), dtype=float)
    for index, (first, second) in enumerate(pairs):
        if (
            channel_variance[first] <= np.finfo(float).eps
            or channel_variance[second] <= np.finfo(float).eps
        ):
            continue
        difference = phases[:, first] - phases[:, second]
        values[index] = float(np.abs(np.mean(np.exp(1j * difference))))
    return np.clip(values, 0.0, 1.0)


def summarize_connectivity_values(
    values: np.ndarray,
    statistics: Sequence[str] = SUMMARY_STATISTICS,
) -> dict[str, float]:
    computed = np.asarray(values, dtype=float)
    if computed.ndim != 1:
        raise ValueError("connectivity values must be one-dimensional")
    if computed.size == 0:
        return {str(name): 0.0 for name in statistics}
    if not np.isfinite(computed).all():
        raise ValueError("computed connectivity values contain NaN or Inf")
    functions = {
        "mean": np.mean,
        "std": np.std,
        "max": np.max,
    }
    return {
        str(name): float(functions[str(name)](computed))
        for name in statistics
    }


def summarize_connectivity_matrix(
    matrix: np.ndarray,
    computed_pairs: Sequence[tuple[int, int]] | None = None,
    statistics: Sequence[str] = SUMMARY_STATISTICS,
) -> dict[str, float]:
    """Summarize only explicitly computed pairs, never placeholder entries."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("connectivity matrix must be square")
    if computed_pairs is None:
        selected = values[np.triu_indices(values.shape[0], k=1)]
        selected = selected[np.isfinite(selected)]
    else:
        pairs = tuple((int(first), int(second)) for first, second in computed_pairs)
        selected = np.asarray([values[first, second] for first, second in pairs])
    return summarize_connectivity_values(selected, statistics)


def compute_correlation_matrix(window: np.ndarray) -> np.ndarray:
    signal_window = validate_window(window)
    pairs = channel_pairs(signal_window.shape[1])
    matrix = np.eye(signal_window.shape[1], dtype=float)
    for value, (first, second) in zip(
        _pairwise_correlation_values(signal_window, pairs), pairs
    ):
        matrix[first, second] = matrix[second, first] = value
    return matrix


def compute_coherence_matrix(
    window: np.ndarray,
    config: ConnectivityConfig,
    band: tuple[float, float],
) -> np.ndarray:
    """Return a pair-budgeted coherence matrix with uncomputed pairs as NaN."""
    signal_window = validate_window(window)
    pairs = channel_pairs(signal_window.shape[1], config.max_channel_pairs)
    matrix = np.full(
        (signal_window.shape[1], signal_window.shape[1]), np.nan, dtype=float
    )
    np.fill_diagonal(matrix, 1.0)
    temporary = ConnectivityConfig(
        sample_rate=config.sample_rate,
        bands={"selected": tuple(band)},
        metrics=("coherence",),
        summary_statistics=config.summary_statistics,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
        window=config.window,
        detrend=config.detrend,
        max_channel_pairs=config.max_channel_pairs,
        plv_mode=config.plv_mode,
        phase_filter_order=config.phase_filter_order,
    )
    pair_values = _coherence_band_values(signal_window, temporary, pairs)["selected"]
    for value, (first, second) in zip(pair_values, pairs):
        matrix[first, second] = matrix[second, first] = value
    return matrix


def compute_plv_matrix(
    window: np.ndarray,
    config: ConnectivityConfig,
    band: tuple[float, float] | None = None,
) -> np.ndarray:
    signal_window = validate_window(window)
    if band is not None:
        low, high = band
        if not 0.0 < low < high < config.sample_rate / 2.0:
            raise ValueError("PLV band must lie strictly below Nyquist")
        sos = signal.butter(
            int(config.phase_filter_order),
            (low, high),
            btype="bandpass",
            fs=float(config.sample_rate),
            output="sos",
        )
        try:
            signal_window = signal.sosfiltfilt(sos, signal_window, axis=0)
        except ValueError as exc:
            raise ValueError("window is too short for band-limited PLV") from exc
    pairs = channel_pairs(signal_window.shape[1], config.max_channel_pairs)
    matrix = np.full(
        (signal_window.shape[1], signal_window.shape[1]), np.nan, dtype=float
    )
    np.fill_diagonal(matrix, 1.0)
    for value, (first, second) in zip(_plv_values(signal_window, pairs), pairs):
        matrix[first, second] = matrix[second, first] = value
    return matrix


def extract_connectivity_features(
    window: np.ndarray,
    config: ConnectivityConfig,
) -> dict[str, np.ndarray]:
    signal_window = validate_window(window)
    pairs = channel_pairs(signal_window.shape[1], config.max_channel_pairs)
    features: dict[str, np.ndarray] = {}
    if "correlation" in config.metrics:
        summary = summarize_connectivity_values(
            _pairwise_correlation_values(signal_window, pairs),
            config.summary_statistics,
        )
        for statistic in config.summary_statistics:
            features[f"correlation_{statistic}"] = np.asarray([summary[statistic]])
    if "coherence" in config.metrics:
        band_values = _coherence_band_values(signal_window, config, pairs)
        for band_name, _ in config.ordered_bands:
            summary = summarize_connectivity_values(
                band_values[band_name], config.summary_statistics
            )
            for statistic in config.summary_statistics:
                features[f"coherence_{band_name}_{statistic}"] = np.asarray(
                    [summary[statistic]]
                )
    if "plv" in config.metrics:
        if config.plv_mode == "broadband":
            summary = summarize_connectivity_values(
                _plv_values(signal_window, pairs), config.summary_statistics
            )
            for statistic in config.summary_statistics:
                features[f"plv_{statistic}"] = np.asarray([summary[statistic]])
        else:
            for band_name, band in config.ordered_bands:
                matrix = compute_plv_matrix(signal_window, config, band)
                summary = summarize_connectivity_matrix(
                    matrix, computed_pairs=pairs, statistics=config.summary_statistics
                )
                for statistic in config.summary_statistics:
                    features[f"plv_{band_name}_{statistic}"] = np.asarray(
                        [summary[statistic]]
                    )
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("connectivity feature extraction produced NaN or Inf")
    return features


def feature_names(config: ConnectivityConfig) -> list[str]:
    names: list[str] = []
    if "correlation" in config.metrics:
        names.extend(
            f"correlation_{statistic}"
            for statistic in config.summary_statistics
        )
    if "coherence" in config.metrics:
        for band_name, _ in config.ordered_bands:
            names.extend(
                f"coherence_{band_name}_{statistic}"
                for statistic in config.summary_statistics
            )
    if "plv" in config.metrics:
        if config.plv_mode == "broadband":
            names.extend(f"plv_{statistic}" for statistic in config.summary_statistics)
        else:
            for band_name, _ in config.ordered_bands:
                names.extend(
                    f"plv_{band_name}_{statistic}"
                    for statistic in config.summary_statistics
                )
    return names
