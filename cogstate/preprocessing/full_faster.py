"""FASTER artifact rejection for epoched EEG and a lightweight online subset.

The batch pipeline follows the four subject-level stages from Nolan et al.:
global channels, epochs, ICA components, and channels within individual epochs.
Arrays use ``[epochs, samples, channels]`` throughout this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.signal import lfilter, welch
from scipy.special import eval_legendre
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


InterpolationMethod = Literal["auto", "mean", "spherical"]


@dataclass
class FasterConfig:
    z_threshold: float = 3.0
    max_iter: int = 1
    interpolate_bad_channels: bool = True
    interpolate_bad_channel_epoch: bool = True
    interpolation_method: InterpolationMethod = "auto"
    spline_stiffness: int = 4
    spline_terms: int = 50
    spline_regularization: float = 1e-5
    hurst_max_lag: int = 100
    spectral_slope_band_hz: Tuple[float, float] = (8.0, 45.0)
    run_ica: bool = True
    ica_n_components: Optional[int] = None
    ica_max_iter: int = 500
    ica_random_state: int = 42
    average_reference: bool = True

    def __post_init__(self) -> None:
        if self.z_threshold <= 0 or self.max_iter < 1:
            raise ValueError("FASTER threshold must be positive and max_iter >= 1")
        if self.interpolation_method not in {"auto", "mean", "spherical"}:
            raise ValueError("Unknown interpolation method")
        if self.spline_stiffness < 2 or self.spline_terms < 1:
            raise ValueError("Invalid spherical-spline parameters")
        if self.spline_regularization < 0:
            raise ValueError("spline_regularization cannot be negative")
        if self.ica_n_components is not None and int(self.ica_n_components) < 2:
            raise ValueError("ica_n_components must be at least 2 or None")
        if int(self.ica_max_iter) < 1:
            raise ValueError("ica_max_iter must be positive")
        low, high = self.spectral_slope_band_hz
        if not np.isfinite((low, high)).all() or not 0 < low < high:
            raise ValueError("spectral_slope_band_hz must satisfy 0 < low < high")


def _signal_matrix(signal: object) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Signal must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    return values


def _epoch_tensor(epochs: object) -> np.ndarray:
    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError("Epochs must be a non-empty [epochs, samples, channels] tensor")
    if not np.isfinite(values).all():
        raise ValueError("Epochs contain non-finite values")
    return values


def _find_outliers(scores: object, config: FasterConfig) -> List[int]:
    """Iteratively find z-score outliers while avoiding masking by extremes."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    remaining = np.arange(len(values))
    outliers: list[int] = []
    for _ in range(config.max_iter):
        if len(remaining) < 2:
            break
        subset = values[remaining]
        std = float(np.std(subset))
        if not np.isfinite(std) or std <= np.finfo(float).eps:
            break
        z_scores = (subset - np.mean(subset)) / std
        found_mask = np.abs(z_scores) > config.z_threshold
        if not np.any(found_mask):
            break
        outliers.extend(int(index) for index in remaining[found_mask])
        remaining = remaining[~found_mask]
    return sorted(set(outliers))


def hurst_exponent(x: np.ndarray, max_lag: int = 100) -> float:
    """Estimate the second-order Hurst exponent used by FASTER."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return 0.5
    # ``max_lag`` is retained for API compatibility with the earlier estimator.
    del max_lag
    accumulated = np.cumsum(np.diff(x))[None, :]
    narrow = lfilter([1.0, -2.0, 1.0], 1.0, accumulated, axis=1)[:, 2:-1]
    wide = lfilter([1.0, 0.0, -2.0, 0.0, 1.0], 1.0, accumulated, axis=1)[:, 4:-1]
    narrow_power = float(np.mean(narrow**2))
    wide_power = float(np.mean(wide**2))
    if narrow_power <= np.finfo(float).eps or wide_power <= 0:
        return 0.5
    return float(0.5 * np.log2(wide_power / narrow_power))


def spectral_slope(
    x: np.ndarray, sample_rate: float, band_hz: Tuple[float, float]
) -> float:
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 1024))
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1]) & (psd > 0)
    if mask.sum() < 2:
        return 0.0
    selected = psd[mask]
    return float(np.mean(np.diff(selected)))


def _channel_metric_scores(
    signal: np.ndarray, config: FasterConfig
) -> dict[str, np.ndarray]:
    centered = signal - np.mean(signal, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((signal.shape[1], signal.shape[1])),
        where=denominator > np.finfo(float).eps,
    )
    np.fill_diagonal(correlation, 0.0)
    mean_correlation = np.sum(correlation, axis=0) / max(signal.shape[1] - 1, 1)
    channel_kurtosis = np.array(
        [
            float(kurtosis(signal[:, ch], fisher=True))
            if np.std(signal[:, ch]) > np.finfo(float).eps
            else 0.0
            for ch in range(signal.shape[1])
        ]
    )
    return {
        "variance": np.var(signal, axis=0),
        "correlation": mean_correlation,
        "hurst": np.array(
            [
                hurst_exponent(signal[:, ch], config.hurst_max_lag)
                for ch in range(signal.shape[1])
            ]
        ),
        "kurtosis": np.nan_to_num(channel_kurtosis, nan=0.0),
    }


def compute_channel_stats(signal: np.ndarray, config: FasterConfig) -> np.ndarray:
    scores = _channel_metric_scores(_signal_matrix(signal), config)
    return np.column_stack(tuple(scores.values()))


def detect_bad_channels_by_metric(
    signal: np.ndarray, config: FasterConfig
) -> dict[str, List[int]]:
    scores = _channel_metric_scores(_signal_matrix(signal), config)
    return {name: _find_outliers(values, config) for name, values in scores.items()}


def _union_by_metric(values: dict[str, Sequence[int]]) -> List[int]:
    return sorted({int(index) for indices in values.values() for index in indices})


def detect_bad_channels(signal: np.ndarray, config: FasterConfig) -> List[int]:
    return _union_by_metric(detect_bad_channels_by_metric(signal, config))


def _normalized_positions(channel_positions: object, n_channels: int) -> np.ndarray:
    positions = np.asarray(channel_positions, dtype=float)
    if positions.shape != (n_channels, 3) or not np.isfinite(positions).all():
        raise ValueError("channel_positions must be a finite [channels, 3] matrix")
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Every channel position must be non-zero")
    return positions / norms


def _spherical_spline_weights(
    positions: np.ndarray,
    good_channels: Sequence[int],
    bad_channels: Sequence[int],
    config: FasterConfig,
) -> np.ndarray:
    good = positions[np.asarray(good_channels)]
    bad = positions[np.asarray(bad_channels)]

    def g(cosine: np.ndarray) -> np.ndarray:
        result = np.zeros_like(cosine)
        for degree in range(1, config.spline_terms + 1):
            scale = (2 * degree + 1) / (
                (degree * (degree + 1)) ** config.spline_stiffness
            )
            result += scale * eval_legendre(degree, cosine)
        return result / (4.0 * np.pi)

    g_good = g(np.clip(good @ good.T, -1.0, 1.0))
    g_bad_good = g(np.clip(bad @ good.T, -1.0, 1.0))
    count = len(good_channels)
    system = np.block(
        [
            [
                g_good + config.spline_regularization * np.eye(count),
                np.ones((count, 1)),
            ],
            [np.ones((1, count)), np.zeros((1, 1))],
        ]
    )
    evaluation = np.column_stack((g_bad_good, np.ones(len(bad_channels))))
    inverse = np.linalg.pinv(system)
    return evaluation @ inverse[:, :count]


def interpolate_channels(
    signal: np.ndarray,
    bad_channels: List[int],
    channel_positions: object | None = None,
    *,
    config: FasterConfig | None = None,
) -> np.ndarray:
    """Interpolate channels using spherical splines or an explicit mean fallback."""
    values = _signal_matrix(signal)
    if not bad_channels:
        return values.copy()
    cfg = config or FasterConfig(run_ica=False)
    bad = sorted({int(index) for index in bad_channels})
    if bad[0] < 0 or bad[-1] >= values.shape[1]:
        raise ValueError("Bad channel index is out of range")
    good = [index for index in range(values.shape[1]) if index not in bad]
    if not good:
        raise ValueError("Cannot interpolate when every channel is bad")

    method = cfg.interpolation_method
    if method == "auto":
        method = "spherical" if channel_positions is not None else "mean"
    result = values.copy()
    if method == "spherical":
        if channel_positions is None:
            raise ValueError("Spherical interpolation requires channel_positions")
        positions = _normalized_positions(channel_positions, values.shape[1])
        weights = _spherical_spline_weights(positions, good, bad, cfg)
        result[:, bad] = values[:, good] @ weights.T
    else:
        result[:, bad] = np.mean(values[:, good], axis=1, keepdims=True)
    return result


def _epoch_metric_scores(epochs: np.ndarray) -> dict[str, np.ndarray]:
    channel_means = epochs.mean(axis=1)
    deviation = np.abs(channel_means - channel_means.mean(axis=0)).mean(axis=1)
    return {
        "amplitude": np.ptp(epochs, axis=1).mean(axis=1),
        "variance": np.var(epochs, axis=1).mean(axis=1),
        "deviation": deviation,
    }


def compute_epoch_stats(epochs: np.ndarray) -> np.ndarray:
    scores = _epoch_metric_scores(_epoch_tensor(epochs))
    return np.column_stack(tuple(scores.values()))


def detect_bad_epochs_by_metric(
    epochs: np.ndarray, config: FasterConfig
) -> dict[str, List[int]]:
    return {
        name: _find_outliers(values, config)
        for name, values in _epoch_metric_scores(_epoch_tensor(epochs)).items()
    }


def detect_bad_epochs(epochs: np.ndarray, config: FasterConfig) -> List[int]:
    return _union_by_metric(detect_bad_epochs_by_metric(epochs, config))


def _eog_component_correlations(sources: np.ndarray, eog_signal: object) -> np.ndarray:
    eog = np.asarray(eog_signal, dtype=float)
    if eog.ndim == 1:
        eog = eog[:, None]
    if eog.ndim != 2 or len(eog) != len(sources):
        raise ValueError("EOG must have the same sample count as ICA sources")
    combined = np.column_stack((sources, eog))
    correlations = np.corrcoef(combined, rowvar=False)
    block = correlations[: sources.shape[1], sources.shape[1] :]
    return np.nan_to_num(np.max(np.abs(block), axis=1), nan=0.0)


def _component_metric_scores(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    scores = {
        "spatial_kurtosis": np.nan_to_num(
            kurtosis(mixing_matrix, axis=0, fisher=True), nan=0.0
        ),
        "hurst": np.array(
            [
                hurst_exponent(sources[:, index], config.hurst_max_lag)
                for index in range(sources.shape[1])
            ]
        ),
        "power_gradient": np.array(
            [
                spectral_slope(
                    sources[:, index], sample_rate, config.spectral_slope_band_hz
                )
                for index in range(sources.shape[1])
            ]
        ),
        "median_gradient": np.median(np.abs(np.diff(sources, axis=0)), axis=0),
    }
    if eog_signal is not None:
        scores["eog_correlation"] = _eog_component_correlations(sources, eog_signal)
    return scores


def compute_component_stats(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    scores = _component_metric_scores(
        _signal_matrix(sources),
        np.asarray(mixing_matrix),
        sample_rate,
        config,
        eog_signal,
    )
    return np.column_stack(tuple(scores.values()))


def detect_bad_components_by_metric(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> dict[str, List[int]]:
    scores = _component_metric_scores(
        _signal_matrix(sources),
        np.asarray(mixing_matrix),
        sample_rate,
        config,
        eog_signal,
    )
    return {name: _find_outliers(values, config) for name, values in scores.items()}


def detect_bad_components(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    sample_rate: float,
    config: FasterConfig,
    eog_signal: Optional[np.ndarray] = None,
) -> List[int]:
    return _union_by_metric(
        detect_bad_components_by_metric(
            sources, mixing_matrix, sample_rate, config, eog_signal
        )
    )


def compute_channel_epoch_stats(epochs: np.ndarray) -> np.ndarray:
    values = _epoch_tensor(epochs)
    channel_means = values.mean(axis=1)
    deviation = np.abs(channel_means - channel_means.mean(axis=0))
    return np.stack(
        (
            np.var(values, axis=1),
            np.median(np.abs(np.diff(values, axis=1)), axis=1),
            np.ptp(values, axis=1),
            deviation,
        ),
        axis=2,
    )


def detect_bad_channel_epoch_pairs(
    epochs: np.ndarray, config: FasterConfig
) -> List[Tuple[int, int]]:
    """Detect channel outliers independently inside every epoch."""
    features = compute_channel_epoch_stats(epochs)
    bad_pairs: set[Tuple[int, int]] = set()
    for epoch_index in range(features.shape[0]):
        for feature_index in range(features.shape[2]):
            outliers = _find_outliers(features[epoch_index, :, feature_index], config)
            for channel_index in outliers:
                bad_pairs.add((epoch_index, channel_index))
    return sorted(bad_pairs)


def interpolate_channel_epoch_pairs(
    epochs: np.ndarray,
    bad_pairs: List[Tuple[int, int]],
    channel_positions: object | None = None,
    *,
    config: FasterConfig | None = None,
) -> np.ndarray:
    values = _epoch_tensor(epochs).copy()
    bad_by_epoch: dict[int, List[int]] = {}
    for epoch_index, channel_index in bad_pairs:
        bad_by_epoch.setdefault(int(epoch_index), []).append(int(channel_index))
    for epoch_index, bad_channels in bad_by_epoch.items():
        values[epoch_index] = interpolate_channels(
            values[epoch_index], bad_channels, channel_positions, config=config
        )
    return values


@dataclass
class FasterReport:
    bad_channels: List[int] = field(default_factory=list)
    bad_epochs: List[int] = field(default_factory=list)
    bad_components: List[int] = field(default_factory=list)
    bad_channel_epoch_pairs: List[Tuple[int, int]] = field(default_factory=list)
    bad_channel_epoch_pairs_original: List[Tuple[int, int]] = field(
        default_factory=list
    )
    kept_epoch_indices: List[int] = field(default_factory=list)
    channel_bads_by_metric: dict[str, List[int]] = field(default_factory=dict)
    epoch_bads_by_metric: dict[str, List[int]] = field(default_factory=dict)
    component_bads_by_metric: dict[str, List[int]] = field(default_factory=dict)
    ica_fitted: bool = False
    ica_converged: bool | None = None
    ica_input_rank: int | None = None
    ica_n_components: int | None = None
    n_input_epochs: int = 0
    interpolation_method: str = "none"

    @property
    def kept_epoch_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_input_epochs, dtype=bool)
        mask[self.kept_epoch_indices] = True
        return mask


def _prepare_eog_epochs(eog_signal: object, epoch_shape: tuple[int, int]) -> np.ndarray:
    eog = np.asarray(eog_signal, dtype=float)
    if eog.ndim == 1:
        eog = eog[:, None]
    if eog.ndim == 2 and eog.shape[0] == epoch_shape[0] * epoch_shape[1]:
        eog = eog.reshape(epoch_shape[0], epoch_shape[1], -1)
    if eog.ndim != 3 or eog.shape[:2] != epoch_shape:
        raise ValueError("EOG must be [epochs, samples, channels] or flattened")
    if not np.isfinite(eog).all():
        raise ValueError("EOG contains non-finite values")
    return eog


def run_faster(
    epochs: np.ndarray,
    config: Optional[FasterConfig] = None,
    *,
    sample_rate: float | None = None,
    eog_signal: object | None = None,
    channel_positions: object | None = None,
) -> Tuple[np.ndarray, FasterReport]:
    """Run the complete four-stage FASTER pipeline.

    Returned epochs exclude globally bad epochs. Use ``report.kept_epoch_mask``
    to apply the identical selection to labels.
    """
    cfg = config or FasterConfig()
    values = _epoch_tensor(epochs).copy()
    if cfg.run_ica and (
        sample_rate is None or not np.isfinite(sample_rate) or sample_rate <= 0
    ):
        raise ValueError("A positive sample_rate is required for the ICA stage")
    report = FasterReport(n_input_epochs=len(values))
    continuous = values.reshape(-1, values.shape[2])

    report.channel_bads_by_metric = detect_bad_channels_by_metric(continuous, cfg)
    report.bad_channels = _union_by_metric(report.channel_bads_by_metric)
    good_channels = [
        index for index in range(values.shape[2]) if index not in report.bad_channels
    ]
    if not good_channels:
        raise ValueError("FASTER classified every EEG channel as bad")

    report.epoch_bads_by_metric = detect_bad_epochs_by_metric(
        values[:, :, good_channels], cfg
    )
    report.bad_epochs = _union_by_metric(report.epoch_bads_by_metric)
    report.kept_epoch_indices = [
        index for index in range(len(values)) if index not in report.bad_epochs
    ]
    if not report.kept_epoch_indices:
        raise ValueError("FASTER classified every epoch as bad")
    clean = values[report.kept_epoch_indices].copy()

    eog_epochs = None
    if eog_signal is not None:
        eog_epochs = _prepare_eog_epochs(eog_signal, values.shape[:2])

    if cfg.run_ica:
        assert sample_rate is not None
        ica_data = clean[:, :, good_channels].reshape(-1, len(good_channels))
        input_rank = int(np.linalg.matrix_rank(ica_data))
        report.ica_input_rank = input_rank
        if input_rank < 2:
            raise ValueError(
                f"FASTER ICA input rank is too low for decomposition: {input_rank}"
            )
        n_components = cfg.ica_n_components or len(good_channels)
        n_components = min(
            int(n_components), len(good_channels), len(ica_data) - 1, input_rank
        )
        if n_components < 2:
            raise ValueError("At least two good channels are required for FASTER ICA")
        report.ica_n_components = n_components
        ica = FastICA(
            n_components=n_components,
            max_iter=cfg.ica_max_iter,
            random_state=cfg.ica_random_state,
            whiten="unit-variance",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            sources = ica.fit_transform(ica_data)
        report.ica_converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        if not report.ica_converged:
            warnings.warn(
                "Full FASTER FastICA did not converge; diagnostics are retained "
                "and no fallback transform is substituted.",
                ConvergenceWarning,
                stacklevel=2,
            )
        flat_eog = None
        if eog_epochs is not None:
            flat_eog = eog_epochs[report.kept_epoch_indices].reshape(len(ica_data), -1)
        report.component_bads_by_metric = detect_bad_components_by_metric(
            sources, ica.mixing_, sample_rate, cfg, flat_eog
        )
        report.bad_components = _union_by_metric(report.component_bads_by_metric)
        if report.bad_components:
            sources[:, report.bad_components] = 0.0
            reconstructed = ica.inverse_transform(sources)
            clean[:, :, good_channels] = reconstructed.reshape(
                len(clean), clean.shape[1], len(good_channels)
            )
        report.ica_fitted = True

    local_pairs = detect_bad_channel_epoch_pairs(clean[:, :, good_channels], cfg)
    report.bad_channel_epoch_pairs = [
        (epoch_index, good_channels[channel_index])
        for epoch_index, channel_index in local_pairs
    ]
    report.bad_channel_epoch_pairs_original = [
        (report.kept_epoch_indices[epoch_index], good_channels[channel_index])
        for epoch_index, channel_index in local_pairs
    ]

    if report.bad_channels and cfg.interpolate_bad_channels:
        clean = np.stack(
            [
                interpolate_channels(
                    epoch, report.bad_channels, channel_positions, config=cfg
                )
                for epoch in clean
            ]
        )
    if report.bad_channel_epoch_pairs and cfg.interpolate_bad_channel_epoch:
        clean = interpolate_channel_epoch_pairs(
            clean,
            report.bad_channel_epoch_pairs,
            channel_positions,
            config=cfg,
        )

    if cfg.average_reference:
        clean -= np.mean(clean, axis=2, keepdims=True)
    if report.bad_channels or report.bad_channel_epoch_pairs:
        report.interpolation_method = (
            "spherical"
            if cfg.interpolation_method == "spherical"
            or (cfg.interpolation_method == "auto" and channel_positions is not None)
            else "mean"
        )
    return clean, report


def apply_faster_online(
    signal: np.ndarray, config: Optional[FasterConfig] = None
) -> np.ndarray:
    """Low-latency subset for one window; it intentionally does not fit ICA."""
    cfg = config or FasterConfig(run_ica=False)
    values = _signal_matrix(signal)
    bad_channels = detect_bad_channels(values, cfg)
    if bad_channels and cfg.interpolate_bad_channels:
        return interpolate_channels(values, bad_channels, config=cfg)
    return values.copy()

