"""Robust, record-local cleaning and encoding of PM target time series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

from cogstate.protocol import PM_METRICS


_SMOOTHING_MODES = {"none", "causal_median", "causal_exponential_smoothing"}
_OUTLIER_POLICIES = {"flag", "nan", "local_median"}


@dataclass(frozen=True)
class PMCleaningConfig:
    """Configuration for leakage-safe cleaning of one physical recording.

    PM values are assumed to be the scaled manufacturer outputs in ``[0, 1]``.
    Detection and smoothing are causal: only earlier values in the same record
    are used.  Defaults preserve every valid source value and only add masks.
    """

    mode: str = "none"
    median_window: int = 5
    exponential_alpha: float = 0.3
    valid_min: float = 0.0
    valid_max: float = 1.0
    outlier_window: int = 7
    outlier_threshold: float = 3.5
    min_outlier_history: int = 3
    min_absolute_deviation: float = 0.05
    outlier_policy: str = "flag"
    warmup_samples: int = 0
    reset_smoothing_on_gap: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _SMOOTHING_MODES:
            raise ValueError(f"mode must be one of {sorted(_SMOOTHING_MODES)}")
        if self.outlier_policy not in _OUTLIER_POLICIES:
            raise ValueError(
                f"outlier_policy must be one of {sorted(_OUTLIER_POLICIES)}"
            )
        if self.median_window < 1 or self.outlier_window < 1:
            raise ValueError("Window sizes must be positive")
        if not 0 < self.exponential_alpha <= 1:
            raise ValueError("exponential_alpha must be in (0, 1]")
        if self.valid_min >= self.valid_max:
            raise ValueError("valid_min must be smaller than valid_max")
        if self.outlier_threshold <= 0 or self.min_absolute_deviation < 0:
            raise ValueError("Outlier thresholds must be non-negative")
        if not 1 <= self.min_outlier_history <= self.outlier_window:
            raise ValueError("min_outlier_history must be within outlier_window")
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples cannot be negative")


@dataclass(frozen=True)
class PMCleaningResult:
    """Cleaned values plus audit information; masks match ``values.shape``."""

    values: np.ndarray
    invalid_mask: np.ndarray
    anomaly_mask: np.ndarray

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.values)

    def summary(self) -> dict[str, int]:
        """Return compact counts suitable for a data-quality report."""
        return {
            "observations": int(self.values.size),
            "invalid": int(self.invalid_mask.sum()),
            "anomalies": int(self.anomaly_mask.sum()),
            "retained": int(np.isfinite(self.values).sum()),
        }


def _as_pm_matrix(values: object) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(PM_METRICS):
        raise ValueError(
            f"Expected PM array [samples, {len(PM_METRICS)}], got {data.shape}"
        )
    return data.copy()


def _hampel_is_outlier(value: float, history: list[float], config: PMCleaningConfig) -> tuple[bool, float]:
    reference = np.asarray(history[-config.outlier_window :], dtype=float)
    if len(reference) < config.min_outlier_history:
        return False, np.nan
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    # 1.4826 makes MAD consistent with standard deviation for Gaussian noise.
    robust_limit = config.outlier_threshold * 1.4826 * mad
    limit = max(robust_limit, config.min_absolute_deviation)
    return abs(value - median) > limit, median


def _clean_column(values: np.ndarray, config: PMCleaningConfig) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    anomalies = np.zeros(len(values), dtype=bool)
    detector_history: list[float] = []
    smoothing_history: list[float] = []
    ema = np.nan

    for index, source_value in enumerate(values):
        if not np.isfinite(source_value):
            if config.reset_smoothing_on_gap:
                detector_history.clear()
                smoothing_history.clear()
                ema = np.nan
            continue

        is_outlier = False
        local_median = np.nan
        if index >= config.warmup_samples:
            is_outlier, local_median = _hampel_is_outlier(
                float(source_value), detector_history, config
            )
        anomalies[index] = is_outlier

        cleaned_value = float(source_value)
        if is_outlier and config.outlier_policy == "nan":
            output[index] = np.nan
            # Do not let a rejected impulse contaminate later detection.
            continue
        if is_outlier and config.outlier_policy == "local_median":
            cleaned_value = local_median

        detector_history.append(cleaned_value)
        if config.mode == "causal_median":
            smoothing_history.append(cleaned_value)
            output[index] = np.median(smoothing_history[-config.median_window :])
        elif config.mode == "causal_exponential_smoothing":
            ema = (
                cleaned_value
                if not np.isfinite(ema)
                else config.exponential_alpha * cleaned_value
                + (1.0 - config.exponential_alpha) * ema
            )
            output[index] = ema
        else:
            output[index] = cleaned_value

    return output, anomalies


def clean_pm(
    values: object, config: PMCleaningConfig = PMCleaningConfig()
) -> PMCleaningResult:
    """Clean a PM matrix belonging to exactly one physical recording.

    Invalid or inactive-detector values must be represented as ``NaN`` and are
    never imputed.  Outliers are always reported; replacement is opt-in through
    ``outlier_policy``.  Call :func:`clean_pm_by_record` when multiple records
    are stored in one matrix.
    """
    data = _as_pm_matrix(values)
    invalid = (
        ~np.isfinite(data)
        | (data < config.valid_min)
        | (data > config.valid_max)
    )
    data[invalid] = np.nan
    output = data.copy()
    anomalies = np.zeros_like(invalid)

    for column in range(data.shape[1]):
        output[:, column], anomalies[:, column] = _clean_column(
            data[:, column], config
        )
    return PMCleaningResult(output, invalid, anomalies)


def clean_pm_by_record(
    values: object,
    record_ids: Iterable[Hashable],
    config: PMCleaningConfig = PMCleaningConfig(),
) -> PMCleaningResult:
    """Clean a matrix without sharing history across recording boundaries."""
    data = _as_pm_matrix(values)
    records = np.asarray(list(record_ids), dtype=object)
    if records.ndim != 1 or len(records) != len(data):
        raise ValueError("record_ids must contain one identifier per PM sample")

    output = np.full_like(data, np.nan)
    invalid = np.zeros(data.shape, dtype=bool)
    anomalies = np.zeros(data.shape, dtype=bool)
    # Stable first-seen order, also supporting mixed hashable identifier types.
    ordered_ids = list(dict.fromkeys(records.tolist()))
    for record_id in ordered_ids:
        mask = records == record_id
        result = clean_pm(data[mask], config)
        output[mask] = result.values
        invalid[mask] = result.invalid_mask
        anomalies[mask] = result.anomaly_mask
    return PMCleaningResult(output, invalid, anomalies)


class TertileDiscretizer:
    """Fit 1/3 and 2/3 quantiles on external-fold training targets only."""

    def __init__(self) -> None:
        self.thresholds_: np.ndarray | None = None

    def fit(self, pm_train: object) -> "TertileDiscretizer":
        values = _as_pm_matrix(pm_train)
        self.thresholds_ = np.nanquantile(values, [1 / 3, 2 / 3], axis=0).T
        if not np.isfinite(self.thresholds_).all():
            raise ValueError("Each PM target needs finite training observations")
        return self

    def transform(self, pm: object) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Call fit on the training fold first")
        values = _as_pm_matrix(pm)
        labels = np.full(values.shape, -1, dtype=np.int8)
        valid = np.isfinite(values)
        classified = (
            (values > self.thresholds_[None, :, 0]).astype(np.int8)
            + (values > self.thresholds_[None, :, 1]).astype(np.int8)
        )
        labels[valid] = classified[valid]
        return labels

    def fit_transform(self, pm_train: object) -> np.ndarray:
        return self.fit(pm_train).transform(pm_train)


def pm_target_columns(prefix: str = "target_") -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in PM_METRICS)
