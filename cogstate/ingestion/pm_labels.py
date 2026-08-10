"""Leakage-safe PM cleaning and three-level target encoding."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from cogstate.protocol import N_PM_CLASSES, PM_METRICS


@dataclass(frozen=True)
class PMCleaningConfig:
    mode: str = "none"  # none | causal_median | causal_exponential_smoothing
    median_window: int = 5
    exponential_alpha: float = 0.3
    valid_min: float = 0.0
    valid_max: float = 1.0

    def __post_init__(self):
        if self.mode not in {"none", "causal_median", "causal_exponential_smoothing"}:
            raise ValueError("Unknown PM cleaning mode")
        if self.median_window < 1 or not 0 < self.exponential_alpha <= 1:
            raise ValueError("Invalid smoothing parameters")


@dataclass
class PMCleaningResult:
    values: np.ndarray
    invalid_mask: np.ndarray
    anomaly_mask: np.ndarray


def clean_pm(values, config: PMCleaningConfig = PMCleaningConfig()) -> PMCleaningResult:
    """Clean one *record* causally; values outside [0, 1] become missing.

    The anomaly mask identifies robust Hampel-style deviations without silently
    discarding them.  Only explicit smoothing modes change valid observations.
    """
    data = np.asarray(values, dtype=float).copy()
    if data.ndim != 2 or data.shape[1] != len(PM_METRICS):
        raise ValueError(f"Expected PM array [windows, {len(PM_METRICS)}]")
    invalid = ~np.isfinite(data) | (data < config.valid_min) | (data > config.valid_max)
    data[invalid] = np.nan
    anomalies = np.zeros_like(invalid)
    output = data.copy()
    for column in range(data.shape[1]):
        history: list[float] = []
        ema = np.nan
        for row, value in enumerate(data[:, column]):
            previous = np.asarray(history[-config.median_window:], dtype=float)
            if len(previous):
                median = np.nanmedian(previous); mad = np.nanmedian(np.abs(previous - median))
                anomalies[row, column] = bool(np.isfinite(value) and mad > 0 and abs(value - median) > 3.5 * 1.4826 * mad)
            if not np.isfinite(value):
                continue
            if config.mode == "causal_median":
                output[row, column] = np.nanmedian(np.append(previous, value))
            elif config.mode == "causal_exponential_smoothing":
                ema = value if not np.isfinite(ema) else config.exponential_alpha * value + (1 - config.exponential_alpha) * ema
                output[row, column] = ema
            history.append(value)
    return PMCleaningResult(output, invalid, anomalies)


class TertileDiscretizer:
    """Fit q1/3 and q2/3 on external-fold training targets only."""
    def __init__(self): self.thresholds_: np.ndarray | None = None
    def fit(self, pm_train):
        values = np.asarray(pm_train, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(PM_METRICS): raise ValueError("Expected PM matrix with seven columns")
        self.thresholds_ = np.nanquantile(values, [1 / 3, 2 / 3], axis=0).T
        if not np.isfinite(self.thresholds_).all(): raise ValueError("Each PM target needs finite training observations")
        return self
    def transform(self, pm):
        if self.thresholds_ is None: raise RuntimeError("Call fit on the training fold first")
        values = np.asarray(pm, dtype=float)
        labels = np.full(values.shape, -1, dtype=np.int8)
        valid = np.isfinite(values)
        classified = (values > self.thresholds_[None, :, 0]).astype(np.int8) + (values > self.thresholds_[None, :, 1]).astype(np.int8)
        labels[valid] = classified[valid]
        return labels
    def fit_transform(self, pm_train): return self.fit(pm_train).transform(pm_train)


def pm_target_columns(prefix: str = "target_"):
    return tuple(f"{prefix}{name}" for name in PM_METRICS)
