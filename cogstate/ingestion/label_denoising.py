"""Causal and advanced denoising methods for continuous PM target series.

The advanced implementations are deliberately record-local and preserve
missing values.  They estimate denoised *continuous* targets; discretization,
if needed, must be fitted afterwards on the training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from cogstate.protocol import PM_METRICS


def denoise_labels(
    labels: np.ndarray,
    *,
    mode: str = "none",
    window: int = 5,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply simple causal smoothing to one PM series.

    This compatibility helper keeps source values for ``mode="none"`` and does
    not perform outlier replacement.  Use :func:`robust_kalman_pm`,
    :func:`huber_trend_pm`, or the audited helpers in ``pm_labels`` for the full
    PM matrix workflow.
    """
    values = np.asarray(labels, dtype=float).reshape(-1)
    if not len(values) or window < 1 or not 0 < alpha <= 1:
        raise ValueError("Invalid label smoothing parameters")
    if mode == "none":
        return values.copy()

    output = values.copy()
    if mode == "causal_median":
        history: list[float] = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                history.append(float(value))
                output[index] = np.median(history[-window:])
            else:
                history.clear()
        return output
    if mode == "causal_exponential_smoothing":
        state = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value):
                state = (
                    value
                    if not np.isfinite(state)
                    else alpha * value + (1.0 - alpha) * state
                )
                output[index] = state
            else:
                state = np.nan
        return output
    raise ValueError(
        "mode must be 'none', 'causal_median', or "
        "'causal_exponential_smoothing'"
    )


@dataclass(frozen=True)
class RobustKalmanConfig:
    """Student-t reweighted random-walk Kalman/RTS smoother parameters."""

    process_variance: float = 0.001
    observation_variance: float = 0.01
    degrees_of_freedom: float = 4.0
    max_iterations: int = 30
    tolerance: float = 1e-5
    outlier_weight_threshold: float = 0.5
    valid_min: float = 0.0
    valid_max: float = 1.0
    clip_output: bool = True

    def __post_init__(self) -> None:
        if self.process_variance <= 0 or self.observation_variance <= 0:
            raise ValueError("Kalman variances must be positive")
        if self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive")
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError("Invalid convergence parameters")
        if self.outlier_weight_threshold <= 0:
            raise ValueError("outlier_weight_threshold must be positive")
        if self.valid_min >= self.valid_max:
            raise ValueError("valid_min must be smaller than valid_max")


@dataclass(frozen=True)
class HuberTrendConfig:
    """Huber loss with sparse first/second-difference trend regularization."""

    huber_delta: float = 0.05
    first_order_penalty: float = 0.05
    second_order_penalty: float = 0.5
    admm_rho: float = 1.0
    max_iterations: int = 500
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-4
    valid_min: float = 0.0
    valid_max: float = 1.0
    clip_output: bool = True

    def __post_init__(self) -> None:
        if self.huber_delta <= 0 or self.admm_rho <= 0:
            raise ValueError("Huber delta and ADMM rho must be positive")
        if self.first_order_penalty < 0 or self.second_order_penalty < 0:
            raise ValueError("Trend penalties cannot be negative")
        if self.first_order_penalty == 0 and self.second_order_penalty == 0:
            raise ValueError("At least one trend penalty must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.absolute_tolerance <= 0 or self.relative_tolerance <= 0:
            raise ValueError("ADMM tolerances must be positive")
        if self.valid_min >= self.valid_max:
            raise ValueError("valid_min must be smaller than valid_max")


@dataclass(frozen=True)
class AdvancedPMCleaningResult:
    """Denoised matrix and method-independent quality diagnostics."""

    values: np.ndarray
    invalid_mask: np.ndarray
    anomaly_mask: np.ndarray
    confidence: np.ndarray
    converged: bool
    iterations: int
    method: str

    def summary(self) -> dict[str, int | bool | str]:
        return {
            "method": self.method,
            "observations": int(self.values.size),
            "invalid": int(self.invalid_mask.sum()),
            "anomalies": int(self.anomaly_mask.sum()),
            "retained": int(np.isfinite(self.values).sum()),
            "converged": self.converged,
            "iterations": self.iterations,
        }


def _validate_matrix(values: object, valid_min: float, valid_max: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(PM_METRICS):
        raise ValueError(
            f"Expected PM array [samples, {len(PM_METRICS)}], got {data.shape}"
        )
    data = data.copy()
    invalid = ~np.isfinite(data) | (data < valid_min) | (data > valid_max)
    data[invalid] = np.nan
    return data, invalid


def _finite_runs(values: np.ndarray) -> Iterable[slice]:
    """Yield maximal finite slices so smoothing never crosses a missing gap."""
    finite = np.isfinite(values)
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def _kalman_rts(y: np.ndarray, q: float, observation_variances: np.ndarray) -> np.ndarray:
    """Scalar random-walk Kalman filter followed by an RTS backward pass."""
    n = len(y)
    filtered_mean = np.empty(n, dtype=float)
    filtered_variance = np.empty(n, dtype=float)
    predicted_variance = np.empty(n, dtype=float)

    mean = float(y[0])
    variance = float(observation_variances[0])
    for index in range(n):
        if index:
            variance += q
        predicted_variance[index] = variance
        gain = variance / (variance + observation_variances[index])
        mean += gain * (y[index] - mean)
        variance *= 1.0 - gain
        filtered_mean[index] = mean
        filtered_variance[index] = max(variance, np.finfo(float).eps)

    smoothed = filtered_mean.copy()
    for index in range(n - 2, -1, -1):
        denominator = filtered_variance[index] + q
        gain = filtered_variance[index] / denominator
        smoothed[index] += gain * (smoothed[index + 1] - filtered_mean[index])
    return smoothed


def _student_t_kalman_run(y: np.ndarray, config: RobustKalmanConfig) -> tuple[np.ndarray, np.ndarray, bool, int]:
    if len(y) == 1:
        return y.copy(), np.ones(1), True, 1

    weights = np.ones(len(y), dtype=float)
    estimate = _kalman_rts(
        y,
        config.process_variance,
        np.full(len(y), config.observation_variance),
    )
    converged = False
    for iteration in range(1, config.max_iterations + 1):
        residual_scale = (y - estimate) ** 2 / config.observation_variance
        new_weights = (config.degrees_of_freedom + 1.0) / (
            config.degrees_of_freedom + residual_scale
        )
        effective_variance = config.observation_variance / np.maximum(
            new_weights, np.finfo(float).eps
        )
        updated = _kalman_rts(y, config.process_variance, effective_variance)
        change = np.max(np.abs(updated - estimate))
        scale = 1.0 + np.max(np.abs(estimate))
        estimate, weights = updated, new_weights
        if change <= config.tolerance * scale:
            converged = True
            break
    return estimate, weights, converged, iteration


def robust_kalman_pm(
    values: object, config: RobustKalmanConfig = RobustKalmanConfig()
) -> AdvancedPMCleaningResult:
    """Estimate PM levels with Student-t observation noise and RTS smoothing."""
    data, invalid = _validate_matrix(values, config.valid_min, config.valid_max)
    output = data.copy()
    confidence = np.full(data.shape, np.nan)
    anomalies = np.zeros(data.shape, dtype=bool)
    all_converged = True
    max_iterations = 0

    for column in range(data.shape[1]):
        for run in _finite_runs(data[:, column]):
            estimate, weights, converged, iterations = _student_t_kalman_run(
                data[run, column], config
            )
            output[run, column] = estimate
            confidence[run, column] = np.minimum(weights, 1.0)
            anomalies[run, column] = weights < config.outlier_weight_threshold
            all_converged &= converged
            max_iterations = max(max_iterations, iterations)

    if config.clip_output:
        output = np.clip(output, config.valid_min, config.valid_max)
    return AdvancedPMCleaningResult(
        output,
        invalid,
        anomalies,
        confidence,
        all_converged,
        max_iterations,
        "robust_kalman",
    )


def _difference_matrix(length: int, order: int) -> sparse.csc_matrix:
    if order == 1:
        return sparse.diags([-np.ones(length - 1), np.ones(length - 1)], [0, 1], shape=(length - 1, length), format="csc")
    if order == 2:
        return sparse.diags(
            [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
            format="csc",
        )
    raise ValueError("Only first and second differences are supported")


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _huber_prox(values: np.ndarray, delta: float, scale: float) -> np.ndarray:
    boundary = delta * (1.0 + scale)
    quadratic = np.abs(values) <= boundary
    output = values - scale * delta * np.sign(values)
    output[quadratic] = values[quadratic] / (1.0 + scale)
    return output


def _huber_trend_run(y: np.ndarray, config: HuberTrendConfig) -> tuple[np.ndarray, bool, int]:
    n = len(y)
    if n == 1:
        return y.copy(), True, 1

    d1 = _difference_matrix(n, 1) if config.first_order_penalty > 0 else sparse.csc_matrix((0, n))
    d2 = _difference_matrix(n, 2) if n > 2 and config.second_order_penalty > 0 else sparse.csc_matrix((0, n))
    identity = sparse.eye(n, format="csc")
    system = identity + d1.T @ d1 + d2.T @ d2
    solve = factorized(system.tocsc())

    x = y.copy()
    residual = np.zeros(n)
    z1, z2 = d1 @ x, d2 @ x
    u0 = np.zeros(n)
    u1, u2 = np.zeros(d1.shape[0]), np.zeros(d2.shape[0])
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        rhs = y - residual + u0
        if d1.shape[0]:
            rhs += d1.T @ (z1 - u1)
        if d2.shape[0]:
            rhs += d2.T @ (z2 - u2)
        x = solve(rhs)

        previous_residual = residual.copy()
        previous_z1, previous_z2 = z1.copy(), z2.copy()
        residual = _huber_prox(
            y - x + u0, config.huber_delta, 1.0 / config.admm_rho
        )
        if d1.shape[0]:
            d1x = d1 @ x
            z1 = _soft_threshold(
                d1x + u1, config.first_order_penalty / config.admm_rho
            )
            u1 += d1x - z1
        if d2.shape[0]:
            d2x = d2 @ x
            z2 = _soft_threshold(
                d2x + u2, config.second_order_penalty / config.admm_rho
            )
            u2 += d2x - z2
        u0 += y - x - residual

        primal_sq = np.sum((y - x - residual) ** 2)
        dual_sq = np.sum((residual - previous_residual) ** 2)
        constraint_count = n
        if d1.shape[0]:
            primal_sq += np.sum((d1 @ x - z1) ** 2)
            dual_sq += np.sum((d1.T @ (z1 - previous_z1)) ** 2)
            constraint_count += d1.shape[0]
        if d2.shape[0]:
            primal_sq += np.sum((d2 @ x - z2) ** 2)
            dual_sq += np.sum((d2.T @ (z2 - previous_z2)) ** 2)
            constraint_count += d2.shape[0]
        primal = np.sqrt(primal_sq)
        dual = config.admm_rho * np.sqrt(dual_sq)
        reference = max(np.linalg.norm(y), np.linalg.norm(x), 1.0)
        epsilon = (
            np.sqrt(constraint_count) * config.absolute_tolerance
            + config.relative_tolerance * reference
        )
        if primal <= epsilon and dual <= epsilon:
            converged = True
            break
    return x, converged, iteration


def huber_trend_pm(
    values: object, config: HuberTrendConfig = HuberTrendConfig()
) -> AdvancedPMCleaningResult:
    """Apply robust piecewise-smooth Huber/L1 trend filtering to PM values."""
    data, invalid = _validate_matrix(values, config.valid_min, config.valid_max)
    output = data.copy()
    anomalies = np.zeros(data.shape, dtype=bool)
    confidence = np.full(data.shape, np.nan)
    all_converged = True
    max_iterations = 0

    for column in range(data.shape[1]):
        for run in _finite_runs(data[:, column]):
            estimate, converged, iterations = _huber_trend_run(
                data[run, column], config
            )
            residual = np.abs(data[run, column] - estimate)
            output[run, column] = estimate
            anomalies[run, column] = residual > config.huber_delta
            confidence[run, column] = np.minimum(
                1.0, config.huber_delta / np.maximum(residual, config.huber_delta)
            )
            all_converged &= converged
            max_iterations = max(max_iterations, iterations)

    if config.clip_output:
        output = np.clip(output, config.valid_min, config.valid_max)
    return AdvancedPMCleaningResult(
        output,
        invalid,
        anomalies,
        confidence,
        all_converged,
        max_iterations,
        "huber_trend",
    )


AdvancedMethod = Literal["robust_kalman", "huber_trend"]


def denoise_pm_by_record(
    values: object,
    record_ids: Iterable[Hashable],
    *,
    method: AdvancedMethod,
    config: RobustKalmanConfig | HuberTrendConfig | None = None,
) -> AdvancedPMCleaningResult:
    """Run an advanced method without sharing state across records."""
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(PM_METRICS):
        raise ValueError(
            f"Expected PM array [samples, {len(PM_METRICS)}], got {raw.shape}"
        )
    records = np.asarray(list(record_ids), dtype=object)
    if records.ndim != 1 or len(records) != len(raw):
        raise ValueError("record_ids must contain one identifier per PM sample")
    if method == "robust_kalman":
        selected_config = config or RobustKalmanConfig()
        if not isinstance(selected_config, RobustKalmanConfig):
            raise TypeError("robust_kalman requires RobustKalmanConfig")
        cleaner = robust_kalman_pm
    elif method == "huber_trend":
        selected_config = config or HuberTrendConfig()
        if not isinstance(selected_config, HuberTrendConfig):
            raise TypeError("huber_trend requires HuberTrendConfig")
        cleaner = huber_trend_pm
    else:
        raise ValueError("method must be 'robust_kalman' or 'huber_trend'")

    output = np.full_like(raw, np.nan)
    invalid = np.zeros(raw.shape, dtype=bool)
    anomalies = np.zeros(raw.shape, dtype=bool)
    confidence = np.full(raw.shape, np.nan)
    all_converged = True
    max_iterations = 0
    for record_id in dict.fromkeys(records.tolist()):
        mask = records == record_id
        result = cleaner(raw[mask], selected_config)
        output[mask] = result.values
        invalid[mask] = result.invalid_mask
        anomalies[mask] = result.anomaly_mask
        confidence[mask] = result.confidence
        all_converged &= result.converged
        max_iterations = max(max_iterations, result.iterations)
    return AdvancedPMCleaningResult(
        output,
        invalid,
        anomalies,
        confidence,
        all_converged,
        max_iterations,
        method,
    )
