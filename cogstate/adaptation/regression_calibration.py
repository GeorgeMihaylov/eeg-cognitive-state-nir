"""Reusable deterministic calibration primitives for multi-output regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from cogstate.protocol import PM_METRICS


CANONICAL_PM_TARGETS = tuple(f"target_{metric}" for metric in PM_METRICS)


def _validated_regression_arrays(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError(
            "Regression arrays must have identical [samples, targets] shape, "
            f"got {truth.shape} and {prediction.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression arrays must be finite")
    return truth, prediction


def fit_bias_correction(y_true: Any, y_pred: Any) -> np.ndarray:
    """Fit one additive bias from calibration observations per target."""
    truth, prediction = _validated_regression_arrays(y_true, y_pred)
    if len(truth) == 0:
        raise ValueError("Bias correction needs at least one fit sample")
    return np.mean(truth - prediction, axis=0, dtype=np.float64)


def apply_bias_correction(y_pred: Any, bias: Any) -> np.ndarray:
    prediction = np.asarray(y_pred, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    if prediction.ndim != 2 or bias_array.shape != (prediction.shape[1],):
        raise ValueError(
            "Bias application expects [samples, targets] predictions and "
            f"one bias per target, got {prediction.shape} and {bias_array.shape}"
        )
    result = prediction + bias_array
    if not np.isfinite(result).all():
        raise ValueError("Bias correction produced non-finite predictions")
    return result


@dataclass(frozen=True)
class AffineCalibration:
    coefficients: np.ndarray
    intercepts: np.ndarray
    parameters: tuple[dict[str, Any], ...]


def fit_affine_calibration(
    y_true: Any,
    y_pred: Any,
    *,
    alpha: float = 1.0,
    variance_epsilon: float = 1e-12,
    target_names: Sequence[str] = CANONICAL_PM_TARGETS,
) -> AffineCalibration:
    """Fit independent Ridge mappings, with deterministic bias fallback."""
    truth, prediction = _validated_regression_arrays(y_true, y_pred)
    if alpha < 0 or variance_epsilon < 0:
        raise ValueError("alpha and variance_epsilon must be non-negative")
    if len(target_names) != truth.shape[1]:
        raise ValueError("target_names must match the regression output width")
    biases = fit_bias_correction(truth, prediction)
    coefficients: list[float] = []
    intercepts: list[float] = []
    parameters: list[dict[str, Any]] = []
    for index, target_name in enumerate(target_names):
        pred = prediction[:, index]
        target = truth[:, index]
        prediction_variance = float(np.var(pred))
        target_variance = float(np.var(target))
        fallback_reason: Optional[str] = None
        if len(pred) < 2:
            fallback_reason = "insufficient_samples"
        elif prediction_variance <= variance_epsilon:
            fallback_reason = "constant_prediction"
        elif target_variance <= variance_epsilon:
            fallback_reason = "constant_target"
        if fallback_reason is None:
            try:
                regressor = Ridge(alpha=float(alpha))
                regressor.fit(pred.reshape(-1, 1), target)
                coefficient = float(regressor.coef_[0])
                intercept = float(regressor.intercept_)
                if not np.isfinite([coefficient, intercept]).all():
                    fallback_reason = "non_finite_coefficients"
            except (ValueError, FloatingPointError):
                fallback_reason = "ridge_fit_failed"
        if fallback_reason is not None:
            coefficient = 1.0
            intercept = float(biases[index])
        coefficients.append(coefficient)
        intercepts.append(intercept)
        parameters.append({
            "target_name": str(target_name),
            "coefficient": coefficient,
            "intercept": intercept,
            "n_fit_samples": int(len(pred)),
            "prediction_variance": prediction_variance,
            "target_variance": target_variance,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
        })
    return AffineCalibration(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercepts=np.asarray(intercepts, dtype=np.float64),
        parameters=tuple(parameters),
    )


def apply_affine_calibration(
    y_pred: Any,
    calibration: AffineCalibration,
) -> np.ndarray:
    prediction = np.asarray(y_pred, dtype=np.float64)
    if prediction.ndim != 2:
        raise ValueError(
            f"Affine predictions must be two-dimensional, got {prediction.shape}"
        )
    if calibration.coefficients.shape != (prediction.shape[1],):
        raise ValueError("Affine coefficients do not match prediction width")
    result = (
        prediction * calibration.coefficients[None, :]
        + calibration.intercepts[None, :]
    )
    if not np.isfinite(result).all():
        raise ValueError("Affine calibration produced non-finite predictions")
    return result


__all__ = [
    "AffineCalibration",
    "CANONICAL_PM_TARGETS",
    "apply_affine_calibration",
    "apply_bias_correction",
    "fit_affine_calibration",
    "fit_bias_correction",
]
