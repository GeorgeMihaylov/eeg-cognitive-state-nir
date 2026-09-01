"""Offline multichannel linear EOG regression."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EOGRegressionReport:
    eog_channels: int
    mean_absolute_correlation_before: float
    mean_absolute_correlation_after: float


class EOGRegression:
    """Estimate and subtract linear EOG propagation into every EEG channel."""

    def __init__(self, *, ridge_alpha: float = 1e-6) -> None:
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("ridge_alpha cannot be negative")
        self.ridge_alpha = float(ridge_alpha)
        self.eog_mean_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None

    @staticmethod
    def _eog_matrix(eog: object, expected_samples: int) -> np.ndarray:
        values = np.asarray(eog, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != expected_samples:
            raise ValueError("EOG must match EEG samples and be one- or two-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("EOG contains non-finite values")
        return values

    def fit(self, eeg: object, eog: object) -> "EOGRegression":
        targets = np.asarray(eeg, dtype=float)
        if targets.ndim != 2 or not len(targets) or not np.isfinite(targets).all():
            raise ValueError("EEG must be a finite [samples, channels] matrix")
        regressors = self._eog_matrix(eog, len(targets))
        self.eog_mean_ = np.mean(regressors, axis=0)
        centered = regressors - self.eog_mean_
        gram = centered.T @ centered
        regularized = gram + self.ridge_alpha * np.eye(gram.shape[0])
        right_hand_side = centered.T @ targets
        self.coefficients_ = (
            np.linalg.solve(regularized, right_hand_side)
            if self.ridge_alpha > 0
            else np.linalg.pinv(gram) @ right_hand_side
        )
        return self

    def transform(self, eeg: object, eog: object) -> np.ndarray:
        if self.eog_mean_ is None or self.coefficients_ is None:
            raise RuntimeError("Call fit before transform")
        targets = np.asarray(eeg, dtype=float)
        if targets.ndim != 2 or not len(targets) or not targets.shape[1]:
            raise ValueError("EEG must be a finite [samples, channels] matrix")
        if not np.isfinite(targets).all():
            raise ValueError("EEG must be a finite [samples, channels] matrix")
        regressors = self._eog_matrix(eog, len(targets))
        if regressors.shape[1] != len(self.eog_mean_):
            raise ValueError("EOG channel count differs from fitted data")
        if targets.shape[1] != self.coefficients_.shape[1]:
            raise ValueError("EEG channel count differs from fitted data")
        return targets - (regressors - self.eog_mean_) @ self.coefficients_

    def fit_transform(self, eeg: object, eog: object) -> np.ndarray:
        return self.fit(eeg, eog).transform(eeg, eog)


def _mean_absolute_correlation(eeg: np.ndarray, eog: np.ndarray) -> float:
    centered_eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
    centered_eog = eog - np.mean(eog, axis=0, keepdims=True)
    numerator = centered_eeg.T @ centered_eog
    denominator = np.sqrt(
        np.sum(centered_eeg**2, axis=0)[:, None]
        * np.sum(centered_eog**2, axis=0)[None, :]
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )
    return float(np.mean(np.abs(correlations)))


def regress_eog(
    eeg: object,
    eog: object,
    *,
    ridge_alpha: float = 1e-6,
) -> tuple[np.ndarray, EOGRegression, EOGRegressionReport]:
    targets = np.asarray(eeg, dtype=float)
    regressors = EOGRegression._eog_matrix(eog, len(targets))
    model = EOGRegression(ridge_alpha=ridge_alpha)
    cleaned = model.fit_transform(targets, regressors)
    report = EOGRegressionReport(
        eog_channels=regressors.shape[1],
        mean_absolute_correlation_before=_mean_absolute_correlation(
            targets, regressors
        ),
        mean_absolute_correlation_after=_mean_absolute_correlation(
            cleaned, regressors
        ),
    )
    return cleaned, model, report
