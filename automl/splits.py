from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InnerSplit:
    inner_train_X: np.ndarray
    inner_train_y: np.ndarray
    inner_val_X: np.ndarray
    inner_val_y: np.ndarray


def build_inner_split(
    calibration_X: np.ndarray,
    calibration_y: np.ndarray,
    inner_train_frac: float = 0.75,
    min_samples: int = 4,
) -> InnerSplit:
    """Chronological split: an early slice for adaptation, the rest for scoring."""
    if not 0.0 < inner_train_frac < 1.0:
        raise ValueError("inner_train_frac must be in (0, 1)")
    n = calibration_X.shape[0]
    if n < min_samples:
        raise ValueError(
            f"Need at least {min_samples} calibration samples for a leakage-safe "
            "inner split; fall back to a zero-min-data candidate and skip "
            "inner search for this subject."
        )
    cut = max(1, min(n - 1, int(round(n * inner_train_frac))))
    return InnerSplit(
        inner_train_X=calibration_X[:cut],
        inner_train_y=calibration_y[:cut],
        inner_val_X=calibration_X[cut:],
        inner_val_y=calibration_y[cut:],
    )
