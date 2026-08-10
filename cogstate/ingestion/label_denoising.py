from __future__ import annotations
import numpy as np


def denoise_labels(labels: np.ndarray, *, mode: str = "none", window: int = 5, alpha: float = 0.3) -> np.ndarray:
    """Optional causal smoothing for a single PM series.

    ``none`` is deliberately the default canonical baseline.  This helper does
    not clip values or remove outliers: callers retain the source values and
    should store anomaly flags from :func:`clean_pm` separately.
    """
    values = np.asarray(labels, dtype=float).reshape(-1)
    if not len(values) or window < 1 or not 0 < alpha <= 1:
        raise ValueError("Invalid label smoothing parameters")
    if mode == "none": return values.copy()
    output = values.copy()
    if mode == "causal_median":
        for index, value in enumerate(values):
            if np.isfinite(value): output[index] = np.nanmedian(values[max(0, index - window + 1):index + 1])
        return output
    if mode == "causal_exponential_smoothing":
        state = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value):
                state = value if not np.isfinite(state) else alpha * value + (1 - alpha) * state
                output[index] = state
        return output
    raise ValueError("mode must be 'none', 'causal_median', or 'causal_exponential_smoothing'")
