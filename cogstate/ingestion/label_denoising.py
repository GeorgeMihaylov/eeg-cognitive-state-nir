from __future__ import annotations
import numpy as np


def denoise_labels(labels: np.ndarray, window: int = 5, z_threshold: float = 3.5) -> np.ndarray:
    """Replace robust outliers and apply a centred moving-median smoother."""
    values = np.asarray(labels, dtype=float).reshape(-1)
    if not len(values) or window < 1:
        raise ValueError("labels must be non-empty and window must be positive")
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    cleaned = values.copy()
    if mad > 0:
        cleaned[np.abs(0.6745 * (values - median) / mad) > z_threshold] = median
    pad = window // 2
    padded = np.pad(cleaned, (pad, pad), mode="edge")
    return np.array([np.nanmedian(padded[i:i + window]) for i in range(len(cleaned))])
