"""Target-free diagnostics on exact-contiguous engineered EEG trajectories."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contiguous_segments(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return original row indices; reject mixed identities and invalid bounds."""
    for key in ("record_id", "subject_id", "outer_fold"):
        if key in frame and (frame[key].isna().any() or frame[key].nunique() != 1):
            raise ValueError(f"Segment input crosses {key}")
    if not frame.index.is_unique:
        raise ValueError("Duplicate row index")
    ordered = frame.sort_values("t_start", kind="stable")
    t = ordered.t_start.to_numpy(dtype=float)
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Invalid or duplicate t_start")
    if "t_end" in ordered and not np.allclose(
        ordered.t_end.to_numpy(dtype=float) - t, 10, atol=1e-6, rtol=0
    ):
        raise ValueError("Expected ten-second windows")
    cuts = np.flatnonzero(~np.isclose(np.diff(t), 10, atol=1e-6, rtol=0)) + 1
    return [a for a in np.split(ordered.index.to_numpy(dtype=np.int64), cuts) if len(a)]


def permute_finite_columns(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle finite values within each trajectory, keeping missing positions fixed."""
    if np.isfinite(matrix).all():
        return matrix[rng.permutation(len(matrix))].copy()
    shuffled = matrix.copy()
    for j in range(matrix.shape[1]):
        valid = np.flatnonzero(np.isfinite(matrix[:, j]))
        shuffled[valid, j] = matrix[rng.permutation(valid), j]
    return shuffled

