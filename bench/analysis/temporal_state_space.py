"""Exploratory Hankel/SSA and bounded FNN feature diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def robust_trajectory(values: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Median/IQR scaling; no interpolation or deletion of missing time points."""
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        return None, "nonfinite_trajectory"
    x = x - np.median(x)
    scale = np.subtract(*np.percentile(x, [75, 25]))
    if scale <= 0:
        return None, "zero_iqr"
    return x / scale, "ok"


def ssa_energy(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hankel squared singular values, via the equivalent symmetric Gram matrix.

    Fixed L=min(60,N//2), N>=100. No reconstruction/forecasting is performed.
    """
    n = len(values)
    if n < 100:
        return np.array([]), {"status": "insufficient_sequence_length"}
    x, status = robust_trajectory(values)
    if x is None:
        return np.array([]), {"status": status}
    length = min(60, n // 2)
    hankel = np.lib.stride_tricks.sliding_window_view(x, length).T.copy()
    squared = np.maximum(np.linalg.eigvalsh(hankel @ hankel.T)[::-1], 0)
    energy = squared / squared.sum()
    positive = energy[energy > 0]
    concentration = float(np.sum(energy ** 2))
    cumulative = np.cumsum(energy)
    return energy, {
        "status": "ok", "n_windows": n, "trajectory_length": length,
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "participation_ratio": 1 / concentration,
        "spectral_concentration": concentration, "leading_energy": float(energy[0]),
        **{f"rank{q}": int(np.searchsorted(cumulative, q / 100) + 1) for q in (80, 90, 95)},
    }


def fnn_curve(values: np.ndarray, tau: int = 1) -> list[dict]:
    """Kennel-style ratio (10) OR attractor-size (2) tests, with Theiler exclusion.

    Length gate is a conservative diagnostic heuristic, not a theorem guaranteeing
    reliable dimension estimation. Zero-distance neighbours are reported/excluded.
    """
    if tau != 1:
        raise ValueError("Frozen diagnostic tau=1")
    x, status = robust_trajectory(values)
    rows = []
    for m in range(1, 9):
        count = len(values) - m * tau
        row = {"dimension": m, "n_vectors": max(count, 0), "theiler_windows": max(12, m * tau)}
        if count < max(500, 50 * (m + 1)):
            rows.append({**row, "status": "insufficient_sequence_length", "fnn_fraction": np.nan})
            continue
        if x is None:
            rows.append({**row, "status": status, "fnn_fraction": np.nan})
            continue
        embedding = np.column_stack([x[j * tau:j * tau + count] for j in range(m)])
        future = x[m * tau:m * tau + count]
        w = row["theiler_windows"]
        distances, indices = cKDTree(embedding).query(embedding, k=2 * w + 2, workers=1)
        allowed = np.abs(indices - np.arange(count)[:, None]) > w
        choice = allowed.argmax(axis=1)
        nearest = indices[np.arange(count), choice]
        distance = distances[np.arange(count), choice]
        valid = allowed.any(axis=1) & (distance > 1e-12)
        n_valid = int(valid.sum())
        row.update(n_valid_neighbors=n_valid, zero_distance_fraction=float(np.mean(distance <= 1e-12)))
        if n_valid < 500 or row["zero_distance_fraction"] > .1:
            rows.append({**row, "status": "unreliable_neighbors", "fnn_fraction": np.nan})
            continue
        difference = np.abs(future - future[nearest])[valid]
        false = (difference / distance[valid] > 10) | (
            np.hypot(distance[valid], difference) / np.std(x) > 2
        )
        rows.append({**row, "status": "ok", "fnn_fraction": float(false.mean())})
    return rows
