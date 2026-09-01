from __future__ import annotations
import numpy as np


def align_pm_with_behavior(pm_timestamps, pm_values, behavior_timestamps, behavior_values, *, max_gap_s: float | None = None):
    """Nearest-time join of PM metrics and behavioural measurements."""
    pt, pv = np.asarray(pm_timestamps, float), np.asarray(pm_values, float)
    bt, bv = np.asarray(behavior_timestamps, float), np.asarray(behavior_values, float)
    if len(pt) != len(pv) or len(bt) != len(bv):
        raise ValueError("timestamps and values must have matching lengths")
    order = np.argsort(bt); bt, bv = bt[order], bv[order]
    index = np.searchsorted(bt, pt).clip(0, max(len(bt) - 1, 0))
    left = np.maximum(index - 1, 0)
    index = np.where(np.abs(bt[index] - pt) < np.abs(bt[left] - pt), index, left) if len(bt) else index
    matched = bv[index] if len(bt) else np.full(len(pt), np.nan)
    if max_gap_s is not None and len(bt): matched[np.abs(bt[index] - pt) > max_gap_s] = np.nan
    return np.column_stack((pt, pv, matched))
