"""Offline EEG re-referencing, including a bad-channel-resistant estimate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .artifact_removal import FasterConfig, detect_bad_channels


ReferenceMethod = Literal["none", "common_average", "median", "robust_average"]


@dataclass(frozen=True)
class ReferenceReport:
    method: str
    excluded_channels: tuple[int, ...] = ()
    iterations: int = 0


def _as_signal(signal: object) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("EEG signal must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError("EEG signal contains non-finite values")
    return values


def common_average_reference(
    signal: object, *, exclude: Iterable[int] = ()
) -> np.ndarray:
    """Subtract the per-sample mean of all non-excluded EEG channels."""
    values = _as_signal(signal)
    excluded = {int(index) for index in exclude}
    if any(index < 0 or index >= values.shape[1] for index in excluded):
        raise ValueError("Excluded channel index is out of range")
    good = [index for index in range(values.shape[1]) if index not in excluded]
    if not good:
        raise ValueError("At least one channel is required to construct a reference")
    reference = np.mean(values[:, good], axis=1, keepdims=True)
    return values - reference


def median_reference(signal: object) -> np.ndarray:
    """Subtract the channel-wise median at each sample."""
    values = _as_signal(signal)
    return values - np.median(values, axis=1, keepdims=True)


def robust_average_reference(
    signal: object,
    *,
    z_threshold: float = 3.0,
    max_iterations: int = 5,
) -> tuple[np.ndarray, ReferenceReport]:
    """Iteratively estimate average reference after excluding bad channels.

    The initial median reference prevents one extreme channel from contaminating
    the first bad-channel estimate.  Detection is then repeated against the
    average of the currently accepted channels until the set stabilizes.
    """
    values = _as_signal(signal)
    if z_threshold <= 0 or max_iterations < 1:
        raise ValueError("Invalid robust-reference parameters")
    config = FasterConfig(z_threshold=z_threshold)
    excluded: set[int] = set()

    for iteration in range(1, max_iterations + 1):
        good = [index for index in range(values.shape[1]) if index not in excluded]
        robust_reference = np.median(values[:, good], axis=1, keepdims=True)
        referenced = values - robust_reference
        detected = set(detect_bad_channels(referenced, config))
        updated = excluded | detected
        if len(updated) >= values.shape[1]:
            # Never construct a reference from zero channels.
            break
        if updated == excluded:
            return common_average_reference(values, exclude=excluded), ReferenceReport(
                "robust_average", tuple(int(index) for index in sorted(excluded)), iteration
            )
        excluded = updated

    referenced = common_average_reference(values, exclude=excluded)
    return referenced, ReferenceReport(
        "robust_average",
        tuple(int(index) for index in sorted(excluded)),
        max_iterations,
    )


def rereference(
    signal: object,
    *,
    method: ReferenceMethod,
    robust_z_threshold: float = 3.0,
    robust_max_iterations: int = 5,
) -> tuple[np.ndarray, ReferenceReport]:
    values = _as_signal(signal)
    if method == "none":
        return values.copy(), ReferenceReport("none")
    if method == "common_average":
        return common_average_reference(values), ReferenceReport("common_average")
    if method == "median":
        return median_reference(values), ReferenceReport("median")
    if method == "robust_average":
        return robust_average_reference(
            values,
            z_threshold=robust_z_threshold,
            max_iterations=robust_max_iterations,
        )
    raise ValueError(f"Unknown reference method: {method!r}")
