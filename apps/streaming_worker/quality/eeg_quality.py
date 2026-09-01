"""Window-level quality gate for EEG streaming."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cogstate.streaming.buffer import Window

from ..config import QualityConfig


@dataclass(frozen=True)
class QualityReport:
    status: str
    valid: bool
    reasons: tuple[str, ...]
    sample_count: int
    expected_sample_count: int
    finite_ratio: float
    estimated_sample_rate: float | None
    missing_ratio: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EEGQualityGate:
    def __init__(
        self,
        *,
        sample_rate: float,
        n_channels: int,
        config: QualityConfig,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self.config = config

    def evaluate(self, window: Window) -> QualityReport:
        values = np.asarray(window.data.get("eeg", np.empty((0, 0))))
        timestamps = np.asarray(window.timestamps.get("eeg", np.empty(0)), dtype=float)
        expected = int(round((window.end_time - window.start_time) * self.sample_rate))
        count = len(values)
        reasons: list[str] = []

        if values.ndim != 2 or values.shape[1] != self.n_channels:
            reasons.append("channel_count_mismatch")
        finite_ratio = float(np.isfinite(values).mean()) if values.size else 0.0
        if finite_ratio < self.config.minimum_finite_ratio:
            reasons.append("non_finite_values")
        missing_ratio = abs(count - expected) / max(expected, 1)
        if missing_ratio > self.config.max_missing_ratio:
            reasons.append("sample_count_mismatch")

        estimated_rate: float | None = None
        if len(timestamps) > 1:
            differences = np.diff(timestamps)
            if np.any(differences <= 0):
                reasons.append("non_monotonic_timestamps")
            else:
                estimated_rate = float(1.0 / np.median(differences))
                rate_error = abs(estimated_rate - self.sample_rate) / self.sample_rate
                if rate_error > self.config.sample_rate_tolerance_ratio:
                    reasons.append("sample_rate_mismatch")
        else:
            reasons.append("insufficient_timestamps")

        valid = not reasons
        status = "good" if valid and missing_ratio == 0 else ("degraded" if valid else "bad")
        return QualityReport(
            status=status,
            valid=valid,
            reasons=tuple(reasons),
            sample_count=count,
            expected_sample_count=expected,
            finite_ratio=finite_ratio,
            estimated_sample_rate=estimated_rate,
            missing_ratio=float(missing_ratio),
        )
