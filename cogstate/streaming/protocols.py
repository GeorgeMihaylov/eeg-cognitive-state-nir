"""Transport-neutral contracts shared by streaming applications."""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .buffer import StreamSample


class SampleSource(Protocol):
    """A live or replayed source that emits timestamped samples."""

    source: str
    sample_rate: float
    n_channels: int

    def start(self, on_sample: Callable[[StreamSample], None]) -> None: ...

    def stop(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class ResultSink(Protocol):
    """A destination for serializable streaming result objects."""

    def publish(self, result: object) -> None: ...

    def close(self) -> None: ...


class StreamingModel(Protocol):
    version: str

    def predict_proba(self, features: np.ndarray) -> dict[str, float]: ...


class StreamingPMModel(Protocol):
    version: str

    def predict_pm_proba(
        self, features: np.ndarray
    ) -> dict[str, dict[str, float]]: ...
