"""Deterministic file/array replay through the live sample interface."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from cogstate.streaming.buffer import StreamSample


class ReplayEEGSource:
    source = "eeg"

    def __init__(
        self,
        signal: np.ndarray,
        *,
        sample_rate: float,
        timestamps: np.ndarray | None = None,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> None:
        data = np.asarray(signal, dtype=np.float32)
        if data.ndim != 2 or not len(data):
            raise ValueError("Replay signal must be [samples, channels] and non-empty")
        if sample_rate <= 0 or speed <= 0:
            raise ValueError("sample_rate and speed must be positive")
        if timestamps is None:
            times = np.arange(len(data), dtype=float) / sample_rate
        else:
            times = np.asarray(timestamps, dtype=float)
            if times.shape != (len(data),) or not np.all(np.diff(times) > 0):
                raise ValueError("timestamps must be strictly increasing and match samples")
        self.signal = data
        self.timestamps = times
        self.sample_rate = float(sample_rate)
        self.speed = float(speed)
        self.realtime = bool(realtime)
        self.n_channels = data.shape[1]
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._error: BaseException | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        sample_rate: float,
        realtime: bool = False,
        speed: float = 1.0,
        delimiter: str = ",",
        timestamp_column: int | None = None,
    ) -> "ReplayEEGSource":
        source_path = Path(path)
        suffix = source_path.suffix.lower()
        timestamps = None
        if suffix == ".npy":
            signal = np.load(source_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(source_path, allow_pickle=False) as payload:
                signal = payload["eeg"]
                timestamps = payload["timestamps"] if "timestamps" in payload else None
        else:
            signal = np.genfromtxt(source_path, delimiter=delimiter)
        signal = np.asarray(signal)
        if timestamp_column is not None:
            timestamps = signal[:, timestamp_column]
            signal = np.delete(signal, timestamp_column, axis=1)
        return cls(
            signal,
            sample_rate=sample_rate,
            timestamps=timestamps,
            realtime=realtime,
            speed=speed,
        )

    @property
    def finished(self) -> bool:
        return self._done_event.is_set()

    def start(self, on_sample: Callable[[StreamSample], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Replay source is already running")
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None

        def run() -> None:
            try:
                previous_time = self.timestamps[0]
                for timestamp, values in zip(self.timestamps, self.signal):
                    if self._stop_event.is_set():
                        break
                    if self.realtime:
                        delay = max(0.0, float(timestamp - previous_time) / self.speed)
                        if self._stop_event.wait(delay):
                            break
                    on_sample(
                        StreamSample(
                            source=self.source,
                            timestamp=float(timestamp),
                            values=values.copy(),
                        )
                    )
                    previous_time = float(timestamp)
            except BaseException as exc:
                self._error = exc
            finally:
                self._done_event.set()

        self._thread = threading.Thread(target=run, daemon=True, name="eeg-replay")
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        finished = self._done_event.wait(timeout)
        if finished and self._error is not None:
            raise RuntimeError("Replay source failed") from self._error
        return finished

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
