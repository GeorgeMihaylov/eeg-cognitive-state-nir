from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class StreamSample:

    source: str
    timestamp: float
    values: np.ndarray
    received_at: float = field(default_factory=time.monotonic)


@dataclass
class Window:

    start_time: float
    end_time: float
    data: Dict[str, np.ndarray]
    timestamps: Dict[str, np.ndarray]


class RingChannelBuffer:

    def __init__(self, source: str, sample_rate: float, max_seconds: float = 30.0):
        self.source = source
        self.sample_rate = sample_rate
        self._maxlen = int(sample_rate * max_seconds)
        self._samples: Deque[StreamSample] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def push(self, sample: StreamSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def slice(self, start_time: float, end_time: float) -> Optional[StreamSample]:
        with self._lock:
            selected = [s for s in self._samples if start_time <= s.timestamp < end_time]
        if not selected:
            return None
        values = np.stack([s.values for s in selected])
        timestamps = np.array([s.timestamp for s in selected])
        return values, timestamps  # type: ignore[return-value]

    def latest_timestamp(self) -> Optional[float]:
        with self._lock:
            if not self._samples:
                return None
            return self._samples[-1].timestamp


class SignalBuffer:
    def __init__(
        self,
        window_size_s: float = 2.0,
        step_size_s: float = 0.5,
        required_sources: Optional[list[str]] = None,
    ):
        if step_size_s <= 0 or step_size_s > window_size_s:
            raise ValueError("step_size_s должен быть в (0, window_size_s]")

        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
        self.required_sources = required_sources or ["eeg"]

        self._channels: Dict[str, RingChannelBuffer] = {}
        self._next_window_start: Optional[float] = None
        self._lock = threading.Lock()

    def register_source(self, source: str, sample_rate: float, max_seconds: float = 30.0) -> None:
        with self._lock:
            self._channels[source] = RingChannelBuffer(source, sample_rate, max_seconds)

    def push(self, sample: StreamSample) -> None:
        channel = self._channels.get(sample.source)
        if channel is None:
            return
        channel.push(sample)

        with self._lock:
            if self._next_window_start is None:
                self._next_window_start = sample.timestamp

    def poll_window(self) -> Optional[Window]:
        with self._lock:
            if self._next_window_start is None:
                return None
            start = self._next_window_start
            end = start + self.window_size_s

        data: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, np.ndarray] = {}

        for source in self.required_sources:
            channel = self._channels.get(source)
            if channel is None:
                return None
            latest = channel.latest_timestamp()
            # Samples are left-closed/right-open.  A regular stream whose last
            # sample is at ``end - 1 / sample_rate`` already completes window.
            if latest is None or latest + 1.0 / channel.sample_rate + 1e-9 < end:
                return None

            sliced = channel.slice(start, end)
            if sliced is None:
                return None
            values, ts = sliced
            data[source] = values
            timestamps[source] = ts

        for source, channel in self._channels.items():
            if source in data:
                continue
            sliced = channel.slice(start, end)
            if sliced is not None:
                values, ts = sliced
                data[source] = values
                timestamps[source] = ts

        with self._lock:
            self._next_window_start = start + self.step_size_s

        return Window(start_time=start, end_time=end, data=data, timestamps=timestamps)
