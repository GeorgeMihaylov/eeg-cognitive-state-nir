"""LSL source isolated from the core library and imported lazily."""
from __future__ import annotations

import threading
from typing import Callable

from cogstate.streaming.buffer import StreamSample
from cogstate.streaming.device_adapters import LSLEEGAdapter


class LSLEEGSource:
    source = "eeg"

    def __init__(self, stream_name: str, sample_rate: float, n_channels: int) -> None:
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self._adapter = LSLEEGAdapter(stream_name, sample_rate, n_channels)
        self._stopped = threading.Event()

    def start(self, on_sample: Callable[[StreamSample], None]) -> None:
        self._stopped.clear()
        self._adapter.start(on_sample)

    def wait(self, timeout: float | None = None) -> bool:
        return self._stopped.wait(timeout)

    def stop(self) -> None:
        self._adapter.stop()
        self._stopped.set()
