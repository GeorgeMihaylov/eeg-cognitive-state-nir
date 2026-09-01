from __future__ import annotations

import threading
from typing import Iterable


class LatestStateSink:
    def __init__(self) -> None:
        self._latest: object | None = None
        self._lock = threading.Lock()

    def publish(self, result: object) -> None:
        with self._lock:
            self._latest = result

    def get(self) -> object | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        return None


class CompositeSink:
    def __init__(self, sinks: Iterable[object]) -> None:
        self.sinks = tuple(sinks)

    def publish(self, result: object) -> None:
        for sink in self.sinks:
            sink.publish(result)

    def close(self) -> None:
        for sink in reversed(self.sinks):
            sink.close()
