"""Thread-safe bridge between the synchronous worker and async API clients."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from ..config import WorkerConfig
from ..runtime import StreamingRuntime
from ..sinks import CompositeSink, ConsoleSink, JsonlSink


class PredictionStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._latest: dict | None = None

    def publish(self, result: object) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else result
        if not isinstance(payload, dict):
            raise TypeError("API prediction sink requires a mapping-like result")
        with self._condition:
            self._sequence += 1
            self._latest = payload
            self._condition.notify_all()

    def snapshot(self) -> tuple[int, dict | None]:
        with self._condition:
            return self._sequence, self._latest

    def wait_after(
        self, sequence: int, timeout: float | None = None
    ) -> tuple[int, dict | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > sequence, timeout)
            if self._sequence <= sequence:
                return sequence, None
            return self._sequence, self._latest

    def close(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True)
class ServiceStatus:
    running: bool
    processed_windows: int
    rejected_windows: int
    rejected_samples: int
    model_version: str | None
    diagnostic_model: bool | None
    last_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "running": self.running,
            "processed_windows": self.processed_windows,
            "rejected_windows": self.rejected_windows,
            "rejected_samples": self.rejected_samples,
            "model_version": self.model_version,
            "diagnostic_model": self.diagnostic_model,
            "last_error": self.last_error,
        }


RuntimeFactory = Callable[[WorkerConfig, object], StreamingRuntime]


class StreamingService:
    """Own one worker instance while keeping FastAPI out of the runtime core."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.config = config
        self.store = PredictionStore()
        self._runtime_factory = runtime_factory or (
            lambda worker_config, sink: StreamingRuntime(worker_config, sink=sink)
        )
        self._runtime: StreamingRuntime | None = None
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._lock = threading.RLock()

    def _build_sink(self) -> CompositeSink:
        sinks: list[object] = [self.store]
        if self.config.output.console:
            sinks.append(ConsoleSink())
        if self.config.output.jsonl_path:
            sinks.append(JsonlSink(self.config.output.jsonl_path))
        return CompositeSink(sinks)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._last_error = None
            sink = self._build_sink()
            try:
                runtime = self._runtime_factory(self.config, sink)
            except Exception as exc:
                sink.close()
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._runtime = None
                self._thread = None
                return False
            self._runtime = runtime

            def run() -> None:
                try:
                    runtime.run()
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"

            self._thread = threading.Thread(
                target=run, daemon=True, name="streaming-worker-service"
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lock:
            runtime = self._runtime
            thread = self._thread
            changed = bool(thread is not None and thread.is_alive())
        if runtime is not None:
            runtime.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return changed

    def status(self) -> ServiceStatus:
        with self._lock:
            runtime = self._runtime
            model = getattr(runtime, "model", None) if runtime is not None else None
            manifest = getattr(model, "manifest", None) if model is not None else None
            return ServiceStatus(
                running=self._thread is not None and self._thread.is_alive(),
                processed_windows=getattr(runtime, "processed_windows", 0),
                rejected_windows=getattr(runtime, "rejected_windows", 0),
                rejected_samples=getattr(runtime, "rejected_samples", 0),
                model_version=getattr(model, "version", None),
                diagnostic_model=getattr(manifest, "diagnostic_only", None),
                last_error=self._last_error,
            )
