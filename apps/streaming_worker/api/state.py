"""Thread-safe bridge between the synchronous worker and async API clients."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from uuid import uuid4

from ..config import WorkerConfig
from ..runtime import StreamingRuntime
from ..sinks import CompositeSink, ConsoleSink, JsonlSink


class PredictionStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._run_id: str | None = None
        self._sequence = 0
        self._latest: dict | None = None
        self._terminal: dict | None = None

    def reset(self, run_id: str) -> None:
        with self._condition:
            self._run_id = run_id
            self._sequence = 0
            self._latest = None
            self._terminal = None
            self._condition.notify_all()

    def publish(self, result: object) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else result
        if not isinstance(payload, dict):
            raise TypeError("API prediction sink requires a mapping-like result")
        with self._condition:
            if self._run_id is None:
                raise RuntimeError("PredictionStore has no active run")
            self._sequence += 1
            self._latest = payload
            self._condition.notify_all()

    def finish(self, event_type: str, error: str | None = None) -> None:
        with self._condition:
            self._terminal = {
                "type": event_type,
                "run_id": self._run_id,
                "sequence": self._sequence,
            }
            if error is not None:
                self._terminal["error"] = error
            self._condition.notify_all()

    def snapshot(self) -> "StoreSnapshot":
        with self._condition:
            return StoreSnapshot(
                run_id=self._run_id,
                sequence=self._sequence,
                prediction=self._latest,
                terminal=self._terminal,
            )

    def wait_after(
        self, run_id: str | None, sequence: int, timeout: float | None = None
    ) -> "StoreSnapshot":
        with self._condition:
            self._condition.wait_for(
                lambda: self._run_id != run_id
                or self._sequence > sequence
                or self._terminal is not None,
                timeout,
            )
            return StoreSnapshot(
                run_id=self._run_id,
                sequence=self._sequence,
                prediction=self._latest if self._sequence > sequence else None,
                terminal=self._terminal,
            )

    def close(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True)
class StoreSnapshot:
    run_id: str | None
    sequence: int
    prediction: dict | None
    terminal: dict | None


class RuntimeState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RuntimeAlreadyRunningError(RuntimeError):
    pass


class RuntimeStartError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceStatus:
    state: str
    run_id: str | None
    running: bool
    processed_windows: int
    rejected_windows: int
    rejected_samples: int
    model_version: str | None
    model_type: str | None
    input_mode: str | None
    class_names: tuple[str, ...] | None
    target_names: tuple[str, ...] | None
    diagnostic_model: bool | None
    last_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "run_id": self.run_id,
            "running": self.running,
            "processed_windows": self.processed_windows,
            "rejected_windows": self.rejected_windows,
            "rejected_samples": self.rejected_samples,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "input_mode": self.input_mode,
            "class_names": list(self.class_names) if self.class_names else None,
            "target_names": list(self.target_names) if self.target_names else None,
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
        self._state = RuntimeState.IDLE
        self._run_id: str | None = None
        self._stop_requested = False
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
                raise RuntimeAlreadyRunningError("Streaming worker is already running")
            self._last_error = None
            self._stop_requested = False
            self._run_id = uuid4().hex
            self._state = RuntimeState.STARTING
            self.store.reset(self._run_id)
            sink = self._build_sink()
            try:
                runtime = self._runtime_factory(self.config, sink)
            except Exception as exc:
                sink.close()
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._state = RuntimeState.FAILED
                self._runtime = None
                self._thread = None
                self.store.finish("runtime_failed", self._last_error)
                raise RuntimeStartError(self._last_error) from exc
            self._runtime = runtime
            self._state = RuntimeState.RUNNING

            def run() -> None:
                try:
                    runtime.run()
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._state = RuntimeState.FAILED
                        error = self._last_error
                    self.store.finish("runtime_failed", error)
                else:
                    with self._lock:
                        stopped = self._stop_requested
                        self._state = (
                            RuntimeState.STOPPED if stopped else RuntimeState.COMPLETED
                        )
                    self.store.finish(
                        "runtime_stopped" if stopped else "runtime_completed"
                    )

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
            if changed:
                self._stop_requested = True
                self._state = RuntimeState.STOPPING
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
            estimator = getattr(model, "estimator", None) if model is not None else None
            class_names = getattr(manifest, "class_names", None)
            if class_names is None and manifest is not None:
                class_names = ("low", "medium", "high")
            target_names = getattr(estimator, "metric_names", None)
            return ServiceStatus(
                state=self._state.value,
                run_id=self._run_id,
                running=self._thread is not None and self._thread.is_alive(),
                processed_windows=getattr(runtime, "processed_windows", 0),
                rejected_windows=getattr(runtime, "rejected_windows", 0),
                rejected_samples=getattr(runtime, "rejected_samples", 0),
                model_version=getattr(model, "version", None),
                model_type=getattr(manifest, "model_type", None),
                input_mode=getattr(manifest, "input_mode", None),
                class_names=tuple(class_names) if class_names is not None else None,
                target_names=tuple(target_names) if target_names is not None else None,
                diagnostic_model=getattr(manifest, "diagnostic_only", None),
                last_error=self._last_error,
            )
