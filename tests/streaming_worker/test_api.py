import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from apps.streaming_worker.api import create_app
from apps.streaming_worker.api.state import StreamingService
from apps.streaming_worker.config import WorkerConfig
from apps.streaming_worker.runtime import StreamingRuntime
from apps.streaming_worker.sources import ReplayEEGSource


class FakeManifest:
    diagnostic_only = True
    model_type = "torch_shallow_convnet"
    input_mode = "raw_eeg"
    class_names = ("low", "medium", "high")


class FakeModel:
    version = "fake-v1"
    manifest = FakeManifest()


class FakeRuntime:
    def __init__(self, config, sink):
        self.config = config
        self.sink = sink
        self.model = FakeModel()
        self.processed_windows = 0
        self.rejected_windows = 0
        self.rejected_samples = 0
        self._stop = threading.Event()
        self.sink.publish(
            {
                "window_start": 0.0,
                "window_end": 2.0,
                "quality": {
                    "status": "good",
                    "valid": True,
                    "reasons": [],
                    "sample_count": 256,
                    "expected_sample_count": 256,
                    "finite_ratio": 1.0,
                    "estimated_sample_rate": 128.0,
                    "missing_ratio": 0.0,
                },
                "prediction": {
                    "label": "high",
                    "probabilities": {"low": 0.1, "medium": 0.2, "high": 0.7},
                    "model_version": self.model.version,
                    "is_calibrated": False,
                    "inference_time_ms": 1.0,
                    "target_labels": None,
                    "target_probabilities": None,
                },
                "stage_latencies_ms": {
                    "preprocessing": 0.5,
                    "feature_extraction": 0.1,
                    "inference": 1.0,
                },
                "model_version": self.model.version,
                "model_type": self.model.manifest.model_type,
                "input_mode": self.model.manifest.input_mode,
                "class_names": list(self.model.manifest.class_names),
                "diagnostic_model": True,
            }
        )

    def run(self):
        self.processed_windows = 1
        self._stop.wait()

    def stop(self):
        self._stop.set()
        self.sink.close()


def api_config():
    return WorkerConfig.from_dict(
        {
            "source": {"type": "replay", "path": "unused.npy"},
            "signal": {
                "sample_rate": 128,
                "channels": ["C1", "C2"],
            },
            "preprocessing": {
                "bandpass_low_hz": 1,
                "bandpass_high_hz": 45,
                "notch_hz": 50,
            },
            "output": {"console": False, "jsonl_path": None},
            "api": {"autostart_worker": False},
        }
    )


def test_http_lifecycle_latest_and_websocket():
    config = api_config()
    service = StreamingService(
        config,
        runtime_factory=lambda worker_config, sink: FakeRuntime(
            worker_config, sink
        ),
    )
    app = create_app(config=config, service=service)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["worker_running"] is False
        assert health.json()["runtime_state"] == "idle"

        started = client.post("/v1/runtime/start")
        assert started.status_code == 200
        assert started.json()["changed"] is True

        status = client.get("/v1/status").json()
        assert status["running"] is True
        assert status["state"] == "running"
        assert status["run_id"]
        assert status["model_version"] == "fake-v1"
        assert status["model_type"] == "torch_shallow_convnet"
        assert status["input_mode"] == "raw_eeg"
        assert status["class_names"] == ["low", "medium", "high"]
        assert status["diagnostic_model"] is True

        duplicate = client.post("/v1/runtime/start")
        assert duplicate.status_code == 409

        latest = client.get("/v1/predictions/latest")
        assert latest.status_code == 200
        assert latest.json()["type"] == "prediction"
        assert latest.json()["run_id"] == status["run_id"]
        assert latest.json()["prediction"]["prediction"]["label"] == "high"

        with client.websocket_connect("/v1/stream") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "prediction"
            assert message["sequence"] == 1
            assert message["prediction"]["model_version"] == "fake-v1"

        stopped = client.post("/v1/runtime/stop")
        assert stopped.status_code == 200
        assert stopped.json()["changed"] is True
        assert stopped.json()["status"]["running"] is False
        assert stopped.json()["status"]["state"] == "stopped"


def test_latest_returns_404_before_first_prediction():
    config = api_config()
    service = StreamingService(config)
    app = create_app(config=config, service=service)

    with TestClient(app) as client:
        response = client.get("/v1/predictions/latest")

    assert response.status_code == 404


def test_api_stays_available_when_worker_autostart_fails(tmp_path):
    config = WorkerConfig.from_dict(
        {
            "source": {
                "type": "replay",
                "path": str(tmp_path / "missing.npy"),
            },
            "signal": {
                "sample_rate": 128,
                "channels": ["C1", "C2"],
            },
            "windowing": {"window_seconds": 2, "step_seconds": 1},
            "preprocessing": {
                "bandpass_low_hz": 1,
                "bandpass_high_hz": 45,
                "notch_hz": 50,
            },
            "model": {
                "artifact_dir": str(tmp_path / "bundle"),
                "allow_bootstrap": True,
            },
            "output": {"console": False, "jsonl_path": None},
            "api": {"autostart_worker": True},
        }
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        health = client.get("/health")
        status = client.get("/v1/status")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["worker_running"] is False
    assert health.json()["runtime_state"] == "failed"
    assert status.json()["state"] == "failed"
    assert "FileNotFoundError" in status.json()["last_error"]


def test_manual_start_failure_returns_service_unavailable(tmp_path):
    config = WorkerConfig.from_dict(
        {
            "source": {"type": "replay", "path": str(tmp_path / "missing.npy")},
            "signal": {"sample_rate": 128, "channels": ["C1", "C2"]},
            "windowing": {"window_seconds": 2, "step_seconds": 1},
            "preprocessing": {
                "bandpass_low_hz": 1,
                "bandpass_high_hz": 45,
                "notch_hz": 50,
            },
            "model": {
                "artifact_dir": str(tmp_path / "bundle"),
                "allow_bootstrap": True,
            },
            "output": {"console": False, "jsonl_path": None},
            "api": {"autostart_worker": False},
        }
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        response = client.post("/v1/runtime/start")
        status = client.get("/v1/status")

    assert response.status_code == 503
    assert status.json()["state"] == "failed"


def test_completed_runtime_emits_terminal_websocket_event():
    class CompletingRuntime(FakeRuntime):
        def run(self):
            self.processed_windows = 1
            self.sink.close()

    config = api_config()
    service = StreamingService(
        config,
        runtime_factory=lambda worker_config, sink: CompletingRuntime(
            worker_config, sink
        ),
    )
    app = create_app(config=config, service=service)

    with TestClient(app) as client:
        started = client.post("/v1/runtime/start")
        assert started.status_code == 200
        for _ in range(50):
            if client.get("/v1/status").json()["state"] == "completed":
                break
            time.sleep(0.01)

        status = client.get("/v1/status").json()
        assert status["state"] == "completed"
        assert status["running"] is False

        with client.websocket_connect("/v1/stream") as websocket:
            prediction = websocket.receive_json()
            terminal = websocket.receive_json()

        assert prediction["type"] == "prediction"
        assert terminal["type"] == "runtime_completed"
        assert terminal["run_id"] == prediction["run_id"]


def test_new_run_clears_previous_latest_prediction():
    class SilentRuntime(FakeRuntime):
        def __init__(self, config, sink):
            self.config = config
            self.sink = sink
            self.model = FakeModel()
            self.processed_windows = 0
            self.rejected_windows = 0
            self.rejected_samples = 0
            self._stop = threading.Event()

    created = 0

    def runtime_factory(config, sink):
        nonlocal created
        created += 1
        return FakeRuntime(config, sink) if created == 1 else SilentRuntime(config, sink)

    config = api_config()
    service = StreamingService(config, runtime_factory=runtime_factory)

    service.start()
    first_run_id = service.status().run_id
    assert service.store.snapshot().prediction is not None
    service.stop()

    service.start()
    second_snapshot = service.store.snapshot()
    try:
        assert second_snapshot.run_id != first_run_id
        assert second_snapshot.sequence == 0
        assert second_snapshot.prediction is None
    finally:
        service.stop()


def test_feature_model_output_matches_typed_api_contract(tmp_path):
    sample_rate = 128
    channels = ("C1", "C2", "C3", "C4")
    timestamps = np.arange(sample_rate * 3) / sample_rate
    signal = np.column_stack(
        [
            np.sin(2 * np.pi * frequency * timestamps)
            for frequency in (6, 10, 18, 30)
        ]
    ).astype(np.float32)
    config = WorkerConfig.from_dict(
        {
            "source": {"type": "replay", "path": "unused.npy"},
            "signal": {"sample_rate": sample_rate, "channels": channels},
            "windowing": {"window_seconds": 2, "step_seconds": 1},
            "preprocessing": {
                "bandpass_low_hz": 1,
                "bandpass_high_hz": 45,
                "notch_hz": 50,
                "mne_faster_enabled": False,
            },
            "model": {
                "artifact_dir": str(tmp_path / "missing_feature_bundle"),
                "allow_bootstrap": True,
            },
            "postprocessing": {
                "minimum_confidence": 0.0,
                "confirmation_windows": 1,
            },
            "output": {"console": False, "jsonl_path": None},
            "api": {"autostart_worker": False},
        }
    )

    def runtime_factory(worker_config, sink):
        source = ReplayEEGSource(signal, sample_rate=sample_rate, realtime=False)
        return StreamingRuntime(worker_config, source=source, sink=sink)

    service = StreamingService(config, runtime_factory=runtime_factory)
    app = create_app(config=config, service=service)

    with TestClient(app) as client:
        assert client.post("/v1/runtime/start").status_code == 200
        for _ in range(100):
            status = client.get("/v1/status").json()
            if status["state"] == "completed":
                break
            time.sleep(0.01)
        latest = client.get("/v1/predictions/latest")

    assert status["input_mode"] == "features"
    assert status["target_names"] == [
        "attention",
        "engagement",
        "excitement",
        "stress",
        "relaxation",
        "interest",
        "focus",
    ]
    assert latest.status_code == 200
    assert latest.json()["prediction"]["input_mode"] == "features"
    assert latest.json()["prediction"]["prediction"]["target_labels"] is not None
