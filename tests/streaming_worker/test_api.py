import threading

from fastapi.testclient import TestClient

from apps.streaming_worker.api import create_app
from apps.streaming_worker.api.state import StreamingService
from apps.streaming_worker.config import WorkerConfig


class FakeManifest:
    diagnostic_only = True


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
                "prediction": {"label": "high"},
                "model_version": self.model.version,
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

        started = client.post("/v1/runtime/start")
        assert started.status_code == 200
        assert started.json()["changed"] is True

        status = client.get("/v1/status").json()
        assert status["running"] is True
        assert status["model_version"] == "fake-v1"
        assert status["diagnostic_model"] is True

        latest = client.get("/v1/predictions/latest")
        assert latest.status_code == 200
        assert latest.json()["prediction"]["prediction"]["label"] == "high"

        with client.websocket_connect("/v1/stream") as websocket:
            message = websocket.receive_json()
            assert message["sequence"] == 1
            assert message["prediction"]["model_version"] == "fake-v1"

        stopped = client.post("/v1/runtime/stop")
        assert stopped.status_code == 200
        assert stopped.json()["changed"] is True
        assert stopped.json()["status"]["running"] is False


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
    assert "FileNotFoundError" in status.json()["last_error"]
