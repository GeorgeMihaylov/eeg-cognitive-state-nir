import json

import numpy as np

from apps.streaming_worker.config import WorkerConfig
from apps.streaming_worker.runtime import StreamingRuntime
from apps.streaming_worker.sources.replay import ReplayEEGSource


class CaptureSink:
    def __init__(self):
        self.results = []
        self.closed = False

    def publish(self, result):
        self.results.append(result)

    def close(self):
        self.closed = True


def test_replay_to_model_end_to_end(tmp_path):
    sample_rate = 128
    channels = ("C1", "C2", "C3", "C4")
    duration_s = 3
    timestamps = np.arange(sample_rate * duration_s) / sample_rate
    signal = np.column_stack(
        [
            np.sin(2 * np.pi * frequency * timestamps)
            for frequency in (6, 10, 18, 30)
        ]
    ).astype(np.float32)
    source = ReplayEEGSource(signal, sample_rate=sample_rate, realtime=False)
    sink = CaptureSink()
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
                "artifact_dir": str(tmp_path / "missing_bundle"),
                "allow_bootstrap": True,
            },
            "postprocessing": {
                "probability_ema_alpha": 0.5,
                "minimum_confidence": 0.0,
                "confirmation_windows": 1,
            },
            "output": {"console": False, "jsonl_path": None},
        }
    )
    runtime = StreamingRuntime(config, source=source, sink=sink)

    runtime.run()

    assert runtime.processed_windows == 2
    assert runtime.rejected_windows == 0
    assert sink.closed
    assert len(sink.results) == 2
    output = sink.results[-1]
    assert output.quality.valid
    assert output.diagnostic_model
    assert output.prediction is not None
    assert set(output.prediction.target_labels) == {
        "attention",
        "engagement",
        "excitement",
        "stress",
        "relaxation",
        "interest",
        "focus",
    }
    assert output.stage_latencies_ms.keys() >= {
        "preprocessing",
        "feature_extraction",
        "inference",
    }


def test_replay_to_shallow_convnet_end_to_end(tmp_path):
    sample_rate = 128
    channels = ("C1", "C2", "C3", "C4")
    timestamps = np.arange(sample_rate * 3) / sample_rate
    signal = np.column_stack(
        [
            np.sin(2 * np.pi * frequency * timestamps)
            for frequency in (6, 10, 18, 30)
        ]
    ).astype(np.float32)
    artifact_dir = tmp_path / "shallow_bundle"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "shallow-diagnostic-v1",
                "model_type": "torch_shallow_convnet_multitask",
                "input_mode": "raw_eeg",
                "input_layout": "batch,1,channels,time",
                "sample_rate": sample_rate,
                "channels": list(channels),
                "window_seconds": 2,
                "n_times": 256,
                "class_names": ["low", "medium", "high"],
                "target_names": ["attention", "engagement", "excitement", "stress", "relaxation", "interest", "focus"],
                "preprocessing": {
                    "bandpass_low_hz": 1,
                    "bandpass_high_hz": 45,
                    "notch_hz": 50,
                    "filter_mode": "causal",
                    "artifact_removal": "none",
                    "artifact_bundle_version": None,
                },
                "bootstrap": True,
                "diagnostic_only": True,
            }
        ),
        encoding="utf-8",
    )
    source = ReplayEEGSource(signal, sample_rate=sample_rate, realtime=False)
    sink = CaptureSink()
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
                "artifact_dir": str(artifact_dir),
                "allow_bootstrap": True,
                "device": "cpu",
            },
            "postprocessing": {
                "minimum_confidence": 0.0,
                "confirmation_windows": 1,
            },
            "output": {"console": False, "jsonl_path": None},
        }
    )

    runtime = StreamingRuntime(config, source=source, sink=sink)
    runtime.run()

    assert runtime.model.manifest.input_mode == "raw_eeg"
    assert runtime.processed_windows == 2
    assert len(sink.results) == 2
    assert set(sink.results[-1].prediction.target_labels) == {
        "attention", "engagement", "excitement", "stress",
        "relaxation", "interest", "focus",
    }
