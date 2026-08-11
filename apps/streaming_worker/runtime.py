"""Composition root and lifecycle for the primary streaming worker."""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cogstate.features.pipeline import build_default_pipeline as build_full_feature_pipeline
from cogstate.features.streaming import build_lightweight_pipeline
from cogstate.preprocessing.artifact_removal import FasterConfig, apply_faster
from cogstate.preprocessing.filtering import FilterConfig, StreamingFilter
from cogstate.streaming.buffer import SignalBuffer, StreamSample, Window
from cogstate.streaming.inference import InferenceService, PredictionResult
from cogstate.streaming.processor import StreamProcessor

from .config import WorkerConfig
from .model_bundle import BundlePMModel, load_model_bundle
from .postprocessing import PredictionFilter
from .quality import EEGQualityGate, QualityReport
from .sinks import CompositeSink, ConsoleSink, JsonlSink, LatestStateSink
from .sources import LSLEEGSource, ReplayEEGSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingOutput:
    window_start: float
    window_end: float
    quality: QualityReport
    prediction: PredictionResult | None
    stage_latencies_ms: dict[str, float]
    model_version: str
    diagnostic_model: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "quality": self.quality.as_dict(),
            "prediction": asdict(self.prediction) if self.prediction else None,
            "stage_latencies_ms": self.stage_latencies_ms,
            "model_version": self.model_version,
            "diagnostic_model": self.diagnostic_model,
        }


class _WindowArtifactPreprocessor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.config = FasterConfig()

    def __call__(self, window: Window) -> np.ndarray:
        signal = window.data["eeg"]
        return apply_faster(signal, self.config) if self.enabled else signal.copy()


class StreamingRuntime:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        source: object | None = None,
        sink: object | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._closed = False
        self.processed_windows = 0
        self.rejected_windows = 0
        self.rejected_samples = 0

        sample_rate = config.signal.sample_rate
        channels = config.signal.channels
        self._filter = StreamingFilter(
            FilterConfig(
                sample_rate=sample_rate,
                bandpass_low_hz=config.preprocessing.bandpass_low_hz,
                bandpass_high_hz=config.preprocessing.bandpass_high_hz,
                notch_freq_hz=config.preprocessing.notch_hz,
            ),
            n_channels=len(channels),
        )
        self._buffer = SignalBuffer(
            window_size_s=config.windowing.window_seconds,
            step_size_s=config.windowing.step_seconds,
            required_sources=["eeg"],
        )
        self._buffer.register_source(
            "eeg",
            sample_rate,
            max_seconds=max(30.0, 3.0 * config.windowing.window_seconds),
        )
        self._quality = EEGQualityGate(
            sample_rate=sample_rate,
            n_channels=len(channels),
            config=config.quality,
        )
        self._features = (
            build_lightweight_pipeline(sample_rate)
            if config.features.profile == "lightweight"
            else build_full_feature_pipeline(sample_rate)
        )
        feature_count = len(self._features.feature_names(len(channels)))
        self.model: BundlePMModel = load_model_bundle(
            config.model.artifact_dir,
            n_features=feature_count,
            sample_rate=sample_rate,
            channels=channels,
            window_seconds=config.windowing.window_seconds,
            feature_profile=config.features.profile,
            allow_bootstrap=config.model.allow_bootstrap,
        )
        self._processor = StreamProcessor(
            buffer=self._buffer,
            preprocessor=_WindowArtifactPreprocessor(config.preprocessing.faster),
            feature_extractor=self._features,
            inference_service=InferenceService(self.model),
        )
        self._prediction_filter = PredictionFilter(config.postprocessing)
        self.latest_state = LatestStateSink()
        self.sink = sink or self._build_sink()
        self.source = source or self._build_source()

        if getattr(self.source, "n_channels", None) != len(channels):
            raise ValueError("Source channel count does not match signal.channels")
        if not np.isclose(getattr(self.source, "sample_rate", np.nan), sample_rate):
            raise ValueError("Source sample rate does not match signal.sample_rate")

    def _build_source(self) -> object:
        source = self.config.source
        if source.type == "replay":
            return ReplayEEGSource.from_path(
                source.path or "",
                sample_rate=self.config.signal.sample_rate,
                realtime=source.realtime,
                speed=source.speed,
                delimiter=source.delimiter,
                timestamp_column=source.timestamp_column,
            )
        return LSLEEGSource(
            source.stream_name,
            self.config.signal.sample_rate,
            len(self.config.signal.channels),
        )

    def _build_sink(self) -> CompositeSink:
        sinks: list[object] = [self.latest_state]
        if self.config.output.console:
            sinks.append(ConsoleSink())
        if self.config.output.jsonl_path:
            sinks.append(JsonlSink(self.config.output.jsonl_path))
        return CompositeSink(sinks)

    def _publish_rejection(self, window: Window, quality: QualityReport) -> None:
        self.rejected_windows += 1
        self.sink.publish(
            StreamingOutput(
                window_start=window.start_time,
                window_end=window.end_time,
                quality=quality,
                prediction=None,
                stage_latencies_ms={},
                model_version=self.model.version,
                diagnostic_model=self.model.manifest.diagnostic_only,
            )
        )

    def ingest(self, sample: StreamSample) -> None:
        """Filter one new sample once, then drain all newly ready windows."""
        with self._lock:
            if self._closed:
                return
            values = np.asarray(sample.values, dtype=float)
            if values.shape != (len(self.config.signal.channels),) or not np.isfinite(values).all():
                self.rejected_samples += 1
                self._filter.reset()
                return
            filtered = self._filter.process(values[None, :])[0].astype(np.float32)
            self._buffer.push(
                StreamSample(
                    source="eeg",
                    timestamp=sample.timestamp,
                    values=filtered,
                    received_at=sample.received_at,
                )
            )

            while True:
                window = self._buffer.poll_window()
                if window is None:
                    break
                quality = self._quality.evaluate(window)
                if not quality.valid:
                    self._publish_rejection(window, quality)
                    continue
                processed = self._processor.process_window(window)
                prediction = self._prediction_filter.apply(processed.prediction)
                self.processed_windows += 1
                self.sink.publish(
                    StreamingOutput(
                        window_start=window.start_time,
                        window_end=window.end_time,
                        quality=quality,
                        prediction=prediction,
                        stage_latencies_ms=processed.stage_latencies_ms,
                        model_version=self.model.version,
                        diagnostic_model=self.model.manifest.diagnostic_only,
                    )
                )

    def start(self) -> None:
        logger.info(
            "Starting streaming worker with model %s%s",
            self.model.version,
            " (diagnostic only)" if self.model.manifest.diagnostic_only else "",
        )
        self.source.start(self.ingest)

    def wait(self, timeout: float | None = None) -> bool:
        return bool(self.source.wait(timeout))

    def run(self) -> None:
        self.start()
        try:
            self.wait()
        finally:
            self.stop()

    def stop(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Stop outside the lock: a source callback may currently be waiting to
        # enter ``ingest`` and must be allowed to observe the closed flag.
        self.source.stop()
        self.sink.close()
