"""Composition root and lifecycle for the primary streaming worker."""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cogstate.features.streaming import (
    build_lightweight_pipeline,
    build_streaming_full_pipeline,
)
from cogstate.preprocessing.artifact_removal import FasterConfig, apply_faster
from cogstate.preprocessing.filtering import FilterConfig, StreamingFilter
from cogstate.preprocessing.mne_faster import MNEFasterBundle
from cogstate.streaming.buffer import SignalBuffer, StreamSample, Window
from cogstate.streaming.inference import InferenceService, PredictionResult
from cogstate.streaming.processor import StreamProcessor

from .config import WorkerConfig
from .contracts import feature_schema_hash, preprocessing_contract, preprocessing_hash
from .model_bundle import (
    FeatureModelBundle,
    ShallowConvNetBundle,
    load_model_bundle,
)
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
    model_type: str
    input_mode: str
    class_names: tuple[str, ...]
    diagnostic_model: bool
    raw_prediction: PredictionResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "quality": self.quality.as_dict(),
            "prediction": asdict(self.prediction) if self.prediction else None,
            "raw_prediction": (
                asdict(self.raw_prediction) if self.raw_prediction else None
            ),
            "postprocessed_prediction": (
                asdict(self.prediction) if self.prediction else None
            ),
            "stage_latencies_ms": self.stage_latencies_ms,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "input_mode": self.input_mode,
            "class_names": list(self.class_names),
            "diagnostic_model": self.diagnostic_model,
        }


class _WindowArtifactPreprocessor:
    def __init__(
        self,
        bundle: MNEFasterBundle | None,
        *,
        legacy_faster_enabled: bool = False,
    ) -> None:
        self.bundle = bundle
        self.legacy_faster_enabled = legacy_faster_enabled
        self.legacy_config = FasterConfig()

    def __call__(self, window: Window) -> np.ndarray:
        signal = window.data["eeg"]
        if self.bundle is not None:
            return self.bundle.transform(signal)
        if self.legacy_faster_enabled:
            return apply_faster(signal, self.legacy_config)
        return signal.copy()


class _RawEEGWindowInput:
    """Convert ``[time, channels]`` windows to model input ``[1, channels, time]``."""

    def __init__(self, n_channels: int, n_times: int) -> None:
        self.expected_shape = (n_times, n_channels)

    def __call__(self, clean_signal: np.ndarray, window: Window) -> np.ndarray:
        values = np.asarray(clean_signal, dtype=np.float32)
        if values.shape != self.expected_shape:
            raise ValueError(
                f"Expected cleaned EEG window {self.expected_shape}, got {values.shape}"
            )
        return np.ascontiguousarray(values.T[None, :, :])


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
                bandpass_enabled=config.preprocessing.bandpass_enabled,
                bandpass_low_hz=config.preprocessing.bandpass_low_hz,
                bandpass_high_hz=config.preprocessing.bandpass_high_hz,
                notch_freq_hz=config.preprocessing.notch_hz,
                notch_enabled=config.preprocessing.notch_enabled,
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
        feature_pipeline = (
            build_lightweight_pipeline(sample_rate)
            if config.features.profile == "lightweight"
            else build_streaming_full_pipeline(sample_rate)
        )
        feature_names = feature_pipeline.feature_names(len(channels))
        feature_count = len(feature_names)
        filter_contract = {
            "bandpass_low_hz": config.preprocessing.bandpass_low_hz,
            "bandpass_high_hz": config.preprocessing.bandpass_high_hz,
            "notch_hz": config.preprocessing.notch_hz,
            "filter_mode": "causal",
        }
        artifact_bundle: MNEFasterBundle | None = None
        if config.preprocessing.mne_faster_enabled:
            assert config.preprocessing.mne_faster_bundle_dir is not None
            artifact_bundle = MNEFasterBundle.load(
                config.preprocessing.mne_faster_bundle_dir
            )
            artifact_bundle.validate(
                sample_rate=sample_rate,
                channel_names=channels,
                preprocessing_contract=filter_contract,
            )
        raw_preprocessing_contract = {
            **filter_contract,
            "artifact_removal": "mne_faster" if artifact_bundle else "none",
            "artifact_bundle_version": artifact_bundle.version if artifact_bundle else None,
        }
        feature_preprocessing_contract = preprocessing_contract(
            sample_rate=sample_rate,
            bandpass_enabled=config.preprocessing.bandpass_enabled,
            bandpass_low_hz=config.preprocessing.bandpass_low_hz,
            bandpass_high_hz=config.preprocessing.bandpass_high_hz,
            notch_enabled=config.preprocessing.notch_enabled,
            notch_hz=config.preprocessing.notch_hz,
            faster=config.preprocessing.faster,
        )
        self.model: ShallowConvNetBundle | FeatureModelBundle = load_model_bundle(
            config.model.artifact_dir,
            sample_rate=sample_rate,
            channels=channels,
            window_seconds=config.windowing.window_seconds,
            preprocessing=raw_preprocessing_contract,
            allow_bootstrap=config.model.allow_bootstrap,
            device=config.model.device,
            n_features=feature_count,
            feature_profile=config.features.profile,
            feature_schema_hash_value=feature_schema_hash(feature_names),
            preprocessing_hash_value=preprocessing_hash(
                feature_preprocessing_contract
            ),
        )
        if self.model.manifest.input_mode == "raw_eeg":
            model_input = _RawEEGWindowInput(
                len(channels), self.model.manifest.n_times
            )
        else:
            model_input = feature_pipeline
        self._processor = StreamProcessor(
            buffer=self._buffer,
            preprocessor=_WindowArtifactPreprocessor(
                artifact_bundle,
                legacy_faster_enabled=config.preprocessing.faster,
            ),
            feature_extractor=model_input,
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
                model_type=self.model.manifest.model_type,
                input_mode=self.model.manifest.input_mode,
                class_names=tuple(
                    getattr(
                        self.model.manifest,
                        "class_names",
                        ("low", "medium", "high"),
                    )
                ),
                diagnostic_model=self.model.manifest.diagnostic_only,
                raw_prediction=None,
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
                raw_prediction = processed.prediction
                prediction = self._prediction_filter.apply(raw_prediction)
                self.processed_windows += 1
                self.sink.publish(
                    StreamingOutput(
                        window_start=window.start_time,
                        window_end=window.end_time,
                        quality=quality,
                        prediction=prediction,
                        stage_latencies_ms=processed.stage_latencies_ms,
                        model_version=self.model.version,
                        model_type=self.model.manifest.model_type,
                        input_mode=self.model.manifest.input_mode,
                        class_names=tuple(
                            getattr(
                                self.model.manifest,
                                "class_names",
                                ("low", "medium", "high"),
                            )
                        ),
                        diagnostic_model=self.model.manifest.diagnostic_only,
                        raw_prediction=raw_prediction,
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
