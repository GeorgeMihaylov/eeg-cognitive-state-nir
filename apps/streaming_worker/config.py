"""Validated YAML configuration for the streaming worker."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cogstate.protocol import EEG_CHANNELS, SAMPLE_RATE, WINDOW_SECONDS


@dataclass(frozen=True)
class SourceConfig:
    type: str = "replay"
    path: str | None = None
    realtime: bool = False
    speed: float = 1.0
    stream_name: str = "EEG"
    delimiter: str = ","
    timestamp_column: int | None = None


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: float = float(SAMPLE_RATE)
    channels: tuple[str, ...] = EEG_CHANNELS


@dataclass(frozen=True)
class WindowingConfig:
    window_seconds: float = float(WINDOW_SECONDS)
    step_seconds: float = 1.0


@dataclass(frozen=True)
class PreprocessingConfig:
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 45.0
    notch_hz: float = 50.0
    faster: bool = True


@dataclass(frozen=True)
class QualityConfig:
    max_missing_ratio: float = 0.05
    sample_rate_tolerance_ratio: float = 0.05
    minimum_finite_ratio: float = 1.0


@dataclass(frozen=True)
class FeatureConfig:
    profile: str = "lightweight"


@dataclass(frozen=True)
class ModelConfig:
    artifact_dir: str = "artifacts/shallow_convnet_diagnostic"
    allow_bootstrap: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class PostprocessingConfig:
    probability_ema_alpha: float = 0.3
    minimum_confidence: float = 0.55
    confirmation_windows: int = 3


@dataclass(frozen=True)
class OutputConfig:
    console: bool = True
    jsonl_path: str | None = "outputs/live_predictions.jsonl"


@dataclass(frozen=True)
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    autostart_worker: bool = True


@dataclass(frozen=True)
class WorkerConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self) -> None:
        if self.source.type not in {"replay", "lsl"}:
            raise ValueError("source.type must be 'replay' or 'lsl'")
        if self.source.type == "replay" and not self.source.path:
            raise ValueError("source.path is required for replay")
        if self.source.speed <= 0:
            raise ValueError("source.speed must be positive")
        if self.signal.sample_rate <= 0 or not self.signal.channels:
            raise ValueError("A positive sample rate and channels are required")
        if not 0 < self.windowing.step_seconds <= self.windowing.window_seconds:
            raise ValueError("step_seconds must be in (0, window_seconds]")
        nyquist = self.signal.sample_rate / 2.0
        if not 0 < self.preprocessing.bandpass_low_hz < self.preprocessing.bandpass_high_hz < nyquist:
            raise ValueError("Bandpass frequencies must be inside the Nyquist range")
        if not 0 < self.preprocessing.notch_hz < nyquist:
            raise ValueError("notch_hz must be inside the Nyquist range")
        if not 0 <= self.quality.max_missing_ratio < 1:
            raise ValueError("max_missing_ratio must be in [0, 1)")
        if not 0 < self.quality.minimum_finite_ratio <= 1:
            raise ValueError("minimum_finite_ratio must be in (0, 1]")
        if self.features.profile not in {"lightweight", "full"}:
            raise ValueError("features.profile must be 'lightweight' or 'full'")
        if not self.model.device:
            raise ValueError("model.device must not be empty")
        if not 0 < self.postprocessing.probability_ema_alpha <= 1:
            raise ValueError("probability_ema_alpha must be in (0, 1]")
        if self.postprocessing.confirmation_windows < 1:
            raise ValueError("confirmation_windows must be positive")
        if not self.api.host or not 1 <= self.api.port <= 65535:
            raise ValueError("A valid API host and port are required")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkerConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("Streaming configuration must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerConfig":
        signal_payload = dict(payload.get("signal", {}))
        if "channels" in signal_payload:
            signal_payload["channels"] = tuple(signal_payload["channels"])
        return cls(
            source=SourceConfig(**payload.get("source", {})),
            signal=SignalConfig(**signal_payload),
            windowing=WindowingConfig(**payload.get("windowing", {})),
            preprocessing=PreprocessingConfig(**payload.get("preprocessing", {})),
            quality=QualityConfig(**payload.get("quality", {})),
            features=FeatureConfig(**payload.get("features", {})),
            model=ModelConfig(**payload.get("model", {})),
            postprocessing=PostprocessingConfig(**payload.get("postprocessing", {})),
            output=OutputConfig(**payload.get("output", {})),
            api=APIConfig(**payload.get("api", {})),
        )
