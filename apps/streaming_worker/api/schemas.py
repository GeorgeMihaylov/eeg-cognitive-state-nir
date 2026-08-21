from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


RuntimeStateName = Literal[
    "idle",
    "starting",
    "running",
    "stopping",
    "completed",
    "stopped",
    "failed",
]


class HealthResponse(BaseModel):
    status: str
    worker_running: bool
    runtime_state: RuntimeStateName
    run_id: str | None = None
    last_error: str | None = None


class RuntimeStatusResponse(BaseModel):
    state: RuntimeStateName
    run_id: str | None = None
    running: bool
    processed_windows: int
    rejected_windows: int
    rejected_samples: int
    model_version: str | None = None
    model_type: str | None = None
    input_mode: str | None = None
    class_names: list[str] | None = None
    target_names: list[str] | None = None
    diagnostic_model: bool | None = None
    last_error: str | None = None


class RuntimeActionResponse(BaseModel):
    changed: bool
    status: RuntimeStatusResponse


class QualityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    valid: bool
    reasons: list[str] | tuple[str, ...]
    sample_count: int
    expected_sample_count: int
    finite_ratio: float
    estimated_sample_rate: float | None
    missing_ratio: float


class ModelPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    probabilities: dict[str, float]
    model_version: str
    is_calibrated: bool
    inference_time_ms: float
    target_labels: dict[str, str] | None = None
    target_probabilities: dict[str, dict[str, float]] | None = None


class StreamingPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    window_start: float
    window_end: float
    quality: QualityResponse
    prediction: ModelPredictionResponse | None
    stage_latencies_ms: dict[str, float]
    model_version: str
    model_type: str
    input_mode: Literal["raw_eeg", "features"]
    class_names: list[str]
    diagnostic_model: bool


class PredictionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["prediction"] = "prediction"
    run_id: str
    sequence: int
    prediction: StreamingPredictionResponse
