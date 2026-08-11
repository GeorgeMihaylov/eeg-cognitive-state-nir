from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    worker_running: bool
    last_error: str | None = None


class RuntimeStatusResponse(BaseModel):
    running: bool
    processed_windows: int
    rejected_windows: int
    rejected_samples: int
    model_version: str | None = None
    diagnostic_model: bool | None = None
    last_error: str | None = None


class RuntimeActionResponse(BaseModel):
    changed: bool
    status: RuntimeStatusResponse


class PredictionEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    sequence: int
    prediction: dict[str, Any]
