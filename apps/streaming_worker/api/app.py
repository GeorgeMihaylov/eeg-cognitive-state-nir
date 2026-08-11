from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from ..config import WorkerConfig
from .schemas import (
    HealthResponse,
    PredictionEnvelope,
    RuntimeActionResponse,
    RuntimeStatusResponse,
)
from .state import StreamingService


def create_app(
    *,
    config: WorkerConfig | None = None,
    config_path: str | Path = "configs/streaming.yaml",
    service: StreamingService | None = None,
) -> FastAPI:
    resolved_config_path = os.environ.get(
        "COGSTATE_STREAMING_CONFIG", str(config_path)
    )
    worker_config = config or WorkerConfig.from_yaml(resolved_config_path)
    worker_service = service or StreamingService(worker_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if worker_config.api.autostart_worker:
            worker_service.start()
        try:
            yield
        finally:
            worker_service.stop()

    app = FastAPI(
        title="Cogstate Streaming API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.streaming_service = worker_service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        status = worker_service.status()
        return HealthResponse(
            status="degraded" if status.last_error else "ok",
            worker_running=status.running,
            last_error=status.last_error,
        )

    @app.get("/v1/status", response_model=RuntimeStatusResponse)
    def runtime_status() -> RuntimeStatusResponse:
        return RuntimeStatusResponse(**worker_service.status().as_dict())

    @app.post("/v1/runtime/start", response_model=RuntimeActionResponse)
    def start_runtime() -> RuntimeActionResponse:
        try:
            changed = worker_service.start()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return RuntimeActionResponse(
            changed=changed,
            status=RuntimeStatusResponse(**worker_service.status().as_dict()),
        )

    @app.post("/v1/runtime/stop", response_model=RuntimeActionResponse)
    def stop_runtime() -> RuntimeActionResponse:
        changed = worker_service.stop()
        return RuntimeActionResponse(
            changed=changed,
            status=RuntimeStatusResponse(**worker_service.status().as_dict()),
        )

    @app.get("/v1/predictions/latest", response_model=PredictionEnvelope)
    def latest_prediction() -> PredictionEnvelope:
        sequence, prediction = worker_service.store.snapshot()
        if prediction is None:
            raise HTTPException(status_code=404, detail="No prediction is available")
        return PredictionEnvelope(sequence=sequence, prediction=prediction)

    @app.websocket("/v1/stream")
    async def prediction_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        sequence, prediction = worker_service.store.snapshot()
        try:
            if prediction is not None:
                await websocket.send_json(
                    PredictionEnvelope(
                        sequence=sequence, prediction=prediction
                    ).model_dump()
                )
            while True:
                next_sequence, next_prediction = await asyncio.to_thread(
                    worker_service.store.wait_after, sequence, 1.0
                )
                if next_prediction is None:
                    await websocket.send_json(
                        {"type": "heartbeat", "sequence": sequence}
                    )
                    continue
                sequence = next_sequence
                await websocket.send_json(
                    PredictionEnvelope(
                        sequence=sequence, prediction=next_prediction
                    ).model_dump()
                )
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
