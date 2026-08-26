from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from ..config import WorkerConfig
from .schemas import (
    HealthResponse,
    PredictionEnvelope,
    RuntimeActionResponse,
    RuntimeStatusResponse,
)
from .state import (
    RuntimeAlreadyRunningError,
    RuntimeStartError,
    StreamingService,
)


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
    jsonl_override = os.environ.get("COGSTATE_STREAMING_JSONL_PATH")
    if jsonl_override is not None:
        worker_config = replace(
            worker_config,
            output=replace(
                worker_config.output,
                jsonl_path=(
                    None
                    if jsonl_override.lower() == "none"
                    else jsonl_override
                ),
            ),
        )
    worker_service = service or StreamingService(worker_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if worker_config.api.autostart_worker:
            try:
                worker_service.start()
            except RuntimeStartError:
                # Keep the control plane available and expose the failure via
                # health/status so data or model mounts can be repaired.
                pass
        try:
            yield
        finally:
            worker_service.stop()

    app = FastAPI(
        title="Cogstate Streaming API",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.streaming_service = worker_service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        status = worker_service.status()
        return HealthResponse(
            status="degraded" if status.state == "failed" else "ok",
            worker_running=status.running,
            runtime_state=status.state,
            run_id=status.run_id,
            last_error=status.last_error,
        )

    @app.get("/v1/status", response_model=RuntimeStatusResponse)
    def runtime_status() -> RuntimeStatusResponse:
        return RuntimeStatusResponse(**worker_service.status().as_dict())

    @app.post("/v1/runtime/start", response_model=RuntimeActionResponse)
    def start_runtime() -> RuntimeActionResponse:
        try:
            changed = worker_service.start()
        except RuntimeAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeStartError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        snapshot = worker_service.store.snapshot()
        if snapshot.prediction is None or snapshot.run_id is None:
            raise HTTPException(status_code=404, detail="No prediction is available")
        return PredictionEnvelope(
            run_id=snapshot.run_id,
            sequence=snapshot.sequence,
            prediction=snapshot.prediction,
        )

    @app.websocket("/v1/stream")
    async def prediction_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        snapshot = worker_service.store.snapshot()
        run_id = snapshot.run_id
        sequence = snapshot.sequence
        try:
            if snapshot.prediction is not None and run_id is not None:
                await websocket.send_json(
                    PredictionEnvelope(
                        run_id=run_id,
                        sequence=sequence,
                        prediction=snapshot.prediction,
                    ).model_dump()
                )
            if snapshot.terminal is not None:
                await websocket.send_json(snapshot.terminal)
                return
            while True:
                next_snapshot = await asyncio.to_thread(
                    worker_service.store.wait_after, run_id, sequence, 1.0
                )
                if next_snapshot.run_id != run_id:
                    await websocket.send_json(
                        {
                            "type": "runtime_replaced",
                            "run_id": run_id,
                            "sequence": sequence,
                        }
                    )
                    return
                if next_snapshot.prediction is not None and run_id is not None:
                    sequence = next_snapshot.sequence
                    await websocket.send_json(
                        PredictionEnvelope(
                            run_id=run_id,
                            sequence=sequence,
                            prediction=next_snapshot.prediction,
                        ).model_dump()
                    )
                if next_snapshot.terminal is not None:
                    await websocket.send_json(next_snapshot.terminal)
                    return
                if next_snapshot.prediction is None:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "run_id": run_id,
                            "sequence": sequence,
                        }
                    )
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
