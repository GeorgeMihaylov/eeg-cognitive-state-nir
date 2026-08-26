# FastAPI layer

The API is an optional transport around `StreamingRuntime`; the signal
processing library and the standalone worker do not import FastAPI.

Run locally:

```powershell
python -m apps.streaming_worker.api --config configs/streaming.yaml
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Endpoints:

- `GET /health` — API/worker health;
- `GET /v1/status` — counters and active model metadata;
- `GET /v1/predictions/latest` — most recent complete result;
- `POST /v1/runtime/start` — create and start a worker;
- `POST /v1/runtime/stop` — stop the active worker;
- `WS /v1/stream` — prediction envelopes and idle heartbeats.

Every worker start creates a new `run_id` and clears the previous latest
prediction. Runtime status distinguishes `idle`, `starting`, `running`,
`stopping`, `completed`, `stopped` and `failed`. It also exposes the active
model's `model_type`, `input_mode`, `class_names` and optional multitask target
names.

WebSocket prediction messages use `type: prediction` and carry the same
typed streaming result as `GET /v1/predictions/latest`. A finite replay or a
stopped/failed runtime ends the stream with one of:

- `runtime_completed`;
- `runtime_stopped`;
- `runtime_failed`.

Starting an already running worker returns HTTP 409. A model, manifest or
source initialization failure returns HTTP 503 while the API remains available
and reports the failure through `/health` and `/v1/status`.

The default host is loopback-only.  Runtime control endpoints do not implement
authentication and must not be exposed to an untrusted network as-is.

If replay data or LSL is unavailable during autostart, FastAPI remains online,
`/health` reports `degraded`, and `/v1/status` contains `last_error`.
