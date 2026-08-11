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

The default host is loopback-only.  Runtime control endpoints do not implement
authentication and must not be exposed to an untrusted network as-is.

If replay data or LSL is unavailable during autostart, FastAPI remains online,
`/health` reports `degraded`, and `/v1/status` contains `last_error`.
