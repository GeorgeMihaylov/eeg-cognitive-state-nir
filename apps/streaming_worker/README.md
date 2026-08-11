# Streaming worker

The worker is the runnable composition layer around the reusable `cogstate`
library.  Its primary path is:

```text
source -> causal sample filter -> ring buffer -> quality gate
       -> window artifact handling -> lightweight features -> model bundle
       -> causal prediction filter -> sinks
```

Run a NumPy replay (`[samples, channels]`) with:

```powershell
python -m apps.streaming_worker --config configs/streaming.yaml
```

The checked-in manifest enables a real logistic-regression model from
`cogstate.model_zoo`, fitted on synthetic anchors at startup.  It validates the
pipeline only: every output contains `diagnostic_model: true`.  Scientific use
requires replacing it with a trained bundle containing `model.joblib`, its
manifest and the same scaler/selector used during training.

LSL is optional and imported only for `source.type: lsl`; install `pylsl` in
the runtime environment before using that source.

`features.profile: lightweight` uses spectral and statistical features and is
the real-time default.  The existing entropy/connectivity-heavy pipeline is
available as `features.profile: full`, but its latency must fit the configured
window step.  A trained model bundle is tied to exactly one feature profile.

## Optional FastAPI service

Install `requirements-streaming-api.txt`, then run:

```powershell
python -m apps.streaming_worker.api --config configs/streaming.yaml
```

The API is a separate layer over the same runtime and exposes health, status,
latest prediction, worker lifecycle and a WebSocket stream.  See `api/README.md`.
