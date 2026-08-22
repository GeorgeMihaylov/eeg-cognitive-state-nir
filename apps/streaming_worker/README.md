# Streaming worker

The worker is the runnable composition layer around the reusable `cogstate`
library.  Its primary path is:

```text
source -> causal sample filter -> ring buffer -> quality gate
       -> window artifact handling -> model input adapter -> model bundle
       -> causal prediction filter -> sinks
```

The model manifest selects one of two inference paths:

```text
input_mode: raw_eeg   -> [1, channels, time] -> ShallowConvNet (.pt)
input_mode: features  -> feature pipeline -> scaler/selector -> model (.joblib)
```

Both modes share exactly the same streaming source, causal filtering, windowing,
quality checks, optional MNE-FASTER handling, postprocessing and outputs. A raw-EEG manifest
also records the preprocessing contract and the worker rejects configurations
that differ from training.

Run a NumPy replay (`[samples, channels]`) with:

```powershell
python -m apps.streaming_worker --config configs/streaming.yaml
```

The default manifest creates an untrained ShallowConvNet at startup to validate
the raw-window path only: every output contains `diagnostic_model: true` and its
predictions have no scientific meaning.  Replace it with a trained bundle
containing `model.pt`.  The older feature-mode diagnostic bundle remains at
`artifacts/pm_model_v1` and feature-mode bundles can still contain
`model.joblib`, `scaler.joblib` and an optional `selector.joblib`.

LSL is optional and imported only for `source.type: lsl`; install `pylsl` in
the runtime environment before using that source.

MNE-FASTER is also optional. The default configuration disables it. To use a
calibration bundle, install `requirements-streaming-mne-faster.txt`, set
`preprocessing.mne_faster_enabled: true`, and provide
`preprocessing.mne_faster_bundle_dir`. The worker never fits ICA on live data;
it loads the fixed ICA and bad-channel decisions produced during calibration.

`features.profile` is used only by manifests with `input_mode: features`.
`lightweight` provides spectral/statistical features; `full` also enables the
more expensive feature pipeline.  ShallowConvNet bypasses both profiles and
consumes the cleaned EEG window directly.

Device-specific ShallowConvNet contracts live under `artifacts/*_shallow_v1`.
They are templates without weights: channel order must be checked against the
actual dataset metadata before training, then the matching `model.pt` is placed
beside the manifest.

## Optional FastAPI service

Install `requirements-streaming-api.txt`, then run:

```powershell
python -m apps.streaming_worker.api --config configs/streaming.yaml
```

The API is a separate layer over the same runtime and exposes health, status,
latest prediction, worker lifecycle and a WebSocket stream.  See `api/README.md`.

## Docker

Build and run the FastAPI layer from the repository root:

```powershell
docker build -t cogstate-streaming:local .
docker run --rm -p 8000:8000 cogstate-streaming:local
```

The image runs as the unprivileged `cogstate` user and exposes port `8000`.
`/health` is used as the container liveness check. The default replay source
expects `data/replay_eeg.npy`, which is intentionally not baked into the image.
Mount replay data and a deployment configuration when needed:

```powershell
docker run --rm -p 8000:8000 `
  -v ${PWD}/data:/app/data:ro `
  -v ${PWD}/configs/streaming.yaml:/app/configs/streaming.yaml:ro `
  -v ${PWD}/outputs:/app/outputs `
  cogstate-streaming:local
```

Files present under `artifacts/` at build time are copied into the image. A
trained bundle may instead be mounted read-only over its configured artifact
directory, which allows replacing a model without rebuilding the application
image.

For an LSL source, add `pylsl` to the image dependencies and ensure that the
container networking can discover the LSL stream. LSL is not installed in the
base image because replay mode does not need it.
