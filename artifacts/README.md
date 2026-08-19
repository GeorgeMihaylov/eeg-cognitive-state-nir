# Streaming model bundles

The worker supports two bundle contracts selected by `manifest.json`:

- `input_mode: raw_eeg` loads a saved `torch_shallow_convnet` adapter from
  `model.pt` and passes cleaned windows as `[1, channels, time]`;
- `input_mode: features` preserves the handcrafted-feature path with a joblib
  estimator, scaler and optional selector.

The checked-in `*_shallow_v1` directories are device/dataset templates.  They
do not contain trained weights.  Before training, verify the ordered channel
list against the source recording metadata; this is especially important for
the OpenNeuro templates, where BIDS sidecars are authoritative.  Save the
trained adapter with `model.save(<artifact_dir>/model.pt)`.

Available raw-EEG templates:

| Bundle | Nominal input |
| --- | --- |
| `openneuro_ds007169_shallow_v1` | 19 channels, 250 Hz |
| `eegmat_shallow_v1` | Neurocom 19 channels, 500 Hz |
| `universe_muse_s_shallow_v1` | Muse S 4 channels, 256 Hz |
| `openneuro_ds007554_shallow_v1` | 32 channels, resampled to 250 Hz |
| `flight_deck_epoc_x_shallow_v1` | Emotiv EPOC X 14 channels, 128 Hz |
| `deap_biosemi_shallow_v1` | BioSemi 32 channels, original 512 Hz |
| `seed_neuroscan_shallow_v1` | Neuroscan 62 channels, original 1000 Hz |

DEAP and SEED manifests describe emotion targets rather than cognitive load.
They are useful for device/throughput validation but their labels must not be
reported as workload classes.
