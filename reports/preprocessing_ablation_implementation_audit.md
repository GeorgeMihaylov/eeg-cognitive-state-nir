# Preprocessing ablation implementation audit

Этот отчёт фиксирует исторический путь на дату аудита. Текущий cache-builder
CLI: `scripts/data/build_raw_eeg_window_cache.py`; каталога `src` в
актуальной архитектуре нет.

Audit date: 2026-07-16  
Branch: `feature/preprocessing-ablation`  
Baseline tests: `80 passed, 1 warning in 5.97s`

## Current preprocessing call graph

```text
scripts/data/build_raw_eeg_window_cache.py
  -> YAML raw_preprocessing (or direct mapping)
  -> normalize_raw_preprocessing()
  -> build_raw_window_index()
  -> build_raw_eeg_cache()
       -> raw_preprocessing_hash()             # namespace identity
       -> _cache_config_hash()                 # per-record shard identity
       -> load_raw_eeg_record()
       -> extract_raw_eeg_window()
            -> regularize timestamp grid
            -> resample_poly()
            -> apply_raw_preprocessing()
                 -> Butterworth band-pass
                 -> IIR notch
                 -> common-average reference
            -> trim padded interval
       -> .npy shard + .json shard metadata
       -> raw window index Parquet

cli.py -> load_config() -> BenchmarkRunner(config)
  -> BenchmarkRunner.load_dataset()
       -> copies top-level raw_preprocessing into dataset config
       -> RawEEGWindowDataset.load()
            -> validates manifest preprocessing_hash
            -> RawEEGWindowArrayView over cache_file/cache_offset
  -> CrossValidator.run_group_kfold()
  -> BenchmarkRunner._evaluate_group_kfold()
  -> BenchmarkRunner._save_split_artifacts()
```

The benchmark consumes a raw-window **index Parquet** through
`datasets.emotiv_raw_eeg.data_path`; it does not receive a cache directory
directly. Each accepted index row contains `cache_file` and `cache_offset`, and
`RawEEGWindowDataset` exposes the files through a lazy memory-mapped view.

## Actual operation order

The order below comes from `extract_raw_eeg_window()` and
`apply_raw_preprocessing()`, not from YAML names.

1. Normalize and validate the preprocessing mapping and target sample rate.
2. Resolve the exact 10-second window bounds.
3. If band-pass or notch is enabled, expand the source interval by 2 seconds on
   each side. CAR alone does not enable padding.
4. Select timestamped source samples, collapse duplicate timestamps, and check
   missingness against the unpadded 10-second source grid.
5. Interpolate the padded interval onto a regular source-rate grid as float32.
6. Resample the padded interval to 256 Hz with `resample_poly` when necessary;
   edge-pad or truncate to the expected padded sample count.
7. Convert to float64 working precision inside `apply_raw_preprocessing()`.
8. Apply the fourth-order Butterworth 1–45 Hz band-pass using zero-phase
   `sosfiltfilt` when enabled.
9. Apply the 50 Hz, Q=30 notch using zero-phase `filtfilt` when enabled.
10. Apply common-average reference when enabled.
11. Convert the filtered padded signal to contiguous float32.
12. Remove the 2-second margins and enforce `[14, 2560]`.
13. Convert the final window to contiguous float32 and reject NaN/Inf.

Thus the effective order is **padding -> resampling -> band-pass -> notch ->
CAR -> trimming -> final float32**. Float32 is also used for the regularized and
resampled buffers before the filtering function temporarily promotes to
float64.

## Current configuration sources

- Cache builder defaults live in `DEFAULT_RAW_PREPROCESSING` and CLI defaults in
  `scripts/data/build_raw_eeg_window_cache.py`.
- `--preprocessing-config` accepts either a document containing
  `raw_preprocessing` or the schema mapping directly.
- The benchmark accepts top-level `raw_preprocessing`; `BenchmarkRunner` copies
  it into the selected dataset config unless the dataset has its own value.
- `RawEEGWindowDataset` normalizes that value and compares its recomputed
  preprocessing hash with hashes stored in the accepted manifest rows.
- Current public keys are `resample_hz`, `bandpass.{enabled,low_hz,high_hz}`,
  `notch.{enabled,frequency_hz,quality_factor}`, `rereference.mode`, and artifact
  rejection settings.
- Butterworth order 4, zero-phase mode, two-second filter padding, and output
  float32 are implementation constants rather than configurable schema fields.

## Current cache hash inputs

There are two different hashes.

`raw_preprocessing_hash()` controls the cache namespace and manifest-level
compatibility. Its canonical JSON payload contains:

- `RAW_PREPROCESSING_VERSION = raw-preprocessing-v1`;
- ordered channel names;
- the fully normalized raw preprocessing mapping, including resample rate,
  enabled flags, frequency values, rereference mode, and artifact-rejection
  settings.

JSON keys are sorted and compactly serialized before SHA-256, so dict insertion
order does not affect this hash.

`_cache_config_hash()` validates each record shard. It additionally contains:

- `RAW_LOADER_VERSION`;
- record ID;
- resolved absolute source path, file size, and nanosecond mtime;
- channel order, target sample rate, and maximum missing fraction;
- the normalized preprocessing mapping;
- every requested `[sample_id, t_start, t_end]` window.

The namespace name is `<variant>-<first 16 preprocessing hash characters>`.
Each `.json` shard stores its full `config_hash`; `_valid_cache_shard()` also
checks `.npy` dtype and shape and samples the first array for finite values.

Parameters that are not explicit fields in the semantic preprocessing hash are
the 2-second padding, filter order, zero-phase flag, output dtype, window length,
and loader schema. Padding/filter implementation/dtype are only indirectly
versioned by code constants. Window boundaries and loader version do enter the
per-record shard hash. Runtime diagnostics such as original sample rate,
accepted window count, cache offsets, rejection reasons, missing fraction, and
amplitude/flatness statistics are metadata outputs and are not hash inputs.

## Existing cache inventory

| Cache | Version | Pairs | Accepted / rejected index rows | Size | Semantic status |
|---|---|---:|---:|---:|---|
| `data/interim/raw_eeg_cache_w10` | v2 | 119 | 45,326 / 58 | 6,507,039,688 B | physically complete; preprocessing identity absent |
| `raw_eeg_cache_w10_v3/raw-2251ca950a467267` | v3 | 119 | 45,326 / 58 | 6,511,440,533 B | exact identity/A match; reusable |
| `raw_eeg_cache_w10_v3/raw-bp-notch-car-445be3721678be51` | v3 | 119 | 45,326 / 58 | 6,510,754,301 B | exact full/H match; reusable |

All 357 JSON/NPY pairs exist and all inspected arrays have the expected float32
dtype and metadata-derived first dimension. The raw and full v3 indices have
identical row count, `sample_id`, record/subject identity, time bounds, fold,
target, status, and accepted sample set. Deduplication occurs later in the
dataset loader; the established model dataset retains 30,958 logical-record
windows.

The v2 cache has no `raw_preprocessing`, `preprocessing_hash`, or variant in its
manifests/index. Although it predates filtering and is plausibly raw, semantic
identity cannot be proven from its own metadata; because a verified v3 raw
cache already exists, automatic ablation reuse should prefer v3 and reject v2.

## Legacy YAML -> semantic pipeline mapping

| YAML | Factorial ID | Variant | Recomputed hash | Existing compatible v3 cache |
|---|---|---|---|---|
| `raw_preprocessing_a_raw.yaml` | A | identity/raw | `2251ca950a467267dcccc1c5b83157f26e02768f46c6073d33f5dc16225bda84` | yes |
| `raw_preprocessing_b_bandpass.yaml` | B | band-pass | `9f1fb83cec499628e53e1bad6a0b31ebb3db82c08f5da847dcbae11bcddcf716` | no |
| `raw_preprocessing_c_bandpass_notch.yaml` | E | band-pass + notch | `54278ee46868b00e2834f7da71f2f0cb05b63a88c3990056421b02ea4379d815` | no |
| `raw_preprocessing_d_bandpass_notch_car.yaml` | H | band-pass + notch + CAR | `445be3721678be517a650e93cc43c0eb0267f8eb54bbf4a9cd05fda0323f236e` | yes |

The current `preprocessing_variant_name()` starts with `raw` and independently
adds `bp`, `notch`, `car`, and `artifact-qc`. It therefore already distinguishes
all eight factor combinations; the four legacy YAML files merely cover A, B, E,
and H.

## Reusable benchmark entry points

- `normalize_raw_preprocessing()`, `apply_raw_preprocessing()`,
  `raw_preprocessing_hash()`, and `preprocessing_variant_name()` preserve the
  validated signal-processing implementation.
- `build_raw_window_index()` and `build_raw_eeg_cache()` can build/reuse a cache
  directly in-process.
- `RawEEGWindowDataset` provides lazy cache loading, logical-record
  deduplication, `max_windows`, and manifest/hash validation.
- `BenchmarkRunner(config).run()` executes one trial programmatically; a
  subprocess is not required.
- `CrossValidator.run_group_kfold()` and the runner preserve precomputed outer
  folds and leakage checks.
- Existing artifact saving already writes fold metrics, predictions, models,
  training logs, validation splits, normalization, preprocessing metadata,
  logical-record selection, rejected windows, unified predictions, root JSON,
  and summary CSV.

The output root is the configured `output_dir`. `BenchmarkRunner` adds a
second-resolution timestamp, followed by dataset/task/model and then
`group_kfold_subject/fold_NN`. Root benchmark JSON and summary CSV live directly
under `output_dir`; resolved config and run metrics live in its timestamped run
directory.

## Required minimal code changes

1. Extend `raw_preprocessing.py` with immutable step/pipeline specifications and
   a registry that delegates execution to the existing filtering function.
2. Keep the current legacy schema adapter and hashes unchanged for cache
   compatibility; add a separate stable semantic serialization/hash for the
   richer specification.
3. Add a small experiment module that expands the Cartesian product, resolves
   semantic cache matches, estimates missing-cache storage, records neutral
   trial state, invokes `build_raw_eeg_cache()` and `BenchmarkRunner` in-process,
   and supports resume.
4. Add one declarative experiment matrix and one thin CLI script.
5. Add focused registry/hash/signal/cache/plan/resume/leakage tests.

No change to `bench_runner.py` or the raw cache builder is required for the
initial implementation. Fold limiting can be implemented in the experiment
layer while preserving the runner's public API; full runs continue to use all
five folds.

## Compatibility risks

- Replacing the legacy hash would orphan both verified v3 caches. The new
  semantic hash must coexist with, not silently redefine, the legacy hash.
- The requested explicit `bandpass.order` and `padding_seconds` are absent from
  the legacy schema. The adapter must validate them against the fixed current
  values rather than pass unsupported values into existing functions.
- Cache namespace hashes alone do not identify source files or window layout;
  shard validation and manifest completeness remain mandatory.
- Absolute raw paths participate in record-shard hashes, so moving the checkout
  can prevent byte-level shard reuse even when preprocessing is semantically
  identical. Semantic lookup must inspect metadata but must not bypass source
  identity and shard-integrity checks.
- CAR-only uses no filter padding in current code. The specification must
  preserve that behavior rather than applying unconditional padding.
- Stateful future steps must be rejected by the global cache builder unless
  `fit_scope=stateless`; otherwise preprocessing before the outer split could
  leak test information.
- Existing cache contents and legacy YAML/config resolution must remain
  read-compatible throughout the ablation work.

