# COG-BCI record-level lazy loader

## Scope and repository state

The implementation was developed on branch
`integration/benchmark-unification` from `HEAD 3e36cfc`. The working tree was
clean at the start.

This stage provides dataset discovery and lazy record access only. It does
not create EEG windows, harmonize channels, define a scientific target or
run training.

## Existing architecture contracts

The following contracts were inspected before implementation:

- `BaseDataset.load() -> EEGData` represents materialized feature rows or
  windows and their labels;
- `BaseEEGDataset`, `EmotivDataset` and `WESADDataset` follow that
  materialized contract;
- `RawEEGWindowDataset` consumes an already built window manifest/cache and
  returns a lazy array view of fixed windows inside `EEGData`;
- `DATASET_REGISTRY` is the single dataset registry;
- `BenchmarkRunner` immediately calls `load()` and requires a scientific
  task, so it cannot yet consume an unwindowed 81.75-hour record collection.

Forcing COG-BCI into `load() -> EEGData` would either materialize the complete
dataset or return an object with false row/target semantics. A new
`BaseRecordDataset` contract was therefore added beside `BaseDataset`.
`COGBCIDataset` is registered in the existing registry, while
`BenchmarkRunner` rejects record-level datasets with an explicit explanation
until a window-materialization stage exists. No second registry and no fake
`load()` were introduced.

## Components

Production code lives in `bench/datasets/cog_bci_dataset.py`:

- `COGBCIRecord` is immutable typed metadata for one `.set/.fdt` pair;
- `COGBCIRecordIndex` discovers, validates, serializes and restores records;
- `COGBCIDataset` provides cache management, filtering, lookup and lazy MNE
  access.

The inventory script remains an independent raw-data management tool and is
not used as the loader's source of truth.

## Record identity and metadata

A record contains:

- stable `record_id`;
- canonical and raw subject/session/task identifiers;
- task family, variant and condition;
- relative `.set` and `.fdt` paths and sizes;
- sampling frequency, sample count and duration;
- total, EEG and auxiliary channel names and counts;
- MNE channel types;
- `has_cz`, `has_ecg1` and channel-layout identity;
- event count and unique annotation values;
- reader, reference, montage status, missing positions and units.

An example identifier is:

```text
cog_bci::sub-01::ses-01::flanker::run-na
```

It is derived only from normalized source identifiers and is independent of
filesystem traversal order. Serialized paths are POSIX-style and relative to
the configured extracted root.

## Normalization

Subjects are normalized to `sub-01` through `sub-29`. Raw session labels such
as `ses-S1` are retained while canonical sessions are `ses-01`, `ses-02` and
`ses-03`.

The explicit task map is:

| Raw label | Family | Variant |
|---|---|---|
| `zeroBACK` | `n_back` | `zero_back` |
| `oneBACK` | `n_back` | `one_back` |
| `twoBACK` | `n_back` | `two_back` |
| `MATBeasy` | `matb` | `matb_easy` |
| `MATBmed` | `matb` | `matb_medium` |
| `MATBdiff` | `matb` | `matb_difficult` |
| `PVT` | `pvt` | `pvt` |
| `Flanker` | `flanker` | `flanker` |
| four `RS_*` names | `resting_state` | begin/end × eyes open/closed |

Unknown subjects, sessions and task labels are rejected rather than guessed.
No workload or other target class is created.

## Discovery and index cache

Discovery recursively scans actual `.set` and `.fdt` files, groups them by
relative parent and case-normalized stem, and requires exactly one of each.
It rejects incomplete pairs, duplicate paths and duplicate `record_id`
values. Records are sorted by `record_id`.

Cache schema 1 contains:

- schema and dataset versions;
- deterministic UTC inventory timestamp derived from the newest source-pair
  modification time;
- source-root fingerprint;
- record count;
- sorted serialized records.

The fingerprint hashes relative pair paths, sizes and modification times
without reading signal payloads. Cache loading recomputes it, checks every
path, and rejects changed roots, incompatible schema versions and inconsistent
record counts. The cache contains no EEG arrays or absolute paths.

The diagnostic cache is under
`benchmark_results/cog_bci_loader/record_index.json`; it is ignored runtime
data. It contains 1,044 records and is approximately 20 MB because annotation
type metadata is retained.

## Filtering and lazy access

`query()` combines filters for:

- subject;
- canonical or raw-style session identifier;
- task family;
- task variant;
- `has_cz`;
- channel-layout ID.

Unknown filter values produce an error that lists available values. Results
remain deterministically sorted. `get_record()` performs exact lookup and
`iter_records()` provides an iterator over a query.

`open_raw(record_id, preload=False)` opens only the selected EEGLAB pair.
Auxiliary channels are excluded from the returned raw object by default but
remain in record metadata and can be retained explicitly. No resampling,
filtering, referencing, montage application, interpolation or array-wide
loading is performed.

MNE warnings are left visible during user-facing `open_raw()` calls. Header
indexing suppresses repetitive MNE informational output for 1,044 files but
records explicit montage and missing-position diagnostics.

## Channels, reference and montage

The real index confirms:

| Native layout | Records | EEG | Auxiliary | Cz |
|---|---:|---:|---:|---|
| 63 total | 324 | 62 | 1 (`ECG1`) | absent |
| 64 total | 720 | 63 | 1 (`ECG1`) | present |

MNE identifies `ECG1` as type `ecg`; name-based classification is retained as
a defensive fallback. `ECG1` is never included in `eeg_channel_names`.
`Cz` is preserved for participants 10–29 and is neither removed nor
interpolated for participants 1–9.

Every record has complete scalp locations for EEG channels. Only `ECG1`
lacks a scalp position, so every record has montage status
`auxiliary_missing_only`. MNE reports
`FIFFV_MNE_CUSTOM_REF_OFF`; no reference transformation is applied. Original
units are not exposed by the reader and remain unavailable.

## Dependency

MNE 1.12.1 is installed in the current analysis environment. The repository
has no tracked `requirements.txt`, `environment.yml`, `pyproject.toml` or
equivalent dependency manifest, so no parallel dependency file was created.
README documents the optional raw-EEG dependency as:

```text
mne>=1.12,<2
```

If MNE is missing, the loader raises an explicit installation instruction.

## Real-data diagnostic smoke-run

The index was rebuilt from the extracted files without consulting the
inventory tables:

```text
records: 1,044
subjects: 29
canonical sessions: 3
subject-session units: 87
complete pairs: 1,044
sampling rates: [500.0]
channel layouts: 2
```

The cache was then restored without rereading EEGLAB headers. Queries returned
36 records for `sub-01`, 348 records for `ses-03`, and 261 N-Back records.

Five records from `sub-01` were opened, covering N-Back, MATB, PVT, Flanker
and resting state. Every raw object had `preload=False`, finite duration,
available annotations, and exact agreement with indexed channel count,
sample count and sampling frequency. Separate Flanker records confirmed
62 EEG + `ECG1` without `Cz` for `sub-01`, and 63 EEG + `ECG1` with `Cz` for
`sub-10`.

MNE emitted expected warnings for boundary annotations, annotations extending
slightly beyond the data range, and the non-scalp ECG channel. These warnings
were not suppressed by the user-facing lazy reader.

The smoke artifact has status `diagnostic` and is stored in
`benchmark_results/cog_bci_loader/smoke_summary.json`.

## Validation

- changed Python modules compile successfully;
- loader unit tests: 26 passed;
- loader, runner, config and existing raw-window integration tests:
  69 passed;
- complete `tests` directory: 834 passed, 12 warnings.
- complete repository invocation (`python -m pytest -q`): 834 passed,
  12 warnings.

The warnings are the existing pytest `cache_dir` warning, expected
small-synthetic-partition metric warnings, and the documented MNE warnings in
the real-data smoke-run.

## Readiness boundary and next stage

```text
dataset discovery and lazy record access: ready
window materialization: not implemented
scientific task: not implemented
training: not implemented
```

The next task must explicitly decide:

1. native common-channel policy for the 62/63 EEG layouts;
2. whether analyses exclude `Cz` globally or use a missing-channel-aware
   representation;
3. record-safe window boundaries around MNE `boundary` annotations;
4. window length, overlap and event alignment;
5. whether and when resampling, filtering and referencing occur;
6. task-specific targets and leakage-safe cross-subject/cross-session splits.

No processed dataset, window cache, target, model or training experiment was
created in this stage.
