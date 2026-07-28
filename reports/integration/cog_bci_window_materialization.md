# COG-BCI raw-window materialization

Status: `diagnostic`

## Repository state

- Branch: `integration/benchmark-unification`
- Audited HEAD: `5dabc71 feat(data): add COG-BCI channel harmonization`
- Initial worktree and staging area: clean
- Source dataset: COG-BCI v4 (`zenodo-7413650-v4`)
- Source inventory: 1,044 paired EEGLAB records, 29 subjects, 3 sessions,
  87 subject-session combinations, 500 Hz, approximately 81.75 hours
- No model training, target construction, split construction, resampling,
  CAR, DANN, transfer learning or contrastive learning was performed.

## Architecture

`COGBCIWindowBuilder` extends the existing lazy `COGBCIDataset`, channel
policy and typed raw-preprocessing registry:

```text
COGBCIDataset record
→ apply explicit fixed channel policy
→ inspect source filter metadata
→ load one physical record
→ optional stateless whole-record preprocessing
→ split at internal boundary annotations
→ enumerate windows independently in each continuous segment
→ window QC and event association
→ atomic float32 NPY record shard + JSON manifest
→ global record/window/event/QC tables
→ checksum and leakage-identity audit
```

Only one physical record is materialized in memory at a time. The builder
does not concatenate records, subjects or sessions and does not create a
train/test split.

## Window contract

The implemented smoke and full technical profile uses:

| Parameter | Value |
|---|---:|
| window duration | 5.12 s |
| stride | 5.12 s |
| sampling rate | 500 Hz |
| samples per window | 2,560 |
| incomplete policy | reject final incomplete window |
| minimum valid fraction | 1.0 |
| segmentation | `record_full` |
| output dtype | `float32` |

Durations and strides are converted with `round(seconds × sampling_rate)`;
the implementation rejects any rounding error above half a sample. Other
durations and overlap are configuration parameters, not COG-BCI constants.
When incomplete retention is selected, the last fixed-shape window is
zero-padded and accepted only when its valid fraction satisfies the configured
threshold.

Every NPY shard has axis order:

```text
[window, channel, time]
```

Only accepted windows are stored in the NPY array. The global index retains
rejected candidates with `cache_offset = -1` and an explicit reason.

## Task-boundary and annotation audit

Each `.set/.fdt` pair is already task/condition-specific. All 1,044 files
contain an MNE/EEGLAB `boundary` annotation at onset zero, with a nonzero
duration spanning the record; none contains an additional internal boundary.
The builder treats an onset-zero marker as record provenance, not as an
internal discontinuity. Any future boundary strictly inside a record will
split the continuous sample range, and no window may cross it.

Observed task-marker behavior:

| Family | Records | File interpretation | Start marker | End marker | Initial strategy |
|---|---:|---|---|---|---|
| N-Back | 261 | separate zero/one/two-back file | official `600/610/620`, absent in materialized files | `601/611/621`, present | `record_full` |
| MATB-II | 261 | separate easy/medium/difficult file | no uniform marker | `MATBeasyend/MATBmedend/MATBdiffend`, present | `record_full` |
| PVT | 87 | separate PVT file | `10`, absent | `15`, present | `record_full` |
| Flanker | 87 | separate Flanker file | `20`, absent | `21`, present | `record_full` |
| Resting state | 348 | separate phase/eyes-condition file | `40/42/50/52`, absent | `41/43/51/53`, present | `record_full` |

The full event table contains 403,004 annotations. MATB dynamic annotation
strings are retained as metadata and are not converted into labels. Since
start markers are absent, exact pre-task removal is not reproducible uniformly;
the whole-file result is therefore a technical baseline, not a scientifically
final task-interval dataset. `task_interval` and `event_interval` fail
explicitly until their boundaries are validated rather than silently falling
back to whole-record segmentation.

Per-window event metadata contains:

```text
event_count_in_window
event_types_in_window
nearest_previous_event
nearest_next_event
contains_task_start
contains_task_end
```

The start/end trigger sets are part of the semantic config hash. Large event
lists are normalized once in `events.parquet`; record JSON manifests do not
duplicate them.

## Channel profiles

Two fixed-shape policies are supported:

| Policy | Channels | Cz | ECG1 | Role |
|---|---:|---|---|---|
| `cog_bci_common` | 62 | excluded | excluded | native COG-BCI spatial baseline |
| `emotiv_common` | 14 | not used | excluded | cross-dataset channel contract |

Channel order comes from the existing policy objects and is never inferred by
sorting names. Each cache stores the policy schema, stable mapping hash,
ordered channel list, count, Cz flags and auxiliary-channel exclusion.
`cog_bci_native` remains available for record-level inspection but is rejected
for a fixed batch cache because its channel count varies.

The source corpus contains physical Cz in part of the records, so final
manifests record `has_physical_cz=true`; both fixed output profiles record
`uses_cz=false`.

## Preprocessing profiles

The builder exposes the existing stateless preprocessing registry as four
profiles:

```text
none
bandpass
notch
bandpass_notch
```

Band-pass defaults to 1–45 Hz, fourth order; notch defaults to 50 Hz, Q=30.
Filters run once over the selected whole physical record before windowing.
CAR is disabled. Sampling remains 500 Hz. Source files are never modified.

MNE reports `highpass=0` and `lowpass=250` for inspected and materialized
records, but this does not prove that no processing occurred before EEGLAB
export. Provenance is therefore stored as
`unknown_eeglab_processing_history`. Any non-identity profile requires the
explicit `allow_filtering_when_source_status_unknown` opt-in to prevent an
unacknowledged double-filtering assumption. The reported smoke and full
materializations use `none`.

## Stable identities and leakage protection

`sample_id` is a SHA-256-derived identifier over:

```text
dataset
record_id
window specification
channel policy
start/stop samples
preprocessing specification hash
```

Each window also carries `subject_id`, `session_id`, `record_id` and
`record_group_id`; for this external corpus one physical COG-BCI record is one
record group. The builder never performs a random window split. Split code can
later group by subject or physical record without parsing file paths.

The full audit found:

- duplicate accepted `sample_id`: 0;
- invalid record-group assignment: 0;
- invalid start/stop bounds: 0;
- accepted records / record groups: 1,044 / 1,044;
- source fingerprint equal to the verified record index: yes;
- absolute paths in record/global manifests: none.

The two channel policies intentionally produce disjoint `sample_id` sets,
because channel policy is part of the identity contract.

## QC

Per-window QC records:

```text
valid_sample_fraction
NaN and Inf flags
constant-channel count
near-zero-variance-channel count
absolute_max
absolute_mean
status
rejection_reason
```

The public status vocabulary is:

```text
accepted
rejected_nonfinite
rejected_constant
rejected_incomplete
rejected_invalid_range
```

The final full cache contains 57,947 window candidates: 56,903 accepted and
1,044 rejected incomplete tails (one per physical record). It contains zero
NaN windows, zero Inf windows and zero windows with a constant selected
channel. Threshold-independent measurements remain available in
`qc_windows.parquet`; rejection thresholds are serialized in the semantic
window specification.

## Cache schema and recovery

Top-level provenance includes schema/dataset/builder versions, source-root
fingerprint, commit, config hash, channel contract, event contract,
preprocessing, sampling rate, window/stride and dtype.

Each record JSON includes relative input/output paths, record identity,
task metadata, accepted/rejected counts, array shape, SHA-256 checksum,
source-record fingerprint, reader filter metadata and config hash. It also
repeats dtype, channel order, sampling rate and window sample count for the
corresponding NPY array.

Writes use a temporary suffix and atomic replacement. A temporary file is not
accepted as a completed shard. Supported operational modes are:

```text
--resume
--overwrite
--verify-only
--subjects
--sessions
--task-families
--task-variants
--max-records
--one-per-subject-family
```

Resume skips only a shard whose config hash, source fingerprint, checksum,
shape and dtype all match. Incompatible config/source/checksum fails clearly;
replacement requires `--overwrite`. `verify-only` writes nothing.

## Synthetic and compatibility tests

New tests cover non-overlap and overlap counts, continuous-segment boundaries,
incomplete-window policies, stable semantic IDs, both channel shapes,
float32 conversion, source preservation, NaN/Inf/constant-channel QC,
event metadata, MATB end markers, no-op/filter execution counts,
unknown-filter-history protection, atomic paths, resume, overwrite,
checksum/config/source invalidation and the leakage audit.

Related COG dataset, channel harmonization, shared preprocessing and legacy
raw-window tests were also run. The final full repository result was:

```text
881 passed, 12 warnings
```

Warnings were the existing pytest `cache_dir` option warning and existing
sklearn metric warnings; there were no failures.

## Real-data smoke

Selection: `sub-01` and `sub-10`, one deterministic record from each of
N-Back, MATB-II, PVT, Flanker and resting state (10 physical records total).

| Policy | Shape per accepted window | Candidates | Accepted | Rejected tails | Size |
|---|---|---:|---:|---:|---:|
| `emotiv_common` | `[14, 2560]` | 782 | 772 | 10 | 0.103 GiB |
| `cog_bci_common` | `[62, 2560]` | 782 | 772 | 10 | 0.457 GiB |

Both runs had finite values, correct fixed order, no ECG1, no constant
channels, no duplicate IDs and no invalid boundaries. Resume skipped all 10
valid shards; `verify-only` validated all checksums.

Accepted smoke windows by family were: Flanker 236, MATB-II 116, N-Back 147,
PVT 251 and resting state 22.

## Full technical materialization

Before launch, the estimated NPY payload was:

| Policy | Estimated accepted windows | Estimated NPY size | Materialized |
|---|---:|---:|---|
| `emotiv_common` | 56,903 | 7.597 GiB | yes |
| `cog_bci_common` | 56,903 | 33.645 GiB | no |

Available free space before launch was approximately 575.8 GiB. The full
`emotiv_common` run completed in approximately 4.8 minutes:

| Measure | Result |
|---|---:|
| physical record shards | 1,044 |
| subjects | 29 |
| sessions | 3 |
| candidates | 57,947 |
| accepted | 56,903 |
| rejected incomplete | 1,044 |
| events | 403,004 |
| total cache size | 7.626 GiB |
| checksum-verified shards | 1,044 |
| errors | 0 |

Accepted windows by family:

| Family | Records | Windows |
|---|---:|---:|
| Flanker | 87 | 10,246 |
| MATB-II | 261 | 15,138 |
| N-Back | 261 | 16,927 |
| PVT | 87 | 10,764 |
| Resting state | 348 | 3,828 |

Final config hash:

```text
4f60a6c7cd9d0dd6613a9338834691d9ee289a749a08551045502aef4da80d72
```

## Runtime artifacts

All generated artifacts are ignored under:

```text
benchmark_results/cog_bci_windows/
├── smoke_emotiv_common/
├── smoke_cog_bci_common/
└── emotiv_common_full/
```

Each cache contains:

```text
dataset_manifest.json
record_manifest.parquet
window_index.parquet
events.parquet
qc_summary.json
qc_windows.parquet
leakage_audit.json
cache_report.md
errors.csv
shards/*.npy
shards/*.json
```

## Commands

Smoke:

```powershell
python scripts\data\cog_bci_window_cache.py `
  --config experiments\cog_bci\window_cache_smoke.json
```

The 62-channel smoke uses the same config plus:

```powershell
--channel-policy cog_bci_common `
--output-dir benchmark_results\cog_bci_windows\smoke_cog_bci_common
```

Full 14-channel materialization:

```powershell
python scripts\data\cog_bci_window_cache.py `
  --output-dir benchmark_results\cog_bci_windows\emotiv_common_full `
  --channel-policy emotiv_common `
  --window-duration-seconds 5.12 `
  --window-stride-seconds 5.12 `
  --preprocessing none `
  --resume
```

Verification replaces `--resume` with `--verify-only`.

## Limitations and readiness

The full 14-channel cache is a validated technical substrate for a later
cross-dataset encoder/DANN experiment, but no domain definition, compatible
scientific target or evaluation protocol has been selected. It must not be
reported as transfer-learning evidence.

The 62-channel path is validated by real-data smoke but was not fully
materialized because its estimated payload is 33.6 GiB. A native COG-BCI task
still requires a justified target, exact task-interval policy (or an explicit
whole-file scientific justification), split protocol and metric contract.

The next stage should define one scientific COG-BCI task and its leakage-safe
subject/record split before any model training. For cross-dataset work, it
should first define a defensible shared target/domain question; only then
should the existing shared encoder/DANN infrastructure consume the verified
14-channel cache.
