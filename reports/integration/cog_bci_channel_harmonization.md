# COG-BCI ↔ Emotiv channel harmonization contract

## Scope and repository state

Task 8Ж was implemented on branch `integration/benchmark-unification` from
`HEAD 04327e4`. The working tree and staging area were clean at the start.
The uncommitted changes in this task are therefore separate from the committed
record-loader work in task 8Е.

This is a technical channel audit and deterministic selection contract. It is
not a scientific validation of cross-dataset transfer. No signal filtering,
resampling, rereferencing, interpolation, windowing, target construction or
model training was performed.

## Inspected contracts

The audit traced channel order through:

- the `gpn_data` and `Old_EEG` record catalog;
- `load_raw_eeg_record()` and `build_raw_eeg_cache()`;
- record-shard JSON metadata and `.npy` tensor shapes;
- `RawEEGWindowDataset`;
- EEGNet and ShallowConvNet raw-input configs;
- `COGBCIRecord`, its cached record index and lazy MNE reader.

The project raw tensor convention is `[window, channel, time]`; model adapters
add the singleton spatial input dimension and consume
`[batch, 1, channel, time]`.

## Project Emotiv channel contract

The single production source of truth is:

```text
bench/datasets/channel_contracts.py::PROJECT_EMOTIV_CHANNEL_ORDER
```

Its exact order is:

```text
EEG.AF3
EEG.F7
EEG.F3
EEG.FC5
EEG.T7
EEG.P7
EEG.O1
EEG.O2
EEG.P8
EEG.T8
EEG.FC6
EEG.F4
EEG.F8
EEG.AF4
```

`raw_eeg_window_dataset.CANONICAL_EEG_CHANNELS` remains as a backward-
compatible alias to this tuple. The raw loader passes the tuple explicitly to
`pandas.read_csv(usecols=...)` and indexes the resulting frame in that same
order; it does not derive tensor order by sorting source columns.

The catalog contains 71 `gpn_data` and 49 `Old_EEG` records. Both sources have
the same ordered set of 20 `EEG.*` fields. Fourteen are signal channels in the
contract above. The other fields (`EEG.Counter`, `EEG.Interpolated`,
`EEG.RawCq`, battery fields and `EEG.MarkerHardware`) are device/support
fields and are not raw model channels.

All 119 supervised raw-cache shard JSON files in the canonical raw namespace
contain the same explicit 14-channel manifest in production order. Their
`.npy` channel axis is therefore recoverable without relying on filename or
column sorting. Existing caches were inspected but not rebuilt.

## COG-BCI layouts

The cached record index was audited across all 1,044 records:

| Layout | Records | EEG | Auxiliary | Cz |
|---|---:|---:|---:|---|
| 63 total | 324 | 62 | `ECG1` | absent |
| 64 total | 720 | 63 | `ECG1` | present |

There are exactly two ordered total-channel layouts and two ordered EEG
layouts. No duplicate names or case-only collisions were found.

The exact EEG intersection contains 62 channels in native first-layout order:

```text
Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 CP5 CP1 Pz P3 P7 O1 Oz O2 P4 P8
TP10 CP6 CP2 FCz C4 T8 FT10 FC6 FC2 F4 F8 Fp2 AF7 AF3 AFz F1 F5
FT7 FC3 C1 C5 TP7 CP3 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2 CPz CP4
TP8 C6 C2 FC4 FT8 F6 AF8 AF4 F2
```

The EEG union contains 63 channels. `Cz` is the only partial channel.
`ECG1` is auxiliary in every record and is not part of either EEG set.
Every EEG channel has a scalp coordinate; only `ECG1` lacks one.

## Cz states

The contract distinguishes three states:

1. physical absence: the 324 records from participants 1–9 do not contain Cz;
2. physical presence: the 720 records from participants 10–29 contain Cz;
3. policy exclusion: `cog_bci_common` and `emotiv_common` do not select Cz.

Policy exclusion does not modify the source Raw and is not interpolation.
A possible `cog_bci_cz_interpolated` policy is explicitly
`not_implemented`.

## ECG1 role

`ECG1` has:

```text
channel role: auxiliary
default inclusion in EEG: false
coordinate absence: expected
```

The record metadata and native loader still expose auxiliary names explicitly.
No EEG policy automatically includes `ECG1`.

## COG-BCI ↔ Emotiv mapping

COG-BCI uses standard sensor labels such as `AF3`; project CSV columns use the
same labels in the `EEG.` namespace. The complete deterministic mapping is:

| Project Emotiv | COG-BCI | Status |
|---|---|---|
| EEG.AF3 | AF3 | explicit_alias_match |
| EEG.F7 | F7 | explicit_alias_match |
| EEG.F3 | F3 | explicit_alias_match |
| EEG.FC5 | FC5 | explicit_alias_match |
| EEG.T7 | T7 | explicit_alias_match |
| EEG.P7 | P7 | explicit_alias_match |
| EEG.O1 | O1 | explicit_alias_match |
| EEG.O2 | O2 | explicit_alias_match |
| EEG.P8 | P8 | explicit_alias_match |
| EEG.T8 | T8 | explicit_alias_match |
| EEG.FC6 | FC6 | explicit_alias_match |
| EEG.F4 | F4 | explicit_alias_match |
| EEG.F8 | F8 | explicit_alias_match |
| EEG.AF4 | AF4 | explicit_alias_match |

No fuzzy matching, case folding, whitespace normalization or nearest-channel
selection is used. Exact source names have priority over aliases. Missing or
multiple configured candidates produce `ChannelHarmonizationError`.
There are no missing or ambiguous entries in the real index.

The mapping is stored in
`bench/datasets/channel_maps/cog_bci_emotiv.json`. Its canonical order must
equal the production Python contract or loading fails.

## Coordinates and montage

COG-BCI coordinates for the 14 mapped labels equal MNE's
`standard_1020` reference positions within floating-point precision; the
largest observed distance is approximately `1.1e-13 mm`.

The project Emotiv CSV/cache artifacts contain channel names and order but no
observed electrode coordinates. Therefore this comparison validates the
COG-BCI standard-label montage only; it is not a coordinate measurement of
the project device and is not used to change mapping or signal samples.

## Typed contract and policies

`bench/datasets/channel_contracts.py` defines:

- `ChannelLayout`;
- `ChannelMappingEntry`;
- `ChannelSelectionPolicy`;
- `ChannelSelectionValidation`;
- `ChannelSelectionResult`;
- `ChannelHarmonizationError`;
- `apply_channel_policy()`.

`apply_channel_policy()` defaults to `copy=True`, selects in policy order and
renames mapped outputs to canonical project names. Its provenance records
source layout, source names, selected source/output names, excluded auxiliary
channels and performed operations. It does not resample, filter or
rereference.

Implemented policies:

- `cog_bci_native`: original 62/63 EEG channels, source order retained,
  variable shape;
- `cog_bci_common`: computed 62-channel intersection, fixed native order;
- `emotiv_common`: 14 mapped channels, exact production Emotiv order and
  canonical `EEG.*` output names.

`COGBCIDataset.get_channel_policy()` builds these policies from the complete
record index. `COGBCIDataset.select_raw_channels()` opens one record with
`preload=False` by default and applies the policy to a copy. Existing
`open_raw(record_id, preload=False)` behavior is unchanged.

## Real-data diagnostic smoke

One Flanker record from `sub-01` and one from `sub-10` were opened lazily.
All three policies were applied to both:

| Subject | Physical Cz | native | common | Emotiv |
|---|---|---:|---:|---:|
| sub-01 | absent | 62 | 62 | 14 |
| sub-10 | present | 63 | 62 | 14 |

For all six selections:

- source and selected objects remained `preload=False`;
- source channel order was unchanged;
- the result was a copy;
- `ECG1` was excluded;
- sampling rate remained 500 Hz;
- sample count was unchanged;
- fixed policies gave identical shapes and orders across layouts;
- `emotiv_common` output names exactly matched the project raw tensor order.

Status: `diagnostic`.

## Runtime artifacts

Ignored artifacts are under `benchmark_results/cog_bci_channel_audit/`:

```text
cog_bci_channel_layouts.csv
cog_bci_channel_presence.csv
cog_bci_common_channels.json
project_emotiv_channel_contract.json
cog_bci_emotiv_mapping.csv
coordinate_audit.csv
channel_policy_smoke.json
channel_audit_summary.json
channel_audit_report.md
errors.csv
```

Artifacts contain relative provenance paths only. `errors.csv` is empty.

## Validation

Tests cover order preservation, Cz/ECG handling, missing and ambiguous
channels, exact-name priority, alias mapping, copy/in-place behavior,
unchanged sampling/time/filter/reference state, deterministic serialization,
loader integration, mapping completeness and raw-window backward
compatibility.

```text
new channel-contract tests: 27 passed
targeted channel/loader/raw/model tests: 78 passed
config audit and curation regression tests: 94 passed
python -m pytest -q tests: 861 passed, 12 warnings
python -m pytest -q: 861 passed, 12 warnings
```

The warnings are the existing pytest `cache_dir` configuration warning and
small synthetic-classification metric warnings. The first full run exposed
that a YAML mapping artifact was being classified as an experiment config;
the artifact was moved to JSON, after which all config-governance and full
tests passed.

## Recommendations and readiness boundary

For the first fixed-shape native COG-BCI baseline, use `cog_bci_common`.
Retain `cog_bci_native` for record-level inspection and any future model that
explicitly supports variable channel counts.

For cross-dataset technical compatibility, use `emotiv_common`. It establishes
names and order only; it does not justify a transfer-learning claim by itself.

Before window materialization:

1. define the native scientific task and event/window alignment;
2. choose and version sampling, filtering and reference policies separately;
3. save channel policy, ordered names and mapping hash in every new cache;
4. preserve source/session/record IDs for leakage-safe splits;
5. decide whether a future Cz experiment is scientifically justified;
6. test window boundaries and event discontinuities without hiding MNE
   boundary warnings.

```text
record discovery: ready
channel audit and deterministic selection: ready
scientific channel-harmonization validation: not performed
window materialization: not implemented
scientific task: not implemented
training: not implemented
```
