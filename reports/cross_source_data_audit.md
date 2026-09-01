# Cross-source data audit

## Canonical inputs

- Feature dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`.
- Logical-record manifest: `data/interim/logical_recording_map.parquet`.
- Raw-record catalog: `data/interim/emotiv_record_catalog.parquet`.
- Target: `label_q5`; 45,384 of 51,308 feature windows are supervised.
- Feature space: 448 EEG + POW columns selected by the canonical `pow_plus_eeg` loader.

The processed dataset contains 120 source records. One 54-window `Old_EEG`
record (`007291c7`) is entirely unlabeled and is therefore absent from the
supervised logical-record manifest. The supervised data contain 119 source
records represented by 86 logical recordings.

## Per-source inventory

| source | all windows | supervised windows | subjects | supervised source records | logical recordings | class distribution 0/1/2/3/4 |
|---|---:|---:|---:|---:|---:|---|
| `gpn_data` | 27,021 | 23,826 | 42 | 71 | 71 | 4,813 / 4,920 / 4,876 / 4,684 / 4,533 |
| `Old_EEG` | 24,287 | 21,558 | 43 | 48 | 48 | 4,267 / 4,155 / 4,199 / 4,394 / 4,543 |

All 120 catalogued raw files exist locally. Both sources declare 256 Hz EEG
and the same 14-channel order: `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8,
FC6, F4, F8, AF4`. Supervised feature rows have no missing logical ID and the
canonical loader drops rows with a missing target or any non-finite selected
feature. Approximate supervised record durations range from 35 to 7,365 s in
`gpn_data` and from 35 to 7,685 s in `Old_EEG`.

## Subject and logical-record overlap

- Shared subject IDs: 31.
- `gpn_data`-exclusive subject IDs: 11.
- `Old_EEG`-exclusive subject IDs: 12.
- Logical recordings exported under both sources: 33, covering 30 shared
  subjects and 28,770 supervised rows when both copies are counted.
- Every one of the 33 pairs has equal time ranges and is marked
  `full_duplicate_export_exact_on_supervised_windows` by the canonical
  manifest. Each side contributes 14,385 duplicated supervised windows.

The exact signal duplicate pairs provide strong direct evidence that the same
subject ID denotes the same person for 30 of the 31 shared IDs. The remaining
ID, `a02151ac`, has different, non-overlapping day-1/day-2 recordings in the
two sources. Catalog metadata do not contain a reliable common demographic
identifier with which to verify that last identity independently, so it is
retained only as a same-ID shared-subject candidate. After the 33 duplicate
logical recordings are removed symmetrically, it is also the only subject with
residual data in both sources: 263 `gpn_data` windows and 456 `Old_EEG`
windows. This uncertainty does not affect CS1 and CS2 is invalid under the
configured sample-size thresholds.

## Duplicate intersection

The complete row-level intersection is stored canonically in
`data/interim/logical_recording_map.parquet`; selecting
`present_in_both_sources == true` yields 33 rows with `subject_id`, `sources`,
`source_record_ids`, `record_group_id`, per-record window counts, label
distribution, time-range equality and signal relationship. The duplicated
logical IDs are:

`0012905a` (3 recordings), `01c2a0d8`, `1081b177`, `20201194`, `2162c09e`,
`2182c1cd`, `219060fa`, `3110e0c7`, `40009139`, `41e2010c`, `50c02189`,
`517001af`, `6030f0fd`, `7072a0e0`, `7150e10a`, `71c09041`, `71e10186`,
`71f21142`, `8030618f`, `81e150c1`, `81f1f0fe`, `9192c107`, `a1721173`
(2 recordings), `a1b210fc`, `b112005d`, `c060c06a`, `c1a150b1`, `d0e2d025`,
`d151b0c4`, and `f121f1e0`.

## Protocol eligibility

| direction | mode | train windows / subjects / records | test windows / subjects / records | Transformer train/test sequences | status |
|---|---|---|---|---|---|
| `gpn_data -> Old_EEG` | source-exclusive (CS1) | 6,348 / 11 / 21 | 6,717 / 12 / 14 | 6,165 / 6,496 | valid |
| `Old_EEG -> gpn_data` | source-exclusive (CS1) | 6,717 / 12 / 14 | 6,348 / 11 / 21 | 6,496 / 6,165 | valid |
| `gpn_data -> Old_EEG` | shared-subject (CS2) | 263 / 1 / 1 | 456 / 1 / 1 | 256 / 449 | invalid: fewer than 5 train and 3 test subjects |
| `Old_EEG -> gpn_data` | shared-subject (CS2) | 456 / 1 / 1 | 263 / 1 / 1 | 449 / 256 | invalid: fewer than 5 train and 3 test subjects |

Sequence counts use length 8, stride 1 and the existing 10.5 s gap rule.
Every valid CS1 partition has all five train classes, all five test classes,
at least 20 test predictions per subject, and zero subject, logical-record,
source-record and sample overlap. In CS2 the 33 duplicate logical recordings
are removed symmetrically before counting; this leaves zero logical-record,
source-record and sample overlap, but insufficient subjects.

## Required implementation

The smallest safe extension is:

1. enrich `EmotivDataset` row metadata with `record_group_id` from the
   configured logical-record manifest;
2. add `CrossValidator.run_cross_source_holdout()` returning the existing
   `TaskSplit` type plus explicit eligibility/leakage metadata;
3. dispatch `evaluation.protocol: cross_source_holdout` in `BenchmarkRunner`
   and reuse its model factory, sequence builder, metrics and split artifacts;
4. add a thin experiment planner that expands at most eight canonical configs
   and invokes `BenchmarkRunner` for valid trials only;
5. write cross-source audit artifacts next to the standard fold artifacts.

No raw cache, preprocessing path, model implementation, independent training
loop, metric implementation, or artifact hierarchy needs to be introduced.
