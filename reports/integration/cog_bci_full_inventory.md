# COG-BCI full extraction and structural inventory

## Scope and provenance

This diagnostic stage was completed on branch
`integration/benchmark-unification` from `HEAD 00a7e9e`. It performs data
verification, safe extraction and structural inventory only. It does not
define targets, load complete EEG arrays, resample, filter, map channels,
create windows or train models.

The input consists of 29 Zenodo archives (`sub-01.zip` through
`sub-29.zip`). The official record is version 4, DOI
`10.5281/zenodo.7413650`. Published MD5 values were saved as runtime metadata
and all 29 local archives matched them. All archives also passed full ZIP CRC
testing.

Before the extraction audit, 576.87 GiB were free on the data volume. The
archives contain 35.83 GiB uncompressed, leaving much more than the required
10 GiB safety margin. After the completed audit, 576.79 GiB remained free.
The source archives were retained.

## Root cause and correction

The original `all --resume` failure was reproduced with `--debug`. The full
traceback ended in `scipy.io.whosmat()`, where SciPy tried to build the shape
of a MATLAB variable whose internal `dims` value was `None`. This occurs in
348 small behavioural files: all `0-Back.mat`, `1-Back.mat`, `2-Back.mat` and
`Flanker.mat` files. `scipy.io.loadmat()` identifies their payload as
`MatlabOpaque`, consistent with MATLAB table/opaque data.

The fix is deliberately narrow:

- `whosmat()` remains the primary metadata-only reader;
- only the observed `TypeError` caused by a `NoneType` header triggers a
  bounded `loadmat()` fallback;
- fallback loading is limited to behavioural MAT files no larger than
  10 MiB and never touches `.fdt` EEG payloads;
- opaque content is recorded as `not_available`, not silently converted to an
  empty collection;
- `annotations=None` is explicitly treated as an allowed empty annotation
  collection;
- missing channel locations are warnings, while required EEG shape fields
  remain errors;
- normal CLI failures remain concise, while `--debug` re-raises the original
  exception and produces a full traceback.

The tool now writes per-archive progress after each completed archive and
retains a partial extraction manifest if a later archive fails. Each archive
has an explicit terminal status. Repeated `--resume` runs are idempotent.

One repository hygiene problem was also found: the unanchored ignore rule
`data/` hid `scripts/data/cog_bci_inventory.py`. It was restricted to the
root data directory (`/data/`) so the source tool is visible to Git while all
runtime data remain ignored.

## Extraction completeness

| Check | Result |
|---|---:|
| Archives processed | 29 |
| Archive status | 29 `already_complete` |
| ZIP members | 3,260 |
| Directory members | 404 |
| File members | 2,856 |
| Extracted bytes | 38,469,799,666 (35.83 GiB) |
| Missing manifest files | 0 |
| Unexpected extracted files | 0 |
| Size mismatches | 0 |
| Duplicate manifest paths | 0 |
| Conflicts | 0 |
| Partial markers / temporary parts | 0 |

The 2,856 files comprise 1,044 `.set`, 1,044 `.fdt`, 696 behavioural
`.mat`, and 72 channel-location `.txt` files. Member paths pass the existing
Zip Slip, absolute-path, symlink, encryption and duplicate-destination
guards. Extraction does not call `extractall()`.

## Dataset structure

The extracted dataset contains:

- 29 participants;
- three session labels (`ses-S1`, `ses-S2`, `ses-S3`) and 87
  participant-session combinations;
- 1,044 EEG records, exactly 36 per participant and 12 per
  participant-session;
- 1,044 unique `subject/session/task/run` combinations;
- no explicit run token in the EEG filenames.

Each of the following recording names occurs 87 times:

| File-level recording name | Scientific mapping | Evidence |
|---|---|---|
| `zeroBACK`, `oneBACK`, `twoBACK` | N-Back workload conditions | filename, N-Back triggers, PDF |
| `MATBeasy`, `MATBmed`, `MATBdiff` | MATB-II workload conditions | filename, behavioural structures, PDF |
| `PVT` | psychomotor vigilance | filename, PVT triggers, PDF |
| `Flanker` | decision/conflict task | filename, Flanker triggers, PDF |
| `RS_Beg_EO`, `RS_Beg_EC` | beginning resting state | filename, triggers, PDF |
| `RS_End_EO`, `RS_End_Ec` | ending resting state | filename, triggers, PDF |

Thus there are four cognitive task families plus resting state, represented
by 12 file-level recording types. The PDF states that task order was
pseudorandomized; the downloaded `notebook.mat` records that order and
interruptions.

## EEG channels, sampling and duration

MNE 1.12.1 was installed only in the current analysis environment with
`python -m pip install mne`. No tracked dependency file changed. All 1,044
EEGLAB headers were read with
`mne.io.read_raw_eeglab(..., preload=False)`; there were no EEG read
failures.

All recordings are sampled at 500 Hz. There are two stable channel layouts:

- 324 records from participants 1–9 have 63 channels and omit `Cz`;
- 720 records from participants 10–29 have 64 channels and include `Cz`.

This agrees with the PDF, which states that `Cz` could not be recorded for
participants 1–9. Both layouts include `ECG1`; its scalp position is
appropriately absent, producing one missing-location warning in every
record. There are no duplicate channel names and no channels marked bad in
the EEGLAB headers. Units are not exposed by the MNE EEGLAB reader and
therefore remain `not_available`. MNE reports that no custom reference has
been applied; the original acquisition-reference semantics still require
confirmation before preprocessing.

Record duration ranges from 59.912 to 721.870 seconds, with a median of
299.336 seconds. Summed header duration is 81.75 hours. The main per-record
medians are approximately 60 seconds for resting state, 299 seconds for each
MATB condition, 324–329 seconds for N-Back conditions, 605 seconds for
Flanker and 633 seconds for PVT.

## Events and triggers

The EEGLAB files contain 403,004 annotation occurrences. Together with 498
normal MATLAB variable-header rows and 348 opaque-header fallback rows, the
event inventory contains 403,850 rows.

The downloaded official trigger list confirms:

- N-Back start/end, block, normal/hit/conflict onset, error and correct
  response codes in the 600-series;
- Flanker trial, congruent/incongruent stimulus, correct/incorrect response
  and feedback codes in the 20-series;
- PVT trial/ISI, stimulus and response codes `10`–`15`;
- beginning/end and eyes-open/eyes-closed resting-state codes `40`–`53`.

Some EEGLAB annotations contain task-specific dynamic values, so raw unique
annotation strings must not be treated automatically as categorical targets.
The inventory preserves them without defining a label.

## Behavioural and subjective data

The 696 extracted behavioural MAT files split evenly:

- 348 MATB/PVT files are structurally decoded;
- 348 N-Back/Flanker files contain SciPy-opaque MATLAB tables and are
  explicitly marked `not_available`.

Decoded PVT fields include reaction times, error trials, inter-stimulus
intervals and trial counts. Decoded MATB structures include tracking,
system-monitoring, resource-management and, where present, communication
subtasks. These are task-level behavioural measurements, not ready-made
benchmark targets.

The five official service files were downloaded into runtime metadata and
matched their published MD5 values:

- `COG-BCI_info.pdf`;
- `KSS.txt`;
- `RSME.txt`;
- `notebook.mat`;
- `triggerlist.txt`.

`KSS.txt` contains 344 of 348 expected participant/session/condition rows;
all four entries for participant 14, session 3 are absent. It includes
beginning, end and after-PVT condition labels. The numeric after-PVT field
contains values below the conventional KSS range, including negative values,
so its precise definition must be confirmed before it is used as a target.

`RSME.txt` contains 685 of 696 expected
participant/session/task-condition rows. Eight values are absent for
participant 14/session 3, plus participant 16/session 2/condition 7,
participant 16/session 3/condition 4 and participant 19/session 1/condition
2. Its condition labels map to the three MATB conditions, three N-Back
conditions, PVT and Flanker.

No target variable was created. Candidate scientific uses include
task-condition classification, workload analysis within N-Back or MATB,
vigilance analysis with PVT behaviour, conflict analysis with Flanker
behaviour, and carefully defined regression on RSME/KSS. Each requires an
explicit label level, missing-data policy and leakage-safe split.

## Pair and error audit

All 1,044 `.set` files have exactly one same-stem `.fdt`, and all 1,044
`.fdt` files have exactly one `.set`. There are no missing, duplicate,
empty or size-inconsistent pairs and no unknown extracted file formats.

The final error table contains 348 non-fatal behavioural entries, all for
the known MATLAB opaque/table payloads. There are zero archive, extraction,
pairing or EEGLAB-reader errors. Channel-location warnings are retained in
the channel table rather than promoted to fatal errors because the only
missing position is the non-scalp `ECG1` channel.

## Runtime artifacts

The ignored runtime directory `benchmark_results/cog_bci_inventory/full`
contains:

- `archive_inventory.csv`;
- `extraction_manifest.csv` and `extraction_progress.json`;
- `record_inventory.csv`;
- `channel_inventory.csv`;
- `event_inventory.csv`;
- `behavioural_inventory.csv`;
- `file_pair_inventory.csv`;
- `task_inventory.csv`;
- `session_inventory.csv`;
- `subject_inventory.csv`;
- `inventory_summary.json`;
- `inventory_report.md`;
- `errors.csv`.

All paths stored in these artifacts are relative.

## Recommended `COGBCIDataset` contract

A later loader can safely build on this inventory if it:

1. selects records by canonical `subject_id`, `session_id` and file-level
   task name;
2. treats `subject/session/task` as the stable record identity while
   retaining the original relative path;
3. requires a validated one-to-one `.set/.fdt` pair;
4. reads native data lazily and keeps 500 Hz as the default source rate;
5. exposes EEG and `ECG1` explicitly rather than silently mixing modalities;
6. preserves the native 63/64-channel layouts and makes the `Cz` policy an
   explicit downstream option;
7. returns annotations and trigger provenance without constructing a target;
8. joins behavioural, KSS, RSME and notebook metadata only through validated
   participant/session/task keys;
9. exposes missingness and read-status fields;
10. performs no filtering, resampling, channel mapping or windowing inside
    the structural loader unless configured through the existing
    preprocessing contracts.

The dataset is technically ready for loader design. Scientific decisions
still required are the handling of missing `Cz`, separation of ECG from EEG,
reference interpretation, decoding strategy for MATLAB tables, precise KSS
after-PVT semantics, target granularity, missing subjective ratings, and
which task families are appropriate for cross-subject or cross-session
evaluation.

## Validation

- `python -m py_compile scripts/data/cog_bci_inventory.py`: passed;
- targeted COG-BCI tests: 33 passed;
- complete repository suite: 808 passed, 12 warnings;
- `git diff --check`: clean.

The warnings comprise the repository's unknown `cache_dir` pytest setting
and expected metric warnings from small synthetic test partitions. No
archive was deleted, and no commit, push, merge, rebase, reset or staging-area
change was performed.
