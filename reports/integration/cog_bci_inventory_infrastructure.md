# COG-BCI archive verification and inventory infrastructure

## Status and scope

- Branch: `integration/benchmark-unification`
- Audited HEAD: `332fc04`
- Initial working tree: clean; staging area empty.
- Result status: `diagnostic`
- Production dataset loader: intentionally not implemented.
- Real EEG training, preprocessing, resampling, channel mapping, windowing,
  target construction, and full archive extraction: not performed.

The implementation provides one importable data-management module and CLI:

```text
scripts/data/cog_bci_inventory.py
```

It is separate from the benchmark runner because it verifies and prepares
external source files; it does not load samples into `EEGData` or define a
scientific task.

## Architecture

The tool has four composable stages:

```text
archive discovery
→ checksum/ZIP verification
→ guarded member-level extraction
→ structural and EEGLAB-header inventory
→ deterministic CSV/JSON/Markdown artifacts
```

The same public functions are used by the CLI and synthetic tests. Source
archives are opened read-only and checked after extraction to ensure their
size did not change.

## CLI

Normal invocation after the download is complete:

```powershell
python scripts/data/cog_bci_inventory.py `
  --archives-dir data/raw/cog_bci/archives `
  --extract-dir data/raw/cog_bci/extracted `
  --output-dir benchmark_results/cog_bci_inventory `
  --mode all `
  --require-complete
```

Supported modes:

- `verify-archives`: discover expected archives, apply an available checksum
  manifest, open each ZIP, and run its member CRC test;
- `extract`: verify first, then extract only archives with status `valid`;
- `inventory`: inspect an existing extracted tree without touching archives;
- `all`: run verification, extraction, and inventory in order.

Flags:

- `--resume` continues an extraction carrying an incomplete marker;
- `--overwrite` explicitly permits replacement of an existing changed file;
- `--verify-only` prevents extraction and inventory even with `--mode all`;
- `--require-complete` returns a non-zero code unless all expected archives
  are valid;
- `--checksum-manifest` selects a JSON, CSV, or checksum-text manifest;
- `--subjects` restricts a diagnostic verification to named subjects;
- `--skip-content-test` reads the ZIP directory only and records
  `zip_test_passed=not_run`; it is not a substitute for CRC verification.

Without `--require-complete`, a missing or still-downloading archive set
produces `overall_status=incomplete` with exit code zero. Corrupt or partial
archives are never deleted.

## ZIP verification

The archive table records:

```text
filename, subject_id, size_bytes, zip_readable, zip_test_passed,
member_count, compressed_size, uncompressed_size, checksum_expected,
checksum_actual, checksum_match, status, error
```

Expected names are `sub-01.zip` through `sub-29.zip`. Missing subjects receive
explicit `missing` rows. The other statuses are `partial`, `valid`, `corrupt`,
`checksum_mismatch`, `unexpected_name`, and `duplicate_subject`.

Checksum verification is never inferred. If no manifest exists,
`checksum_expected` and `checksum_match` are `not_available` and
`checksum_actual` is `not_computed`. A 32-character expected digest selects
MD5 for compatibility with an explicit source manifest; otherwise SHA-256 is
used. ZIP integrity remains a separate CRC-based check.

## Safe extraction and Zip Slip protection

The implementation never calls `ZipFile.extractall`. Before creating any
output, it validates every member and rejects:

- `..` path components;
- POSIX absolute paths;
- Windows drive or UNC absolute paths;
- backslash variants of traversal paths;
- encrypted members;
- symbolic-link members;
- duplicate normalized destinations;
- any resolved destination outside the extraction root.

All members are preflighted before the incomplete marker is created. Existing
files are compared by size and CRC:

- an identical file is reported as `already_correct`;
- a changed file causes a fatal conflict by default;
- `--overwrite` writes the new content to a sibling temporary file, verifies
  its size and CRC, and atomically replaces the destination.

An extraction marker is written per subject. A pre-existing marker requires
`--resume`; already correct members are skipped, so interrupted extraction
continues member by member. The marker is removed only after all members pass.
Archives and other source files are never removed or rewritten.

## Structural inventory

The record inventory recognizes:

```text
.set, .fdt, .tsv, .json, .txt, .mat
```

It derives subject, session, task, and run from actual relative paths, retains
the original task stem, and records `.set/.fdt` pair paths. It reports
unpaired files, unknown extensions, service files, missing participant
identifiers, and duplicate `subject/session/task/run` combinations.

TSV event rows are read with a bounded streaming parser. JSON metadata is
bounded by file size and item count. MATLAB files are inspected with
`scipy.io.whosmat`, which reads variable headers rather than payload arrays.
These sources populate descriptive event, trigger, rating-field, and
behavioural-outcome inventories. No value is promoted to a target.

## EEGLAB header reading

When MNE is available, each `.set` is opened with:

```python
mne.io.read_raw_eeglab(path, preload=False)
```

The inventory attempts to record channel count and names, sampling frequency,
sample count, duration, units, annotations/events, reference, and bad
channels. Each reader failure is written to `errors.csv`; it does not stop
other records. The current environment does not contain MNE, so the synthetic
smoke deliberately exercised this local-failure path. The tool does not parse
`.fdt` samples or load complete raw arrays.

Channel layouts and sampling rates are only summarized descriptively. No
automatic 64/32/14-channel correspondence or resampling policy is introduced.

## Artifact contract

Runtime artifacts are ignored by Git and written under
`benchmark_results/cog_bci_inventory/`:

```text
archive_inventory.csv
extraction_manifest.csv
record_inventory.csv
channel_inventory.csv
event_inventory.csv
inventory_summary.json
inventory_report.md
errors.csv
```

All data paths in tables are relative to the configured roots. Reports contain
no local absolute paths, timestamps, or nondeterministic row ordering.

## Actual download snapshot and smoke runs

The live directory was changing while this task ran. The final structural
snapshot found:

```text
expected archives                  29
ZIP directories readable          25 (sub-01 through sub-25)
active/partial archive              1 (sub-26)
missing archives                    3 (sub-27 through sub-29)
checksum manifest                   not available
overall status                      incomplete
```

The structural scan used `--skip-content-test`; its readable rows explicitly
say `zip_test_passed=not_run`, and its `crc_verified_archive_count` is zero.
It did not claim full CRC verification.

A separate real diagnostic smoke fully tested `sub-01.zip`:

```text
archive size                 1,096,627,416 bytes
members                                  112
compressed bytes             1,096,608,146
uncompressed bytes           1,350,645,191
ZIP readable                           true
member CRC test passed                 true
checksum verification         not available
real extraction                     not run
```

The archive directory showed three sessions (`ses-S1` to `ses-S3`), 36 paired
`.set/.fdt` records, and 24 behavioural `.mat` files. Actual EEG stems include
Flanker, three MATB levels, PVT, zero/one/two-back, and four resting-state
records. These are preliminary filename observations, not task or target
definitions.

A second smoke used a tiny synthetic archive and ran `--mode all`. Verification,
safe extraction, set/fdt pairing, TSV event inventory, artifact writing, and
the per-record missing-MNE error path all completed. Its status is
`diagnostic`, not a dataset result.

## Tests

The synthetic suite covers:

- valid, corrupt, partial, zero-byte, missing, duplicate-subject, and
  unexpected-name archives;
- explicit checksum match, mismatch, and unavailable states;
- traversal through `../`, POSIX absolute paths, and Windows absolute paths;
- first extraction, repeat extraction, resume, overwrite refusal, and
  explicit overwrite;
- preservation of source archive bytes;
- set/fdt pairs and both unpaired directions;
- unknown/service files and participant naming mismatch;
- bounded TSV metadata extraction without target creation;
- local EEGLAB-reader failure;
- empty directories and `--require-complete`;
- all four CLI modes, `--verify-only`, fatal exit codes, subject subsets,
  deterministic reports, relative paths, and sorted tables.

Baseline before implementation:

```text
775 passed, 12 warnings
```

The baseline used a unique ignored `--basetemp`; an earlier overlapping retry
had only a Windows temporary-directory lock and did not reveal a code failure.

## Limitations and next step

- The full 29-archive download was not complete.
- Only `sub-01.zip` received a complete member CRC test in this task.
- No upstream checksum manifest was present, so cryptographic provenance
  remains unresolved.
- No real archive was extracted.
- MNE is optional and absent in the current environment.
- Channel units, reference, exact events, behavioural values, missing records,
  and cross-subject consistency remain unknown until complete extraction and
  inventory.

After all downloads finish, obtain the publisher's checksum manifest if one is
available, run the normal `all --require-complete` command, review every error
and structural inconsistency, and prepare the data card plus explicit channel,
sampling, event, and task contracts. A production `COGBCIDataset` should be
implemented only from that verified inventory, because the current archive
snapshot and publication-level assumptions are insufficient to define a
loader safely.
