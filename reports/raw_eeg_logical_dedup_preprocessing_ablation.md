# Raw EEG logical-record deduplication and preprocessing ablation

## Scope

This report covers the `label_q5` EEGNet benchmark on 10-second raw EEG
windows. All comparisons use the same five precomputed outer folds grouped by
`subject_id`, the same EEGNet architecture, seed 42, AdamW optimizer, batch size
32, maximum 15 epochs and early-stopping patience 4. Inner validation is grouped
by `record_group_id` and is disjoint from the outer test records.

The comparison conditions are:

1. unfiltered raw EEG with all source-specific records;
2. unfiltered raw EEG with one deterministic source record per logical recording;
3. bandpass + notch + common-average-reference EEG with the same deduplicated records.

Artifact rejection is disabled in all three runs so that the third comparison
changes preprocessing only.

## Logical-record audit

- Source-specific records: **119**
- Logical recordings: **86**
- Logical recordings present in both `gpn_data` and `Old_EEG`: **33**
- Source records removed by deduplication: **33**
- Accepted windows before deduplication: **45,326**
- Accepted windows after deduplication: **30,958**
- Accepted windows removed with duplicate source records: **14,368**
- Cross-source pairs with matching timestamp ranges: **33/33**
- Cross-source pairs with matching labels: **33/33**
- Cross-source pairs with exactly equal float32 cached signal tensors on all
  common supervised windows: **33/33**
- Logical recordings spanning more than one outer fold: **0**
- Inner train/validation logical-record overlaps in the five-fold audit: **0**
- Selected sources: **71 `gpn_data`**, **15 `Old_EEG`**

The deterministic source selection ranks accepted-window fraction (descending),
available raw EEG rows (descending), accepted-window missing fraction
(ascending), fixed source priority (`gpn_data`, then `Old_EEG`) and lexical
`record_id`. The complete per-record table and selection reasons are in
`reports/logical_recording_audit.md` and
`data/interim/logical_recording_map.parquet`.

All five test-subject lists are identical before and after deduplication:
11, 11, 11, 10 and 11 subjects in folds 1–5.

## Artifact audit

The unfiltered artifact audit covers all 45,326 accepted windows. Peak-to-peak,
variance and flatline distributions use every window; amplitude quantiles use a
deterministic sample of up to 2,048 values per record/channel.

- Source/channel amplitude p1 range: **3,095.503–4,117.612**
- Source/channel amplitude p99 range: **4,554.791–5,374.213**
- Source/channel p99.9 peak-to-peak range: **6,119.658–8,202.464**
- Source/channel p99.9 variance range: **1,493,255.954–6,187,561.177**
- Source/channel p99.9 flat-fraction range: **0.019330–0.118717**
- Export-extrema clipping proxy: **0–0.00001995%**
- NaN/Inf: **0%** in every source/channel group

Because the unreferenced channels contain large DC offsets, no absolute-amplitude
threshold was introduced. A future artifact-rejection experiment should first
inspect the measured distribution after its fixed preprocessing, freeze thresholds
before outer-fold evaluation, and report it as a separate experimental factor.
Detailed channel/source statistics are in `reports/raw_eeg_artifact_audit.md`.

## Preprocessing and caches

Filtering is applied independently within each source record to a reconstructed
14-second interval: the central 10-second window plus 2 seconds of padding on
each side. The central interval is retained after zero-phase filtering. Source
CSVs are never modified.

| Cache | Processing | Accepted / rejected | Size (bytes) | Namespace hash |
|---|---|---:|---:|---|
| raw v3 | resample 256 Hz; no filters; no rereference | 45,326 / 58 | 6,511,440,533 | `2251ca950a467267dcccc1c5b83157f26e02768f46c6073d33f5dc16225bda84` |
| filtered v3 | Butterworth order 4 bandpass 1–45 Hz; IIR notch 50 Hz, Q=30; common-average reference | 45,326 / 58 | 6,510,754,301 | `445be3721678be517a650e93cc43c0eb0267f8eb54bbf4a9cd05fda0323f236e` |

Both caches use float32 output and the canonical 14-channel order. Cache hashes
also include artifact settings, resampling, channel order, loader version and the
source file size/mtime. A 100-window real-data check found a maximum residual CAR
channel mean of `1.31e-05`, consistent with float32 rounding. One hundred sampled
raw-v3 windows were exactly equal to the previous v2 baseline cache, and all
45,384 status/rejection rows were equal.

## One-fold smoke run

The filtered deduplicated smoke run used 1,000 windows and fold 1 on
`NVIDIA GeForce RTX 5060 Ti` (`device=cuda`). It trained for 3 epochs; the best
epoch was 1 with validation loss 1.646908.

| Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|
| 0.233503 | 0.232821 | 0.226710 | 0.227202 |

## Full fold-level results

| Condition | Fold | Test windows | Epochs | Best epoch | Best validation loss | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all raw | 1 | 9,113 | 8 | 4 | 1.597745 | 0.268298 | 0.286081 | 0.229193 |
| all raw | 2 | 8,977 | 5 | 1 | 1.603522 | 0.205637 | 0.212480 | 0.190423 |
| all raw | 3 | 9,094 | 11 | 7 | 1.556395 | 0.245107 | 0.252389 | 0.233682 |
| all raw | 4 | 9,048 | 11 | 7 | 1.598069 | 0.255637 | 0.250906 | 0.220280 |
| all raw | 5 | 9,094 | 6 | 2 | 1.543505 | 0.226633 | 0.223747 | 0.218445 |
| dedup raw | 1 | 6,931 | 6 | 2 | 1.590619 | 0.277738 | 0.284423 | 0.223801 |
| dedup raw | 2 | 6,192 | 9 | 5 | 1.576492 | 0.277132 | 0.276959 | 0.266504 |
| dedup raw | 3 | 6,037 | 12 | 8 | 1.526380 | 0.251781 | 0.252133 | 0.238408 |
| dedup raw | 4 | 5,776 | 11 | 7 | 1.552248 | 0.239958 | 0.237967 | 0.215537 |
| dedup raw | 5 | 6,022 | 7 | 3 | 1.574824 | 0.251578 | 0.241407 | 0.197567 |
| dedup filtered | 1 | 6,931 | 5 | 1 | 1.629856 | 0.230414 | 0.243542 | 0.201773 |
| dedup filtered | 2 | 6,192 | 5 | 1 | 1.693273 | 0.239018 | 0.242230 | 0.214636 |
| dedup filtered | 3 | 6,037 | 7 | 3 | 1.646330 | 0.244492 | 0.247662 | 0.225168 |
| dedup filtered | 4 | 5,776 | 5 | 1 | 1.588840 | 0.215201 | 0.215447 | 0.151490 |
| dedup filtered | 5 | 6,022 | 8 | 4 | 1.559976 | 0.225008 | 0.256901 | 0.181620 |

## Aggregated results and deltas

| Condition | Accuracy mean ± std | Balanced accuracy mean ± std | Macro F1 mean ± std | Weighted F1 mean ± std | AUC mean ± std |
|---|---:|---:|---:|---:|---:|
| all raw | 0.240262 ± 0.022048 | 0.245120 ± 0.025637 | 0.218405 ± 0.015075 | 0.216510 ± 0.015132 | 0.568436 ± 0.030696 |
| dedup raw | 0.259637 ± 0.015150 | 0.258578 ± 0.018799 | 0.228363 ± 0.023192 | 0.227848 ± 0.024191 | 0.585061 ± 0.019674 |
| dedup filtered | 0.230827 ± 0.010315 | 0.241157 ± 0.013842 | 0.194937 ± 0.026129 | 0.191784 ± 0.027321 | 0.568321 ± 0.012620 |

Dedup raw minus all raw:

- accuracy: **+0.019375**
- balanced accuracy: **+0.013457**
- macro F1: **+0.009959**
- weighted F1: **+0.011337**

Filtered dedup minus raw dedup:

- accuracy: **−0.028811**
- balanced accuracy: **−0.017421**
- macro F1: **−0.033426**
- weighted F1: **−0.036064**

The observed deduplication delta is favorable for this fixed run, while the fixed
bandpass/notch/CAR variant is unfavorable. With only five subject folds and one
seed/configuration, these deltas are descriptive and are not a significance claim.

## Run artifacts

- All raw: `benchmark_results/groupkfold_torch_eegnet_raw_all_label_q5/20260715_081334`
- Deduplicated raw: `benchmark_results/groupkfold_torch_eegnet_raw_dedup_label_q5/20260715_082819`
- Deduplicated filtered: `benchmark_results/groupkfold_torch_eegnet_preprocessed_dedup_label_q5/20260715_083924`
- Smoke: `benchmark_results/smoke_torch_eegnet_dedup_preprocessed_label_q5/20260715_081312`

Each full run contains `config.yaml` and run-level `metrics.json`. Each fold contains
`metrics.json`, `predictions.parquet`, `model.pt`, `training_log.csv`,
`normalization_stats.json`, `validation_split.json`, `raw_eeg_stats.json`,
`preprocessing_metadata.json`, `selected_logical_records.parquet` and
`rejected_windows.parquet`. The deduplicated raw and filtered prediction tables
contain the same 30,958 `(sample_id, fold, subject_id, record_id)` rows.

## Verification and limitations

- Tests: **76 passed, 1 expected sklearn warning**.
- All fold validation artifacts have empty logical `group_overlap` and empty
  `outer_test_record_overlap`.
- The raw/all run exactly reproduces the earlier v2 aggregate metrics.
- The two source exports are proven exact over all common supervised cached
  windows; the audit does not claim byte equality of differently compressed CSVs.
- The amplitude audit is partly sampled, while window-level peak-to-peak,
  variance and flatline distributions are exhaustive.
- Artifact rejection remains a separate future experiment.
- Results cover one EEGNet configuration and one seed; they do not establish
  universal preprocessing superiority or inferential significance.
