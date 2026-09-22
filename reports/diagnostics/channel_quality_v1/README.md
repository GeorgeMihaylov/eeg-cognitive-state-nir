# Frozen 14-channel signal quality audit

Status: diagnostic, read-only input audit; no models, labels, signal transformations, channel removal or selection.

## Data and definitions

30,958 windows, 54 participants, 86 records, five existing subject folds; 14 x 2560 samples at 256 Hz. All raw shards and all frozen sample/offset mappings were verified using existing preflight utilities. Full output: 433,412 window-channel rows, 1,204 record-channel rows, 756 participant-channel rows, 70 fold-channel rows.

Quality metrics: mean, population SD, unscaled temporal MAD, RMS, peak-to-peak, absolute maximum, exact repeats, existing near-flat fraction (absolute first difference <= 1e-6), median absolute and RMS first difference, diff_RMS / signal_RMS, total Welch power and relative delta/theta/alpha/beta/gamma power. High-frequency ratio is 30-45 Hz power / sum of 1-45 Hz band powers. No additional filtering, referencing or resampling. All amplitude units are stored raw-cache units; no conversion to physical units is assumed.

Within-window score: (channel value - median of other 13 channels) / unscaled MAD of those channels. Descriptive high/low anomaly: z > 3 / z < -3, fixed before smoke/full results. Undefined if MAD=0 or insufficient finite peers; no artificial epsilon or automatic removal. Undefined counts are retained. The full real-data scores were defined for every window/channel.

Record summaries use window medians for continuous metrics and means for anomaly/violation rates. Participant summaries average records equally. Global/fold summaries use participant medians and IQRs; window-pooled summaries are separately named. Thus the percentages below are median record-balanced participant rates, not pooled-window percentages. Elevated subject fraction means that channel's absolute SD-anomaly rate exceeds that subject's median across 14 channels; it is not a hardware QC threshold or a classification of bad channels.

## Global 14-channel summary

| Channel | SD median | SD subject IQR | High-SD rate % | High-SD subject IQR pp | Low-SD rate % | High-max rate % | High-roughness rate % | Subjects elevated SD % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AF3 | 41.084 | 15.803 | 15.155 | 14.534 | 0.218 | 4.780 | 3.739 | 70.370 |
| F7 | 47.156 | 21.064 | 32.652 | 23.958 | 0.198 | 11.085 | 7.445 | 94.444 |
| F3 | 24.976 | 21.171 | 1.127 | 2.536 | 0.356 | 0.435 | 0.330 | 11.111 |
| FC5 | 32.768 | 23.875 | 3.749 | 6.950 | 0.188 | 1.791 | 4.154 | 24.074 |
| T7 | 21.022 | 20.284 | 1.835 | 2.807 | 1.894 | 0.612 | 4.143 | 16.667 |
| P7 | 27.112 | 28.404 | 4.071 | 12.355 | 1.812 | 0.444 | 1.344 | 55.556 |
| O1 | 21.609 | 20.659 | 1.136 | 2.137 | 1.250 | 0.373 | 1.527 | 9.259 |
| O2 | 23.583 | 17.374 | 2.283 | 2.716 | 0.367 | 0.623 | 3.071 | 12.963 |
| P8 | 33.973 | 26.356 | 6.739 | 12.563 | 0.194 | 0.908 | 1.814 | 44.444 |
| T8 | 28.864 | 22.639 | 2.676 | 8.392 | 0.301 | 0.667 | 8.885 | 27.778 |
| FC6 | 41.843 | 28.383 | 22.949 | 18.126 | 0.000 | 60.943 | 7.466 | 88.889 |
| F4 | 28.552 | 19.008 | 6.503 | 9.501 | 0.000 | 40.063 | 1.167 | 46.296 |
| F8 | 52.873 | 28.388 | 47.072 | 25.038 | 0.000 | 40.792 | 17.658 | 100.000 |
| AF4 | 48.154 | 23.803 | 35.226 | 29.571 | 0.000 | 65.058 | 8.037 | 90.741 |

## Participant-level stability

F8 has a high SD-anomaly rate in the typical participant (47.1%); its absolute SD-anomaly rate exceeds the within-subject channel median in 54/54 participants. The corresponding counts are F7 51/54, AF4 49/54 and FC6 48/54. This pattern is not limited to one recording, although record/participant distributions remain heterogeneous.

The between-participant IQR of the high-SD anomaly rate is largest for AF4 (29.57 percentage points), F8 (25.04) and F7 (23.96). By absolute SD, the widest participant IQRs are P7, F8 and FC6 (about 28.4 stored units); the smallest are AF3 (15.80), O2 (17.37), F4 (19.01). These are different notions of variability and are not combined into a ranking.

O1, T7, O2 and F3 have the smallest between-participant IQRs of absolute SD-anomaly rates (about 3.86, 3.99, 4.55 and 4.56 percentage points). F3 has low median high-SD and high-roughness anomaly rates (1.13% and 0.33%). This supports relative stability on these metrics only, not a universal good-quality label.

F8 versus F3 differs by about 46 percentage points in high-SD rate and 17.3 points in high-roughness rate. AF4 versus AF3 differs in high-SD rate (35.2% versus 15.2%). Fold ranges below are descriptive consistency checks only; no folds were used for selection.

| channel | min | max |
| --- | --- | --- |
| AF3 | 0.112 | 0.192 |
| F7 | 0.222 | 0.423 |
| F3 | 0.005 | 0.017 |
| FC5 | 0.017 | 0.086 |
| T7 | 0.014 | 0.022 |
| P7 | 0.022 | 0.069 |
| O1 | 0.004 | 0.022 |
| O2 | 0.017 | 0.036 |
| P8 | 0.033 | 0.090 |
| T8 | 0.015 | 0.099 |
| FC6 | 0.155 | 0.327 |
| F4 | 0.046 | 0.138 |
| F8 | 0.392 | 0.531 |
| AF4 | 0.289 | 0.450 |

## Quality versus existing redundancy

The redundancy source is the existing balanced 533-window pilot covering 54 participants; the QC audit uses the full frozen cohort. This is a descriptive per-channel cross-reference, not a same-window causal comparison. Redundancy score is participant median mean absolute correlation; partner is the most common strongest partner. No composite score or ranking is calculated.

| channel | participant_median_mean_abs_corr | most_common_strongest_partner | participant_median_anomaly_std_high | participant_iqr_anomaly_std_any | participant_median_anomaly_roughness_ratio_high |
| --- | --- | --- | --- | --- | --- |
| AF3 | 0.573 | AF4 | 0.152 | 0.132 | 0.037 |
| F7 | 0.440 | FC5 | 0.327 | 0.233 | 0.074 |
| F3 | 0.656 | F4 | 0.011 | 0.046 | 0.003 |
| FC5 | 0.554 | F7 | 0.037 | 0.068 | 0.042 |
| T7 | 0.511 | FC5 | 0.018 | 0.040 | 0.041 |
| P7 | 0.455 | O1 | 0.041 | 0.183 | 0.013 |
| O1 | 0.520 | P7 | 0.011 | 0.039 | 0.015 |
| O2 | 0.544 | P8 | 0.023 | 0.045 | 0.031 |
| P8 | 0.556 | T8 | 0.067 | 0.136 | 0.018 |
| T8 | 0.567 | P8 | 0.027 | 0.092 | 0.089 |
| FC6 | 0.616 | F4 | 0.229 | 0.181 | 0.075 |
| F4 | 0.675 | F3 | 0.065 | 0.105 | 0.012 |
| F8 | 0.563 | FC6 | 0.471 | 0.250 | 0.177 |
| AF4 | 0.624 | AF3 | 0.352 | 0.277 | 0.080 |

F3 combines relatively high redundancy (0.656; partner F4) with low relative QC anomaly rates. F7 has lower redundancy (0.440; partner FC5) yet a high variability-anomaly rate. F8 has intermediate redundancy (0.563; partner FC6) with high variability and roughness anomaly rates. AF4 and FC6 have relatively high redundancy together with more frequent relative scale anomalies. These combinations illustrate why redundancy alone is not a signal-quality measure.

## Interpretation and diagnostic candidates

F8, F7, AF4 and FC6 have patterns worth inspecting diagnostically, with F8 showing both higher variability and roughness indicators. P7 shows substantial between-subject scale variability. None is automatically classified as bad, and this audit does not establish that any channel impairs model transfer.

A material limitation: raw means are around 4,200-4,500 stored units, so RMS and absolute maxima are strongly influenced by DC offset. F4 has a high absolute-maximum anomaly rate despite modest SD/roughness rates. Absolute-amplitude anomalies must not be equated with transient artifacts or clipping. The diff_RMS / signal_RMS denominator is also offset-sensitive, so diff_RMS and PSD diagnostics are retained separately. The high-frequency PSD ratios do not show the same simple ordering as roughness; unusually large first differences do not establish a physiological or hardware mechanism.

No non-finite or exactly zero-SD channel windows were found. Local near-flat extremes still exist: the maximum near-flat fraction is 62.84% for P8 at sample 8361 (subject 5001d09a); F4 in the same window reaches 46.50%, and O2 at sample 7122 reaches 45.41%. These are window-level observations, not evidence of generally bad channels; all continuous values are retained in the runtime table. The existing artifact-rejection contract is disabled and has no amplitude/flatness limits; violation rates are undefined rather than zero. No hardware saturation threshold or confirmed acquisition mains frequency was available, so clipping and mains-neighbourhood ratios were not invented. Analysis is conditional on the already frozen/QC-accepted cohort and cannot characterize excluded acquisition windows.

## Runtime and validation

Smoke: 54 windows, 756 channel rows, 2.74 seconds, PASS. Full: 163.00 seconds, PASS. Targeted tests: 42 passed; py_compile PASS; sample-universe/order/fold/record consistency and code hashes PASS; git diff --check exit 0. Existing raw file sizes/mtimes remained unchanged. No PM targets loaded and no training executed.

## Reproduction and artifacts

Implementation: bench/analysis/channel_quality.py; thin CLI: scripts/analysis/run_channel_quality.py; frozen specification: experiments/pm_diagnostics/channel_quality_v1.json; tests: tests/test_channel_quality.py.

Run smoke first with --mode smoke and output reports/diagnostics/channel_quality_smoke_v1, then full with --mode full and output reports/diagnostics/channel_quality_v1. Both require --feature-cache and --runtime. Existing output directories are never overwritten.

Full identities, definitions, hashes and configuration: run_summary.json and protocol.json. Compact tables: channel_global_quality_summary.csv, channel_subject_quality.csv, channel_record_quality.csv, channel_fold_quality.csv, channel_quality_vs_redundancy.csv. Continuous window-level metrics and scores: benchmark_results/channel_quality_v1/window_channel_quality.parquet (runtime artifact). Validation and Git snapshot: validation.json, git_status_short.txt.

Branch: analysis/pm-target-validity-audit-20260827. HEAD: 222dbaf6ebe090b4211308ca673607aa04aabd55. Existing dirty files preserved. No commit, staging or cleanup.
