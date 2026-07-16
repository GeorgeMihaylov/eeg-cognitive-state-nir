# Selected preprocessing trials across three seeds

## Scope and validation

This report compares the selected ShallowConvNet preprocessing trials on the deduplicated raw-EEG benchmark:

- **A:** raw;
- **B:** band-pass 1-45 Hz;
- **E:** band-pass 1-45 Hz + 50 Hz notch;
- seeds **7, 42, 123**;
- precomputed 5-fold GroupKFold by `subject_id`;
- 30,958 accepted windows, 54 subjects, input shape `[1, 14, 2560]`;
- inner validation grouped by `record_group_id`.

Seed 42 uses the semantically validated existing standard benchmark runs; it was not retrained. Seeds 7 and 123 were run through the canonical matrix -> resolved config -> `BenchmarkRunner` path. Across all nine runs, unified predictions contain exactly 30,958 unique `sample_id` values, identical `sample_id`/fold/`y_true` assignments, finite class probabilities summing to one, zero outer subject overlap, and zero inner logical-record overlap.

Commands for the new full runs:

```powershell
python cli.py --experiment-matrix experiments/preprocessing_ablation_shallowconvnet.yaml --seed 7 --trial-ids A,B,E --resume --verbose
python cli.py --experiment-matrix experiments/preprocessing_ablation_shallowconvnet.yaml --seed 123 --trial-ids A,B,E --resume --verbose
```

## Aggregate results

Values below are mean +/- sample SD across all 15 matched seed-fold observations per trial. Training time is the sum across those 15 folds.

| Trial | Preprocessing | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs mean | Training min |
|---|---|---|---|---|---|---|---|---|---|
| A | raw | 0.2825 +/- 0.0139 | 0.2839 +/- 0.0143 | 0.2647 +/- 0.0137 | 0.2649 +/- 0.0155 | 0.1037 +/- 0.0165 | 0.6047 +/- 0.0189 | 11.80 | 38.7 |
| B | band-pass | 0.2835 +/- 0.0163 | 0.2855 +/- 0.0156 | 0.2632 +/- 0.0261 | 0.2629 +/- 0.0284 | 0.1055 +/- 0.0199 | 0.6078 +/- 0.0182 | 10.00 | 32.9 |
| E | band-pass + notch | 0.2830 +/- 0.0162 | 0.2851 +/- 0.0155 | 0.2623 +/- 0.0254 | 0.2622 +/- 0.0278 | 0.1050 +/- 0.0196 | 0.6075 +/- 0.0187 | 10.47 | 34.6 |

Between-seed SD is the sample SD of the three seed-level fold means:

| Trial | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC |
|---|---|---|---|---|---|---|
| A | 0.0028 | 0.0015 | 0.0043 | 0.0047 | 0.0025 | 0.0076 |
| B | 0.0025 | 0.0046 | 0.0020 | 0.0018 | 0.0049 | 0.0089 |
| E | 0.0021 | 0.0044 | 0.0031 | 0.0030 | 0.0045 | 0.0088 |

## Seed-level results

| Trial | Seed | Balanced accuracy | Macro F1 | Epochs mean | Training min |
|---|---|---|---|---|---|
| A | 7 | 0.2854 +/- 0.0139 | 0.2658 +/- 0.0117 | 14.0 | 15.2 |
| A | 42 | 0.2824 +/- 0.0190 | 0.2599 +/- 0.0168 | 10.6 | 11.9 |
| A | 123 | 0.2838 +/- 0.0126 | 0.2684 +/- 0.0138 | 10.8 | 11.6 |
| B | 7 | 0.2889 +/- 0.0163 | 0.2629 +/- 0.0266 | 11.6 | 12.4 |
| B | 42 | 0.2873 +/- 0.0175 | 0.2653 +/- 0.0303 | 11.8 | 13.1 |
| B | 123 | 0.2802 +/- 0.0151 | 0.2614 +/- 0.0273 | 6.6 | 7.4 |
| E | 7 | 0.2862 +/- 0.0165 | 0.2600 +/- 0.0262 | 13.0 | 13.8 |
| E | 42 | 0.2889 +/- 0.0165 | 0.2659 +/- 0.0279 | 11.8 | 13.4 |
| E | 123 | 0.2803 +/- 0.0156 | 0.2610 +/- 0.0278 | 6.6 | 7.4 |

## Paired deltas against raw

Each value is the mean paired difference on identical folds (`candidate - A`). The `all` row averages all 15 matched seed-fold pairs.

| Comparison | Seed | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC |
|---|---|---|---|---|---|---|---|
| B - A | 7 | +0.0009 | +0.0035 | -0.0029 | -0.0040 | +0.0033 | -0.0018 |
| B - A | 42 | +0.0054 | +0.0049 | +0.0053 | +0.0051 | +0.0065 | +0.0114 |
| B - A | 123 | -0.0036 | -0.0036 | -0.0070 | -0.0072 | -0.0042 | -0.0002 |
| B - A | all | +0.0009 | +0.0016 | -0.0015 | -0.0020 | +0.0018 | +0.0032 |
| E - A | 7 | -0.0008 | +0.0008 | -0.0057 | -0.0063 | +0.0004 | -0.0036 |
| E - A | 42 | +0.0055 | +0.0065 | +0.0059 | +0.0055 | +0.0077 | +0.0121 |
| E - A | 123 | -0.0034 | -0.0035 | -0.0074 | -0.0075 | -0.0041 | -0.0002 |
| E - A | all | +0.0004 | +0.0012 | -0.0024 | -0.0027 | +0.0013 | +0.0028 |

## Interpretation

The balanced-accuracy advantage seen at seed 42 is not stable across initializations. B exceeds raw by +0.0035 at seed 7 and +0.0049 at seed 42, but is -0.0036 at seed 123; E is +0.0008, +0.0065, and -0.0035 respectively. Across all folds and seeds, the gains are only +0.0016 for B and +0.0012 for E, while macro F1 is lower than raw by -0.0015 and -0.0024. Raw also has the smallest between-seed SD for balanced accuracy (0.0015 versus 0.0046 for B and 0.0044 for E).

For the current benchmark, **raw is the defensible default** because it is simplest, has the best average macro F1, and is the most stable across seeds. If balanced accuracy is the sole optimization target, B remains a candidate with a very small aggregate edge, but the sign change at seed 123 means this should not be treated as a robust improvement. E provides no consistent advantage over B. No statistical-significance claim is made from these folds/seeds alone.

## Standard result references

The ablation layer stores references only; model, fold metrics, logs and predictions remain in the standard benchmark run directories.

| Trial | Seed | Origin | Reference | Standard run |
|---|---|---|---|---|
| A | 7 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_A\seed_7\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\14c967f7ff382bc0e87e\20260716_152802` |
| A | 42 | legacy semantic reuse | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_A\seed_42\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\full\trial_A\seed_42\20260716_131347` |
| A | 123 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_A\seed_123\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\cb823a65948bcb046918\20260716_161035` |
| B | 7 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_B\seed_7\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\3be662913d664e4cffc9\20260716_154318` |
| B | 42 | legacy semantic reuse | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_B\seed_42\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\full\trial_B\seed_42\20260716_132542` |
| B | 123 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_B\seed_123\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\2b6cc4eb3f9cf3c95325\20260716_162211` |
| E | 7 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_E\seed_7\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\50e4e5dbb76aebe1cb9d\20260716_155543` |
| E | 42 | legacy semantic reuse | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_E\seed_42\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\full\trial_E\seed_42\20260716_140244` |
| E | 123 | canonical | `benchmark_results\preprocessing_ablation_shallowconvnet\references\full\trial_E\seed_123\trial_reference.json` | `benchmark_results\preprocessing_ablation_shallowconvnet\runs\a00977b6a97537ce6b3d\20260716_162935` |

## Definitions

- **Fold-level SD:** sample SD across the 15 seed-fold observations for a trial.
- **Between-seed SD:** sample SD of the three seed-level fold means.
- **Paired delta:** difference on the same precomputed outer fold and seed.
- The machine-readable fold table is `reports/preprocessing_selected_trials_multiseed.csv`.
