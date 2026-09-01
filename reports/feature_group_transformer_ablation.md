# Transformer feature-group ablation

## 1. Objective

Test whether the RF EEG/POW feature-group conclusions persist when eight-window temporal context is encoded by the canonical Transformer.

## 2. Canonical architecture

`Linear(input,128) -> learned positions -> 2 x TransformerEncoder(nhead=4, FF=256, GELU, dropout=0.1) -> last pooling -> 5-class head`; sequence length 8, AdamW, seed 42, record-group inner validation and train-only standardization. `device: auto` resolved to ['NVIDIA GeForce RTX 5060 Ti'].

## 3. Feature-group definitions

| Group | Count | Ordered-list SHA-256 | Input shape |
| --- | ---: | --- | --- |
| eeg_only | 168 | `6e822ee172422e7138945b47b2b27c947393b828b72d96b7a8e22850aded8aca` | `[B, 8, 168]` |
| pow_only | 280 | `c3106631e4ad3eff1694c874a9ff5c4e26470aa587c7294cc5c3047c53d832dd` | `[B, 8, 280]` |
| eeg_pow | 448 | `8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51` | `[B, 8, 448]` |

## 4. Exact sequence alignment

Canonical sequences: **44142**; sequence-index SHA-256: `1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa`. All feature groups have identical sequence IDs, folds, subjects, records, sources, target windows, times and labels; total alignment mismatches are zero. The sequences contain **53 of 54** supervised subjects; ['9192c107'] has no valid length-8 sequence and is absent identically from the published baseline and all three trials.

## 5. Leakage checks

Artifact/leakage audit valid: **True**. Outer subject overlap, inner record/group overlap and outer-test record overlap are zero in all 15 folds. Normalization is fitted on inner train only.

## 6. Fold results

| Group | Fold | BA | Macro F1 | Ordinal MAE | Severe error | Epochs | Best val loss | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only | fold_01 | 0.3276 | 0.3231 | 1.1874 | 0.3461 | 8 | 1.3297 | 13.0 |
| eeg_only | fold_02 | 0.3396 | 0.3404 | 1.0268 | 0.2728 | 12 | 1.2149 | 17.1 |
| eeg_only | fold_03 | 0.3160 | 0.3123 | 1.1860 | 0.3188 | 12 | 1.1427 | 17.3 |
| eeg_only | fold_04 | 0.3764 | 0.3702 | 0.8813 | 0.2005 | 6 | 1.2981 | 8.9 |
| eeg_only | fold_05 | 0.3684 | 0.3552 | 1.0009 | 0.2537 | 8 | 1.1841 | 11.7 |
| pow_only | fold_01 | 0.3322 | 0.3196 | 1.1400 | 0.3203 | 7 | 1.4586 | 10.6 |
| pow_only | fold_02 | 0.3196 | 0.3046 | 1.1821 | 0.3335 | 12 | 1.2779 | 17.0 |
| pow_only | fold_03 | 0.2908 | 0.2779 | 1.3236 | 0.3776 | 14 | 1.2484 | 20.6 |
| pow_only | fold_04 | 0.2926 | 0.2866 | 1.2376 | 0.3442 | 14 | 1.2886 | 20.3 |
| pow_only | fold_05 | 0.3329 | 0.3312 | 1.0752 | 0.2880 | 7 | 1.3861 | 10.6 |
| eeg_pow | fold_01 | 0.3834 | 0.3687 | 0.9647 | 0.2444 | 15 | 1.1683 | 22.0 |
| eeg_pow | fold_02 | 0.3725 | 0.3506 | 1.0231 | 0.2801 | 9 | 1.1465 | 13.8 |
| eeg_pow | fold_03 | 0.3342 | 0.3359 | 1.1150 | 0.2960 | 12 | 1.1839 | 17.8 |
| eeg_pow | fold_04 | 0.3660 | 0.3666 | 0.8974 | 0.2086 | 7 | 1.1976 | 10.8 |
| eeg_pow | fold_05 | 0.3874 | 0.3856 | 0.9191 | 0.2274 | 13 | 1.0613 | 19.3 |

## 7. Aggregate results

| Group | Balanced accuracy | Macro F1 | Accuracy | Weighted F1 | Kappa | AUC | Ordinal MAE | Severe error | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only | 0.3456 +/- 0.0232 | 0.3403 +/- 0.0210 | 0.3446 | 0.3411 | 0.1799 | 0.6835 | 1.0565 | 0.2784 | 305029 |
| pow_only | 0.3136 +/- 0.0185 | 0.3040 +/- 0.0198 | 0.3111 | 0.3042 | 0.1393 | 0.6344 | 1.1917 | 0.3327 | 319365 |
| eeg_pow | 0.3687 +/- 0.0189 | 0.3615 +/- 0.0169 | 0.3664 | 0.3620 | 0.2080 | 0.7036 | 0.9838 | 0.2513 | 340869 |

EEG-only BA retention is **93.7%**; POW-only retention is **85.1%**. Combined minus EEG-only BA is +0.0231; combined minus POW-only is +0.0551. EEG-only remains better than POW-only, and combined is best for BA, macro F1, ordinal MAE and severe-error rate.

## 8. Subject-level statistics

Positive differences favor the left group; error differences use `right_error - left_error`. CIs use 10,000 paired subject bootstraps; Holm correction covers the Transformer comparison family.

| Comparison | Metric | Mean Delta | Median Delta | 95% CI | Improved/degraded/ties | Wilcoxon p | Holm p | Sign p | Rank-biserial |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| eeg_only_vs_pow_only | balanced_accuracy | 0.0430 | 0.0363 | [0.0261, 0.0595] | 40 / 13 / 0 | 0.0000 | 0.0002 | 0.0003 | 0.6785 |
| eeg_only_vs_pow_only | macro_f1 | 0.0417 | 0.0460 | [0.0183, 0.0642] | 37 / 16 / 0 | 0.0008 | 0.0056 | 0.0055 | 0.5290 |
| eeg_only_vs_pow_only | ordinal_mae | 0.1375 | 0.1205 | [0.0194, 0.2515] | 33 / 20 / 0 | 0.0104 | 0.0483 | 0.0984 | 0.4046 |
| eeg_only_vs_pow_only | severe_error_rate | 0.0545 | 0.0752 | [0.0071, 0.1005] | 33 / 20 / 0 | 0.0137 | 0.0483 | 0.0984 | 0.3892 |
| eeg_only_vs_eeg_pow | balanced_accuracy | -0.0210 | -0.0298 | [-0.0351, -0.0066] | 18 / 35 / 0 | 0.0091 | 0.0483 | 0.0270 | -0.4116 |
| eeg_only_vs_eeg_pow | macro_f1 | -0.0246 | -0.0206 | [-0.0423, -0.0069] | 20 / 33 / 0 | 0.0158 | 0.0483 | 0.0984 | -0.3809 |
| eeg_only_vs_eeg_pow | ordinal_mae | -0.0982 | -0.0617 | [-0.1647, -0.0347] | 19 / 34 / 0 | 0.0104 | 0.0483 | 0.0534 | -0.4046 |
| eeg_only_vs_eeg_pow | severe_error_rate | -0.0399 | -0.0329 | [-0.0668, -0.0137] | 20 / 32 / 1 | 0.0080 | 0.0483 | 0.1263 | -0.4224 |
| pow_only_vs_eeg_pow | balanced_accuracy | -0.0641 | -0.0513 | [-0.0831, -0.0455] | 9 / 44 / 0 | 0.0000 | 0.0000 | 0.0000 | -0.8239 |
| pow_only_vs_eeg_pow | macro_f1 | -0.0663 | -0.0530 | [-0.0908, -0.0410] | 10 / 43 / 0 | 0.0000 | 0.0001 | 0.0000 | -0.7051 |
| pow_only_vs_eeg_pow | ordinal_mae | -0.2357 | -0.1971 | [-0.3332, -0.1372] | 12 / 41 / 0 | 0.0000 | 0.0002 | 0.0001 | -0.6785 |
| pow_only_vs_eeg_pow | severe_error_rate | -0.0943 | -0.0774 | [-0.1347, -0.0533] | 13 / 39 / 1 | 0.0001 | 0.0005 | 0.0004 | -0.6372 |

## 9. Source-level results

Source results are descriptive; overlapping people are not treated as independent units in paired tests.

| Group | Source | Sequences | Subjects | BA | Macro F1 | Ordinal MAE | Severe error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only | Old_EEG | 20980 | 42 | 0.3463 | 0.3478 | 1.0583 | 0.2787 |
| eeg_only | gpn_data | 23162 | 41 | 0.3439 | 0.3447 | 1.0548 | 0.2780 |
| pow_only | Old_EEG | 20980 | 42 | 0.3199 | 0.3153 | 1.1735 | 0.3261 |
| pow_only | gpn_data | 23162 | 41 | 0.3035 | 0.3006 | 1.2086 | 0.3389 |
| eeg_pow | Old_EEG | 20980 | 42 | 0.3680 | 0.3694 | 0.9854 | 0.2532 |
| eeg_pow | gpn_data | 23162 | 41 | 0.3651 | 0.3679 | 0.9826 | 0.2496 |

Best source-specific groups: {'Old_EEG': 'eeg_pow', 'gpn_data': 'eeg_pow'}.

## 10. RF versus Transformer interpretation

RF EEG/POW BA retention was 95.7% / 89.2%; Transformer retention is 93.7% / 85.1%. POW therefore does not gain relative importance from temporal context; its retention falls, while the combined-minus-POW gap grows from 0.0329 to 0.0551. The static and temporal models support the same feature-group ordering. This is descriptive, not a paired RF-vs-Transformer test.

## 11. Circularity implications

POW and Focus originate from the same proprietary headset ecosystem, so shared algorithmic content cannot be excluded. The relative POW-only result indicates that temporal context does not make POW disproportionately predictive: POW-only is the weakest group overall and within both sources. This argues against simple Focus reconstruction from POW alone, but cannot exclude partial proprietary circularity.

## 12. Recommendation for target formulation

Keep global `label_q5` for benchmark comparability and retain leakage-safe labels as sensitivity analysis. Because the classes are ordered and temporal models reduce large errors differently from nominal F1, ordinal classification is the most direct next target-formulation extension.

## 13. Recommendation for next experiment

Future main modeling should retain both EEG-only and EEG+POW variants: EEG-only tests signal-specific validity, while EEG+POW measures the best available feature representation. The matched temporal comparison is now complete; implement an ordinal classifier next, before a regression or joint ordinal-regression Transformer. The combined advantage over EEG-only is subject-level Holm-significant at 0.0483 for BA, macro F1, ordinal MAE and severe error, while the advantage over POW-only is stronger across the same outcomes.

## 14. Limitations

Only seed 42 and one fixed architecture were evaluated. Baseline config sections match: True; maximum normalization-stat delta: 0.00000000; baseline y-pred differences: 0; probability max absolute delta: 0.00000000. One supervised subject has no valid length-8 sequence. Source analyses are descriptive, and no RF-vs-Transformer significance test was performed.
