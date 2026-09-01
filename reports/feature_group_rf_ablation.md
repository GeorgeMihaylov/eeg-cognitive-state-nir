# Random Forest feature-group and regression audit

## 1. Objective

Compare EEG-only, headset spectral-power (POW)-only, and EEG+POW inputs under identical five-fold subject GroupKFold splits for global `label_q5` classification and continuous `target_focus` regression.

## 2. Feature-group definitions

| Group | Features | Ordered-list SHA-256 | Valid |
| --- | ---: | --- | --- |
| eeg_only | 168 | `6e822ee172422e7138945b47b2b27c947393b828b72d96b7a8e22850aded8aca` | True |
| pow_only | 280 | `c3106631e4ad3eff1694c874a9ff5c4e26470aa587c7294cc5c3047c53d832dd` | True |
| eeg_pow | 448 | `8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51` | True |

EEG features are deterministic `EEG.*` window statistics. POW features are deterministic `POW.*` headset spectral-power columns aggregated by mean/std/min/max; they are not Performance Metrics (`PM.*`). The full ordered lists are stored in the JSON report and fold feature manifests.

## 3. Leakage checks

- Supervised windows: 45384; subjects: 54.
- Targets, PM fields, subject/record/source/sample identifiers, and time metadata are absent from all model feature lists.
- Every outer fold has zero train/test subject overlap.
- The global `label_q5` is used unchanged; leakage-safe sensitivity labels are not trained on here.

## 4. Exact alignment

All classification groups, all regression groups, and matching classification/regression trials have exact sample/fold/subject/record/source alignment. Total mismatches: **0**. Canonical baseline alignment is True.

## 5. Classification results

| Group | Balanced accuracy | Macro F1 | Accuracy | Weighted F1 | Kappa | AUC | Ordinal MAE | Adjacent accuracy | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only | 0.2928 +/- 0.0249 | 0.2829 +/- 0.0202 | 0.2903 | 0.2841 | 0.1153 | 0.6081 | 1.3113 | 0.6224 | 0.3776 |
| pow_only | 0.2731 +/- 0.0243 | 0.2610 +/- 0.0209 | 0.2666 | 0.2591 | 0.0878 | 0.5882 | 1.3743 | 0.5972 | 0.4028 |
| eeg_pow | 0.3059 +/- 0.0255 | 0.2955 +/- 0.0217 | 0.3021 | 0.2953 | 0.1297 | 0.6219 | 1.2685 | 0.6371 | 0.3629 |

EEG-only retains 95.7% of combined balanced accuracy; POW-only retains 89.2%. Adding EEG to POW changes balanced accuracy by +0.0329.

## 6. Regression results

| Group | MAE | RMSE | R2 | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| eeg_only | 0.0925 +/- 0.0060 | 0.1185 +/- 0.0076 | 0.0870 | 0.3347 | 0.3084 |
| pow_only | 0.0937 +/- 0.0059 | 0.1198 +/- 0.0075 | 0.0696 | 0.3028 | 0.2834 |
| eeg_pow | 0.0902 +/- 0.0064 | 0.1160 +/- 0.0082 | 0.1265 | 0.3836 | 0.3694 |

Lowest fold-mean MAE: **eeg_pow**. Highest fold-mean Spearman: **eeg_pow**.

## 7. Regression-to-class results

Fixed global thresholds `[0.330177, 0.387786, 0.444458, 0.526585]` were applied without refitting. This is a diagnostic comparison, not an optimized ordinal method.

| Group | Quantized BA | Quantized macro F1 | Quantized ordinal MAE | Adjacent accuracy | Severe error | Direct BA / ordinal MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only | 0.2422 | 0.2020 | 1.1522 | 0.6731 | 0.3269 | 0.2928 / 1.3113 |
| pow_only | 0.2329 | 0.1811 | 1.1656 | 0.6665 | 0.3335 | 0.2731 / 1.3743 |
| eeg_pow | 0.2505 | 0.2140 | 1.1108 | 0.7018 | 0.2982 | 0.3059 / 1.2685 |

## 8. Subject-level comparisons

Positive differences always favor the left group; error differences are oriented as `right_error - left_error`. CIs use 10,000 paired subject bootstraps. Holm correction is separate for classification and regression families.

| Task | Comparison | Metric | Mean Delta | Median Delta | 95% CI | Improved/degraded/ties | Wilcoxon p | Holm p | Sign p | Holm sign p | Rank-biserial |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| classification | eeg_only_vs_pow_only | balanced_accuracy | 0.0021 | 0.0118 | [-0.0290, 0.0231] | 32 / 22 / 0 | 0.0501 | 0.2005 | 0.2203 | 0.6610 | 0.3064 |
| classification | eeg_only_vs_pow_only | macro_f1 | 0.0088 | 0.0108 | [-0.0127, 0.0268] | 29 / 25 / 0 | 0.0754 | 0.2262 | 0.6835 | 0.6835 | 0.2781 |
| classification | eeg_only_vs_pow_only | ordinal_mae | 0.0564 | 0.0397 | [-0.0476, 0.1515] | 33 / 21 / 0 | 0.1444 | 0.2262 | 0.1337 | 0.5347 | 0.2283 |
| classification | eeg_only_vs_eeg_pow | balanced_accuracy | -0.0072 | -0.0058 | [-0.0145, 0.0001] | 22 / 31 / 1 | 0.0804 | 0.2262 | 0.2717 | 0.6610 | -0.2760 |
| classification | eeg_only_vs_eeg_pow | macro_f1 | -0.0101 | -0.0142 | [-0.0182, -0.0018] | 18 / 35 / 1 | 0.0151 | 0.0906 | 0.0270 | 0.1891 | -0.3836 |
| classification | eeg_only_vs_eeg_pow | ordinal_mae | -0.0342 | -0.0316 | [-0.0652, -0.0041] | 20 / 33 / 1 | 0.0304 | 0.1521 | 0.0984 | 0.4919 | -0.3417 |
| classification | pow_only_vs_eeg_pow | balanced_accuracy | -0.0093 | -0.0221 | [-0.0294, 0.0217] | 17 / 37 / 0 | 0.0005 | 0.0043 | 0.0091 | 0.0726 | -0.5461 |
| classification | pow_only_vs_eeg_pow | macro_f1 | -0.0189 | -0.0315 | [-0.0359, 0.0027] | 20 / 34 / 0 | 0.0009 | 0.0063 | 0.0759 | 0.4554 | -0.5192 |
| classification | pow_only_vs_eeg_pow | ordinal_mae | -0.0907 | -0.0909 | [-0.1761, 0.0066] | 16 / 38 / 0 | 0.0007 | 0.0060 | 0.0038 | 0.0345 | -0.5273 |
| regression | eeg_only_vs_pow_only | mae | 0.0016 | 0.0014 | [-0.0014, 0.0047] | 32 / 22 / 0 | 0.2799 | 0.2799 | 0.2203 | 0.2203 | 0.1690 |
| regression | eeg_only_vs_pow_only | spearman | 0.0782 | 0.0451 | [0.0195, 0.1524] | 35 / 19 / 0 | 0.0173 | 0.0518 | 0.0402 | 0.0804 | 0.3724 |
| regression | eeg_only_vs_eeg_pow | mae | -0.0018 | -0.0018 | [-0.0033, -0.0001] | 16 / 38 / 0 | 0.0075 | 0.0300 | 0.0038 | 0.0192 | -0.4182 |
| regression | eeg_only_vs_eeg_pow | spearman | -0.0063 | -0.0377 | [-0.0428, 0.0438] | 17 / 37 / 0 | 0.0213 | 0.0518 | 0.0091 | 0.0363 | -0.3603 |
| regression | pow_only_vs_eeg_pow | mae | -0.0034 | -0.0041 | [-0.0063, -0.0000] | 11 / 43 / 0 | 0.0001 | 0.0007 | 0.0000 | 0.0001 | -0.5960 |
| regression | pow_only_vs_eeg_pow | spearman | -0.0844 | -0.0799 | [-0.1229, -0.0462] | 17 / 37 / 0 | 0.0000 | 0.0002 | 0.0091 | 0.0363 | -0.6471 |

## 9. Source-level results

Source results are descriptive. A person present in both sources is not counted as two independent units in paired tests.

| Task | Group | Source | Windows | Subjects | Metrics |
| --- | --- | --- | ---: | ---: | --- |
| classification | eeg_only | Old_EEG | 21558 | 43 | BA=0.2876; macro-F1=0.2799; ordinal-MAE=1.3282 |
| classification | eeg_only | gpn_data | 23826 | 42 | BA=0.2925; macro-F1=0.2875; ordinal-MAE=1.2963 |
| classification | pow_only | Old_EEG | 21558 | 43 | BA=0.2665; macro-F1=0.2632; ordinal-MAE=1.3872 |
| classification | pow_only | gpn_data | 23826 | 42 | BA=0.2658; macro-F1=0.2649; ordinal-MAE=1.3629 |
| classification | eeg_pow | Old_EEG | 21558 | 43 | BA=0.3026; macro-F1=0.2954; ordinal-MAE=1.2773 |
| classification | eeg_pow | gpn_data | 23826 | 42 | BA=0.3013; macro-F1=0.2970; ordinal-MAE=1.2610 |
| regression | eeg_only | Old_EEG | 21558 | 43 | MAE=0.0952; Spearman=0.3009; quantized ordinal-MAE=1.1588 |
| regression | eeg_only | gpn_data | 23826 | 42 | MAE=0.0901; Spearman=0.2862; quantized ordinal-MAE=1.1466 |
| regression | pow_only | Old_EEG | 21558 | 43 | MAE=0.0969; Spearman=0.2609; quantized ordinal-MAE=1.1780 |
| regression | pow_only | gpn_data | 23826 | 42 | MAE=0.0908; Spearman=0.2717; quantized ordinal-MAE=1.1546 |
| regression | eeg_pow | Old_EEG | 21558 | 43 | MAE=0.0929; Spearman=0.3570; quantized ordinal-MAE=1.1200 |
| regression | eeg_pow | gpn_data | 23826 | 42 | MAE=0.0878; Spearman=0.3524; quantized ordinal-MAE=1.1029 |

Classification source gaps are at most about 0.005 balanced-accuracy points. Old_EEG regression MAE is about 0.005-0.006 higher than gpn_data, but the feature-group ranking is unchanged; there is no strong qualitative source reversal.

## 10. Feature importance

| Task | Group | Feature | Mean +/- SD | Folds in top-20 | Family | Channel | Band |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| classification | eeg_only | `EEG.T7__robust_iqr` | 0.018057 +/- 0.001732 | 5 | EEG | T7 |  |
| classification | eeg_only | `EEG.F3__std` | 0.015831 +/- 0.003608 | 5 | EEG | F3 |  |
| classification | eeg_only | `EEG.T7__std` | 0.013048 +/- 0.001441 | 5 | EEG | T7 |  |
| classification | eeg_only | `EEG.F3__zero_crossing_rate` | 0.011143 +/- 0.002145 | 5 | EEG | F3 |  |
| classification | eeg_only | `EEG.F3__robust_iqr` | 0.010752 +/- 0.002222 | 4 | EEG | F3 |  |
| classification | pow_only | `POW.T7.Theta__max` | 0.010491 +/- 0.002016 | 5 | POW | T7 | Theta |
| classification | pow_only | `POW.T7.Theta__mean` | 0.009176 +/- 0.002718 | 5 | POW | T7 | Theta |
| classification | pow_only | `POW.T7.Theta__std` | 0.008159 +/- 0.001497 | 5 | POW | T7 | Theta |
| classification | pow_only | `POW.F8.Alpha__mean` | 0.007785 +/- 0.001753 | 4 | POW | F8 | Alpha |
| classification | pow_only | `POW.AF4.Alpha__mean` | 0.007126 +/- 0.000634 | 5 | POW | AF4 | Alpha |
| classification | eeg_pow | `EEG.T7__robust_iqr` | 0.012704 +/- 0.001721 | 5 | EEG | T7 |  |
| classification | eeg_pow | `EEG.F3__std` | 0.009360 +/- 0.001748 | 5 | EEG | F3 |  |
| classification | eeg_pow | `EEG.T7__std` | 0.007919 +/- 0.000973 | 5 | EEG | T7 |  |
| classification | eeg_pow | `POW.F8.Alpha__mean` | 0.006620 +/- 0.001318 | 5 | POW | F8 | Alpha |
| classification | eeg_pow | `EEG.F3__robust_iqr` | 0.005721 +/- 0.000896 | 5 | EEG | F3 |  |
| regression | eeg_only | `EEG.T7__robust_iqr` | 0.096274 +/- 0.048789 | 5 | EEG | T7 |  |
| regression | eeg_only | `EEG.F3__robust_iqr` | 0.086372 +/- 0.056531 | 4 | EEG | F3 |  |
| regression | eeg_only | `EEG.T7__std` | 0.040918 +/- 0.025233 | 5 | EEG | T7 |  |
| regression | eeg_only | `EEG.O2__mean_abs_diff` | 0.038133 +/- 0.017457 | 5 | EEG | O2 |  |
| regression | eeg_only | `EEG.AF4__mean_abs_diff` | 0.016408 +/- 0.005973 | 5 | EEG | AF4 |  |
| regression | pow_only | `POW.T7.Theta__std` | 0.053370 +/- 0.030739 | 4 | POW | T7 | Theta |
| regression | pow_only | `POW.T7.Theta__max` | 0.037874 +/- 0.026451 | 5 | POW | T7 | Theta |
| regression | pow_only | `POW.F8.Alpha__mean` | 0.031005 +/- 0.017653 | 4 | POW | F8 | Alpha |
| regression | pow_only | `POW.O2.BetaH__mean` | 0.023056 +/- 0.012027 | 5 | POW | O2 | BetaH |
| regression | pow_only | `POW.O2.Alpha__mean` | 0.020181 +/- 0.017360 | 5 | POW | O2 | Alpha |
| regression | eeg_pow | `EEG.T7__robust_iqr` | 0.084924 +/- 0.044774 | 5 | EEG | T7 |  |
| regression | eeg_pow | `EEG.F3__robust_iqr` | 0.074165 +/- 0.050671 | 4 | EEG | F3 |  |
| regression | eeg_pow | `EEG.T7__std` | 0.031896 +/- 0.026707 | 5 | EEG | T7 |  |
| regression | eeg_pow | `POW.F8.Alpha__mean` | 0.030439 +/- 0.014129 | 5 | POW | F8 | Alpha |
| regression | eeg_pow | `POW.AF4.Alpha__mean` | 0.014937 +/- 0.005353 | 5 | POW | AF4 | Alpha |

Importance is impurity-based, descriptive, and not used for feature selection. Correlated EEG/POW variables can divide or inflate importance. Optional permutation importance was not run to keep this six-trial audit bounded.

## 11. Circularity implications

POW and Focus are both exported by the same headset ecosystem. The source files identify POW as spectral-power features and PM.Focus as a separate proprietary metric, but the vendor's Focus computation is not available. Strong POW performance would therefore be compatible with shared signal content or partial algorithmic circularity; it would not prove either causality or direct leakage. Here POW-only is weaker than EEG-only, and adding EEG improves POW on classification and regression, so the results do not look like trivial reconstruction of Focus from POW alone. Proprietary-algorithm circularity nevertheless cannot be excluded.

## 12. Limitations

This audit uses one fixed RF configuration and one global label definition. RF importance is biased toward correlated/high-variance features, source analyses are descriptive, and regression quantization is not a trained ordinal objective.

## 13. Recommendation for Transformer experiment

Run the next Transformer comparison on **eeg_pow** and **eeg_only**, the two groups with the best combined classification-BA and regression-MAE ranks. Do not launch it automatically; retain the third group only if the subject-level or circularity interpretation requires a dedicated control.
