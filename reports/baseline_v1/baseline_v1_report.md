# Baseline v1: personal calibration of latent EEG proxy-state trajectories

Generated: `2026-06-17T14:29:55`

## Purpose

This baseline integrates the main experimental line of the project: latent proxy-state targets built from smoothed PM metrics, EEG/POW sequence modeling, Transformer prediction, subject-wise validation, and personal head-only calibration.

The baseline uses the term `proxy-state`: latent targets are derived from PM annotations and should not be interpreted as direct objective measurements of human cognitive-affective state.

## Fixed configuration

| Parameter | Value |
|---|---|
| feature_set | `pow_plus_eeg` |
| seq_len | `8` |
| targets | `slow_pca_1, slow_pca_2, slow_pca_3` |
| calibration_lr | `0.0001` |
| calibration_frac | `0.2` |
| split level | `subject` |
| split seeds | `42, 123, 2024, 3407, 777` |

## Hypothesis baseline matrix

| hypothesis_id | hypothesis                                                                                 | baseline_or_check                                                    | key_result                                                                                          | main_metric                | status                           | caution                                                                                     |
| ---           | ---                                                                                        | ---                                                                  | ---                                                                                                 | ---                        | ---                              | ---                                                                                         |
| H1            | Smoothed PM metrics can be represented as interpretable latent proxy-state axes.           | PM dynamics analysis + slow-PM PCA; exclude unstable axes.           | Final targets are slow_pca_1, slow_pca_2, slow_pca_3; slow_pca_4 excluded as unstable.              |                            | supported within PM proxy labels | Latent axes reflect PM annotations, not direct objective cognitive-affective states.        |
| H2            | Temporal context improves prediction of latent proxy-state trajectories.                   | last_window_mlp / mean_pool_mlp / GRU / Transformer comparison.      | Best zero-shot temporal model: transformer with mean R2=0.2614, Spearman=0.6094.                    | R2=0.2614; Spearman=0.6094 | supported                        | Extreme negative R2 for simple MLPs should be treated as sanity-check degradation.          |
| H3            | POW and EEG features provide complementary information.                                    | pow vs eeg vs pow_plus_eeg feature ablation.                         | pow_plus_eeg selected; test R2=0.2398, Spearman=0.5804.                                             | R2=0.2398; Spearman=0.5804 | supported                        | Feature ablation depends on the current corpus and feature selection procedure.             |
| H4            | Personal head-only calibration improves held-out subject performance.                      | zero-shot vs head-only calibration on held-out subject tail.         | Test mean R2 changed from -0.0530 to 0.2398; gain=0.2928; subject positive-rate=1.0000.             | R2 gain=0.2928             | supported                        | Held-out subject count is limited; do not claim universal improvement.                      |
| H5            | Calibration effect is not an artifact of a single subject-wise split.                      | Fixed protocol evaluated across several random subject-wise splits.  | Across 5 seeds: mean zero R2=-0.0299, mean calibrated R2=0.2085, mean gain=0.2384, gain std=0.1114. | mean R2 gain=0.2384        | supported                        | Gain magnitude remains split-sensitive.                                                     |
| H6            | Transformer + calibration should be compared against simple statistical/persistence rules. | train_mean / subject_calibration_mean / previous_state if available. | Best naive baseline: previous_state / zero_full with mean R2=0.9381, Spearman=0.9637.               | best naive R2=0.9381       | available                        | previous_state uses target history and is a sanity-check, not an EEG-only deployable model. |

## Feature ablation

| feature_set  | selected_seq_len | selected_lr | selected_frac | val_selected_mean_r2 | val_selected_spearman | test_at_selected_mean_r2 | test_at_selected_spearman | test_best_lr | test_best_frac | test_best_mean_r2 | test_best_spearman |
| ---          | ---              | ---         | ---           | ---                  | ---                   | ---                      | ---                       | ---          | ---            | ---               | ---                |
| pow_plus_eeg | 8                | 0.0001      | 0.2000        | 0.0924               | 0.5207                | 0.2398                   | 0.5804                    | 0.0010       | 0.2000         | 0.2410            | 0.5797             |
| eeg          | 8                | 0.0010      | 0.2000        | 0.1292               | 0.5092                | 0.1915                   | 0.5433                    | 0.0001       | 0.2000         | 0.2126            | 0.5537             |
| pow          | 8                | 0.0001      | 0.2000        | 0.1926               | 0.5412                | 0.1410                   | 0.5243                    | 0.0001       | 0.2000         | 0.1410            | 0.5243             |

## Final calibration result

| eval_split | n_subjects | mean_r2_zero | mean_r2_calibrated | mean_r2_gain | subject_mean_r2_positive_rate | target_subject_r2_positive_rate | mean_spearman_zero | mean_spearman_calibrated | mean_spearman_gain |
| ---        | ---        | ---          | ---                | ---          | ---                           | ---                             | ---                | ---                      | ---                |
| test       | 8.0000     | -0.0530      | 0.2398             | 0.2928       | 1.0000                        | 0.8333                          | 0.5478             | 0.5804                   | 0.0326             |

## Temporal architecture baseline

| rank_by_test_r2 | model           | eval_split | phase     | mean_r2   | mean_spearman | mean_mae | mean_rmse |
| ---             | ---             | ---        | ---       | ---       | ---           | ---      | ---       |
| 1               | transformer     | test       | zero_full | 0.2614    | 0.6094        | 0.9295   | 1.1593    |
| 2               | gru             | test       | zero_full | 0.0442    | 0.5231        | 1.0461   | 1.2984    |
| 3               | mean_pool_mlp   | test       | zero_full | -8.2988   | 0.4731        | 1.1381   | 3.8132    |
| 4               | last_window_mlp | test       | zero_full | -122.3203 | 0.4877        | 1.1805   | 11.4887   |

## Split-seed robustness

| eval_split | n_seeds | mean_r2_zero_mean | mean_r2_zero_std | mean_r2_calibrated_mean | mean_r2_calibrated_std | mean_r2_gain_mean | mean_r2_gain_std | mean_spearman_zero_mean | mean_spearman_calibrated_mean | mean_spearman_gain_mean | subject_mean_r2_positive_rate_mean | target_subject_r2_positive_rate_mean |
| ---        | ---     | ---               | ---              | ---                     | ---                    | ---               | ---              | ---                     | ---                           | ---                     | ---                                | ---                                  |
| test       | 5       | -0.0299           | 0.0935           | 0.2085                  | 0.1027                 | 0.2384            | 0.1114           | 0.5388                  | 0.5804                        | 0.0416                  | 0.8500                             | 0.7333                               |
| val        | 5       | -0.2007           | 0.2437           | -0.0298                 | 0.2020                 | 0.1709            | 0.0897           | 0.5296                  | 0.5489                        | 0.0193                  | 0.7000                             | 0.6667                               |

## Optional naive baselines

| eval_split | phase                 | baseline                 | uses_target_history | n_seeds | mean_r2_mean | mean_r2_std | mean_spearman_mean | mean_mae_mean | mean_rmse_mean |
| ---        | ---                   | ---                      | ---                 | ---     | ---          | ---         | ---                | ---           | ---            |
| test       | post_calibration_tail | previous_state           | 1                   | 5       | 0.9380       | 0.0061      | 0.9635             | 0.2455        | 0.3409         |
| test       | post_calibration_tail | train_mean               | 0                   | 5       | -0.0198      | 0.0120      |                    | 1.0473        | 1.3677         |
| test       | post_calibration_tail | subject_calibration_mean | 0                   | 5       | -0.0735      | 0.0239      | 0.1743             | 1.0394        | 1.3990         |
| test       | post_calibration_tail | subject_calibration_last | 0                   | 5       | -0.3298      | 0.1328      | 0.1535             | 1.1738        | 1.5432         |
| test       | zero_full             | previous_state           | 1                   | 5       | 0.9381       | 0.0062      | 0.9637             | 0.2354        | 0.3284         |
| test       | zero_full             | train_mean               | 0                   | 5       | -0.0201      | 0.0168      |                    | 1.0091        | 1.3175         |
| val        | post_calibration_tail | previous_state           | 1                   | 5       | 0.9383       | 0.0035      | 0.9639             | 0.2508        | 0.3461         |
| val        | post_calibration_tail | train_mean               | 0                   | 5       | -0.0192      | 0.0066      |                    | 1.0753        | 1.3946         |
| val        | post_calibration_tail | subject_calibration_mean | 0                   | 5       | -0.1352      | 0.0822      | 0.1206             | 1.1121        | 1.4637         |
| val        | post_calibration_tail | subject_calibration_last | 0                   | 5       | -0.4324      | 0.1263      | 0.0342             | 1.2577        | 1.6438         |
| val        | zero_full             | previous_state           | 1                   | 5       | 0.9389       | 0.0031      | 0.9640             | 0.2426        | 0.3359         |
| val        | zero_full             | train_mean               | 0                   | 5       | -0.0149      | 0.0061      |                    | 1.0439        | 1.3577         |

## Artifact index

| name                    | path                                                                                                           | status      |
| ---                     | ---                                                                                                            | ---         |
| dataset                 | D:\PycharmProjects\eeg-cognitive-state-nir\reports\slow_latent_states\pm_w10\slow_pm_latent_states_w10.parquet | file_exists |
| feature_ablation_dir    | D:\PycharmProjects\eeg-cognitive-state-nir\reports\feature_ablation_v2                                         | dir_exists  |
| subject_diagnostics_dir | D:\PycharmProjects\eeg-cognitive-state-nir\reports\feature_ablation_v2\subject_diagnostics_pow_plus_eeg        | dir_exists  |
| temporal_baselines_dir  | D:\PycharmProjects\eeg-cognitive-state-nir\reports\temporal_baselines\pow_plus_eeg_seq8_pca123                 | dir_exists  |
| final_summary_dir       | D:\PycharmProjects\eeg-cognitive-state-nir\reports\final_experiment_summary                                    | dir_exists  |
| split_seed_dir          | D:\PycharmProjects\eeg-cognitive-state-nir\reports\split_seed_robustness\pow_plus_eeg_seq8_pca123              | dir_exists  |
| naive_dir               | D:\PycharmProjects\eeg-cognitive-state-nir\reports\naive_hypothesis_baselines\pow_plus_eeg_seq8_pca123         | dir_exists  |
| script_44               | D:\PycharmProjects\eeg-cognitive-state-nir\src\44_run_seq_len_sensitivity.py                                   | file_exists |
| script_46               | D:\PycharmProjects\eeg-cognitive-state-nir\src\46_run_reliable_axes_calibration_val_test.py                    | file_exists |
| script_48               | D:\PycharmProjects\eeg-cognitive-state-nir\src\48_train_temporal_baselines.py                                  | file_exists |
| script_49               | D:\PycharmProjects\eeg-cognitive-state-nir\src\49_summarize_final_experiments.py                               | file_exists |
| script_50               | D:\PycharmProjects\eeg-cognitive-state-nir\src\50_run_split_seed_robustness.py                                 | file_exists |
| script_51_optional      | D:\PycharmProjects\eeg-cognitive-state-nir\src\51_run_naive_hypothesis_baselines.py                            | file_exists |

## Commands executed or registered

_No commands executed. The script was run in summarize mode._

## Main interpretation

Baseline v1 supports the cautious claim that, within the current EEG/PM corpora, PM-derived latent proxy-states can be predicted from EEG/POW sequences, and subject-specific head-only calibration improves held-out subject performance on average. The effect is positive across several subject-wise split seeds, although its magnitude remains sensitive to the composition of held-out subjects.

## Limitations

- Targets are PM-derived proxy-states, not direct objective cognitive-affective measurements.
- Available corpora are close in device/protocol characteristics; cross-device generalization is not established.
- Head-only calibration uses part of the held-out subject sequence and is not a pure zero-shot setting.
- Feature-ablation outputs are summarized here but not fully reproduced unless upstream artifacts already exist.
- Naive baselines are optional because script 51 may be absent from the current branch.
