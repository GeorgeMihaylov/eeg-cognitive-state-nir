# Ordinal Transformer multiseed runs

Seeds 7 and 123 were trained with the canonical seed-42 outer and inner splits. Seed 42 was reused.

New trials: 12 (60 fold-runs). All 18 method/group/seed prediction artifacts align exactly.

## Categorical baseline audit

Existing seed-7/123 EEG+POW runs were excluded when their validation/task split seed differed from 42; comparable baselines were rerun.

| Group | Seed | Eligible | Candidate | Reason |
| --- | ---: | --- | --- | --- |
| eeg_only | 42 | True | `benchmark_results/feature_group_transformer_ablation/runs/transformer_classification_eeg_only/20260718_124023` | comparable |
| eeg_pow | 42 | True | `benchmark_results/feature_group_transformer_ablation/runs/transformer_classification_eeg_pow/20260718_124412` | comparable |
| eeg_pow | 42 | True | `benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246` | comparable |
| eeg_pow | 7 | False | `benchmark_results/groupkfold_torch_transformer_label_q5_seed7/20260716_191618` | inner validation split seed is not 42; outer split seed is not 42; task split seed is not 42 |
| eeg_pow | 123 | False | `benchmark_results/groupkfold_torch_transformer_label_q5_seed123/20260716_191837` | inner validation split seed is not 42; outer split seed is not 42; task split seed is not 42 |
| eeg_only | 123 | True | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_only_seed123/20260718_182015` | comparable |
| eeg_only | 7 | True | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_only_seed7/20260718_181522` | comparable |
| eeg_pow | 123 | True | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_pow_seed123/20260718_182227` | comparable |
| eeg_pow | 7 | True | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_pow_seed7/20260718_181731` | comparable |

## New and reused runs

## Runs

| Method | Feature group | Seed | Run directory |
| --- | --- | ---: | --- |
| categorical | eeg_only | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_only_seed7/20260718_181522` |
| coral | eeg_only | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/coral_eeg_only_seed7/20260718_182450` |
| corn | eeg_only | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/corn_eeg_only_seed7/20260718_183046` |
| categorical | eeg_only | 42 | `benchmark_results/feature_group_transformer_ablation/runs/transformer_classification_eeg_only/20260718_124023` |
| coral | eeg_only | 42 | `benchmark_results/ordinal_transformer_full_seed42/runs/coral_eeg_only/20260718_160001` |
| corn | eeg_only | 42 | `benchmark_results/ordinal_transformer_full_seed42/runs/corn_eeg_only/20260718_160527` |
| categorical | eeg_only | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_only_seed123/20260718_182015` |
| coral | eeg_only | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/coral_eeg_only_seed123/20260718_183557` |
| corn | eeg_only | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/corn_eeg_only_seed123/20260718_184620` |
| categorical | eeg_pow | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_pow_seed7/20260718_181731` |
| coral | eeg_pow | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/coral_eeg_pow_seed7/20260718_182748` |
| corn | eeg_pow | 7 | `benchmark_results/ordinal_transformer_multiseed/runs/corn_eeg_pow_seed7/20260718_183312` |
| categorical | eeg_pow | 42 | `benchmark_results/feature_group_transformer_ablation/runs/transformer_classification_eeg_pow/20260718_124412` |
| coral | eeg_pow | 42 | `benchmark_results/ordinal_transformer_full_seed42/runs/coral_eeg_pow/20260718_160242` |
| corn | eeg_pow | 42 | `benchmark_results/ordinal_transformer_full_seed42/runs/corn_eeg_pow/20260718_160719` |
| categorical | eeg_pow | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/categorical_eeg_pow_seed123/20260718_182227` |
| coral | eeg_pow | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/coral_eeg_pow_seed123/20260718_183829` |
| corn | eeg_pow | 123 | `benchmark_results/ordinal_transformer_multiseed/runs/corn_eeg_pow_seed123/20260718_185223` |

## Split, normalization, probability, and checkpoint audits

All outer subject overlaps and inner record-group overlaps are zero. For every matching feature group/fold, inner validation groups, feature order, normalization mean, and normalization scale match the canonical seed-42 baseline exactly.

All new ordinal checkpoints loaded strictly into a fresh factory model; recomputed predictions match saved predictions within 1e-7. Class probabilities are finite, non-negative, normalized; cumulative probabilities are monotone.

## Aggregate metrics for new trials

| Trial | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: |
| categorical_eeg_only_seed123 | 0.3494 | 0.3531 | 0.4967 | 1.0021 | 0.2593 |
| categorical_eeg_only_seed7 | 0.3436 | 0.3466 | 0.4706 | 1.0320 | 0.2691 |
| categorical_eeg_pow_seed123 | 0.3646 | 0.3676 | 0.5023 | 0.9798 | 0.2486 |
| categorical_eeg_pow_seed7 | 0.3690 | 0.3699 | 0.5092 | 0.9903 | 0.2559 |
| coral_eeg_only_seed123 | 0.3429 | 0.3499 | 0.4923 | 0.9916 | 0.2552 |
| coral_eeg_only_seed7 | 0.3352 | 0.3393 | 0.4823 | 1.0194 | 0.2636 |
| coral_eeg_pow_seed123 | 0.3616 | 0.3684 | 0.5426 | 0.9399 | 0.2330 |
| coral_eeg_pow_seed7 | 0.3617 | 0.3660 | 0.5287 | 0.9482 | 0.2348 |
| corn_eeg_only_seed123 | 0.3314 | 0.3343 | 0.4658 | 1.0137 | 0.2579 |
| corn_eeg_only_seed7 | 0.3423 | 0.3453 | 0.4692 | 1.0019 | 0.2560 |
| corn_eeg_pow_seed123 | 0.3609 | 0.3677 | 0.5296 | 0.9394 | 0.2280 |
| corn_eeg_pow_seed7 | 0.3708 | 0.3747 | 0.5318 | 0.9406 | 0.2287 |

## Training by fold

| Trial | Epochs | Best epochs | Best validation loss | Training seconds |
| --- | --- | --- | --- | ---: |
| categorical_eeg_only_seed123 | 8/7/10/7/12 | 4/3/6/3/8 | 1.3311/1.2264/1.1701/1.2792/1.1184 | 87.0 |
| categorical_eeg_only_seed7 | 12/8/7/7/11 | 8/4/3/3/7 | 1.2790/1.2187/1.2288/1.2767/1.1039 | 82.8 |
| categorical_eeg_pow_seed123 | 8/7/15/6/15 | 4/3/11/2/11 | 1.2317/1.1704/1.1394/1.2479/1.0657 | 97.1 |
| categorical_eeg_pow_seed7 | 15/14/8/8/15 | 11/10/4/4/11 | 1.2014/1.1262/1.1924/1.2120/1.0675 | 118.4 |
| coral_eeg_only_seed123 | 8/10/12/12/12 | 4/6/8/8/8 | 0.4028/0.3878/0.3671/0.4068/0.3328 | 109.6 |
| coral_eeg_only_seed7 | 12/13/11/12/15 | 8/9/7/8/12 | 0.4089/0.3649/0.3633/0.3916/0.3409 | 133.2 |
| coral_eeg_pow_seed123 | 10/13/12/12/15 | 6/9/8/8/11 | 0.3890/0.3321/0.3493/0.3692/0.3110 | 411.0 |
| coral_eeg_pow_seed7 | 8/15/14/7/15 | 4/12/10/3/13 | 0.3786/0.3374/0.3412/0.3832/0.2958 | 131.6 |
| corn_eeg_only_seed123 | 8/7/15/10/8 | 4/3/13/6/4 | 0.4442/0.4546/0.3872/0.4304/0.3755 | 304.3 |
| corn_eeg_only_seed7 | 8/12/8/10/11 | 4/8/4/6/7 | 0.4674/0.4379/0.4025/0.4181/0.3759 | 100.2 |
| corn_eeg_pow_seed123 | 11/15/12/6/7 | 7/12/8/2/3 | 0.4222/0.4220/0.3888/0.4273/0.3666 | 354.9 |
| corn_eeg_pow_seed7 | 11/14/8/9/15 | 7/10/4/5/11 | 0.4208/0.4320/0.3921/0.3956/0.3518 | 119.5 |

## CORAL cutpoints and CORN risk sets

CORAL cutpoints remain strictly ordered in every audited fold. CORN risk counts remain positive and non-increasing across thresholds. Absolute CORAL and CORN loss values are not compared.

## Source- and class-level artifacts

Each ordinal trial manifest stores source-level metrics and per-class precision/recall/F1. Categorical and ordinal unified predictions remain available for the common downstream multiseed analysis.
