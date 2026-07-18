# Ordinal Transformer multiseed statistics

The inferential unit is one subject. Seeds are repeated initializations and were averaged within each of 53 subjects before paired inference.

## Primary hypotheses

| Group | Head | Metric | Mean improvement | 95% bootstrap CI | Holm p | Improved/degraded/tied | Positive seeds |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| eeg_only | coral | ordinal_mae | 0.03719 | [0.01653, 0.05822] | 0.0094595 | 33/20/0 | 3/3 |
| eeg_only | coral | severe_error_rate | 0.01592 | [0.00554, 0.02653] | 0.0094595 | 35/18/0 | 3/3 |
| eeg_only | corn | ordinal_mae | 0.02925 | [0.00636, 0.05264] | 0.011489 | 37/16/0 | 2/3 |
| eeg_only | corn | severe_error_rate | 0.01882 | [0.00805, 0.02932] | 0.0053255 | 38/15/0 | 3/3 |
| eeg_pow | coral | ordinal_mae | 0.02288 | [0.00536, 0.04053] | 0.026899 | 33/19/1 | 2/3 |
| eeg_pow | coral | severe_error_rate | 0.01221 | [0.00461, 0.01999] | 0.011871 | 36/17/0 | 2/3 |
| eeg_pow | corn | ordinal_mae | 0.03385 | [0.01237, 0.05698] | 0.011963 | 37/16/0 | 2/3 |
| eeg_pow | corn | severe_error_rate | 0.01981 | [0.00995, 0.03044] | 0.0030234 | 39/14/0 | 3/3 |

## Aggregate metrics by seed

| Method/group | Seed | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| categorical_eeg_only | 7 | 0.3436 | 0.3466 | 0.4706 | 1.0320 | 0.2691 |
| categorical_eeg_pow | 7 | 0.3690 | 0.3699 | 0.5092 | 0.9903 | 0.2559 |
| coral_eeg_only | 7 | 0.3352 | 0.3393 | 0.4823 | 1.0194 | 0.2636 |
| coral_eeg_pow | 7 | 0.3617 | 0.3660 | 0.5287 | 0.9482 | 0.2348 |
| corn_eeg_only | 7 | 0.3423 | 0.3453 | 0.4692 | 1.0019 | 0.2560 |
| corn_eeg_pow | 7 | 0.3708 | 0.3747 | 0.5318 | 0.9406 | 0.2287 |
| categorical_eeg_only | 42 | 0.3451 | 0.3463 | 0.4561 | 1.0565 | 0.2784 |
| categorical_eeg_pow | 42 | 0.3667 | 0.3690 | 0.5057 | 0.9839 | 0.2513 |
| coral_eeg_only | 42 | 0.3386 | 0.3434 | 0.4941 | 1.0058 | 0.2649 |
| coral_eeg_pow | 42 | 0.3541 | 0.3550 | 0.5026 | 1.0001 | 0.2575 |
| corn_eeg_only | 42 | 0.3307 | 0.3308 | 0.4728 | 1.0096 | 0.2517 |
| corn_eeg_pow | 42 | 0.3605 | 0.3648 | 0.5148 | 0.9628 | 0.2388 |
| categorical_eeg_only | 123 | 0.3494 | 0.3531 | 0.4967 | 1.0021 | 0.2593 |
| categorical_eeg_pow | 123 | 0.3646 | 0.3676 | 0.5023 | 0.9798 | 0.2486 |
| coral_eeg_only | 123 | 0.3429 | 0.3499 | 0.4923 | 0.9916 | 0.2552 |
| coral_eeg_pow | 123 | 0.3616 | 0.3684 | 0.5426 | 0.9399 | 0.2330 |
| corn_eeg_only | 123 | 0.3314 | 0.3343 | 0.4658 | 1.0137 | 0.2579 |
| corn_eeg_pow | 123 | 0.3609 | 0.3677 | 0.5296 | 0.9394 | 0.2280 |

## Secondary hypotheses

| Group | Head | Metric | Mean improvement | 95% bootstrap CI | Holm p |
| --- | --- | --- | ---: | --- | ---: |
| eeg_only | coral | balanced_accuracy | -0.01028 | [-0.01937, -0.00218] | 0.18106 |
| eeg_only | coral | macro_f1 | 0.00148 | [-0.00606, 0.00901] | 1 |
| eeg_only | coral | quadratic_weighted_kappa | 0.01319 | [0.00056, 0.02583] | 0.22267 |
| eeg_only | coral | adjacent_accuracy | 0.01592 | [0.00554, 0.02653] | 0.038472 |
| eeg_only | coral | expected_rank_mae | 0.00203 | [-0.01349, 0.01757] | 1 |
| eeg_only | coral | expected_rank_spearman | -0.00988 | [-0.02024, 0.00050] | 0.26968 |
| eeg_only | corn | balanced_accuracy | -0.01354 | [-0.02045, -0.00692] | 0.0085047 |
| eeg_only | corn | macro_f1 | -0.00966 | [-0.01621, -0.00335] | 0.12314 |
| eeg_only | corn | quadratic_weighted_kappa | 0.00052 | [-0.01133, 0.01219] | 1 |
| eeg_only | corn | adjacent_accuracy | 0.01882 | [0.00805, 0.02932] | 0.014645 |
| eeg_only | corn | expected_rank_mae | -0.00983 | [-0.02740, 0.00781] | 0.72658 |
| eeg_only | corn | expected_rank_spearman | -0.00690 | [-0.01716, 0.00278] | 0.97747 |
| eeg_pow | coral | balanced_accuracy | -0.00813 | [-0.02025, 0.00155] | 0.76134 |
| eeg_pow | coral | macro_f1 | 0.00172 | [-0.00660, 0.00973] | 1 |
| eeg_pow | coral | quadratic_weighted_kappa | 0.01281 | [0.00163, 0.02399] | 0.35608 |
| eeg_pow | coral | adjacent_accuracy | 0.01221 | [0.00461, 0.01999] | 0.044052 |
| eeg_pow | coral | expected_rank_mae | 0.00064 | [-0.01331, 0.01480] | 1 |
| eeg_pow | coral | expected_rank_spearman | -0.00826 | [-0.01852, 0.00156] | 1 |
| eeg_pow | corn | balanced_accuracy | -0.01114 | [-0.01901, -0.00380] | 0.13021 |
| eeg_pow | corn | macro_f1 | -0.00168 | [-0.00913, 0.00554] | 1 |
| eeg_pow | corn | quadratic_weighted_kappa | 0.00899 | [-0.00294, 0.02076] | 1 |
| eeg_pow | corn | adjacent_accuracy | 0.01981 | [0.00995, 0.03044] | 0.0096504 |
| eeg_pow | corn | expected_rank_mae | 0.00859 | [-0.00850, 0.02608] | 1 |
| eeg_pow | corn | expected_rank_spearman | -0.00429 | [-0.01458, 0.00505] | 1 |

## Seed consistency

| Group | Head | Metric | Seed 7 | Seed 42 | Seed 123 | Positive seeds | Label |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| eeg_only | coral | ordinal_mae | 0.01986 | 0.06141 | 0.03031 | 3/3 | positive_in_3_of_3 |
| eeg_only | coral | severe_error_rate | 0.01070 | 0.02261 | 0.01445 | 3/3 | positive_in_3_of_3 |
| eeg_only | coral | balanced_accuracy | -0.00967 | -0.01354 | -0.00764 | 0/3 | positive_in_0_of_3 |
| eeg_only | coral | macro_f1 | -0.00058 | 0.00568 | -0.00067 | 1/3 | changes_sign |
| eeg_only | coral | quadratic_weighted_kappa | 0.01193 | 0.01554 | 0.01211 | 3/3 | positive_in_3_of_3 |
| eeg_only | coral | adjacent_accuracy | 0.01070 | 0.02261 | 0.01445 | 3/3 | positive_in_3_of_3 |
| eeg_only | coral | expected_rank_mae | -0.01254 | 0.01141 | 0.00723 | 2/3 | changes_sign |
| eeg_only | coral | expected_rank_spearman | -0.00808 | -0.02070 | -0.00085 | 0/3 | positive_in_0_of_3 |
| eeg_only | corn | ordinal_mae | 0.03418 | 0.05926 | -0.00569 | 2/3 | changes_sign |
| eeg_only | corn | severe_error_rate | 0.01837 | 0.03348 | 0.00461 | 3/3 | positive_in_3_of_3 |
| eeg_only | corn | balanced_accuracy | -0.01052 | -0.01399 | -0.01609 | 0/3 | positive_in_0_of_3 |
| eeg_only | corn | macro_f1 | -0.00321 | -0.00728 | -0.01850 | 0/3 | positive_in_0_of_3 |
| eeg_only | corn | quadratic_weighted_kappa | 0.00272 | 0.01156 | -0.01271 | 2/3 | changes_sign |
| eeg_only | corn | adjacent_accuracy | 0.01837 | 0.03348 | 0.00461 | 3/3 | positive_in_3_of_3 |
| eeg_only | corn | expected_rank_mae | -0.00683 | 0.00411 | -0.02676 | 1/3 | changes_sign |
| eeg_only | corn | expected_rank_spearman | -0.00354 | -0.01017 | -0.00700 | 0/3 | positive_in_0_of_3 |
| eeg_pow | coral | ordinal_mae | 0.04003 | -0.01936 | 0.04797 | 2/3 | changes_sign |
| eeg_pow | coral | severe_error_rate | 0.02453 | -0.00648 | 0.01860 | 2/3 | changes_sign |
| eeg_pow | coral | balanced_accuracy | -0.01089 | -0.01251 | -0.00099 | 0/3 | positive_in_0_of_3 |
| eeg_pow | coral | macro_f1 | -0.00087 | -0.00647 | 0.01251 | 1/3 | changes_sign |
| eeg_pow | coral | quadratic_weighted_kappa | 0.01427 | -0.00524 | 0.02941 | 2/3 | changes_sign |
| eeg_pow | coral | adjacent_accuracy | 0.02453 | -0.00648 | 0.01860 | 2/3 | changes_sign |
| eeg_pow | coral | expected_rank_mae | 0.00828 | -0.03300 | 0.02664 | 2/3 | changes_sign |
| eeg_pow | coral | expected_rank_spearman | 0.00232 | -0.02268 | -0.00443 | 1/3 | changes_sign |
| eeg_pow | corn | ordinal_mae | 0.05026 | -0.00125 | 0.05254 | 2/3 | changes_sign |
| eeg_pow | corn | severe_error_rate | 0.02928 | 0.00376 | 0.02639 | 3/3 | positive_in_3_of_3 |
| eeg_pow | corn | balanced_accuracy | -0.00471 | -0.02436 | -0.00435 | 0/3 | positive_in_0_of_3 |
| eeg_pow | corn | macro_f1 | 0.00238 | -0.01057 | 0.00317 | 2/3 | changes_sign |
| eeg_pow | corn | quadratic_weighted_kappa | 0.00887 | -0.00788 | 0.02597 | 2/3 | changes_sign |
| eeg_pow | corn | adjacent_accuracy | 0.02928 | 0.00376 | 0.02639 | 3/3 | positive_in_3_of_3 |
| eeg_pow | corn | expected_rank_mae | 0.01691 | -0.02236 | 0.03123 | 2/3 | changes_sign |
| eeg_pow | corn | expected_rank_spearman | -0.01253 | -0.01229 | 0.01194 | 1/3 | changes_sign |

## Subject heterogeneity and hard subjects

Primary comparison rows include the 10th/25th/50th/75th/90th percentiles, worst- and best-quartile means, and improved/degraded/tied subject counts. Hard-subject summaries use the lowest categorical baseline quartile within each feature group.

| Group | Candidate | Difficulty quartile | Subjects | Ordinal-MAE improvement | Severe-error improvement | Fraction improved |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| eeg_only | coral | worst_quartile | 14 | 0.08021 | 0.03000 | 0.714 |
| eeg_only | coral | best_quartile | 14 | 0.02293 | 0.00695 | 0.571 |
| eeg_only | corn | worst_quartile | 14 | 0.03124 | 0.01277 | 0.571 |
| eeg_only | corn | best_quartile | 14 | -0.00798 | 0.00368 | 0.714 |
| eeg_pow | coral | worst_quartile | 14 | 0.06853 | 0.02547 | 0.714 |
| eeg_pow | coral | best_quartile | 14 | -0.00651 | 0.00588 | 0.500 |
| eeg_pow | corn | worst_quartile | 14 | 0.07663 | 0.03308 | 0.714 |
| eeg_pow | corn | best_quartile | 14 | -0.01209 | 0.00350 | 0.429 |

## BA and macro-F1 trade-offs

| Group | Head | Quality metric | Ordinal+quality improved | Ordinal improved/quality degraded | Ordinal degraded/quality improved | Both degraded | Ties |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| eeg_only | coral | balanced_accuracy | 15 | 18 | 8 | 12 | 0 |
| eeg_only | coral | macro_f1 | 22 | 11 | 6 | 14 | 0 |
| eeg_only | corn | balanced_accuracy | 18 | 19 | 2 | 14 | 0 |
| eeg_only | corn | macro_f1 | 20 | 17 | 1 | 15 | 0 |
| eeg_pow | coral | balanced_accuracy | 17 | 16 | 2 | 17 | 1 |
| eeg_pow | coral | macro_f1 | 24 | 9 | 4 | 15 | 1 |
| eeg_pow | corn | balanced_accuracy | 19 | 18 | 2 | 14 | 0 |
| eeg_pow | corn | macro_f1 | 23 | 14 | 2 | 14 | 0 |

## Feature-group interpretation

EEG+POW is the primary feature group and EEG-only is the control. Effects are reported separately; a benefit confined to EEG-only is not interpreted as a universal ordinal-head advantage.

## Source- and class-level results

Source-level and per-class metrics for all 18 runs are stored in the JSON summary. They are descriptive because sources and classes are not independent inferential units.

## Limitations

Seeds are not independent people; source/fold views are descriptive; three seeds do not characterize the full initialization distribution.
