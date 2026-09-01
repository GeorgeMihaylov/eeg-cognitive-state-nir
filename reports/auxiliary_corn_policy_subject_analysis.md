# Finalized auxiliary-CORN policy: subject-level analysis

Inference uses one paired observation per subject after averaging seeds 7, 42, and 123. All comparisons use exactly aligned outer-test sequences.

## Policy composition

- Joint auxiliary-CORN units: 25.
- Categorical fallback units: 5.
- Selected lambda counts: {'0.25': 16, '0.5': 5, '1.0': 4}.

## Primary policy-versus-categorical hypotheses

| Group | Metric | Reference mean | Policy mean | Mean improvement | 95% bootstrap CI | Holm p | Improved/degraded/tied |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| eeg_only | ordinal_mae | 1.02612 | 1.02960 | -0.00348 | [-0.02419, 0.01662] | 1 | 29/24/0 |
| eeg_only | severe_error_rate | 0.26974 | 0.26835 | 0.00139 | [-0.00782, 0.01073] | 1 | 29/24/0 |
| eeg_only | balanced_accuracy | 0.33109 | 0.32941 | -0.00167 | [-0.00709, 0.00375] | 1 | 28/25/0 |
| eeg_pow | ordinal_mae | 0.96675 | 0.97013 | -0.00337 | [-0.02014, 0.01299] | 1 | 25/27/1 |
| eeg_pow | severe_error_rate | 0.24573 | 0.24645 | -0.00072 | [-0.00829, 0.00655] | 1 | 25/28/0 |
| eeg_pow | balanced_accuracy | 0.34661 | 0.34289 | -0.00372 | [-0.00922, 0.00167] | 0.5659 | 24/29/0 |

## Secondary paired comparisons

| Group | Candidate | Reference | Metric | Mean improvement | 95% bootstrap CI | Holm p |
| --- | --- | --- | --- | ---: | --- | ---: |
| eeg_only | policy | categorical | macro_f1 | 0.00028 | [-0.00525, 0.00595] | 1 |
| eeg_only | policy | categorical | quadratic_weighted_kappa | -0.00388 | [-0.01449, 0.00694] | 1 |
| eeg_only | policy | categorical | adjacent_accuracy | 0.00139 | [-0.00782, 0.01073] | 1 |
| eeg_only | policy | categorical | expected_rank_mae | -0.01162 | [-0.02719, 0.00345] | 0.97747 |
| eeg_only | policy | categorical | expected_rank_spearman | -0.00627 | [-0.01491, 0.00260] | 0.53087 |
| eeg_only | policy | corn | balanced_accuracy | 0.01186 | [0.00568, 0.01828] | 0.0072987 |
| eeg_only | policy | corn | macro_f1 | 0.00995 | [0.00420, 0.01589] | 0.013649 |
| eeg_only | policy | corn | quadratic_weighted_kappa | -0.00441 | [-0.01510, 0.00663] | 0.45442 |
| eeg_only | policy | corn | ordinal_mae | -0.03273 | [-0.05975, -0.01017] | 0.025401 |
| eeg_only | policy | corn | severe_error_rate | -0.01743 | [-0.02818, -0.00721] | 0.0072987 |
| eeg_pow | policy | categorical | macro_f1 | -0.00036 | [-0.00600, 0.00524] | 1 |
| eeg_pow | policy | categorical | quadratic_weighted_kappa | -0.00190 | [-0.01045, 0.00686] | 1 |
| eeg_pow | policy | categorical | adjacent_accuracy | -0.00072 | [-0.00829, 0.00655] | 1 |
| eeg_pow | policy | categorical | expected_rank_mae | -0.00596 | [-0.01982, 0.00783] | 1 |
| eeg_pow | policy | categorical | expected_rank_spearman | -0.00267 | [-0.00933, 0.00410] | 1 |
| eeg_pow | policy | corn | balanced_accuracy | 0.00742 | [0.00014, 0.01509] | 0.34278 |
| eeg_pow | policy | corn | macro_f1 | 0.00132 | [-0.00582, 0.00862] | 0.92594 |
| eeg_pow | policy | corn | quadratic_weighted_kappa | -0.01088 | [-0.02306, 0.00172] | 0.19476 |
| eeg_pow | policy | corn | ordinal_mae | -0.03722 | [-0.05944, -0.01541] | 0.012979 |
| eeg_pow | policy | corn | severe_error_rate | -0.02053 | [-0.03063, -0.01059] | 0.0016547 |

## Aggregate outer-test metrics by seed

| Method/group | Seed | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| categorical_eeg_only | 7 | 0.3436 | 0.3466 | 0.4706 | 1.0320 | 0.2691 |
| categorical_eeg_pow | 7 | 0.3690 | 0.3699 | 0.5092 | 0.9903 | 0.2559 |
| corn_eeg_only | 7 | 0.3423 | 0.3453 | 0.4692 | 1.0019 | 0.2560 |
| corn_eeg_pow | 7 | 0.3708 | 0.3747 | 0.5318 | 0.9406 | 0.2287 |
| policy_eeg_only | 7 | 0.3430 | 0.3449 | 0.4650 | 1.0367 | 0.2694 |
| policy_eeg_pow | 7 | 0.3650 | 0.3680 | 0.5164 | 0.9778 | 0.2511 |
| categorical_eeg_only | 42 | 0.3451 | 0.3463 | 0.4561 | 1.0565 | 0.2784 |
| categorical_eeg_pow | 42 | 0.3667 | 0.3690 | 0.5057 | 0.9839 | 0.2513 |
| corn_eeg_only | 42 | 0.3307 | 0.3308 | 0.4728 | 1.0096 | 0.2517 |
| corn_eeg_pow | 42 | 0.3605 | 0.3648 | 0.5148 | 0.9628 | 0.2388 |
| policy_eeg_only | 42 | 0.3484 | 0.3514 | 0.4636 | 1.0289 | 0.2648 |
| policy_eeg_pow | 42 | 0.3668 | 0.3690 | 0.5038 | 0.9903 | 0.2532 |
| categorical_eeg_only | 123 | 0.3494 | 0.3531 | 0.4967 | 1.0021 | 0.2593 |
| categorical_eeg_pow | 123 | 0.3646 | 0.3676 | 0.5023 | 0.9798 | 0.2486 |
| corn_eeg_only | 123 | 0.3314 | 0.3343 | 0.4658 | 1.0137 | 0.2579 |
| corn_eeg_pow | 123 | 0.3609 | 0.3677 | 0.5296 | 0.9394 | 0.2280 |
| policy_eeg_only | 123 | 0.3435 | 0.3488 | 0.4644 | 1.0336 | 0.2682 |
| policy_eeg_pow | 123 | 0.3633 | 0.3669 | 0.5100 | 0.9716 | 0.2441 |

## Hard-subject analysis

Hard subjects are the lowest quartile by categorical balanced accuracy within each feature group.

| Group | Subset | Subjects | BA improvement | Ordinal-MAE improvement | Severe-error improvement | Both ordinal metrics improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eeg_only | hard_quartile | 14 | -0.00023 | -0.00142 | 0.00264 | 0.500 |
| eeg_only | remaining_subjects | 39 | -0.00219 | -0.00422 | 0.00094 | 0.513 |
| eeg_pow | hard_quartile | 14 | -0.00637 | -0.02999 | -0.01367 | 0.357 |
| eeg_pow | remaining_subjects | 39 | -0.00277 | 0.00618 | 0.00393 | 0.436 |

## Seed consistency

| Group | Reference | Metric | Seed 7 | Seed 42 | Seed 123 | Positive seeds | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| eeg_only | categorical | balanced_accuracy | -0.00598 | 0.00105 | -0.00009 | 1/3 | changes_sign |
| eeg_only | categorical | macro_f1 | -0.00511 | 0.00910 | -0.00315 | 1/3 | changes_sign |
| eeg_only | categorical | quadratic_weighted_kappa | -0.01622 | 0.01622 | -0.01165 | 1/3 | changes_sign |
| eeg_only | categorical | ordinal_mae | -0.02290 | 0.03534 | -0.02287 | 1/3 | changes_sign |
| eeg_only | categorical | severe_error_rate | -0.00806 | 0.01810 | -0.00587 | 1/3 | changes_sign |
| eeg_only | corn | balanced_accuracy | 0.00455 | 0.01505 | 0.01600 | 3/3 | consistent_positive |
| eeg_only | corn | macro_f1 | -0.00189 | 0.01638 | 0.01535 | 2/3 | changes_sign |
| eeg_only | corn | quadratic_weighted_kappa | -0.01895 | 0.00466 | 0.00106 | 2/3 | changes_sign |
| eeg_only | corn | ordinal_mae | -0.05708 | -0.02392 | -0.01718 | 0/3 | consistent_nonpositive |
| eeg_only | corn | severe_error_rate | -0.02644 | -0.01537 | -0.01047 | 0/3 | consistent_nonpositive |
| eeg_pow | categorical | balanced_accuracy | -0.00429 | -0.00582 | -0.00104 | 0/3 | consistent_nonpositive |
| eeg_pow | categorical | macro_f1 | -0.00204 | -0.00223 | 0.00319 | 1/3 | changes_sign |
| eeg_pow | categorical | quadratic_weighted_kappa | -0.00414 | -0.00992 | 0.00838 | 1/3 | changes_sign |
| eeg_pow | categorical | ordinal_mae | 0.00183 | -0.01931 | 0.00736 | 2/3 | changes_sign |
| eeg_pow | categorical | severe_error_rate | 0.00068 | -0.00824 | 0.00540 | 2/3 | changes_sign |
| eeg_pow | corn | balanced_accuracy | 0.00042 | 0.01854 | 0.00331 | 3/3 | consistent_positive |
| eeg_pow | corn | macro_f1 | -0.00441 | 0.00834 | 0.00002 | 2/3 | changes_sign |
| eeg_pow | corn | quadratic_weighted_kappa | -0.01301 | -0.00204 | -0.01760 | 0/3 | consistent_nonpositive |
| eeg_pow | corn | ordinal_mae | -0.04844 | -0.01805 | -0.04518 | 0/3 | consistent_nonpositive |
| eeg_pow | corn | severe_error_rate | -0.02859 | -0.01200 | -0.02100 | 0/3 | consistent_nonpositive |

## Disagreement and policy-gain associations

These correlations are descriptive and use subject-level seed-averaged values.

| Group | Signal | Outcome | Subjects | Spearman rho | p-value | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| eeg_only | categorical_aux_disagreement_rate | balanced_accuracy_improvement | 53 | -0.0152 | 0.91421 | completed |
| eeg_only | categorical_aux_disagreement_rate | ordinal_mae_improvement | 53 | -0.2076 | 0.13575 | completed |
| eeg_only | categorical_aux_disagreement_rate | severe_error_improvement | 53 | -0.2408 | 0.082446 | completed |
| eeg_only | auxiliary_coverage_fraction | balanced_accuracy_improvement | 53 | 0.1026 | 0.46457 | completed |
| eeg_only | auxiliary_coverage_fraction | ordinal_mae_improvement | 53 | -0.1975 | 0.15626 | completed |
| eeg_only | auxiliary_coverage_fraction | severe_error_improvement | 53 | -0.1656 | 0.23607 | completed |
| eeg_pow | categorical_aux_disagreement_rate | balanced_accuracy_improvement | 53 | 0.0606 | 0.66625 | completed |
| eeg_pow | categorical_aux_disagreement_rate | ordinal_mae_improvement | 53 | 0.1477 | 0.29117 | completed |
| eeg_pow | categorical_aux_disagreement_rate | severe_error_improvement | 53 | 0.1843 | 0.18641 | completed |
| eeg_pow | auxiliary_coverage_fraction | balanced_accuracy_improvement | 53 | -0.1627 | 0.24439 | completed |
| eeg_pow | auxiliary_coverage_fraction | ordinal_mae_improvement | 53 | -0.1677 | 0.22998 | completed |
| eeg_pow | auxiliary_coverage_fraction | severe_error_improvement | 53 | -0.1953 | 0.16119 | completed |

## Fallback units

| Selection unit | Group | Seed | Fold | Outer rows |
| --- | --- | ---: | ---: | ---: |
| eeg_only_seed123_fold01 | eeg_only | 123 | 1 | 8800 |
| eeg_only_seed42_fold02 | eeg_only | 42 | 2 | 8801 |
| eeg_only_seed7_fold01 | eeg_only | 7 | 1 | 8800 |
| eeg_pow_seed123_fold05 | eeg_pow | 123 | 5 | 8826 |
| eeg_pow_seed42_fold01 | eeg_pow | 42 | 1 | 8800 |

## Decision

Primary-feature-group classification: **not_supported**.
The categorical fallback is a post-execution protocol amendment and is reported explicitly. No outer-test result was used for lambda or branch selection.

## Limitations

Seeds are repeated model initializations, not independent subjects. The fallback policy was added after the protective aborts and must not be presented as preregistered. Disagreement associations are descriptive and do not establish a calibrated decision rule for new users.
