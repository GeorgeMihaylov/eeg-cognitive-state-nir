# H120 history-order control

Conditional confirmatory subset: excitement, focus, interest, relaxation, stress. Five fixed outer folds, H120 strict common support; exactly 25 shuffled fits. Ordered H120 reused without retraining.

Implementation commit: 6d332f7211cb97455d7229949dd01f1ca7615d26. Tests: 31 passed. Dry-run: 25 planned fits, no training. Execution: 25/25; executed 25, restored 0, failed 0.
Runtime: 1337.11 s (22.29 min); summed model fit/predict: 954.65 s.

## Frozen protocol

LightGBM: n_estimators=200, n_jobs=4, random_state=42, no tuning. Fold-local Q33/Q67 fitted on canonical outer-training continuous complete cases before history filtering. Ordered/shuffled thresholds, samples, labels, subjects, folds and counts match. Exact record-relative 10-second lookup; no gaps, substitution, future windows or record crossing.
Existing history_matrix shuffle permutes only the 11 older blocks (-120 through -20 s), deterministically per target sample. Canonical -10 s block stays last. Saved provenance and prediction identity were checked for all 25 cells; ordered reference checksums remain unchanged.

## Primary results

Primary metric: participant-macro balanced accuracy. Delta = ordered H120 minus shuffled H120. Paired participant percentile 95% CI, 10,000 resamples, seed 42; two-sided centered-null bootstrap p with plus-one correction; BH across exactly five BA contrasts.

| pm | n_subjects | ordered_ba | shuffled_ba | delta_ba | ci95_low | ci95_high | bootstrap_two_sided_p | bh_q_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excitement | 53 | 0.860613 | 0.853213 | 0.007400 | 0.002297 | 0.012736 | 0.006799 | 0.008499 |
| focus | 53 | 0.812677 | 0.733485 | 0.079192 | 0.065372 | 0.093650 | 0.000100 | 0.000250 |
| interest | 53 | 0.762386 | 0.753800 | 0.008586 | 0.002551 | 0.014468 | 0.006199 | 0.008499 |
| relaxation | 53 | 0.802520 | 0.788174 | 0.014346 | 0.008500 | 0.020517 | 0.000100 | 0.000250 |
| stress | 53 | 0.773203 | 0.763517 | 0.009687 | -0.000739 | 0.020272 | 0.070793 | 0.070793 |

Ordered BA exceeds shuffled BA for all five PM. The five-contrast BH correction retains excitement, focus, interest and relaxation (q<0.05); stress does not pass (q=0.070793). This supports a contribution from older-block order for four PM under this representation and fixed control. It does not establish that all H120 benefit is due to order, nor that stress is order-invariant.

## Consistency

| pm | participants_ordered_better | participants_shuffled_better | participants_tied | folds_ordered_better | folds_shuffled_better | fold_delta_mean | fold_delta_sd | fold_delta_min | fold_delta_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excitement | 30 | 21 | 2 | 4 | 1 | 0.007409 | 0.006228 | -0.001607 | 0.014984 |
| focus | 52 | 1 | 0 | 5 | 0 | 0.079665 | 0.012195 | 0.067575 | 0.095711 |
| interest | 35 | 16 | 2 | 4 | 1 | 0.008707 | 0.010333 | -0.006976 | 0.018908 |
| relaxation | 35 | 12 | 6 | 5 | 0 | 0.014408 | 0.006104 | 0.003682 | 0.018751 |
| stress | 35 | 18 | 0 | 4 | 1 | 0.009713 | 0.014169 | -0.009885 | 0.028894 |

## Secondary descriptive metrics

Participant-macro metrics; paired differences use the same sign. No secondary multiplicity claim.

| pm | metric | ordered | shuffled | status | n_subjects | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excitement | macro_f1 | 0.842068 | 0.835353 | ok | 53 | 0.006714 | 0.000446 | 0.013490 |
| excitement | roc_auc | 0.960151 | 0.955034 | ok | 53 | 0.005118 | 0.002902 | 0.007435 |
| excitement | pr_auc | 0.961274 | 0.954249 | ok | 53 | 0.007025 | 0.003764 | 0.010922 |
| focus | macro_f1 | 0.781416 | 0.702804 | ok | 53 | 0.078612 | 0.065064 | 0.092077 |
| focus | roc_auc | 0.940895 | 0.878115 | ok | 53 | 0.062780 | 0.049386 | 0.077281 |
| focus | pr_auc | 0.913609 | 0.830199 | ok | 53 | 0.083410 | 0.060261 | 0.109037 |
| interest | macro_f1 | 0.717034 | 0.707865 | ok | 53 | 0.009169 | 0.002297 | 0.015926 |
| interest | roc_auc | 0.884929 | 0.879905 | ok | 52 | 0.005024 | -0.003605 | 0.012041 |
| interest | pr_auc | 0.873358 | 0.870150 | ok | 52 | 0.003207 | -0.001199 | 0.007518 |
| relaxation | macro_f1 | 0.789307 | 0.774318 | ok | 53 | 0.014989 | 0.008401 | 0.021963 |
| relaxation | roc_auc | 0.941706 | 0.930548 | ok | 53 | 0.011158 | 0.006899 | 0.015660 |
| relaxation | pr_auc | 0.938570 | 0.925707 | ok | 53 | 0.012863 | 0.007180 | 0.019018 |
| stress | macro_f1 | 0.729244 | 0.720728 | ok | 53 | 0.008516 | -0.001999 | 0.018910 |
| stress | roc_auc | 0.908923 | 0.901178 | ok | 52 | 0.007746 | 0.002156 | 0.012946 |
| stress | pr_auc | 0.900747 | 0.893421 | ok | 52 | 0.007326 | 0.002564 | 0.012259 |

All primary comparisons include 53 participants. ROC-AUC and PR-AUC use 52 valid paired participants for interest and stress, and 53 for the other PM; undefined metrics are excluded pairwise.

## Validation commands

Targeted: `python -m pytest tests/test_pm_temporal_predictability.py tests/test_pm_temporal_order_control.py -q` (31 passed). CLI: `scripts/analysis/run_pm_temporal_predictability.py --feature-cache-dir <cache> --order-control-dry-run`, then the same command with `--order-control`. The modes are mutually exclusive with the existing dry-run, smoke and full-run modes.
Post-run: `python scripts/analysis/report_pm_temporal_order_control.py` verifies all 25 saved cells without training. No experiment failures, restores or training warnings occurred. One intermediate read-only validation command had a shell-quoting SyntaxError; it was corrected and passed, with no impact on training or artifacts.

## Limitations and stop

Targets were selected from the matched experiment on the same cohort. Training folds overlap. This control uses one deterministic shuffle realization. Confidence intervals and inference are conditional on this design; non-significance is not evidence of equivalence. No physiological or causal inference is made.
No additional experiments were run: no other horizons/targets, no ordered retraining, no horizon-specific support, AutoML, neural models, wavelets or personalization. STOP after the 25-fit control.

## Artifacts

protocol.json; dry_run_summary.json; run_matrix.csv; execution_status.json; fold_metrics.csv; participant_metrics.csv; ordered_vs_shuffled_participant_deltas.csv; ordered_vs_shuffled_ba.csv; ordered_vs_shuffled_fold_deltas.csv; secondary_metrics.csv; inference_protocol.json; validation.json. Runtime runs/ remains outside Git. git_status_at_completion.txt records the working tree before this report commit; unrelated changes were preserved.
