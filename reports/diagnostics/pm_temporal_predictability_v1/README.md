# Temporal Predictability: completed predictive experiments

Current status, 2026-11-06: **full matched 175/175 complete**, executed 175, restored 0, failed 0; **H120 order control 25/25 complete**, executed 25, restored 0, failed 0. These are separate from the earlier five-fit smoke.

## Frozen scientific contract

All seven PM: attention, engagement, excitement, stress, relaxation, interest, focus. Five fixed subject-disjoint outer folds. Canonical 371 features; LightGBM n_estimators=200, n_jobs=4, random_state=42, no tuning. Q33/Q67 fitted once on canonical outer-train continuous complete cases before history filtering: LOW <= Q33, HIGH >= Q67, middle excluded from binary labels but retained in elapsed-time/support accounting. Thresholds are applied unchanged to test.

H10/H20/H30/H60/H120 concatenate 1/2/3/6/12 exact previous 10-second blocks. Oldest first, canonical -10 s last. No gaps, substitution, future windows or subject/record/fold crossing. Strict H120 common support preserves identical target samples, labels, participant IDs, fold IDs and threshold hashes across horizons within each PM/fold.

H120 support before binary label filtering: 29,437 windows, 53 participants, 84 records; 95.56% of H10 support. Primary metric is participant-macro balanced accuracy; Macro-F1, ROC-AUC and PR-AUC are secondary. Horizon-specific support remains an unexecuted sensitivity option, not part of these results.

## Full matched H10–H120

[Execution status](full_matched/execution_status.json), [run matrix](full_matched/run_matrix.csv), [overall metrics](full_matched/overall.csv), [inference protocol](full_matched/inference_protocol.json).

Equal-PM means of participant-macro metrics (descriptive; not a substitute for target-specific paired inference):

|   horizon_s |   balanced_accuracy |   macro_f1 |   roc_auc |   pr_auc |
|------------:|--------------------:|-----------:|----------:|---------:|
|   10.000000 |            0.750499 |   0.717453 |  0.882299 | 0.868681 |
|   20.000000 |            0.760806 |   0.727677 |  0.899573 | 0.885892 |
|   30.000000 |            0.766316 |   0.733107 |  0.905152 | 0.890468 |
|   60.000000 |            0.770759 |   0.738212 |  0.907277 | 0.892689 |
|  120.000000 |            0.781939 |   0.750086 |  0.911315 | 0.897289 |

H120 minus H10; participant-paired percentile 95% bootstrap CI, 10,000 samples, seed 42. BH family contains all **28** target × H>10 contrasts:

| pm         |   delta_ba |   ci95_low |   ci95_high |   bh_q_28 |
|:-----------|-----------:|-----------:|------------:|----------:|
| attention  |  -0.000085 |  -0.009802 |    0.009744 |  0.999600 |
| engagement |   0.005203 |  -0.004451 |    0.014226 |  0.407728 |
| excitement |   0.019671 |   0.012016 |    0.027583 |  0.000467 |
| focus      |   0.107084 |   0.086030 |    0.128018 |  0.000467 |
| interest   |   0.034538 |   0.018673 |    0.050348 |  0.000467 |
| relaxation |   0.019535 |   0.010264 |    0.029864 |  0.001600 |
| stress     |   0.034139 |   0.018233 |    0.053749 |  0.003150 |

H120 gains pass BH q<0.05 for **5/7 PM**: excitement, focus, interest, relaxation, stress. Attention and engagement have no confirmed H120 gain. Predictive value of additional EEG history is target-dependent.

## H120 history-order control

[Full report](order_control_h120/README.md), [status](order_control_h120/execution_status.json), [inference protocol](order_control_h120/inference_protocol.json), [validation](order_control_h120/validation.json).

Only the five H120-positive PM, H120 and folds 1–5: **25 fits**. Ordered H120 was reused without retraining. The existing per-sample deterministic shuffle preserves all 11 older blocks (-120 through -20 s) and never moves the canonical -10 s block. Samples, labels, subjects, folds, thresholds and counts match; hashes and saved provenance pass.

Contrast = ordered H120 minus shuffled H120. Paired participant bootstrap: 10,000, seed 42, percentile 95% CI, two-sided centered-null p, BH across exactly five primary BA contrasts:

| pm         |   ordered_ba |   shuffled_ba |   delta_ba |   ci95_low |   ci95_high |   bh_q_5 |
|:-----------|-------------:|--------------:|-----------:|-----------:|------------:|---------:|
| excitement |     0.860613 |      0.853213 |   0.007400 |   0.002297 |    0.012736 | 0.008499 |
| focus      |     0.812677 |      0.733485 |   0.079192 |   0.065372 |    0.093650 | 0.000250 |
| interest   |     0.762386 |      0.753800 |   0.008586 |   0.002551 |    0.014468 | 0.008499 |
| relaxation |     0.802520 |      0.788174 |   0.014346 |   0.008500 |    0.020517 | 0.000250 |
| stress     |     0.773203 |      0.763517 |   0.009687 |  -0.000739 |    0.020272 | 0.070793 |

Order effect passes BH for **4/5 gated PM**: excitement, focus, interest, relaxation. Stress is **inconclusive**: positive point estimate, CI crosses zero, q≈0.0708. Focus: 52/53 participants and 5/5 folds favor ordered history.

This is a two-stage analysis: the 28-contrast history family followed by the prespecified conditional five-target order control. It is not an independent unconditional seven-PM test. One fixed shuffle realization and overlapping training folds limit inference. No physiological memory, attractor or causal mechanism is established.

## Scientific decision and next step

Additional history carries target-dependent predictive information; correct older-block order adds value for four gated PM under this representation. One bounded, preregistered H120 sequence-model comparison against ordered-H120 LightGBM is justified and **planned, not executed**. No architecture sweep is justified; no AutoML, Mamba, PatchTST or TimesNet sweep is authorized by these results. Wavelets and synchronization remain separate later representation hypotheses.

## Bounded smoke

One run: Attention, fold 1, all five horizons, n_estimators=10. **5 executed, 0 restored, 5 complete**. Every cell uses 14,756 training and 4,098 test examples.

| Horizon | Participant-macro BA | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| H10 | 0.6557 | 0.5998 | 0.7963 | 0.8207 |
| H20 | 0.6680 | 0.6124 | 0.8020 | 0.8260 |
| H30 | 0.6762 | 0.6239 | 0.7996 | 0.8177 |
| H60 | 0.6728 | 0.6178 | 0.7970 | 0.8176 |
| H120 | 0.6598 | 0.6025 | 0.7921 | 0.8165 |

These are smoke metrics, **not evidence for a preferred horizon or model**. No bootstrap inference is made from this smoke. Participant metrics and paired H10 deltas are saved. Runtime prediction/history-provenance artifacts have SHA256 checksums; resume rejects incomplete or corrupt results instead of silently retraining.


## Reproducibility

Existing CLI modes: --dry-run / --smoke / --full-run / --order-control-dry-run / --order-control (mutually exclusive). Reproduction requires the canonical feature cache and the frozen reference artifacts; no rerun was performed during documentation reconciliation. Full matched implementation c4b688e/results 8584d96; order control implementation 6d332f7/results fbd100f. Order-control targeted tests: 31 passed; full runtime and failure/restoration counts are recorded in each status file.

[Diagnostic structure report](../pm_temporal_predictability_audit_v1/README.md) contains ACF/MI/PE/SSA/FNN evidence, which is distinct from these PM prediction comparisons.
