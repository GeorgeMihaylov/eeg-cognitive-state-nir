# Confirmatory multi-fold DANN protocol

- Branch/HEAD: `integration/benchmark-unification` / `f8c6e58`.
- Hypothesis: DANN improves Old_EEG-to-gpn_data label_q5 transfer over a source-update-matched EEGNet.
- Diagnostic provenance: fold 1, seed 42, `diagnostic/proceed`; mean participant Δmacro F1 +0.013364, Δbalanced accuracy +0.019079, 6/8 wins, bootstrap CI crossing zero.
- The diagnostic is limited to one fold and seed; its target-test result did not retune any confirmatory hyperparameter.
- Raw universe: 30,958 windows, 54 participants, 86 logical records; `308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf`.
- Direction/policy: `Old_EEG -> gpn_data`, `strict_cross_domain_subject_disjoint`.
- Model seeds: `42, 123, 2026`; source-validation split seed is always 42 and is shared across model seeds.
- Production EEGNet: `[B,1,14,2560]`, 8,501 parameters, latent 1,280; fixed 172,354-parameter domain head.
- Execution is disabled: no optimizer, backward, training, CUDA tensor, target-test EEG read, inference, or metric calculation occurred.

## Five-fold inventory and eligibility

| fold | outer split hash | source train s/w | source val s/w | target train s/w | target test s/w | matched steps | status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `b8591f6a0ff5…` | 8/4433 | 2/776 | 33/18555 | 8/4973 | 580 | eligible |
| 2 | `c3062fa8f721…` | 8/4749 | 2/776 | 33/19241 | 9/4550 | 602 | eligible |
| 3 | `adc7b66241ef…` | 9/4995 | 3/1456 | 31/18207 | 10/5321 | 569 | eligible |
| 4 | `e6117577010d…` | 8/4557 | 2/985 | 34/19377 | 7/4151 | 606 | eligible |
| 5 | `a11dbdf7654f…` | 8/4790 | 2/1151 | 33/18732 | 8/4796 | 586 | eligible |

All five folds contain every class in source train and source validation, meet participant thresholds, and have zero subject/sample/logical-record overlap.

## Shared participant and protected target partitions

`a02151ac` is in outer train for folds 1/3/4/5: it remains source-side and is excluded from unlabeled target train. In fold 2 it is outer-test and therefore absent from both training domains. The rule is deterministic and target-metric independent.

Target-train manifests expose EEG provenance/domain fields only in the future training contract; `label_q5`, `target`, `task_label`, and `y` are forbidden. Target-test references contain IDs/counts only. Raw target-test tensors read: 0.

## Inherited training contract

AdamW, learning rate 0.001, weight decay 0.0001, source/target batch size 32/32, maximum 12 epochs, patience 3, gradient clipping 5.0, logistic GRL alpha, constant domain lambda 1.0, and matched joint early stopping are inherited verbatim from executable diagnostic preregistration `f5e7cd…a817`.

Each fold/seed/mode pair must share the preregistered source batch sequence hashes. Checkpoints are selected independently by source-validation macro F1, then balanced accuracy. Target data and domain accuracy never select checkpoints.

## Future aggregation and decision

The primary unit is the unique participant. Deltas are first averaged within participant across seeds, then across unique participants; fold-, seed-, and overall variability are reported separately. Windows are not independent observations.

`confirmed` requires mean Δmacro F1 ≥0.01, positive median, nonnegative mean Δbalanced accuracy, ≥60% participant wins, and ≥4/5 folds with nonnegative mean Δmacro F1. Positive but incomplete evidence is `partially_confirmed`; nonpositive overall effect, only one positive fold, or <40% wins is `not_confirmed`; methodological failure is `blocked`.

Protocol hash: `a261d6081b4924af82752021fa24bbd50a75ed83ac3672db1e691709ad2cad71`.
Disabled preregistration hash: `f4862dbf09d6eccd04438eebd5bbd99899dc4f1530ba3554c87a45a13dba59c4`.
Readiness: **confirmatory_protocol_ready** (5/5 folds eligible).

Execution requires a separately authorized stage that first passes a clean full pytest and preserves every fold manifest, source-validation split, model seed, hyperparameter, batch hash requirement, and target-test lock. No confirmatory run was started here.
