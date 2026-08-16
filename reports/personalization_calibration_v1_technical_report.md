# Leakage-safe personalization/calibration v1: technical audit and protocol

## Scope

This change prepares an executable, but not yet scientifically executed, unified participant-personalization
experiment for the seven canonical Performance Metrics (PM).  The protocol uses
the existing deduplicated raw-window universe, fixed participant folds, target
registry, fold-local target transforms, model factory and shared Torch adapter.
No project EEG model was trained and no project checkpoint was created during this task.

## Existing infrastructure audit

| Area | Existing behavior at `48676159` | Reuse / limitation |
|---|---|---|
| Classification personalization | `bench/experiments/user_calibration.py` provides zero-shot, subject normalization, head-only and full-model modes, checkpoint cloning, per-condition artifacts and resume hashes. | Reused split, fixed-evaluation and adapter contracts. Its configured scientific target is legacy `label_q5`, so it is not the new primary target. |
| PM regression personalization | `bench/experiments/pm_regression_personalization.py` supports the canonical seven-output PM order, zero-shot, bias, affine, head-only and full-model adaptation. | Scientifically useful prior implementation, but fixed to multi-output complete cases, 20%, Torch MLP and seed 42. It does not provide seven target-specific regression/Q3 cells. |
| Calibration split | Existing code uses `chronological_prefix`, a calibration-only chronological validation holdout, one late reference evaluation suffix across budgets, and explicit sample-overlap audits. | Reused. The new plan strengthens ordering by using `absolute_t_start` across all recordings of a participant. |
| Outer folds | Canonical raw manifest stores `outer_fold`; each subject belongs to exactly one of five folds. | Reused without recomputation. Real plan audit: outer train/test subject overlap is zero. |
| Q3 | Seven `pm_*_q3_fold_local` targets and `FoldLocalQuantileTargetTransform` already exist. Cross-validation fits boundaries on outer train and stores transform hashes. | Reused without new discretization code. One frozen transform per PM/fold is shared by calibration and evaluation. |
| Model adaptation | `TorchClassificationAdapter.clone()` preserves the loaded checkpoint. `fine_tune()` accepts explicit calibration train/validation arrays and supports head-only/full-model. | Reused; no new optimizer or training loop. |
| Model heads | ShallowConvNet and EEGNet implement the shared encoder/head API; Torch MLP exposes an explicit output-head prefix. | Head-only is structurally supported for all three models where the task factory supports the model. |
| Inner validation | Canonical base runs can use `record_group_id`-disjoint inner validation. Personal adaptation validation is a chronological suffix of calibration only and never uses final evaluation. | Both contracts are retained and must be checked during execution. |
| Checkpoints/resume | Existing experiments hash base checkpoints, clone state, audit frozen/trainable parameters and reject incompatible resume state. | The execution bridge now validates the complete base identity and independently resumes base folds and participant adaptations. |

No existing personalization path was found that is simultaneously target-specific
for all seven PM, supports both scalar regression and fold-local Q3, covers raw
CNNs and produces one deterministic full run matrix.  That missing orchestration,
not a second model-training implementation, is the scope of v1.

## Execution bridge

`bench/experiments/personalization_calibration_execution.py` connects the
materialized protocol to the existing production layers. One base unit is
`outer fold × PM × task type × model`; it is delegated to `BenchmarkRunner`,
which retains fixed subject folds, record-group inner validation, fold-local Q3,
normalization, model factory construction and standard fold artifacts. The
resulting adapter is loaded once and reused read-only for every outer-test
participant. `zero_shot_shared_eval` performs inference on the common late
suffix. Head-only and full-model conditions start from independent
`TorchClassificationAdapter.clone()` instances and call the existing
`fine_tune()` loop with explicit chronological calibration-train and validation
arrays.

The bridge fits no normalization. Raw channel statistics or feature
`standard_clip` state loaded from the base checkpoint remain frozen during
adaptation, and their hash is asserted before and after each condition. Base
identity is not inferred from a filename: target, fold, preprocessing hashes,
sample-universe hash, input shape, normalization hash, task/output shape, seed,
model-config hash, Q3 hash, benchmark config hash and checkpoint SHA-256 are all
recorded. Participant resume additionally binds the protocol/plan/base hashes,
subject, mode, budget and calibration/evaluation sample hashes.

For feature MLP runs, `EmotivDataset.cohort_manifest_path` selects the exact
PM-specific raw-deduplicated `sample_id` universe and attaches the canonical
`outer_fold`; this avoids a parallel dataset or split implementation. No change
to `BenchmarkRunner`, the model factory, or either training loop was required.

## Scientific protocol

For each fixed outer fold, the global model is trained only on outer-train
participants.  Base-model early stopping and normalization must use
`record_group_id`-disjoint inner validation and outer-train data only.  Every
outer-test participant is treated as a new user.

The participant's accepted windows are ordered by `absolute_t_start`, with
`source`, `record_id`, relative time and `sample_id` as deterministic tie-breakers.
The maximum 20% prefix defines one fixed late evaluation suffix. Smaller budgets
use shorter prefixes and reserve the intermediate windows, so every mode and
budget is compared on identical evaluation sample IDs. Personalized parameters
may persist forward across recordings of the same participant; no recurrent or
BatchNorm state is propagated window-to-window at evaluation. This explicit
participant-level rule prevents a prefix from a later recording being used to
predict an earlier recording.

Budgets reuse the existing classification calibration set: 0%, 1%, 5%, 10% and
20%. At 10-second windows these correspond nominally to 0, 10, 50, 100 and 200
seconds per 100 available windows; actual durations are stored per participant.
Zero budget exists only for `zero_shot`; head-only and full-model use positive
budgets. The calibration-only 80/20 chronological holdout is the adaptation
validation source. If minimum calibration/evaluation sizes are not met, the cell
is `insufficient_data`; windows are never duplicated and evaluation is never
borrowed.

For Q3, `FoldLocalQuantileTargetTransform` fits exactly once from the continuous
target values of the outer-train fold. Its frozen boundaries and hash are then
applied to all calibration/evaluation windows for that PM/fold. New-user labels,
global quantiles and evaluation targets never contribute to the fit.

## Compatibility and run matrix

Factory probing at plan time produced the following contract:

- ShallowConvNet: Q3 and scalar regression; zero-shot, head-only and full-model.
- EEGNet: Q3 only; scalar regression is explicitly `unsupported` because the
  current factory exposes EEGNet as classification-only.
- Torch MLP: Q3 and scalar regression through the shared adapter. Its input is
  the 448-feature EEG+POW view matched to the same canonical sample IDs; it must
  not be presented as a raw-input CNN.

The full matrix contains 1,890 fold-level conditions:

`7 PM × 2 task types × 3 models × 5 folds × (1 zero-shot + 4 head-only + 4 full-model)`.

Of these, 1,575 are supported and 315 EEGNet-regression conditions are explicitly
unsupported. The real participant plan contains 1,885 PM/participant/budget rows
and 15,455 supported participant executions after minimum-cohort checks. There
are 54 unique participants; Attention has 53 target-available participants and
the other six PMs have 54.

The real plan uses 29,570 Attention, 32,535 Engagement, 34,354 Excitement, 30,958
Stress, 30,964 Relaxation, 31,002 Interest and 30,958 Focus accepted deduplicated
windows. Median actual calibration sizes are 0, 5, 28, 57 and 114 windows for the
five budgets. At 1%, 135 of 377 PM/participant cells are below the five-window
minimum; these are retained as `insufficient_data` in the participant manifest,
not silently removed.

## Leakage and reproducibility gates

The planner verifies:

1. fixed-fold outer train/test participant overlap is zero;
2. each participant maps to exactly one outer fold;
3. calibration/evaluation sample overlap is zero;
4. all calibration timestamps precede all evaluation timestamps;
5. one identical evaluation sample hash is used across budgets;
6. no partition contains more than one participant;
7. Q3 fit scope is outer-train only and each transform hash is valid;
8. repeated planning produces deterministic sample and condition hashes;
9. insufficient cells do not duplicate or borrow windows;
10. resume requires both the immutable protocol hash and the filter-specific plan hash.

Real `plan-only` result: outer subject overlap 0, calibration/evaluation overlap
0, all calibration-before-evaluation checks true, and fixed evaluation hashes
true. Protocol hash: `9e985d1df33a8d46d20e25f71753a187e3864e2379996dbc9ad8bd18c7a20c0e`.

## Dry execution result

The full metadata-only dry execution completed without loading raw EEG tensors
or fitting a model. It resolved 175 unique supported base units: 70
ShallowConvNet (35 Q3 + 35 regression), 70 MLP (35 + 35), and 35 EEGNet Q3.
No existing checkpoint passed the exact new identity contract, so 0 can be
reused and all 175 would require base training. Historical `label_q5` and
seven-output PM checkpoints are deliberately rejected as incompatible.

The supported plan contains 1,865 pure shared-evaluation zero-shot inferences,
6,795 head-only adaptations and 6,795 full-model adaptations. Thus the future
full execution has 13,590 adaptation trainings and 13,765 total training jobs
after adding the 175 shared base trainings. This is 1,690 fewer trainings than
the 15,455 participant executions; the difference is explained by replacing
1,865 zero-shot participant trainings with inference while adding 175 reusable
base trainings. If every adapted model is retained, the upper-bound checkpoint
count is 13,765. There are 1,510 insufficient-data participant-condition
occurrences across the supported expanded matrix; the 1% budget is preserved
and reported rather than silently removed.

| Scope | Base | Zero-shot inference | Head adaptations | Full adaptations | Training jobs |
|---|---:|---:|---:|---:|---:|
| Full | 175 | 1,865 | 6,795 | 6,795 | 13,765 |
| ShallowConvNet only | 70 | 746 | 2,718 | 2,718 | 5,506 |
| Head-only + shared zero-shot | 175 | 1,865 | 6,795 | 0 | 6,970 |
| Fold 1 | 35 | 385 | 1,380 | 1,380 | 2,795 |
| Fold 2 | 35 | 360 | 1,340 | 1,340 | 2,715 |
| Fold 3 | 35 | 385 | 1,395 | 1,395 | 2,825 |
| Fold 4 | 35 | 350 | 1,285 | 1,285 | 2,605 |
| Fold 5 | 35 | 385 | 1,395 | 1,395 | 2,825 |
| Fold-1 Focus-Q3 Shallow head-only smoke | 1 | 11 | 39 | 0 | 40 |

No defensible runtime estimate is reported because no exact compatible
base/adaptation runtime artifact exists. A benchmark was not launched merely to
measure time.

## Commands

Plan-only (executed):

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --personalization-calibration experiments\calibration\personalization_calibration_v1.json `
  --plan-only `
  --data-root F:\EEG `
  --output-dir benchmark_results\personalization_calibration_v1 `
  --verbose
```

Dry execution (executed; no training):

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --personalization-calibration experiments\calibration\personalization_calibration_v1.json `
  --dry-execution --data-root F:\EEG --verbose
```

Minimal future real smoke (specified, not executed):

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --personalization-calibration experiments\calibration\personalization_calibration_v1.json `
  --run --outer-fold 1 --pm focus --task-type classification `
  --models torch_shallow_convnet --calibration-mode head_only `
  --calibration-budget-fraction 0.05 --subject-limit 1 `
  --max-calibration-epochs 1 --device cpu `
  --data-root F:\EEG --resume --verbose
```

Intended final command (specified, not executed):

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --personalization-calibration experiments\calibration\personalization_calibration_v1.json `
  --run --data-root F:\EEG --resume --verbose
```

The CLI now accepts exactly one of `--plan-only`, `--dry-execution`, or `--run`.
The real smoke and final commands above were intentionally not executed.

## Scientific limitations

- Q3 is a fold-local ordinal proxy, not a direct clinical or cognitive-state
  ground truth.
- PM values are device-derived signals and target availability differs slightly
  by PM; participant-macro results must therefore retain target-specific cohorts.
- The 1% budget is frequently too small for five calibration windows and should
  be reported as a feasibility point, not optimized away.
- Cross-record persistence models a returning user whose personalized parameters
  are retained. A session-reset policy would be a different preregistered study.
- MLP and raw CNN results share target/sample identities where available but use
  different input representations; this must remain explicit in comparisons.
- No claim about personalization benefit is possible from this dry-execution stage.

## Short description for a colleague

We prepared a unified leakage-safe personalization protocol for all seven EEG
Performance Metrics in both continuous regression and fold-local Q3 form. The
global model is trained without each held-out participant using the existing five
fixed subject folds. For a new participant, calibration is a strict chronological
prefix and evaluation is a later fixed suffix shared by all budgets and modes.
Q3 thresholds are fitted only on outer-train targets and are frozen before any
new-user data are processed. The comparison includes zero-shot, head-only and
full-model adaptation with 0%, 1%, 5%, 10% and 20% budgets. Results will be
computed per participant and then macro-averaged, so users with many windows do
not dominate. ShallowConvNet and MLP support both tasks, while current EEGNet
regression is explicitly unsupported. The execution bridge and full dry cost
plan are now implemented; no project EEG model training was performed.
