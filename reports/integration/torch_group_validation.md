# Group-aware inner validation for Torch models

## Scope

This change replaces window-level inner validation with a shared,
group-aware option for Torch models. The outer evaluation protocol is not
changed. The implementation was validated on the canonical feature dataset
and on a single outer fold of the seven-target Performance Metrics regression
task.

Branch and audited baseline:

- branch: `integration/benchmark-unification`;
- HEAD at implementation start: `733a85c`;
- baseline suite before code changes: `394 passed, 11 warnings`.

## Previous risk

The shared Torch adapter created its inner train/validation partition with
`train_test_split`. Classification used stratification, but neither
classification nor regression used subject or recording identity. Windows
from one subject could therefore occur in both inner partitions and make
validation loss, early stopping, and best-epoch selection optimistic.

Record-disjoint validation already existed for sequence and raw-window
pipelines, but the runner only configured it for those observation units.
Feature-based `torch_mlp` training did not receive validation groups.

## Configuration contract

The existing top-level validation section is retained:

```yaml
validation:
  strategy: group_holdout
  group_column: subject_id
  fraction: 0.15
  random_state: 42
```

Supported strategies:

- `group_holdout`: recommended group-aware mode;
- `random_holdout`: explicit legacy window-level mode;
- `group_record`: retained for existing sequence and raw-EEG configs.

`fraction` and the legacy name `validation_size` are both accepted. Invalid
strategies, missing group columns, invalid fractions, and invalid seeds are
rejected during CLI config validation.

If validation is omitted, the runner prefers `group_holdout` by
`subject_id`. For an outer protocol that is not subject-disjoint, or an outer
training partition with fewer than two groups, it uses `random_holdout` only
with an explicit warning and a `fallback_reason` in metadata. An explicitly
configured `group_holdout` never falls back silently.

## Split algorithm and class coverage

The shared `TorchClassificationAdapter` uses `GroupShuffleSplit` only on the
outer training rows. It evaluates a bounded deterministic set of candidates
(`min(128, max(32, 4 * n_groups))`) and prefers:

1. complete class coverage in inner train and validation for classification;
2. validation size closest to the requested fraction;
3. the smallest class-distribution difference, or the smallest standardized
   target-mean difference for regression.

All windows of a selected subject stay in exactly one inner partition.
Both partitions contain at least one group. If classification cannot produce
a class-complete `group_holdout`, training stops before the first epoch with a
clear error and the per-group class distribution. The existing
`group_record` compatibility mode retains its prior warning behavior.

## Scaling and early stopping

The adapter resolves inner indices first and fits the feature mean and scale
only on `X[inner_train_indices]`. It then transforms inner train and inner
validation separately. The outer test data are transformed later with those
same inner-train statistics.

No target scaler is currently used by the Torch regression path, so there is
no target-scaler fitting scope to change. Early stopping and best-state
selection use only the inner validation loss.

## Audit artifacts

`validation_split.json` now includes the strategy, group column, requested
fraction, seed, sample/group counts, class or target summaries, group lists,
overlap lists and counts, and fallback reason.

Each Torch fold also writes `inner_validation_audit.csv`. GroupKFold combines
the fold rows into a protocol-level `inner_validation_audit.csv`. The row
contains the dataset, task, model, outer fold, strategy, group column,
inner sample/group counts, all required overlap counts, seed, best epoch,
best validation loss, and fallback reason.

## Tests

The new `tests/test_torch_group_validation.py` covers:

- subject-disjoint and deterministic grouped partitions;
- full assignment of each subject to one inner partition;
- a changed selection under another seed;
- one-group and impossible class-complete cases;
- one-dimensional classification targets;
- scalar and seven-output regression shapes;
- feature scaling on inner train only;
- metadata and audit contents;
- explicit legacy `random_holdout`;
- validation of the PM smoke config.

Targeted suite:

```text
47 passed in 5.21s
```

Final full suite:

```text
402 passed, 11 warnings in 93.28s
```

## Real PM regression smoke

Command:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --config experiments\pm_regression\pm_regression_group_validation_smoke.yaml `
  --models torch_mlp `
  --fold-limit 1 `
  --verbose
```

Resolved dataset/model:

- complete-case windows: 43,174;
- subjects: 53;
- input shape: `(448,)`;
- output shape: `(n, 7)`;
- trainable parameters: 66,183;
- device: CUDA, NVIDIA GeForce RTX 5060 Ti.

Outer fold 1:

- outer train: 34,581 windows, 43 subjects;
- outer test: 8,593 windows, 10 subjects;
- outer test subjects: `01c2a0d8`, `7092f07b`, `7150e10a`, `71a251fa`,
  `71c09041`, `81f1f0fe`, `b0700166`, `c060c06a`, `d18000a3`,
  `f121f1e0`;
- outer train/test subject overlap: 0.

Inner split:

- inner train: 29,395 windows, 36 subjects;
- inner validation: 5,186 windows, 7 subjects;
- validation subjects: `2162c09e`, `30c140ca`, `3110e0c7`, `50c02189`,
  `517001af`, `8191f1d9`, `c112918e`;
- inner group overlap: 0;
- inner train/outer test overlap: 0;
- inner validation/outer test overlap: 0;
- fallback: none.

Training and fold metrics:

- epochs trained: 3;
- best epoch: 2;
- best validation loss: 356.038853;
- training and evaluation time reported by the runner: 2.4291 seconds;
- macro MAE: 0.110233;
- macro RMSE: 0.152369;
- macro R2: -0.159733;
- macro Pearson: 0.268806;
- macro Spearman: 0.286777.

The prediction artifact contains 8,593 rows, seven `y_true_*` and seven
`y_pred_*` columns, and no duplicate `sample_id`. This is a short technical
smoke, not a final scientific estimate.

Primary artifacts:

- `benchmark_results/pm_regression_group_validation_smoke/benchmark_results_20260724_123659.json`;
- `benchmark_results/pm_regression_group_validation_smoke/summary_20260724_123659.csv`;
- `benchmark_results/pm_regression_group_validation_smoke/20260724_123659/emotiv_pm_regression/performance_metrics_regression/torch_mlp/group_kfold_subject/inner_validation_audit.csv`;
- fold directory:
  `benchmark_results/pm_regression_group_validation_smoke/20260724_123659/emotiv_pm_regression/performance_metrics_regression/torch_mlp/group_kfold_subject/fold_01/`.

The fold directory contains `model.pt`, `training_log.csv`,
`predictions.parquet`, `metrics.json`, `validation_split.json`,
`normalization_stats.json`, and the fold-level
`inner_validation_audit.csv`.

## Validation-loss aggregation audit

A follow-up audit investigated the apparently implausible PM regression
validation loss `356.038853`. The proposed diagnosis—that the epoch
aggregator had discarded the `n_samples * n_outputs` denominator—was not
confirmed.

The shared objective and adapter already use the correct contract:

```text
batch mean used by backward
    = batch numerator / batch denominator

epoch component mean
    = sum(batch numerators) / sum(batch denominators)
```

For multi-output MSE, each batch numerator is the sum of squared errors and
its denominator is `raw_outputs.numel()`. `loss.backward()` receives
`LossParts.mean`, not the unnormalised numerator. The same numerator and
denominator accumulation is used for train and validation. The fine-tuning
path is classification-only and likewise uses `LossParts.mean`.

Recomputing the loss from the saved best checkpoint on the exact inner
validation partition produced:

```text
validation samples:       5,186
outputs:                       7
elements:                 36,302
sum of squared errors: 12,924,922.998460
mean squared error:         356.038868
stored best loss:           356.038853
```

Therefore, dividing `356.038853` by `5,186 * 7` would divide by the
denominator a second time. The logged value is already the mean MSE.

The actual cause is extreme held-subject feature values and resulting
predictions. Subject `8191f1d9` accounts for nearly all of the large loss:

```text
subject validation windows:       714
subject MSE:                  2,573.466883
overall prediction range:       -4.214132 .. 1,040.749878
maximum standardised feature: 41,472.621094
```

Examples include `POW.T8.BetaL__min=12,781.04` and
`POW.T8.Alpha__min=19,893.57`, while their inner-train means and standard
deviations are approximately `0.37 +/- 0.31` and `0.56 +/- 0.52`.
Group-aware validation correctly exposes this held-subject distribution
shift; changing loss normalisation would conceal it rather than fix it.

The existing aggregation was moved from a nested training-loop helper to the
module-level `_aggregate_loss_component_values` function without changing its
mathematics. New tests verify:

- exact NumPy-equivalent scalar and multi-output MSE;
- denominator `n` for scalar regression and `n * 7` for seven outputs;
- invariance to batch sizes 1, 2, and full/partial final batches;
- mean Smooth L1 aggregation;
- unchanged categorical, CORAL, and CORN aggregation;
- regression gradients equal the analytical mean-MSE gradient;
- `best_validation_loss_` equals the minimum logged validation loss.

Follow-up targeted suite:

```text
27 passed in 4.02s
```

Follow-up full suite:

```text
414 passed, 11 warnings in 92.46s
```

The requested one-fold diagnostic rerun was saved under
`benchmark_results/pm_regression_group_validation_smoke_loss_fixed`, although
no numerical loss correction was necessary. Its epoch logs were exactly
equal to the original:

| Epoch | Train loss | Validation loss | Best |
| ---: | ---: | ---: | :---: |
| 1 | 0.075435 | 399.943634 | yes |
| 2 | 0.032835 | 356.038853 | yes |
| 3 | 0.026461 | 369.431113 | no |

The rerun retained best epoch 2, identical split sizes and zero overlap
counts. Test predictions were identical (`max_abs_difference=0`) and the
macro metrics remained MAE `0.110233`, RMSE `0.152369`, R2 `-0.159733`,
Pearson `0.268806`, and Spearman `0.286777`.

## Limitations and next step

Only one outer fold and three epochs were run. No five-fold experiment,
multi-seed experiment, hyperparameter search, LSTM, or Transformer training
was performed. Existing record-level configs remain supported, while
canonical future Torch configs should state `group_holdout` and
`subject_id` explicitly.

Before extending multi-output regression to LSTM and Transformer, the
extreme POW feature values should be audited as a separate data-quality and
robust-scaling question. This task deliberately did not change feature
preprocessing, clipping, or scaling.
