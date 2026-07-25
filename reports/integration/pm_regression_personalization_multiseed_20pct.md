# Three-seed PM regression personalization at 20% calibration

## Purpose and provenance

This experiment tests whether personalization of the seven-output Performance
Metrics regression model is robust to model initialization and training
stochasticity. The data split is fixed with `split_seed=42`; only the model RNG
changes across `model_seed in {7, 42, 2026}`. The evaluated methods are
`zero_shot`, `head_only`, and `full_model`; the calibration budget is fixed at
20%.

The run was performed on branch `integration/benchmark-unification` at
`b1f47f4 feat: add PM regression personalization benchmark`. The working tree
was initially clean. The pre-existing seed-42 run was found at
`benchmark_results/pm_regression_personalization_20pct/`.

Strict reuse of that run was rejected. Its dataset hash, prediction schema,
target order, and completed-condition set matched, but its configuration and
implementation hashes differed and it did not contain the complete split hashes
needed to prove strict compatibility. Therefore all 15 global models were
trained again. The newly trained seed-42 result nevertheless reproduced the
old result exactly: all common numeric metrics for 159 selected subject-method
conditions and all 725,739 matching prediction rows had maximum absolute
difference 0.

## Orchestration

`PMRegressionPersonalizationMultiseedExperiment` is a thin orchestration layer:
it resolves one seed-specific configuration, invokes the existing
`PMRegressionPersonalizationExperiment`, validates provenance and artifacts,
and aggregates completed results. It does not duplicate global training,
fine-tuning, preprocessing, split, checkpoint, or metric logic.

- `split_seed=42` controls outer and inner splits, chronological calibration
  and evaluation samples, preprocessing fit, and bootstrap resampling.
- `model_seed` controls Python, NumPy, Torch/CUDA, DataLoader, dropout,
  initialization, and optimization RNG.
- Resume is manifest/config/implementation aware. A second invocation skipped
  all three completed seeds and returned the existing run in 3.4 seconds.

## Data, targets, and protocol

- Dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`
- Dataset SHA-256:
  `26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8`
- Complete-case feature shape: `(43174, 448)`
- Target shape: `(43174, 7)`
- Subjects: 53
- Model: `torch_mlp`, input 448, output 7
- Outer evaluation: five-fold GroupKFold by `subject_id`
- Inner validation: group holdout by `subject_id`, fraction 0.15
- Personalization: first chronological 20% per target subject is the
  calibration pool; 80/20 of that pool is adaptation train/validation; the
  remaining 80% is final evaluation
- Chronological order: `source`, `record_id`, `t_start`, `sample_id`
- Preprocessing: `standard_clip`, quantiles 0.005 and 0.995, fitted only on
  global inner-train subjects

Canonical target order:

1. `target_attention`
2. `target_engagement`
3. `target_excitement`
4. `target_stress`
5. `target_relaxation`
6. `target_interest`
7. `target_focus`

No PM target, `label_q5`, identifier, temporal, or service column is included
among the 448 features.

## Environment and execution

- Python 3.11.15
- Torch 2.11.0+cu128
- CUDA 12.8
- GPU: NVIDIA GeForce RTX 5060 Ti
- Peak allocated GPU memory: 20,894,720 bytes (19.93 MiB)
- Global trainings: 15 (3 seeds x 5 folds)
- Subjects: 53
- Conditions: 477 completed, 0 incomplete, 0 failed
- Prediction rows: 2,177,217
- Wall time: 247.65 s
- Sum of global-training times: 57.74 s
- Sum of recorded fine-tuning times: 20.81 s

Global training used eight epochs in all seed-7 and seed-42 folds. Seed 2026
used 4--8 epochs (mean 7.2) because of early stopping. Mean best validation
loss was 0.019334, 0.019469, and 0.020506 for seeds 7, 42, and 2026,
respectively.

Full command:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --calibration-experiment experiments\calibration\pm_regression_personalization_multiseed_20pct.yaml `
  --resume `
  --verbose
```

## CUDA smoke test

The successful smoke run used seeds 7 and 2026, fold 1, one Old_EEG subject and
one gpn_data subject, two global epochs, and two fine-tuning epochs. It completed
12/12 conditions with two global trainings, 27,342 prediction rows, no
failures, and a wall time of 14.83 s. Its resume invocation skipped both seeds.

Smoke artifacts:
`benchmark_results/pm_regression_personalization_multiseed_cuda_smoke/20260725_222110/`.

## Seed-level results

Positive gain means lower error for MAE/RMSE/absolute bias and higher value for
R2/Spearman.

| seed | method | macro MAE | MAE gain | macro RMSE | RMSE gain | macro R2 | R2 gain | macro Spearman | Spearman gain | macro abs bias |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | zero_shot | 0.104844 | 0 | 0.132275 | 0 | -0.172508 | 0 | 0.378014 | 0 | 0.041875 |
| 7 | head_only | 0.102491 | 0.002353 | 0.130749 | 0.001526 | -0.156389 | 0.016118 | 0.381309 | 0.003295 | 0.039687 |
| 7 | full_model | 0.101896 | 0.002948 | 0.129645 | 0.002630 | -0.143181 | 0.029326 | 0.387492 | 0.009477 | 0.038367 |
| 42 | zero_shot | 0.103848 | 0 | 0.131560 | 0 | -0.148883 | 0 | 0.380569 | 0 | 0.042615 |
| 42 | head_only | 0.102325 | 0.001523 | 0.130435 | 0.001125 | -0.140564 | 0.008319 | 0.383056 | 0.002487 | 0.040315 |
| 42 | full_model | 0.101279 | 0.002569 | 0.129105 | 0.002455 | -0.120986 | 0.027898 | 0.394836 | 0.014267 | 0.039722 |
| 2026 | zero_shot | 0.106573 | 0 | 0.134722 | 0 | -0.222467 | 0 | 0.354283 | 0 | 0.043681 |
| 2026 | head_only | 0.104795 | 0.001778 | 0.133663 | 0.001059 | -0.242704 | -0.020236 | 0.357074 | 0.002792 | 0.042046 |
| 2026 | full_model | 0.104035 | 0.002538 | 0.132574 | 0.002148 | -0.204344 | 0.018123 | 0.366495 | 0.012212 | 0.040987 |

The seed-level mean gain +/- sample standard deviation is:

| method | macro MAE | macro RMSE | macro R2 | macro Spearman | macro abs bias |
|:---|---:|---:|---:|---:|---:|
| head_only | 0.001885 +/- 0.000425 | 0.001237 +/- 0.000253 | 0.001401 +/- 0.019139 | 0.002858 +/- 0.000408 | 0.002041 +/- 0.000356 |
| full_model | 0.002685 +/- 0.000228 | 0.002411 +/- 0.000244 | 0.025116 +/- 0.006098 | 0.011985 +/- 0.002403 | 0.003031 +/- 0.000424 |

Thus MAE, RMSE, and Spearman gains are positive for both methods in all three
seeds. Full-model R2 gain is positive in every seed; head-only R2 gain is
negative for seed 2026.

## Subject-averaged multiseed results

Seeds were first averaged within each subject. The 53 resulting independent
subject summaries were then aggregated; the confidence intervals use 1,000
subject-level bootstrap resamples with seed 42.

| method | metric | mean metric | mean gain | median gain | improved subjects | 95% bootstrap CI |
|:---|:---|---:|---:|---:|---:|:---|
| head_only | macro MAE | 0.103204 | 0.001885 | 0.001028 | 64.15% | [0.000774, 0.003101] |
| head_only | macro RMSE | 0.131616 | 0.001237 | 0.000475 | 62.26% | [0.000094, 0.002455] |
| head_only | macro R2 | -0.179886 | 0.001401 | 0.000770 | 50.94% | [-0.086062, 0.071098] |
| head_only | macro Spearman | 0.373813 | 0.002858 | 0.001436 | 77.36% | [0.001492, 0.004454] |
| head_only | macro abs bias | 0.040683 | 0.002041 | 0.001439 | 54.72% | [-0.000477, 0.004764] |
| full_model | macro MAE | 0.102404 | 0.002685 | 0.001533 | 67.92% | [0.001506, 0.003980] |
| full_model | macro RMSE | 0.130441 | 0.002411 | 0.001667 | 67.92% | [0.001145, 0.003789] |
| full_model | macro R2 | -0.156171 | 0.025116 | 0.015193 | 62.26% | [-0.043542, 0.085802] |
| full_model | macro Spearman | 0.382941 | 0.011985 | 0.007698 | 67.92% | [0.006442, 0.018499] |
| full_model | macro abs bias | 0.039692 | 0.003031 | 0.002697 | 60.38% | [0.000432, 0.005695] |

For macro MAE, full-model improved 35/53 subjects in at least two seeds
(66.04%) and 28/53 in all three (52.83%). Head-only improved 34/53 in at least
two seeds (64.15%) and 25/53 in all three (47.17%). Mean within-subject seed
standard deviation of MAE gain was 0.001940 for full-model and 0.001595 for
head-only. The latter is slightly more stable per subject, while full-model has
less variation among the three aggregate seed means.

After averaging seeds, the mean number of targets with positive MAE gain per
subject was 4.38/7 for full-model (median 5, range 0--7) and 4.11/7 for
head-only (median 4, range 1--7).

## Method comparisons

After subject-wise seed averaging, full-model outperformed head-only by:

- macro MAE gain: 0.000800, 95% CI [0.000331, 0.001314], 69.81% positive;
- macro RMSE gain: 0.001174, CI [0.000682, 0.001720], 71.70% positive;
- macro R2 gain: 0.023715, CI [-0.009792, 0.053087], 77.36% positive;
- macro Spearman gain: 0.009128, CI [0.004310, 0.015101], 66.04% positive;
- macro abs-bias gain: 0.000990, CI [-0.000221, 0.002234], 62.26% positive.

The full-model advantage is therefore clear for MAE, RMSE, and Spearman, but
the R2 and absolute-bias comparison intervals cross zero.

## Seven-target analysis

Values below are computed after averaging seeds within subjects.

| target | method | MAE before | MAE after | MAE gain | 95% CI | improved | improved >=2 seeds | improved all 3 |
|:---|:---|---:|---:|---:|:---|---:|---:|---:|
| attention | head_only | 0.100296 | 0.101698 | -0.001403 | [-0.004109, 0.001344] | 45.28% | 47.17% | 37.74% |
| attention | full_model | 0.100296 | 0.099013 | 0.001282 | [-0.000605, 0.003346] | 58.49% | 54.72% | 41.51% |
| engagement | head_only | 0.096009 | 0.093570 | 0.002439 | [0.000590, 0.004208] | 71.70% | 60.38% | 43.40% |
| engagement | full_model | 0.096009 | 0.094134 | 0.001875 | [0.000278, 0.003667] | 60.38% | 62.26% | 35.85% |
| excitement | head_only | 0.151190 | 0.145445 | 0.005744 | [0.003305, 0.008421] | 75.47% | 71.70% | 60.38% |
| excitement | full_model | 0.151190 | 0.143947 | 0.007243 | [0.004463, 0.010646] | 75.47% | 75.47% | 67.92% |
| stress | head_only | 0.097439 | 0.096256 | 0.001184 | [-0.001429, 0.003807] | 47.17% | 49.06% | 45.28% |
| stress | full_model | 0.097439 | 0.095571 | 0.001869 | [-0.000749, 0.004562] | 56.60% | 54.72% | 39.62% |
| relaxation | head_only | 0.124332 | 0.121591 | 0.002741 | [0.000928, 0.004778] | 66.04% | 67.92% | 52.83% |
| relaxation | full_model | 0.124332 | 0.121857 | 0.002475 | [0.000877, 0.004119] | 69.81% | 69.81% | 50.94% |
| interest | head_only | 0.071294 | 0.070795 | 0.000499 | [-0.002279, 0.003351] | 52.83% | 47.17% | 32.08% |
| interest | full_model | 0.071294 | 0.070395 | 0.000899 | [-0.001394, 0.003201] | 56.60% | 52.83% | 39.62% |
| focus | head_only | 0.095059 | 0.093071 | 0.001988 | [-0.000403, 0.004639] | 52.83% | 52.83% | 39.62% |
| focus | full_model | 0.095059 | 0.091909 | 0.003150 | [0.000749, 0.006043] | 60.38% | 60.38% | 35.85% |

Full-model has positive mean MAE gain for all seven targets. The bootstrap
interval excludes zero for engagement, excitement, relaxation, and focus.
Attention, stress, and interest remain weaker or heterogeneous. Excitement is
the clearest target; interest has the smallest effect. Head-only has positive
mean gain for six targets, with intervals excluding zero for engagement,
excitement, and relaxation; attention worsens on average.

## Descriptive subset analysis

The subsets reflect different study membership and are descriptive only.
Subjects belonging to `both` are not duplicated into the two exclusive groups.

| subset | subjects | method | macro MAE gain | 95% CI | macro Spearman gain |
|:---|---:|:---|---:|:---|---:|
| Old_EEG | 12 | head_only | 0.000057 | [-0.001916, 0.002137] | 0.002706 |
| Old_EEG | 12 | full_model | 0.000628 | [-0.000987, 0.002523] | 0.007242 |
| both | 30 | head_only | 0.002229 | [0.001052, 0.003572] | 0.003835 |
| both | 30 | full_model | 0.002990 | [0.001831, 0.004290] | 0.018176 |
| gpn_data | 11 | head_only | 0.002939 | [-0.000410, 0.007477] | 0.000358 |
| gpn_data | 11 | full_model | 0.004095 | [0.000304, 0.009293] | 0.000277 |

The full-model mean MAE effect is positive in all three subsets, although the
Old_EEG interval crosses zero. The strongest precise effect occurs in `both`;
gpn_data has a larger point estimate but only 11 subjects. The overall result
is not produced by a single exclusive subset, but the subset heterogeneity is
material and should not be interpreted as a separate transfer experiment.

## Integrity audits

### Split and preprocessing

- 159 fold-subject-seed split rows were checked.
- All outer-train, inner-train, inner-validation, calibration,
  adaptation-train, adaptation-validation, evaluation, and preprocessor hashes
  are identical across seeds for the same fold/subject.
- Outer train/test, inner train/test, inner validation/test, preprocessing-fit
  target, calibration/evaluation, adaptation-train/validation, and
  adaptation/evaluation overlap counts are all zero.
- Duplicate sample-ID count is zero.
- Preprocessor hash has exactly one value per outer fold across all seeds.

### Checkpoints

- All 477 conditions start from their seed/fold global checkpoint.
- Each outer fold has three distinct semantic global checkpoint hashes.
- Zero-shot initial and final hashes are identical for all 159 conditions;
  before and after predictions have maximum absolute difference 0.
- Head-only changes its final checkpoint in all 159 conditions while all frozen
  parameters remain unchanged; 455 parameters are trainable and 65,728 frozen.
- Full-model changes its final checkpoint in all 159 conditions; all 66,183
  parameters are trainable.
- No subject, method, or seed inherits a fine-tuned state from another
  condition.

### Predictions

- All 477 conditions contain seven targets and satisfy
  `rows = evaluation samples x 7`.
- The prediction key
  `(model_seed, outer_fold, subject_id, method, sample_id, target_name)` is
  unique; duplicate count is zero.
- There are 241,913 canonical evaluation sample-target observations. Every one
  appears exactly nine times (three methods x three seeds), with exactly one
  `y_true`.
- Evaluation membership and `y_true` are identical across methods and seeds.
- All `y_true`, before predictions, and after predictions are finite.
- Predictions differ between model seeds, as expected.

## Artifacts

Main run:
`benchmark_results/pm_regression_personalization_multiseed_20pct/20260725_222221/`.

It contains:

- `run_manifest.json`
- `resolved_multiseed.yaml`
- `seed_provenance.csv`
- `global_fold_summary.csv`
- `multiseed_subject_metrics.csv`
- `per_seed_aggregate.csv`
- `multiseed_aggregate.csv`
- `multiseed_target_summary.csv`
- `multiseed_source_summary.csv`
- `stability_summary.csv`
- `paired_comparisons_by_seed.csv`
- `paired_comparisons_multiseed.csv`
- `split_consistency_audit.csv`
- `checkpoint_audit.csv`
- `predictions.parquet`
- `failures.csv`
- the complete seed-specific runs under `seed_7/`, `seed_42/`, and
  `seed_2026/`

Runtime outputs remain ignored and were not added to Git.

## Testing

- Initial baseline: 489 passed, 11 warnings.
- New multiseed unit tests: 31 passed.
- Requested focused regression/personalization tests: 126 passed.
- Full suite before CUDA execution: 520 passed, 11 warnings.
- Final full suite after CUDA execution: 520 passed, 11 warnings. It was run
  with the short `F:\EEG\t` base directory to avoid Windows `MAX_PATH`.

## Limitations and conclusion

This experiment covers one MLP architecture, one 20% calibration budget, and
three model seeds. It does not establish behavior for smaller budgets, other
architectures, or external datasets. Subject-level R2 is noisy and its
multiseed confidence intervals cross zero, so conclusions should not be based
on R2 alone.

Under the prespecified criteria, `full_model` personalization is robust:
macro-MAE gain is positive for every seed, its subject-bootstrap interval is
strictly positive, a majority of users improve in at least two seeds, all seven
targets have positive mean MAE gain, and macro Spearman improves in every seed.
`head_only` is also a robust, lower-update alternative for MAE and Spearman, but
its effect is smaller, attention worsens on average, and R2 is not stable.
Full-model is therefore recommended as the primary 20% personalization method,
with head-only retained as a cheaper conservative baseline. The next useful
step is a preregistered budget-response comparison at 5%, 10%, and 20% using
the same fixed splits and seeds, rather than further tuning on these results.
