# Multi-output Performance Metrics regression

## Scope

This change integrates direct seven-output Performance Metrics (PM) regression
into the standard dataset, task, model-factory, runner, metrics, and artifact
pipeline. It does not add a standalone trainer and does not change the
canonical `label_q5` classification configuration in `configs.yaml`.

## Targets and complete-case cohort

The canonical Parquet schema contains these seven regression targets, in the
order used by the experiment:

1. `target_attention`
2. `target_engagement`
3. `target_excitement`
4. `target_stress`
5. `target_relaxation`
6. `target_interest`
7. `target_focus`

The source table has 51,308 rows. Requiring all seven finite targets leaves
43,174 complete cases and drops 8,134 rows. All 448 selected EEG+POW features
are finite on this cohort, so no additional row is dropped by feature
validation. The supervised cohort contains 53 subjects and 117 source records:
22,808 windows from `gpn_data` and 20,366 from `Old_EEG`.

The final arrays have shapes:

```text
X: (43174, 448)
y: (43174, 7)
```

The loader stores the ordered `target_cols`, `n_outputs`, before/after row
counts, and separate target/feature drop counts in dataset metadata. PM
aggregate columns (`PM.*`), every `target_*` column, and service columns are
excluded from input features. `target_col` and `target_cols` are mutually
exclusive.

## Architecture

- Dataset registry: `emotiv_pm_regression` reuses `EmotivDataset`.
- Task registry: `performance_metrics_regression` requires finite 2D targets
  and preserves seven outputs through every split.
- Model factory:
  - `mean_regressor` creates `DummyRegressor(strategy="mean")`;
  - `random_forest` creates `RandomForestRegressor`;
  - `torch_mlp` supports classification and multi-output regression;
  - other Torch architectures remain classification-only.
- Shared Torch adapter: the existing AdamW/DataLoader/early-stopping loop now
  selects either the existing classification objective or a regression
  objective. Regression supports `mse` (default) and `smooth_l1`; it uses a
  linear seven-unit output and does not expose probabilities.
- No target scaler is used. The PM targets are already scaled and metrics and
  predictions remain in their stored units. Feature standardization for the
  MLP is fitted only on the inner-training partition.

## Metrics and artifacts

At window level, MAE, RMSE, R², Pearson, and Spearman are calculated for every
target and macro-averaged across targets. Constant truth or prediction makes
R²/correlation undefined for that target. Macro values use the finite targets,
and every metric records its valid-target count; in particular,
`pearson_valid_targets` and `spearman_valid_targets` make the correlation rule
explicit.

At subject level, true and predicted test-window values are averaged for each
subject and target, then the same metrics are calculated across subjects.
Window metrics retain their established unprefixed names and also have
`window_*` aliases; subject metrics use `subject_*`.

Each fold stores:

- `metrics.json`;
- `predictions.parquet`, with one row per sample and seven `y_true_*` plus
  seven `y_pred_*` columns;
- `per_target_metrics.csv`;
- `subject_target_predictions.csv`.

The protocol directory contains unified versions of the three tabular
artifacts. The Torch MLP additionally stores `model.pt`,
`training_log.csv`, `validation_split.json`, and
`normalization_stats.json`. Standard benchmark JSON, summary CSV, run config,
metrics, and manifest remain unchanged.

## One-fold technical smoke

Command:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --config experiments\pm_regression\pm_regression_smoke.yaml `
  --verbose
```

Run timestamp: `20260724_120934`. The deterministic first GroupKFold split
contains 34,581 train windows from 43 subjects and 8,593 test windows from 10
subjects. Train/test subject overlap is empty. Every model produced an
`(8593, 7)` finite prediction matrix with unique `sample_id`.

| Model | MAE macro | RMSE macro | R² macro | Pearson macro | Spearman macro | Training time, s |
|---|---:|---:|---:|---:|---:|---:|
| mean_regressor | 0.112097 | 0.146009 | -0.008594 | undefined | undefined | 0.002 |
| random_forest | 0.100739 | 0.135586 | 0.118088 | 0.362846 | 0.322378 | 13.746 |
| torch_mlp | 0.103721 | 0.141921 | 0.017724 | 0.309301 | 0.320222 | 2.128 |

Per-target Random Forest metrics:

| Target | MAE | RMSE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Attention | 0.103072 | 0.134018 | 0.029008 | 0.184052 | 0.157686 |
| Engagement | 0.094076 | 0.116364 | 0.134780 | 0.390884 | 0.371696 |
| Excitement | 0.136115 | 0.189405 | 0.230860 | 0.491146 | 0.414239 |
| Stress | 0.095460 | 0.144058 | 0.138903 | 0.413363 | 0.372676 |
| Relaxation | 0.114819 | 0.148970 | 0.130469 | 0.418976 | 0.410169 |
| Interest | 0.071040 | 0.101095 | 0.082091 | 0.353543 | 0.303620 |
| Focus | 0.090591 | 0.115191 | 0.080504 | 0.287962 | 0.226559 |

The MLP used CUDA (`NVIDIA GeForce RTX 5060 Ti`), trained all 3 configured
epochs, selected epoch 3, and reached best inner-validation MSE
`0.01881945`. It has 66,183 trainable parameters.

These are technical one-fold smoke results, not final scientific estimates.

## Verification

Changed-file compilation succeeded. Targeted tests:

```text
55 passed
```

Full suite:

```text
394 passed, 11 warnings
```

The warning count is unchanged from the pre-task baseline and comes from
existing classification edge-case tests.

## Artifacts

Run-level:

```text
benchmark_results/pm_regression_smoke/benchmark_results_20260724_120934.json
benchmark_results/pm_regression_smoke/summary_20260724_120934.csv
benchmark_results/pm_regression_smoke/20260724_120934/config.yaml
benchmark_results/pm_regression_smoke/20260724_120934/metrics.json
benchmark_results/pm_regression_smoke/20260724_120934/run_manifest.json
```

Model artifacts are under:

```text
benchmark_results/pm_regression_smoke/20260724_120934/
  emotiv_pm_regression/performance_metrics_regression/<model>/
  group_kfold_subject/
```

Generated benchmark outputs remain ignored and are not intended for Git.

## Current limitations and next work

- Only fold 1 and seed 42 were run; no five-fold or multiseed inference is
  justified by this smoke.
- The MLP inner validation is a deterministic random subset of the outer
  training fold. Outer evaluation remains subject-disjoint.
- Subject-level fold-1 metrics use only 10 test subjects and are diagnostic.
- No target scaling, missing-target imputation, latent proxy construction, or
  hyperparameter tuning is included.
- Regression support is intentionally limited to the feature MLP; LSTM,
  Transformer, EEGNet, and ShallowConvNet were not changed.

Recommended next stages are full five-fold validation of the direct PM
baselines, followed by regression-capable sequence models, joint PM/proxy
objectives, and fold-safe construction of latent proxy states.

## Five-fold direct PM regression baseline

A five-fold subject-independent GroupKFold experiment was completed for the
mean baseline and a lightweight Random Forest.

### Dataset

- Samples: 43,174
- Subjects: 53
- Input features: 448 EEG + POW features
- Outputs: 7 Performance Metrics
- Subject overlap between train and test: none

### Results

| Model | Macro MAE | Macro RMSE | Macro R? | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Mean regressor | 0.1110 ? 0.0046 | 0.1443 ? 0.0046 | -0.0078 ? 0.0026 | undefined | undefined |
| Random Forest | 0.1003 ? 0.0039 | 0.1314 ? 0.0046 | 0.1443 ? 0.0302 | 0.3838 ? 0.0329 | 0.3315 ? 0.0275 |

The Random Forest consistently outperformed the mean baseline. This confirms
that the EEG and POW feature representation contains predictive information
for the joint Performance Metrics target.

The experiment used the lightweight smoke configuration and should be treated
as the first reproducible baseline rather than the final optimized result.
