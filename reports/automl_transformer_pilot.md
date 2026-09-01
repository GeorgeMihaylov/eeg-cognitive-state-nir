# Transformer nested AutoML pilot

Date: 2026-07-16
Branch: `feature/automl-integration`
Study: `transformer_label_q5_pilot`
Backend: Optuna 4.9.0 with SQLite

## Result

The nested pilot completed successfully for outer fold 1 only. All 15 trials
were selected exclusively from three subject-grouped inner folds over the 43
outer-train subjects. After selection, the best configuration was retrained by
the canonical `BenchmarkRunner` on outer-train and evaluated once on the 11
untouched outer-test subjects.

The best inner balanced accuracy was **0.362181** and inner macro F1 was
**0.348580**. On the untouched outer fold, balanced accuracy was **0.364535**
and macro F1 was **0.349577**. This is below the existing seed-42 Transformer
baseline on the identical outer observations by 0.018840 balanced accuracy and
0.019075 macro F1. This single-fold difference is descriptive, not a claim of
statistical significance.

Given the absence of an outer-fold improvement and the limited 15-trial search,
the pilot does not justify automatically expanding nested AutoML to all five
outer folds. A separate decision should first consider whether to revise the
search prior, training budget or objective robustness.

## Protocol

```text
fixed outer GroupKFold(subject_id), fold 1
  outer train: 43 subjects, 36,261 feature windows
  outer test:  11 subjects,  9,123 feature windows
        |
        +-- 15 Optuna trials
              3-fold inner GroupKFold(subject_id)
              canonical sequence builder (length 8)
              canonical Transformer factory/adapter/metrics/artifacts
        |
        +-- select max mean inner balanced_accuracy
        |
        +-- canonical retrain/evaluate once on outer fold 1
```

Fixed choices were EEG+POW features, `label_q5`, input `[B, 8, 448]`, last-token
pooling, learned positional encoding, existing sequence gap policy and grouped
`record_group_id` adapter validation. Preprocessing, target, representation,
sequence length, calibration and model family were not searched.

## Best parameters

| Parameter | Value |
|---|---:|
| `d_model` | 128 |
| `nhead` | 8 |
| `num_layers` | 2 |
| `dim_feedforward` | 256 |
| `dropout` | 0.3472567 |
| `learning_rate` | 0.0002075861 |
| `weight_decay` | 0.0077771123 |
| `batch_size` | 256 |

The selected architecture has 340,869 trainable parameters, the same parameter
count as the canonical baseline. Training used CUDA on an NVIDIA GeForce RTX
5060 Ti.

## Best-trial inner folds

| Inner fold | Train seq. | Test seq. | Accuracy | Balanced accuracy | Macro F1 | Epochs | Best epoch | Best validation loss | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23,513 | 11,829 | 0.366895 | 0.376024 | 0.367463 | 12 | 8 | 1.176129 | 7.627 |
| 2 | 23,554 | 11,788 | 0.359433 | 0.353405 | 0.338104 | 12 | 9 | 1.265562 | 7.815 |
| 3 | 23,617 | 11,725 | 0.360597 | 0.357114 | 0.340173 | 7 | 3 | 1.301146 | 5.121 |
| **Mean ± SD** | — | — | **0.362308 ± 0.003278** | **0.362181 ± 0.009905** | **0.348580 ± 0.013379** | **10.33 ± 2.36** | **6.67 ± 2.62** | **1.247613 ± 0.052592** | **20.563 total** |

## Untouched outer-fold comparison

The comparison uses exactly the same 8,800 `sequence_id/fold/y_true` rows for
both models.

| Metric | AutoML selected | Baseline Transformer | Difference |
|---|---:|---:|---:|
| Accuracy | 0.358182 | 0.370682 | -0.012500 |
| Balanced accuracy | 0.364535 | 0.383375 | -0.018840 |
| Macro F1 | 0.349577 | 0.368652 | -0.019075 |
| Weighted F1 | 0.350277 | 0.365078 | -0.014801 |
| Kappa | 0.202713 | 0.221099 | -0.018386 |
| AUC (weighted OVR) | 0.698637 | 0.721201 | -0.022564 |
| Epochs trained | 10 | 15 | -5 |
| Best validation loss | 1.343987 | 1.168292 | +0.175695 |
| Fold training time (s) | 9.552 | 23.353 | -13.801 |

The pilot used the requested 12-epoch cap whereas the existing baseline config
allowed 15 epochs. Early stopping selected epoch 6 for the AutoML configuration.
This budget difference and the single outer fold limit the interpretation.

## Search execution

- Trials: 15 complete, 0 failed, 0 pruned/rejected.
- Unique resolved config hashes: 15.
- Unique canonical benchmark references: 15.
- Summed inner-trial runtime: 762.715 seconds.
- End-to-end CLI wall time including final outer evaluation: 787.7 seconds.
- Smoke: 3/3 complete, two inner folds, three epochs, 3,000 windows; resume
  retained three trials and did not create another benchmark run.
- No epoch pruning was used. The safe fallback avoids adding Optuna coupling to
  `TorchClassificationAdapter`.

Parameter importance from this small study is exploratory:

| Rank | Parameter | Importance |
|---:|---|---:|
| 1 | `dropout` | 0.384942 |
| 2 | `batch_size` | 0.194501 |
| 3 | `weight_decay` | 0.121586 |
| 4 | `d_model` | 0.117707 |
| 5 | `learning_rate` | 0.087416 |
| 6 | `dim_feedforward` | 0.068241 |
| 7 | `nhead` | 0.023673 |
| 8 | `num_layers` | 0.001934 |

## Leakage and artifact audit

- Every trial config contains exactly the 43 outer-train subject IDs and none
  of the 11 outer-test IDs.
- Every trial contains three subject-grouped benchmark folds with zero outer
  train/validation subject overlap.
- Adapter validation manifests report zero `record_group_id` overlap.
- The final outer split reports zero train/test subject overlap.
- Outer predictions contain 8,800 unique sequence IDs, finite `proba_0`–
  `proba_4`, and maximum probability-row-sum error `2.38e-7`.
- Final outer `sequence_id`, `fold` and `y_true` exactly match the seed-42
  baseline fold-1 predictions.
- AutoML study artifacts contain references; model checkpoints, predictions,
  logs and fold artifacts remain authoritative in standard benchmark runs.

## Artifacts

Study directory:
`benchmark_results/automl/transformer_label_q5/transformer_label_q5_pilot`

SQLite storage:
`benchmark_results/automl/transformer_label_q5/study.db`

Best inner benchmark run:
`benchmark_results/automl/transformer_label_q5/transformer_label_q5_pilot/benchmark_runs/d05ce34ab5f24eb643df/20260716_204106`

Selected outer benchmark run:
`benchmark_results/automl/transformer_label_q5/transformer_label_q5_pilot/outer_evaluation/9f1b5fca9a09483de9f5/20260716_204146`

Baseline run:
`benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246`

Study manifests are `study_spec.yaml`, `search_space.yaml`, `outer_folds.json`,
`study_summary.json`, `trials.parquet`, `best_trials.json` and
`environment.json`. The versioned flat trial table is
`reports/automl_transformer_trials.csv`.
