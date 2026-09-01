# AutoML architecture audit

Date: 2026-07-16
Branch at audit time: `feature/automl-integration`
Canonical base config: `configs/groupkfold_torch_transformer_label_q5.yaml`

This audit was completed before probing or importing an AutoML backend.

## Existing reusable components

- `BenchmarkRunner` is the canonical programmatic execution API. Constructing
  it with a resolved dictionary and calling `run()` reuses the dataset/task
  registries, model factory, sequence builder, validation, training adapter,
  metrics, predictions, fold artifacts and top-level result serialization.
- `model_zoo.factory.build_model()` already creates `torch_transformer` after
  the runner knows `input_shape` and `num_outputs`. Every GroupKFold fold gets
  a newly constructed model.
- `TorchClassificationAdapter` owns DataLoader creation, optimization, early
  stopping, grouped inner validation, prediction and model/training artifacts.
- `CrossValidator.run_group_kfold()` is the canonical deterministic outer and
  inner fold generator. With `group_column=subject_id`, it verifies zero group
  and subject overlap and records the subject lists in split metadata.
- `MetricsCalculator.calculate_all_metrics()` is the only metric calculator
  needed by a trial. `BenchmarkRunner._aggregate_group_metrics()` already
  produces mean/std values including `balanced_accuracy` and `macro_f1`.
- `benchmark_config_hash()` canonicalizes the scientific config, excludes only
  execution placement (`output_dir`), and calculates SHA-256. Therefore outer
  fold identity, subject restriction and trial parameters must remain in the
  resolved config to affect reuse identity.
- `BenchmarkRunner.find_completed_run()` and `validate_completed_run()` provide
  validated reuse of standard runs by config hash and required artifacts.
- The preprocessing-ablation layer already exposes the required neutral API:
  `resolve_trial_config(base_config, trial_parameters)`. It is the correct
  shared resolver to extend instead of creating an AutoML-only resolver.

## Current config resolution

The canonical Transformer config has one dataset (`emotiv_cognitive`), one
task (`cognitive_load_5class`) and one model (`torch_transformer`). Its fixed
scientific choices are EEG+POW features, `label_q5`, 448 input features,
sequence length 8, last-token pooling, learned positional encoding, record-
grouped adapter validation and 5-fold subject GroupKFold.

`resolve_trial_config()` currently accepts preprocessing switches plus
`training.random_state`, `training.max_epochs`, `dataset.max_windows` and
`evaluation.folds`. It deep-copies and JSON-normalizes the config, but does not
yet accept `model.name`, `model.params.*` or generic `training.*` model
hyperparameters. The minimal compatible change is to extend this resolver with
an explicit allow-list and keep all current paths and semantics unchanged.

The resolver should interpret `model.name` against the single configured model,
`model.params.<name>` as a parameter of that model, and
`training.<name>` as the same adapter/training parameter location. Unknown
paths must continue to fail before model construction.

## Current trial execution path

The preprocessing experiment demonstrates the reusable execution chain:

```text
neutral parameter dictionary
  -> resolve_trial_config(...)
  -> benchmark_config_hash(...)
  -> BenchmarkRunner.find_completed_run(...)
  -> BenchmarkRunner(resolved_config).run()
  -> BenchmarkRunner.completed_run()
```

A completed standard run contains `config.yaml`, `metrics.json`, a run
manifest, top-level JSON/summary CSV, unified predictions and fold-level model,
training-log, prediction and validation artifacts. AutoML needs to retain only
references to those artifacts.

## Outer/inner validation structure

The loaded supervised dataset contains 45,384 rows, 448 features, 54 subjects
and five classes. The current `GroupKFold(n_splits=5)` is deterministic (the
stored `random_state` is provenance; sklearn GroupKFold is not shuffled here).
For outer fold 1 it gives 43 train subjects / 11 untouched test subjects and
36,261 / 9,123 feature rows before sequence construction.

Outer-fold-1 test subjects are:

```text
30c140ca, 3110e0c7, 40f0714a, 7150e10a, 71f0603f, 81f1f0fe,
a1721173, c060c06a, c112918e, d111e017, f1d06060
```

Each search trial must first restrict the feature dataset to the 43 outer-train
subjects and then ask the same runner for 3-fold `group_kfold_subject` inner
evaluation. The objective is the runner's aggregated inner
`balanced_accuracy_mean`; `macro_f1_mean` is secondary. After selection, the
best configuration can be evaluated once through the normal runner with the
original full dataset and `evaluation.folds: [1]`, which retrains on the full
outer-train partition and evaluates the untouched outer test partition once.

The adapter's record-grouped validation remains inside every model fit, so
logical recording overlap is also excluded during early stopping.

## Leakage risks

- Current `EmotivDataset` has no subject inclusion filter. Running a three-fold
  trial directly on the full dataset would expose outer-test subjects to search.
- Outer-test labels must not be passed to the AutoML objective. The orchestration
  layer needs identities only for planning; trial configs must contain only the
  outer-train inclusion list.
- Trial selection must use only aggregated inner metrics. Outer-fold metrics
  are produced once, after the best trial is fixed.
- Changing sequence length, gap threshold, pooling, positional encoding,
  feature representation, preprocessing or class definition would broaden the
  search beyond the declared first track and is disallowed.
- A config hash that omitted the outer fold or subject restriction could reuse
  scientifically different runs. Both must be part of canonical config data.
- Benchmark artifacts are large. Copying models or predictions into the study
  directory would create competing artifact authorities; store references only.

## Required extension points

1. Add a generic, validated `include_subject_ids` option to
   `EmotivDataset.load()`, applied before optional `max_windows` selection and
   recorded in metadata. This is useful beyond AutoML and leaves default loads
   byte-for-byte unchanged.
2. Extend the existing preprocessing-ablation `resolve_trial_config()` for the
   eight declared Transformer search paths and `model.name`, preserving its
   existing paths and serialization.
3. Add neutral, Optuna-independent study/search/trial dataclasses plus search
   constraint validation under `bench/automl/`.
4. Add an objective adapter that resolves a canonical inner config, calls only
   `BenchmarkRunner`, and reads only standard aggregated metrics.
5. Put Optuna and SQLite resume logic exclusively in the orchestration layer.
6. Add a thin main-CLI branch for study plan/run; do not add a second CLI.

No `BenchmarkRunner`, adapter, model factory, metrics, CrossValidator or
sequence-builder changes are required for the first pilot. Pruning will use the
safe no-pruning fallback because adding a generic epoch callback is not needed
to validate the integration and would enlarge the adapter change.

## Components that must not be duplicated

- dataset or task registry;
- feature or sequence loader/builder;
- outer or inner fold generator;
- model factory or Transformer implementation;
- PyTorch training loop, early stopping or probability inference;
- metric calculation or fold aggregation;
- predictions/model/training-log serialization;
- benchmark config hashing and completed-run validation.

## Initial AutoML scope

The initial track is only `torch_transformer` on EEG+POW feature sequences,
`label_q5`, sequence length 8, objective `balanced_accuracy`, secondary
`macro_f1`. Search is limited to `d_model`, `nhead`, `num_layers`,
`dim_feedforward`, `dropout`, learning rate, weight decay and batch size, with
`d_model % nhead == 0` validated before a benchmark run. The pilot is limited
to outer fold 1. It does not search preprocessing, calibration, sequence length,
pooling, representation, target definition or any second model.
