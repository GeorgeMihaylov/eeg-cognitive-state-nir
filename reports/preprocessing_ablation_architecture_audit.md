# Preprocessing ablation architecture audit

Этот отчёт фиксирует историческую архитектуру на момент аудита. Указанный ниже
`src/13_run_preprocessing_ablation.py` впоследствии был заменён каноническим
`scripts/run_preprocessing_ablation.py` и больше не является рабочим путём.

## Current call graph

```text
scripts/run_preprocessing_ablation.py
  -> PreprocessingAblation.plan()
     -> expand_factorial_trials()
     -> resolve_cache()
  -> PreprocessingAblation.execute()
     -> PreprocessingAblation.run_trial()
        -> _FoldLimitedBenchmarkRunner(config).run()
           -> dataset registry
           -> task registry
           -> CrossValidator
           -> model_zoo.build_model
           -> MetricsCalculator
           -> standard fold and prediction artifact saving
        -> second result/summary/prediction serialization
        -> independent trial_status.json completion state
```

The numerical training and evaluation loop is already delegated to the existing
benchmark. The architectural duplication is around it: run identity, fold
limiting, completion state, output roots, and copies of standard artifacts.

## Responsibility mapping

| New component/function | Responsibility | Existing benchmark equivalent | Duplicate responsibility | Keep/refactor/remove | Reason |
|---|---|---|---|---|---|
| `PreprocessingSpec`, `PreprocessingStepSpec` and preprocessing registry | Validate, serialize, hash and apply signal preprocessing; declare `fit_scope` and cacheability | Raw EEG data/preprocessing layer | No | Keep | This is data-layer functionality and is reusable by cache building and future optimizers. |
| `load_experiment_spec()` | Load and validate the factorial declaration | No matrix equivalent | No | Keep, narrow | The benchmark accepts one resolved config, not a parameter matrix. Matrix-only validation belongs here. |
| `expand_factorial_trials()` | Expand the Cartesian product and attach display IDs A–H | No matrix equivalent | No | Keep, refactor | Trial IDs may label results, but parameter semantics must come exclusively from the parameter dictionary. |
| `_validate_semantic_cache()` / `resolve_cache()` | Validate preprocessing/cache compatibility and locate an index | Raw cache builder validates individual shards, but no cross-cache semantic lookup exists | Partial, cache-specific | Keep | Cache resume is explicitly a preprocessing-layer responsibility. |
| `build_cache()` | Invoke existing raw index/cache builders and validate the result | `build_raw_window_index()` and `build_raw_eeg_cache()` | No numerical duplication | Keep as orchestration | It calls the established cache implementation rather than reimplementing filtering or shard writing. |
| `TrialPlan.model_parameters` and `_benchmark_config()` | Construct a benchmark config from matrix fields and CLI overrides | Canonical benchmark config consumed by `BenchmarkRunner` | Partial | Replace with one public `resolve_trial_config()` | CLI, matrix execution and future AutoML must use the same parameter resolver and deterministic config hash. |
| `_FoldLimitedBenchmarkRunner` | Truncate the split dictionary for smoke runs | `BenchmarkRunner.run_for_dataset()` already supports `evaluation.folds` | Yes | Remove | Fold selection belongs in the canonical benchmark config. A runner subclass is unnecessary. |
| `PreprocessingAblation.run_trial()` runner call | Invoke training/evaluation | `BenchmarkRunner(config).run()` | No | Keep as one direct call | The ablation layer is allowed to invoke the existing runner once per resolved trial. |
| `run_trial()` model creation | None directly; runner creates a model per fold | `BenchmarkRunner._create_model()` → `model_zoo.build_model()` | No | Keep delegated | The current runner correctly creates fresh GroupKFold models through the shared factory. |
| `run_trial()` fold loop | None directly; `_FoldLimitedBenchmarkRunner` intercepts the fold mapping | `CrossValidator.run_group_kfold()` and `BenchmarkRunner._evaluate_group_kfold()` | The limiter duplicates fold selection policy | Remove limiter | The matrix layer must not own or modify the fold loop. |
| `run_trial()` metric calculation and aggregation | Reads aggregated runner output | `MetricsCalculator` and `BenchmarkRunner._aggregate_group_metrics()` | No calculation duplication | Keep read-only aggregation | Factorial reporting may consume standard metrics but must not recalculate benchmark metrics. |
| `artifact_path` and `plan._artifact_path()` | Create per-trial output root | `BenchmarkRunner.output_dir`, timestamp run directory and `_model_artifact_dir()` | Yes | Refactor to reference directory plus standard benchmark output directory | Standard model/fold artifacts must have one owner and one location. |
| `benchmark_results.json` copy | Serialize `runner.results` a second time | `BenchmarkRunner._save_results()` writes timestamped result JSON and run `metrics.json` | Yes | Remove | The standard benchmark result is authoritative. |
| `summary.csv` copy | Serialize the summary a second time | `BenchmarkRunner._save_csv_summary()` | Yes | Remove | The canonical summary already exists. |
| root `predictions.parquet` copy | Duplicate unified predictions outside the standard protocol directory | `BenchmarkRunner._evaluate_group_kfold()` | Yes | Remove | Predictions must exist only in the standard run directory. |
| ablation `preprocessing_metadata.json` and `cache_manifest.json` | Duplicate metadata beside the second output root | Dataset/fold preprocessing artifacts and cache manifests | Yes for run artifacts | Remove from trial output | The resolved trial can reference the preprocessing hash and cache index without copying benchmark artifacts. |
| `trial_status.json` and `_completed_status_matches()` | Independent benchmark completion truth and resume signature | No public standard resume API; standard run artifacts contain the necessary evidence | Yes, caused by missing extension point | Remove and add minimal standard run manifest/config-hash validation | Cache readiness must not imply benchmark completion, and a reference must not become a second truth source. |
| `resolved_trial.yaml` | Record matrix parameters and the exact resolved config | Standard run `config.yaml` stores only the benchmark config | No | Keep | This is allowed experiment-layer metadata and supports auditability. |
| proposed `trial_reference.json` | Point from trial/seed to the authoritative standard benchmark run | No existing reference artifact | No | Add | It is a pointer whose validity is checked against the standard run, not a completion database. |
| `execute()` loop over trials | Sequentially invoke one benchmark per matrix row | Benchmark runner handles datasets/tasks/models/folds within one config | No | Keep | A matrix necessarily has a trial loop; GPU trials remain sequential. |
| dataset loading | Pass cache index and deduplication mode to the benchmark | Dataset registry → `RawEEGWindowDataset.load()` | No | Keep delegated | Logical-record deduplication remains in the established dataset implementation. |
| prediction validation | Runner checks unified row count and duplicate observation IDs | `BenchmarkRunner._evaluate_group_kfold()` | Any second validation would duplicate benchmark ownership | Standardize completion validation in runner | Resume validation may verify that required standard artifacts exist, but must not create or rewrite predictions. |

## Required target architecture

```text
Experiment matrix
  -> parameter dictionary independent of trial ID
  -> semantic cache resolution
  -> resolve_trial_config(base_config, parameters)
  -> deterministic benchmark config hash
  -> BenchmarkRunner.find_completed_run(...) or BenchmarkRunner(config).run()
  -> standard benchmark result and artifacts
  -> trial_reference.json
  -> read-only factorial/multiseed aggregation
```

## Minimal changes

1. Add deterministic config serialization/hash and completed-run discovery to
   `BenchmarkRunner`. The standard runner will write a small `run_manifest.json`
   beside its existing run-level `config.yaml` and `metrics.json`.
2. Express smoke limits through existing canonical fields:
   `evaluation.folds`, dataset `max_windows`, and model `max_epochs`.
3. Replace `_benchmark_config()` with a public, AutoML-ready
   `resolve_trial_config(base_config, trial_parameters)` and reject unknown
   dotted parameters.
4. Remove `_FoldLimitedBenchmarkRunner`, `trial_status.json`, copied results,
   copied summary, copied predictions, and copied benchmark metadata.
5. Store only `resolved_trial.yaml` and `trial_reference.json` in the matrix
   reference directory.
6. Add `--experiment-matrix` dispatch to the existing `cli.py`; expose it
   through `scripts/run_preprocessing_ablation.py`.

## Compatibility risks

- Existing seed-42 directories contain valid standard timestamped benchmark
  runs plus legacy duplicate files at the ablation root. They must not be
  deleted. Semantic migration must compare the saved standard `config.yaml`
  while ignoring output-location-only differences.
- The current benchmark has no explicit completed-run manifest. Legacy lookup
  therefore needs a read-only fallback that validates `config.yaml`, result
  JSON, unified predictions and fold artifact paths.
- `output_dir` cannot participate in the scientific config hash if the output
  directory itself is derived from that hash. It is treated as execution
  placement, while dataset, model, split, preprocessing and seed remain hashed.
- On Windows the standard nested artifact path would exceed `MAX_PATH` with a
  full 64-character hash directory. Placement therefore uses `runs/<20-char
  prefix>` while the full hash remains in the config, reference and manifest;
  resume always validates the full hash, so prefix collisions cannot be reused.
- Smoke and full hashes must differ through canonical scientific fields, not
  through directory names: selected folds, `max_windows`, and `max_epochs`.
- Paths must be normalized to serializable strings without changing the paths
  consumed by the existing dataset loader.
