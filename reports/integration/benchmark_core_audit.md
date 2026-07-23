# Benchmark core audit

## Scope

Branch: `integration/benchmark-unification`

Reviewed:

- `configs.yaml`
- `cli.py`
- `bench/bench_runner.py`
- `bench/core/abstract_task.py`
- `bench/tasks/tasks_registry.py`
- `model_zoo/factory.py`

## Verification results

| Check | Result |
|---|---|
| Python compilation | passed |
| Structural imports and YAML parsing | passed |
| CLI `--help` | passed |
| Full test suite | passed |

Full test output: `reports/integration/benchmark_core_pytest.txt`

Branch diff: `reports/integration/benchmark_core_branch_diff.txt`

## Structural facts

| Property | Value |
|---|---|
| `tasks` | `cognitive_load_3class, cognitive_load_5class, focus_regression, wesad_4class` |
| `sklearn_models` | `logistic_regression, mlp, random_forest, svm, xgboost` |
| `torch_models` | `torch_bilstm, torch_eegnet, torch_lstm, torch_mlp, torch_shallow_convnet, torch_transformer` |
| `runner_import` | `True` |
| `config_top_level_keys` | `datasets, model_config, models, output_dir, task_config, tasks` |
| `config_tasks` | `cognitive_load_3class, wesad_4class` |
| `config_datasets` | `emotiv_cognitive, wesad` |
| `runner_uses_shared_factory` | `True` |
| `runner_has_get_model_for_split` | `True` |
| `runner_legacy_default_task` | `True` |
| `factory_torch_classification_only` | `True` |
| `cli_has_legacy_3class_default` | `True` |
| `registry_has_5class` | `True` |
| `registry_has_focus_regression` | `True` |

## Errors

- No structural errors detected.

## Warnings and required changes

- configs.yaml has no configured models; it is not a runnable canonical config
- dataset 'emotiv_cognitive' has an empty data_path in configs.yaml
- BenchmarkRunner still references cognitive_load_3class
- Torch factory currently rejects regression; PM/proxy targets require extension
- CLI still references cognitive_load_3class

## Preliminary decision

1. Keep the current shared model factory and current benchmark runner as the integration base.
2. Do not replace them with the colleague versions.
3. Replace the legacy root config with explicit runnable experiment configs instead of mixing all datasets and tasks in one incomplete file.
4. Remove implicit fallback to `cognitive_load_3class` from the canonical execution path.
5. Extend the shared Torch model interface to support regression and later multi-output regression.
6. Preserve `cognitive_load_5class` and `focus_regression` as legacy/diagnostic tasks.
7. Add PM and latent proxy task names only after target builders and multi-output metrics exist.
8. Do not activate experimental mixins before fold-scoped leakage checks are implemented.

## Next patch block

The next code change should be limited to:

- canonical config validation;
- removal of implicit task defaults;
- explicit task/model compatibility checks;
- tests for configuration, task registry and per-split model creation.

PM/proxy regression support should be implemented in the following dedicated block, not mixed into configuration cleanup.
