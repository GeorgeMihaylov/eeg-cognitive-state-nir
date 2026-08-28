# Final package unification — baseline audit

## Safety baseline

- Branch: `refactor/final-package-unification-20260828`
- HEAD: `3b50f35d45522cb6faa1c40da10409b32089b081`
- Working tree and staging before the task: clean
- Protected integration branch was not checked out or modified.

## Architecture before migration

`model_zoo/` contains the canonical model implementations and factory.
`cogstate/model_zoo/` is a facade over those implementations plus streaming and
masked multitask extensions.  `automl/` contains the application/personalized
portfolio while `bench/automl/` contains leakage-safe scientific nested
optimization.  `bench/meta/` combines dataset-dependent episode orchestration
with reusable FOMAML mechanics.  Regression calibration primitives are embedded
in `bench/experiments/pm_regression_personalization.py`.

The source tree has no root `src/` package.

## Dependency inventory

- Python files with direct `model_zoo` imports: 107.
- Python files with direct root `automl` imports: 0; the package is currently
  effectively disconnected and `automl/bindings.py` contains stale imports.
- `cogstate -> bench` executable Python imports: 0.  Two docstrings under
  `cogstate/evaluation` mention `bench.validation`, but do not import it.
- Non-Python `model_zoo.*` paths occur only in historical Markdown provenance;
  no active YAML/JSON/TOML config contains a model module path.
- Dynamic config loader: `bench.analysis.experiment_config_audit` imports a
  configured loader module.  Its active registry entry for scientific AutoML is
  `bench.automl.study_runner` and must migrate explicitly.

## Model implementation inventory

Canonical root modules:

- `model_zoo/base.py`, `factory.py`;
- `model_zoo/DL/{adapter,contrastive,dann,eegnet,encoder,feature_preprocessing,lstm,mlp,ordinal,regression,sequence_utils,shallow_convnet,shallow_fusion,transformer}.py`;
- `model_zoo/ML/{multitask,sklearn_models,xgboost_personalization}.py`.

Application extensions already under `cogstate.model_zoo`:

- `TorchMultiTaskClassificationAdapter` and masked-label training;
- `TorchShallowConvNetMultiTaskClassifier`;
- `StreamingModelAdapter`, `StreamingPMMultiTaskAdapter`;
- `load_torch_weights`;
- application-level multitask and streaming exports.

The facade files are not independent canonical implementations.  During the
migration the extensions will be retained in dedicated local modules and
re-exported through the sole `cogstate.model_zoo` factory/API.

## AutoML inventory

| Current path | Responsibility | Destination |
|---|---|---|
| `bench/automl/*.py` | scientific nested optimization | `bench/automl/scientific/*.py` |
| `automl/*.py` | personalized candidate portfolio/adaptation | `bench/automl/personalized/*.py` |
| `automl/bindings.py` | stale application bindings | repaired under `bench.automl.personalized` using `cogstate.model_zoo` |

The two search/split/objective semantics remain separate and are not merged.

## Symbol and path migration map

| Old symbol/path | New symbol/path | Consumers | Metadata impact | Test coverage |
|---|---|---|---|---|
| `model_zoo.factory.build_model` | `cogstate.model_zoo.factory.build_model` | benchmark, experiments, apps, tests | model name/params unchanged | factory and model suites |
| `model_zoo.DL.*` | `cogstate.model_zoo.DL.*` | raw/sequence/transfer experiments | Python module path changes; state-dict format must not | model, streaming, DANN, contrastive tests |
| `model_zoo.ML.*` | `cogstate.model_zoo.ML.*` | sklearn/multitask/personalization | estimator params unchanged | factory, multitask, XGBoost tests |
| `bench.meta.{buffers,protocol,fomaml}` | `cogstate.adaptation.meta_learning.*` | FOMAML core and audits | class module path changes; state hashes do not | FOMAML/buffer/clone tests |
| calibration functions in `bench.experiments.pm_regression_personalization` | `cogstate.adaptation.regression_calibration` | PM personalization | numeric contract unchanged | regression personalization tests |
| `bench.automl.*` | `bench.automl.scientific.*` | CLI/config audit/tests | configured loader module path changes only | scientific AutoML tests |
| `automl.*` | `bench.automl.personalized.*` | application portfolio | package relocation only | personalized AutoML tests |

## Metadata and hash risk

`bench.meta.buffers.architecture_schema_signature` currently hashes
`model.__class__.__module__`.  Moving model classes would therefore change a
signature even when architecture and weights are identical.  The migration must
introduce a stable logical class identifier that maps historical
`model_zoo.*` and current `cogstate.model_zoo.*` paths to the same canonical
identity, with a regression test proving legacy/current equality.  No protocol,
fold, target, metric, prediction, checkpoint tensor or scientific artifact will
be rewritten.

PyTorch persistence stores model type, input shape, training config and state
dict rather than pickling the Python class.  Existing bundle loading must be
tested after relocation.  Historical Markdown provenance is intentionally not
rewritten.

## Baseline tests

Full baseline command used the required interpreter and an isolated short
temporary path.  Result:

```text
1540 passed, 11 skipped, 80 failed, 74 errors, 38 warnings
```

The failures/errors are pre-existing and dominated by absent local data,
`benchmark_results`, generated CSV/report artifacts, and byte-exact provenance
fixtures.  A first invalid attempt using a missing basetemp parent produced
setup-path errors and is not treated as the baseline.

Targeted post-block suites will cover factory/sklearn, adapters, EEGNet, MLP,
LSTM/BiLSTM, ShallowConvNet, Transformer, sequence helpers, ordinal, regression,
fusion, DANN, contrastive, multitask, streaming, FOMAML/meta, calibration,
scientific AutoML, personalized AutoML, and architecture boundaries.
