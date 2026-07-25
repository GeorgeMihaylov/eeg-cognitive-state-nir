# Audit of transfer-learning mixins from `feature/benchmarking`

## Executive result

None of the four prototypes is a scientifically runnable benchmark task at commit
`6a1fbd7`. All modules import, but none is registered in `TASK_REGISTRY`, none has
an existing experiment config, and none defines a complete source/target or
support/query evaluation protocol. A real-data one-off call confirms that
`TransferMixin` can train a source model, but the current Torch adapter resets
that state during the calibration `fit`; consequently the measured “after”
score is a calibration-only baseline and not transfer learning.

The recommended integration decision is:

- transfer learning: **requires reimplementation** of the small protocol layer
  around the existing model adapter;
- domain adaptation: **requires reimplementation** of the DANN/model-feature
  contract;
- meta-learning: **defer** until the dependency and episode contract are
  deliberately selected;
- contrastive learning: **requires reimplementation** of encoder injection and
  split handling, although its isolated loss loop runs after one local shape
  fix.

No code was copied to the integration repository.

## Environment

| Item | Observed value |
| --- | --- |
| Worktree | `F:\EEG_mixin_smoke` |
| Git state | detached HEAD |
| Commit | `6a1fbd7ce15b24f97bec3df63925443999e09062` |
| Source ref | `origin/feature/benchmarking` |
| Python | `C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe`, 3.11.15 |
| Torch | `2.11.0+cu128` |
| CUDA | available, CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Feature dataset | `F:\EEG\data\processed\windowed_eeg_pm_dataset_w10.parquet` |
| Dataset availability | present; not copied or modified |
| Supervised rows | 45,384 |
| Shape used by feature models | `[45384, 448]` |
| Subjects / records | 54 / 119 |
| Sources | `gpn_data`: 23,826; `Old_EEG`: 21,558 |
| Raw cache inspected | `F:\EEG\data\interim\raw_eeg_cache_w10` |
| Raw cache shape | 119 arrays; individual arrays `[windows, 14, 2560]`, float32 |

The integration repository remained at branch
`integration/benchmark-unification`, commit
`b96550669e4fcb86cced6d9b8fdd4926b56a330b`, with a clean working tree.

## Current mixin composition

### Transfer learning

`TransferMixin` exposes `pretrain_models(models_dict, pretrain_data=None,
pretrain_task=None)` and `prepare_model(model)`. It expects a task with `config`
and optionally `data`, model entries shaped as
`{"alias": {"model": sklearn_like_model}}`, and a model implementing `fit`.
Weights are captured through `get_weights` or `model.model.state_dict`.

Expected data are a feature matrix and classification labels. There is no
target-user selector, calibration selector, evaluation selector, source-domain
selector, or metric code. `pretrain_epochs` and `finetune_epochs` are read but
never passed to a model. Before the local fix, the configured-dataset path used
an invalid relative import and returned a dataset object without calling
`load()`.

### Domain adaptation

`DomainAdaptationMixin` exposes `pretrain_models`, `prepare_model`, an internal
DANN wrapper, and `train_dann`. It expects an `nn.Module` backbone whose forward
returns a 128-dimensional latent representation, plus explicit source and
target loaders. Neither requirement is represented in the benchmark model
interface.

The prototype infers one domain per `subject_id` (54 on the real dataset),
rather than the requested two source datasets. The discriminator input is
hard-coded to 128. The generic runner passes only `fit(X, y)`, so the wrapper
silently skips DANN optimization when loaders are absent. Wrapping
`TorchClassificationAdapter` then fails at prediction because that adapter is
not a callable `nn.Module`.

### Meta-learning

`MetaMixin` exposes `pretrain_models`, `_maml_train`, and `prepare_model`.
Classification with `learn2learn`, an `is_maml` model flag, and support/query
episodes are expected. `learn2learn` is absent in the active environment, so
the method logs a warning and returns without training.

Static inspection finds additional blockers behind the missing dependency:
real `subject_id` values are strings but are passed directly to
`torch.as_tensor`; the subject episode path builds three-item datasets while
the training loop accepts only two-item tasks; all episodes are constructed
from `self.data` without an external held-out subject split; and no registered
model advertises `is_maml`.

### Contrastive learning

`ContrastiveMixin` implements amplitude scaling, temporal masking, channel
dropout, a small EEG convolutional encoder, projection head, and an NT-Xent-like
loss. The intended shape is raw EEG `[N, C, T]` or `[N, 1, C, T]`.

For a two-dimensional feature matrix it silently creates a pseudo-signal
`[N, 1, 1, 448]`; that is not a valid substitute for raw EEG. The mixin uses
all of `self.data` with no outer split, saves no encoder weights, and can attach
the trained encoder only to a downstream model implementing `set_encoder`.
No model in this branch implements that method.

## As-is results

| Method | Import | Registered | Existing config | Training | Evaluation | Metrics | Initial status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transfer learning | yes | no | no | isolated synthetic source fit completed | no integrated evaluation | no | `not_wired_to_runner` |
| Domain adaptation | yes | no | no | wrapper only; no optimization | failed | no | `blocked_by_architecture` |
| Meta-learning | yes, with warning | no | no | skipped | no | no | `blocked_by_dependency` |
| Contrastive learning | yes | no | no | started, then failed | no | no | `blocked_by_architecture` |

The initial contrastive exception was:

```text
RuntimeError: The size of tensor a (32) must match the size of tensor b (16)
at non-singleton dimension 1
```

The initial full test suite did not reach tests:

```text
NameError: name 'WESADDataset' is not defined
```

The same dataset-registry exception made the CLI and `BenchmarkRunner`
unreachable before configuration parsing.

## Minimal local fixes

| File | Defect | Minimal change | Effect |
| --- | --- | --- | --- |
| `bench/datasets/base_eeg_data_loader.py` | import of nonexistent `bench.core.dataset` | use `abstract_dataset` | Emotiv loader imports |
| `bench/datasets/datasets_registry.py` | loader classes not imported; Emotiv absent | import both loaders and register `emotiv_cognitive` | runner can load the canonical Parquet |
| `bench/bench_runner.py` | nonexistent `bench.models.factory` | use `model_zoo.factory` | runner imports |
| `bench/datasets/emotiv_loader.py` | persisted `label_q5` loaded as float | validate integral values and cast to int64 | classification task accepts the target without recomputing it |
| `bench/tasks/mixin/transfer_learning.py` | wrong relative import and unloaded dataset | import from `bench.datasets` and call `load()` | configured pretrain dataset becomes an `EEGData` object |
| `bench/tasks/mixin/contrastive_learning.py` | identity mask was twice the similarity-matrix size | use `all_z.size(0)` | isolated contrastive epoch completes |
| `cli.py` | missing pandas import; `--verbose` bypassed config loading; `--test` unhandled | add import and independent config selection | CLI is reachable with `--config ... --verbose` |

These changes do not register the prototypes and do not change their scientific
algorithms.

## Results after minimal fixes

| Method | Import | Registered | Local config | Training started | Training completed | Evaluation completed | Metrics | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfer learning | yes | no | yes, one-off | yes | yes | yes, diagnostic | yes | `methodologically_invalid` |
| Domain adaptation | yes | no | no | no actual batches | no | no | no | `blocked_by_architecture` |
| Meta-learning | yes | no | no | no | no | no | no | `blocked_by_dependency` |
| Contrastive learning | yes | no | no | yes, synthetic raw shape | yes | no downstream | no | `not_wired_to_runner` |

The control runner command after fixes was:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --config experiments\mixin_smoke\runner_reachability_label_q5.yaml `
  --verbose
```

It loaded 45,384 rows and 448 EEG+POW features. A 10-tree RF control completed
the existing random within-subject split with 38,553 train and 6,831 test
windows: accuracy 0.325721, macro F1 0.295382, weighted F1 0.295535, and kappa
0.157094. This is only a runner reachability check, not a mixin result and not a
scientific subject-independent estimate.

## Transfer real-data diagnostic

The one-off smoke used:

- source: `gpn_data`, a seed-42 stratified limit of 2,000 windows from 41
  subjects;
- target: `Old_EEG`, subject `8191f1d9`;
- calibration: 100 target windows;
- final evaluation: 642 target windows;
- model: Torch MLP `[448, 64, 32, 5]`;
- one adapter epoch for source and one adapter epoch for calibration;
- CUDA; seed 42.

Leakage checks:

- source and target subjects overlap: 0;
- source and target row IDs overlap: 0;
- calibration and evaluation row IDs overlap: 0;
- the Parquet has no canonical `sample_id`, so its immutable row index was used
  for this disposable smoke and hashed in `split_audit.json`.

| Metric | Source-pretrained model | After calibration `fit` | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.183801 | 0.204050 | +0.020249 |
| Balanced accuracy | 0.235662 | 0.211461 | -0.024201 |
| Macro F1 | 0.125731 | 0.117301 | -0.008430 |

Source training took 1.144 s and calibration fit 0.013 s. The values must not be
reported as transfer gain: `TorchClassificationAdapter.fit` reloads
`_initial_state` before every fit. `prepare_model` also cannot load the captured
state into the adapter (`Model does not support weight loading`). The second
fit therefore trains a fresh randomly initialized model on 100 calibration
windows. The experiment is valuable as a technical failure reproduction, not
as a model comparison.

Artifacts:

- `benchmark_results/mixin_smoke/transfer_learning/config.yaml`
- `benchmark_results/mixin_smoke/transfer_learning/metrics.json`
- `benchmark_results/mixin_smoke/transfer_learning/predictions.parquet`
- `benchmark_results/mixin_smoke/transfer_learning/split_audit.json`
- `benchmark_results/mixin_smoke/transfer_learning/training_log.csv`
- `benchmark_results/mixin_smoke/transfer_learning/console.log`

## Other real/synthetic attempts

### Domain adaptation

The real dataset was attached to the task and the prototype inferred 54 subject
domains. The runner-contract call `fit(X, y)` logged:

```text
No source/target loaders provided; falling back to standard training.
```

No optimization occurs in that branch. Prediction then failed with:

```text
TypeError: 'TorchClassificationAdapter' object is not callable
```

There is no valid source-only versus DANN metric. Result:
`benchmark_results/mixin_smoke/domain_adaptation/result.json`.

### Meta-learning

`import learn2learn` raises `ModuleNotFoundError`. The dependency was not
installed. The mixin inspected the real 45,384-row/54-subject contract and
skipped training. Result:
`benchmark_results/mixin_smoke/metalearning/result.json`.

### Contrastive learning

Before the fix, a correct synthetic raw-like tensor `[24, 1, 4, 64]` failed in
the contrastive mask. After the one-line shape fix, one epoch over three
batches completed and produced an encoder and projection head.

The available raw cache was inspected without rebuilding it and has compatible
physical windows `[14, 2560]`. A real pretraining run was not started because
the remaining required conditions are absent: no runnable task is registered
and no downstream model has `set_encoder`. Running on the 448-feature Parquet
as `[N,1,1,448]` was rejected as methodologically invalid. Result:
`benchmark_results/mixin_smoke/contrastive_learning/result.json`.

## Leakage and methodological risks

| Risk | Transfer | Domain | Meta | Contrastive |
| --- | --- | --- | --- | --- |
| Full-data pretraining | avoidable only in one-off manual split; runner path unsafe | `self.data` is global | `self.data` is global | `self.data` is global |
| Calibration/evaluation overlap | zero in one-off split; no built-in check | no calibration protocol | support/query generator only | not applicable |
| Target-test labels used in training | no in one-off split | labels not separated by runner | all labels available globally | labels unused |
| Subject overlap | zero in one-off split | no outer split | no outer split | no outer split |
| Source/target record overlap | zero by source and subject in one-off | not checked | not checked | not checked |
| Support/query overlap | not applicable | not applicable | sampling intends disjoint indices but no outer isolation | positive views intentionally share a window |
| Preprocessing learned on test | adapter standardizer is source-train-only before evaluation; calibration fit learns anew | undefined | undefined | undefined |

## Tests

Baseline:

```text
pytest collection error: NameError WESADDataset is not defined
```

After local fixes:

```text
11 passed, 13 failed
```

All 13 failures are pre-existing runner/test contract drift now exposed after
the collection blocker was removed: tests expect `runner.models` and
`_create_model`, mocks use obsolete `get_all_splits` behavior, and integration
fixtures request an unregistered `test_dataset`. Fixing that suite would exceed
this smoke audit. Final direct checks succeeded for:

- all four mixin imports;
- `BenchmarkRunner` import;
- CLI `--help`;
- one-epoch Torch MLP source fit;
- one-epoch synthetic contrastive fit after the mask fix.

## Integration recommendation

1. **Transfer first, but not as-is.** Reuse the current adapter/model factory,
   add an explicit `load_state` or fine-tune-without-reset contract, and
   implement a small outer-subject plus target calibration/evaluation protocol
   with persisted sample IDs. Do not port this mixin unchanged.
2. **Contrastive second only after encoder contracts exist.** Define
   `encode`/`set_encoder`, consume raw windows, train only on outer-train data,
   and save encoder weights. The loss-mask fix alone is insufficient.
3. **Domain adaptation requires redesign.** A backbone must expose latent
   features with a declared dimension, and the runner must build source and
   unlabeled-target loaders. Do not integrate the current wrapper.
4. **Meta-learning should be deferred.** Select and pin the dependency, define
   subject-disjoint episodes, encode string IDs safely, and add a compatible
   MAML model before attempting metrics.

The machine-readable status/metrics table is
`reports/mixin_smoke/mixin_smoke_metrics.csv`.
