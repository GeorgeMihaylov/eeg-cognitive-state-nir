# Contrastive pretraining infrastructure for raw-EEG encoders

## Status and scope

- Branch: `integration/benchmark-unification`
- Audited HEAD: `65290df`
- Initial working tree: clean; staging area empty.
- Result status: `diagnostic`
- Supported encoders: EEGNet and ShallowConvNet.
- Excluded model: the aggregated-feature Transformer.
- No dataset loader, model registry, `BenchmarkRunner`, private training loop,
  COG-BCI loader, or scientific experiment was added.

## Historical prototype audit

The historical prototype was inspected read-only with:

```text
git show 8ecbee9:bench/tasks/mixin/contrastive_learning.py
```

Reusable ideas:

- create two stochastic views of the same raw EEG window;
- attach a projection head after an encoder;
- use the other samples in the current batch as negatives;
- discard the projection head for downstream tasks.

Ideas retained with changes:

- the projection input is now `model.latent_dim`, not a hardcoded width;
- augmentations operate only on validated `[B, 1, channels, time]` tensors;
- the existing EEGNet/ShallowConvNet encoder is reused through `encode`;
- the loss exposes numerator and denominator for dataset-level aggregation;
- the loader receives explicit fold indices and preserves provenance;
- CPU is supported without assuming CUDA.

Rejected unsafe parts:

- training over all `self.data`;
- the independent 64-dimensional `EEGEncoder`;
- reshaping two-dimensional aggregated features into pseudo-raw EEG;
- hardcoded projection and encoder widths;
- an incompatible standalone epoch loop;
- access to outer-test samples, subjects, records, or labels;
- unconditional CUDA selection;
- a global negative queue.

## Architecture

The implemented flow is:

```text
authorized raw EEG window
├─ augmentation A → shared encoder → latent A → projection head → z1
└─ augmentation B → shared encoder → latent B → projection head → z2

z1 + z2 → in-batch NT-Xent objective
```

`ContrastiveModule` receives an already constructed encoder-compatible model.
It does not replace or modify the model's ordinary output head. Consequently:

```text
model(X)
```

retains its existing classification or regression behavior. The projection
head is a separate module and never appears in the ordinary EEGNet or
ShallowConvNet state dict.

The projection head is:

```text
Linear(model.latent_dim, projection_hidden_dim)
→ ReLU
→ Linear(projection_hidden_dim, projection_dim)
→ explicit L2 normalization
```

`projection_dim` and the optional hidden width are configuration values.
`latent_dim` always comes from the model:

| Encoder | Raw input | Latent representation |
|---|---|---|
| EEGNet | `[B, 1, channels, time]` | model-derived flattened convolution output |
| ShallowConvNet | `[B, 1, channels, time]` | configured filter count |

The feature Transformer is excluded because its input is a sequence of
aggregated EEG+POW features rather than compatible raw EEG.

## EEG augmentation pipeline

The pipeline applies a fixed, serialized order:

1. Gaussian noise;
2. amplitude scaling;
3. time masking;
4. channel masking;
5. temporal shift.

Every transformation:

- preserves `[B, 1, channels, time]`;
- is independently enabled or disabled;
- has an explicit probability and numerical parameters;
- uses the supplied PyTorch generator;
- works on CPU and can use a device-matched CUDA generator;
- clones its input instead of mutating it;
- validates finite output;
- receives no labels.

Channel masking only sets selected channels to zero; it never reorders
electrodes. Temporal shift is a circular shift and therefore introduces a
wrap-around boundary. It is included only as a conservative configurable
primitive, not as a scientifically selected augmentation policy.

No aggressive frequency warping, channel permutation, cross-window mixing, or
label-dependent transformation is implemented.

## Contrastive objective

The objective is an in-batch NT-Xent/InfoNCE equivalent:

- two projections of source window `i` form its positive pair;
- all non-self, non-positive elements of the current batch are negatives;
- no global queue or cross-fold memory is used;
- representations are explicitly L2-normalized;
- self-comparisons are masked with a finite minimum value;
- batches smaller than two windows fail explicitly;
- temperature must be finite and positive.

The result preserves:

```text
contrastive_loss
positive_similarity
negative_similarity
embedding_norm
```

The differentiable loss component stores:

```text
numerator   = summed cross entropy over 2 × batch rows
denominator = 2 × batch
```

Epoch reporting must sum numerators and denominators before division. Similarity
and norm diagnostics are denominator-weighted across batches. These values
remain diagnostic until a predeclared scientific experiment is completed.

## Fold-scoped data and leakage protection

`ContrastiveFoldData.from_indexed_source` requires explicit:

```text
training_indices
inner_validation_indices
outer_test_indices
target_final_evaluation_indices
sample_ids
record_group_ids
subject_ids
fold_id
```

Only `training_indices` are reachable through the DataLoader. The loader has no
label field and the contract contains no `self.data` lookup.

Before loader construction, the contract verifies:

- unique global `sample_id`;
- valid, unique indices inside each partition;
- zero training-index overlap with validation, outer test, and target final
  evaluation;
- zero `record_group_id` overlap between training and forbidden partitions;
- zero `subject_id` overlap between training and forbidden partitions;
- string-safe provenance identifiers;
- raw input shape and finite values for each authorized window.

The training provenance contains fold ID, authorized sample count, hashes of
the training indices and sample IDs, training subjects and logical records,
and excluded-partition counts. It can be embedded in the contrastive
checkpoint without exposing task labels.

If inner validation is later used to select temperature or augmentation
parameters, it must remain outside the pretraining loader and outside outer
test, as enforced by this contract.

## Checkpoint formats

### Full contrastive checkpoint

The full checkpoint stores:

```text
schema_version
encoder_state_dict
projection_head_state_dict
optimizer_state_dict, when supplied
encoder_architecture
configuration
latent_dim
projection_dim
projection_hidden_dim
augmentation_configuration
seed
epoch
training_provenance
```

The encoder state excludes the downstream output head. Loading validates the
encoder class, input metadata, latent width, encoder key names and tensor
shapes, then restores encoder and projection states strictly.

### Encoder-only checkpoint

The downstream export contains only:

```text
encoder_state_dict
encoder_architecture
latent_dim
metadata
```

It can be loaded into a compatible model with a different output width. The
current downstream head is retained byte-for-byte. Loading an EEGNet encoder
into ShallowConvNet, changing the latent width, or changing encoder tensor
shapes produces an explicit incompatibility error.

Projection-head parameters never enter an ordinary downstream checkpoint.

## Downstream contract

After encoder-only loading, the existing shared interface supports:

- five-class classification;
- seven-target PM regression;
- linear evaluation / `head_only`;
- `full_model` fine-tuning.

The CPU tests verify that a seven-output head can differ from the pretraining
head, a head-only optimizer step leaves all encoder parameters and buffers
unchanged in evaluation mode, and a full-model step produces non-zero encoder
gradients and changes encoder parameters.

No complete downstream experiment was run.

## Pytest discovery correction

Before this task, root:

```text
python -m pytest -q
```

recursed into ignored runtime directories under
`benchmark_results/pytest_tmp` and failed with 33 Windows
`PermissionError` collection errors. Explicit `python -m pytest -q tests`
did not have this problem.

The new `pytest.ini` sets:

```text
testpaths = tests
addopts = --basetemp=benchmark_results/.pytest_runtime
cache_dir = benchmark_results/.pytest_cache
norecursedirs = .git .pytest_cache benchmark_results data logs outputs pytest_tmp
```

This excludes runtime and data directories, not project source. The dedicated
ignored basetemp also avoids the separate Windows permission failure in the
shared system temporary directory; pytest cache metadata is kept in the same
ignored runtime tree. Root and explicit discovery both collect the same 744
pre-existing tests before the new contrastive suite is added.

## Tests

The new CPU suite contains 31 tests covering:

- all five augmentation shapes, input immutability and finite output;
- same-seed determinism and different-seed variability;
- neutral disabled transformations;
- two distinct finite views;
- model-derived projection input width and normalized projections;
- finite NT-Xent loss, aligned positives and excluded self-comparisons;
- explicit rejection of a one-window batch;
- EEGNet and ShallowConvNet integration;
- unchanged ordinary forward and state dict;
- authorized-only loader access;
- outer-test index and logical-record overlap rejection;
- string subject identifiers and fold provenance;
- an optimizer step updating encoder and projection head;
- full contrastive checkpoint restoration;
- encoder-only export into a different downstream head;
- incompatible architecture rejection;
- seven-output head, head-only, and full-model behavior;
- numerator/denominator aggregation;
- absence of label and `self.data` dependencies.

Observed results:

```text
new contrastive suite:                 31 passed
shared encoder suite:                  20 passed
DANN suite:                            21 passed
root `python -m pytest -q`:           775 passed, 11 warnings
explicit `python -m pytest -q tests`: 775 passed, 11 warnings
```

## Synthetic diagnostic smoke

The CPU smoke used only synthetic raw EEG:

```text
authorized indices              [0, 1, 2, 3]
validation/test/final indices   excluded
input                           [4, 1, 4, 128]
latent width                    4
projection width                8
contrastive optimizer step      finite
encoder changed                 true
projection head changed         true
full checkpoint saved           true
encoder-only checkpoint saved   true
encoder parameters transferred  true
downstream output               [4, 7]
head-only encoder unchanged     true
downstream head changed         true
status                          diagnostic
```

Temporary checkpoint files were written outside the repository test artifacts
and are not intended for Git.

## Model-specific limitations

### EEGNet

- Its flattened latent width depends on the input duration and pooling
  configuration, so an encoder checkpoint requires matching channel count,
  sample count, and architecture.
- The latent representation can be much wider than the base model's filter
  count; projection-head capacity needs inner-validation rather than arbitrary
  scaling.

### ShallowConvNet

- Its adaptive-pooled latent width equals the filter count and is relatively
  narrow in the current architecture.
- The global pooling that benefits the supervised baseline may discard
  temporal detail useful to a contrastive objective.
- The architecture was not enlarged solely for this infrastructure task.

## Remaining work for COG-BCI

Before a scientific external-data experiment:

1. download, checksum, extract, and inventory the actual COG-BCI files;
2. validate channel names/order, sampling rate, units, windowability, subject
   IDs, sessions, events, missingness, and license;
3. define compatible source/target raw preprocessing without pseudo-raw
   reshaping;
4. define outer-train-only pretraining folds and independent downstream test
   folds;
5. select augmentation policy, temperature, projection width, epoch and
   checkpoint using inner validation only;
6. integrate checkpoint and provenance artifacts through the existing
   experiment orchestration;
7. compare against the unchanged randomly initialized/source-supervised
   baselines;
8. run a one-fold real smoke before any complete GroupKFold experiment.

No scientific result is reported because no real pretraining, external loader,
downstream evaluation, fold comparison, or predeclared hyperparameter
selection was performed.
