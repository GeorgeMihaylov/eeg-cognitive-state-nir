# Fold-scoped DANN infrastructure

## Status and scope

- Branch: `integration/benchmark-unification`
- Audited HEAD: `dda2293`
- Result status: `diagnostic`
- This change extends the uncommitted Task 8A shared encoder interface.
- No dataset loader, model registry, `BenchmarkRunner`, experiment config, or
  scientific training run was added.
- No COG-BCI data was loaded.

## Historical prototype audit

The historical implementation was inspected read-only from commit `8ecbee9`
with `git show 8ecbee9:bench/tasks/mixin/domain_adaptation.py`.

The reusable ideas were:

- a gradient-reversal autograd operation;
- a domain discriminator attached to latent features;
- a joint task/domain objective.

The following parts were deliberately rejected:

- hardcoded `latent_dim = 128`;
- discovery of all domains from `self.data.subject_ids`;
- training over the full `self.data`;
- a private DANN adapter and training loop;
- an assumption that `backbone(x)` returns latent features;
- unconditional CUDA selection;
- synthetic target label `-1` stored beside task labels;
- batch-mean epoch accumulation without numerator/denominator accounting.

## Architecture

The current flow is:

```text
source X ─┐
          ├─ shared raw-EEG encoder ─ latent ─ task head ─ source task output
target X ─┘                         │
                                   └─ gradient reversal ─ domain head
                                                              │
                                                   source+target domain output
```

`DANNModule` receives an existing encoder-compatible task model. It obtains
the domain input width from `model.latent_dim`; it does not add dataset, fold,
or domain knowledge to EEGNet or ShallowConvNet. The task model remains
directly callable through its unchanged `model(X)` surface.

The domain classifier is a separate configurable MLP:

```text
Linear(latent_dim, hidden_dims[0])
→ ReLU
→ Dropout
→ optional additional hidden layers
→ Linear(last_hidden, n_domains)
```

`n_domains`, hidden dimensions, dropout, and gradient-reversal coefficient are
construction parameters. CPU is supported and tested; CUDA remains optional
through ordinary PyTorch device movement.

## Supported encoders and latent format

Primary support is limited to raw-EEG EEGNet and ShallowConvNet:

| Model | Input | Latent representation | Current full-config width |
|---|---|---|---:|
| EEGNet | `[batch, 1, channels, time]` | flattened convolution output | 1280 |
| ShallowConvNet | `[batch, 1, channels, time]` | adaptive-pooled filters | 40 |

The displayed widths are model-derived for `[batch, 1, 14, 2560]` and the
current seed-42 raw-deduplicated configs. They are examples, not constants in
DANN. The feature Transformer is intentionally excluded.

ShallowConvNet exposes only `n_filters` latent values after global adaptive
pooling. This narrow representation is an experimental limitation for domain
discrimination; its architecture was not enlarged merely to accommodate DANN.

## Objective and aggregation

The objective keeps two independent coefficients:

```text
gradient_reversal_alpha
    controls sign and magnitude of the domain gradient entering the encoder

lambda_domain
    controls the contribution of domain loss to the optimized objective

total_loss = task_loss + lambda_domain * domain_loss
```

The result preserves:

```text
task_loss
domain_loss
total_loss
domain_accuracy
```

Task and domain loss components retain differentiable numerators and
denominators. Cross-batch reporting sums numerators and denominators before
division. Classification task loss uses `sum(cross_entropy) / n_source`;
domain loss uses `sum(cross_entropy) / n_source_plus_target`. A minimal
regression objective uses summed squared error divided by the number of output
elements, but no regression DANN experiment is claimed here.

## Fold-scoped data contract

`DANNFoldData` requires four explicit partitions:

```text
source_train
target_unlabelled_or_calibration
inner_validation
outer_test
```

Each `DANNPartition` carries only explicitly supplied arrays plus:

```text
domain_ids
sample_ids
record_group_ids
subject_ids
optional task_labels
```

String subject identifiers are preserved. The contract has no `self.data` and
does not discover a full dataset.

The training loader is built only from `source_train` and
`target_unlabelled_or_calibration`. Its batch contains source task labels and
source/target domain IDs, but deliberately has no target task-label field.
Target calibration labels may exist in the partition for another authorized
stage; they are not returned by this loader. Inner-validation and outer-test
objects are not reachable through the training dataset.

Before a loader can be created, the contract rejects:

- source-train/outer-test `sample_id` overlap;
- source-train/outer-test `record_group_id` overlap;
- target-training/outer-test `sample_id` overlap;
- inner-validation `sample_id` overlap with train or outer test;
- duplicate sample IDs inside a partition;
- inconsistent lengths, shapes, non-integer domain IDs, and non-finite input.

Feature normalization is not fitted by this helper. A future experiment
orchestrator must fit the existing scaler only on its authorized training
partition and pass the transformed arrays into this fold contract. Outer test
must not select the scaler, `alpha`, `lambda_domain`, epoch, or checkpoint.

## Checkpoints

An ordinary task-model checkpoint is unchanged and contains no domain-head
keys. A DANN checkpoint stores separate mappings:

```text
task_model_state_dict
domain_discriminator_state_dict
dann:
  latent_dim
  n_domains
  gradient_reversal_alpha
  domain_hidden_dims
  domain_dropout
metadata
```

Loading validates schema, latent width, and domain count, then restores both
state dicts strictly. This avoids silently treating a DANN checkpoint as an
ordinary EEGNet/ShallowConvNet checkpoint.

## Tests

The new CPU suite contains 21 tests covering:

- gradient-reversal forward identity, sign, and scale;
- dynamic domain input width;
- EEGNet and ShallowConvNet integration;
- unchanged ordinary forward and ordinary state-dict surface;
- identical source latent values sent to task and domain paths;
- separate DANN checkpoint round-trip;
- exclusion of target task labels, inner validation, and outer test from the
  training loader;
- string subject IDs;
- finite CPU optimizer step that updates encoder, task head, and domain head;
- `lambda_domain = 0` equivalence to task-only gradients;
- `gradient_reversal_alpha = 0` preserving task-model gradients;
- explicit dimension errors;
- numerator/denominator aggregation;
- sample and logical-record overlap rejection.

Results at the time of this report:

```text
config + encoder + DANN + personalization regression set: 182 passed
new DANN suite:                                       21 passed
full suite before DANN implementation:               723 passed, 11 warnings
full suite after DANN implementation:                 744 passed, 11 warnings
```

## Synthetic diagnostic smoke

One CPU optimizer step used:

```text
source input                 [6, 1, 4, 128]
source task labels           supplied
target input                 [6, 1, 4, 128]
target task labels           not supplied
latent width                 4
outer-test samples in batch  false
encoder changed              true
task head changed            true
domain head changed          true
all reported values finite   true
status                       diagnostic
```

The synthetic loss and domain accuracy are implementation diagnostics only and
are not scientific results.

## Remaining work before a scientific DANN experiment

A real experiment still requires:

1. a scientifically justified source/target domain definition;
2. a validated COG-BCI loader and channel/sampling/label compatibility audit;
3. fold-scoped orchestration using only outer-train source data and authorized
   target calibration/unlabelled data;
4. train-only preprocessing and normalization;
5. an inner-validation policy for `alpha`, `lambda_domain`, epoch selection,
   and early stopping;
6. checkpoint, split, prediction, and provenance artifacts integrated into the
   existing experiment layer;
7. comparison against unchanged source-only and target-calibration baselines;
8. a synthetic runner integration test before any real multi-fold run.

Consequently, this task establishes reusable infrastructure only. It does not
show that DANN improves cognitive-state prediction or transfers to COG-BCI.
