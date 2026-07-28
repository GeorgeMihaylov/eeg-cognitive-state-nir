# Shared encoder interface for raw-EEG models

## Status

- Branch: `integration/benchmark-unification`
- Audited HEAD: `dda2293`
- Result status: `diagnostic`
- Initial working tree: clean; staging area empty.
- No dataset, training experiment, registry, benchmark runner, or
  personalization pipeline was added.

## Prototype audit

The four historical prototypes were absent from the current tree and were
inspected read-only from commit `8ecbee9` using `git show`.

### Transfer learning

`TransferMixin` fitted each model on an optional pretraining dataset and saved
either `get_weights()` or the wrapped Torch `state_dict`. Its reusable idea is
checkpoint capture followed by explicit restoration. The prototype did not
define an encoder/head boundary. Its historical calibration path was not
restored because the later audit showed that it rebuilt the model and lost the
pretrained state. Current leakage-safe `zero_shot`, `head_only`, and
`full_model` orchestration remains the only personalization path.

### Domain adaptation / DANN

The prototype contained a reusable gradient-reversal operation and a domain
discriminator. The wrapper assumed that `backbone(x)` returned latent features,
then applied the classifier again even though current models return task
outputs. It hardcoded discriminator input width `128`, inferred domains from
all `self.data.subject_ids`, and introduced a separate source/target training
loop. These parts must be replaced by an explicit domain contract, fold-scoped
loaders, model-derived `latent_dim`, and the shared adapter loop before DANN is
scientifically runnable.

### Meta-learning

The prototype exposed useful episodic concepts (`n_ways`, `n_shots`,
`n_queries`, support/query adaptation), but constructed episodes from all
`self.data`, used a separate MAML loop, depended optionally on `learn2learn`,
and attempted to convert subject identifiers directly to tensors. It did not
enforce outer-fold isolation. Future episodes must be constructed by experiment
orchestration from the allowed outer-train subjects only.

### Contrastive learning

The prototype supplied useful raw-EEG augmentation, projection-head, and
contrastive-loss ideas. It nevertheless created a separate encoder with
hardcoded latent width `64`, projection width `128`, automatically consumed
all `self.data`, converted two-dimensional features into pseudo-raw inputs, and
used an independent training loop. Future contrastive pretraining must operate
on real raw windows from the current fold and attach a projection head to the
shared model `encode` output.

## Final contract

The minimal model contract is:

```python
latent = model.encode(X)                # [batch, latent_dim]
outputs = model.forward_head(latent)    # [batch, num_outputs]
outputs = model(X)                      # identical composition
```

The shared utility layer also provides:

```text
latent_dim
get_output_head()
replace_output_head(num_outputs)
freeze_encoder()
unfreeze_encoder()
output_head_parameter_prefixes()
```

`latent_dim` is calculated by each model during construction. No experiment,
mixin, dataset, or domain discriminator supplies it.

The adapter exposes the same capabilities through `get_encoder`, `encode`,
`get_output_head`, `replace_output_head`, `freeze_encoder`, and
`unfreeze_encoder`. Adapter feature extraction accepts only explicitly passed
features, applies already-fitted train-only preprocessing, performs no fitting,
uses no labels, batches on the configured CPU/CUDA device, and returns detached
features on CPU.

## Supported models and shapes

| Model | Input | Latent output | Current head |
|---|---|---|---|
| EEGNet | `[batch, 1, channels, time]` | `[batch, model.latent_dim]` | `Linear(latent_dim, num_outputs)` |
| ShallowConvNet | `[batch, 1, channels, time]` | `[batch, n_filters]` | `Linear(n_filters, num_outputs)` |

For the current ShallowConvNet configuration, `latent_dim` is the configured
number of temporal/spatial filters. For EEGNet it is the model-computed
flattened convolutional width after both pooling stages.

The feature-sequence Transformer already has an `encode` method, but its input
contract is `[batch, sequence, aggregated EEG+POW features]`. It was not forced
into the raw-EEG transfer contract because aggregated features must not be
reshaped or described as external raw EEG. A future feature-sequence transfer
question may extend it separately after defining a compatible source
representation.

Sklearn models are intentionally outside this neural encoder API.

## Output-head replacement and PM compatibility

A proxy classification head can be replaced by a new linear head without
changing encoder parameters. The shared adapter updates its objective handler
when changing between classification and regression. A seven-output regression
head therefore supports the canonical PM target order:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

This task verifies the architecture and synthetic adapter path only. It does
not train a PM model or claim transfer quality.

## Checkpoint compatibility

The implementation keeps existing registered module names and state-dict keys:

```text
EEGNet: features.* and classifier.*
ShallowConvNet: temporal.*, spatial.*, features.*, classifier.*
```

The mixin registers no parameters or buffers, and `_latent_dim` is plain
metadata. Existing checkpoints with an unchanged output head therefore remain
strict-load compatible. New checkpoint metadata adds `latent_dim` and
`encoder_api_version`, while loading older metadata remains supported.

After an intentional head replacement, a target-specific checkpoint must be
loaded into a factory-built model whose head and adapter task contract have
been replaced identically. This migration path is covered by the seven-output
adapter round-trip test.

## Tests

CPU tests verify:

- unchanged forward shapes for EEGNet and ShallowConvNet;
- `[batch, latent_dim]` encoder shapes;
- model-derived latent width;
- exact `forward_head(encode(X)) == forward(X)` in evaluation mode;
- three-class and seven-output head replacement;
- unchanged encoder parameters during replacement;
- encoder freeze/unfreeze flags;
- a head-only optimizer step changes the head but not encoder parameters;
- CPU operation and optional CUDA device preservation;
- adapter-side explicit feature extraction;
- seven-output regression fit, prediction, save, and restore;
- the existing classification adapter `fit`, `predict`, and `predict_proba`;
- strict state-dict loading with legacy key names;
- absence of dataset, fold, subject, or label arguments in the encoder API;
- explicit rejection of the feature Transformer as a raw-transfer encoder.

Targeted result:

```text
60 passed, 1 warning
```

The warning is the existing sklearn class-coverage warning from the small
EEGNet runner smoke fixture.

Full suite result:

```text
722 passed, 1 failed, 11 warnings
```

The single failure is the pre-existing
`test_config_curation.py::test_23_all_31_unclassified_configs_receive_review_status`.
It expects exactly 29 configs without registry/report links, while the current
committed registry/report state yields 0. The failure was reproduced in
isolation and is unrelated to the encoder changes; config curation was not
modified in this task.

## Synthetic diagnostic smoke

The smoke used a random CPU raw-EEG batch and no external dataset:

```text
input                  [6, 1, 4, 128]
latent                 [6, 4]
original output        [6, 5]
replacement output     [6, 7]
encoder unchanged      true
head changed           true
loss finite            true
status                 diagnostic
```

The synthetic loss value is deliberately not reported as a scientific metric.

## Leakage and API boundaries

`encode` receives only its explicit tensor/array argument. It does not know
about a dataset, fold, target subject, record, sample index, or label. The
adapter does not discover or materialize an entire dataset automatically.
Future DANN, contrastive, and meta-learning orchestration remains responsible
for passing only fold-allowed data.

## Readiness and limitations

### DANN

Ready: stable raw-EEG latent features, model-derived width, separable task head,
and encoder freezing. Still required: a justified source/target definition,
fold-scoped paired loaders, domain labels, a shared multi-objective adapter
extension, and leakage/artifact tests. No DANN training was implemented.

### Contrastive learning

Ready: a reusable real-raw-EEG encoder output to which a projection head can be
attached. Still required: approved augmentations, fold-scoped pretraining data,
checkpoint provenance, and downstream comparison. The historical independent
encoder and automatic `self.data` access were not reused.

### Meta-learning

Ready: explicit encoder/head separation and head replacement. Still required:
string-safe subject episode construction, support/query isolation inside outer
train, optional-dependency policy, and integration with the shared adapter.

### General limitations

- Encoder checkpoint-only export/import is not yet a dedicated artifact
  operation; this task preserves full checkpoint compatibility.
- Transformer support is intentionally deferred for raw cross-source work.
- No external loader, COG-BCI data, PM training, DANN, contrastive training, or
  meta-learning was run.

## Recommended next step

Implement a small checkpoint-transfer utility around the shared adapter:
load a validated raw-EEG source checkpoint, verify input/channel metadata,
replace the task head, save an encoder/checkpoint audit, and run only a
synthetic or approved native-dataset smoke. Define source/target domains and
fold-scoped loaders before adding a DANN objective.
