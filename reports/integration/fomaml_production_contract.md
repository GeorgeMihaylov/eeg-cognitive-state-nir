# Production FOMAML buffer and validation contract

## Scope and status

- Branch: `integration/benchmark-unification`.
- Base HEAD: `851faf5` (`feat(meta): add synthetic FOMAML contract`).
- Task 8U status at entry: a deterministic synthetic CPU FOMAML contract
  existed, while production EEGNet and ShallowConvNet adaptation was blocked
  because `buffer_policy=frozen` had no safe BatchNorm semantics.
- Current decision: `production_contract_ready`.
- This is a read-only architecture/checkpoint audit plus synthetic one-step
  compatibility smoke. No real EEG tensor was loaded for adaptation, no EEG
  model was trained, and no policy was selected from outer-test results.

## Architecture and latent dimension audit

All dimensions are obtained from `model.encode(...)`, `model.latent_dim`, and
the registered output head. No latent width is hard-coded in the FOMAML path.
The canonical/checkpoint/explicit production rows agree within each model.

| model row | example input | encoder output | latent / head input | parameters | semantic architecture signature |
|---|---:|---:|---:|---:|---|
| EEGNet canonical | `[2,1,14,2560]` | `[2,1280]` | 1280 / 1280 | 8,501 | `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7` |
| EEGNet fold-01 checkpoint | `[2,1,14,2560]` | `[2,1280]` | 1280 / 1280 | 8,501 | `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7` |
| EEGNet explicit production smoke | `[2,1,14,2560]` | `[2,1280]` | 1280 / 1280 | 8,501 | `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7` |
| EEGNet task-8U mini audit | `[2,1,4,128]` | `[2,128]` | 128 / 128 | 503 | `4d2b1a634925ba59acfab99a0d22e889a4573690d0518c8d8e78936b1ba6dac0` |
| ShallowConvNet canonical | `[2,1,14,2560]` | `[2,40]` | 40 / 40 | 1,925 | `ee9eb659f0ad02214e19bd5879fbc6436f578a39689d50adf01b794800afdd58` |
| ShallowConvNet fold-01 checkpoint | `[2,1,14,2560]` | `[2,40]` | 40 / 40 | 1,925 | `ee9eb659f0ad02214e19bd5879fbc6436f578a39689d50adf01b794800afdd58` |
| ShallowConvNet explicit production smoke | `[2,1,14,2560]` | `[2,40]` | 40 / 40 | 1,925 | `ee9eb659f0ad02214e19bd5879fbc6436f578a39689d50adf01b794800afdd58` |
| ShallowConvNet task-8U mini audit | `[2,1,4,128]` | `[2,4]` | 4 / 4 | 83 | `65def0ba538931509a31d91573bae31970417c715664da8d2e27cf239b14ee98` |

Production EEGNet uses 256 Hz, 10 seconds, temporal/separable kernels of
128/32 samples, F1=8, depth multiplier=2, F2=16, and pooling 4 then 8. Its
flattened time-dependent representation is therefore 1280 rather than the
task-8U mini model's 128. Production ShallowConvNet uses 40 filters, a
25-sample temporal kernel, pooling 75/15, and adaptive pooling to `(1,1)`;
its latent width is consequently 40, while the four-filter mini model has
width 4. The earlier discrepancy is fully explained by different architecture
signatures, not by a model or encoder bug.

Checkpoint schema signatures are
`541b9c4ecd399493da51b01f9670ce7daaa7415320925ca77652a4f5cb67015c`
for EEGNet and
`404f0a729ef5a9fe27eaeebaaf693b7454a908a8bf558bfcc99efee493b479f1`
for ShallowConvNet. Strict state loading succeeds without modifying either
checkpoint.

## BatchNorm and buffer inventory

| model | module | features | buffers |
|---|---|---:|---|
| EEGNet | `features.2` | 8 | `running_mean`, `running_var`, `num_batches_tracked` |
| EEGNet | `features.4` | 16 | `running_mean`, `running_var`, `num_batches_tracked` |
| EEGNet | `features.11` | 16 | `running_mean`, `running_var`, `num_batches_tracked` |
| ShallowConvNet | `features.0` | 40 | `running_mean`, `running_var`, `num_batches_tracked` |

The functional EEGNet state has 12 named parameter tensors and 9 named
buffers; ShallowConvNet has 8 and 3. Names, order, shapes, dtype, and device
are validated against the production module. Parameters and buffers are
cloned into independent storage before every episode. Missing, extra, or
shape-mismatched entries fail explicitly.

## Explicit buffer policies

`frozen_global` clones the global running statistics, runs support and query
with BatchNorm in evaluation mode, and permits adaptation only of ordinary
fast parameters, including BatchNorm affine parameters. Running statistics
never change. This is deterministic and robust for small support sets, but
global statistics may not fit a new participant.

`support_local` creates independent episode buffers, enables only BatchNorm
modules during support forward passes, freezes the resulting local statistics
before query, and destroys the episode state afterwards. The base model is
never updated. It can adapt statistics to the participant without query use,
but is more variable and requires at least two support samples per batch.

Dropout is explicitly disabled for all functional support and query forwards
under both policies. No mode jointly updates BatchNorm from support and query.

## Functional and leakage audits

For EEGNet and ShallowConvNet, for both policies, the CPU smoke used synthetic
`[4,1,14,2560]` tensors and performed one support forward, one first-order
inner gradient step, one query forward, and one query gradient. Outputs had
five classes and all losses/gradients were finite and nonzero.

Two different query tensors produced identical pre-query fast weights and
identical support-derived buffers; only query loss and query gradients could
differ. Query forward did not change any buffer hash. Changing support changed
local buffers under `support_local`, but not the base buffers. Parameter and
buffer hashes of both original production models remained unchanged in every
audit.

## Nested meta-validation protocol

The manifest reuses outer fold 1 from the existing five-fold subject-level
GroupKFold prediction artifact. Its file hash is checked before and after
materialization. No outer split is regenerated or modified.

| partition | subjects | prepared episodes |
|---|---:|---:|
| outer train | 43 | — |
| meta-train (only from outer train) | 34 | 23 |
| meta-validation (only from outer train) | 9 | 9 |
| protected outer-test | 11 | 8 |

Meta-validation selection is a deterministic seed-42 hash ordering of eligible
outer-train subjects. Support budget is 32 windows and query budget is 64.
The existing episode builder uses session, then record, then chronology;
support/query sample and record IDs are disjoint. Subject intersections are
zero for outer train/test and meta-train/meta-validation, and outer-test never
enters policy, hyperparameter, checkpoint, or early-stopping selection.

Forty safe episodes were materialized. Fourteen subject episodes were skipped
rather than weakening the protocol: thirteen participants had fewer than two
records, and one support record contained only three windows. These errors are
preserved in `errors.csv`; there is no silent within-record fallback.
Protocol hash:
`a3e6ff5ee2dbfa1638ffee9180ddff582dbab8aa6186e164320dd92f082871e8`.

## Disabled future experiment

`experiments/meta_learning/fomaml_production_contract.json` preregisters
EEGNet, `label_q5`, outer fold 1, both buffer policies, one inner step,
learning rates, support/query budgets, checkpoint criterion, metrics, and a
decision rule based only on meta-validation. `execution_enabled` is `false`.
Both policies remain preregistered candidates; this task does not choose a
winner.

Runtime artifacts are under
`benchmark_results/meta_learning_fomaml_production_contract/` and include the
architecture, parameter/buffer, BatchNorm, policy, compatibility, functional
state, query leakage, protocol, episode, future-config, decision, error, and
contract-report files. They contain no absolute local paths and remain ignored
runtime output.

## Verification, limitations, and launch conditions

- Targeted FOMAML tests: 21 passed, 1 pre-existing pytest-config warning.
- Full `python -m pytest -q tests`: 1059 passed, 13 warnings.
- Full root `python -m pytest -q`: 1059 passed, 13 warnings.
- Result status: `production_contract_ready`.

This status establishes infrastructure safety only; it makes no claim about
FOMAML quality on EEG. A real diagnostic run still requires explicit approval,
the disabled execution guard to be deliberately changed, preregistered use of
both policies or a meta-validation-only selection rule, train-only checkpoint
creation, and a final one-use outer-test evaluation. Real data, GPU training,
new folds/seeds, policy search on outer-test, and production-architecture
changes are outside this audit.
