# Leakage-safe episodic meta-learning infrastructure

## Status and scope

- Branch audited: `integration/benchmark-unification`.
- Base HEAD: `fd4b1f1b354fcb3c070c725cf14652f90f5e1e3e`.
- Result status: `diagnostic`.
- Episodic infrastructure: implemented.
- Meta-learning algorithm: not implemented.
- Meta-learning experiment or model training: not performed.

This change adds only the metadata, partition, validation, cloning, and
artifact contracts needed before an algorithmic meta-learning experiment can
be approved. It does not add a new splitter, model factory, optimizer,
checkpoint format, metric implementation, or training loop.

## Historical prototype

The design input was
`feature/benchmarking@8ecbee9:bench/tasks/mixin/metalearning.py`, inspected
with `git show` together with the historical transfer-learning, DANN, and
contrastive mixins.

Preserved ideas:

- an episode is divided into support and query subsets;
- an episode can represent adaptation to one participant;
- adaptation must operate on a model clone;
- future algorithms need a repeated episode-level interface.

Reworked ideas:

- episodes receive explicit allowed and forbidden outer-partition IDs;
- string participant IDs are first-class values;
- sample, record, session, and subject boundaries are audited;
- IDs use canonical serialization and SHA-256 rather than process-local
  hashing;
- manifests store identities and targets, never EEG or feature arrays;
- session and record boundaries take precedence over within-record slicing;
- every skipped episode has an explicit policy and error record.

Rejected prototype components:

- the optional `learn2learn` dependency and its silent dummy fallback;
- direct access to global `self.data`;
- implicit random selection of neighboring windows;
- hard-coded ways, shots, query count, task count, and optimizer settings;
- the embedded MAML/First-Order MAML training loop;
- direct mixin inheritance and model replacement through shared mutable state.

The prototype could coerce subject identifiers into tensors, see the entire
dataset before an outer split, and create support/query samples without
record or temporal provenance. Those paths are not connected.

## Episode structures

`MetaEpisodeSpec` records the episode type, task type, target, support/query
units and sizes, class policy, chronology, group fields, seed, minimum data,
and explicit insufficiency policy. Supported task contracts distinguish
classification, ordinal classification, scalar regression, and multi-output
regression.

`MetaEpisode` contains dataset/task/fold/entity identity, one subject,
session and record identities, support/query sample IDs and targets, split
level, seed, specification hash, and deterministic episode ID. It contains no
EEG or feature arrays. `MetaEpisodeManifest`, `MetaEpisodeIndex`, and
`MetaEpisodeError` provide stable serialization and explicit failures.

The primary materialized mode is `subject_personalization`: limited support
data and independent later query data from the same participant. The split
preference is session, then record, then an explicitly enabled chronological
within-record boundary. `session_adaptation` is materialized with query-session
rotations. `few_shot_classification` and `cross_task_adaptation` are typed
contracts only; cross-task adaptation is not claimed to be scientifically
valid without a shared target.

## Safety invariants

Every build receives allowed and forbidden sample IDs. Meta-training entity
indices use outer-train participants and mark outer-test participants as
forbidden. External-subject personalization is built in a separate
outer-test evaluation scope with outer-train samples forbidden.

The validator enforces:

- `support ∩ query = ∅`;
- no duplicate support or query sample IDs;
- support/query cannot contain a forbidden sample or subject;
- one subject per personalization episode;
- disjoint support/query records in strict mode;
- a recorded and verified time boundary when within-record splitting is
  explicitly enabled.

Query IDs are reserved for future evaluation. This diagnostic does not
consume them for training, early stopping, episode-size selection, algorithm
selection, or hyperparameter selection.

Insufficient-data policies are `error`, `skip_episode`, `reduce_support`, and
`reduce_query`. Sampling with replacement, duplication, borrowing from
another participant, and borrowing from the forbidden partition are not
fallbacks.

Classification policies are `none`, `require_all_classes`,
`equal_per_class`, and `at_least_one_per_class`. Equalization only removes
examples and never oversamples. Regression rejects every non-`none` class
policy.

## Dataset and model contracts

`EpisodeDatasetView` retains a reference to the original array or DataFrame
and stores only support/query positions plus small metadata subsets. It
preserves `sample_id`, `subject_id`, `session_id`, and `record_id`, does not
mutate the source, and does not fit a scaler or apply transformations.

The clone helper uses `copy.deepcopy` only after verification. Its contract
requires identical module type and state, distinct parameter storage,
preserved output shape, explicit device placement, and an unchanged original
model. CPU tests cover the production EEGNet and ShallowConvNet modules. The
existing fitted `TorchClassificationAdapter.clone()` remains the integration
point for future personalization; no second adapter was introduced.

`MetaLearnerProtocol` reserves only `meta_train_step`, `adapt`, and
`evaluate`. A future implementation must use the current model factory,
encoder/head API, Torch adapter, checkpoint contract, and metrics.

## Real-data diagnostic smoke

The smoke reused outer fold 1 from the existing canonical Random Forest
`label_q5` prediction artifact. No outer split was recomputed.

- Main dataset episodes: 6.
- Main outer-train entity indices: 3 subjects, record-disjoint.
- Main outer-test personalization episodes: 3 subjects, record-disjoint.
- Main support samples: 192.
- Main query samples: 384.

The COG-BCI smoke reused fold 1 of the existing N-Back protocol and its
target index.

- COG-BCI subjects: 3.
- Query-session rotations per subject: 3.
- COG-BCI episodes: 9.
- COG-BCI support samples: 3,608.
- COG-BCI query samples: 1,804.

The three COG rotations are session-disjoint sensitivity views; rotations
with session 1 or 2 as query do not claim that support chronologically
precedes query. Chronological support-before-query remains mandatory for the
primary `subject_personalization` episodes.

Across all 15 episodes, sample overlap was zero, record overlap was zero,
forbidden-sample overlap was zero, forbidden-subject overlap was zero, and
all episode audits passed. SHA-256 hashes of the three consumed split
artifacts were identical before and after materialization.

## Runtime artifacts

Ignored runtime output is written below
`benchmark_results/meta_learning_infrastructure/`:

- `episode_spec.json`;
- `episode_index.parquet`;
- `episode_manifest.json`;
- `episode_summary.json`;
- `episode_balance.csv`;
- `episode_leakage_audit.json`;
- `prototype_mapping.json`;
- `errors.csv`;
- `infrastructure_report.md`.

Files contain repository-relative provenance only. The generated manifest
contains no EEG arrays, feature arrays, model weights, or local absolute
paths.

## Limitations and algorithm readiness

No gradient adaptation, meta-optimizer, learned initialization, episodic
sampler for batch training, algorithm checkpoint, or scientific comparison
exists. The current query semantics are evaluation-only; an approved
algorithm must state whether a distinct inner meta-validation partition is
needed without reusing outer query data.

First-Order MAML still needs a differentiable adapter path, explicit
support-loss adaptation, meta-train/meta-validation separation, optimizer
state artifacts, and proof that outer query labels never influence
selection. Reptile needs deterministic episode batching, parameter-delta
aggregation, clone/reset audits, and checkpoint/resume tests. ProtoNet needs
a scientifically justified shared embedding, class-complete support/query
episodes, prototype computation without leakage, and ordinal/regression
scope decisions.

The recommended next algorithmic smoke is a CPU-only, synthetic
First-Order MAML contract test after a separate protocol review. It should
not run on real EEG until nested selection and query semantics are approved.
