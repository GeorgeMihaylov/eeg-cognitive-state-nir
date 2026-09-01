# Leakage-safe raw-domain DANN protocol for `label_q5`

## Scope and provenance

- Branch: `integration/benchmark-unification`.
- Audited HEAD: `34072c7232de8f2535139c5e706b7ac8e6b8f1f4`.
- Protocol ID: `dann_label_q5_raw_deduplicated_source_transfer_v1`.
- Result status: `diagnostic`; readiness: `dann_protocol_ready`.
- Execution is disabled. No optimizer, backward pass, model training, target-test
  inference, predictions, metrics, checkpoint, CUDA tensor, new split, or cache
  build was produced.

The future scientific question is whether unlabeled target-domain raw EEG can
improve `label_q5` transfer over source-only EEGNet on new target-domain
participants under an otherwise identical split and training budget.

## Historical prototype and current replacements

The historical reference is
`8ecbee9:bench/tasks/mixin/domain_adaptation.py` from `feature/benchmarking`.
It was audited with `git show`; it was neither cherry-picked nor connected.

| Historical component | Current replacement | Preserved idea | Rejected implementation | Reason |
|---|---|---|---|---|
| Gradient-reversal autograd function | `model_zoo.DL.dann.GradientReversal` | reverse domain gradients with explicit alpha | mixin-local implementation | one shared, tested component is already available |
| Fixed domain discriminator | `DomainDiscriminator(latent_dim, 2, ...)` | separate domain head | hard-coded latent width 128 | production EEGNet exposes `latent_dim=1280` through the encoder API |
| Inline source/task and domain losses | `DANNObjective` | source task loss plus source/target domain loss | experiment-local loss code | shared objective enforces shapes and safe aggregation |
| Source/target loaders stopped at the shorter loader | `DANNFoldData` deterministic cycling | paired source and target batches | `min(len(source), len(target))` epoch | the smaller loader must cycle under an explicit epoch policy |
| `self.data` and subject IDs used as implicit domains | explicit metadata manifests and `source` mapping | domain-conditioned training | global data access and participant-as-domain labels | domains are only `gpn_data=0` and `Old_EEG=1` |
| Mixin-local optimizer and training loop | future integration through the shared adapter/runner | adversarial encoder optimization | a second production training loop | benchmark artifacts, validation and checkpoints must stay unified |

The current protocol reuses the production encoder API, EEGNet, DANN module,
objective, label-free target batch contract, raw-deduplicated cache metadata,
existing outer fold 1, and existing checkpoint schema. No production DANN
component required modification.

## Domain definition and canonical raw universe

The only domain label is the confirmed source field:

```text
domain 0 = gpn_data
domain 1 = Old_EEG
```

Participant, session, record, device assumptions, and arbitrary clusters are
not domain labels. The two sources are not described as different devices.

The reused canonical universe contains 30,958 unique samples, 54 subjects,
86 retained records / logical records, five non-null `label_q5` classes, and
raw windows shaped `[14, 2560]` at 256 Hz (10 seconds). Its immutable identity
is:

```text
308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf
```

This stage matched the canonical sample-ID and metadata hashes, verified that
all referenced cache shards exist, and read zero tensor values. In particular,
no target-test EEG value was opened. Tensor shape, dtype, offsets and finite
samples rely on the existing canonical raw-universe audit referenced by the
new manifest.

## Domain inventory

| Domain | Partition | Windows | Subjects | Records | Class counts 0/1/2/3/4 |
|---|---:|---:|---:|---:|---:|
| `Old_EEG` | outer train | 5,209 | 10 | 11 | 1279 / 1075 / 975 / 1036 / 844 |
| `Old_EEG` | outer test | 1,958 | 3 | 4 | 462 / 294 / 295 / 322 / 585 |
| `gpn_data` | outer train | 18,818 | 34 | 55 | 3542 / 3848 / 3867 / 3808 / 3753 |
| `gpn_data` | outer test | 4,973 | 8 | 16 | 1256 / 1068 / 1006 / 868 / 775 |

There is one participant represented by retained records in both domains:
`a02151ac`. The logical-record map contains 33 recordings originally present
in both sources.

## Logical-record deduplication audit

The retained-source rule remains the canonical deterministic ranking:
accepted fraction descending, available EEG samples descending, missing
fraction ascending, source priority, then lexical record ID. The full audit
records original sources, selected source/record, retained and discarded
accepted-window counts, selection reason, and signal relationship.

After selection there are zero duplicate sample IDs, zero duplicate logical
window keys, and zero logical records represented by both domains. Thus a
logical window cannot enter source and target simultaneously. The reused
outer-fold artifact also has zero train/test subject overlap.

## Outer split and adaptation directions

Outer fold 1 is reused exactly: 43 outer-train and 11 outer-test subjects.
Its source artifact SHA-256 is:

```text
41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341
```

Both directions and both overlap policies were audited independently:

| Direction | Policy | Source train subjects/windows | Target train subjects/windows | Target test subjects/windows | Shared train subjects | Feasible |
|---|---|---:|---:|---:|---:|---:|
| `gpn_data -> Old_EEG` | allow cross-domain | 34 / 18,818 | 10 / 5,209 | 3 / 1,958 | 1 | no: target test < 5 |
| `gpn_data -> Old_EEG` | strict disjoint | 34 / 18,818 | 9 / 4,754 | 3 / 1,958 | 0 | no: target test < 5 |
| `Old_EEG -> gpn_data` | allow cross-domain | 10 / 5,209 | 34 / 18,818 | 8 / 4,973 | 1 | count-feasible but not preferred |
| `Old_EEG -> gpn_data` | strict disjoint | 10 / 5,209 | 33 / 18,555 | 8 / 4,973 | 0 | yes |

For strict mode, a cross-domain outer-train participant is retained in the
source loader and excluded from the target loader. The rule depends only on
source membership, not on target labels.

The preregistered selection first requires at least five target-test subjects,
then maximizes target-test subjects, target-train subjects, source-train
subjects, and finally uses lexical source-domain order. Therefore the primary
direction is `Old_EEG -> gpn_data` with
`strict_cross_domain_subject_disjoint`. The reverse direction remains an
unexecuted secondary protocol; it is not treated as equivalent.

## Source validation and protected target partitions

For the primary candidate, the 10 source outer-train participants are split by
stable seed-42 subject hashing:

| Partition | Subjects | Windows | Records |
|---|---:|---:|---:|
| source task train | 7 | 3,753 | 8 |
| source validation | 3 | 1,456 | 3 |
| unlabeled target train | 33 | 18,555 | 54 |
| protected target test | 8 | 4,973 | 16 |

All sample, logical-record and required subject overlaps are zero. Source
validation alone may choose a future source-only or DANN checkpoint and early
stopping epoch, using source validation macro F1. Balanced accuracy, domain
accuracy, domain loss and task loss are monitoring values only.

The target training manifest exposes sample provenance and domain ID but no
`label_q5`, target task logits, target task loss, or classification metric.
The existing `DANNTrainingBatch` has no target-task-label field and
`DANNObjective` accepts task labels only for source outputs. Target test is a
metadata-only reference and cannot select direction, checkpoint, alpha,
lambda, batching, or stopping epoch.

## Batching, objective and architecture

The future fixed batching contract uses source batch size 32, target batch
size 32, no `drop_last`, no class weighting, domain weight 1.0, and
deterministic cycling of the smaller loader. Steps per epoch are the maximum
of the source and target batch counts (580 for the primary manifests), so the
target loader length does not silently define an epoch count.

The fixed schedules are:

```text
alpha(p) = 2 / (1 + exp(-10 p)) - 1
p = global_step / max(total_steps - 1, 1)
lambda_domain = 1.0 (constant)
```

The objective is source classification cross-entropy plus weighted domain
cross-entropy over both domains. Target task logits and target task labels do
not enter it.

A CPU-only `torch.no_grad()` forward used the production EEGNet signature
`248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7`.
It confirmed latent width 1280, source task shape `[2, 5]`, domain shape
`[5, 2]`, separate task/domain parameters, unchanged encoder state, finite
losses, and no gradients. EEGNet has 8,501 parameters; the configured domain
head has 172,354 (180,855 combined). No optimizer or backward call occurred.

## Preregistration and future decision rule

```text
protocol hash:
7f5642109e1ed26dd6de96aa88fe0711bfa08e8f3a58422b17364301d693f7c5

primary candidate hash:
a47141952a1a517555a32d3c6b091bf159e2c6f254f1ab63be99f8a78e5a3551

secondary candidate hash:
acb0289f69ea2d149bd57ef7632a0befcde9471d1e4698954b19b14ce90b8317

preregistration hash:
bd7b25e2b3057d2174001fcc105d19705e1b6a22c1cae057cde3feb399a472ca
```

The protocol hash binds the raw universe, outer split, direction, overlap
policy, source train/validation, target train/test sample/subject/logical IDs,
batching, schedules, seed and architecture signature. The immutable
preregistration has `execution_enabled: false`.

Future modes are `source_only`, `dann`, and a clearly labeled optional
`target_supervised_upper_bound`. The primary comparison is DANN minus
source-only on the same target-test subjects using participant-level macro F1
and balanced accuracy. `proceed` requires mean macro-F1 gain at least 0.01,
non-negative balanced-accuracy gain, and wins for at least 60% of target-test
subjects. `do_not_proceed` applies if both primary metrics worsen or wins are
below 30%. With eight target-test subjects, any result remains diagnostic.

## Readiness, artifacts and limitations

Readiness is `dann_protocol_ready`: both domains exist, the strict primary
candidate meets 10/5/5 thresholds, outer and logical leakage audits pass,
target labels are inaccessible to the training contract, the production
DANN forward passes, hashes are deterministic, and execution is disabled.

Runtime artifacts are under
`benchmark_results/domain_adaptation_dann_raw_protocol/` and are ignored by
Git. They include the canonical raw reference, domain/subject/logical audits,
both direction manifests, overlap audit, source validation and unlabeled
target manifests, batching/schedule/objective/architecture contracts,
protocol and preregistration hashes, readiness decision, empty errors table,
and runtime report.

Validation completed with the required interpreter: changed Python files
compiled successfully; the new protocol suite passed 22 tests; both
`python -m pytest -q tests` and repository-root `python -m pytest -q` passed
1,125 tests with 13 non-failing warnings. `git diff --check` is clean.

Limitations: source sizes are strongly asymmetric; only one outer fold and one
seed are prepared; only eight target-test subjects support the selected
direction; source provenance may encode acquisition/experiment differences
beyond the intended domain contrast; no causal or device-domain claim is
made; and no performance evidence exists because training and inference were
explicitly out of scope.

A future diagnostic run requires separate authorization after this tracked
protocol, full tests, manifests, hashes and source-only/DANN budget equality
are reviewed. Target-test must remain sealed until direction, schedules,
batching, checkpoint rule and stopping policy are fixed.
