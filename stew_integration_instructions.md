# STEW Integration Instructions

## 1. Purpose

This file contains task-specific instructions for integrating the STEW
dataset into the existing EEG benchmark and for using the existing
transfer-learning, domain-adaptation, meta-learning, and contrastive
mixin prototypes as the basis for subsequent experiments.

Read `AGENTS.md` first. All general repository, architecture, leakage,
testing, artifact, reporting, and Git rules from `AGENTS.md` remain in
force.

This file narrows the scope to:

```text
STEW dataset integration
native STEW baselines
shared EEG representation
all-seven-PM downstream evaluation
separate proxy-state evaluation
transfer-learning experiments
domain-adaptation experiments
meta-learning experiments
contrastive-pretraining experiments
```

The main project task must not be simplified to a single binary
workload classification problem.

---

## 2. Scientific scope

The project keeps two separate experimental tracks.

### Track A — Performance Metrics regression

The primary downstream task is prediction of all seven Performance
Metrics (PM):

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

Required result levels:

```text
per-target regression metrics for all seven targets
multi-output regression metrics
macro aggregation across targets
window-level metrics
subject-level metrics where supported
```

No method is complete if it reports only:

```text
target_focus
target_stress
one selected PM target
an averaged target
only a macro score
```

The canonical target order must be preserved in configs, predictions,
reports, and tests.

### Track B — proxy-state tasks

Proxy-state classification is a separate experimental track.

Examples may include:

```text
native STEW workload conditions
STEW subjective workload categories
project label_q5
approved ordinal or categorical proxies derived from project annotations
```

A proxy task must have its own:

```text
task name
config
target derivation
class order
metrics
predictions
summary table
limitations
```

Proxy-state results must never:

- replace the seven PM targets;
- be inserted into PM-regression tables;
- be described as equivalent to a PM target without evidence;
- be used to claim that PM regression has been solved;
- use thresholds or quantiles fitted on validation or test data.

---

## 3. Role of STEW

STEW is the selected external EEG dataset for the current cross-source
track.

It is used as an external source for:

```text
native cross-subject EEG baselines
representation pretraining
encoder transfer
domain-robust training
contrastive pretraining
subject-level episodic training
proxy-state experiments
```

Use a public mirror such as Kaggle only as a practical download source.
Project documentation and provenance must identify:

```text
the original dataset/publication
the exact downloaded archive or version
local file inventory
deterministic hashes where practical
license and citation requirements
```

Do not describe STEW as automatically compatible with the project data
only because both use Emotiv-class EEG hardware.

Verify from the actual files and source documentation:

```text
number of subjects
channel names
channel order
reference scheme
sampling rate
signal units or scale
recording duration
conditions
subjective annotations
file format
missing files
malformed records
```

Do not silently hardcode these values from memory.

---

## 4. Relationship between project data and STEW

The project data and STEW are separate EEG sources.

The compatibility contract must explicitly define:

```text
project dataset role
STEW dataset role
subject identifiers
record identifiers
channel intersection
canonical channel order
sampling-rate policy
window length
stride
signal preprocessing
shared raw-window shape
feature schema or encoder interface
task track
label availability by dataset
training labels allowed
calibration policy
final-evaluation policy
metrics
artifact provenance
```

The following may differ and must be audited:

```text
experimental protocol
recording organization
label semantics
channel order
sampling rate
reference
filtering
signal scale
record length
event segmentation
class balance
```

Do not concatenate datasets until these contracts are validated.

---

## 5. Required integration stages

Perform the work in stages. Do not start long mixin experiments before
the data and task contracts are stable.

### Stage 1 — source and data audit

Inspect the current repository for existing STEW code:

```text
bench/datasets/
bench/preprocessing/
bench/tasks/
bench/tasks/mixin/
bench/experiments/
bench/validation/
model_zoo/
experiments/
tests/
reports/
```

Search for:

```text
STEW
stew
workload
cognitive workload
cross_source
domain adaptation
contrastive
meta learning
transfer
```

Determine:

- whether a loader already exists;
- whether it is registered;
- whether imports work;
- whether it uses shared preprocessing or a private pipeline;
- whether it preserves subject and recording metadata;
- which tasks and configs already exist;
- whether prior results are smoke, diagnostic, baseline, final, or
  invalidated.

Historical issue descriptions must be reproduced on the current `HEAD`
before production code is changed.

### Stage 2 — data card and inventory

Create or update:

```text
reports/datasets/stew_data_card.md
reports/datasets/stew_file_inventory.md
```

The data card must include:

1. source and citation;
2. license;
3. archive/version;
4. subjects;
5. file layout;
6. channel names and order;
7. reference scheme;
8. sampling rate;
9. conditions and labels;
10. subjective annotations;
11. selected records;
12. excluded records;
13. missingness;
14. signal quality observations;
15. windowing;
16. preprocessing;
17. limitations;
18. differences from project data;
19. suitability for native and cross-source tasks.

No local absolute paths may appear in tracked documentation.

### Stage 3 — loader integration

Integrate STEW through the existing dataset abstraction and
`DATASET_REGISTRY`.

Do not instantiate the loader directly inside the runner.

The loader must preserve at least:

```text
dataset
subject_id
record_id
condition
subjective_workload
channel_order
sampling_rate
window_start
window_end
sample_id
```

`sample_id` must be deterministic and unique.

String subject IDs must be supported.

The loader must fail clearly on:

```text
missing source directory
unknown file layout
missing required channels
duplicate sample IDs
inconsistent sampling rate
malformed labels
empty selected subset
```

It must not modify source files.

### Stage 4 — channel and sampling contract

Create a deterministic project ↔ STEW channel mapping artifact.

Recommended tracked report:

```text
reports/integration/project_stew_channel_contract.md
```

It must include:

```text
project channel list
STEW channel list
intersection
excluded channels
canonical order
reference notes
sampling-rate comparison
resampling policy
window-size policy
shape after preprocessing
known uncertainties
```

Do not silently reorder channels.

Do not treat equal tensor dimensions as proof of equal semantics.

Resampling must be deterministic and documented.

Signal-level deterministic preprocessing may happen before an outer
split only when it does not fit statistics from the complete dataset.

Any fitted transform must use permitted training data only.

### Stage 5 — native STEW tasks and baselines

Implement native STEW tasks before cross-source transfer.

At minimum consider:

```text
native condition/workload classification
subjective workload proxy classification or ordinal prediction
```

The exact task definitions must come from the dataset documentation and
actual labels.

Do not silently invent a binary mapping.

Each task must explicitly define:

```text
included conditions
excluded conditions
class order
label mapping
subjective-score handling
transition handling
missing-label handling
```

Use group-aware outer evaluation by `subject_id`.

Obtain at least:

```text
Dummy baseline
Random Forest baseline
one shared-adapter neural baseline when the raw contract is ready
```

Native STEW results are not PM results.

### Stage 6 — shared EEG representation

Define one supported cross-source representation.

Preferred initial route:

```text
raw 14-channel EEG windows
→ deterministic channel alignment
→ deterministic resampling
→ shared encoder input contract
```

A handcrafted common feature schema is allowed only if every feature has
the same definition and physiological meaning in both datasets.

Do not reshape aggregated 448 EEG+POW project features into a fake raw
EEG tensor.

Do not directly combine incompatible feature matrices.

### Stage 7 — all-seven-PM downstream contract

Create a task-specific contract for evaluating transferred
representations on all seven PM targets.

Recommended report:

```text
reports/integration/stew_to_pm_transfer_contract.md
```

It must specify:

```text
pretraining dataset
downstream project dataset
encoder architecture
checkpoint format
head architecture
seven-target order
loss
per-target metrics
macro metrics
outer split
inner validation
calibration budget
checkpoint audit
artifact schema
```

STEW does not provide direct labels for the seven project PM targets.

Therefore, valid STEW contributions may include:

```text
self-supervised EEG pretraining
contrastive EEG pretraining
native workload-supervised encoder pretraining
domain-robust encoder training
meta-learning initialization
```

After STEW pretraining, the downstream model must be evaluated on every
PM target.

Do not treat STEW workload labels as direct labels for:

```text
attention
engagement
excitement
stress
relaxation
interest
focus
```

### Stage 8 — proxy-state contract

Create a separate report:

```text
reports/integration/stew_proxy_state_contract.md
```

For every proxy define:

```text
proxy_definition_id
dataset
source labels or source columns
derivation rule
thresholds or quantiles
threshold fit scope
class order
class counts
subject counts
scientific interpretation
limitations
```

Any threshold or quantile learned from data must be fitted within the
permitted training split.

A common project ↔ STEW proxy is allowed only after its semantic
compatibility is justified.

---

## 6. Transfer-learning integration

The historical transfer mixin is a reference prototype, not the
production path.

The existing leakage-safe fine-tuning pipeline is the production basis.

The original transfer prototype reset pretrained weights during
calibration. Do not restore that behavior.

Use the prototype as conceptual and implementation input for:

```text
checkpoint transfer
head replacement
head-only fine-tuning
full-model fine-tuning
calibration budgets
per-subject evaluation
```

Required transfer conditions for the all-PM track:

```text
project-only initialization baseline
STEW-pretrained encoder
head-only fine-tuning
full-model fine-tuning
```

A zero-shot PM result is only meaningful if a compatible PM head exists.
Otherwise mark it:

```text
not_applicable
```

Do not fabricate a zero-shot PM mapping.

For each transfer condition verify:

```text
pretrained checkpoint hash
fine-tune initial hash
final hash
frozen parameters for head-only
calibration/evaluation separation
no outer-test labels
all seven PM outputs
per-target metrics
macro metrics
```

Proxy-state transfer must use separate configs and reports.

---

## 7. Domain adaptation / DANN

The existing DANN/domain-adaptation prototype must be inspected and used
as the basis for implementation.

Do not write an unrelated DANN pipeline from scratch unless the report
explains why the prototype cannot be reused.

Before integration, define:

```text
source domain
target domain
shared encoder
task head
domain head
source labels available
target labels available
unlabeled target data allowed
loss terms
loss weights
batch composition
evaluation protocol
```

### DANN for proxy-state tasks

DANN may be used for a common proxy-state task only after the proxy
mapping is approved.

Required baselines:

```text
project-only or STEW-only source baseline
target-native baseline where available
zero-shot cross-source baseline
ordinary transfer baseline
DANN
```

### DANN for all-PM regression

Do not reduce the all-PM task to binary workload merely to fit a
standard DANN implementation.

STEW has no direct labels for the seven project PM targets.

A PM-oriented DANN design must explicitly use one of the following or
another justified contract:

```text
labeled project PM data + unlabeled STEW EEG
multi-task encoder with STEW proxy supervision and project PM supervision
label-agnostic domain alignment followed by PM fine-tuning
```

The PM task head must preserve seven outputs.

All seven targets must be reported separately.

The report must distinguish:

```text
task loss
domain loss
PM performance
domain-classification performance
```

No target final-evaluation labels may enter domain training.

---

## 8. Contrastive-learning integration

The existing contrastive prototype must be inspected and used as the
basis for implementation.

Required work:

```text
shared raw-EEG encoder contract
EEG-appropriate augmentations
project-window sampling
STEW-window sampling
positive-pair definition
negative-pair policy
checkpoint export
downstream adapter integration
```

Do not use augmentations that destroy label-relevant temporal or
spectral structure without justification.

Do not reshape the aggregated project feature table into pseudo-raw EEG.

Required controls:

```text
project-only contrastive pretraining
STEW-only contrastive pretraining
combined project + STEW contrastive pretraining
randomly initialized encoder
```

Downstream evaluation must include:

```text
all seven PM targets
multi-output PM macro metrics
approved proxy-state tasks separately
```

Embedding loss, t-SNE, UMAP, or clustering alone is not a final
scientific result.

---

## 9. Meta-learning integration

The existing meta-learning prototype must be inspected and used as the
basis for implementation.

Do not start with a full MAML sweep.

First define:

```text
episode task
support groups
query groups
subject-level sampling
dataset-level sampling
number of support windows
number of query windows
inner updates
outer updates
classification or regression loss
```

Possible episode units:

```text
project subject
STEW subject
project task
STEW proxy task
dataset
```

PM meta-learning must preserve a seven-output regression head or a
documented multi-head equivalent.

Every PM result must include all seven targets.

Proxy-state episodes must remain separate from PM episodes.

Calibration and final evaluation sample IDs must not overlap.

Do not add `learn2learn` or another dependency until the task contract
and current environment have been audited.

---

## 10. Required configs

Use topic-specific configs, for example:

```text
experiments/stew/
experiments/cross_source/
experiments/pm_regression/
experiments/proxy_states/
```

Recommended config families:

```text
stew_native_<task>_<model>_smoke.yaml
stew_native_<task>_<model>_groupkfold.yaml
stew_encoder_pretraining_<method>_smoke.yaml
stew_to_project_pm_transfer_<method>_smoke.yaml
stew_to_project_pm_transfer_<method>_full.yaml
project_stew_proxy_<task>_<method>_smoke.yaml
project_stew_proxy_<task>_<method>_full.yaml
```

Every cross-source config must explicitly contain:

```text
source_dataset
target_dataset
task_track: all_pm or proxy_state
task
target list
proxy_definition_id when applicable
channel_mapping
sampling_rate_policy
window_length
stride
preprocessing
encoder
task_head
method
outer_protocol
inner_validation
seed
calibration_budget
output_dir
result_status
```

No tracked config may contain a local absolute path.

---

## 11. Metrics

### All-PM regression

For each of the seven targets report, where supported:

```text
MAE
RMSE
R²
Pearson
Spearman
absolute bias
```

Also report macro aggregation.

Preserve negative R² values.

Do not replace missing values with zero.

Report sample and subject counts.

### Proxy-state classification

Report separately:

```text
accuracy
balanced accuracy
macro F1
weighted F1
kappa
AUC when defined
ordinal MAE when ordered
adjacent accuracy when ordered
severe error rate when ordered
```

Do not insert these values into PM-regression summaries.

### Transfer/adaptation gains

Report:

```text
baseline metric
adapted metric
absolute gain
relative gain
bootstrap confidence interval when already supported
subjects improved fraction
```

For error metrics, define the gain direction explicitly.

---

## 12. Artifacts

Native STEW and cross-source runs must use the existing artifact
infrastructure.

Required metadata should include:

```text
experiment_id
result_status
source_dataset
target_dataset
task_track
task
subject_id
record_id
sample_id
channel_mapping
sampling_rate
window_length
target names
proxy_definition_id
model
encoder
method
seed
fold
evaluation_protocol
commit
config hash
dataset inventory hash
```

All-PM predictions must preserve:

```text
seven target names in canonical order
y_true per target
y_pred per target
per-target metrics
macro metrics
```

Proxy-state predictions must preserve:

```text
proxy_definition_id
class order
y_true
y_pred
probabilities
```

Keep runtime artifacts under ignored output directories.

Small deterministic summaries may be written under
`reports/summary/` according to `AGENTS.md`.

---

## 13. Leakage rules specific to STEW

In addition to the general rules from `AGENTS.md`:

- no subject may occur in both outer train and outer test;
- project and STEW subject namespaces must be explicit;
- no final-evaluation labels may enter pretraining, adaptation, feature
  fitting, threshold fitting, or calibration;
- target calibration and final evaluation sample IDs must be disjoint;
- learned proxy thresholds must fit on training data only;
- channel normalization must not use final-evaluation statistics;
- resampling configuration must be deterministic;
- subject-level subjective annotations must not be copied to other
  subjects;
- repeated windows from one recording must not cross a prohibited split;
- checkpoint initialization must be audited;
- source-only, transfer, and adaptation results must not reuse different
  test subsets without explicit reporting.

---

## 14. Tests

Add focused tests without requiring the full real STEW dataset.

Use small deterministic fixtures for unit and integration tests.

Required coverage:

### Loader

1. registry lookup;
2. loader construction;
3. string subject IDs;
4. deterministic sample IDs;
5. file inventory;
6. channel names;
7. canonical channel order;
8. sampling-rate validation;
9. malformed-file errors;
10. missing-channel errors;
11. no target columns in features.

### Native tasks

12. explicit label mapping;
13. excluded labels;
14. group-aware subject split;
15. deterministic folds;
16. class/output shape;
17. metric compatibility;
18. artifact writing.

### Shared representation

19. project/STEW channel intersection;
20. deterministic reordering;
21. deterministic resampling;
22. common raw-window shape;
23. clear failure on incompatible shapes.

### All-PM track

24. canonical seven-target order;
25. seven-output prediction shape;
26. per-target metrics;
27. macro metrics;
28. no silent single-target reduction;
29. negative R² preservation;
30. PM summaries contain no proxy rows.

### Proxy-state track

31. proxy-definition version;
32. train-only thresholds;
33. class order;
34. proxy summaries contain no PM rows.

### Transfer and mixins

35. pretrained checkpoint loaded;
36. initial checkpoint hash audit;
37. head-only frozen parameters;
38. full-model update;
39. no final-test labels;
40. calibration/evaluation separation;
41. DANN source/target batch contract;
42. DANN seven-output PM head where applicable;
43. contrastive checkpoint reuse;
44. support/query separation for meta-learning;
45. CPU smoke path;
46. deterministic seed behavior.

Existing project tests must continue to pass.

---

## 15. Execution order

Use the following order:

```text
source audit
→ loader fixtures
→ loader integration
→ data card
→ channel/sampling contract
→ native STEW tasks
→ native classical baseline smoke
→ native full baseline
→ shared encoder contract
→ all-seven-PM transfer smoke
→ proxy-state transfer smoke
→ DANN smoke
→ contrastive smoke
→ meta-learning smoke
→ selected full experiments
```

For every implementation stage:

```text
py_compile
→ targeted tests
→ related integration tests
→ full pytest
→ real smoke-run
→ artifact audit
→ split/leakage audit
→ git diff --check
```

Do not launch a long multi-fold or multi-seed experiment until:

- data contracts validate;
- full tests pass;
- smoke succeeds;
- artifacts are complete;
- leakage checks pass;
- the user has explicitly requested the full run.

---

## 16. Expected reports

Create or update as applicable:

```text
reports/datasets/stew_data_card.md
reports/datasets/stew_file_inventory.md
reports/integration/project_stew_channel_contract.md
reports/integration/stew_native_baselines.md
reports/integration/stew_to_pm_transfer_contract.md
reports/integration/stew_proxy_state_contract.md
reports/integration/stew_transfer_results.md
reports/integration/stew_dann_results.md
reports/integration/stew_contrastive_results.md
reports/integration/stew_meta_learning_results.md
reports/summary/stew_cross_source_summary.md
```

Do not create empty placeholder reports for methods that were not run.

A blocked method should receive a concise audit entry with:

```text
blocker
evidence
required prerequisite
decision
```

---

## 17. Result status

Use the existing result-status vocabulary:

```text
final
baseline
smoke
diagnostic
invalidated
```

Examples:

```text
native STEW Random Forest GroupKFold → baseline
one-fold reduced-epoch transfer → smoke
channel compatibility audit → diagnostic
method with target-test leakage → invalidated
```

Never report a smoke result as a scientific comparison.

---

## 18. Stop conditions

Stop a specific method and report a blocker rather than forcing a run
when:

```text
real STEW files are absent
required channels are absent
sampling information is unresolved
label mapping is not defensible
shared encoder shapes are incompatible
target-test isolation cannot be guaranteed
the prototype resets transferred weights
a method requires an undefined task contract
```

A blocker in one method must not prevent completion of other valid
integration stages.

Do not fabricate metrics.

---

## 19. Definition of done

The STEW integration program is complete only when:

- STEW is registered through the existing architecture;
- a data card and file inventory exist;
- channel and sampling contracts are documented;
- native STEW baselines are reproducible;
- the shared raw-EEG representation is explicit;
- all seven PM targets are preserved;
- per-target and macro PM metrics are reported;
- proxy-state tasks are separate;
- transfer uses the current leakage-safe fine-tuning pipeline;
- existing mixin prototypes were inspected and used as implementation
  bases;
- DANN does not silently reduce PM regression to binary workload;
- contrastive pretraining is evaluated downstream;
- meta-learning has explicit support/query episodes;
- unit and integration tests pass;
- full pytest passes;
- smoke-runs succeed;
- artifacts are complete;
- leakage audits pass;
- no forbidden Git action was performed.

---

## 20. Required final task report

After each STEW-related task, report:

1. branch and `HEAD`;
2. task scope;
3. relevant instruction file used;
4. real-data availability;
5. files changed;
6. loader and registry changes;
7. channel and sampling contract;
8. task track: `all_pm` or `proxy_state`;
9. exact targets or proxy definition;
10. data shapes;
11. subject counts;
12. split protocol;
13. leakage checks;
14. prototype/mixin code reused;
15. prototype code replaced and why;
16. configs;
17. artifacts;
18. smoke result status;
19. full result status when run;
20. per-target PM metrics when applicable;
21. macro PM metrics when applicable;
22. proxy metrics when applicable;
23. limitations;
24. blockers;
25. targeted tests;
26. full pytest;
27. `git diff --check`;
28. `git status --short`;
29. confirmation that no forbidden Git action was performed;
30. recommended next stage.

Never hide a negative or blocked result.
