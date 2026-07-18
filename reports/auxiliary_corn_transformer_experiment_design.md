# Auxiliary CORN Transformer experiment design

Protocol status: preregistered design, 2026-07-18. Source revision:
`d8428d9dfadb17078fd523b03ebda3c1e805b996`. No training, lambda search,
checkpoint creation, or outer-test evaluation was performed while writing this
document.

## 1. Research hypothesis

For the fixed five-class `label_q5` sequence task, adding a CORN auxiliary objective
to the categorical Transformer will reduce ordinal MAE and severe-error rate of the
primary categorical prediction while preserving more balanced accuracy and macro F1
than a pure CORN output head. The mechanism under study is representation-level
regularization by an ordinal task, not ensembling or choosing between heads at test
time.

The primary estimand is the paired subject-level difference between the joint model's
categorical output and the existing categorical model. Pure CORN is a secondary
reference. All claims are internal to the existing 53-subject benchmark; external
generalization is not tested here.

## 2. Baselines

The comparisons are:

1. Primary: `categorical_corn` primary categorical output versus the existing
   `categorical` Transformer.
2. Secondary: `categorical_corn` primary categorical output versus the existing pure
   `corn` Transformer.
3. Diagnostic only: auxiliary CORN output versus both primary outputs within the same
   joint model.

Existing comparable categorical and pure-CORN runs for seeds 7, 42, and 123 are
reused only after exact identity and protocol audits. A reference must have the same
sequence IDs, targets, outer folds, split seed 42, model seed, feature group,
normalization scope, and inner validation groups. Older categorical seed-7/123 runs
whose split seed varied with model seed remain ineligible. CORAL is not part of the
new experiment.

## 3. Feature groups

Two prespecified feature groups are evaluated separately:

| Role | Feature group | Features/token | Canonical feature SHA-256 |
| --- | --- | ---: | --- |
| Primary | EEG+POW (`eeg_pow`) | 448 | `8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51` |
| Control | EEG-only (`eeg_only`) | 168 | `6e822ee172422e7138945b47b2b27c947393b828b72d96b7a8e22850aded8aca` |

No POW-only, preprocessing, label-definition, raw-EEG, or sequence-length variant is
added. Features are standardized from inner-train only using the existing adapter.

## 4. Seeds and splits

Model initialization seeds are exactly 7, 42, and 123. Seeds are repeated
initializations, not independent statistical units.

The data and split contract is fixed:

```text
source parquet SHA-256:
  26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8
supervised windows: 45,384
sequences: 44,142
subjects: 53
sequence length / stride / target: 8 / 1 / last
sequence-index SHA-256:
  1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa
outer protocol: five-fold GroupKFold by subject_id
outer folds: 1, 2, 3, 4, 5
outer split random_state: 42
inner validation: group_record by record_group_id, validation_size 0.15
inner split random_state: 42 for every model seed and lambda
```

Outer train/test subject overlap and inner train/validation logical-record overlap
must each be zero. The same inner indices and normalization statistics are used by all
lambda candidates in a given feature-group/seed/fold cell. The Transformer is newly
initialized for each lambda and fold; only the split and preprocessing state are
shared.

## 5. Auxiliary-weight strategies

Three strategies were considered before implementation:

| Strategy | Statistical validity | Leakage risk | New fold fits | Complexity | Interpretation | Pipeline fit |
| --- | --- | --- | ---: | --- | --- | --- |
| One fixed lambda | Valid if fixed before results, but sensitive to loss scale | Low | 30 | Low | Simple, but weight is weakly justified | Direct adapter/config support |
| Nested grid per outer fold | Strict if candidates use only outer-train and outer-test is untouched | Low with a dedicated fit/select seam | 90 | Moderate | Transparent fold-specific choice from three declared values | Requires minimal candidate orchestration |
| Learned task weight | Valid if fully prespecified, but introduces another optimized parameter | Low | 30 | High | Harder to attribute gains specifically to CORN | Requires new objective/optimizer state logic |

The recommended primary strategy is a nested grid. A fixed weight is retained only as
a possible future sensitivity analysis; a learned weight is rejected for the first
study because it adds a second methodological innovation and weakens interpretation.

## 6. Recommended lambda-selection method

The candidate grid is fixed before implementation:

```text
auxiliary_weight in {0.25, 0.5, 1.0}
```

Each feature-group/seed/outer-fold cell selects its weight independently from the same
three candidates. The experiment therefore has 18 candidate configurations and 90
candidate fold fits. There is no post-selection refit: the selected candidate's best
inner-validation checkpoint is the checkpoint applied once to outer-test.

Selection uses constrained optimization (approach B) on the primary categorical
output. For each candidate, compute inner-validation balanced accuracy, macro F1,
severe-error rate, and ordinal MAE. The categorical reference is evaluated on the
identical inner-validation rows with its paired seed/fold checkpoint.

The preregistered balanced-accuracy non-inferiority margin is 0.0100 absolute:

```text
candidate is eligible iff
BA_candidate >= BA_categorical_reference - 0.0100
```

The one-percentage-point margin is an absolute, interpretable tolerance fixed before
joint-model results. It is also strictly smaller than the already observed mean
EEG+POW pure-CORN BA cost of 0.01114, matching the stated goal that the joint model
lose less discrimination than pure CORN. It must not be changed after inspecting any
joint outer-test prediction.

Among eligible candidates, select lexicographically:

1. minimum categorical-output severe-error rate;
2. minimum categorical-output ordinal MAE;
3. maximum categorical-output balanced accuracy;
4. maximum categorical-output macro F1;
5. smaller auxiliary weight.

Values are compared after deterministic rounding to `1e-8`; this is a numerical tie
rule, not a practical-equivalence margin. If no candidate is eligible, choose the
candidate with maximum balanced accuracy, then maximum macro F1, then minimum severe
error, minimum ordinal MAE, and smaller weight. Mark the cell
`ba_guard_fallback=true`. Any such fallback is reported and prevents a claim that the
constraint was uniformly satisfied.

Approach A was rejected because optimizing only ordinal MAE can reproduce the pure-
CORN BA cost. Approach C was rejected because its metric weights have no natural
scale. Approach D is deterministic only after another priority rule is supplied and
does not directly encode the categorical-quality constraint.

## 7. Early-stopping criterion

Every candidate uses the current maximum of 15 epochs, patience 4, and strict
improvement in validation categorical cross-entropy. The monitor is:

```text
validation_categorical_loss
```

Total validation loss is logged but not used for checkpoint choice because its scale
changes with lambda and it can improve solely through the auxiliary head. Ordinal
loss, balanced accuracy, ordinal MAE, and outer-test metrics are not stopping
monitors. The selected epoch, monitor name/value, total/categorical/ordinal validation
losses at that epoch, stopping reason, and epoch count are fold artifacts.

## 8. Nested-validation protocol

For each feature group, model seed, and canonical outer fold:

1. Materialize the canonical outer-train and outer-test sequence split without
   changing sequence IDs or ordering.
2. Resolve the single canonical `group_record` inner train/validation split from
   outer-train using split seed 42.
3. Fit normalization on inner-train only; verify identical feature order, mean, and
   scale for all candidates and the categorical reference.
4. Evaluate the paired categorical reference checkpoint on inner-validation only.
5. For each lambda in ascending order 0.25, 0.5, 1.0, initialize a fresh joint model
   with the cell's model seed, train on the same inner-train, early-stop on categorical
   validation CE, and compute selection metrics on the same inner-validation rows.
6. Do not call `BenchmarkRunner.run()` for each candidate: the current method predicts
   outer-test immediately. Use a minimal in-process fit-candidate/evaluate-fitted-model
   seam that reuses adapter and runner behavior without a second training loop.
7. Select lambda with the rule in section 6. The selection manifest is finalized
   before any joint-model outer-test prediction is computed or loaded.
8. Apply only the selected best checkpoint to outer-test once; save standard fold
   metrics and predictions.
9. Audit zero outer subject overlap, zero inner record overlap, exact sequence and
   target identity, finite probabilities, strict checkpoint reload, and selected-
   lambda provenance.
10. Continue to the next cell. Resume is keyed by resolved candidate hashes and the
    immutable selection manifest.

Outer-test labels, metrics, predictions, and existing outer-test comparison reports
are not inputs to lambda selection. Candidates may be trained independently, but all
three candidate results for a cell must be complete and audited before selection.

## 9. Metrics

Primary metrics, all computed from categorical `y_pred`:

- balanced accuracy;
- macro F1;
- ordinal MAE;
- severe-error rate (`abs(y_pred - y_true) >= 2`).

Secondary primary-head metrics:

- AUC from the five categorical softmax probabilities only;
- Cohen's kappa and quadratic weighted kappa;
- adjacent accuracy;
- categorical expected-rank MAE and Spearman correlation, where expected rank is
  computed from categorical class probabilities.

Auxiliary diagnostics, never substituted for primary metrics:

- auxiliary ordinal MAE and severe-error rate from the CORN threshold prediction;
- auxiliary expected-rank MAE;
- categorical/auxiliary exact agreement and absolute prediction distance;
- auxiliary probability finiteness, normalization, and cumulative monotonicity.

Window/sequence-level, fold-level, source-level, class-level, and head-agreement views
are descriptive. The inferential unit remains the subject.

## 10. Statistical analysis

For each method, feature group, seed, and subject, first compute metrics from all of
that subject's held-out outer-fold predictions. Average each subject's metric over the
three model seeds. The paired analysis then contains 53 values per method; seeds and
folds are not treated as independent samples.

The two prespecified primary Holm families are separate:

```text
EEG+POW: joint categorical output vs categorical
  balanced accuracy, macro F1, ordinal MAE, severe-error rate

EEG-only: joint categorical output vs categorical
  balanced accuracy, macro F1, ordinal MAE, severe-error rate
```

For each comparison report mean/median paired improvement, 10,000 paired subject
bootstrap 95% CI, two-sided Wilcoxon result, Holm-adjusted p-value within its family,
sign test, rank-biserial correlation, improved/degraded/tied subjects, and 10th/25th/
50th/75th/90th percentiles. Error-metric signs are oriented so positive means the
joint model improves.

Joint versus pure CORN forms a separate secondary family per feature group with the
same four metrics. Secondary metrics and auxiliary-head diagnostics are reported in
separate exploratory families or descriptively; they do not alter the primary
decision. Direction by seed is computed before across-seed averaging, and a positive
direction must appear in at least two of the three seeds where required.

The difficult-subject analysis uses the lowest categorical-performance quartile
defined independently within each feature group before examining joint-model changes.
The categorical definition is frozen across methods and seeds as in the existing
ordinal analysis.

## 11. Success criteria

The joint model is successful only if the primary categorical prediction:

1. reduces ordinal MAE relative to categorical;
2. reduces severe-error rate relative to categorical;
3. has no stable, confirmed macro-F1 degradation;
4. loses less balanced accuracy than pure CORN;
5. retains improvement in the worst categorical-performance subject quartile;
6. has a favorable direction for the main effects in at least two of three seeds;
7. is directionally consistent for EEG+POW and EEG-only, or any difference has a
   prespecified, scientifically plausible interpretation.

A strong result improves ordinal MAE and severe error, is statistically non-worse on
balanced accuracy and macro F1, and has higher balanced accuracy than pure CORN. A
negative result is recorded if it repeats the pure-CORN trade-off or provides no
stable ordinal improvement. A fold using the BA-guard fallback is disclosed and
precludes a claim of uniform inner-validation non-inferiority.

No difference is called statistically significant without the prespecified paired
subject analysis and multiplicity correction.

## 12. Computational estimate

Observed multiseed work required 2,049.6 aggregate training seconds for 60 new fold
fits (34.16 seconds/fold on average), with substantial run-to-run variation. The joint
model uses one encoder pass and adds only one small head, so this is the best available
planning reference, not a runtime guarantee.

| Strategy | New trial configurations | Fold fits | Point estimate | Practical single-GPU envelope | Checkpoint policy |
| --- | ---: | ---: | ---: | --- | --- |
| One fixed lambda | 6 | 30 | about 1,025 s (17 min) | 20-45 min | keep 30 selected folds |
| Nested grid (recommended) | 18 | 90 | about 3,074 s (51 min) | 1-2 h | keep only 30 selected folds after selection audit |
| Learned weight | 6 | 30 | at least 1,025 s | 20-60 min plus development risk | keep 30 folds plus learned-weight state |

Existing categorical and pure-CORN references add no training. The current mean
checkpoint is about 1.32 MB and mean outer-fold predictions artifact about 1.75 MB.
Thirty selected joint checkpoints are expected to require roughly 42 MB and their
outer predictions roughly 53 MB. With logs, manifests, summaries, and safety margin,
the selected-artifact plan is approximately 100-160 MB. Keeping all 90 candidate
checkpoints would add roughly 125 MB and is unnecessary after the selection manifest,
hashes, metrics, and selected checkpoint are atomically finalized. Full predictions
are not written for non-selected candidates.

Sequence tensors, feature lists, folds, and deterministic inner-train normalization
can be reused. Candidate fits are independent and resumable by candidate config hash.
Within a cell, temporary candidate checkpoints are retained until selection; after an
atomic selection manifest is written, only the selected model is a required durable
weight artifact. The experiment can resume completed candidates without rebuilding
sequences or changing splits.

## 13. Artifact schema

Experiment-level artifacts:

```text
resolved experiment config and hash
canonical_sequence_index.parquet
candidate_plan.json
lambda_selection_summary.json
run_manifest.json
aligned_predictions.parquet
subject_metrics.parquet
source_metrics.json
class_metrics.json
statistical_summary.json
Markdown report
```

Each feature-group/seed/fold selection cell records:

```text
feature_group, model_seed, split_seed, outer_fold
outer train/test subjects and overlap audit
inner train/validation record groups and overlap audit
sequence_index_sha256, feature_list_sha256
normalization hash
candidate auxiliary weights and config hashes
candidate best epoch and all three best-epoch validation losses
candidate categorical BA, macro F1, severe error, ordinal MAE
categorical-reference identity and inner-validation metrics
BA margin, eligibility, deterministic ranking, selected weight
ba_guard_fallback
selection finalized timestamp/hash before outer evaluation
```

Only the selected fold receives the standard durable fold artifacts:

```text
model.pt
metrics.json
training_log.csv
predictions.parquet
validation_split.json
normalization_stats.json
selection_manifest.json
```

Required prediction fields are the standard dataset/task/model/split/protocol/fold,
sample/sequence, subject, record, source, `y_true`, and categorical `y_pred` identity
columns; categorical `proba_0..4` and `class_probability_0..4`; `head_type`; the four
auxiliary cumulative-threshold probabilities; five auxiliary class probabilities;
auxiliary expected rank, threshold prediction, argmax; and `auxiliary_weight`.
Component losses are epoch/fold metadata, not per-sample fields.

## 14. Test plan

The implementation and orchestration tasks must cover at least these checks:

1. `encode()` is called exactly once per joint forward.
2. The categorical head returns `[B, 5]`.
3. The auxiliary CORN head returns `[B, 4]`.
4. Joint forward returns the declared typed object.
5. Categorical CE equals a manual/PyTorch reference.
6. Auxiliary CORN loss exactly equals the existing `corn_loss` implementation.
7. Total loss exactly equals `CE + lambda * CORN`.
8. With lambda zero, auxiliary gradients are absent or exactly zero as specified.
9. With positive lambda, gradients in both heads are finite.
10. Shared-encoder gradients receive finite contributions from both tasks; isolated
    backward checks show each task can affect encoder parameters.
11. Primary `predict_proba()` returns categorical `[N, 5]` probabilities.
12. AUC receives only categorical probabilities.
13. Auxiliary probabilities are returned and saved under distinct names.
14. Primary `y_pred` equals categorical softmax argmax.
15. Auxiliary ordinal prediction equals the existing CORN threshold rule.
16. Checkpoints contain both `classifier.*` and `auxiliary_ordinal_head.*`.
17. Strict load of a matching joint checkpoint reproduces both heads' outputs.
18. Loading a pure categorical checkpoint as joint is rejected before strict load
    unless a future explicit initialization mode is requested.
19. Loading a pure CORN checkpoint as joint is rejected.
20. `head_type` and `auxiliary_weight` each change the config hash.
21. A missing/non-finite auxiliary weight is rejected.
22. A negative auxiliary weight is rejected; zero is accepted for gradient testing
    but is not in the full-experiment grid.
23. Outer-test data and labels are never passed to lambda selection.
24. Inner train/validation indices and normalization statistics are identical for all
    lambda candidates in a cell.
25. Early stopping uses categorical validation loss and records the declared monitor.
26. User calibration fails explicitly for `categorical_corn`.
27. Existing categorical, CORAL, CORN, factory, adapter, runner, checkpoint, and
    calibration tests continue to pass.
28. Sequence IDs, folds, subjects, records, sources, targets, feature order, and
    sequence-index hash remain canonical.
29. The predictions writer saves all required primary/auxiliary columns with finite,
    normalized probabilities; AUC and saved primary metrics recompute exactly from
    categorical columns.
30. `.gitignore` and the source Parquet are unchanged.

Additional orchestration tests cover deterministic ranking/ties, the no-eligible
fallback flag, resume after a partial candidate set, atomic selection before outer
prediction, no full predictions for non-selected candidates, and exact paired
alignment with categorical and pure-CORN references.

## 15. Implementation stages

### Task 7B: infrastructure only

Implement typed dual output, one encoder call, the composite objective handler,
component logging, categorical primary decoding, auxiliary diagnostics, strict
checkpoint validation, explicit calibration rejection, and unit/integration tests.
Do not perform full training or lambda selection.

### Task 7C: technical smoke

Use EEG+POW, seed 42, outer fold 1, three epochs, and the declared lambda strategy.
Verify shapes, losses, gradients, device execution, logs, checkpoint reload, artifact
columns, and that primary probabilities feed metrics. This is technical only.

### Task 7D: full nested experiment

Run both feature groups, all three seeds, all five folds, and the three candidate
weights. Finalize each selection before its one outer-test evaluation. Audit all
identity, leakage, probability, checkpoint, and resume constraints.

### Task 7E: paired multiseed analysis

Compare the joint primary output with categorical and pure CORN at subject level using
the prespecified families, bootstrap, Wilcoxon, Holm, sign test, rank-biserial effect,
seed consistency, and difficult-subject analysis. Only then decide whether external
validation, subject-risk optimization, or calibration should be studied.

## 16. Decision rules

The following rules are frozen before implementation and outer-test use:

- model name: `torch_transformer`;
- joint head name: `categorical_corn`;
- primary inference: categorical softmax/argmax only;
- auxiliary inference: diagnostic only;
- objective: unweighted CE plus lambda times existing normalized CORN;
- candidate grid: 0.25, 0.5, 1.0;
- lambda selection: section 6 constrained lexicographic rule on inner validation;
- BA margin: 0.0100 absolute;
- early-stopping monitor: categorical validation CE, patience 4, maximum 15 epochs;
- no post-selection refit and no outer-test access before selection finalization;
- first experiment starts from scratch; no categorical checkpoint warm start;
- canonical 44,142 sequences, split seed 42, and inner group-record validation remain
  unchanged;
- primary statistical unit: subject; seeds averaged within subject;
- separate primary Holm families for EEG+POW and EEG-only;
- no CORAL, preprocessing, calibration, regression, label, raw-EEG, or ensemble
  extension in this experiment;
- failed constraints and null/negative findings are reported rather than repaired by
  changing weights, heads, or decision rules.
