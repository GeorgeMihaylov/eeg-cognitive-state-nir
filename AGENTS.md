# AGENTS.md

Current scientific status: 2026-11-06. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) and the completed [temporal evidence](reports/diagnostics/pm_temporal_predictability_v1/README.md).

## 1. Purpose

This repository contains the research and experimental platform for EEG-based cognitive-state recognition.

The project is no longer primarily a model-zoo implementation task. The current scientific focus is:

- cross-subject recognition of cognitive states from EEG;
- seven Performance Metrics (PM) as project targets;
- leakage-safe LOW/HIGH classification;
- temporal alignment between EEG and PM;
- temporal dynamics and useful history in EEG;
- subject variability and personalization;
- reproducible confirmatory and diagnostic experiments.

The repository should remain a reproducible experimental platform rather than a collection of isolated scripts.

The main local working repository is:

```text
F:\EEG
```

This path is workstation context only. Do not hard-code absolute local paths into reusable source code when a configuration or CLI argument can be used instead.

---

## 2. Source of truth

Before making changes, use the following priority order:

1. current repository code and tests;
2. current experiment specifications, manifests, and generated reports;
3. this `AGENTS.md`;
4. current research-status documentation;
5. root `README.md`;
6. historical reports and old context files.

Historical files may describe obsolete branches, paths, targets, protocols, or model plans.

Do not silently treat historical documentation as current truth.

If documentation conflicts with current code, experiment specifications, or validated reports:

- identify the conflict;
- preserve historical artifacts;
- use the current validated contract;
- report the mismatch before making broad changes.

`PROJECT_CONTEXT.md` may contain historical snapshots. Inspect its date and scope before relying on it.

---

## 3. Required working procedure

Before editing code:

1. Inspect the current branch and working tree:

```powershell
git branch --show-current
git status --short
git rev-parse HEAD
```

2. Read the files directly related to the requested task.

3. Inspect existing:

- experiment runners;
- target contracts;
- fold definitions;
- feature caches;
- alignment utilities;
- model factories;
- metric aggregation;
- bootstrap utilities;
- manifests;
- tests;
- report formats.

4. Search for an existing implementation before creating a new abstraction.

5. Reuse existing infrastructure whenever practical.

6. Make the smallest coherent change that answers the current scientific question.

7. Add targeted tests.

8. Run the narrowest relevant test subset.

9. Run a dry-run or smoke test before a large computational experiment.

10. Only run a full expensive experiment when the task explicitly requires it or when the implementation has already passed its audits and smoke tests.

Do not begin with a broad repository refactor unless the requested task genuinely requires one.

Do not discard unrelated user changes.

---

# 4. Main scientific target contract

The primary current classification problem covers all seven PM targets:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

Do not reduce new primary experiments to Focus-only unless the experiment explicitly studies Focus.

Historical `label_q5` / Focus-only and internal fold-local Q3 experiments are legacy, not current primary classification. Preserve their results/configurations; never relabel their metrics as LOW/HIGH. Native COG-BCI N-Back/MATB, CL-Drive/CLARE three-class targets and synthetic three-class tests are valid separate tasks and must not be reinterpreted.

---

## 5. LOW/HIGH classification protocol

The current primary classification formulation is binary:

```text
LOW / HIGH
```

For each PM and each outer fold:

- LOW <= fold-local Q33;
- HIGH >= fold-local Q67;
- the middle third is excluded.

Thresholds are fitted once on canonical outer-train continuous complete cases before history filtering and reused unchanged across matched horizons and outer-test. Middle samples consume elapsed time/support but never become binary training labels.

Never calculate Q33/Q67 globally before the train/test split.

Never allow test-subject values to affect:

- target thresholds;
- scaling;
- normalization statistics;
- feature selection;
- imputation;
- dimensionality reduction;
- calibration;
- hyperparameter selection.

Do not return to the old three-class/Q3 formulation as the primary protocol without a new scientific reason.

---

# 6. Validation contract

The main scientific question is generalization to unseen participants.

Therefore the primary evaluation is cross-subject.

Use the existing five fixed outer folds by `subject_id` whenever the current experiment belongs to the main protocol.

All samples belonging to one subject must remain in the same outer fold.

Preserve record/session/source identifiers.

Do not use random row-level splits as evidence of generalization to unseen participants.

Random splits may only be used as explicitly labelled diagnostics.

Where sequence construction is involved, additionally prevent:

- record leakage;
- session leakage;
- temporal leakage;
- overlapping target leakage;
- history crossing recording boundaries.

---

## 7. Primary metric and aggregation

The primary classification endpoint is:

```text
participant-macro balanced accuracy
```

unless a preregistered experiment specification explicitly defines another primary endpoint.

Important:

- windows are not independent experimental units;
- participants are the main statistical units for final comparisons;
- subjects with more EEG windows must not dominate the aggregate result simply because they contribute more rows.

Secondary metrics normally include:

```text
macro F1
ROC-AUC
PR-AUC
```

Preserve per-fold and per-participant results.

For paired comparisons, prefer participant-level paired differences and participant-level bootstrap confidence intervals.

Do not bootstrap individual EEG windows as independent observations when estimating participant-level uncertainty.

---

# 8. Canonical EEG–PM temporal contract

A confirmatory absolute-lag experiment has already been completed on the current LOW/HIGH protocol.

The canonical temporal relationship is:

```text
EEG(t - 10 s) -> PM(t)
```

In explicit window notation:

```text
PM window:
[T, T+10]

canonical EEG window:
[T-10, T]
```

The absolute-lag convention is:

```text
absolute_lag_s = EEG.t_start - PM.t_start
```

The completed confirmatory experiment tested:

```text
[-30, -20, -10, 0, +10] seconds
```

Interpretation:

```text
-30  EEG precedes PM by 30 s
-20  EEG precedes PM by 20 s
-10  canonical PRIMARY REFERENCE
  0  contemporaneous EEG
+10  EEG follows PM; NON-CAUSAL DIAGNOSTIC ONLY
```

The `-10 s` alignment remains the canonical project reference.

Do not silently change it in a new experiment.

If a future experiment studies another alignment, it must explicitly state whether it is:

```text
exploratory
diagnostic
confirmatory
```

The canonical absolute-lag report is located under:

```text
reports/diagnostics/pm_eeg_absolute_lag_v1/
```

The experiment specification is under:

```text
experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json
```

Inspect these artifacts before modifying temporal-alignment logic.

---

# 9. Rules for temporal history

For a target PM window:

```text
PM = [T, T+10]
```

the current canonical EEG block is:

```text
block 0 = [T-10, T]
```

Longer causal history must extend backwards:

```text
block 1 = [T-20, T-10]
block 2 = [T-30, T-20]
...
```

A history sequence must remain strictly inside:

- one subject;
- one logical recording;
- one continuous temporal segment.

Never:

- cross a recording boundary;
- cross a subject boundary;
- borrow missing history from another record;
- silently interpolate large temporal gaps;
- include EEG after the target PM interval in a causal experiment.

Timestamp arithmetic must be audited with real examples before a full run.

Do not rely only on parameter names such as `lag` or `shift`; verify actual timestamps.

---

# 10. Matched/common support

Temporal comparisons are especially vulnerable to support changes.

Whenever different lags or history lengths remove different samples, explicitly audit support.

Prefer matched/common support for primary causal comparisons when feasible.

At minimum record:

```text
n_subjects
n_records
n_samples
n_train
n_test
retention_fraction
```

for each relevant configuration.

If horizon-specific support is also reported, label it separately.

Never interpret a metric improvement as an effect of temporal history without checking whether the evaluated population changed.

---

# 11. Established research results

The following directions have already been investigated and should not be repeated without a new hypothesis.

### Strong tabular baselines

On the current LOW/HIGH cross-subject formulation, LightGBM and XGBoost are strong baselines, with balanced accuracy around the mid-0.7 range across the seven PM targets.

Any new complex model must be compared against these baselines under the same protocol.

### LSTM and Transformer

Sequence models based on LSTM and Transformer have already been evaluated.

They did not show a sufficiently stable advantage over strong tabular baselines to justify repeating the same experiment.

Do not rerun them simply with minor architecture changes.

A new temporal architecture requires evidence that additional temporal history contains useful information.

### Nested AutoML

Full LOW/HIGH nested AutoML is complete: 2,730 inner trainings and 35 final outer runs. The old label_q5 Transformer AutoML pilot is historical.

### Personalization

Personalization has already been studied using a limited amount of target-subject data.

Previously investigated approaches include:

- subject-specific thresholds;
- probability calibration;
- model adaptation / fine-tuning;
- limited subject-specific training data.

Model adaptation produced the most stable improvement.

Do not start another broad personalization sweep.

After a new representation/model is selected, one confirmatory personalization experiment is appropriate.

### Channel diagnostics

Simple channel-removal experiments and inter-channel lag diagnostics have already been performed.

No stable useful inter-channel absolute lag was established.

Removing F3/F7/F8 degraded performance.

Do not continue arbitrary channel-drop search.

If spatial EEG structure is revisited, prefer interaction/connectivity representations over repeated electrode deletion.

### Absolute EEG–PM lag

The confirmatory absolute-lag experiment is complete.

Do not repeat the `[-30,-20,-10,0,+10]` sweep unless the underlying dataset or target contract changes materially.

---

# 12. Completed Temporal Predictability Audit

The completed audit addressed:

> Does EEG history before the canonical `[T-10,T]` block contain additional information about LOW/HIGH cognitive state beyond the canonical 10-second window itself?

Completed history horizons are:

```text
H10
H20
H30
H60
H120
```

where the canonical `-10 s` block is always the most recent causal block.

The audit separates two different questions:

1. does additional history contain information?
2. does a particular model architecture know how to exploit it?

Do not answer both questions by immediately training another large neural network.

---

## 13. Completed diagnostics and retained methodological rules

Support/ACF/MI/surrogates/PE/SSA/restricted FNN are complete. No reliable FNN embedding dimension or physical attractor was established. Preserve these rules for future diagnostic work; do not rerun completed diagnostics without a new request.

### 13.1 Support audit

Measure how much usable data remains for:

```text
10
20
30
60
120 seconds
```

of continuous causal history.

### 13.2 Autocorrelation

Estimate temporal dependence at relevant delays.

Compute within recordings first.

Do not concatenate unrelated recordings into an artificial time series.

Aggregate subsequently at recording and participant levels.

### 13.3 Permutation entropy

Permutation Entropy may be used as an exploratory nonlinear complexity measure.

Record:

- embedding parameters;
- number of valid sequences;
- minimum sequence-length rule.

Do not report unstable estimates from sequences that are too short.

### 13.4 Mutual information and delay selection

Use mutual information as an exploratory method for estimating a useful temporal delay `tau`.

The goal is to determine whether adjacent 10-second blocks are largely redundant or contain complementary state information.

Do not select `tau` using final test classification performance.

### 13.5 Takens-style delay embedding / False Nearest Neighbours

Takens-style state-space reconstruction and False Nearest Neighbours may be used to estimate an effective embedding dimension.

Treat these as diagnostic evidence about the dimensionality of temporal dynamics.

Do not claim that noisy EEG exactly satisfies an ideal deterministic dynamical-system model.

### 13.6 Hankel / Singular Spectrum Analysis

Use Hankel trajectory matrices and Singular Spectrum Analysis (SSA) to inspect:

- singular-value spectrum;
- explained energy;
- effective rank;
- dominant temporal components;
- signal-versus-surrogate structure.

SSA is primarily diagnostic in the current project.

Do not turn it into an unrelated forecasting benchmark.

### 13.7 Hurst / DFA

Long-memory measures may be used only when the available sequence length supports a defensible estimate.

If a sequence is too short, explicitly report:

```text
insufficient_sequence_length
```

rather than generating unreliable values.

---

# 14. Surrogate and temporal-order controls

Temporal structure should be tested against appropriate controls.

A surrogate transformation must remain inside the same recording and subject.

Useful controls include:

- within-record permutation;
- block permutation;
- other explicitly documented structure-destroying surrogates.

Surrogates should preserve the relevant marginal values while destroying temporal organization.

Compare observed and surrogate:

- autocorrelation;
- permutation entropy;
- SSA structure;
- other temporal statistics.

The conditional H120 order control is complete (25/25). Do not rerun it without a new request. The following describes the retained control principle.

For example:

```text
ordered:
-60 -50 -40 -30 -20 -10

shuffled:
-30 -60 -20 -50 -40 -10
```

The canonical `-10 s` block remains fixed.

This distinguishes:

```text
more historical observations are useful
```

from:

```text
their temporal ordering is useful
```

---

# 15. Predictive history benchmark

The first matched predictive history test is complete: 175/175, strict common H120 support, LightGBM 200 trees / n_jobs=4 / seed 42. H120 gains pass BH across 28 contrasts for 5/7 PM: excitement, focus, interest, relaxation, stress. Attention/engagement have no confirmed H120 gain. The gated order control reused ordered H120 and completed 25/25 shuffled fits; order effects pass BH across five contrasts for excitement, focus, interest and relaxation (4/5); stress is inconclusive. This two-stage analysis is conditional, not an unconditional independent seven-PM test.

Use the existing LightGBM configuration.

Do not perform a new hyperparameter search.

Represent temporal history initially by concatenating lagged feature blocks.

Example:

```text
H30 =
features@-30
+
features@-20
+
features@-10
```

Use:

```text
H10
```

as the primary reference.

Primary comparison:

```text
delta(H) = metric(H) - metric(H10)
```

Use matched evaluation support for the primary comparison.

Only after this baseline establishes useful temporal information should a new temporal architecture be considered.

---

# 16. Time-series methods relevant to future work

The time-series course materials suggest several methods that are scientifically relevant to this project.

They are not all equal priorities.

## 16.1 Wavelet representation — high priority

Wavelet features are a planned follow-up experiment.

They are relevant because EEG is:

- nonstationary;
- oscillatory;
- time-frequency structured.

A future wavelet ablation should compare:

```text
current features
wavelet features
current + wavelet features
```

under exactly the same cross-subject LOW/HIGH protocol.

Do not combine this experiment with simultaneous architecture changes.

---

## 16.2 Phase synchronization / Kuramoto-style features — high priority after wavelets

EEG channels may carry useful information through phase synchronization rather than simple absolute lag.

A future experiment may derive band-specific instantaneous phases and synchronization summaries such as a Kuramoto-style order parameter.

Possible features include:

```text
mean synchronization
std synchronization
quantiles
temporal variability
band-specific synchronization
```

This direction is preferred over another arbitrary channel-removal sweep.

Keep interpretation conservative:

phase synchronization is an association measure and does not by itself establish directed physiological causation.

---

## 16.3 Reservoir computing / ESN — conditional inexpensive baseline

Echo State Networks or related reservoir methods may be useful as a cheap temporal baseline if longer history is shown to contain predictive information.

Use them before a large new architecture when they can answer the same question more cheaply.

---

## 16.4 RevIN / Instance Normalization — conditional

Reversible Instance Normalization or Instance Normalization may be evaluated when a new neural temporal model is introduced.

They are particularly relevant to inter-subject distribution shift and nonstationarity.

Do not treat RevIN as a standalone scientific contribution.

---

## 16.5 PCMCI / Granger — exploratory only

PCMCI or Granger-style analysis may be used for exploratory inter-channel temporal dependency analysis.

Do not equate predictive dependency with physiological causation.

This is not currently a primary project direction.

---

## 16.6 Kalman/state-space smoothing — downstream application

State-space filtering may later be useful for stabilizing online cognitive-state probability trajectories.

It is downstream of the current classification research and should not replace the main temporal-information audit.

---

## 16.7 Hawkes / point processes — currently out of scope

The main project uses regularly sampled/windowed EEG classification rather than an event-stream formulation.

Do not introduce Hawkes processes unless a future experiment explicitly reformulates cognitive-state transitions as discrete events.

---

# 17. Policy for new temporal architectures

Do not create an architecture zoo.

Model choice must follow empirical evidence.

Current evidence justifies one bounded, prespecified H120 sequence architecture compared directly against ordered-H120 LightGBM on identical support. This comparison is planned, not executed. Do not launch broad AutoML, Mamba, PatchTST or TimesNet sweeps merely because history/order effects are positive. Freeze the model and comparison before test outcomes; preserve target-dependent interpretation and negative results.

Only one new temporal architecture should normally be evaluated at a time.

Its scientific motivation must be written before inspecting its final test results.

---

# 18. PM interpretation and scientific claims

Performance Metrics are proxy/device-derived targets and must not automatically be described as direct ground truth of human cognitive state.

Use careful wording such as:

- recognition of PM-defined cognitive states;
- EEG association with PM-derived LOW/HIGH states;
- prediction of Performance Metric state.

Avoid unsupported statements such as:

- direct measurement of cognition;
- physiological causation established by lag;
- EEG causes PM;
- a specific brain region causes a PM state.

The confirmed `-10 s` temporal relationship is a predictive/alignment result.

It is not by itself proof of a physiological 10-second causal delay.

The internal PM generation process may include smoothing or proprietary processing and should be considered when interpreting temporal effects.

---

# 19. Dataset handling

Current canonical cache: 371 features (182 spectral, 140 statistical, 28 entropy, 21 connectivity); 30,958 deduplicated windows, 54 participants, 86 records, raw [1,14,2560] at 256 Hz. EEG 168 + POW 280 = 448 is a historical separate representation.

Primary internal data sources include:

```text
gpn_data
Old_EEG
```

Preserve source information throughout the pipeline.

Auxiliary/open datasets may be used for:

- pretraining;
- transfer;
- external validation;
- robustness experiments.

COG-BCI has previously been used as an auxiliary EEG source.

Do not merge datasets merely because they all contain EEG.

Before cross-dataset experiments, explicitly verify:

- channel compatibility;
- sampling rates;
- preprocessing;
- task semantics;
- target semantics;
- session structure;
- participant identifiers.

WESAD is a physiological-signal dataset, not an EEG dataset.

Do not describe it as EEG.

Never commit raw private EEG data to the repository.

---

# 20. External validation

External validation is an important future step for a publication-quality result.

A suitable external dataset should preferably contain:

- EEG;
- independent participants;
- cognitively meaningful workload/stress/attention-like labels;
- enough metadata to construct leakage-safe subject-level evaluation.

Do not force incompatible label spaces into the seven internal PM targets.

External validation may test whether a scientific conclusion transfers even when exact labels differ.

Examples of transferable conclusions include:

- useful temporal horizon;
- benefit of wavelet representation;
- value of personalization;
- usefulness of synchronization features.

---

# 21. Experimental categories

Every experiment should be explicitly labelled as one of:

```text
exploratory
diagnostic
confirmatory
```

### Exploratory

Used to discover hypotheses or promising configurations.

### Diagnostic

Used to understand data, protocol, leakage, or mechanism.

### Confirmatory

Used to test a previously specified comparison under a frozen protocol.

Do not describe an exploratory search as confirmatory.

Do not change a reference configuration after looking at test results and still call the same experiment confirmatory.

---

# 22. Reproducibility requirements

A substantial experiment should record, where applicable:

```text
experiment_id
experiment_type
dataset/input identifiers
input hashes
target contract
feature contract
temporal contract
fold contract
support rule
model configuration
random seed
runtime information
run matrix
sample counts
subject counts
record counts
metrics
artifact paths
failure/skip reason
```

Prefer deterministic reconstruction.

Large experiments should support resume where practical.

Resume must verify configuration and input integrity before reusing results.

---

# 23. Required experiment artifacts

For substantial experiments, prefer the existing project structure and produce an appropriate subset of:

```text
experiment specification
manifest
run matrix
alignment audit
support audit
fold-level metrics
participant-level metrics
aggregate metrics
paired deltas
bootstrap confidence intervals
plots
README/report
runtime bundles
```

Do not create a parallel reporting system if an existing one can be extended.

Do not overwrite historical experiment artifacts.

Create a new versioned namespace when the scientific contract materially changes.

---

# 24. Statistical discipline

The effective independent sample size is closer to the number of participants than to the number of EEG windows.

Therefore:

- report subject counts;
- preserve participant-level metrics;
- use participant-level resampling for main confidence intervals;
- avoid treating tens of thousands of windows as tens of thousands of independent subjects.

Confidence intervals are descriptive unless a formal inferential protocol has been specified.

If many targets/configurations are tested, do not overstate nominal significance without addressing multiplicity.

Do not select a new canonical protocol simply because it achieves the best point estimate in an exploratory sweep.

---

# 25. Model and feature comparisons

When testing one scientific factor, avoid changing several other factors simultaneously.

Bad comparison:

```text
old features + LightGBM
vs
wavelets + new normalization + Mamba + new thresholds
```

because the cause of any change is uninterpretable.

Preferred sequence:

```text
same model, new representation
same representation, new model
same model/representation, personalization
```

Use ablations whenever practical.

---

# 26. Code architecture rules

Reuse existing project abstractions.

Prefer:

- extending an existing runner;
- extending existing metrics;
- reusing model factories;
- reusing fold contracts;
- reusing bootstrap utilities;
- reusing feature/alignment lookups.

Avoid:

- duplicated split logic;
- duplicated metric implementations;
- duplicated threshold code;
- parallel experiment frameworks;
- hidden global state;
- import-time data loading;
- import-time GPU initialization.

Use:

- `pathlib.Path`;
- type hints on public APIs;
- `logging` for library/runtime diagnostics;
- deterministic ordering;
- explicit validation errors.

Keep:

- dataset logic;
- target logic;
- model logic;
- temporal alignment;
- evaluation;
- reporting

separated where practical.

---

# 27. Backward compatibility

Several older experiments depend on shared runners and utilities.

When extracting or modifying common functionality:

- preserve historical behavior unless the task explicitly changes it;
- add targeted regression tests;
- avoid silently changing previous experiment semantics;
- document any unavoidable contract change.

Historical experiment reports should remain reproducible.

---

# 28. Testing rules

Use targeted tests appropriate to the changed code.

Tests should cover scientific invariants, not only syntax.

Typical temporal tests include:

- correct lag sign;
- exact window alignment;
- no subject crossing;
- no record crossing;
- no temporal-gap crossing;
- no PM overlap in causal history;
- fold-local Q33/Q67;
- train/test subject isolation;
- common-support construction;
- deterministic feature ordering;
- deterministic seeds;
- surrogate preservation properties;
- canonical block preservation in shuffled controls.

Tests should not require:

- private raw datasets;
- network access;
- large GPU runs.

Use small synthetic fixtures when possible.

For expensive pipelines:

1. targeted unit tests;
2. dry-run;
3. one-target/one-fold smoke test;
4. full experiment only after these pass.

---

# 29. Runtime warnings and failures

Do not hide warnings that may affect scientific validity.

Classify them.

Examples:

### Benign/runtime compatibility warning

Record it, but execution may continue.

### Data/shape/alignment warning

Investigate before using results.

### Leakage or protocol violation

Stop the experiment.

### Missing run / corrupted artifact / hash mismatch

Do not silently resume from it.

Report actual completed/restored/failed run counts.

Never claim:

```text
175/175 complete
```

unless this was verified from the artifacts or execution output.

---

# 30. Generated artifacts and repository hygiene

Generated experiment outputs belong in configured result/report directories.

Do not commit by default:

- raw private data;
- large caches;
- large runtime bundles;
- model checkpoints;
- temporary logs;
- local credentials;
- `.env` files;
- machine-specific artifacts.

Small specifications, manifests, compact reports, tests, and documentation may be appropriate repository artifacts depending on existing project conventions.

Inspect `.gitignore` before introducing new generated paths.

---

# 31. Git safety

Do not perform the following unless explicitly requested:

```text
git add
git commit
git push
git reset --hard
git clean
git rebase
git checkout -- <user file>
```

Do not:

- rewrite history;
- force-push;
- delete branches;
- discard unrelated user modifications;
- remove untracked user work;
- stage files automatically after implementation.

The user manages final staging and commits unless explicitly stated otherwise.

At the end of a task, report branch, HEAD, and working-tree summary when relevant.

---

# 32. Current publication-oriented research path

The current project should move toward a coherent scientific story rather than maximizing the number of tested architectures.

The preferred research sequence is:

```text
validated LOW/HIGH protocol
        ↓
cross-subject baseline
        ↓
confirmed EEG–PM temporal alignment
        ↓
completed temporal predictability / effective history
        ↓
one bounded H120 sequence comparison (planned)
        ↓
time-frequency representation
        ↓
inter-channel synchronization

        ↓
confirmatory personalization
        ↓
external validation
```

The central research question is approximately:

> What temporal information in EEG is useful for cross-subject recognition of PM-defined cognitive states, how should that information be represented, and how much can limited subject-specific adaptation improve the result?

Do not replace this question with:

> Which of many neural architectures gives the highest score?

---

# 33. Immediate next milestone

The temporal milestone is complete (175/175 matched, 25/25 order control). The next milestone is one bounded H120 sequence-model comparison against the frozen ordered-H120 LightGBM reference, only on explicit authorization. Wavelet and phase-synchronization representations remain separate later ablations. No architecture has yet been evaluated as this next milestone.

---

# 34. Definition of done for an experiment

An experiment is complete only when the required subset of the following is satisfied:

- scientific question is explicit;
- experiment category is explicit;
- protocol is frozen or clearly exploratory;
- target contract is explicit;
- temporal contract is explicit;
- support rule is explicit;
- leakage checks pass;
- tests pass;
- dry-run passes;
- smoke test passes;
- full run completes when requested;
- run counts are verified;
- metrics are saved;
- participant-level results are preserved;
- report is generated;
- limitations are documented;
- results are interpreted without exceeding what the protocol supports.

A successful implementation is not the same as a successful scientific result.

Negative results must be preserved and reported.

---

# 35. Codex task response format

At the end of a task, respond in the following structure:

```text
Repository state
- branch:
- HEAD:
- working tree:

Summary
- what was implemented
- what scientific question it addresses

Reused infrastructure
- ...

Changed files
- path: purpose

Protocol / semantics
- target contract
- temporal contract
- support rule
- model/reference configuration

Validation
- tests:
- dry-run:
- smoke test:
- full run, if requested:

Results / diagnostics
- only results that were actually produced

Artifacts
- paths to generated reports/tables/manifests

Risks / limitations
- ...

Recommended next step
- ...
```

Do not claim results from runs that were not executed.

Do not infer scientific conclusions from smoke tests.

Do not automatically continue to the next large experiment unless explicitly requested.