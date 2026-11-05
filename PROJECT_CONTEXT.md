# PROJECT_CONTEXT.md

## 1. Project summary

The project studies **cross-subject recognition of cognitive states from electroencephalography (EEG)**.

The current work combines several related research questions:

1. how reliably cognitive-state proxies can be recognized from EEG for previously unseen participants;
2. how EEG should be temporally aligned with Performance Metrics (PM);
3. whether temporal EEG history beyond the canonical 10-second window contains additional predictive information;
4. whether time-frequency and inter-channel representations improve generalization;
5. how much limited target-subject data can improve predictions through personalization;
6. whether the main findings reproduce on independent external EEG data.

The project is not primarily a model-zoo implementation exercise anymore.

The main objective is to develop a **scientifically controlled and reproducible experimental pipeline** and use it to answer the questions above.

The current local working repository is:

```text
F:\EEG
```

Historical paths such as `F:\eeg-final-unification` or old `D:\PycharmProjects\...` locations should not be treated as the active project path.

---

# 2. Current research status

Status summarized from the validated project work completed through the latest active research cycle.

The project has moved through several stages:

```text
dataset harmonization
        ↓
baseline ML/DL benchmarking
        ↓
cross-subject validation
        ↓
domain adaptation / self-supervised / meta-learning experiments
        ↓
seven-PM target unification
        ↓
LOW/HIGH reformulation
        ↓
EEG–PM temporal alignment audit
        ↓
personalization
        ↓
absolute-lag confirmatory experiment
        ↓
CURRENT: temporal predictability and representation analysis
```

The current scientific focus is therefore not to add many new architectures.

The next goal is to understand **what temporal information is present in EEG and how it should be represented**.

---

# 3. Main internal datasets

## 3.1 `gpn_data`

`gpn_data` is the main newer internal EEG dataset.

It contains:

- multichannel EEG;
- recordings from a wearable Emotiv-class EEG device;
- Performance Metrics;
- markers;
- annotations;
- recording metadata;
- service files.

The relevant PM variables are:

```text
Attention
Engagement
Excitement
Stress
Relaxation
Interest
Focus
```

---

## 3.2 `Old_EEG`

`Old_EEG` is the older internal EEG source.

It contains conceptually similar information:

- EEG;
- Performance Metrics;
- markers;
- recording metadata.

However, its original structure and storage conventions differ from `gpn_data`.

The two sources were previously harmonized into a common experimental representation.

---

## 3.3 Combined internal data

The processed internal dataset contains approximately:

```text
~45k accepted supervised EEG windows before deduplication
~31k deduplicated windows
~54 participants
~119 source records
~86 logical recordings
```

Historical counts vary slightly between pipeline versions because several audits corrected filtering, record identity, and duplicate handling.

Do not use an old count as a hard-coded invariant.

Always derive actual support from the current canonical cache/manifest used by the experiment.

The established handcrafted representation contains approximately:

```text
168 EEG statistical features
280 power/spectral features
448 features total
```

and has historically been referred to as:

```text
pow_plus_eeg
```

Exact feature contracts must be taken from the current feature cache or experiment manifest rather than reconstructed from memory.

---

# 4. Auxiliary / public EEG data

## 4.1 COG-BCI

COG-BCI was prepared as an external auxiliary EEG source.

The completed inventory contained approximately:

```text
29 participants
3 sessions per participant
87 participant-session combinations
1,044 EEG records
~81.75 hours of EEG
500 Hz sampling
63 or 64 channels depending on participant
```

Tasks include variants of:

```text
N-Back
MATB
PVT
Flanker
Resting EO/EC
```

COG-BCI was used for contrastive/self-supervised EEG pretraining experiments.

It should not automatically be treated as directly label-compatible with the seven internal PM targets.

---

## 4.2 WESAD

WESAD has been supported in parts of the benchmark infrastructure.

WESAD contains physiological signals related to affect/stress.

It is **not an EEG dataset**.

Do not describe WESAD as EEG.

---

# 5. Seven-target PM contract

The current main target space contains seven Performance Metrics:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

Current experiments should normally address all seven.

Focus-only experiments and the historical `label_q5` target remain legacy/reference experiments.

Do not reduce a new primary experiment to Focus without an explicit scientific reason.

---

# 6. Historical target formulations

Several formulations have been used during the project.

## 6.1 Five-class Focus target

A historical quantile-based Focus target:

```text
label_q5
```

was used extensively during the early benchmarking and deep-learning stages.

It remains useful for historical comparison and some transfer experiments.

It is not the current primary classification target.

---

## 6.2 Continuous PM regression

All seven PM were also formalized as continuous regression targets.

Regression remains scientifically relevant, but the current main classification line is LOW/HIGH.

---

## 6.3 Previous three-class formulation

A three-class PM formulation was investigated during intermediate project stages.

It was later replaced as the main classification protocol.

Do not return to it as the primary formulation without a new scientific hypothesis.

---

# 7. Current primary classification protocol: LOW/HIGH

The current main classification problem is binary:

```text
LOW / HIGH
```

For each PM and each outer training fold:

```text
LOW  = values below train-fold Q33
HIGH = values above train-fold Q67
```

The middle third is excluded.

Important:

- Q33/Q67 are calculated from training subjects only;
- thresholds are fold-local;
- test-subject target values must not influence threshold construction;
- threshold logic must remain identical when models or temporal configurations are compared.

This reformulation substantially improved the usefulness and stability of the cross-subject classification task.

Strong tabular models on the modern protocol achieved balanced accuracy around:

```text
~0.75
```

with ROC-AUC approximately in the high-0.8 range for several configurations.

Exact values must always be taken from the corresponding report rather than copied from this context file.

---

# 8. Validation protocol

The primary scientific problem is:

> performance on previously unseen participants.

Therefore the main protocol is **cross-subject validation**.

The project uses fixed subject-level outer folds.

Primary rules:

- one participant must never appear in both train and test within one outer fold;
- recording/session metadata must be preserved;
- temporal sequences must not cross recording boundaries;
- all fitted preprocessing must use training data only;
- target thresholds are fold-local;
- calibration and feature selection are train-only.

Random row/window splits are not evidence of cross-subject generalization.

They may only be used as explicit diagnostics.

---

# 9. Historical baseline lesson: random split vs subject split

One of the earliest important findings was the large discrepancy between random-window and subject-grouped evaluation.

Historical five-class Focus experiments produced approximately:

```text
Random split RF accuracy:
~0.73

Subject-grouped RF balanced accuracy:
~0.32
```

This demonstrated that EEG windows contain strong participant/session-specific structure.

The result motivated the current strict subject-wise evaluation strategy.

Do not compare random-split metrics directly with current cross-subject metrics as though they represented the same task.

---

# 10. Classical ML baselines

Historical and current experiments have included:

```text
Logistic Regression
Random Forest
Histogram Gradient Boosting
LightGBM
XGBoost
```

Under the modern LOW/HIGH formulation, LightGBM and XGBoost became strong baselines.

These models must remain comparison points for future temporal or representation models.

A complex model is not considered useful merely because it is architecturally newer.

It should show added value under the same support and cross-subject protocol.

---

# 11. Deep-learning experiments

The project has evaluated multiple neural approaches, including:

```text
MLP
EEGNet
ShallowConvNet
LSTM
BiLSTM
Transformer-based temporal models
```

Historical raw-EEG models generally showed relatively weak cross-subject performance compared with later engineered-feature LOW/HIGH baselines.

Examples from earlier five-class experiments included balanced accuracy approximately in the:

```text
0.24–0.29
```

range for EEGNet/ShallowConvNet variants.

These numbers belong to older task formulations and must not be compared directly with modern LOW/HIGH results.

---

# 12. EEG preprocessing ablation

A dedicated preprocessing ablation was previously performed with ShallowConvNet.

Variants included combinations of:

```text
raw
band-pass
notch
CAR
band-pass + notch
band-pass + CAR
notch + CAR
band-pass + notch + CAR
```

The best historical configurations were based on band-pass and notch filtering.

Common Average Reference (CAR) reduced performance in that specific experiment.

This result is specific to that data/model/protocol combination and should not be generalized beyond it without validation.

---

# 13. Domain adaptation and representation learning

Several advanced approaches have already been investigated.

## 13.1 Domain-Adversarial Neural Network

Domain-Adversarial Neural Network (DANN) experiments were implemented to study transfer between internal EEG sources.

The direction was scientifically useful but did not become the main current research line.

Do not restart a broad DANN search without a new transfer question.

---

## 13.2 Contrastive pretraining

Contrastive EEG pretraining was implemented using COG-BCI.

Approximately:

```text
56,903 EEG windows
```

were used in the pretraining pipeline.

The project therefore already contains self-supervised representation-learning experience.

Do not treat self-supervised EEG pretraining as an untested future idea.

---

## 13.3 Meta-learning

A First-Order Model-Agnostic Meta-Learning (FOMAML) protocol was implemented and diagnostically evaluated.

The tested configuration did not show sufficiently convincing improvement and was marked effectively as a direction not to continue under the same formulation.

Do not restart it without a materially new hypothesis.

---

# 14. Temporal alignment between EEG and PM

Temporal alignment became one of the main scientific questions.

The current canonical temporal contract is:

```text
EEG(t - 10 s) -> PM(t)
```

If the target PM window is:

```text
[T, T+10]
```

the canonical EEG window is:

```text
[T-10, T]
```

This relationship is now supported by a dedicated confirmatory experiment.

---

# 15. Confirmatory absolute-lag experiment

A confirmatory experiment was implemented after the LOW/HIGH protocol had already been established.

Absolute lags tested:

```text
-30 s
-20 s
-10 s
0 s
+10 s
```

with:

```text
-10 s = PRIMARY REFERENCE
```

and:

```text
+10 s = NON-CAUSAL DIAGNOSTIC ONLY
```

The experiment used:

```text
5 lags
× 7 PM
× 5 outer folds
= 175 model fits
```

All 175 fits completed.

A common-support strategy ensured that different lags were compared on matched available examples.

The main result was a clear average maximum around:

```text
-10 s
```

Approximate mean balanced accuracy over the seven PM:

```text
-30 s   ~0.623
-20 s   ~0.672
-10 s   ~0.751
  0 s   ~0.672
+10 s   ~0.615
```

Therefore:

```text
EEG(t-10s) -> PM(t)
```

remains the canonical project alignment.

For six PM, the advantage of `-10 s` over nearby alternatives was clear in participant-level paired analysis.

Focus showed a broader plateau over approximately:

```text
-30 ... -10 s
```

rather than a sharply unique optimum.

Do not claim that the experiment proves a universal physiological 10-second delay.

It demonstrates predictive alignment under the current PM/data-generation contract.

Canonical artifacts are under:

```text
reports/diagnostics/pm_eeg_absolute_lag_v1/
```

Experiment specification:

```text
experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json
```

---

# 16. Temporal sequence models: current result

Temporal context has already been investigated with models such as:

```text
LSTM
Transformer
```

These models did not produce a sufficiently stable advantage over strong LightGBM/XGBoost baselines.

This negative result is important.

The current question is therefore not:

> Which sequence architecture should be tried next?

The correct question is:

> Does longer EEG history contain useful information beyond the canonical 10-second block?

This motivates the current Temporal Predictability Audit.

---

# 17. Personalization

Personalization is a second major completed research line.

Approaches investigated include:

```text
subject-specific thresholds
probability calibration
model adaptation / fine-tuning
limited target-subject calibration data
```

A practically relevant scenario used approximately:

```text
30 minutes of target-participant data
```

The most stable gains came from model adaptation.

Approximate ROC-AUC improvements observed in the completed experiments were around:

```text
XGBoost   +0.016
LightGBM  +0.015
```

The exact values must be taken from the corresponding personalization report.

Personalization should not currently be expanded into another large exploratory sweep.

After a new best representation/model is selected, one confirmatory personalization experiment is appropriate.

---

# 18. Channel-level diagnostics

Channel-level experiments have included:

- searches for stable temporal lag between EEG channels;
- removal of selected channels.

No robust useful inter-channel absolute lag was established.

Removing:

```text
F3
F7
F8
```

degraded performance in the tested configurations.

Therefore arbitrary channel-drop search is considered a low-value direction.

If spatial EEG structure is revisited, prefer:

```text
phase synchronization
functional connectivity
graph-based representations
```

rather than repeatedly deleting electrodes.

---

# 19. Current scientific question

The immediate main question is:

> Does EEG history before the canonical `[T-10,T]` block contain additional predictive information about PM-defined LOW/HIGH states?

The planned history horizons are:

```text
H10
H20
H30
H60
H120
```

with:

```text
H10 = canonical reference
```

Example:

```text
PM:
[T, T+10]

H10:
[T-10, T]

H20:
[T-20, T-10]
[T-10, T]

H30:
[T-30, T-20]
[T-20, T-10]
[T-10, T]
```

and so forth.

All history must remain causal and inside one continuous recording.

---

# 20. Temporal Predictability Audit

The next experiment should first determine whether longer history is scientifically useful before another major neural architecture is introduced.

The planned audit includes several layers.

## 20.1 History-support audit

For each:

```text
target × fold × history horizon
```

measure:

```text
number of usable samples
participants
records
train/test counts
retention fraction
```

Longer histories may remove examples near the start of recordings.

This support reduction must not be confused with a modeling effect.

---

## 20.2 Autocorrelation

Measure temporal dependence of canonical EEG features at delays such as:

```text
10
20
30
60
120 seconds
```

Compute temporal statistics inside recordings first.

Do not concatenate unrelated records into artificial time series.

---

## 20.3 Permutation entropy

Permutation Entropy is planned as an exploratory nonlinear complexity measure.

It may help determine whether EEG trajectories have different temporal complexity from shuffled/surrogate trajectories.

Short sequences must be excluded when estimates are unreliable.

---

## 20.4 Mutual information

Mutual information can be used diagnostically to estimate a useful temporal delay.

This provides a data-driven complement to the manually selected H10/H20/H30/H60/H120 horizons.

The delay must not be selected by maximizing final test-set classification performance.

---

## 20.5 Takens-style delay embedding

State-space reconstruction ideas from Takens' theorem may be used diagnostically to assess temporal embedding.

False Nearest Neighbours may help estimate an effective embedding dimension.

This should be interpreted cautiously because real EEG is noisy, nonstationary, and not assumed to be a clean deterministic dynamical system.

The method is a diagnostic representation tool rather than proof of an underlying deterministic attractor.

---

## 20.6 Hankel / Singular Spectrum Analysis

Hankel trajectory matrices and Singular Spectrum Analysis (SSA) should be used to assess:

```text
temporal redundancy
singular-value spectrum
explained energy
effective rank
dominant temporal components
```

A low effective rank would suggest that long history is largely redundant.

A richer stable spectrum compared with surrogate data would support further temporal modeling.

---

## 20.7 Hurst / DFA

Long-memory diagnostics may be considered if recording sequences are sufficiently long.

Do not produce Hurst-like estimates on sequences too short for meaningful inference.

---

## 20.8 Surrogate controls

Observed temporal structure should be compared against surrogate sequences created inside the same recording.

Potential controls include:

```text
within-record permutation
block permutation
```

The goal is to destroy temporal organization while preserving relevant marginal values.

Observed-vs-surrogate comparisons should include temporal statistics such as:

```text
autocorrelation
permutation entropy
SSA structure
```

---

# 21. Predictive temporal-history experiment

If the support audit is acceptable, the first predictive test should use the existing LightGBM baseline.

No new hyperparameter search is required.

History is initially represented by concatenating feature blocks.

Example:

```text
H30 =
features@-30
+
features@-20
+
features@-10
```

Compare:

```text
H20 vs H10
H30 vs H10
H60 vs H10
H120 vs H10
```

under matched support.

Primary endpoint:

```text
participant-macro balanced accuracy
```

Secondary endpoints:

```text
macro F1
ROC-AUC
PR-AUC
```

Participant-level paired deltas and bootstrap confidence intervals should be preserved.

---

# 22. Temporal-order control

If longer history improves prediction, test whether temporal order itself matters.

For example:

```text
ordered:
-60 -50 -40 -30 -20 -10
```

versus:

```text
shuffled:
-30 -60 -20 -50 -40 -10
```

The canonical `-10` block remains fixed.

If shuffling older blocks does not reduce performance, the benefit may come from additional state observations rather than sequential ordering.

In that case, statistical aggregation/Hankel features may be more appropriate than a large sequence model.

---

# 23. Planned wavelet experiment

Wavelet representation is currently one of the highest-priority follow-up directions.

EEG is nonstationary and oscillatory.

Standard power features summarize spectral content over a window but lose information about where within the window spectral activity occurred.

Wavelet representations retain both:

```text
frequency localization
temporal localization
```

The planned clean ablation is:

```text
A. current canonical features
B. wavelet features
C. current + wavelet features
```

using the same:

```text
LOW/HIGH targets
subject folds
canonical -10 s alignment
LightGBM baseline
participant-level metrics
```

Do not simultaneously introduce a new architecture during this comparison.

---

# 24. Planned phase-synchronization experiment

The time-series course material on oscillatory systems and the Kuramoto model suggests a potentially useful EEG-specific direction.

Instead of searching for a fixed delay between electrodes, investigate **phase synchronization between EEG channels**.

For each relevant frequency band, instantaneous phase may be derived and summarized using measures such as a Kuramoto-style order parameter:

```text
R(t) = | (1/N) Σ exp(i φ_j(t)) |
```

Potential features:

```text
mean synchronization
standard deviation
median
quantiles
temporal variability
band-specific synchronization
```

This may capture coordination between channels that is invisible to simple channel deletion or absolute-lag analysis.

Interpretation must remain conservative.

Synchronization does not establish directed physiological causality.

This experiment is planned **after the main temporal-history audit and wavelet ablation** unless results motivate changing the order.

---

# 25. Possible future temporal architectures

New architectures should only be introduced after temporal information is demonstrated.

Decision logic:

```text
If H10 ≈ H20/H30/H60/H120:
    do not add a temporal neural network only for novelty.

If useful history is concentrated around H20/H30:
    consider TimesNet or PatchTST.

If strong periodic/multi-periodic temporal structure is observed:
    TimesNet becomes particularly motivated.

If H60/H120 is useful and temporal ordering matters:
    PatchTST becomes well motivated.

If very long useful memory is demonstrated:
    consider Mamba / selective State Space Models.

If history helps but ordering is unimportant:
    prefer lagged features / aggregation / SSA over a large sequence model.

If temporal information exists but effective data volume is limited:
    Echo State Network / reservoir computing may be a useful inexpensive baseline.
```

Do not run TimesNet, PatchTST, and Mamba together as a model zoo.

---

# 26. RevIN / normalization

Reversible Instance Normalization (RevIN) and related instance-normalization strategies are potentially relevant because EEG distributions vary strongly between participants and over time.

However, RevIN is not currently a standalone priority experiment.

It should be evaluated together with a justified future neural temporal architecture.

For tabular engineered-feature LightGBM experiments, it is not the primary normalization question.

---

# 27. Connectivity / causal-discovery methods

Methods such as:

```text
Granger causality
PCMCI
Convergent Cross Mapping
```

are scientifically interesting but are not current primary experiments.

They may become relevant if the project moves toward directed connectivity between EEG channels.

Important:

- predictive dependency is not equivalent to structural physiological causation;
- PM are proxy/device-derived targets;
- hidden confounding is expected.

Do not use causal terminology casually.

---

# 28. Point-process methods

Poisson/Hawkes processes from the time-series course are currently outside the main task.

The current data formulation consists primarily of regularly sampled/windowed EEG states.

Hawkes processes would become relevant only if a future project explicitly reformulates the data around discrete events such as:

```text
LOW -> HIGH transitions
stress events
attention drops
state bursts
```

Do not introduce them into the current LOW/HIGH window classification without a separate scientific motivation.

---

# 29. PM interpretation limitation

A major scientific limitation is the nature of Performance Metrics.

PM are proxy/device-derived variables rather than independent clinical ground truth.

Therefore results should be phrased as:

```text
recognition of PM-defined cognitive states
prediction of LOW/HIGH PM states from EEG
EEG–PM temporal alignment
```

rather than strong claims such as:

```text
direct measurement of cognition
proof of a physiological causal delay
EEG causes a PM change after 10 seconds
```

The internal PM-generation algorithm may contain smoothing or proprietary processing.

This is especially important when interpreting the strong `-10 s` alignment.

---

# 30. Publication-oriented scientific story

The project should converge toward a coherent research question rather than additional isolated benchmarks.

A useful current formulation is:

> What temporal information in EEG is useful for cross-subject recognition of PM-defined cognitive states, how should this information be represented, and how much can limited subject-specific adaptation improve recognition?

The current result chain is:

```text
cross-subject problem identified
        ↓
seven PM unified
        ↓
LOW/HIGH protocol established
        ↓
strong LightGBM/XGBoost baseline
        ↓
canonical EEG(t-10s) -> PM(t) confirmed
        ↓
LSTM/Transformer did not provide clear benefit
        ↓
personalization provides small stable improvement
        ↓
CURRENT:
determine useful temporal information
        ↓
wavelet representation
        ↓
phase synchronization
        ↓
one justified temporal model if needed
        ↓
external validation
```

---

# 31. External validation

External validation is considered important for a strong publication.

A useful external EEG dataset should preferably provide:

- independent participants;
- EEG;
- workload/stress/attention-like cognitive labels;
- sufficient recording metadata;
- subject-safe validation.

Exact seven-PM label compatibility is not required.

A scientifically transferable result may instead be:

```text
effective temporal horizon
benefit of wavelet features
benefit of synchronization features
value of limited personalization
```

Do not force incompatible datasets into the internal seven-PM label space.

---

# 32. Current immediate milestone

The immediate milestone is:

```text
Temporal Predictability Audit
```

It should answer:

1. How much continuous causal EEG history is actually available?
2. How quickly does dependence between EEG states decay?
3. Is observed temporal structure stronger than surrogate structure?
4. What temporal delay is suggested by mutual-information analysis?
5. What effective embedding dimension is suggested by delay-embedding diagnostics?
6. Is long history low-rank or information-rich according to Hankel/SSA?
7. Does H20/H30/H60/H120 improve classification over H10?
8. If history helps, does temporal ordering matter?

Until these questions are answered, do not start a new large neural temporal architecture experiment.

---

# 33. Planned near-term work

The rational near-term sequence is:

```text
1. Update repository instructions/context
   AGENTS.md
   PROJECT_CONTEXT.md
   related outdated instruction files

2. Temporal Predictability Audit
   support
   ACF
   permutation entropy
   mutual information
   Takens/FNN diagnostics
   Hankel/SSA
   surrogate controls
   predictive H10-H120 test if support permits

3. Wavelet feature ablation

4. Phase-synchronization / Kuramoto-style feature ablation

5. Decide whether one new temporal architecture is justified

6. Confirmatory personalization of the selected best representation/model

7. External validation

8. Freeze experimental section and proceed to manuscript preparation
```

The intermediate report to the scientific supervisor should be prepared after the Temporal Predictability Audit and preferably after the wavelet ablation, because this creates a coherent scientific checkpoint.

---

# 34. Repository state note

The last explicitly reported research branch associated with the absolute-lag implementation was:

```text
analysis/pm-eeg-absolute-lag-20260928
```

and an explicitly reported implementation HEAD was:

```text
807ca0e
```

This information is historical context only.

Before new work, always inspect the actual repository:

```powershell
Set-Location F:\EEG

git branch --show-current
git rev-parse HEAD
git status --short
```

Do not assume the recorded branch or HEAD is still current.

---

# 35. Current definition of success

The next phase should not be judged by whether every new experiment increases balanced accuracy.

A scientifically useful result may be:

### Positive temporal result

Longer history provides reliable additional information.

This motivates a dedicated temporal model.

### Negative temporal result

Most useful information is localized to the canonical 10-second block.

This explains why LSTM/Transformer did not outperform tabular baselines and argues against unnecessary architectural complexity.

### Representation result

Wavelet or synchronization features improve cross-subject recognition.

This provides a more informative EEG representation without requiring a large model.

### Personalization result

A small amount of target-user data consistently improves a subject-independent model.

This supports a practical global-model-plus-adaptation scenario.

Negative findings should be retained and reported when obtained under a valid protocol.

---

# 36. Documents to read before major new research work

At minimum inspect:

```text
AGENTS.md
PROJECT_CONTEXT.md
README.md
```

and the canonical reports/specifications directly relevant to the requested task.

For temporal work, inspect at least:

```text
reports/diagnostics/pm_eeg_absolute_lag_v1/
experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json
```

and the current LOW/HIGH target/fold implementation.

The repository and generated experiment artifacts remain the final source of truth.