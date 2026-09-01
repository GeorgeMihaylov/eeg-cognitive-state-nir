# Auxiliary-CORN Transformer smoke experiment setup

Implementation date: 2026-07-20. Source revision: `e071904`.

This setup completes the code-preparation part of task 7V. It adds a technical
one-fold experiment for the joint categorical + auxiliary CORN Transformer. No EEG
training, outer-test inference, or lambda selection was performed in the isolated
implementation environment.

## 1. Technical protocol

The smoke experiment is fixed to:

```text
head_type: categorical_corn
feature group: EEG+POW
input shape: [8, 448]
model seed: 42
outer fold: 1
maximum epochs: 3
auxiliary weights: 0.25, 0.5, 1.0
```

The complete canonical outer fold is used because the benchmark has no safe limiter
that operates after canonical sequence construction. The expected data contract is:

```text
supervised windows: 45,384
sequences: 44,142
subjects: 53
sequence-index SHA-256:
1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa
source Parquet SHA-256:
26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8
EEG+POW feature-list SHA-256:
8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51
```

The smoke run does not select a preferred auxiliary weight. All outer-test metrics
are technical diagnostics only.

## 2. Experiment layer

The new module is:

```text
bench/experiments/auxiliary_corn_transformer.py
```

It reuses `OrdinalTransformerSmokeExperiment` for canonical sequence construction,
fold reconstruction, and standard benchmark artifact access. It introduces no
optimizer, backward pass, data loader, normalization implementation, or independent
training loop. Every trial delegates fitting and prediction to `BenchmarkRunner`.

The shared CLI dispatcher recognizes:

```yaml
experiment:
  type: auxiliary_corn_transformer_smoke
```

through the existing command-line option:

```text
--ordinal-transformer-experiment
```

## 3. Trial matrix

The resolver creates exactly three trials:

```text
categorical_corn_eeg_pow_lambda_0p25
categorical_corn_eeg_pow_lambda_0p5
categorical_corn_eeg_pow_lambda_1
```

Each resolved config includes:

```text
head_type: categorical_corn
auxiliary_weight: trial-specific value
random_state: 42
validation.random_state: 42
evaluation.random_state: 42
evaluation.folds: [1]
```

The three config hashes must be distinct, while their sequence subset hash, inner
validation split, feature order, and normalization statistics must be identical.
The canonical joint model has 358,153 trainable parameters for `[8, 448]` input at
each auxiliary weight.

## 4. Per-trial audits

After each standard benchmark run, the experiment validates:

- all total, categorical, and ordinal train/validation losses are finite;
- `total = categorical + lambda * ordinal` within numerical tolerance;
- legacy `train_loss` and `validation_loss` remain total-loss aliases;
- early stopping monitors `validation_categorical_loss`;
- the run does not exceed three epochs;
- both `classifier.*` and `auxiliary_ordinal_head.*` parameters change;
- CORN risk counts are positive and non-increasing;
- primary categorical and auxiliary CORN probabilities are finite and normalized;
- auxiliary cumulative probabilities are monotone;
- primary prediction is categorical argmax;
- auxiliary prediction is the CORN threshold count;
- categorical and auxiliary expected ranks reproduce from probabilities;
- AUC and standard metrics use primary categorical probabilities;
- auxiliary metrics are stored under `aux_*` names;
- the best checkpoint strictly reloads into a factory-built joint model;
- primary and auxiliary predictions reproduce after reload;
- outer subjects and inner logical-record groups do not overlap.

Additional audit artifacts are written beside standard fold artifacts:

```text
joint_probability_validation_summary.json
joint_checkpoint_reload_audit.json
joint_objective_audit.json
auxiliary_corn_fold_manifest.json
```

## 5. Cross-lambda audit

The combined audit requires:

- exact identity alignment for all outer-test sequences and targets;
- identical validation split JSON;
- identical normalization mean, scale, and feature order;
- one common smoke subset SHA-256;
- three distinct config hashes;
- zero primary and auxiliary checkpoint-reload mismatches.

Passing these checks sets:

```text
ready_for_nested_lambda_experiment: true
```

It does not set or imply a preferred lambda.

## 6. Configuration and generated reports

The fixed experiment configuration is:

```text
experiments/auxiliary_corn_transformer_smoke.yaml
```

A successful local run writes generated benchmark outputs under:

```text
benchmark_results/auxiliary_corn_transformer_smoke/
```

and creates the concise tracked result reports:

```text
reports/auxiliary_corn_transformer_smoke_results.md
reports/auxiliary_corn_transformer_smoke_results.json
```

The result report explicitly labels all metrics as technical and does not rank or
select the three weights.

## 7. Commands

Plan-only:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --ordinal-transformer-experiment experiments\auxiliary_corn_transformer_smoke.yaml `
  --plan-only `
  --verbose
```

Technical run:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --ordinal-transformer-experiment experiments\auxiliary_corn_transformer_smoke.yaml `
  --run `
  --resume `
  --verbose
```

## 8. Tests

New smoke-layer tests cover:

1. strict configuration loading and dispatcher routing;
2. three fixed auxiliary weights;
3. common subset hash and distinct config hashes;
4. unchanged `categorical_corn` head semantics;
5. plan-only producing no output;
6. runner delegation and resume;
7. generated report creation;
8. primary and auxiliary probability audits;
9. detection of corrupted categorical decoding;
10. detection of auxiliary monotonicity violations;
11. absence of a duplicated training loop.

New and task-7B focused tests:

```text
30 passed
```

Broader ordinal, runner, factory, checkpoint, artifact, and calibration checks:

```text
149 passed, 3 skipped
```

The repository-wide run in the isolated archive produced:

```text
340 passed, 3 skipped, 7 failed, 11 warnings
```

All seven failures require the omitted private Parquet or completed canonical
`benchmark_results`; none is caused by the new smoke layer. On the full local project,
the expected total is the previous 344 passing tests plus six new smoke tests.

## 9. Remaining task 7V work

The local EEG run must still be executed. After it completes, inspect the generated
Markdown/JSON reports and the manifest before committing. Task 7G, nested lambda
selection, full seeds/folds, and scientific comparison remain out of scope.
