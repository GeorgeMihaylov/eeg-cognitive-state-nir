# Auxiliary-CORN nested lambda-selection infrastructure

## Scope

This implementation is task 7G-1. It prepares the leakage-safe inner-validation layer required before the full auxiliary-CORN lambda experiment. It does not train joint models, select a lambda, or evaluate any candidate on an outer-test partition.

## Implemented components

### Deterministic validation reconstruction

`TorchClassificationAdapter` now exposes:

- `resolve_validation_indices(y)` to rebuild the configured deterministic inner split without fitting;
- `validation_partition_detailed(X, y, validation_indices=...)` to obtain predictions from the fitted or loaded checkpoint only on inner-validation rows.

New checkpoints persist `inner_train_indices` and `inner_validation_indices`. Legacy checkpoints remain supported because the indices can be reconstructed from the fixed record-group metadata and validation seed 42.

### Categorical baseline index

The setup reads the six already audited categorical runs from:

`reports/ordinal_transformer_multiseed_runs_summary.json`

Required references:

- EEG-only, seeds 7, 42, 123;
- EEG+POW, seeds 7, 42, 123.

Each reference is checked for:

- completed run manifest;
- categorical head;
- matching model seed;
- canonical validation, evaluation, and task split seed 42;
- matching feature group;
- five canonical outer folds;
- strict checkpoint loading.

### Baseline validation materialization

For every feature group, seed, and outer fold, the setup:

1. rebuilds the canonical outer split;
2. configures the same record-group inner-validation protocol;
3. loads the completed categorical checkpoint with `strict=True`;
4. reconstructs the exact inner-validation indices;
5. checks the selected validation groups against the saved `validation_split.json`;
6. predicts only the inner-validation rows;
7. writes validation predictions and metrics;
8. checks feature order and normalization statistics;
9. records an identity hash for cross-seed alignment.

The setup produces 30 baseline fold materializations:

`2 feature groups × 3 seeds × 5 outer folds = 30`

No optimizer or fitting call exists in the setup experiment.

### Future candidate matrix

The planned joint-model matrix remains:

`2 feature groups × 3 seeds × 5 folds × 3 lambda values = 90 fold fits`

Lambda grid:

- 0.25;
- 0.50;
- 1.00.

These models are not trained in task 7G-1.

## Selection rule

The pure selection function accepts one paired categorical validation result and exactly three joint candidate results.

For each outer fold:

1. calculate the minimum allowed balanced accuracy:
   `categorical validation BA - 0.0100`;
2. reject candidates below the threshold;
3. among eligible candidates, minimize validation severe-error rate;
4. then minimize validation ordinal MAE;
5. break exact ties with the lower lambda.

If no candidate satisfies the balanced-accuracy guard, the fold is aborted before outer-test evaluation. There is no automatic relaxation and no fallback based on outer-test results.

## Artifact schema

Runtime setup output:

`benchmark_results/auxiliary_corn_lambda_selection_setup/`

For each categorical baseline fold:

- `validation_predictions.parquet`;
- `validation_metrics.json`;
- `validation_manifest.json`.

Generated reports after the local setup run:

- `reports/auxiliary_corn_lambda_selection_setup.md`;
- `reports/auxiliary_corn_lambda_selection_setup_summary.json`.

The validation prediction table contains sequence identity, outer fold, subject, record, record group, source, target time, true class, categorical prediction, class probabilities, and categorical expected rank. It contains no outer-test rows.

## Safety properties

- Statistical tuning unit: inner-validation partition within each outer fold.
- Validation split seed: fixed at 42 for all model initialization seeds.
- Outer split seed: fixed at 42.
- Task split seed: fixed at 42.
- Outer-test use during setup: false.
- Candidate model training during setup: false.
- No eligible lambda: abort fold.
- Baseline checkpoints: strict load.
- Cross-seed validation identity: exact hash equality required for each feature group and fold.

## Tests

The new tests cover:

- exact lambda grid validation;
- BA guard;
- severe-error, ordinal-MAE, and lower-lambda ordering;
- no-eligible failure;
- six-reference baseline index;
- 30 baseline materializations and 90 future candidate fits;
- validation-only prediction frame;
- persisted and reconstructed validation indices;
- strict checkpoint-based materialization without fitting;
- cross-seed alignment;
- setup report generation;
- absence of a training loop in the setup layer.

## Remaining work

Task 7G-2 must implement the actual nested candidate training flow. For every outer fold it must:

1. train the three lambda candidates on the common inner-training partition;
2. save predictions on the common inner-validation partition;
3. apply the fixed selection rule;
4. load only the selected checkpoint;
5. evaluate only that selected model on the outer-test partition;
6. preserve the selection trace and rejected candidates without storing their outer-test predictions.

Task 7G-1 deliberately stops before this training stage.
