# Canonical target contract

## Decision

`canonical_target_contract_ready`. This is an analysis-only integration result; no model was trained and no benchmark metric was produced.

## Registry authority

The executable contract is derived from `reports/summary/target_registry.yaml` and its provenance audit. It defines 9 executable targets and 23 registered-but-disabled candidates.

## Executable targets

Seven scalar PM regressions, the fixed-order seven-output PM regression, and the legacy global `label_q5` benchmark label are executable. The physical `label_q5` column is exposed only as `label_focus_q5_legacy` with registry status `legacy_global_benchmark_label`.

## Disabled candidates

Activity proxies, the seven-output activity multilabel target, fold-local Q3/Q5 ordinal candidates, and long-term excitement remain registered but disabled until their scientific or materialization prerequisites are approved.

## Feature target view

The shared feature view accepts `eeg`, `pow`, and `eeg_pow`, preserves source row/sample order, returns a target-specific availability mask, and excludes all `PM.*`, `target_*`, `label_*`, and identifier columns from model inputs.

## Raw target view

9 executable raw target views were validated against the existing manifest and deduplicated logical-record selection without reading or rebuilding raw window tensors. Inputs remain `[1, 14, 2560]`; scalar regression labels use `float32`, classification labels use integer dtype, and multi-output regression follows the canonical seven-target order.

## Cohort policy

Outer subject-to-fold assignments are immutable. Missing targets create target-specific complete-case cohorts inside those fixed folds; folds are never rebuilt after target filtering.

## Fold-local ordinal transforms

Q3/Q5 boundaries are fit only on finite outer-train values, then applied unchanged to train, validation, and outer-test partitions. Duplicate boundaries are reported through the actual class count; there is no global fallback and no derived column is materialized.

## Legacy aliases

`label_q5` maps to `label_focus_q5_legacy`; `target_focus` maps to `pm_focus_regression`; `target_main` is a warned legacy alias for focus and is never an implicit fallback.

## Task integration

The existing task registry now exposes explicit PM scalar, PM multi-output, and legacy Q5 task IDs while preserving `focus_regression`, `performance_metrics_regression`, and `cognitive_load_5class` compatibility.

## Metrics contract

Regression targets recommend MAE, RMSE, R², and Spearman correlation. Classification targets recommend accuracy, balanced accuracy, macro/weighted F1, kappa, and applicable ordinal/AUC metrics. These are contracts only, not new results.

## Leakage controls

Target values and metadata never enter features. Target transforms fit on outer-train only. Raw attachment validates sample, subject, and record identifiers. Target missingness is filtered rather than zero-filled.

## Artifacts

Deterministic machine-readable artifacts are under `reports/summary/target_contract/`. CSV files are ignored by the repository-wide `*.csv` rule and require explicit force-add only if a future commit is requested.

## Compatibility

Legacy feature/raw label-Q5 configurations, scalar focus regression, seven-output PM regression, and config-only FOMAML/DANN paths remain loadable. Compatibility aliases emit explicit warnings.

## Limitations

Fold-local ordinal candidates remain disabled as benchmark tasks, activity semantics remain unapproved, and long-term excitement is not materialized in the processed table. Full test-suite status is recorded separately after generation.
