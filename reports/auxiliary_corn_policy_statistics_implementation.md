# Auxiliary-CORN finalized-policy subject analysis: implementation report

## Scope

Task 7Г-4 adds the final subject-level comparison of three already completed Transformer systems:

1. paired categorical Transformer;
2. paired pure CORN Transformer;
3. finalized selective policy with 25 jointly trained categorical+CORN units and 5 categorical fallbacks.

No model fitting, checkpoint modification, lambda selection, or fallback selection is performed by this analysis.

## Statistical unit

The independent inferential unit is `subject_id`. Predictions are first evaluated separately for each seed (7, 42, 123), then subject metrics are averaged across the three repeated model initializations. Paired inference therefore uses 53 subject-level observations rather than folds, windows, sequences, or seeds as independent samples.

## Input audit

For every combination of feature group and seed, the analysis requires exact outer-test alignment among categorical, pure CORN, and finalized-policy predictions on:

- `sequence_id`;
- outer fold;
- `subject_id`;
- `record_id`;
- source;
- `y_true`.

The expected dimensions are 44,142 sequences and 53 subjects for each of six `feature group × seed` policy models.

## Outputs

The implementation writes:

- `subject_seed_metrics.parquet` — 954 subject × method × seed rows;
- `subject_multiseed_metrics.parquet` — 318 subject × method rows after seed averaging;
- `paired_comparisons.parquet` — bootstrap, Wilcoxon, sign-test, effect-size and Holm-adjusted comparisons;
- `seed_consistency.parquet`;
- `subject_effects.parquet`;
- `fallback_units.csv`;
- `disagreement_analysis.json`;
- `decision.json`;
- Markdown and JSON reports under `reports/`.

## Confirmatory comparisons

Primary comparisons evaluate finalized policy versus paired categorical Transformer separately for EEG-only and EEG+POW:

- ordinal MAE;
- severe-error rate;
- balanced accuracy.

Secondary comparisons include macro F1, quadratic weighted kappa, adjacent accuracy, expected-rank MAE and expected-rank Spearman correlation. The policy is also compared with pure CORN on the five headline metrics.

## Heterogeneity analyses

The module reports:

- hard-subject results for the lowest categorical balanced-accuracy quartile;
- remaining-subject results;
- subject-level trade-offs between ordinal gains and balanced-accuracy losses;
- fallback exposure after averaging repeated seeds;
- descriptive Spearman associations between categorical/auxiliary-head disagreement and policy gains.

## Decision rule

For the primary EEG+POW group, the policy is classified as:

- `supported_with_ba_guard` when at least one ordinal primary metric has a positive bootstrap confidence interval and mean balanced-accuracy change is not below -0.01;
- `ordinal_gain_ba_tradeoff` when ordinal support is present but the mean BA guard is violated;
- `not_supported` otherwise.

This outer-test BA guard is descriptive. All actual lambda and fallback branch decisions remain based only on inner validation.

## Validation

Focused synthetic end-to-end tests validate:

- exact three-way identity alignment;
- rejection of corrupted subject identity;
- 954 subject-seed rows;
- 318 seed-averaged subject rows;
- full artifact generation;
- fallback accounting;
- primary hypothesis table construction.

Targeted and related tests completed successfully. Environment-dependent legacy tests requiring unpublished local `benchmark_results` are not executable from the source-only archive.
