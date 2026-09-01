# Auxiliary-CORN nested lambda fallback finalization

## Purpose

The original nested experiment intentionally aborts a selection unit when none of the three joint `categorical_corn` candidates remains within `0.0100` balanced accuracy of its paired categorical validation baseline. The completed run produced 25 joint selections and 5 protected aborts.

Task 7G-3 adds a separate deterministic finalization stage. It does not overwrite the original incomplete report. For each protected abort it selects the already trained paired categorical Transformer as a safe fallback. The decision remains based exclusively on the original inner-validation guard; no outer-test metric is used to choose a branch.

## Final policy

- 25 units reuse the outer-test predictions of the selected joint model.
- 5 units reuse the existing outer-test predictions of the paired categorical baseline.
- 90 joint candidate artifacts are audited as completed and validation-only.
- No model fitting, checkpoint modification, or hyperparameter selection is performed.
- The source `incomplete` summary remains the audit record of the original protective protocol.

## Outputs

The finalizer writes:

- one normalized outer-test prediction file for each of 30 selection units;
- six complete policy prediction files for `feature_group × seed`;
- `selection_policy.csv` with the selected branch for every fold;
- `subject_level_analysis_input.parquet` containing all finalized policy predictions;
- a final Markdown and JSON report;
- semantic cross-policy identity audits for all five outer folds, with canonical hashes and per-column mismatch diagnostics.

Categorical fallback rows contain valid primary probabilities and expected ranks. Auxiliary CORN columns are present but null, and `aux_available=false`, so downstream analysis cannot silently interpret the categorical fallback as an auxiliary prediction.


## Outer-identity compatibility fix

The first real finalization attempt reached the six-policy comparison for outer fold 1 and stopped before report generation. The previous implementation required byte-identical hashes over parquet values. Historical categorical artifacts and newly materialized joint artifacts can encode the same semantic identity with different storage dtypes, for example integer versus floating-point `target_sample_id`, or sub-nanosecond differences in `target_time`.

The finalizer now compares identities column by column after deterministic sorting by `sequence_id`:

- `sequence_id`, subject, record, source, and split remain exact string identities;
- fold and target class are compared numerically;
- numeric and string forms of `target_sample_id` are canonicalized;
- `target_time` is compared with a strict numerical tolerance;
- a canonical semantic hash is emitted after normalization.

A real mismatch is still fatal. Before raising, the finalizer writes `cross_policy_outer_alignment_failure.json` with mismatch counts and examples, so a failed run always leaves an actionable diagnostic artifact.

## Counter correction

The source implementation counted candidate actions only inside completed selection units, which reported 75 instead of 90. The nested runner now stores candidate manifests for protected aborts and counts candidates across all outcomes. The finalizer also independently audits all 90 candidate directories.

## Protocol status

This is explicitly recorded as a post-execution safe fallback amendment. It is methodologically preferable to weakening the balanced-accuracy tolerance, selecting a rejected lambda using outer-test results, or dropping five folds from the analysis.
