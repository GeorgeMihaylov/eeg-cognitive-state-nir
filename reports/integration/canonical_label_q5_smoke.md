# Canonical label_q5 smoke validation

## Configuration

- Dataset: `emotiv_cognitive`
- Target: `label_q5`
- Features: EEG + POW
- Feature count: 448
- Evaluation: subject-level GroupKFold
- Executed folds: 1
- Model: Random Forest

## Dataset validation

- Samples: 45,384
- Subjects: 54
- Classes: 5
- Train samples: 36,261
- Test samples: 9,123
- Train subjects: 43
- Test subjects: 11

## Fold 1 metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.3571 |
| Balanced accuracy | 0.3631 |
| Macro F1 | 0.3441 |
| Weighted F1 | 0.3448 |
| Cohen kappa | 0.2007 |
| AUC | 0.6724 |
| Ordinal MAE | 1.1577 |
| Adjacent accuracy | 0.6705 |
| Severe error rate | 0.3295 |

## Conclusion

The corrected canonical target contract was validated end to end.

The benchmark now explicitly uses the stored `label_q5` target, excludes
unlabeled rows, retains 54 supervised subjects and evaluates five classes.

The earlier 51,302-row smoke run is not a valid result for `label_q5`.
This single-fold run is a technical validation and not a final scientific
benchmark.
