# Auxiliary-CORN nested lambda selection setup

The six categorical baselines were loaded from completed checkpoints. Their deterministic inner-validation partitions were reconstructed without fitting and without using outer-test predictions.

- Baseline folds materialized: 30.
- Future candidate fold fits: 90.
- Lambda grid: [0.25, 0.5, 1.0].
- BA tolerance: 0.0100 absolute.
- No-eligible action: abort the fold before outer-test evaluation.
- Cross-seed inner-validation identity alignment: exact.

## Selection rule

1. Reject candidates below categorical validation BA minus 0.0100.
2. Minimize validation severe-error rate.
3. Minimize validation ordinal MAE.
4. Break exact ties with the lower lambda.

No lambda has been selected and no joint model has been trained in this setup task.
