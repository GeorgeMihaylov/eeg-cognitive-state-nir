# Target pipeline resolution

## Root cause

The canonical Emotiv configuration did not explicitly define `target_col`.
The loader could therefore use the default `target_main` or fallback target
selection instead of enforcing the stored `label_q5` contract.

The loader also applied quantile discretization because `discretize: true`.
The previous generic discretizer converted missing values to class `0`.

## Corrected contract

- Five-class classification explicitly uses `target_col: label_q5`.
- Stored categorical labels use `discretize: false`.
- Explicit target columns cannot silently fall back to another target.
- Optional discretization preserves missing values.
- Already categorical integer labels are preserved instead of being passed
  through `qcut` again.
- Continuous targets can still be discretized for legacy experiments.
- Dataset metadata records the target column and discretization state.

## Expected canonical counts

- supervised rows: 45,384;
- supervised subjects: 54;
- classes: 0, 1, 2, 3 and 4.

Metrics produced by the earlier 51,302-row technical smoke are not scientific
results for `label_q5`.
