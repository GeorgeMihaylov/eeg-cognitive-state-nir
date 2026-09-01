# Preprocessing ablation cache validation

Validation was completed before the full factorial run. All eight cache variants
are semantically reusable and contain the same supervised sample universe.

## Validation result

| Trial | Pipeline | JSON/NPY | Size (bytes) | Rows | Accepted | Deduplicated | Subjects | Identity vs A | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| A | raw | 119/119 | 6,511,440,533 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| B | band-pass | 119/119 | 6,510,749,713 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| C | notch | 119/119 | 6,511,352,202 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| D | CAR | 119/119 | 6,510,883,924 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| E | band-pass + notch | 119/119 | 6,510,749,723 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| F | band-pass + CAR | 119/119 | 6,510,753,517 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| G | notch + CAR | 119/119 | 6,510,764,599 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |
| H | band-pass + notch + CAR | 119/119 | 6,510,754,301 | 45,384 | 45,326 | 30,958 | 54 | exact | pass |

Every NPY shard was opened with memory mapping and every value was checked.
All arrays have trailing shape `[14, 2560]`, dtype `float32`, and contain only
finite values. JSON and NPY stem sets are identical for every cache.

Semantic validation recomputed the established preprocessing hash and each
record-level cache configuration hash from the source-file identity, channel
order, sampling rate, missing-data threshold, preprocessing parameters, and
the indexed windows. The accepted sample IDs in shard metadata match the
corresponding index exactly.

The complete indices match trial A on `sample_id`, `subject_id`, `record_id`,
`record_group_id`, `outer_fold`, `label_q5`, and `status`. Each index contains
58 rejected rows in addition to 45,326 accepted rows. Accepted data cover all
54 subjects and all five outer folds; selection through the fixed logical
recording map leaves 30,958 windows.

The machine-readable result is in
`reports/preprocessing_ablation_cache_validation.csv`.
