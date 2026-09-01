# Temporal label structure

Sequences are formed strictly within `source + subject_id + record_id` and ordered by `t_start, sample_id`. Positive lags use only earlier windows.

## Target autocorrelation

| Lag | Pairs | Pooled Pearson autocorrelation |
| --- | ---: | ---: |
| 1 | 45183 | 0.881551 |
| 2 | 44994 | 0.769071 |
| 3 | 44810 | 0.684817 |
| 5 | 44464 | 0.556229 |
| 10 | 43673 | 0.441127 |
| 20 | 42158 | 0.422511 |

## Adjacent target change

Mean absolute change is `0.045170`; median `0.033705`, 95th percentile `0.126743`.

## Class stability

- Same next class: 58.4047%
- Adjacent-class transition: 34.4798%
- Two-or-more-class transition: 7.1155%
- Mean run length: 2.389 windows
- Median run length: 1.000 windows
- 95th percentile run length: 7.000 windows
- Mean / median duration: 23.893 / 10.000 seconds

| Source | Pairs | Same class | Adjacent class | Two or more classes |
| --- | ---: | ---: | ---: | ---: |
| Old_EEG | 21463 | 58.4028% | 34.1099% | 7.4873% |
| gpn_data | 23720 | 58.4064% | 34.8145% | 6.7791% |

Transition probabilities, with rows as previous class and columns as next class:

| Previous \ Next | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7123 | 0.2085 | 0.0589 | 0.0197 | 0.0006 |
| 1 | 0.2197 | 0.4877 | 0.2126 | 0.0693 | 0.0107 |
| 2 | 0.0535 | 0.2420 | 0.4583 | 0.2068 | 0.0394 |
| 3 | 0.0127 | 0.0578 | 0.2421 | 0.5005 | 0.1869 |
| 4 | 0.0003 | 0.0042 | 0.0284 | 0.2044 | 0.7626 |

## Previous-label diagnostic

This is a structural diagnostic using the true preceding label, not a deployable model. First windows of every record and pairs crossing missing target windows are excluded.

`n_samples | accuracy | balanced_accuracy | macro_f1 | ordinal_mae | adjacent_accuracy | severe_error_rate`

45183 | 0.584047 | 0.584250 | 0.584269 | 0.496935 | 0.928845 | 0.071155

## Blocked-time check

- `blocked_time_cross_gap`: 109 | 0.311927 | 0.296639 | 0.287509 | 1.321101 | 0.605505 | 0.394495
- `blocked_time_early_adjacent`: 19390 | 0.613976 | 0.622569 | 0.622529 | 0.438834 | 0.951521 | 0.048479
- `blocked_time_late_adjacent`: 16794 | 0.560200 | 0.543385 | 0.543527 | 0.548589 | 0.906514 | 0.093486

Close-neighbor early/late results and the cross-gap bridge are descriptive checks. Every subject/record is an outer-test observation in exactly one canonical fold. The cross-gap rule uses the last early true label to predict only the first late label of each eligible record.

## Interpretation risk

The high adjacent-window autocorrelation and class persistence mean that a sequence model can obtain apparent benefit from local smoothness, record position, or access to correlated neighbouring windows. The previous-label result is an upper diagnostic that uses unavailable true history, not a fair competitor. Subject GroupKFold prevents subject identity leakage, but claims about temporal decoding should additionally report blocked or forward-time checks and must not attribute all sequential gain to EEG physiology.
