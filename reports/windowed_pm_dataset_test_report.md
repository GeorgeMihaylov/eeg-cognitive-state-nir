# Windowed PM/POW dataset report

## Output files

- Parquet: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_test.parquet`
- CSV: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_test.csv`

## Dataset summary

- Rows/windows: **3129**
- Columns: **340**
- Records: **3**
- Subjects: **1**
- Sources: `{'gpn_data': 3129}`
- Days: `{'day1': 3129}`

## Record processing status

| status   |   count |
|:---------|--------:|
| ok       |       3 |

## Windows by source

| source   |   windows |
|:---------|----------:|
| gpn_data |      3129 |

## Windows by day

| day   |   windows |
|:------|----------:|
| day1  |      3129 |

## Target columns

|                   |   count |     mean |      std |      min |      25% |      50% |      75% |      max |
|:------------------|--------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| target_attention  |     575 | 0.430615 | 0.107679 | 0.102595 | 0.381044 | 0.436419 | 0.482399 | 0.80636  |
| target_engagement |     582 | 0.566694 | 0.140856 | 0.251973 | 0.457022 | 0.547943 | 0.660416 | 0.943535 |
| target_excitement |     621 | 0.419906 | 0.305618 | 0.024632 | 0.168601 | 0.278956 | 0.687155 | 1        |
| target_stress     |     576 | 0.463974 | 0.156868 | 0.21928  | 0.359296 | 0.423346 | 0.519708 | 0.999447 |
| target_relaxation |     580 | 0.351821 | 0.123597 | 0.079369 | 0.261608 | 0.336801 | 0.414596 | 0.697059 |
| target_interest   |     578 | 0.501253 | 0.116434 | 0.249688 | 0.422875 | 0.485789 | 0.550032 | 0.996604 |
| target_focus      |     576 | 0.467627 | 0.18793  | 0.182429 | 0.317574 | 0.402425 | 0.599445 | 0.991193 |
| target_main       |     576 | 0.467627 | 0.18793  | 0.182429 | 0.317574 | 0.402425 | 0.599445 | 0.991193 |

## Quantile label `label_q5`

|   label_q5 |   count |
|-----------:|--------:|
|          0 |     116 |
|          1 |     115 |
|          2 |     115 |
|          3 |     115 |
|          4 |     115 |
|        nan |    2553 |

## Example columns

```text
record_id
source
subject_id
day
part
datetime_from_name
POW.AF3.Alpha__mean
POW.AF3.Alpha__std
POW.AF3.Alpha__min
POW.AF3.Alpha__max
POW.AF3.BetaH__mean
POW.AF3.BetaH__std
POW.AF3.BetaH__min
POW.AF3.BetaH__max
POW.AF3.BetaL__mean
POW.AF3.BetaL__std
POW.AF3.BetaL__min
POW.AF3.BetaL__max
POW.AF3.Gamma__mean
POW.AF3.Gamma__std
POW.AF3.Gamma__min
POW.AF3.Gamma__max
POW.AF3.Theta__mean
POW.AF3.Theta__std
POW.AF3.Theta__min
POW.AF3.Theta__max
POW.AF4.Alpha__mean
POW.AF4.Alpha__std
POW.AF4.Alpha__min
POW.AF4.Alpha__max
POW.AF4.BetaH__mean
POW.AF4.BetaH__std
POW.AF4.BetaH__min
POW.AF4.BetaH__max
POW.AF4.BetaL__mean
POW.AF4.BetaL__std
POW.AF4.BetaL__min
POW.AF4.BetaL__max
POW.AF4.Gamma__mean
POW.AF4.Gamma__std
POW.AF4.Gamma__min
POW.AF4.Gamma__max
POW.AF4.Theta__mean
POW.AF4.Theta__std
POW.AF4.Theta__min
POW.AF4.Theta__max
POW.F3.Alpha__mean
POW.F3.Alpha__std
POW.F3.Alpha__min
POW.F3.Alpha__max
POW.F3.BetaH__mean
POW.F3.BetaH__std
POW.F3.BetaH__min
POW.F3.BetaH__max
POW.F3.BetaL__mean
POW.F3.BetaL__std
POW.F3.BetaL__min
POW.F3.BetaL__max
POW.F3.Gamma__mean
POW.F3.Gamma__std
POW.F3.Gamma__min
POW.F3.Gamma__max
POW.F3.Theta__mean
POW.F3.Theta__std
POW.F3.Theta__min
POW.F3.Theta__max
POW.F4.Alpha__mean
POW.F4.Alpha__std
POW.F4.Alpha__min
POW.F4.Alpha__max
POW.F4.BetaH__mean
POW.F4.BetaH__std
POW.F4.BetaH__min
POW.F4.BetaH__max
POW.F4.BetaL__mean
POW.F4.BetaL__std
POW.F4.BetaL__min
POW.F4.BetaL__max
POW.F4.Gamma__mean
POW.F4.Gamma__std
```

## Interpretation

1. This is the first working dataset based on Emotiv POW and PM streams.
2. Raw EEG channels are not used yet; they should be added in the next stage.
3. The field `source` must be preserved for domain-specific validation.
4. The column `target_main` is based on PM.Focus when available.
5. The quantile label is preliminary and should be treated as weak labeling.