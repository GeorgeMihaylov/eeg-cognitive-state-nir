# Windowed PM/POW dataset report

## Output files

- Parquet: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_test_w10.parquet`
- CSV: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_test_w10.csv`

## Dataset summary

- Rows/windows: **1029**
- Columns: **340**
- Records: **5**
- Subjects: **2**
- Sources: `{'gpn_data': 1029}`
- Days: `{'day1': 629, 'day2': 400}`

## Record processing status

| status   |   count |
|:---------|--------:|
| ok       |       5 |

## Windows by source

| source   |   windows |
|:---------|----------:|
| gpn_data |      1029 |

## Windows by day

| day   |   windows |
|:------|----------:|
| day1  |       629 |
| day2  |       400 |

## Target columns

|                   |   count |     mean |      std |      min |      25% |      50% |      75% |      max |
|:------------------|--------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| target_attention  |     971 | 0.440572 | 0.103307 | 0.102595 | 0.386028 | 0.444973 | 0.494532 | 0.80636  |
| target_engagement |     978 | 0.605643 | 0.149195 | 0.15024  | 0.489681 | 0.618209 | 0.716716 | 0.943535 |
| target_excitement |    1017 | 0.385445 | 0.268074 | 0.008618 | 0.178764 | 0.271978 | 0.523297 | 1        |
| target_stress     |     972 | 0.459612 | 0.139171 | 0.21928  | 0.368804 | 0.426929 | 0.507538 | 0.999447 |
| target_relaxation |     976 | 0.369733 | 0.13553  | 0.075858 | 0.270014 | 0.348041 | 0.450048 | 0.697059 |
| target_interest   |     974 | 0.512155 | 0.106023 | 0.249688 | 0.443061 | 0.49874  | 0.564044 | 0.996604 |
| target_focus      |     972 | 0.441287 | 0.161036 | 0.182429 | 0.32692  | 0.396756 | 0.499201 | 0.991193 |
| target_main       |     972 | 0.441287 | 0.161036 | 0.182429 | 0.32692  | 0.396756 | 0.499201 | 0.991193 |

## Quantile label `label_q5`

|   label_q5 |   count |
|-----------:|--------:|
|          0 |     195 |
|          1 |     194 |
|          2 |     194 |
|          3 |     194 |
|          4 |     195 |
|        nan |      57 |

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