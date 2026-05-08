# Windowed PM/POW dataset report

## Output files

- Parquet: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_w10.parquet`
- CSV: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_pm_dataset_w10.csv`

## Dataset summary

- Rows/windows: **51308**
- Columns: **340**
- Records: **120**
- Subjects: **55**
- Sources: `{'gpn_data': 27021, 'Old_EEG': 24287}`
- Days: `{'day1': 45979, 'day2': 5329}`

## Record processing status

| status   |   count |
|:---------|--------:|
| ok       |     120 |

## Windows by source

| source   |   windows |
|:---------|----------:|
| gpn_data |     27021 |
| Old_EEG  |     24287 |

## Windows by day

| day   |   windows |
|:------|----------:|
| day1  |     45979 |
| day2  |      5329 |

## Target columns

|                   |   count |     mean |       std |      min |      25% |      50% |      75% |      max |
|:------------------|--------:|---------:|----------:|---------:|---------:|---------:|---------:|---------:|
| target_attention  |   43175 | 0.462492 | 0.127495  | 0.028539 | 0.380783 | 0.453675 | 0.539397 | 0.925407 |
| target_engagement |   48254 | 0.618477 | 0.131689  | 0.053498 | 0.542603 | 0.633587 | 0.708927 | 0.946502 |
| target_excitement |   50983 | 0.336521 | 0.234904  | 2.4e-05  | 0.164449 | 0.264059 | 0.448988 | 1        |
| target_stress     |   45384 | 0.46215  | 0.139686  | 0.054506 | 0.384973 | 0.435071 | 0.492026 | 0.999447 |
| target_relaxation |   45394 | 0.348829 | 0.167695  | 0.075858 | 0.224523 | 0.314581 | 0.451964 | 0.697059 |
| target_interest   |   45440 | 0.517458 | 0.0977083 | 0.134833 | 0.463496 | 0.503331 | 0.551239 | 0.996604 |
| target_focus      |   45384 | 0.430048 | 0.124997  | 0.004077 | 0.346808 | 0.414493 | 0.501859 | 0.991193 |
| target_main       |   45384 | 0.430048 | 0.124997  | 0.004077 | 0.346808 | 0.414493 | 0.501859 | 0.991193 |

## Quantile label `label_q5`

|   label_q5 |   count |
|-----------:|--------:|
|          0 |    9080 |
|          1 |    9075 |
|          2 |    9075 |
|          3 |    9078 |
|          4 |    9076 |
|        nan |    5924 |

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