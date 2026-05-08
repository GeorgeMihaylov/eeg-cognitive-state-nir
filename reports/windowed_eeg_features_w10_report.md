# EEG feature dataset report

## Output files

- eeg_features_parquet: `D:\PycharmProjects\eeg-cognitive-state-nir\data\interim\windowed_eeg_features_w10.parquet`
- eeg_features_csv: `D:\PycharmProjects\eeg-cognitive-state-nir\data\interim\windowed_eeg_features_w10.csv`
- record_report_csv: `D:\PycharmProjects\eeg-cognitive-state-nir\data\interim\windowed_eeg_features_w10_record_report.csv`
- merged_parquet: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_eeg_pm_dataset_w10.parquet`
- merged_csv: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\windowed_eeg_pm_dataset_w10.csv`
- eeg_missing_csv: `D:\PycharmProjects\eeg-cognitive-state-nir\data\interim\windowed_eeg_features_w10_missingness.csv`
- report_md: `D:\PycharmProjects\eeg-cognitive-state-nir\reports\windowed_eeg_features_w10_report.md`

## Parameters

- window_s: **10.0**
- EEG channels: `EEG.AF3, EEG.F7, EEG.F3, EEG.FC5, EEG.T7, EEG.P7, EEG.O1, EEG.O2, EEG.P8, EEG.T8, EEG.FC6, EEG.F4, EEG.F8, EEG.AF4`

## Record processing status

| status   |   count |
|:---------|--------:|
| ok       |     120 |

## EEG feature table summary

- Rows/windows: **51308**
- Columns: **177**
- Records: **120**
- Subjects: **55**
- Sources: `{'gpn_data': 27021, 'Old_EEG': 24287}`

## Merged PM/POW + EEG dataset summary

- Rows/windows: **51308**
- Columns: **508**
- Records: **120**
- Subjects: **55**
- Sources: `{'gpn_data': 27021, 'Old_EEG': 24287}`

## EEG feature columns

- EEG feature columns in merged dataset: **168**

## EEG missingness preview

| column                      |   missing_count |   missing_ratio |   non_null_count | dtype   |
|:----------------------------|----------------:|----------------:|-----------------:|:--------|
| EEG.AF3__mean               |               0 |               0 |            51308 | float64 |
| EEG.AF3__std                |               0 |               0 |            51308 | float64 |
| EEG.AF3__min                |               0 |               0 |            51308 | float64 |
| EEG.AF3__max                |               0 |               0 |            51308 | float64 |
| EEG.AF3__median             |               0 |               0 |            51308 | float64 |
| EEG.AF3__robust_iqr         |               0 |               0 |            51308 | float64 |
| EEG.AF3__skew               |               0 |               0 |            51308 | float64 |
| EEG.AF3__kurt               |               0 |               0 |            51308 | float64 |
| EEG.AF3__signal_energy      |               0 |               0 |            51308 | float64 |
| EEG.AF3__zero_crossing_rate |               0 |               0 |            51308 | float64 |
| EEG.AF3__line_length        |               0 |               0 |            51308 | float64 |
| EEG.AF3__mean_abs_diff      |               0 |               0 |            51308 | float64 |
| EEG.F7__mean                |               0 |               0 |            51308 | float64 |
| EEG.F7__std                 |               0 |               0 |            51308 | float64 |
| EEG.F7__min                 |               0 |               0 |            51308 | float64 |
| EEG.F7__max                 |               0 |               0 |            51308 | float64 |
| EEG.F7__median              |               0 |               0 |            51308 | float64 |
| EEG.F7__robust_iqr          |               0 |               0 |            51308 | float64 |
| EEG.F7__skew                |               0 |               0 |            51308 | float64 |
| EEG.F7__kurt                |               0 |               0 |            51308 | float64 |
| EEG.F7__signal_energy       |               0 |               0 |            51308 | float64 |
| EEG.F7__zero_crossing_rate  |               0 |               0 |            51308 | float64 |
| EEG.F7__line_length         |               0 |               0 |            51308 | float64 |
| EEG.F7__mean_abs_diff       |               0 |               0 |            51308 | float64 |
| EEG.F3__mean                |               0 |               0 |            51308 | float64 |
| EEG.F3__std                 |               0 |               0 |            51308 | float64 |
| EEG.F3__min                 |               0 |               0 |            51308 | float64 |
| EEG.F3__max                 |               0 |               0 |            51308 | float64 |
| EEG.F3__median              |               0 |               0 |            51308 | float64 |
| EEG.F3__robust_iqr          |               0 |               0 |            51308 | float64 |
| EEG.F3__skew                |               0 |               0 |            51308 | float64 |
| EEG.F3__kurt                |               0 |               0 |            51308 | float64 |
| EEG.F3__signal_energy       |               0 |               0 |            51308 | float64 |
| EEG.F3__zero_crossing_rate  |               0 |               0 |            51308 | float64 |
| EEG.F3__line_length         |               0 |               0 |            51308 | float64 |
| EEG.F3__mean_abs_diff       |               0 |               0 |            51308 | float64 |
| EEG.FC5__mean               |               0 |               0 |            51308 | float64 |
| EEG.FC5__std                |               0 |               0 |            51308 | float64 |
| EEG.FC5__min                |               0 |               0 |            51308 | float64 |
| EEG.FC5__max                |               0 |               0 |            51308 | float64 |

## Record report preview

| record_id                                                 | source   | subject_id   | day   | status   |   n_input_rows |   n_output_windows |   duration_s | missing_eeg_channels   |
|:----------------------------------------------------------|:---------|:-------------|:------|:---------|---------------:|-------------------:|-------------:|:-----------------------|
| Old_EEG__0001508a__day1____2023.12.18T14.22.47p03.00      | Old_EEG  | 0001508a     | day1  | ok       |         856562 |                336 |     3344.08  | []                     |
| Old_EEG__0012905a__day1__2part__2023.11.28T12.51.55p03.00 | Old_EEG  | 0012905a     | day1  | ok       |         516588 |                203 |     2016.79  | []                     |
| Old_EEG__0012905a__day1__3part__2023.11.28T13.15.06p03.00 | Old_EEG  | 0012905a     | day1  | ok       |         161186 |                 64 |      629.278 | []                     |
| Old_EEG__0012905a__day1____2023.11.28T11.47.42p03.00      | Old_EEG  | 0012905a     | day1  | ok       |         924360 |                362 |     3608.76  | []                     |
| Old_EEG__007291c7__day1____2023.12.22T13.45.44p03.00      | Old_EEG  | 007291c7     | day1  | ok       |         136426 |                 54 |      532.613 | []                     |
| Old_EEG__01c2a0d8__day1____2024.01.12T19.11.49p03.00      | Old_EEG  | 01c2a0d8     | day1  | ok       |        2046217 |                800 |     7988.58  | []                     |
| Old_EEG__1081b177__day1____2023.12.08T11.39.57p03.00      | Old_EEG  | 1081b177     | day1  | ok       |        1225055 |                480 |     4782.77  | []                     |
| Old_EEG__20201194__day1____2023.12.03T22.02.18p03.00      | Old_EEG  | 20201194     | day1  | ok       |        1137168 |                445 |     4439.58  | []                     |
| Old_EEG__2162c09e__day1____2023.12.18T12.15.08p03.00      | Old_EEG  | 2162c09e     | day1  | ok       |        1542604 |                604 |     6022.82  | []                     |
| Old_EEG__2182c1cd__day1__part2__2023.11.28T13.51.38p03.00 | Old_EEG  | 2182c1cd     | day1  | ok       |        1128588 |                442 |     4406.15  | []                     |
| Old_EEG__219060fa__day1____2023.12.15T16.47.17p03.00      | Old_EEG  | 219060fa     | day1  | ok       |        1222360 |                479 |     4772.18  | []                     |
| Old_EEG__21a031f6__day1____2024.01.12T16.19.41p03.00      | Old_EEG  | 21a031f6     | day1  | ok       |        1837512 |                718 |     7173.78  | []                     |
| Old_EEG__30c140ca__day1____2023.12.18T09.42.40p03.00      | Old_EEG  | 30c140ca     | day1  | ok       |         971368 |                380 |     3792.29  | []                     |
| Old_EEG__3110e0c7__day1____2024.01.29T09.42.53p01.00      | Old_EEG  | 3110e0c7     | day1  | ok       |        1752665 |                685 |     6842.53  | []                     |
| Old_EEG__40009139__day1____2023.12.04T13.23.34p03.00      | Old_EEG  | 40009139     | day1  | ok       |         752865 |                295 |     2939.23  | []                     |
| Old_EEG__41e2010c__day1____2023.11.29T17.22.52p03.00      | Old_EEG  | 41e2010c     | day1  | ok       |         874009 |                342 |     3412.19  | []                     |
| Old_EEG__50c02189__day1____2023.11.27T11.56.06p03.00      | Old_EEG  | 50c02189     | day1  | ok       |         814991 |                638 |     6365.31  | []                     |
| Old_EEG__517001af__day1____2023.11.27T13.40.34p03.00      | Old_EEG  | 517001af     | day1  | ok       |        1581026 |                618 |     6172.51  | []                     |
| Old_EEG__6030f0fd__day1____2023.12.05T15.32.40p03.00      | Old_EEG  | 6030f0fd     | day1  | ok       |         895988 |                350 |     3498     | []                     |
| Old_EEG__7072a0e0__day1____2023.12.11T13.28.50p03.00      | Old_EEG  | 7072a0e0     | day1  | ok       |        1396969 |                546 |     5453.87  | []                     |
| Old_EEG__7092f07b__day1____2024.01.15T09.47.46p03.00      | Old_EEG  | 7092f07b     | day1  | ok       |        1208166 |                473 |     4716.77  | []                     |
| Old_EEG__7150e10a__day1____2023.12.09T09.37.19p03.00      | Old_EEG  | 7150e10a     | day1  | ok       |        1765903 |                691 |     6895.94  | []                     |
| Old_EEG__71c09041__day1____2023.12.06T13.27.03p03.00      | Old_EEG  | 71c09041     | day1  | ok       |        1496367 |                585 |     5843.88  | []                     |
| Old_EEG__71e10186__day1____2023.12.05T11.06.22p03.00      | Old_EEG  | 71e10186     | day1  | ok       |        1490070 |                583 |     5817.33  | []                     |
| Old_EEG__71f0603f__day1____2023.11.30T17.41.49p03.00      | Old_EEG  | 71f0603f     | day1  | ok       |         973564 |                382 |     3801.35  | []                     |
| Old_EEG__71f0603f__day1____2023.11.28T09.25.58p03.00      | Old_EEG  | 71f0603f     | day1  | ok       |        1676829 |                656 |     6546.45  | []                     |
| Old_EEG__71f21142__day1____2024.01.23T17.38.50p03.00      | Old_EEG  | 71f21142     | day1  | ok       |        1469075 |                574 |     5735.37  | []                     |
| Old_EEG__8030618f__day1____2023.12.11T11.51.47p03.00      | Old_EEG  | 8030618f     | day1  | ok       |        1168433 |                457 |     4561.65  | []                     |
| Old_EEG__8191f1d9__day1____2023.12.22T16.25.50p03.00      | Old_EEG  | 8191f1d9     | day1  | ok       |        1970269 |                770 |     7692.07  | []                     |
| Old_EEG__81e150c1__day1____2023.11.29T11.58.41p03.00      | Old_EEG  | 81e150c1     | day1  | ok       |        1000489 |                391 |     3905.98  | []                     |
| Old_EEG__81f1f0fe__day1____2023.11.27T09.37.48p03.00      | Old_EEG  | 81f1f0fe     | day1  | ok       |        1494165 |                585 |     5833.4   | []                     |
| Old_EEG__9192c107__day1____2023.12.13T15.29.18p03.00      | Old_EEG  | 9192c107     | day1  | ok       |        1959837 |                766 |     7651.35  | []                     |
| Old_EEG__a02151ac__day1____2023.11.29T14.45.43p03.00      | Old_EEG  | a02151ac     | day1  | ok       |        1169087 |                457 |     4564.19  | []                     |
| Old_EEG__a1721173__day1____2023.11.30T14.11.21p03.00      | Old_EEG  | a1721173     | day1  | ok       |          75529 |                 30 |      294.866 | []                     |
| Old_EEG__a1721173__day1__part2__2023.11.30T14.18.22p03.00 | Old_EEG  | a1721173     | day1  | ok       |         686015 |                269 |     2678.25  | []                     |
| Old_EEG__a1b210fc__day1____2023.12.15T13.50.40p03.00      | Old_EEG  | a1b210fc     | day1  | ok       |        1537968 |                601 |     6004.34  | []                     |
| Old_EEG__b112005d__day1____2024.02.02T12.13.43p01.00      | Old_EEG  | b112005d     | day1  | ok       |        1887529 |                738 |     7369.05  | []                     |
| Old_EEG__b1c2f044__day1____2023.12.27T17.13.43p03.00      | Old_EEG  | b1c2f044     | day1  | ok       |        1414397 |                553 |     5521.91  | []                     |
| Old_EEG__c060c06a__day1____2023.11.29T09.22.25p03.00      | Old_EEG  | c060c06a     | day1  | ok       |         969628 |                380 |     3785.49  | []                     |
| Old_EEG__c1a150b1__day1____2024.02.01T16.29.53p01.00      | Old_EEG  | c1a150b1     | day1  | ok       |        1665518 |                651 |     6502.56  | []                     |

## Interpretation

1. This dataset extends the previous PM/POW windowed dataset with raw EEG-derived statistical features.
2. The EEG features are computed on the same 10-second windows as PM/POW features.
3. The current EEG features are time-domain statistics. Spectral features from raw EEG should be added in a later stage.
4. The first comparison should check whether adding EEG features improves GroupKFold and cross-source no-overlap metrics.
5. PM-derived columns must still be excluded from model features to avoid target leakage.