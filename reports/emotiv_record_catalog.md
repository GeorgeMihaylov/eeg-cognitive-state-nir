# Emotiv record catalog

Корень проекта: `D:\PycharmProjects\eeg-cognitive-state-nir`

## Общая сводка

- Всего записей: **120**
- Уникальных субъектов: **55**
- Суммарный размер main-файлов: **38152.69 MB**

## Статусы

| status   |   count |
|:---------|--------:|
| ok       |     120 |

## Источники

| source   |   count |
|:---------|--------:|
| gpn_data |      71 |
| Old_EEG  |      49 |

## Дни

| day   |   count |
|:------|--------:|
| day1  |      94 |
| day2  |      26 |

## Наличие companion-файлов

|       |   has_json |   has_marker |
|:------|-----------:|-------------:|
| True  |         71 |           70 |
| False |         49 |           50 |

## Количество колонок по потокам

|                         |   count |    mean |     std |   min |   25% |   50% |   75% |   max |
|:------------------------|--------:|--------:|--------:|------:|------:|------:|------:|------:|
| n_cols                  |     120 | 182.975 | 0.15678 |   182 |   183 |   183 |   183 |   183 |
| time_columns_count      |     120 |   1.975 | 0.15678 |     1 |     2 |     2 |     2 |     2 |
| eeg_columns_count       |     120 |  20     | 0       |    20 |    20 |    20 |    20 |    20 |
| pm_columns_count        |     120 |  36     | 0       |    36 |    36 |    36 |    36 |    36 |
| bandpower_columns_count |     120 |  70     | 0       |    70 |    70 |    70 |    70 |    70 |
| motion_columns_count    |     120 |  15     | 0       |    15 |    15 |    15 |    15 |    15 |
| facial_columns_count    |     120 |   6     | 0       |     6 |     6 |     6 |     6 |     6 |

## Первые 40 записей каталога

| source   | subject_id   | day   | part   | datetime_from_name        |   size_mb | has_json   | has_marker   |   n_cols |   eeg_columns_count |   pm_columns_count |   bandpower_columns_count |   motion_columns_count |   facial_columns_count | main_rel_path                                                                                                         |
|:---------|:-------------|:------|:-------|:--------------------------|----------:|:-----------|:-------------|---------:|--------------------:|-------------------:|--------------------------:|-----------------------:|-----------------------:|:----------------------------------------------------------------------------------------------------------------------|
| gpn_data | 0012905a     | day1  | 2part  | 2023.11.28T12.51.55+03.00 | 24.843    | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0012905a\day1\eeg\0012905a_1day_2part_EPOCX_202378_2023.11.28T12.51.55+03.00.md.mc.pm.fe.bp.csv.bz2 |
| gpn_data | 0012905a     | day1  | 3part  | 2023.11.28T13.15.06+03.00 |  7.87399  | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0012905a\day1\eeg\0012905a_1day_3part_EPOCX_202378_2023.11.28T13.15.06+03.00.md.mc.pm.fe.bp.csv.bz2 |
| gpn_data | 0012905a     | day1  | nan    | 2023.11.28T11.47.42+03.00 | 42.9466   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0012905a\day1\eeg\0012905a_1day_EPOCX_202378_2023.11.28T11.47.42+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 0012905a     | day2  | nan    | 2023.12.06T12.12.20+03.00 | 26.1469   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0012905a\day2\eeg\0012905a_2day_EPOCX_202378_2023.12.06T12.12.20+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 0110f12e     | day2  | nan    | 2023.12.12T17.41.19+03.00 | 22.6019   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0110f12e\day2\eeg\0110f12e_2day_EPOCX_202378_2023.12.12T17.41.19+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 0110f12e     | day1  | nan    | 2023.12.04T18.03.21+03.00 | 59.3793   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0110f12e\eeg\0110f12e_1day_EPOCX_202378_2023.12.04T18.03.21+03.00.md.mc.pm.fe.bp.csv.bz2            |
| gpn_data | 0182e16c     | day1  | 2part  | 2023.11.27T15.11.45+03.00 |  0.516613 | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0182e16c\day1\0182e16c_1day_2part_EPOCX_202378_2023.11.27T15.11.45+03.00.md.mc.pm.fe.bp.csv.bz2     |
| gpn_data | 0182e16c     | day1  | nan    | 2023.11.27T13.58.33+03.00 | 20.9191   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0182e16c\day1\0182e16c_1day_EPOCX_202378_2023.11.27T13.58.33+03.00.md.mc.pm.fe.bp.csv.bz2           |
| gpn_data | 0182e16c     | day2  | nan    | 2023.12.14T13.40.35+03.00 | 24.8877   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\0182e16c\day2\eeg\0182e16c_2day_EPOCX_202378_2023.12.14T13.40.35+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 01c2a0d8     | day1  | nan    | 2024.01.12T19.11.49+03.00 | 94.6834   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\01c2a0d8\day1\eeg\01c2a0d8_1day_EPOCX_202378_2024.01.12T19.11.49+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 1081b177     | day1  | nan    | 2023.12.08T11.39.57+03.00 | 59.2715   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\1081b177\day1\eeg\1081b177_1day_EPOCX_202449_2023.12.08T11.39.57+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 1081b177     | day2  | nan    | 2023.12.13T13.16.31+03.00 | 27.6351   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\1081b177\day2\eeg\1081b177_2day_EPOCX_202378_2023.12.13T13.16.31+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 20201194     | day1  | nan    | 2023.12.03T22.02.18+03.00 | 54.3783   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\20201194\day1\eeg\20201194_1day_EPOCX_202378_2023.12.03T22.02.18+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 2162c09e     | day1  | nan    | 2023.12.18T12.15.08+03.00 | 75.3644   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\2162c09e\day1\eeg\2162c09e_1day_EPOCX_202378_2023.12.18T12.15.08+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 2182c1cd     | day1  | part2  | 2023.11.28T13.51.38+03.00 | 55.4796   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\2182c1cd\day1\eeg\2182c1cd_day1_part2_EPOCX_202449_2023.11.28T13.51.38+03.00.md.mc.pm.fe.bp.csv.bz2 |
| gpn_data | 2182c1cd     | day2  | nan    | 2023.12.06T11.28.11+03.00 | 17.7541   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\2182c1cd\day2\eeg\2182c1cd_2day_EPOCX_202378_2023.12.06T11.28.11+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 219060fa     | day1  | nan    | 2023.12.15T16.47.17+03.00 | 58.9479   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\219060fa\day1\eeg\219060fa_1day_EPOCX_202378_2023.12.15T16.47.17+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 30908049     | day1  | nan    | 2023.12.04T11.14.04+03.00 | 57.5792   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\30908049\day1\eeg\30908049_1day_EPOCX_202378_2023.12.04T11.14.04+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 30908049     | day2  | nan    | 2023.12.10T11.03.58+03.00 | 29.5657   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\30908049\day2\eeg\30908049_2day_EPOCX_202378_2023.12.10T11.03.58+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 3110e0c7     | day1  | nan    | 2024.01.29T09.42.53+01.00 | 81.4257   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\3110e0c7\day1\eeg\3110e0c7_1day_EPOCX_202378_2024.01.29T09.42.53+01.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 3110e0c7     | day2  | nan    | 2024.02.02T09.44.32+01.00 | 26.7954   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\3110e0c7\day2\eeg\3110e0c7_day2_EPOCX_202378_2024.02.02T09.44.32+01.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 40009139     | day1  | nan    | 2023.12.04T13.23.34+03.00 | 36.9018   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\40009139\day1\eeg\40009139_EPOCX_202378_2023.12.04T13.23.34+03.00.md.mc.pm.fe.bp.csv.bz2            |
| gpn_data | 40009139     | day2  | nan    | 2023.12.10T14.19.11+03.00 | 21.109    | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\40009139\day2\eeg_2day\40009139_2day_EPOCX_202378_2023.12.10T14.19.11+03.00.md.mc.pm.fe.bp.csv.bz2  |
| gpn_data | 40f0714a     | day2  | nan    | 2023.12.15T11.55.06+03.00 | 18.4965   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\40f0714a\day2\eeg\40f0714a_2day_EPOCX_202378_2023.12.15T11.55.06+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 41e2010c     | day1  | nan    | 2023.11.29T17.22.52+03.00 | 40.3357   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\41e2010c\day1\eeg\41e2010c_1day_EPOCX_202378_2023.11.29T17.22.52+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 5001d09a     | day1  | 2part  | 2023.11.27T16.27.12+03.00 | 17.2948   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\5001d09a\day1\5001d09a_1day_2part_EPOCX_202378_2023.11.27T16.27.12+03.00.md.mc.pm.fe.bp.csv.bz2     |
| gpn_data | 5001d09a     | day1  | nan    | 2023.11.27T15.49.07+03.00 | 23.2218   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\5001d09a\day1\5001d09a_1day_EPOCX_202378_2023.11.27T15.49.07+03.00.md.mc.pm.fe.bp.csv.bz2           |
| gpn_data | 50c02189     | day1  | nan    | 2023.11.27T11.56.06+03.00 | 46.0292   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\50c02189\day1\eeg\50c02189_day1_EPOCX_202378_2023.11.27T11.56.06+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 50c02189     | day2  | nan    | 2023.12.06T15.28.05+03.00 | 37.4228   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\50c02189\day2\eeg\50c02189_2day_EPOCX_202378_2023.12.06T15.28.05+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 517001af     | day1  | nan    | 2023.11.27T13.40.34+03.00 | 85.0365   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\517001af\day1\eeg\517001af_1day_EPOCX_202449_2023.11.27T13.40.34+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 517001af     | day2  | nan    | 2023.12.09T14.46.37+03.00 | 19.8305   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\517001af\day2\eeg\517001af_2day_EPOCX_202378_2023.12.09T14.46.37+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 6030f0fd     | day1  | nan    | 2023.12.05T15.32.40+03.00 | 41.999    | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\6030f0fd\day1\eeg\6030f0fd_EPOCX_202378_2023.12.05T15.32.40+03.00.md.mc.pm.fe.bp.csv.bz2            |
| gpn_data | 6030f0fd     | day2  | nan    | 2023.12.10T18.03.09+03.00 | 17.6119   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\6030f0fd\day2\eeg_2day\6030f0fd_2day_EPOCX_202378_2023.12.10T18.03.09+03.00.md.mc.pm.fe.bp.csv.bz2  |
| gpn_data | 7072a0e0     | day1  | nan    | 2023.12.11T13.28.50+03.00 | 65.1885   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\7072a0e0\day1\eeg\7072a0e0_1day_EPOCX_202378_2023.12.11T13.28.50+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 7072a0e0     | day2  | nan    | 2023.12.15T15.47.17+03.00 | 32.77     | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\7072a0e0\day2\eeg\7072a0e0_2day_EPOCX_202378_2023.12.15T15.47.17+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 7150e10a     | day1  | nan    | 2023.12.09T09.37.19+03.00 | 80.0012   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\7150e10a\day1\eeg\7150e10a_1day_EPOCX_202378_2023.12.09T09.37.19+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 7150e10a     | day2  | nan    | 2023.12.15T09.26.01+03.00 |  2.72306  | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\7150e10a\day2\eeg\Cerebral Circles_EPOCX_202378_2023.12.15T09.26.01+03.00.md.mc.pm.fe.bp.csv.bz2    |
| gpn_data | 7150e10a     | day2  | nan    | 2023.12.15T09.30.21+03.00 | 18.2042   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\7150e10a\day2\eeg\Cerebral Circles_EPOCX_202378_2023.12.15T09.30.21+03.00.md.mc.pm.fe.bp.csv.bz2    |
| gpn_data | 71a251fa     | day2  | nan    | 2023.12.06T16.55.33+03.00 | 29.1401   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\71a251fa\day2\eeg\71a251fa_2day_EPOCX_202378_2023.12.06T16.55.33+03.00.md.mc.pm.fe.bp.csv.bz2       |
| gpn_data | 71a251fa     | day1  | nan    | 2023.11.28T17.13.45+03.00 | 82.5841   | True       | True         |      183 |                  20 |                 36 |                        70 |                     15 |                      6 | data\raw\gpn_data\71a251fa\eeg\71a251fa__day1_EPOCX_202378_2023.11.28T17.13.45+03.00.md.mc.pm.fe.bp.csv.bz2           |

## Примеры колонок

### time_columns

`["Timestamp", "OriginalTimestamp"]`

### eeg_columns

`["EEG.Counter", "EEG.Interpolated", "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4", "EEG.RawCq", "EEG.Battery", "EEG.BatteryPercent", "EEG.MarkerHardware"]`

### pm_columns

`["PM.Attention.IsActive", "PM.Attention.Scaled", "PM.Attention.Raw", "PM.Attention.Min", "PM.Attention.Max", "PM.Engagement.IsActive", "PM.Engagement.Scaled", "PM.Engagement.Raw", "PM.Engagement.Min", "PM.Engagement.Max", "PM.Excitement.IsActive", "PM.Excitement.Scaled", "PM.Excitement.Raw", "PM.Excitement.Min", "PM.Excitement.Max", "PM.LongTermExcitement", "PM.Stress.IsActive", "PM.Stress.Scaled", "PM.Stress.Raw", "PM.Stress.Min", "PM.Stress.Max", "PM.Relaxation.IsActive", "PM.Relaxation.Scaled", "PM.Relaxation.Raw", "PM.Relaxation.Min", "PM.Relaxation.Max", "PM.Interest.IsActive", "PM.Interest.Scaled", "PM.Interest.Raw", "PM.Interest.Min", "PM.Interest.Max", "PM.Focus.IsActive", "PM.Focus.Scaled", "PM.Focus.Raw", "PM.Focus.Min", "PM.Focus.Max"]`

### bandpower_columns

`["POW.AF3.Theta", "POW.AF3.Alpha", "POW.AF3.BetaL", "POW.AF3.BetaH", "POW.AF3.Gamma", "POW.F7.Theta", "POW.F7.Alpha", "POW.F7.BetaL", "POW.F7.BetaH", "POW.F7.Gamma", "POW.F3.Theta", "POW.F3.Alpha", "POW.F3.BetaL", "POW.F3.BetaH", "POW.F3.Gamma", "POW.FC5.Theta", "POW.FC5.Alpha", "POW.FC5.BetaL", "POW.FC5.BetaH", "POW.FC5.Gamma", "POW.T7.Theta", "POW.T7.Alpha", "POW.T7.BetaL", "POW.T7.BetaH", "POW.T7.Gamma", "POW.P7.Theta", "POW.P7.Alpha", "POW.P7.BetaL", "POW.P7.BetaH", "POW.P7.Gamma", "POW.O1.Theta", "POW.O1.Alpha", "POW.O1.BetaL", "POW.O1.BetaH", "POW.O1.Gamma", "POW.O2.Theta", "POW.O2.Alpha", "POW.O2.BetaL", "POW.O2.BetaH", "POW.O2.Gamma", "POW.P8.Theta", "POW.P8.Alpha", "POW.P8.BetaL", "POW.P8.BetaH", "POW.P8.Gamma", "POW.T8.Theta", "POW.T8.Alpha", "POW.T8.BetaL", "POW.T8.BetaH", "POW.T8.Gamma", "POW.FC6.Theta", "POW.FC6.Alpha", "POW.FC6.BetaL", "POW.FC6.BetaH", "POW.FC6.Gamma", "POW.F4.Theta", "POW.F4.Alpha", "POW.F4.BetaL", "POW.F4.BetaH", "POW.F4.Gamma", "POW.F8.Theta", "POW.F8. ...`

### motion_columns

`["MOT.CounterMems", "MOT.InterpolatedMems", "MOT.Q0", "MOT.Q1", "MOT.Q2", "MOT.Q3", "MOT.AccX", "MOT.AccY", "MOT.AccZ", "MOT.MagX", "MOT.MagY", "MOT.MagZ", "MC.Action", "MC.ActionPower", "MC.IsActive"]`

### facial_columns

`["FE.BlinkWink", "FE.HorizontalEyesDirection", "FE.UpperFaceAction", "FE.UpperFaceActionPower", "FE.LowerFaceAction", "FE.LowerFaceActionPower"]`

## Интерпретация

1. Если `n_cols = 183`, структура main-файлов стабильна.
2. Если `has_json = True`, JSON можно использовать как источник метаданных.
3. Если `has_marker = True`, marker-файлы можно сохранить в каталоге, даже если они пустые.
4. Для первого рабочего датасета лучше использовать только `gpn_data`, не смешивая со старым `Old_EEG`.
5. Следующий этап — построение оконных сегментов по EEG и PM-колонкам.