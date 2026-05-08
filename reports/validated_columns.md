# Валидация Emotiv-каталога и колонок

Корень проекта: `D:\PycharmProjects\eeg-cognitive-state-nir`

## Общая сводка

- Записей: **120**
- Субъектов: **55**
- Источники: `{'gpn_data': 71, 'Old_EEG': 49}`
- Распределение `n_cols`: `{182: 3, 183: 117}`
- Общих колонок во всех записях: **182**
- Уникальных колонок суммарно: **183**

## Проблемные записи

| source   | subject_id   | day   |   part |   n_cols | main_rel_path                                                                                                               |
|:---------|:-------------|:------|-------:|---------:|:----------------------------------------------------------------------------------------------------------------------------|
| gpn_data | 71c09041     | day1  |    nan |      182 | data\raw\gpn_data\71c09041\day1\eeg\71c09041_1day_EPOCX_202378_2023.12.06T13.27.03+03.00.md.mc.pm.fe.bp.csv.bz2             |
| Old_EEG  | 71c09041     | day1  |    nan |      182 | data\raw\Old_EEG\experiment-1\raw\71c09041\day1\eeg\71c09041_1day_EPOCX_202378_2023.12.06T13.27.03+03.00.md.mc.pm.fe.bp.csv |
| Old_EEG  | e0c0408a     | day1  |    nan |      182 | data\raw\Old_EEG\experiment-1\raw\e0c0408a\day1\eeg\e0c0408a_1day_EPOCX_202378_2023.12.05T19.36.11+03.00.md.mc.pm.fe.bp.csv |

## Общие колонки по группам

| field             |   n_common_all |   n_union_all |
|:------------------|---------------:|--------------:|
| time_columns      |              1 |             2 |
| eeg_columns       |             20 |            20 |
| pm_columns        |             36 |            36 |
| bandpower_columns |             70 |            70 |
| motion_columns    |             15 |            15 |
| facial_columns    |              6 |             6 |

## Рекомендуемый минимальный набор колонок

### Time

`Timestamp`

### EEG signal channels

`EEG.AF3, EEG.F7, EEG.F3, EEG.FC5, EEG.T7, EEG.P7, EEG.O1, EEG.O2, EEG.P8, EEG.T8, EEG.FC6, EEG.F4, EEG.F8, EEG.AF4`

### PM scaled columns

`PM.Attention.Scaled, PM.Engagement.Scaled, PM.Excitement.Scaled, PM.Stress.Scaled, PM.Relaxation.Scaled, PM.Interest.Scaled, PM.Focus.Scaled`

### PM active columns

`PM.Attention.IsActive, PM.Engagement.IsActive, PM.Excitement.IsActive, PM.Stress.IsActive, PM.Relaxation.IsActive, PM.Interest.IsActive, PM.Focus.IsActive`

### POW columns

Количество: **70**

`POW.AF3.Alpha, POW.AF3.BetaH, POW.AF3.BetaL, POW.AF3.Gamma, POW.AF3.Theta, POW.AF4.Alpha, POW.AF4.BetaH, POW.AF4.BetaL, POW.AF4.Gamma, POW.AF4.Theta, POW.F3.Alpha, POW.F3.BetaH, POW.F3.BetaL, POW.F3.Gamma, POW.F3.Theta, POW.F4.Alpha, POW.F4.BetaH, POW.F4.BetaL, POW.F4.Gamma, POW.F4.Theta, POW.F7.Alpha, POW.F7.BetaH, POW.F7.BetaL, POW.F7.Gamma, POW.F7.Theta, POW.F8.Alpha, POW.F8.BetaH, POW.F8.BetaL, POW.F8.Gamma, POW.F8.Theta, POW.FC5.Alpha, POW.FC5.BetaH, POW.FC5.BetaL, POW.FC5.Gamma, POW.FC5.Theta, POW.FC6.Alpha, POW.FC6.BetaH, POW.FC6.BetaL, POW.FC6.Gamma, POW.FC6.Theta ... (+30)`

### Motion columns

Количество: **15**

`MC.Action, MC.ActionPower, MC.IsActive, MOT.AccX, MOT.AccY, MOT.AccZ, MOT.CounterMems, MOT.InterpolatedMems, MOT.MagX, MOT.MagY, MOT.MagZ, MOT.Q0, MOT.Q1, MOT.Q2, MOT.Q3`

### Facial columns

Количество: **6**

`FE.BlinkWink, FE.HorizontalEyesDirection, FE.LowerFaceAction, FE.LowerFaceActionPower, FE.UpperFaceAction, FE.UpperFaceActionPower`

## Сравнение источников

| field             | source   |   n_common |   n_union |
|:------------------|:---------|-----------:|----------:|
| time_columns      | Old_EEG  |          1 |         2 |
| time_columns      | gpn_data |          1 |         2 |
| eeg_columns       | Old_EEG  |         20 |        20 |
| eeg_columns       | gpn_data |         20 |        20 |
| pm_columns        | Old_EEG  |         36 |        36 |
| pm_columns        | gpn_data |         36 |        36 |
| bandpower_columns | Old_EEG  |         70 |        70 |
| bandpower_columns | gpn_data |         70 |        70 |
| motion_columns    | Old_EEG  |         15 |        15 |
| motion_columns    | gpn_data |         15 |        15 |
| facial_columns    | Old_EEG  |          6 |         6 |
| facial_columns    | gpn_data |          6 |         6 |

## Вывод

1. Если число общих колонок близко к 183, можно строить единый загрузчик для `gpn_data` и `Old_EEG`.
2. Если полный набор PM/EEG-каналов общий, первый датасет можно строить сразу по двум источникам с обязательным полем `source`.
3. Для моделей надо делать отдельные эксперименты: `gpn_data`, `Old_EEG`, `all`, а также cross-source проверку.
4. Следующий этап — построение оконных сегментов и агрегация PM-метрик.