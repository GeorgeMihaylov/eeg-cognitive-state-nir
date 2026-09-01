# Аудит выбросов и устойчивого масштабирования EEG/POW-признаков

## Область проверки

Проверка выполнена для `performance_metrics_regression`, `torch_mlp`,
outer fold 1, seed 42 и трёх эпох. В supervised-наборе 43 174 окна,
53 испытуемых, 448 входов (168 EEG + 280 POW) и семь PM-целей.
Outer train/test содержат 34 581/8 593 окна и 43/10 испытуемых.
Inner train/validation содержат 29 395/5 186 окон и 36/7 испытуемых.

Loss aggregation не является причиной исходной аномалии. Адаптер суммирует
batch numerator и denominator, а затем вычисляет
`sum(squared_error) / (n_samples * n_outputs)`. Исходный validation MSE
около 355 воспроизводится реальными экстремальными входами и
предсказаниями, а не повторным делением или ошибкой усреднения.

## Происхождение экстремума

Субъект `8191f1d9` представлен 714 окнами только из `Old_EEG`, без
повторяющихся `sample_id`, NaN или Inf. Все окна относятся к одной записи:

`Old_EEG__8191f1d9__day1____2023.12.22T16.25.50p03.00`

и одному `record_group_id`; доступный диапазон `t_start` — 125–7675 секунд.
В исходном каталоге запись соответствует файлу
`8191f1d9_1day_EPOCX_202378_2023.12.22T16.25.50+03.00.md.mc.pm.fe.bp.csv`.

`POW.*` — готовые спектральные показатели экспорта Emotiv, а не band power,
повторно рассчитанный этим benchmark из raw EEG. Построитель датасета
агрегирует каждый исходный POW-столбец в 10-секундном окне функциями
mean/std/min/max. Поэтому `POW.T8.BetaL__min = 12 781.0449` и
`POW.T8.Alpha__min = 19 893.5664` являются минимумами исходного headset
показателя внутри соответствующих окон. По сохранённым данным можно
локализовать эффект до одной записи и источника, но нельзя доказать,
является ли он ошибкой единиц или реальным артефактом исходного экспорта.
Автоматическое удаление субъекта или его окон не выполнялось.

Медианы и IQR двух ключевых признаков близки между источниками, но хвост
`Old_EEG` существенно тяжелее:

| Source | Feature | Median | IQR | Max | Доля выше train q99.9 |
|---|---|---:|---:|---:|---:|
| Old_EEG | `POW.T8.BetaL__min` | 0.3115 | 0.2580 | 12 781.0449 | 1.7613% |
| gpn_data | `POW.T8.BetaL__min` | 0.3046 | 0.2547 | 21.5206 | 0.3107% |
| Old_EEG | `POW.T8.Alpha__min` | 0.4746 | 0.4049 | 19 893.5664 | 1.7798% |
| gpn_data | `POW.T8.Alpha__min` | 0.4438 | 0.3577 | 55.7215 | 0.2780% |

Таким образом, это не общий multiplicative unit shift всей
`Old_EEG`: центр распределения сопоставим, а различается главным образом
редкий хвост, связанный с конкретной записью.

## Train-relative feature audit

При порогах `std < 1e-8` или `IQR < 1e-8` почти постоянных признаков нет
(0 из 448). Максимум 41 472.62σ не вызван почти нулевой train-дисперсией:
для `POW.T8.BetaL__min` inner-train std равен 0.3082, IQR — 0.2352.
Причина — record-local значение порядка 12 781 при обычном масштабе около
0.3.

Топ-20 признаков субъекта `8191f1d9` по максимальному абсолютному
StandardScaler z-score:

| Rank | Feature | Max \|z\| |
|---:|---|---:|
| 1 | `POW.T8.BetaL__min` | 41 472.62 |
| 2 | `POW.T8.Alpha__min` | 38 586.14 |
| 3 | `POW.T8.Theta__min` | 23 143.94 |
| 4 | `POW.P8.BetaL__min` | 22 365.72 |
| 5 | `POW.P8.Alpha__min` | 18 141.06 |
| 6 | `POW.T8.BetaH__min` | 16 249.92 |
| 7 | `POW.P8.BetaH__min` | 14 277.79 |
| 8 | `POW.FC6.Alpha__min` | 12 952.18 |
| 9 | `POW.P8.Theta__min` | 12 027.50 |
| 10 | `POW.T8.Gamma__min` | 10 838.62 |
| 11 | `POW.FC6.BetaL__min` | 9 745.90 |
| 12 | `POW.FC6.Theta__min` | 8 801.20 |
| 13 | `POW.FC6.BetaH__min` | 6 889.12 |
| 14 | `POW.P8.Gamma__min` | 5 488.04 |
| 15 | `POW.FC6.Gamma__min` | 4 218.43 |
| 16 | `POW.T8.Gamma__mean` | 2 851.50 |
| 17 | `POW.F4.Theta__min` | 2 623.67 |
| 18 | `POW.F4.Alpha__min` | 2 597.27 |
| 19 | `POW.T8.BetaH__mean` | 2 430.63 |
| 20 | `POW.T8.BetaL__mean` | 1 943.04 |

## Реализованный preprocessing contract

Все параметры оцениваются только по inner train. Порядок операций:

1. Для POW-log вариантов преобразуются только 280 `POW.*` столбцов.
2. При clipping границы q0.5/q99.5 оцениваются по inner train и
   применяются к inner train, validation и outer test.
3. Scaler обучается на уже преобразованном/clipped inner train.
4. То же сохранённое состояние применяется к validation/test.

Поддерживаются:

- `standard`: `(x - mean_train) / std_train`;
- `robust`: `(x - median_train) / IQR_train`, quantile range 25–75;
- `standard_clip`: clip q0.5/q99.5, затем StandardScaler;
- `robust_clip`: clip q0.5/q99.5, затем RobustScaler;
- `pow_log_standard`: `log1p` только для POW, затем StandardScaler;
- `pow_log_robust`: `log1p` только для POW, затем RobustScaler.

Все POW в train неотрицательны, поэтому выбран `log1p(x)`. Реализация также
поддерживает `sign(x) * log1p(abs(x))`, если в train встречаются
отрицательные POW. EEG-столбцы не логарифмируются. Scale floor равен
`1e-8`; порядок и SHA-256 списка признаков сохраняются в checkpoint и
JSON-артефактах.

## Smoke matrix A–F

Канонический результат этой проверки:
`benchmark_results_20260724_171854.json`. Более ранний запуск
`20260724_171558` исключён: он помог выявить, что GroupKFold-path не
передавал model-level scaling config, поэтому все шесть его моделей были
фактически `standard`. Ошибка интеграции исправлена; результаты не удалены.

| Trial | Strategy | Val loss | Val max \|x\| | Val max \|pred\| | 8191 MSE | Test MAE | Test RMSE | Test R² | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | standard | 354.7887 | 41 472.62 | 1 039.06 | 2 564.3972 | 0.1102 | 0.1524 | -0.1594 | 0.2684 | 0.2864 |
| B | robust | 84.0484 | 65 739.13 | 493.57 | 608.1452 | 0.2277 | 0.3201 | -4.6330 | -0.0851 | -0.1164 |
| C | standard_clip | 0.02693 | 11.93 | 1.217 | 0.02872 | 0.1049 | 0.1399 | 0.0471 | 0.3451 | 0.3077 |
| D | robust_clip | 0.04555 | 942.12 | 2.264 | 0.06489 | 0.1439 | 0.1936 | -0.9817 | 0.1463 | 0.1136 |
| E | pow_log_standard | 0.13831 | 223.76 | 6.863 | 0.74520 | 0.1169 | 0.1555 | -0.2162 | 0.2148 | 0.1849 |
| F | pow_log_robust | 0.07450 | 478.74 | 5.242 | 0.22545 | 0.1338 | 0.1755 | -0.6019 | 0.2158 | 0.1838 |

Best epochs A–F: 2, 3, 3, 3, 1, 1. Training time составил
1.70–2.77 секунды на trial на CUDA. Все prediction arrays конечны.

У `standard_clip` p99 абсолютного transformed feature равен 9.13 против
8.76 у baseline; важное улучшение относится к редкому максимуму, а не к
центру распределения. Число validation-значений с `|x| > 100` уменьшилось
с 4 407 до нуля. Максимальное validation-предсказание уменьшилось с
1 039.06 до 1.217.

## Subject-level сравнение и leakage audit

Для `standard_clip` MSE улучшился у всех 7 inner-validation subjects;
медианное изменение MAE равно -0.1362, и даже наименее улучшившийся субъект
имеет изменение -0.00255. Субъект `8191f1d9` перестал быть худшим:
его MSE уменьшился с 2 564.3972 до 0.02872. Для сравнения, robust без
clipping улучшил 4/7 subjects, robust+clip — 5/7, POW-log+standard — 6/7,
POW-log+robust — 5/7.

Для каждого trial совпадают:

- outer test: 8 593 уникальных sample_id;
- inner validation: те же 7 subjects;
- fit/validation overlap: 0;
- fit/outer-test overlap: 0;
- validation/outer-test overlap: 0.

Clipping bounds, center и scale имеют `scope=inner_train_only`. Outer test
использован только для диагностической оценки заранее определённой матрицы,
а не для подбора percentiles.

## Решение и ограничения

Рекомендуемый preprocessing для следующего ограниченного этапа —
`standard_clip` с фиксированными train-only q0.5/q99.5. Выбор основан
прежде всего на устранении экстремума, конечных значениях, улучшении всех
inner-validation subjects и простом переносимом контракте; outer-test
метрики лишь согласуются с этим решением (MAE ниже baseline примерно на
4.9%, RMSE — на 8.2%).

Это технический однофолдовый, односидовый, трёхэпоховый smoke, а не итоговый
научный результат. Он не оценивает устойчивость между folds/seeds и не
доказывает, что исходные POW-значения ошибочны. До расширения обучения
полезно отдельно проверить экспорт одной локализованной записи, не меняя
данные benchmark. После этого рекомендуемый следующий модельный этап —
персонализация с зафиксированным `standard_clip`; transfer-learning mixins
следует подключать после проверки того же preprocessing contract в
последовательных моделях.

## Артефакты

- `benchmark_results/pm_regression_robust_scaling_smoke/feature_distribution_audit.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/subject_shift_audit.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/extreme_windows.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/near_constant_features.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/robust_scaling_trials.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/robust_scaling_subject_metrics.csv`
- `benchmark_results/pm_regression_robust_scaling_smoke/benchmark_results_20260724_171854.json`
- `benchmark_results/pm_regression_robust_scaling_smoke/summary_20260724_171854.csv`

Каждый fold/trial содержит `model.pt`, `training_log.csv`,
`predictions.parquet`, `feature_scaling.json`, `feature_clipping.json`,
`feature_transform.json` и subject-level audit.
