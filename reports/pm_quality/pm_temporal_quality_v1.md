# Эксперимент по качеству временных рядов PM (п. 10.2.1)

Статус временного аудита: **diagnostic**. Дополняющий downstream-этап Random
Forest выполнен как полный пятифолдовый sensitivity-анализ. Ни один этап не
изменяет канонический Parquet или target registry.

## Подтверждённый исходный контракт

В `src/04_build_windowed_pm_dataset.py::read_and_aggregate_record` каждый
physical recording обрабатывается отдельно. Окна определяются как
`floor(Timestamp / 10 s)`, поэтому размер и шаг равны 10 секундам, а overlap
равен нулю. `target_*` копирует `PM.*.Scaled__mean`, то есть среднее PM внутри
абсолютного временного bin. В каноническом пути нет PM-интерполяции,
дополнительного smoothing и удаления выбросов.

## Протокол

Порядок внутри `source + subject_id + record_group_id + record_id` задаётся
`t_start, sample_id`. Gap больше 10.01 s разрывает сегмент. Любой исходный NaN
сохраняется и сбрасывает causal state. Сравниваются заранее заданные варианты:

- `baseline_raw` — identity;
- `causal_median_w3` — trailing median текущего и максимум двух прошлых окон;
- `causal_ema_a05` — causal EMA с alpha=0.5;
- `causal_hampel_w5_k3` — trailing median/MAD, window=5, threshold=3 robust
  sigma; статистически аномальная текущая точка заменяется local median.

Hampel не ставит flag при нулевом MAD: в этом случае robust scale не определён,
и консервативная политика не объявляет обычное изменение ошибкой. Будущие PM
не используются ни одним преобразованием.

## Данные

| PM | Доступно окон | Пропущено | Участников | Source-records |
| --- | ---: | ---: | ---: | ---: |
| attention | 43 175 | 8 133 | 53 | 117 |
| engagement | 48 254 | 3 054 | 54 | 119 |
| excitement | 50 983 | 325 | 54 | 119 |
| stress | 45 384 | 5 924 | 54 | 119 |
| relaxation | 45 394 | 5 914 | 54 | 119 |
| interest | 45 440 | 5 868 | 54 | 119 |
| focus | 45 384 | 5 924 | 54 | 119 |

Всего в исходном Parquet: 51 308 окон, 55 участников и 120 source-records.

## Временная устойчивость

Средние значения по семи PM (window-weighted overall):

| Variant | Mean abs first diff | Lag-1 autocorrelation | MAE vs raw | Pearson vs raw | Spearman vs raw | Изменено значений |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_raw | 0.069583 | 0.749723 | 0.000000 | 1.000000 | 1.000000 | 0.0000% |
| causal_median_w3 | 0.033513 | 0.890900 | 0.045833 | 0.838341 | 0.836709 | 72.4808% |
| causal_ema_a05 | 0.035042 | 0.918843 | 0.034905 | 0.941571 | 0.934750 | 99.5143% |
| causal_hampel_w5_k3 | 0.059509 | 0.784235 | 0.016174 | 0.909387 | 0.916626 | 10.4537% |

Participant-macro результаты дают тот же порядок: mean absolute first
difference составляет 0.068688 / 0.032882 / 0.034564 / 0.058563 для raw /
median / EMA / Hampel. Следовательно, median и EMA заметно снижают
кратковременную вариативность и увеличивают temporal persistence, но это не
доказательство улучшения target quality.

## Статистически аномальные точки Hampel

| PM | Flags | Fraction |
| --- | ---: | ---: |
| attention | 4 351 | 10.0776% |
| engagement | 4 232 | 8.7703% |
| excitement | 5 708 | 11.1959% |
| stress | 5 521 | 12.1651% |
| relaxation | 4 668 | 10.2833% |
| interest | 4 867 | 10.7108% |
| focus | 4 526 | 9.9727% |

Всего отмечено 33 873 из 324 014 доступных PM-точек (10.4542%). Это
статистически аномальные точки, а не подтверждённые ошибки разметки.

## Fold-local Q3

Для всех 7 PM × 4 variants × 5 фиксированных subject-folds сохранены три
класса. Пороговые значения fitted только на outer-train через существующий
`FoldLocalQuantileTargetTransform`; outer-test не влияет на q1/q2.

Средняя доля изменённых outer-test Q3 labels по 35 PM×fold сравнениям:

| Variant | Mean | Min fold/PM | Max fold/PM |
| --- | ---: | ---: | ---: |
| causal_median_w3 | 21.9244% | 16.5703% | 28.1585% |
| causal_ema_a05 | 17.6306% | 12.8718% | 21.8965% |
| causal_hampel_w5_k3 | 9.5901% | 5.3065% | 15.8017% |

Это сравнение альтернативных определений target, а не строго одной и той же
classification task.

## Temporal lag

Дискретная cross-correlation диагностика показывает best lag +1 окно (10 s)
для trailing median у всех семи PM. Для EMA максимум корреляции остаётся при
нулевом дискретном lag, хотя теоретическая низкочастотная group delay при
alpha=0.5 равна `(1-alpha)/alpha = 1` окну (10 s). Hampel имеет best lag 0.
Эта разница показывает, что smoothness и responsiveness нельзя считать одним
свойством.

## Behavioral audit

Найдены 55 Old_EEG annotation CSV (6 195 raw rows). Поля
`Time Spent (Seconds)` и `Correct Answer` являются реальными event-level
behavioral measurements; первое означает время на слайде/до завершения, но не
подтверждённое reaction time. `First Attempt Answered At` и единичное
`Latest Attempt Answered At` дают timestamps; course/lesson/slide поля задают
контекст. В детерминированном smoke первые 123 проверенных события дали 123
уникальных same-subject interval matches и 200 long-form строк (123 time-spent,
77 correctness); nearest substitution не использовалась.

В gpn_data найдены 80 JSON sidecars (322 marker objects) и 79 intervalMarker
файлов; только 2 intervalMarker-файла непусты, всего 440 строк. Их поля
`timestamp`, `latency`, `duration`, `type`, `marker_value`, `key`, `marker_id`
позволяют временную привязку, но подтверждают прежде всего event/context
markers. Без внешнего определения `plain_hit`, `pattern` и других marker types
они не считаются behavioral outcomes.

## Артефакты и вывод

Runtime-артефакты находятся в
`benchmark_results/pm_temporal_quality_ablation_v1/`; диагностические графики —
в его `figures/`. Полные q1/q2 по folds, participant-macro статистика,
outlier-level audit и behavioral inventory сохранены отдельными таблицами.

Дополнительное causal smoothing заметно меняет PM и 9.6–21.9% fold-local Q3
labels в зависимости от метода. Оно снижает first-difference variability, но
median одновременно даёт наблюдаемый lag 10 s.

## Полный downstream EEG → PM

После временного аудита выполнена заранее зафиксированная Random Forest
матрица: 7 PM × 4 варианта цели × classification/regression × 5 folds = 280
запусков, по 56 на fold. Во всех пяти fold manifests отсутствуют failed runs;
Q3-пороги fitted только на outer-train, матрица из 371 EEG-признака читалась из
существующего feature cache без перестроения. Средние ниже рассчитаны по 35
сопоставимым PM×fold результатам каждого варианта.

| Variant | Classification Macro F1 | Balanced Accuracy | Regression MAE | RMSE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_raw | 0.473036 | 0.479122 | 0.098373 | 0.128938 | 0.185193 | 0.445013 | 0.390826 |
| causal_median_w3 | 0.450473 | 0.456828 | 0.095423 | 0.124984 | 0.121694 | 0.368132 | 0.325278 |
| causal_ema_a05 | 0.467406 | 0.473061 | 0.087494 | 0.114079 | 0.156884 | 0.418815 | 0.374152 |
| causal_hampel_w5_k3 | 0.457189 | 0.462928 | 0.097546 | 0.128670 | 0.145751 | 0.398696 | 0.348878 |

EMA и median снижают абсолютные ошибки относительно изменённой сглаженной
цели, однако raw превосходит их по classification, R² и корреляциям. Поскольку
варианты меняют сам target, одно снижение MAE нельзя трактовать как улучшение
исходной PM-задачи. Универсального downstream-выигрыша нет; `baseline_raw`
остаётся каноническим вариантом.

Первичные runtime-артефакты находятся в
`benchmark_results/pm_temporal_quality_rf_final_v1/` рабочего дерева
PM-quality. В исторических fold manifests поле `result_status` осталось
`diagnostic` из-за исправленной позднее ошибки propagation metadata; это не
меняет splits, predictions или метрики. Исправляющий migration script не
запускался в рамках настоящего аудита.

Config hash: `5a36f2cc3675bce384e87dde3d58003cfbf818c6adb467a524e015bc6ef8fbc5`.

Protocol hash: `b7f28ab750082f96a7c7c8594fabc8444857875e2a74f102654483cbcea62424`.
