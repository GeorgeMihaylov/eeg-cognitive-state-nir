# EEG Cognitive State Benchmark

Единая воспроизводимая платформа для исследования когнитивного состояния по
электроэнцефалографии (ЭЭГ): загрузка и предобработка данных, явные контракты
целевых переменных, межсубъектная оценка, персонализация и перенос, единые
метрики и артефакты, а также потоковый replay и прикладной API.

Актуализация README: **16 августа 2026 года**. Проект находится на этапе
консолидации результатов и подготовки отчётных материалов. Основные
исследовательские и программные контуры реализованы, но физический end-to-end
тест с реальным EEG-устройством ещё не выполнен.

## Текущее состояние

Репозиторий поддерживает:

- семь непрерывных Performance Metrics (PM) и получаемые из них fold-local
  категориальные proxy-состояния;
- историческую пяти-классовую задачу `label_q5` для сопоставимости;
- пятифолдовый межсубъектный benchmark;
- feature-window, feature-sequence и raw-EEG модели;
- временной аудит PM, персонализацию, DANN, внешние мультимодальные наборы и
  автоматический отбор признаков;
- научный streaming replay, streaming worker и FastAPI;
- единые manifests, predictions, split-аудиты, checkpoints и сводные метрики.

Последняя полная проверка текущего кода на этом состоянии ветки:
**1475 passed, 1 skipped, 37 warnings**. Длительные эксперименты не запускаются
автоматически тестами.

## Данные и целевые переменные

### `gpn_data` и `Old_EEG`

Основной набор объединяет два исходных источника Emotiv-класса: `gpn_data` и
`Old_EEG`. Они отличаются организацией файлов и экспериментальным дизайном,
но не считаются автоматически разными устройствами или независимыми доменами.
Совпадающие логические записи отслеживаются через `record_group_id`.

Канонический feature parquet:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
SHA-256: 26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8
```

| Представление | Выборка | Размер |
|---|---|---:|
| EEG+POW | все окна до фильтрации целей | 51 308 × 448 |
| `label_q5` | 54 участника | 45 384 × 448 |
| семь PM, complete-case | 53 участника | 43 174 × 448 |
| raw EEG, deduplicated | `[1, 14, 2560]`, 256 Гц | 30 958 окон |

В 448 инженерных признаков входят 168 EEG- и 280 POW-признаков. Цели и
служебные PM-колонки в матрицу признаков не включаются.

### Семь PM

Современный основной научный контур охватывает:

```text
Attention, Engagement, Excitement, Stress, Relaxation, Interest, Focus
```

Непрерывные цели имеют фиксированный порядок:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

Для классификации каждого PM используются три состояния low/medium/high.
Пороговые значения тертилей Q3 вычисляются отдельно в каждом fold **только по
outer-train** и затем без переоценки применяются к outer-test. Реализация
находится в [`target_registry.py`](bench/tasks/target_registry.py) и
[`target_transforms.py`](bench/tasks/target_transforms.py).

`label_q5` — историческая Focus-specific benchmark-метка на основе заранее
вычисленных глобальных квантилей. Она сохранена для сравнения с предыдущими
экспериментами, но не описывает всё пространство целей проекта. Полный реестр:
[`target_registry.yaml`](reports/summary/target_registry.yaml).

### COG-BCI и внешние наборы

COG-BCI используется как отдельный диагностический и transfer-контур:

- 29 участников, 3 сеанса и 1 044 EEGLAB-записи;
- нативные layouts с 62/63 EEG-каналами и сопоставленный 14-канальный профиль;
- 56 903 record-safe окна 500 Гц × 5,12 с;
- 28 910 time-aligned окон 256 Гц × 10 с;
- N-Back, MATB-II, spectral, CNN и contrastive-transfer диагностики.

Для мультимодальной проверки дополнительно используются MEFAR, CL-Drive и
CLARE. Их единица анализа и target-контракты не смешиваются с основным Emotiv
benchmark.

## Экспериментальный протокол

Основная оценка — пятифолдовый `GroupKFold` по `subject_id`. Это протокол без
утечки информации между обучением и тестом: участники outer-train и outer-test
не пересекаются.

Дополнительные правила:

- inner validation учитывает группы участников или `record_group_id`;
- preprocessing, imputation, normalization, clipping, Q3-пороги и feature
  selection обучаются только на текущей train-части;
- outer-test не используется для early stopping или выбора модели;
- последовательности не пересекают `source + subject_id + record_group_id`,
  сортируются по времени и используют стабильный `sample_id`;
- где это задано протоколом, основной единицей итоговой агрегации является
  участник с равным весом — participant-macro, а не число его окон;
- random-window split допускается только для smoke/diagnostic, но не как
  основной научный результат.

## Архитектура проекта

```text
данные и кэши
→ target registry и group-aware splits
→ preprocessing и признаки
→ model factory и общие Torch adapters
→ метрики, predictions и manifests
→ эксперименты, resume-аудит и сводные отчёты
→ streaming worker и API
```

Основные каталоги:

```text
bench/                  воспроизводимый benchmark и экспериментальные протоколы
cogstate/               переиспользуемые preprocessing/features/streaming-компоненты
model_zoo/               канонический factory и модели sklearn/PyTorch
apps/streaming_worker/   прикладной потоковый worker и FastAPI
configs/                 benchmark- и streaming-конфигурации
experiments/             тематические конфигурации экспериментов
scripts/                 проверенные CLI-точки запуска
reports/                 научные отчёты и сводные таблицы
artifacts/               небольшие manifests развёртываемых model bundles
```

Runtime-выходы, predictions, checkpoints и большие кэши находятся в
`benchmark_results/` и не отслеживаются Git.

## Реализованные модели

Канонический `model_zoo` включает:

- Random Forest, Logistic Regression, SVM/SVR, Ridge, HistGradientBoosting,
  LightGBM, XGBoost и sklearn MLP;
- Torch MLP, LSTM, BiLSTM и Transformer;
- EEGNet и ShallowConvNet для raw EEG;
- ShallowFusion для мультимодальных задач;
- categorical, CORAL, CORN и auxiliary-CORN варианты Transformer;
- общий encoder-интерфейс для transfer, DANN и contrastive-компонентов.

## Основные результаты

### Сравнение моделей

Новый предварительный model-zoo эксперимент сравнивает семь Q3-задач на
**одном outer fold, seed 42**. Это инженерный `preliminary`-результат, а не
замена полному пятифолдовому benchmark; raw/feature/sequence модели также
имеют разные допустимые когорты.

| Модель | Вход | Средний Macro F1 | Средняя Balanced Accuracy |
|---|---|---:|---:|
| LSTM | sequence | 0,4932 | 0,5048 |
| BiLSTM | sequence | 0,4857 | 0,5003 |
| XGBoost | feature window | 0,4848 | 0,4923 |
| Random Forest | feature window | 0,4846 | 0,4941 |
| Transformer | sequence | 0,4672 | 0,4799 |

В однократном regression-срезе Random Forest получил средние по семи PM:
MAE 0,1017, R² 0,2280 и Pearson 0,4854. Эти числа относятся только к outer
fold 1. Ранее опубликованные значения около Macro F1 0,36 относятся к другой
постановке — полному пятифолдовому `label_q5` benchmark — и поэтому не
сравниваются с этой таблицей напрямую.

### Временная обработка PM

В полном Random Forest downstream-сравнении проверены исходные PM, causal
median, causal EMA и causal Hampel. Для `raw` средние по 7 PM × 5 folds:

- классификация: Macro F1 0,4730, Balanced Accuracy 0,4791;
- регрессия: MAE 0,09837, RMSE 0,12894, R² 0,18519,
  Pearson 0,44501, Spearman 0,39083.

Сглаживание уменьшало кратковременную вариативность, но не давало
универсального downstream-улучшения: classification-метрики ухудшались, а
снижение MAE в отдельных регрессионных вариантах сопровождалось ухудшением
R² и корреляций. Поэтому `raw` сохранён как канонический вариант. Подробный
аудит происхождения и временной структуры:
[`pm_temporal_quality_v1.md`](reports/pm_quality/pm_temporal_quality_v1.md).

### LightGBM и автоматический отбор признаков

Завершены 140 запусков: 7 PM × classification/regression × 448/50 признаков ×
5 folds. Корреляционный фильтр и Random Forest importance обучаются только на
outer-train; outer-test не участвует в выборе 50 признаков.

| Задача, participant-macro | 448 признаков | 50 признаков |
|---|---:|---:|
| Classification Macro F1 | 0,41869 | 0,41155 |
| Classification Balanced Accuracy | 0,46236 | 0,45268 |
| Classification Accuracy | 0,48263 | 0,47534 |
| Regression MAE | 0,09843 | 0,09925 |
| Regression Pearson | 0,47945 | 0,46708 |
| Regression Spearman | 0,43597 | 0,42604 |

Размерность уменьшилась на 88,84%. Среднее время downstream-обучения
LightGBM сократилось с 5,79 до 0,85 с — примерно в 6,78 раза; inference стал
примерно в 1,15 раза быстрее. Отбор существенно снижает вычислительную
стоимость, но в среднем немного ухудшает качество и не рассматривается как
способ повышения точности. Конфигурация:
[`lightgbm_feature_selection_v1.yaml`](experiments/feature_selection/lightgbm_feature_selection_v1.yaml).

### Предобработка и удаление артефактов

В завершённой восьмивариантной raw-EEG ablation ShallowConvNet лучшие
описательные значения получены для band-pass + notch (Balanced Accuracy
0,2889), band-pass (0,2873) и raw (0,2824). Варианты с CAR находились ниже;
средний описательный эффект CAR составил −0,0285. Статистическая значимость
различий не заявлялась.

Отдельно реализован fold-safe artifact-removal протокол:

```text
raw
FASTER-like
ICA
FASTER-like + ICA
```

FASTER-like означает статистическое обнаружение плохих каналов с
mean-channel interpolation, а не полную каноническую реализацию FASTER. ICA
калибруется только на outer-train. Протокол и smoke-проверка существуют, но
полный сравнительный 5-fold/140-run эксперимент не выполнялся; вывод о влиянии
FASTER-like/ICA на качество не делается.

### Персонализация нового пользователя

Современный confirmatory-контур использует семь непрерывных PM, три seed и
хронологическое разделение ранней calibration-части и поздней evaluation-части
каждого нового участника. Для `full_model` относительно `zero_shot`:

- MAE уменьшился на 0,002685, 95% bootstrap CI [0,001506; 0,003980];
- RMSE уменьшился на 0,002411, CI [0,001145; 0,003789];
- R² вырос на 0,025116, но CI [−0,043542; 0,085802] включает ноль;
- Spearman вырос на 0,011985, CI [0,006442; 0,018499].

После адаптации participant-macro: MAE 0,102404, RMSE 0,130441 и Spearman
0,382941. Эффект небольшой и неоднородный между участниками.

Формальное требование classification Accuracy ≥ 75% для исторического
`label_q5` **не достигнуто**: среднее значение изменилось примерно с 0,2967
до 0,3138, и 0 из 53 участников достигли 0,75. Это проверенный отрицательный
результат, а не незавершённый подбор параметров. Сводка:
[`colleague_metrics_summary.md`](reports/summary/colleague_metrics_summary.md).

### Перенос между источниками и DANN

Подтверждающий DANN-эксперимент выполнен в направлении
`Old_EEG → gpn_data` на пяти folds и primary seeds 123/2026. Относительно
matched source-only:

- ΔMacro F1 = +0,008048;
- ΔBalanced Accuracy = +0,008332;
- ΔOrdinal MAE = −0,034008;
- 23/42 участников улучшились, 19 ухудшились;
- bootstrap 95% CI для ΔMacro F1 [−0,001672; 0,017882] включает ноль.

Статус — `partially_confirmed`: наблюдается небольшой положительный эффект,
но статистическая значимость и полная доменная инвариантность не доказаны.
[`Отчёт DANN`](reports/integration/dann_label_q5_confirmatory_v2.md).

### Мультимодальные данные

Для MEFAR, CL-Drive и CLARE сопоставлены EEG-only, peripheral-only и fusion
EEG + peripheral в одинаковых folds. Основной общий baseline — XGBoost;
ShallowConvNet/ShallowFusion дополнительно реализованы там, где входной
контракт это допускает.

| Набор | EEG-only Macro F1 | Peripheral-only | Fusion |
|---|---:|---:|---:|
| MEFAR | 0,3972 | 0,5776 | 0,5111 |
| CL-Drive | 0,3805 | 0,3723 | 0,3916 |
| CLARE | 0,3083 | 0,3016 | 0,2703 |

Эффект мультимодальности зависит от набора и модели: на MEFAR сильнее
peripheral-only, на CL-Drive fusion даёт небольшой прирост, на CLARE fusion
ухудшает результат. Универсальное улучшение на 5–10% не подтверждено.

- [`MEFAR`](reports/external_datasets/mefar_multimodal_xgboost_protocol.md)
- [`CL-Drive`](reports/external_datasets/cl_drive_multimodal_protocol.md)
- [`CLARE`](reports/external_datasets/clare_multimodal_protocol.md)

### COG-BCI, contrastive transfer и FOMAML

Расширение COG-BCI с 14 до 62 каналов не прошло заранее заданный порог: прирост
Balanced Accuracy составил около +0,0077 при требовании +0,03. Сохранён
14-канальный кэш. Shape-only и time-aligned contrastive transfer не дали
устойчивого downstream-улучшения; решение — `close_transfer_track`.

Raw-deduplicated FOMAML проверен как ограниченный diagnostic: один fold,
seed 42, пять участников и EEGNet. ΔMacro F1 относительно supervised
full-model составил −0,046338; решение — `do_not_proceed`. Это отрицательный
результат конкретного протокола, а не доказательство бесполезности
метаобучения вообще.

## Потоковая обработка

### Вычислительная задержка

Scientific replay использует 10-секундное окно с шагом обновления 1 секунда.
Проверены два диагностических профиля:

| Профиль | Признаков | Feature P95 | Model P95 | Total P95 | Realtime factor |
|---|---:|---:|---:|---:|---:|
| full | 399 | 3047,411 мс | 4,723 мс | 3052,311 мс | 0,356× |
| lightweight | 336 | 6,573 мс | 5,624 мс | 12,215 мс | 63,825× |

Полный профиль не укладывается в бюджет обновления 1 секунду; облегчённый
профиль укладывается с большим запасом. Эти значения измеряют программную
обработку replay и **не** являются полной задержкой
`сенсор → передача → буфер → обработка → API/UI`. Live end-to-end тест с
физическим EEG-устройством пока не выполнен.

### Streaming worker и FastAPI

[`apps/streaming_worker/`](apps/streaming_worker/) содержит:

- replay- и Lab Streaming Layer (LSL) sources;
- буферизацию окон и EEG quality checks;
- model bundle, prediction postprocessing и latest-state sink;
- standalone worker;
- FastAPI transport layer.

Проверенные маршруты:

```text
GET       /health
GET       /v1/status
POST      /v1/runtime/start
POST      /v1/runtime/stop
GET       /v1/predictions/latest
WebSocket /v1/stream
```

## Воспроизводимость и запуск

Установка зависимостей выполняется в окружении проекта. Примеры ниже
предполагают запуск из корня репозитория и наличие локальных данных/кэшей.

Исторический пятифолдовый Random Forest для `label_q5`:

```powershell
python cli.py --config configs/groupkfold_rf_label_q5.yaml --verbose
```

Канонический seven-PM feature baseline:

```powershell
python scripts/run_pm_all_targets_feature_baseline.py `
  --config experiments/pm_regression/pm_all_targets_feature_baseline.yaml `
  --resume
```

LightGBM и fold-local отбор 50 признаков — полный запуск требует явного
подтверждающего флага и выполняет 140 ячеек:

```powershell
python scripts/run_lightgbm_feature_selection.py `
  --config experiments/feature_selection/lightgbm_feature_selection_v1.yaml `
  --confirm-full
```

Scientific lightweight replay:

```powershell
python scripts/run_streaming_scientific.py `
  --config configs/streaming_scientific_lightweight_v1.yaml `
  --action replay
```

FastAPI поверх streaming worker:

```powershell
python -m apps.streaming_worker.api `
  --config configs/streaming_scientific_lightweight_v1.yaml
```

Тесты:

```powershell
python -m pytest -q
```

Подготовленный selected-model confirmatory seven-PM протокол находится в
[`selected_models_5fold_v1.json`](experiments/pm_confirmatory/selected_models_5fold_v1.json),
но полный эксперимент ещё не выполнен и не включён в завершённые результаты.

## Итоговые материалы

- [Сводный пакет результатов](reports/summary/final_project_results.md)
- [Итоговые таблицы и рисунки](reports/summary/final_result_tables/)
- [Итоговое состояние](reports/integration/project_final_state.md)
- [Научные выводы](reports/integration/project_scientific_conclusions.md)
- [Отрицательные результаты](reports/integration/project_negative_results.md)
- [Аудит воспроизводимости](reports/integration/project_reproducibility_audit.md)
- [Покрытие требований](reports/requirements/final_requirement_coverage.md)

Сводные документы были сформированы до части августовских экспериментов;
поэтому для LightGBM, PM-quality, streaming и мультимодальных результатов
приоритет имеют соответствующие experiment manifests и первичные runtime
CSV/JSON. Ссылки сохранены как общий индекс ранее консолидированных результатов.

## Текущее состояние и ограничения

Реализованы target infrastructure семи PM, model zoo, временной PM-аудит,
персонализация, transfer/DANN, мультимодальные эксперименты, fold-local feature
selection, scientific streaming replay и streaming worker/API.

Остаются отдельными задачами:

- физический end-to-end live тест с реальным EEG-устройством;
- прикладная live-демонстрация с устройством;
- полная quantitative 5-fold FASTER-like/ICA ablation, если она потребуется;
- полный selected-model confirmatory benchmark, если будет принято решение о
  его запуске;
- окончательное оформление методических рекомендаций, отчёта и презентации.

Недостижение Accuracy 75% в персонализации `label_q5` и отсутствие
универсального мультимодального прироста — уже проверенные результаты, а не
задачи, которые следует скрывать дополнительным подбором параметров. Raw
proprietary Emotiv data, большие runtime outputs и веса моделей намеренно не
хранятся в Git.
