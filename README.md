# EEG Cognitive State Benchmark

Воспроизводимая исследовательская платформа для распознавания PM-defined когнитивных состояний по данным электроэнцефалографии (ЭЭГ) с межсубъектной оценкой, персонализацией, внешней валидацией, исследованием предобработки и потоковым программным контуром.

**Актуализация README: 5 ноября 2026 года.**

## Текущее состояние

Основной научный контур проекта ориентирован на классификацию всех семи Performance Metrics (PM):

- `target_attention`;
- `target_engagement`;
- `target_excitement`;
- `target_stress`;
- `target_relaxation`;
- `target_interest`;
- `target_focus`.

Основная постановка — **LOW / HIGH**: нижняя и верхняя трети распределения PM, средняя треть исключается. Q33/Q67 вычисляются отдельно для каждого PM и outer fold **только по outer-train** и без переоценки применяются к outer-test. Точная семантика текущего кода и спецификаций: `LOW ≤ Q(1/3)`, `HIGH ≥ Q(2/3)`; значения на границах включаются в крайние классы.

Трёхклассовая Q3-постановка и пяти-классовая Focus-разметка `label_q5` — **исторические (legacy)** протоколы, сохранённые для воспроизводимости.

Завершены межсубъектные baseline, исследования preprocessing, временной привязки EEG–PM, последовательностных моделей, персонализации, transfer и streaming. Следующий этап — **Temporal Predictability Audit**, проверка полезности истории до канонического 10-секундного блока.

Подробный научный контекст: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md). Правила работы и экспериментальные инварианты: [AGENTS.md](AGENTS.md). Точные контракты и результаты определяются кодом, спецификациями и артефактами конкретного эксперимента.

## Исходные данные и каноническая когорта

Основной внутренний набор сформирован из двух исходных источников:

- `gpn_data`;
- `Old_EEG`.

Они используют ЭЭГ-устройства класса Emotiv и отличаются структурой файлов и организацией записей. Эти источники **не рассматриваются как независимые научные домены**, поскольку часть физических записей представлена в обоих источниках. Совпадающие записи отслеживаются через `record_group_id`.

Основная цепочка формирования выборки:

| Этап | Размер |
|---|---:|
| Полная обработанная таблица до фильтрации по целевой разметке | 51 308 окон |
| Окна, пригодные для обучения с учителем | 45 384 |
| После контроля качества raw EEG | 45 326 |
| После устранения повторного представления логически одинаковых записей | **30 958** |

После дедупликации каноническая когорта сырого ЭЭГ содержит **30 958 окон из 86 логических записей, относящихся к 54 участникам**.

Контракт raw EEG:

```text
shape: [B, 1, 14, 2560]
channels: 14
sample rate: 256 Hz
window: 10 s
dtype: float32
```

Для `attention` число complete-case окон составляет **29 569**. Для остальных шести PM доступны все **30 958** окон. Это complete-case support до LOW/HIGH-фильтрации и временного сопоставления; фактические размеры train/test определяются целевыми масками и support audit конкретного эксперимента внутри фиксированных внешних разбиений.

## Признаковые представления

В проекте используются два различных инженерных представления; они не являются последовательными стадиями отбора признаков.

### Унифицированное представление — 371 признак

Современный `FeaturePipeline` вычисляет признаки непосредственно из окна ЭЭГ и формирует фиксированное целенезависимое представление:

- 182 спектральных;
- 140 статистических;
- 28 энтропийных;
- 21 характеристика межканальной связности.

Итого: **371 признак**.

Это представление используется в основном сравнении признаковых и последовательностных моделей.

### Исходное EEG+POW-представление — 448 признаков

Исторический обработанный parquet содержит:

- 168 EEG-признаков;
- 280 POW-признаков.

Итого: **448 признаков**.

Для этого представления отдельно проверен fold-local автоматический отбор признаков: корреляционный фильтр `|r| > 0.95` и Random Forest importance сокращают размерность **448 → 50**. Это отдельный эксперимент и не является происхождением 371-признакового пространства.

## Экспериментальный протокол

Основная оценка использует пять фиксированных внешних разбиений по `subject_id`.

Основные правила:

- участники outer-train и outer-test не пересекаются;
- внутренняя валидация учитывает `record_group_id`;
- Q33/Q67 для LOW/HIGH вычисляются только по outer-train;
- нормализация, заполнение пропусков и отбор признаков обучаются только на train-части;
- outer-test не используется для early stopping или выбора модели;
- последовательности остаются внутри одного участника, логической записи и непрерывного временного сегмента;
- основная итоговая агрегация в подтверждающих экспериментах — participant-macro;
- результаты разных входных когорт не сравниваются как полностью сопоставимые без matched-протокола.

Primary classification endpoint — **participant-macro balanced accuracy**. Secondary: Macro-F1, ROC-AUC, PR-AUC. Участники имеют одинаковый вес; парные сравнения и bootstrap выполняются на уровне участников, а не отдельных окон. Random window split допустим как диагностика, но не является основной научной оценкой.

### Временная привязка EEG и PM

Канонический контракт: **`EEG(t−10 s) → PM(t)`**. Для PM-окна `[T,T+10]` используется EEG-окно `[T−10,T]`; `absolute_lag_s = EEG.t_start − PM.t_start`.

Отдельный **confirmatory** эксперимент проверил `[-30, -20, -10, 0, +10] s` на common support с неизменными порогами и LightGBM (`n_estimators=200`, `random_state=42`). `−10 s` заранее задан как **PRIMARY REFERENCE**, `+10 s` — **non-causal diagnostic only**. Завершены все **175 fits = 5 lags × 7 PM × 5 folds**.

| Абсолютный лаг, s | Среднее participant-macro Balanced Accuracy по семи PM |
|---|---:|
| −30 | ≈0.623 |
| −20 | ≈0.672 |
| **−10** | **≈0.751** |
| 0 | ≈0.672 |
| +10 | ≈0.615 |

Результат поддерживает каноническую predictive temporal alignment; для Focus наблюдается более широкое плато −30…−10 s. Это не доказательство физиологической причинности или универсальной задержки мозга 10 секунд. Лаг 0 использует contemporaneous EEG, доступную лишь к концу PM-окна; внутреннее сглаживание PM неизвестно.

Источники: [отчёт и артефакты](reports/diagnostics/pm_eeg_absolute_lag_v1/), [спецификация](experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json), [итоговые метрики](reports/diagnostics/pm_eeg_absolute_lag_v1/overall.csv), [статус выполнения](reports/diagnostics/pm_eeg_absolute_lag_v1/execution_status.json).

## Модели

Поддерживаются:

- Logistic Regression;
- Random Forest;
- HistGradientBoosting;
- LightGBM;
- XGBoost;
- SVM;
- sklearn MLP;
- PyTorch MLP;
- LSTM;
- BiLSTM;
- Transformer;
- EEGNet;
- ShallowConvNet;
- ShallowFusion.

Для transfer-задач реализован общий encoder-интерфейс; для доменной адаптации — DANN.

## Основные результаты

### Текущий LOW/HIGH-контур

LightGBM и XGBoost — сильные табличные baseline на 371 признаке с канонической привязкой −10 s. Ниже — средние participant-macro метрики по 35 строкам `PM × outer fold` из [model robustness summary](reports/diagnostics/pm_low_high_model_robustness_v1/summary_by_model.csv); это не pooled window metrics.

| Модель | Balanced Accuracy | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 0.7528 | 0.7163 | 0.8790 | 0.8681 |
| XGBoost | 0.7501 | 0.7152 | 0.8732 | 0.8624 |

Таким образом, participant-macro Balanced Accuracy находится около 0.75, ROC-AUC — в high-0.8 range. Это характеристики конкретного LOW/HIGH-протокола, а не заявление о «точности системы 75%».

**Temporal models.** LSTM, BiLSTM и Transformer уже исследовались. Устойчивого преимущества последовательностных моделей над сильными табличными baseline не установлено. В [современном matched-сравнении](reports/diagnostics/pm_low_high_temporal_context_matched_v1/summary_by_model.csv) средняя participant-macro BA составляет около 0.750 для LightGBM/XGBoost и 0.725 для LSTM/Transformer. BiLSTM представлен в историческом сравнении ниже. Следующая задача — определить информационную ценность более длинной истории.

**Персонализация LOW/HIGH.** Исследованы индивидуальные пороги решения, probability calibration и model adaptation при ограниченном объёме target-subject data. В практически важном сценарии первых **30 минут** данных пользователя наиболее устойчивый результат дала адаптация модели через weighted augmentation. На operational support (50 участников, 345 participant–PM пар, с fallback для непригодных к адаптации случаев) получены небольшие стабильные приросты относительно zero-shot:

| Модель | ΔROC-AUC | ΔPR-AUC |
|---|---:|---:|
| XGBoost | +0.0162 | +0.0148 |
| LightGBM | +0.0147 | +0.0153 |

Источники: [итоговые estimands](reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/summary_by_estimand.csv), [participant bootstrap](reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/bootstrap_operational.csv). Для этих четырёх приростов описательные 95% интервалы выше нуля. Адаптация применялась к 285 participant–PM парам у 48 участников; оценка выполнялась на последующем отрезке. Пороги LOW/HIGH оставались outer-train; персональные пороги решения не меняли целевую разметку. README этого эксперимента сохранил раннюю пометку «dry-run», поэтому численные результаты здесь взяты из итоговых CSV.

**Channel diagnostics.** Устойчивого полезного fixed inter-channel lag не обнаружено; удаление F3/F7/F8 ухудшало качество в проверенных конфигурациях ([channel-drop report](reports/diagnostics/pm_qc_channel_drop_confirmatory_v1/README.md), [inter-channel diagnostic](reports/diagnostics/interchannel_lag_pilot_v1/run_summary.json)). Дальнейший arbitrary channel-drop search не является приоритетом; перспективнее phase synchronization, functional connectivity и graph representations.

### Исторические Q3 / legacy experiments

Следующие результаты обработки PM, preprocessing A–H, ICA/FASTER, предварительного model comparison, отбора признаков и ShallowConvNet calibration относятся к предыдущей Q3-постановке. Они сохранены без пересчёта и не являются текущим итоговым LOW/HIGH benchmark.

#### Временная обработка PM (legacy Q3)

В пятифолдовой классификационной проверке всех семи PM исходные значения PM оказались лучшим вариантом:

| Вариант | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| raw | **0.4730** | **0.4791** |
| EMA, α=0.5 | 0.4674 | 0.4731 |
| Hampel | 0.4572 | 0.4629 |
| Median, w=3 | 0.4505 | 0.4568 |

Причинное сглаживание не дало универсального улучшения, поэтому исходные PM сохранены как основной вариант разметки.

#### Факторная абляция предобработки A–H (legacy Q3)

Эксперимент `preprocessing_factorial_q3_all_pm_v1` завершён полностью:

```text
8 preprocessing variants × 7 PM × 5 outer folds = 280 runs
completed_runs = 280
unsupported_runs = 0
protocol_hash = 7ca105e55b4ec84064f60d3a6c99d1251fdd89000a15ac1500f7ae6b3008f0a9
```

Во всех вариантах использовалась идентичная matched-когорта A–H.

| Вариант | Band-pass 1–45 Hz | Notch 50 Hz | CAR | Participant-macro Macro-F1 | Participant-macro Balanced Accuracy |
|---|:---:|:---:|:---:|---:|---:|
| A | − | − | − | 0.3646 | 0.4172 |
| B | + | − | − | **0.3702** | **0.4282** |
| C | − | + | − | 0.3660 | 0.4196 |
| D | − | − | + | 0.3602 | 0.4097 |
| E | + | + | − | 0.3695 | 0.4279 |
| F | + | − | + | 0.3630 | 0.4183 |
| G | − | + | + | 0.3560 | 0.4054 |
| H | + | + | + | 0.3631 | 0.4181 |

Matched participant-level факторные эффекты:

- CAR: ΔMacro-F1 = **−0.0072**, Holm `p = 0.2905`;
- CAR: ΔBalanced Accuracy = **−0.0106**, Holm `p = 0.0026`;
- band-pass: ΔMacro-F1 = `+0.0048`, Holm `p = 0.5675`;
- band-pass: ΔBalanced Accuracy = `+0.0102`, Holm `p = 0.0604`;
- notch: средний эффект близок к нулю для обеих основных метрик.

Следовательно, отрицательный средний эффект CAR статистически подтверждается для Balanced Accuracy, но не для Macro-F1. Полосовая фильтрация показывает положительную описательную тенденцию, однако после поправки Холма она не достигает уровня `p < 0.05`. Универсального эффекта режекторного фильтра не обнаружено.

Файлы эксперимента:

- [`bench/experiments/preprocessing_factorial_q3_all_pm.py`](bench/experiments/preprocessing_factorial_q3_all_pm.py)
- [`experiments/preprocessing/preprocessing_factorial_q3_all_pm_v1.json`](experiments/preprocessing/preprocessing_factorial_q3_all_pm_v1.json)
- [`scripts/run_preprocessing_factorial_q3_all_pm.py`](scripts/run_preprocessing_factorial_q3_all_pm.py)

Полные runtime-артефакты находятся в `benchmark_results/` и не отслеживаются Git.

#### Удаление артефактов (legacy Q3)

Семиметрическая диагностическая абляция ShallowConvNet:

| Вариант | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| raw | 0.3733 | 0.4250 |
| ICA | 0.3722 | 0.4265 |
| FASTER-like | 0.3719 | 0.4227 |
| FASTER-like + ICA | 0.3540 | 0.4063 |

Усложнение процедуры очистки не обеспечило устойчивого повышения качества.

В отдельном matched-сравнении полного MNE-FASTER выполнены все 140 запусков для семи PM, ShallowConvNet и XGBoost. Из 30 958 окон после MNE-FASTER осталось 29 899.

| Модель | Предобработка | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| ShallowConvNet | baseline | 0.4433 | 0.4265 | 0.3768 |
| ShallowConvNet | MNE-FASTER | 0.4214 | 0.4043 | 0.3511 |
| XGBoost | baseline | 0.4658 | 0.4497 | 0.4137 |
| XGBoost | MNE-FASTER | 0.4473 | 0.4357 | 0.3969 |

MNE-FASTER не выбран как предобработка по умолчанию.

#### Предварительное сравнение моделей (legacy Q3)

На первом внешнем разбиении, seed 42:

| Модель | Вход | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|
| LSTM | sequence, 10 × 371 | **0.4932** | **0.5048** |
| BiLSTM | sequence, 10 × 371 | 0.4857 | 0.5003 |
| XGBoost | 371 признаков | 0.4848 | 0.4923 |
| Random Forest | 371 признаков | 0.4846 | 0.4941 |
| Transformer | sequence, 10 × 371 | 0.4672 | 0.4799 |
| PyTorch MLP | 371 признаков | 0.4483 | 0.4573 |
| Logistic Regression | 371 признаков | 0.4201 | 0.4283 |
| ShallowConvNet | raw EEG | 0.4158 | 0.4236 |
| EEGNet | raw EEG | 0.3975 | 0.4164 |

Этот эксперимент является предварительным: sequence- и single-window модели используют разные допустимые наборы окон, поэтому таблица не трактуется как окончательный рейтинг архитектур.

#### Автоматический отбор признаков (legacy Q3)

Для исходного 448-признакового EEG+POW-представления выполнен отдельный пятифолдовый LightGBM-эксперимент.

| Режим | Macro-F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|
| 448 признаков | **0.4187** | **0.4624** | **0.4826** |
| 50 признаков | 0.4116 | 0.4527 | 0.4753 |

Размерность уменьшилась на 88.84%, а среднее время downstream-обучения LightGBM — примерно в 6.8 раза. Сокращение признакового пространства полезно прежде всего для уменьшения вычислительной стоимости, а не как способ повышения качества.

#### Персонализация нового пользователя (legacy Q3)

Полный эксперимент персональной калибровки ShallowConvNet завершён для всех семи PM.

Медианный фактический объём калибровки:

- 5%: 28 окон, ≈ 4.67 мин;
- 10%: 57 окон, ≈ 9.50 мин;
- 20%: 115 окон, ≈ 19.17 мин.

| Режим | Бюджет | Macro-F1 | Δ к zero-shot |
|---|---:|---:|---:|
| zero-shot | 0% | 0.3788 | — |
| full-model | 5% | 0.3875 | +0.0086 |
| full-model | 10% | 0.3898 | +0.0109 |
| full-model | 20% | **0.3924** | **+0.0136** |
| head-only | 5% | 0.3226 | −0.0563 |
| head-only | 10% | 0.3344 | −0.0445 |
| head-only | 20% | 0.3449 | −0.0339 |

Для full-model прирост participant-macro Macro-F1 сохраняется после поправки Холма для всех трёх бюджетов. Head-only не улучшает Macro-F1 и Balanced Accuracy.

### Внешняя мультимодальная проверка

Этот отдельный контур использует собственные внешние labels; его метрики не эквивалентны внутреннему seven-PM LOW/HIGH benchmark.

Используются MEFAR, CL-Drive и CLARE.

XGBoost:

| Набор | EEG-only Macro-F1 | Peripheral-only | Fusion |
|---|---:|---:|---:|
| MEFAR | 0.3972 | **0.5776** | 0.5111 |
| CL-Drive | 0.3805 | 0.3723 | **0.3916** |
| CLARE | **0.3083** | 0.3016 | 0.2703 |

Дополнительная физиологическая модальность не даёт универсального улучшения: результат зависит от набора данных и способа объединения.

### Независимый cross-dataset перенос CL-Drive ↔ CLARE

Для научной проверки domain shift используются независимые CL-Drive и CLARE. Внутренние `gpn_data`/`Old_EEG` для доказательства междоменного переноса не применяются.

EEG-only контракт:

```text
channels: TP9, AF7, AF8, TP10
sample rate: 256 Hz
window: 10 s
target: subjective cognitive load, fixed 3 classes
```

Participant-macro Macro-F1:

| Направление | Target-only | Source-only | DANN |
|---|---:|---:|---:|
| CL-Drive → CLARE | **0.2175** | 0.1890 | 0.1695 |
| CLARE → CL-Drive | **0.2876** | 0.2202 | 0.2095 |

Прямой перенос ухудшает качество в обоих направлениях. DANN не компенсировал межнаборный сдвиг и в среднем дал дополнительное снижение. После поправки Холма статистически подтверждённого преимущества или ухудшения DANN не получено.

### Потоковая обработка

Scientific replay использует 10-секундное окно со сдвигом 1 секунда.

| Профиль | Признаков | Feature P95 | Model P95 | Total P95 |
|---|---:|---:|---:|---:|
| full | 399 | 3047.411 ms | 4.723 ms | 3052.311 ms |
| lightweight | 336 | 6.573 ms | 5.624 ms | **12.215 ms** |

Полный профиль не укладывается в шаг обновления 1 с. Облегчённый профиль имеет большой вычислительный запас и позволяет после накопления первого окна формировать новое предсказание раз в секунду.

Эти измерения относятся к программной обработке готового окна. Полная задержка `сенсор → транспорт → буфер → обработка → пользователь` с физическим ЭЭГ-устройством пока не измерена.

## Внешние наборы

### COG-BCI

Используется как отдельный диагностический и transfer-контур:

- 29 участников;
- 3 сессии;
- 1 044 EEG-записи;
- ≈ 81.75 ч;
- 500 Hz;
- 62/63 EEG-канала в нативных схемах;
- N-Back, MATB-II, PVT, Flanker и resting state.

Расширение числа каналов и contrastive/meta-learning эксперименты сохранены как диагностические результаты; устойчивого downstream-улучшения в проверенных конфигурациях не получено.

### CL-Drive и CLARE

Для EEG-only cross-dataset эксперимента:

- CL-Drive: 3 086 пригодных EEG-окон, 21 участник;
- CLARE: 3 829 пригодных EEG-окон, 19 участников.

Для мультимодального matched-контура используются более узкие общие когорты EEG + ECG + EDA; их размеры не следует смешивать с EEG-only DANN-когортой.

## Структура репозитория

```text
bench/
  datasets/          загрузчики и dataset-контракты
  experiments/       воспроизводимые экспериментальные протоколы
  tasks/             target registry и target transforms
  validation/        канонические scientific splits и метрики
  data_quality/      переиспользуемые inventory/QC-аудиты
  preprocessing/     fold-local transforms, caches и orchestration
  features/          dataset-specific feature caches/adapters
  automl/            scientific/ и personalized/ orchestration
  analysis/          статистический и диагностический анализ

cogstate/
  ingestion/         generic parsing и canonical records
  preprocessing/     фильтрация и удаление артефактов
  features/          каноническое target-free извлечение 371 признака
  model_zoo/         единственные реализации и единая фабрика моделей
  adaptation/        FOMAML core и reusable calibration primitives
  streaming/         потоковая обработка
  evaluation/        application-level evaluation components

experiments/          конфигурации экспериментов
scripts/              тонкие актуальные CLI-точки запуска
apps/streaming_worker/ application inference через cogstate
reports/              отчёты и сводные артефакты
artifacts/            небольшие model-bundle manifests
```

Reusable-логика исторических команд перенесена в `bench/` и `cogstate/`.
Все актуальные запуски выполняются через entry points в `scripts/data/`,
`scripts/analysis/` и `scripts/`; отдельного legacy-слоя команд нет.

Большие кэши, predictions, checkpoints, веса моделей и `benchmark_results/` не хранятся в Git.

## Воспроизводимость

Основной Python-окружение проекта должно содержать PyTorch, scikit-learn, pandas, NumPy, SciPy и зависимости конкретных моделей.

Пример воспроизведения исторического Q3 A–H эксперимента (не текущий LOW/HIGH benchmark):

```powershell
python scripts\run_preprocessing_factorial_q3_all_pm.py `
  --config experiments\preprocessing\preprocessing_factorial_q3_all_pm_v1.json `
  --run `
  --resume
```

План без обучения:

```powershell
python scripts\run_preprocessing_factorial_q3_all_pm.py `
  --config experiments\preprocessing\preprocessing_factorial_q3_all_pm_v1.json `
  --plan-only
```

Тест исторического preprocessing-контура:

```powershell
python -m pytest tests\test_preprocessing_factorial_q3_all_pm.py -q
```

Scientific lightweight replay:

```powershell
python scripts\run_streaming_scientific.py `
  --config configs\streaming_scientific_lightweight_v1.yaml `
  --action replay
```

FastAPI:

```powershell
python -m apps.streaming_worker.api `
  --config configs\streaming_scientific_lightweight_v1.yaml
```

Прикладной streaming CLI и terminal dashboard:

```powershell
python -m apps.streaming_worker validate --config configs\streaming.yaml
python -m apps.streaming_worker run --config configs\streaming.yaml --dashboard
python -m apps.streaming_worker run --config configs\streaming.yaml --demo --dashboard
```

Docker-образ использует тот же `configs/streaming.yaml` и запускает FastAPI:

```powershell
docker build -t eeg-cogstate-streaming .
docker run --rm -p 8000:8000 eeg-cogstate-streaming
```

Для production-запуска необходимо смонтировать совместимые replay/device data
и model bundle; bootstrap-manifests предназначены только для диагностического
сквозного smoke-теста.

## Текущий исследовательский этап

**Temporal Predictability Audit** должен ответить: содержит ли EEG-history до канонического `[T−10,T]` дополнительную predictive information сверх одного 10-секундного блока?

Планируемые горизонты: **H10, H20, H30, H60, H120**. История расширяется назад, оставаясь внутри одной непрерывной записи. План включает:

- support/retention audit с числом участников, записей и train/test окон;
- autocorrelation, permutation entropy и mutual information для выбора диагностического delay;
- Takens-style delay embedding / False Nearest Neighbours и Hankel / Singular Spectrum Analysis;
- surrogate controls внутри записи, с контролем достаточной длины последовательностей;
- predictive LightGBM H10–H120 comparison на matched support, с прежними гиперпараметрами и конкатенацией lagged features;
- при наличии прироста — temporal-order shuffle control с фиксированным каноническим блоком −10 s.

Это план, а не уже полученный результат. Сначала проверяется наличие полезной информации, затем способность архитектуры её использовать. Если H10 ≈ длинным histories, новая temporal NN не обоснована. Польза H20/H30 допускает TimesNet/PatchTST; польза H60/H120 при важности порядка — PatchTST; доказанная длинная память — Mamba/SSM. Если история помогает независимо от порядка, предпочтительнее lag/SSA/statistical features.

Дальнейшая последовательность: wavelet feature ablation → phase-synchronization / Kuramoto-style features → максимум одна обоснованная temporal architecture → confirmatory personalization → external validation научных выводов → manuscript preparation. Подробности — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Ограничения

- Основная внутренняя когорта сравнительно невелика: 54 независимых участника; доступный support зависит от эксперимента.
- PM — проприетарные/device-derived proxy targets, а не независимая прямая разметка когнитивного состояния.
- EEG→PM lag характеризует predictive alignment; он не устанавливает физиологическую причинность и может зависеть от внутренней обработки PM.
- Participant-macro balanced accuracy зависит от PM, support и протокола; её нельзя смешивать с Accuracy или историческими Q3-метриками.
- External dataset labels не полностью эквивалентны внутренним PM; внешняя проверка текущих научных выводов остаётся отдельной задачей.
- Мультимодальность не обеспечивает универсального прироста; DANN не устранил cross-dataset shift между CL-Drive и CLARE.
- Полный MNE-FASTER и CAR не улучшили качество в соответствующих исторических Q3-экспериментах; это не универсальный вывод для всех протоколов.
- Live hardware end-to-end latency не измерена; streaming-цифры относятся к программной обработке готового окна.
- Предварительный legacy model-zoo срез не является окончательным пятифолдовым рейтингом архитектур.

## Итог

Создан воспроизводимый межсубъектный EEG pipeline, унифицированы семь PM и установлен основной LOW/HIGH-протокол. Сформированы сильные LightGBM/XGBoost baseline и подтверждена predictive temporal alignment `EEG(t−10 s) → PM(t)`. Изучены preprocessing, transfer, personalization и streaming; сложные temporal models пока не показали стабильного преимущества. Текущая работа направлена на выяснение temporal information content EEG, далее — wavelet и synchronization representations и внешняя проверка выводов. Отрицательные результаты сохраняются вместе с условиями, в которых они получены.
