# EEG Cognitive State Benchmark

Единая воспроизводимая платформа для классификации и регрессии когнитивного
состояния по агрегированным EEG/POW-признакам и сырым окнам ЭЭГ.

Актуальное состояние: 29 июля 2026 года. Проект находится на этапе
консолидации результатов, формального закрытия требований и подготовки
публикационных материалов.

## Данные и задачи

Основной набор объединяет `gpn_data` и `Old_EEG`. Это источники одного общего
класса Emotiv-записей с разной организацией экспериментов; их нельзя
автоматически трактовать как разные устройства или независимые sensor domains.
Совпадающие логические записи отслеживаются через `record_group_id`.

Канонический feature parquet:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
SHA-256: 26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8
```

| Представление | Выборка | Размер |
|---|---|---:|
| EEG+POW | все окна до target filtering | 51 308 × 448 |
| `label_q5` | supervised окна, 54 участника | 45 384 × 448 |
| семь PM targets | complete-case, 53 участника | 43 174 × 448 |
| raw EEG deduplicated | `[1, 14, 2560]`, 256 Гц | 30 958 окон |

Поддерживаются:

- пяти-классовая классификация `label_q5`;
- скалярная и семивыходная PM-регрессия;
- категориальная и порядковая классификация;
- leakage-safe персональная настройка;
- raw-window, feature-window и feature-sequence inputs.

`label_q5` является заранее определённой benchmark-целью на основе глобальных
квантилей `target_focus`. Cross-fitted sensitivity analysis изменила 2.6816%
окон; поэтому глобальная метка сохранена для сопоставимости, а leakage-safe
вариант обязателен как анализ чувствительности.

## COG-BCI

Внешний корпус COG-BCI полностью распакован и проинвентаризирован:

- 29 участников, 3 сеанса, 1 044 EEGLAB-записи;
- нативные layouts 62/63 EEG-канала и явный 14-канальный `emotiv_common`;
- record-safe cache 500 Гц × 5.12 с: 56 903 окна;
- time-aligned cache 256 Гц × 10 с: 28 910 окон;
- N-Back и MATB-II protocol manifests;
- CNN, preprocessing, spectral и contrastive-transfer diagnostics.

Решение spectral screening — `retain_14_channel_cache`: прирост 62 каналов
составил только +0.0077 balanced accuracy и не достиг порога +0.03. Shape-only
и time-aligned contrastive transfer не превзошли случайную инициализацию
downstream. Решение — `close_transfer_track`; эти эксперименты не возобновляются
без новой утверждённой научной гипотезы.

## Архитектура

```text
datasets and caches
→ task registry and leakage-safe splits
→ model factory and shared Torch adapters
→ metrics and predictions
→ manifests, checkpoints and resume
→ experiment registry
→ reproducible summaries and publication package
```

Основной outer protocol — пятифолдовый `GroupKFold` по `subject_id`.
Torch-модели используют group-aware inner validation. Outer test не участвует
в preprocessing fit, early stopping, выборе loss/threshold/lambda или эпохи.
Последовательности не пересекают `source + subject_id + record_id`.

Реализованы Random Forest, sklearn/Torch MLP, LSTM, BiLSTM, Transformer,
EEGNet и ShallowConvNet. Transformer поддерживает categorical, CORAL, CORN и
auxiliary-CORN objectives. Общий encoder interface, DANN-компоненты и
contrastive trainer готовы как инфраструктура.

Важно: DANN не имеет подтверждённого научного domain experiment. Contrastive
infrastructure протестирована, но внешний перенос не дал downstream-выигрыша.

## Основные результаты

- Feature-sequence LSTM/BiLSTM/Transformer дают macro F1 около 0.36 на
  subject-disjoint `label_q5`; RF — около 0.30.
- Pure CORN снижает ordinal MAE и severe-error rate, но не даёт устойчивого
  выигрыша balanced accuracy; auxiliary-CORN policy также не поддержана.
- Random Forest превосходит mean baseline в семивыходной PM-регрессии.
- Leakage-safe персонализация даёт небольшие средние эффекты с выраженной
  межсубъектной вариативностью; full-model не универсально лучше head-only.
- CAR не улучшил ShallowConvNet на основном наборе.
- COG-BCI CNN лишь немного превышают трёхклассовый chance level 0.333.

Отрицательные результаты считаются научными исходами проверенных гипотез, а
не ошибками реализации.

## Воспроизводимые команды

Каноническая диагностика:

```powershell
python cli.py --config configs.yaml --verbose
```

Полная RF-классификация:

```powershell
python cli.py --config configs/groupkfold_rf_label_q5.yaml --verbose
```

PM-регрессия:

```powershell
python cli.py --config experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml --verbose
```

COG-BCI N-Back protocol и baseline:

```powershell
python scripts/data/cog_bci_task_protocol.py --config experiments/cog_bci/nback_3class_protocol.json
python scripts/cog_bci_nback_baseline.py --config experiments/cog_bci/nback_eegnet_baseline.json
```

Перегенерация итогового пакета без обучения:

```powershell
python src/19_build_project_final_package.py --repo-root .
```

Диагностическая материализация leakage-safe meta-episodes без обучения:

```powershell
python scripts/build_meta_learning_episodes.py --config experiments/meta_learning/episode_infrastructure_smoke.json --verbose
```

Эпизодическая инфраструктура, synthetic FOMAML contract и production
BatchNorm policies реализованы. Они являются инженерными контрактами, а не
доказательством качества метаобучения.

Синтетический CPU-контракт FOMAML реализован и проверяется отдельно:

```powershell
python scripts/run_fomaml_synthetic_smoke.py --config experiments/meta_learning/fomaml_synthetic_smoke.json --verbose
```

Raw-deduplicated FOMAML проверен как ограниченный EEGNet diagnostic (один
fold, seed 42, пять участников). Participant-level macro F1 ухудшился на
−0.046338 относительно supervised full-model adaptation; preregistered
решение — `do_not_proceed`.

DANN в направлении `Old_EEG → gpn_data` прошёл подтверждающий анализ на
пяти folds и двух primary seeds (123, 2026). Статус —
`partially_confirmed`: primary Δmacro F1 **+0.008048**, Δbalanced accuracy
**+0.008332**, Δordinal MAE **−0.034008**. Seed 42 остаётся отдельным
sensitivity seed; diagnostic fold 1 / seed 42 не включён в primary decision.
Всего подтверждающий этап потребовал 28 новых trainings. Bootstrap interval
включает ноль, поэтому статистическая значимость и полная доменная
инвариантность не заявляются. Итог: [сводный отчёт](reports/summary/final_project_results.md)
и [таблицы](reports/summary/final_result_tables/).

Тесты:

```powershell
python -m pytest -q tests
python -m pytest -q
```

## Итоговые материалы

- [Итоговое состояние](reports/integration/project_final_state.md)
- [Научные выводы](reports/integration/project_scientific_conclusions.md)
- [Отрицательные результаты](reports/integration/project_negative_results.md)
- [Аудит воспроизводимости](reports/integration/project_reproducibility_audit.md)
- [Покрытие требований](reports/requirements/final_requirement_coverage.md)
- [Инвентаризация экспериментов](reports/summary/final_experiment_inventory.csv)
- [Таблицы и рисунки статьи](reports/summary/final_result_tables/)

## Ограничения

- Raw proprietary Emotiv data и большие runtime artifacts не хранятся в Git.
- Ранние RF/MLP baselines не имеют отдельного сохранённого split hash и
  используются в итоговом provenance audit только как supporting evidence.
- COG-BCI transfer — диагностический one-fold screening, не финальная оценка.
- Неизвестная часть upstream EEGLAB preprocessing history сохранена как
  ограничение provenance.
- Формальный авторитетный текст ТЗ отсутствует в tracked-дереве; карта
  требований основана на утверждённом плане проекта.
- Streaming replay, latency measurement, demo и финальная презентация остаются
  открытыми deliverables.

Новые DANN, contrastive, AutoML, 62-channel cache и COG-BCI seed sweeps не
являются приоритетом без новой научной гипотезы.
