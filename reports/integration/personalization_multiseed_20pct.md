# Трёхсидовая устойчивость персонализации `label_q5` при бюджете 20%

## Цель и исходная точка

Эксперимент проверяет, сохраняется ли эффект персонализации при изменении
случайной инициализации и стохастики обучения, но при неизменном составе
данных. Исходный run для `seed=42`
`benchmark_results/calibration_label_q5_full_subjects/personalization/20260725_145534`
дал для `full_model` средние приросты accuracy `+0.019232`, balanced accuracy
`+0.004486` и macro F1 `+0.010017` на 53 complete-case субъектах.

Исторический manifest не содержит SHA-256 датасета, поэтому строгую
совместимость старого результата с новыми seeds доказать нельзя. Старый
seed 42 не переиспользовался: он был повторён в том же трёхсидовом протоколе.
Повтор воспроизвёл старый 20%-срез точно: совпали ключи и значения всех 162
condition-строк, все checkpoint hashes, split-поля, метрики, 108 963
prediction-строки, `y_true`, `y_pred` и `proba_0`–`proba_4` с максимальным
абсолютным различием `0`.

## Протокол

- Dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`.
- Target: `label_q5`, 5 классов, только строки с конечным target.
- Features: 448 EEG + POW признаков; PM, target, ID и временные служебные
  колонки исключены.
- Model: `torch_mlp`, hidden dimensions `[256, 128]`, dropout `0.3`,
  batch size `256`.
- Outer evaluation: 5-fold GroupKFold по `subject_id`.
- Inner validation: group holdout по `subject_id`, fraction `0.15`.
- Preprocessing: `standard_clip`, квантили `0.005` и `0.995`, fit только на
  global inner-train.
- Calibration: первые 20% хронологических окон; оставшиеся 80% — неизменяемая
  final evaluation; методы `zero_shot`, `head_only`, `full_model`.
- Global training: максимум 8 эпох; personalization: максимум 5 эпох.
- Bootstrap: 1000 subject-level resamples, bootstrap seed 42.
- `split_seed=42` фиксирует outer/inner folds, calibration, adaptation и
  evaluation samples. `model_seed ∈ {7, 42, 2026}` меняет только модельную
  стохастику.

Многосидовый orchestration layer создаёт для каждого model seed resolved
single-seed config и вызывает существующий `UserCalibrationExperiment`.
Отдельный training loop не создан. Единица resume остаётся
`seed × fold × subject × budget × method`; завершённые seed-runs и
условия повторно не обучаются.

## Окружение и объём запуска

- Python `3.11.15`.
- PyTorch `2.11.0+cu128`, CUDA `12.8`.
- GPU: NVIDIA GeForce RTX 5060 Ti; CPU fallback запрещён и не использовался.
- 15 новых global trainings: 5 folds × 3 seeds.
- 54 исходных субъекта; 53 complete-case субъекта во всех трёх seeds.
- 486 условий: 477 `completed`, 6 `insufficient_calibration_samples`,
  3 `insufficient_evaluation_samples`, 0 runtime failures.
- Все 9 неполных условий относятся к `9192c107`: по каждому seed zero-shot
  не имеет достаточной evaluation-части, а head/full — достаточной
  calibration-части. Субъект не включён в complete-case агрегаты.
- 326 889 итоговых prediction-строк.
- Wall-clock полного orchestration run: `181.232 s`.
- Суммарное время global training: `44.528 s`
  (seed 7: `16.543 s`, seed 42: `14.405 s`, seed 2026: `13.580 s`).
- Global models: в среднем `4.47` эпох, диапазон `4–8`; best validation loss
  `1.539936–1.665078`.
- Head-only и full-model: в среднем `4.81` эпох, диапазон `4–5`.
- Суммарное fine-tuning time: `7.168 s`
  (head-only `3.404 s`, full-model `3.764 s`).
- Peak allocated GPU memory: global `21 535 232 bytes` (`20.54 MiB`);
  fine-tuning `22 111 744 bytes` (`21.09 MiB`). Batch size не уменьшался.

## Технический smoke и resume

CUDA smoke использовал seeds 7 и 2026, один fold, одного exclusive
`gpn_data` и одного exclusive `Old_EEG` субъекта, три метода, 20%,
2 global и 2 fine-tuning эпохи. Получено 2 global trainings, 12/12
завершённых условий, 6822 predictions, 0 failures за `12.133 s`.
Split/preprocessing hashes совпали, global hashes различались, overlap был
нулевым, вероятности были корректны. Повтор smoke с resume пропустил оба
завершённых seed-runs. Повтор полного запуска с resume пропустил все три
model seeds.

## Результаты по seeds

В таблице указаны subject-level mean score и в скобках mean gain относительно
zero-shot того же seed. Для zero-shot gain равен нулю.

| Seed | Метод | Accuracy | Balanced accuracy | Macro F1 |
|---:|---|---:|---:|---:|
| 7 | zero-shot | 0.298128 | 0.268740 | 0.231454 |
| 7 | head-only | 0.304874 (+0.006746) | 0.269296 (+0.000556) | 0.232175 (+0.000721) |
| 7 | full-model | 0.314481 (+0.016353) | 0.272476 (+0.003736) | 0.234972 (+0.003518) |
| 42 | zero-shot | 0.294658 | 0.265424 | 0.216965 |
| 42 | head-only | 0.304279 (+0.009621) | 0.267651 (+0.002227) | 0.224855 (+0.007890) |
| 42 | full-model | 0.313890 (+0.019232) | 0.269910 (+0.004486) | 0.226982 (+0.010017) |
| 2026 | zero-shot | 0.297200 | 0.267397 | 0.221907 |
| 2026 | head-only | 0.303801 (+0.006600) | 0.268267 (+0.000870) | 0.226264 (+0.004357) |
| 2026 | full-model | 0.312965 (+0.015765) | 0.272226 (+0.004828) | 0.228079 (+0.006172) |

Среднее и sample standard deviation между тремя seed-level means:

| Метод | Метрика | Score mean ± seed SD | Gain mean ± seed SD |
|---|---|---:|---:|
| zero-shot | accuracy | 0.296662 ± 0.001797 | 0 |
| zero-shot | balanced accuracy | 0.267187 ± 0.001668 | 0 |
| zero-shot | macro F1 | 0.223442 ± 0.007365 | 0 |
| head-only | accuracy | 0.304318 ± 0.000538 | 0.007656 ± 0.001704 |
| head-only | balanced accuracy | 0.268405 ± 0.000831 | 0.001218 ± 0.000888 |
| head-only | macro F1 | 0.227764 ± 0.003884 | 0.004323 ± 0.003584 |
| full-model | accuracy | 0.313779 ± 0.000764 | 0.017117 ± 0.001856 |
| full-model | balanced accuracy | 0.271537 ± 0.001415 | 0.004350 ± 0.000559 |
| full-model | macro F1 | 0.230011 ± 0.004331 | 0.006569 ± 0.003268 |

## Итог после усреднения seeds внутри субъекта

Агрегация сначала усредняет три seeds для каждого из 53 субъектов и только
после этого считает group statistics и bootstrap CI. Seeds не трактуются как
159 независимых субъектов.

| Метод | Метрика | Mean score | Mean gain | Median gain | Positive subjects | Bootstrap 95% CI gain |
|---|---|---:|---:|---:|---:|---:|
| head-only | accuracy | 0.304318 | +0.007656 | +0.006093 | 69.81% | [0.004084, 0.011650] |
| head-only | balanced accuracy | 0.268405 | +0.001218 | +0.001659 | 62.26% | [-0.001074, 0.003497] |
| head-only | macro F1 | 0.227764 | +0.004323 | +0.004360 | 64.15% | [0.000578, 0.008337] |
| full-model | accuracy | 0.313779 | +0.017117 | +0.010217 | 69.81% | [0.009000, 0.027363] |
| full-model | balanced accuracy | 0.271537 | +0.004350 | +0.003019 | 58.49% | [-0.000372, 0.011044] |
| full-model | macro F1 | 0.230011 | +0.006569 | +0.003974 | 64.15% | [0.001499, 0.012117] |

Balanced accuracy остаётся статистически неопределённой по bootstrap CI для
обоих методов. Accuracy и macro F1 имеют положительные CIs для обоих методов.

## Устойчивость на уровне субъектов

| Метод | Метрика | Улучшение ≥2 seeds | Улучшение во всех 3 | Mean within-subject seed SD gain |
|---|---|---:|---:|---:|
| head-only | accuracy | 38/53 (71.70%) | 20/53 (37.74%) | 0.011111 |
| head-only | balanced accuracy | 33/53 (62.26%) | 16/53 (30.19%) | 0.009825 |
| head-only | macro F1 | 32/53 (60.38%) | 15/53 (28.30%) | 0.012694 |
| full-model | accuracy | 36/53 (67.92%) | 23/53 (43.40%) | 0.013497 |
| full-model | balanced accuracy | 33/53 (62.26%) | 15/53 (28.30%) | 0.013273 |
| full-model | macro F1 | 34/53 (64.15%) | 17/53 (32.08%) | 0.013905 |

Head-only устойчивее по accuracy-доле «минимум 2 seeds» и имеет меньшую
внутрисубъектную seed variability. Full-model даёт более крупные средние
приросты и немного более высокую устойчивость macro F1.

## Парное сравнение методов после усреднения seeds

| Сравнение | Метрика | Mean difference | Positive subjects | Bootstrap 95% CI |
|---|---|---:|---:|---:|
| full-model − head-only | accuracy | +0.009461 | 69.81% | [0.003578, 0.016908] |
| full-model − head-only | balanced accuracy | +0.003132 | 58.49% | [-0.000605, 0.008995] |
| full-model − head-only | macro F1 | +0.002247 | 58.49% | [-0.001304, 0.005975] |

Преимущество full-model над head-only подтверждается положительным CI только
для accuracy; для приоритетных macro F1 и balanced accuracy CIs пересекают
ноль. Поэтому результат не сводится к выбору метода по accuracy.

## Анализ по источникам

Субъекты распределены в три взаимоисключающие группы: exclusive `gpn_data`,
exclusive `Old_EEG` и `both`. Общие субъекты не дублируются в двух
source-specific группах. Complete-case состав: 11 `gpn_data`, 12 `Old_EEG`,
30 `both`.

### Exclusive `gpn_data`

| Метод | Accuracy (gain) | Balanced accuracy (gain) | Macro F1 (gain) |
|---|---:|---:|---:|
| zero-shot | 0.284978 | 0.259088 | 0.219081 |
| head-only | 0.291455 (+0.006477) | 0.261027 (+0.001939) | 0.218890 (-0.000191) |
| full-model | 0.305358 (+0.020380) | 0.262687 (+0.003599) | 0.222757 (+0.003676) |

Для full-model доли улучшения минимум в двух seeds составляют 54.55%,
63.64% и 54.55% по accuracy, balanced accuracy и macro F1; соответствующие
mean within-subject seed SD gains — 0.015890, 0.016146 и 0.013879. Для
head-only доли равны 63.64%, 63.64% и 45.45%, а SD — 0.013509, 0.010267
и 0.012686. Full-model accuracy gain CI положителен
`[0.002771, 0.042968]`; CIs balanced accuracy и macro F1 пересекают ноль.

### Exclusive `Old_EEG`

| Метод | Accuracy (gain) | Balanced accuracy (gain) | Macro F1 (gain) |
|---|---:|---:|---:|
| zero-shot | 0.305384 | 0.287484 | 0.238062 |
| head-only | 0.311080 (+0.005696) | 0.284378 (-0.003107) | 0.237664 (-0.000398) |
| full-model | 0.321052 (+0.015668) | 0.283176 (-0.004309) | 0.239392 (+0.001330) |

Для full-model доли улучшения минимум в двух seeds составляют 66.67%,
50.00% и 50.00%; mean seed SD gains — 0.014681, 0.015300 и 0.017327. Для
head-only доли равны 66.67%, 33.33% и 41.67%, а SD — 0.013478, 0.013125
и 0.014732. Full-model accuracy gain CI положителен
`[0.002466, 0.029171]`, но balanced accuracy имеет отрицательную точечную
оценку, а CIs balanced accuracy и macro F1 пересекают ноль.

### Субъекты `both`

У 30 общих субъектов full-model gains равны `+0.016499`, `+0.008089`,
`+0.009726`, а head-only — `+0.008872`, `+0.002683`, `+0.007866` для
accuracy, balanced accuracy и macro F1 соответственно. Для full-model
улучшение минимум в двух seeds наблюдается у 73.33%, 66.67% и 73.33%
субъектов. Эта группа анализируется отдельно и не усиливает искусственно
exclusive source-выборки.

Положительный accuracy gain наблюдается в обеих exclusive source-группах,
то есть overall accuracy-эффект не определяется одним источником. Однако
balanced accuracy и macro F1 на небольших exclusive группах неоднородны;
это ограничивает вывод о межисточниковой обобщаемости.

## Порог accuracy 0.75

Ни один из 53 complete-case субъектов ни для одного метода не достиг
accuracy `≥0.75` хотя бы в одном seed. Следовательно, также нулевы числа
субъектов с порогом минимум в двух seeds, во всех трёх seeds и со средней
accuracy по seeds `≥0.75`. Результат одинаков для exclusive `gpn_data`,
exclusive `Old_EEG` и `both`.

## Leakage, split и checkpoint audit

- `split_consistency_audit.csv`: 162 строки `fold × subject × seed`;
  все consistency flags истинны.
- Для каждого `fold × subject` число уникальных hashes между seeds равно 1
  для outer train, inner train, inner validation, calibration, adaptation
  train, adaptation validation, final evaluation и preprocessing.
- В каждом fold один preprocessing hash и три разных global checkpoint
  hashes.
- Суммы `global_target_overlap`, `calibration_evaluation_overlap`,
  `adaptation_validation_overlap`, `evaluation_overlap` и
  `duplicate_sample_ids` равны нулю.
- `checkpoint_audit.csv`: 486 строк, из них 477 валидных завершённых.
  Во всех валидных строках initial checkpoint совпадает с global checkpoint,
  initial predictions совпадают с global predictions; head-only не изменяет
  frozen backbone; zero-shot final hash равен global hash.
- Все 326 889 probability vectors конечны и находятся в `[0, 1]`;
  максимальная ошибка суммы равна `1.6764e-7`; duplicate
  `seed × fold × subject × method × sample_id` отсутствуют.

## Артефакты

Полный run:

```text
benchmark_results/calibration_label_q5_multiseed_20pct/20260725_172240
```

В корне run сохранены manifest/progress, provenance, global/subject/source,
per-seed/multiseed/stability/paired/threshold сводки, split/checkpoint audits,
failures/incomplete tables и unified `predictions.parquet`. В `seed_7`,
`seed_42`, `seed_2026` сохранены resolved configs и все стандартные
single-seed fold/subject артефакты.

Smoke:

```text
benchmark_results/calibration_label_q5_multiseed_20pct_smoke/20260725_172142
```

## Ограничения и вывод

Проверены только feature-based MLP, `label_q5` и calibration budget 20%;
результат не переносится автоматически на другие архитектуры, targets,
бюджеты или источники. Bootstrap CIs отражают subject-level uncertainty, но
не являются отдельной поправленной множественной проверкой гипотез.

Full-model удовлетворяет критериям устойчивого эффекта: mean gains
положительны, accuracy и macro F1 CIs не пересекают ноль, большинство
субъектов улучшаются минимум в двух seeds, а accuracy gain положителен в
обеих exclusive source-группах. Balanced accuracy остаётся неопределённой.
Head-only также устойчив: accuracy и macro F1 CIs положительны, а accuracy
улучшается минимум в двух seeds у большей доли субъектов и с меньшей
seed variability, но средние gains ниже. Для основного следующего baseline
рекомендуется full-model, а head-only следует сохранять как более дешёвую и
несколько более стабильную альтернативу. Следующий научно полезный шаг —
leakage-safe cross-source transfer benchmark с теми же фиксированными
split/model-seed контрактами, поскольку текущие exclusive source-подгруппы
малы и не дают чистой проверки переноса между источниками.
