# Leakage-safe fine-tuning для нового пользователя: smoke

## Итог

В существующий `UserCalibrationExperiment` добавлен feature-window путь для
`torch_mlp`; отдельный calibration runner не создавался. Pilot выполнен для
`8191f1d9` на fold 05, seed 42, пяти дробных бюджетах и трёх режимах. Все 15
условий завершились, но это техническая проверка одного пользователя, а не
оценка ожидаемого эффекта персонализации.

Лучшее значение pilot дал `full_finetuning` при бюджете 20%: accuracy
`0.3523`, balanced accuracy `0.3165`, macro F1 `0.2652`. Порог accuracy
`0.75` не достигнут. Рост accuracy не интерпретируется отдельно от balanced
accuracy и macro F1.

## Аудит старого TransferMixin

Прототип из `feature/benchmarking` не был runnable-задачей: он не был
зарегистрирован, не имел experiment config и полного source/target протокола.
Главный методический дефект воспроизведён ранее: второй вызов обычного `fit`
восстанавливал начальное случайное состояние адаптера. Полученная модель была
calibration-only baseline, а не продолжением обучения pretrained модели.
Прямой перенос старого runner/mixin не выполнялся.

## Реализованный путь

`TorchClassificationAdapter.fine_tune()` требует fitted/loaded model и обучает
текущие параметры новым AdamW optimizer без вызова `fit`, переинициализации
модели или переобучения preprocessing. Поддерживаются:

- `no_adaptation` (`zero_shot` внутри legacy-compatible runner);
- `head_only_finetuning` (`head_only`), где MLP явно публикует префикс
  последнего `Linear`;
- `full_finetuning` (`full_model`).

Каждое условие получает независимый clone одного fold checkpoint.
`global_checkpoint_hash` и `fine_tune_initial_hash` во всех 15 случаях равны
`0b58a00728cb4d2f7f7e37e3fd0a18bb9ddd8a85901eba7a616abb16b74bc2c2`;
предсказания clone до первого update совпадают с global predictions. Для
`no_adaptation` initial/final hash совпадают. Head-only обновляет 645
параметров и оставляет 147 840 параметров неизменными; frozen hash совпал
до/после во всех бюджетах. Full fine-tuning допускает обновление всех 148 485
параметров и использует learning rate `0.0001`, head-only — `0.001`.

## Global и target split

Канонический Parquet содержит 45 384 размеченных окна, 54 испытуемых и 448
EEG+POW признаков. Глобальный fold 05 содержит 43 outer-train и 11 outer-test
испытуемых. Внутри outer train выполнен `group_holdout` по `subject_id`:
36 испытуемых / 30 843 окна в inner train и 7 испытуемых / 5 435 окон в
inner validation. `8191f1d9` отсутствует в обеих частях.

У target subject 742 окна; классы: `{0: 297, 1: 99, 2: 109, 3: 132, 4: 105}`.
Доля начала выбирается отдельно в каждом `(source, record_id)` после сортировки
по `t_start`, затем `sample_id`. Для сопоставимости бюджетов final evaluation
зафиксирован как поздний suffix после максимального 20% префикса: 599 окон,
классы `{0: 189, 1: 81, 2: 103, 3: 122, 4: 104}`. Окна между меньшим
префиксом и этим suffix резервируются и не используются при обучении или
оценке данного условия.

| Budget | Calibration | Actual fraction | Adapt train/validation | Final evaluation | Calibration classes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0% | 0 | 0.0000 | 0 / 0 | 599 | none |
| 1% | 6 | 0.0081 | 4 / 2 | 599 | `0:5, 1:1` |
| 5% | 33 | 0.0445 | 26 / 7 | 599 | `0:32, 1:1` |
| 10% | 69 | 0.0930 | 53 / 16 | 599 | `0:61, 1:4, 2:2, 3:2` |
| 20% | 143 | 0.1927 | 111 / 32 | 599 | `0:108, 1:18, 2:6, 3:10, 4:1` |

Бюджеты не увеличиваются до минимального размера молча. Малое class coverage
оставлено как реальное свойство few-shot calibration. Adaptation validation
является последней 20%-частью calibration pool; final evaluation не
используется для early stopping.

## Preprocessing и leakage audit

`standard_clip` с границами q0.5/q99.5 fitted только на 30 843 inner-train
окнах. Один frozen state затем применён к global validation, target
calibration и final evaluation. `fit_target_overlap = 0`, hash preprocessing
state — `6a91f3646ec77989c3161192103ea5d9d20638934077007131f7d6a821e91364`.

Во всех 15 строках audit:

- global target overlap: 0;
- calibration/evaluation overlap: 0;
- adaptation train/validation overlap: 0;
- adaptation/final-evaluation overlap: 0;
- duplicate sample IDs: 0.

## Метрики pilot

Метрики до адаптации рассчитаны на одном и том же 599-window suffix и поэтому
одинаковы для всех бюджетов: accuracy `0.3205`, balanced accuracy `0.2984`,
macro F1 `0.2418`.

| Budget | Method | Calibration | Train/val | Accuracy before → after | Balanced accuracy before → after | Macro F1 before → after | Epochs | Best validation loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | no adaptation | 0 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 0% | head only | 0 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 0% | full | 0 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 1% | no adaptation | 6 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 1% | head only | 6 | 4/2 | 0.3205 → 0.3306 | 0.2984 → 0.3041 | 0.2418 → 0.2506 | 3 | 1.8722 |
| 1% | full | 6 | 4/2 | 0.3205 → 0.3439 | 0.2984 → 0.3140 | 0.2418 → 0.2617 | 3 | 1.8190 |
| 5% | no adaptation | 33 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 5% | head only | 33 | 26/7 | 0.3205 → 0.3356 | 0.2984 → 0.3073 | 0.2418 → 0.2524 | 3 | 0.4869 |
| 5% | full | 33 | 26/7 | 0.3205 → 0.3489 | 0.2984 → 0.3149 | 0.2418 → 0.2599 | 3 | 0.3603 |
| 10% | no adaptation | 69 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 10% | head only | 69 | 53/16 | 0.3205 → 0.3356 | 0.2984 → 0.3039 | 0.2418 → 0.2427 | 3 | 1.1066 |
| 10% | full | 69 | 53/16 | 0.3205 → 0.3506 | 0.2984 → 0.3145 | 0.2418 → 0.2630 | 3 | 1.0443 |
| 20% | no adaptation | 143 | 0/0 | 0.3205 → 0.3205 | 0.2984 → 0.2984 | 0.2418 → 0.2418 | 0 | — |
| 20% | head only | 143 | 111/32 | 0.3205 → 0.3322 | 0.2984 → 0.3018 | 0.2418 → 0.2418 | 3 | 0.7810 |
| 20% | full | 143 | 111/32 | 0.3205 → 0.3523 | 0.2984 → 0.3165 | 0.2418 → 0.2652 | 3 | 0.7638 |

Accuracy `0.75` не достигнута ни в одном условии. Full fine-tuning улучшил все
три основные метрики во всех ненулевых бюджетах и выглядит лучшим кандидатом
этого одного pilot. Head-only даёт небольшое улучшение при 1–5%, но при 10–20%
macro F1 почти не меняется. Эти различия не являются доказательством
преимущества метода: один пользователь и три эпохи не позволяют оценить
межсубъектную устойчивость.

## Конфигурация, устройство и артефакты

Global model обучена 3 эпохи; лучший global validation loss `1.53994` на эпохе
1. Устройство pilot: `NVIDIA GeForce RTX 5060 Ti` (`cuda`). Runtime:

- base run: `benchmark_results/calibration_label_q5_smoke/base_run/20260725_141123`;
- personalization run:
  `benchmark_results/calibration_label_q5_smoke/personalization/20260725_141627`;
- сводки: `calibration_summary.csv`, `calibration_subject_metrics.csv`;
- audits: `calibration_split_audit.csv`, `checkpoint_audit.csv`;
- unified predictions: `predictions.parquet`;
- исходный checkpoint target: `fold_05/8191f1d9/global_model.pt`;
- для каждого budget/method: `config.yaml`, before/after metrics and
  predictions, calibration/evaluation IDs, `training_log.csv`,
  `fine_tuning_summary.json`, `model.pt`.

Команда:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' cli.py `
  --calibration-experiment experiments\calibration\label_q5_finetuning_smoke.yaml `
  --subject-limit 1 `
  --max-calibration-epochs 3 `
  --verbose
```

## Проверки и ограничения

Targeted suite: `47 passed`. Полный suite после реализации:
`441 passed, 11 warnings`. Вероятности всех 8 985 unified prediction rows
конечны и суммируются до единицы; идентификаторы условий уникальны.

Рекомендуемый следующий эксперимент — сохранить все четыре ненулевых бюджета
и сравнить в первую очередь `full_finetuning` с `no_adaptation` на всех
outer-test subjects, оставив head-only как дешёвый контроль. Нужны
subject-level paired оценки и несколько seeds; выбирать learning rate/epochs
по final evaluation нельзя. PM-регрессия не является следующим обязательным
этапом этой ветки: сначала следует подтвердить классификационный эффект
персонализации. Из старого TransferMixin концептуально переиспользована только
идея «source checkpoint → независимая target adaptation»; код и ошибочный
повторный `fit` не переносились.
