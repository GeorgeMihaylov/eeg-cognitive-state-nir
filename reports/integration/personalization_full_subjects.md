# Полный межсубъектный эксперимент персонализации `label_q5`

## Итог

Leakage-safe персонализация выполнена на CUDA для всех 54 испытуемых и всех пяти
внешних `GroupKFold`-фолдов, seed 42. На каждом фолде обучалась ровно одна
глобальная `torch_mlp`, после чего независимые копии одного и того же лучшего
checkpoint использовались для `zero_shot`, `head_only` и `full_model` при бюджетах
0%, 1%, 5%, 10% и 20%.

Лучший средний результат получен у `full_model`, 20%: accuracy `0.3139`,
balanced accuracy `0.2699`, macro F1 `0.2270`. Относительно zero-shot это
соответственно `+0.0192`, `+0.0045` и `+0.0100`. Subject-level bootstrap CI
исключает ноль для accuracy (`[+0.0088, +0.0312]`) и macro F1
(`[+0.0038, +0.0172]`), но не для balanced accuracy
(`[-0.0007, +0.0105]`). Это описывает устойчивость оценки в данном seed, но не
заменяет многосидовую проверку.

Порог accuracy 0.75 не достигнут ни одним испытуемым ни в одном условии. Поэтому
требование 75% систематически не выполнено; высокая accuracy не скрывает здесь
низкие balanced accuracy или macro F1, поскольку даже единичных случаев выше
порога нет.

## Связь с аудитом `TransferMixin`

Старый `TransferMixin` из `feature/benchmarking` не был перенесён как исполняемый
путь. Аудит показал, что повторный обычный `fit()` реинициализировал модель и
превращал адаптацию в calibration-only baseline. Использован реализованный в
задаче 9Б.1 путь `TorchClassificationAdapter.fine_tune()`: он продолжает обучение
уже fitted/loaded модели новым AdamW optimizer, не вызывает повторный `fit()` и
не переобучает preprocessing. Из старого подхода сохранена только корректная
идея «source checkpoint → независимая target adaptation».

## Протокол

- Данные: `data/processed/windowed_eeg_pm_dataset_w10.parquet`.
- Supervised-контракт: 45 384 окна, 54 испытуемых, 448 EEG+POW признаков,
  `label_q5`, пять классов, без повторной дискретизации.
- Внешняя оценка: пять `GroupKFold` по `subject_id`; каждый испытуемый входит
  ровно в один outer-test fold.
- Внутренняя валидация глобальной модели: group-aware holdout по `subject_id`,
  36/7 субъектов в folds 01, 02, 03 и 05 и 37/7 в fold 04.
- Модель: MLP `448 → 256 → 128 → 5`, dropout 0.3, 148 485 параметров.
- Global training: AdamW, максимум 8 эпох, patience 3, learning rate 0.001.
- Fine-tuning: максимум 5 эпох; head-only learning rate 0.001, full-model
  learning rate 0.0001. Head-only обновляет 645 параметров и сохраняет
  неизменными 147 840 параметров backbone.
- Preprocessing: `standard_clip`, q0.5/q99.5, fit только на global inner train.
  Один frozen state применяется к inner validation, target calibration,
  adaptation validation и final evaluation.

Для каждого target subject окна детерминированно упорядочены по записи и времени.
Первые floor(20%) образуют максимальный calibration pool, оставшийся suffix —
фиксированную final evaluation. Бюджеты 1%, 5% и 10% являются вложенными
префиксами внутри максимального pool; неиспользованный остаток pool не
возвращается в evaluation. Calibration subset делится хронологически 80/20 на
adaptation train/validation. При недостатке validation используется
`none_fixed_epochs`; final evaluation никогда не участвует в early stopping.

## Глобальные модели

| Fold | Outer train/test subjects | Inner train/val subjects | Global epochs | Best epoch | Best validation loss | Fit time, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 43 / 11 | 36 / 7 | 6 | 3 | 1.5741 | 5.15 |
| 02 | 43 / 11 | 36 / 7 | 4 | 1 | 1.6646 | 3.11 |
| 03 | 43 / 11 | 36 / 7 | 5 | 2 | 1.5738 | 3.45 |
| 04 | 44 / 10 | 37 / 7 | 4 | 1 | 1.6273 | 3.13 |
| 05 | 43 / 11 | 36 / 7 | 4 | 1 | 1.5399 | 2.94 |

Всего выполнено пять, а не 54, global training. Суммарное время их `fit` —
17.78 с.

## Полнота условий

Создано 810 записей `subject × budget × method`: 779 условий завершены с
предсказаниями, 24 имеют статус `insufficient_calibration_samples`, семь —
`insufficient_evaluation_samples`, ошибок обучения и non-finite результатов нет.
Все 54 субъекта представлены.

Один субъект (`9192c107`, всего шесть окон) не имеет достаточной final evaluation
ни при одном бюджете. Ещё восемь субъектов (`0001508a`, `30c140ca`, `40f0714a`,
`5001d09a`, `7092f07b`, `71a251fa`, `d142d110`, `e0c0408a`) не имеют минимальных
пяти calibration samples только при бюджете 1%; zero-shot для них сохранён.
Иными словами: failed subjects — 0, полностью неоцениваемых — 1, частично
недостаточных — 8. Среди частично недостаточных четыре относятся только к
`gpn_data`, четыре только к `Old_EEG`; полностью недостаточный субъект представлен
в обоих источниках.

## Overall: средние метрики по испытуемым

Zero-shot одинаков для всех бюджетов: accuracy `0.2947`, balanced accuracy
`0.2654`, macro F1 `0.2170` (`n=53`).

| Method | Budget | n | Accuracy | Balanced accuracy | Macro F1 | Δ accuracy | Δ balanced | Δ macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head_only | 1% | 45 | 0.2992 | 0.2667 | 0.2206 | +0.0029 | +0.0002 | +0.0008 |
| head_only | 5% | 53 | 0.2965 | 0.2656 | 0.2206 | +0.0019 | +0.0001 | +0.0036 |
| head_only | 10% | 53 | 0.2988 | 0.2657 | 0.2209 | +0.0041 | +0.0003 | +0.0039 |
| head_only | 20% | 53 | 0.3043 | 0.2677 | 0.2249 | +0.0096 | +0.0022 | +0.0079 |
| full_model | 1% | 45 | 0.3023 | 0.2659 | 0.2201 | +0.0061 | -0.0006 | +0.0003 |
| full_model | 5% | 53 | 0.3062 | 0.2675 | 0.2228 | +0.0116 | +0.0021 | +0.0059 |
| full_model | 10% | 53 | 0.3108 | 0.2697 | 0.2260 | +0.0162 | +0.0043 | +0.0090 |
| full_model | 20% | 53 | 0.3139 | 0.2699 | 0.2270 | +0.0192 | +0.0045 | +0.0100 |

Доля испытуемых с положительным gain при 20%:

| Method | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| head_only | 71.7% | 67.9% | 64.2% |
| full_model | 67.9% | 54.7% | 66.0% |

Медианные paired gains:

| Method | Budget | Median Δ accuracy | Median Δ balanced | Median Δ macro F1 |
| --- | ---: | ---: | ---: | ---: |
| head_only | 1% | +0.0008 | +0.0006 | +0.0004 |
| head_only | 5% | +0.0018 | +0.0023 | +0.0046 |
| head_only | 10% | +0.0027 | +0.0010 | +0.0048 |
| head_only | 20% | +0.0055 | +0.0031 | +0.0076 |
| full_model | 1% | +0.0000 | -0.0002 | -0.0008 |
| full_model | 5% | +0.0056 | +0.0023 | +0.0027 |
| full_model | 10% | +0.0056 | +0.0038 | +0.0039 |
| full_model | 20% | +0.0083 | +0.0019 | +0.0037 |

Head-only улучшает испытуемых чаще по accuracy и balanced accuracy, но
full-model даёт больший средний эффект. Парная разность `full_model − head_only`
при 20% составляет `+0.0096` accuracy (bootstrap CI `[+0.0018, +0.0193]`),
`+0.0023` balanced accuracy (`[-0.0022, +0.0073]`) и `+0.0021` macro F1
(`[-0.0026, +0.0072]`). Уверенное преимущество full-model над head-only
наблюдается только для обычной accuracy.

## Bootstrap-интервалы paired gain относительно zero-shot

Интервалы рассчитаны на уровне испытуемого, 1000 resamples, bootstrap seed 42.

| Method | Budget | Accuracy gain, 95% CI | Balanced gain, 95% CI | Macro F1 gain, 95% CI |
| --- | ---: | --- | --- | --- |
| head_only | 1% | +0.0029 [-0.0017, +0.0073] | +0.0002 [-0.0030, +0.0031] | +0.0008 [-0.0035, +0.0046] |
| head_only | 5% | +0.0019 [-0.0028, +0.0061] | +0.0001 [-0.0033, +0.0036] | +0.0036 [-0.0008, +0.0083] |
| head_only | 10% | +0.0041 [-0.0007, +0.0085] | +0.0003 [-0.0032, +0.0038] | +0.0039 [-0.0007, +0.0085] |
| head_only | 20% | +0.0096 [+0.0050, +0.0140] | +0.0022 [-0.0012, +0.0058] | +0.0079 [+0.0032, +0.0132] |
| full_model | 1% | +0.0061 [-0.0032, +0.0175] | -0.0006 [-0.0062, +0.0058] | +0.0003 [-0.0063, +0.0074] |
| full_model | 5% | +0.0116 [+0.0023, +0.0213] | +0.0021 [-0.0032, +0.0074] | +0.0059 [-0.0003, +0.0123] |
| full_model | 10% | +0.0162 [+0.0068, +0.0267] | +0.0043 [-0.0012, +0.0103] | +0.0090 [+0.0026, +0.0163] |
| full_model | 20% | +0.0192 [+0.0088, +0.0312] | +0.0045 [-0.0007, +0.0105] | +0.0100 [+0.0038, +0.0172] |

Рост обычной accuracy с бюджетом выражен сильнее и устойчивее, чем рост
class-balanced метрик.

## Результаты по источникам

Субъект оценивается отдельно на окнах каждого источника; поэтому 31 общая
идентичность входит в обе source-specific таблицы. Доступны 41 испытуемый в
`gpn_data` и 42 в `Old_EEG`; это не 83 независимых человека. Ещё 11 идентичностей
принадлежат только `gpn_data`, 12 — только `Old_EEG`.

### `gpn_data`

Zero-shot: accuracy `0.2879`, balanced accuracy `0.2531`, macro F1 `0.2086`.

| Method | Budget | n | Accuracy | Balanced accuracy | Macro F1 | Δ accuracy | Δ balanced | Δ macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head_only | 1% | 37 | 0.2900 | 0.2553 | 0.2113 | +0.0034 | +0.0007 | +0.0021 |
| head_only | 5% | 41 | 0.2905 | 0.2543 | 0.2137 | +0.0026 | +0.0012 | +0.0050 |
| head_only | 10% | 41 | 0.2948 | 0.2561 | 0.2157 | +0.0069 | +0.0029 | +0.0071 |
| head_only | 20% | 41 | 0.3002 | 0.2579 | 0.2191 | +0.0123 | +0.0047 | +0.0105 |
| full_model | 1% | 37 | 0.2945 | 0.2555 | 0.2114 | +0.0079 | +0.0009 | +0.0023 |
| full_model | 5% | 41 | 0.3028 | 0.2573 | 0.2167 | +0.0150 | +0.0042 | +0.0080 |
| full_model | 10% | 41 | 0.3093 | 0.2613 | 0.2216 | +0.0214 | +0.0081 | +0.0130 |
| full_model | 20% | 41 | 0.3120 | 0.2622 | 0.2220 | +0.0241 | +0.0091 | +0.0134 |

При 20% все три CI для full-model положительны: accuracy
`[+0.0130, +0.0381]`, balanced accuracy `[+0.0038, +0.0156]`, macro F1
`[+0.0060, +0.0218]`.

### `Old_EEG`

Zero-shot: accuracy `0.2994`, balanced accuracy `0.2660`, macro F1 `0.2061`.

| Method | Budget | n | Accuracy | Balanced accuracy | Macro F1 | Δ accuracy | Δ balanced | Δ macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head_only | 1% | 38 | 0.3030 | 0.2646 | 0.2066 | +0.0007 | +0.0000 | -0.0004 |
| head_only | 5% | 42 | 0.3019 | 0.2706 | 0.2112 | +0.0025 | +0.0046 | +0.0051 |
| head_only | 10% | 42 | 0.3000 | 0.2660 | 0.2091 | +0.0006 | -0.0000 | +0.0030 |
| head_only | 20% | 42 | 0.3060 | 0.2675 | 0.2138 | +0.0066 | +0.0014 | +0.0077 |
| full_model | 1% | 38 | 0.3055 | 0.2659 | 0.2071 | +0.0032 | +0.0013 | +0.0001 |
| full_model | 5% | 42 | 0.3065 | 0.2682 | 0.2100 | +0.0071 | +0.0021 | +0.0039 |
| full_model | 10% | 42 | 0.3083 | 0.2691 | 0.2121 | +0.0089 | +0.0031 | +0.0060 |
| full_model | 20% | 42 | 0.3127 | 0.2678 | 0.2136 | +0.0133 | +0.0018 | +0.0075 |

Эффект на `Old_EEG` слабее: при 20% CI исключает ноль для full-model accuracy
(`[+0.0003, +0.0280]`), но не для balanced accuracy или macro F1. Это
source-stratified наблюдение внутри данного эксперимента, а не доказательство
сенсорного переноса.

## Class coverage

При 1% только один из 45 обученных calibration subsets содержит все пять
классов; распределение числа классов — 8/14/9/13/1 для 1/2/3/4/5 классов. При
5% полных subsets 25 из 53, при 10% — 35 из 53, при 20% — 42 из 53.

Связь macro-F1 gain с числом представленных calibration-классов невелика и
не монотонна: Spearman rho для full-model по бюджетам 1/5/10/20% равен
`-0.032 / 0.232 / 0.120 / -0.016`, для head-only —
`0.120 / 0.230 / 0.085 / -0.017`. Корреляция с числом calibration samples также
слабая или умеренная: full-model `0.183 / 0.177 / 0.257 / 0.151`, head-only
`0.202 / 0.232 / 0.290 / 0.119`.

Class-incomplete calibration особенно заметно ограничивает head-only при 5%:
средний macro-F1 gain `-0.0002` против `+0.0079` у class-complete subsets. Для
full-model при 5% это `+0.0019` против `+0.0103`. Однако при 10–20% эта
зависимость не повторяется строго, поэтому class coverage нельзя считать
единственным объяснением различий.

## Leakage и checkpoint audit

Проверка всех 810 записей дала:

- outer train/test subject overlap: 0;
- preprocessing-fit/outer-test subject overlap: 0;
- global-target sample overlap: 0;
- calibration/final-evaluation overlap: 0;
- adaptation train/validation overlap: 0;
- adaptation/final-evaluation overlap: 0;
- duplicate sample IDs в split audit: 0;
- duplicate `condition + sample_id` и `condition + sequence_id`: 0;
- final evaluation IDs и `y_true` одинаковы для всех доступных условий каждого
  испытуемого;
- 541 051 unified prediction rows совпадают с суммой `n_final_evaluation` по
  завершённым условиям;
- `proba_0`–`proba_4` конечны, лежат в `[1.02e-5, 0.9596]`, максимальное
  отклонение суммы от единицы `1.64e-7`.

Для всех 779 условий с предсказаниями:

- `global_checkpoint_hash == fine_tune_initial_hash`;
- исходные predictions clone совпадают с global predictions;
- все бюджеты одного fold стартуют с одного checkpoint;
- у head-only frozen parameters не изменились;
- у zero-shot initial/final hashes и on-disk before/after predictions совпали.

## CUDA, runtime и resume

- Python 3.11.15, PyTorch 2.11.0+cu128.
- Устройство всех global и fine-tuning запусков:
  `cuda`, NVIDIA GeForce RTX 5060 Ti.
- Пиковая выделенная GPU-память: 21.09 MiB для full-model и 19.37 MiB для
  head-only; CUDA OOM не было, batch fallback не потребовался.
- Среднее число fine-tuning эпох: 4.81 для head-only и 4.84 для full-model.
- Сумма измеренного fine-tuning времени: 10.37 с; большая часть wall time
  приходится на клонирование, inference, метрики и запись condition artifacts.
- Персонализационный этап: 265.21 с; global fit: 17.78 с; полный путь занял
  примерно 283 с (4 мин 43 с) измеренного времени.

CUDA integration smoke предварительно завершил 18/18 условий для двух
source-exclusive субъектов (`0110f12e`, `8191f1d9`), одного фолда, бюджетов
0/5/20%, всех методов, двух global и двух fine-tuning эпох. Повторный запуск с
`--resume` пропустил все 18 условий и переиспользовал global checkpoint.

Full-run resume хранит результаты после каждого условия, проверяет config,
implementation и base-config hashes, пропускает завершённые условия и
переобучает failed/незавершённые. Ошибка отдельного условия сохраняется в
`failures.csv`; системные нарушения CUDA, dataset, leakage, checkpoint или
resume-state останавливают запуск.

## Артефакты и воспроизведение

Runtime-артефакты не входят в Git:

- global base:
  `benchmark_results/calibration_label_q5_full_subjects/global_base/20260725_145514`;
- personalization:
  `benchmark_results/calibration_label_q5_full_subjects/personalization/20260725_145534`;
- общие файлы: `run_manifest.json`, `progress.json`, `failures.csv`,
  `global_fold_summary.csv`, `calibration_summary.csv`,
  `calibration_subject_metrics.csv`, `calibration_split_audit.csv`,
  `checkpoint_audit.csv`, `aggregate_metrics.csv`, `paired_comparisons.csv`,
  `source_summary.csv`, `threshold_75_summary.csv`,
  `subjects_accuracy_ge_075.csv`, `predictions.parquet`;
- condition-level: resolved config, calibration/evaluation sample IDs,
  split/checkpoint audits, before/after metrics и predictions, training log,
  model reference или `model.pt`, completion record.

Команда полного запуска:

```powershell
python cli.py `
  --calibration-experiment experiments\calibration\label_q5_finetuning_full_subjects.yaml `
  --verbose
```

Fraction budgets берутся из YAML. Существующий CLI-флаг
`--calibration-budgets` задаёт бюджеты в секундах и поэтому намеренно не
использовался для этого fractional-протокола.

## Ограничения и следующий этап

Эксперимент использует один seed, одну MLP-архитектуру и только `label_q5`.
Результаты показывают небольшой, но растущий с бюджетом выигрыш; на 20%
full-model лучше всего по всем трём средним метрикам, однако balanced accuracy
остаётся около 0.27, а требование 75% недостижимо. Нельзя подбирать
гиперпараметры по final evaluation или интерпретировать source-stratified
различия как самостоятельный cross-source transfer.

Следующий приоритет — повторить заранее зафиксированный 20%-й протокол как
минимум на seeds 7 и 2026. Если эффект сохраняется, затем проверить
PM-regression personalization и отдельный cross-source transfer; если 75%
остаётся систематически недостижимым, пересмотреть научную формулировку
пятиуровневой классификационной цели, не подгоняя её по этим test-результатам.
