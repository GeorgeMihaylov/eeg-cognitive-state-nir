# Ordinal Transformer technical smoke

Дата: 2026-07-18

Ветка: `feature/ordinal-transformer`

Базовая инфраструктура: commit `42f845c`

## 1. Цель технической проверки

Проверен полный путь `BenchmarkRunner` для одного и того же EEG-only набора
последовательностей и трёх выходных частей `torch_transformer`: `categorical`,
`coral`, `corn`. Проверка включала group split, внутреннюю record-group validation,
train-only normalization, три эпохи, восстановление лучшего состояния,
стандартные predictions/metrics/checkpoint artifacts и независимую strict reload
проверку.

## 2. Ограничения интерпретации

Использованы только outer fold 1, seed 42, EEG-only и максимум три эпохи. Подбора
параметров, дополнительных seeds, POW-only, EEG+POW, cross-source, calibration и
полного пятифолдового сравнения не было.

> Результаты получены на ограниченном техническом запуске и не являются оценкой научного качества методов.

Метрики ниже приведены только как доказательство работоспособности pipeline. Они
не используются для ранжирования categorical, CORAL и CORN.

## 3. Конфигурация

Общий encoder: input `[B,8,168]`, `d_model=128`, четыре attention heads, два
Transformer layers, FFN 256, GELU, dropout 0.1, learned positional encoding и last
pooling. Общие параметры обучения: AdamW, batch size 128, learning rate 0.001,
weight decay 0.0001, validation fraction 0.15, patience 4, seed 42 и `device=auto`.
Фактически использовалась CUDA: NVIDIA GeForce RTX 5060 Ti.

Различался только `head_type`. Число параметров: categorical — 305 029, CORAL —
304 517, CORN — 304 900.

## 4. Размеры train, validation и test

Безопасного лимита последовательностей после канонического sequence builder в
репозитории нет. Существующий `max_windows` действует до построения
последовательностей и изменил бы канонический путь. Поэтому лимит 2 000 не
применялся; использован полный outer fold 1.

| Partition | Sequences | Classes 0/1/2/3/4 |
|---|---:|---|
| Outer train до inner split | 35 342 | — |
| Inner train | 30 038 | 5 569 / 6 075 / 6 075 / 6 251 / 6 068 |
| Inner validation | 5 304 | 949 / 974 / 1 167 / 1 123 / 1 091 |
| Outer test | 8 800 | 2 184 / 1 813 / 1 631 / 1 548 / 1 624 |

Outer train содержит 43 субъектов; после выделения validation inner train содержит
42 субъекта. Inner validation включает 14 record groups от 13 субъектов. Outer
test содержит 11 субъектов.

## 5. Общий индекс последовательностей

Полный индекс: 44 142 последовательности и 53 субъекта. Канонический SHA-256:

```text
1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa
```

Общий технический split manifest построен один раз после канонического индекса.
Его semantic SHA-256 по sequence identity, outer fold, metadata, target и
train/validation/test membership:

```text
6201df099c015dbbf08cfdea20eea2aa2b80f4e8904020032e859061b6aa5c32
```

Hash одинаков для всех heads.

## 6. Проверка разбиений и alignment

Outer train/test subject overlap равен нулю. Inner train/validation
`record_group_id` overlap равен нулю; inner/outer-test record overlap также равен
нулю. Validation manifests трёх trials совпали точно.

Между categorical, CORAL и CORN проверены `sequence_id`, `fold`, `subject_id`,
`record_id`, `source`, `target_sample_id`, `target_time`, `y_true` и `split`.
Каждая колонка имеет 0 расхождений на 8 800 test-последовательностях.

Normalization mean и scale совпали с максимальной абсолютной разностью 0.0;
feature order и feature-list hash также совпали.

## 7. Результаты обучения

| Head | Train loss epochs 1/2/3 | Validation loss epochs 1/2/3 | Best epoch / loss | Training time |
|---|---|---|---|---:|
| categorical | 1.308097 / 1.164583 / 1.087686 | 1.384661 / 1.390573 / 1.358291 | 3 / 1.358291 | 8.20 s |
| CORAL | 0.428300 / 0.371629 / 0.340719 | 0.463336 / 0.416609 / 0.463400 | 2 / 0.416609 | 7.70 s |
| CORN | 0.466579 / 0.412036 / 0.383498 | 0.498803 / 0.468891 / 0.478365 | 2 / 0.468891 | 7.50 s |

Все losses и learning rates конечны. Loss не был константным. После обучения все
параметры конечны; изменились соответственно 6, 8 и 6 параметрических tensors
выходной части. В checkpoint сохранено лучшее, а не последнее состояние.

## 8. CORAL cutpoints

Cutpoints в лучшем checkpoint (epoch 2):

```text
[-1.5061053, -0.3681812, 0.8537537, 2.1491194]
```

Они строго возрастают; минимальный соседний gap равен 1.1379241. Cutpoints также
сохранены для каждой эпохи в `training_log.csv`.

## 9. CORN risk sets

Во всех трёх эпохах inner-train risk counts равны:

```text
[30038, 24469, 18394, 12319]
```

Выполняется `risk_count_0 >= ... >= risk_count_3`, первый risk set не пуст.

## 10. Проверка вероятностей

| Head | Shape | Minimum class p | Max `abs(sum(p)-1)` | Max monotonicity violation | Corrections |
|---|---|---:|---:|---:|---:|
| categorical | `[8800,5]` | 4.77e-6 | 1.68e-7 | n/a | n/a |
| CORAL | `[8800,5]` | 5.11e-4 | 1.30e-7 | 0.0 | 0 |
| CORN | `[8800,5]` | 1.96e-5 | 1.06e-7 | 0.0 | 0 |

Все class и threshold probabilities конечны. Содержательных нарушений
монотонности и отрицательных вероятностей нет. Expected rank лежит в `[0.1692,
3.9731]` для CORAL и `[0.1229,3.9735]` для CORN. Максимальная разность при
повторном суммировании сохранённых float32 threshold probabilities равна 2.24e-7,
что находится внутри audit tolerance 1e-6.

## 11. Проверка правила `y_pred`

Повторный расчёт `count(q_k >= 0.5)` дал 0 расхождений с сохранённым `y_pred` для
CORAL и CORN. Повторный расчёт `sum(q_k)` также совпал в пределах float32
погрешности. Диагностический `ordinal_argmax` не подменяет primary prediction.

Доля `ordinal_argmax != y_pred`: CORAL — 36.6818% (3 228 строк), CORN — 21.4432%
(1 887 строк). Это техническая диагностика разных правил декодирования.

## 12. Технические метрики

| Head | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | AUC | Kappa | QWK | Ordinal MAE | Adjacent accuracy | Severe error | Expected-rank MAE | Rank Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| categorical | 0.3428 | 0.3455 | 0.3478 | 0.3480 | 0.6790 | 0.1795 | 0.4556 | 1.0835 | 0.6922 | 0.3078 | — | — |
| CORAL | 0.3251 | 0.3245 | 0.3249 | 0.3275 | 0.6518 | 0.1557 | 0.4541 | 1.0885 | 0.6885 | 0.3115 | 1.0598 | 0.4570 |
| CORN | 0.3085 | 0.3144 | 0.3121 | 0.3078 | 0.6586 | 0.1381 | 0.4177 | 1.0826 | 0.7115 | 0.2885 | 1.0748 | 0.4143 |

AUC получил `[N,5]` class probabilities, QWK — primary `y_pred`, expected-rank
метрики — `expected_rank`. В test присутствуют все пять классов, поэтому все
перечисленные метрики конечны.

## 13. Checkpoint save/reload

Каждый `model.pt` загружен в заново созданную factory-модель через strict
`load_state_dict`. Для всех heads:

- `y_pred` mismatches: 0;
- maximum class-probability delta: 0.0;
- maximum threshold-probability delta: 0.0 для ordinal heads;
- maximum expected-rank delta: 0.0 для ordinal heads;
- checkpoint `head_type` совпадает с resolved config;
- все checkpoint tensors конечны.

Автоматическое преобразование весов между heads отсутствует и продолжает
проверяться unit tests.

## 14. Категориальная совместимость

Categorical trial использовал прежние CrossEntropy, argmax и softmax пути.
Checkpoint сохранил `classifier.0.*`, `classifier.1.*`, `classifier.4.*`.
Формат categorical predictions не получил фиктивных threshold-колонок. Точного
предыдущего run с теми же fold/subset/3 epochs не найдено; strict-load legacy
checkpoint и точное старое forward-поведение продолжают покрываться тестами.

## 15. Предупреждения

Baseline до EEG: `280 passed, 10 warnings`; финальный suite: `289 passed, 10
warnings`. Полный перечень десяти instances:

1. `test_cross_source_metrics_include_severe_error_rate` — `UserWarning: y_pred contains classes not in y_true`.
2. `test_source_summary_does_not_treat_overlap_as_independent_subjects` — то же предупреждение.
3. `test_subject_metrics_and_explicit_auc_policy` — то же предупреждение.
4. `test_runner_executes_small_lazy_raw_eeg_smoke` — то же предупреждение.
5. `test_source_summary_does_not_treat_overlap_as_independent_subjects` — `UserWarning: A single label was found in y_true and y_pred...`.
6. Тот же тест и тот же single-label warning.
7. Тот же тест и тот же single-label warning.
8. Тот же тест и тот же single-label warning.
9. Тот же тест — `UndefinedMetricWarning: y1, y2 and labels have only one label in common; cohen_kappa_score is undefined ... nan`.
10. Тот же тест и тот же `UndefinedMetricWarning`.

Относительно прежних восьми warnings добавление QWK вызывает на искусственном
одно-классовом source slice ещё один single-label `UserWarning` и один
`UndefinedMetricWarning`. Значение остаётся `NaN`, а не заменяется нулём. Это не
связано с формой probabilities, checkpoint loading, скрытым преобразованием NaN
или устаревшим PyTorch save API. Фильтры предупреждений не добавлялись. Во время
реального EEG smoke новых warnings не было.

## 16. Обнаруженные и исправленные проблемы

- Специального ordinal experiment resolver не было: добавлен тонкий matrix/plan/
  resume/audit слой, вызывающий стандартный runner.
- Безопасного post-sequence limit нет: вместо нового несовместимого sampler выбран
  полный fold 1.
- Training log не содержал learning rate, CORAL cutpoints и CORN risk counts:
  добавлены нейтральные diagnostics без нового training loop.
- Не было автоматической reload/probability/exact-alignment проверки smoke runs:
  добавлены технические manifests и audits.
- Добавлен `fold_manifest.json`; стандартные artifacts не заменялись.

## 17. Готовность к полному эксперименту

Технические критерии выполнены: три heads прошли runner, losses конечны, best
checkpoints strict-load совместимы, probabilities валидны, alignment и
normalization точны, cutpoints/risk sets корректны, ordinal metrics сохраняются.
Статус: готово к проектированию полного эксперимента 6Г, но он не запускался.

Generated outputs находятся в `benchmark_results/ordinal_transformer_smoke/` и
не добавляются в Git. Основные run directories:

- `runs/categorical_eeg_only/20260718_152907`;
- `runs/coral_eeg_only/20260718_152926`;
- `runs/corn_eeg_only/20260718_152944`.

## 18. Точные изменения перед задачей 6Г

Модельный или runner bug перед полным запуском не обнаружен. Для 6Г потребуется
отдельная full-experiment конфигурация: пять folds вместо `[1]`, утверждённое
число эпох, EEG-only и EEG+POW по проекту 6А, сохранение того же sequence index и
парный analysis на одинаковых predictions. Следует заранее зафиксировать политику
seeds и обработку неопределённого QWK в малых описательных подгруппах. Технический
smoke config не должен переиспользоваться как научный full config.

Команды:

```powershell
python cli.py `
  --ordinal-transformer-experiment experiments\ordinal_transformer_smoke.yaml `
  --plan-only --verbose

python cli.py `
  --ordinal-transformer-experiment experiments\ordinal_transformer_smoke.yaml `
  --run --resume --verbose
```

Исходный Parquet сохранил SHA-256
`26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8`.
`.gitignore` не изменён.
