# Confirmatory benchmark семи Performance Metrics: план исполнения

Статус: `confirmatory_plan`. Реальное обучение EEG в рамках подготовки плана не выполнялось.

## Зафиксированный протокол

- Семь PM: attention, engagement, excitement, stress, relaxation, interest, focus.
- Две постановки: outer-train-only Q3 classification и scalar continuous regression.
- Модели: Random Forest, XGBoost, ShallowConvNet, LSTM.
- Пять неизменяемых participant-disjoint outer folds из колонки `outer_fold`.
- Inner validation для Torch: `group_record` по `record_group_id`, seed 42.
- Q3 вычисляется `FoldLocalQuantileTargetTransform` на target-complete outer-train и фиксируется одним hash для всех моделей данного `fold × PM`.
- Feature cache read-only: 34 354 окна, 371 признак, float32, без feature selection и sample entropy.
- Raw contract: `[B, 1, 14, 2560]`, float32, 256 Hz, 10 s, canonical raw preprocessing.
- Sequence contract: length 10, stride 1, last-window target; группы `source + subject_id + record_group_id`; разрыв при gap больше 10.01 s; дополнительная проверка запрещает объединять разные `record_id`.

## Матрица

Полная декартова матрица содержит 280 ячеек (`5 folds × 7 PM × 2 tasks × 4 models`). Поддерживаются 245 training units. Не поддерживаются 35 ячеек `torch_lstm × regression`: текущий factory и adapter предоставляют LSTM только для classification.

Для каждой поддерживаемой training unit предусмотрены две evaluation views без второго checkpoint:

1. `native` — канонический cohort модели;
2. `common_sequence_eligible` — одинаковые LSTM endpoint `sample_id` для всех моделей.

Итого: 245 native и 245 common-cohort evaluations. LSTM native cohort уже является sequence-eligible; для single-window моделей common view выполняет только повторный inference на тех же checkpoint.

Across 35 `fold × PM` cohorts metadata-only аудит дал:

- 220 341 target-complete test-вхождений (сумма по PM и folds);
- 214 123 sequence-eligible test-вхождения;
- 6 218 исключений из-за отсутствующей истории или временных разрывов;
- диапазон native cohort: 5 564–8 066 окон;
- диапазон common cohort: 5 456–7 868 endpoints;
- диапазон исключений: 108–244 на `fold × PM`.

## Checkpoint reuse

Строгий gate проверяет model/task/target/fold/seed, resolved model parameters, preprocessing hash, feature identity, Q3 hash, sequence contract, normalization scope, input/output shapes и SHA-256 checkpoint.

- Reusable: 14 ShallowConvNet (Q3 + regression) и 7 classification LSTM, всего 21 fold-1 units.
- Не reusable: 14 Random Forest и 14 XGBoost fold-1 units. Их preliminary metrics и predictions существуют, но sklearn estimator не был сериализован, поэтому common-cohort inference без переобучения невозможен.
- Unsupported: 7 fold-1 LSTM regression и 35 таких ячеек во всей матрице.
- Новых обучений требуется 224: 196 поддерживаемых units folds 2–5 и 28 sklearn units fold 1.

## Resume и отказоустойчивость

Каждая training unit имеет отдельный `checkpoint_manifest.json`. Resume принимает checkpoint только после полного identity и file-hash gate. Native и common evaluation status ведутся отдельно, поэтому common view можно достроить без повторного fit. Ошибка одной ячейки записывается в `execution_status.csv` и не останавливает оставшуюся матрицу.

## Артефакты

Plan-only создаёт:

- `plan_manifest.json`;
- `training_units.csv`;
- `evaluation_units.csv`;
- `fixed_fold_audit.csv`;
- `cohort_inventory.csv`;
- `common_sequence_eligible_samples.parquet`;
- `checkpoint_reuse_audit.csv`;
- `q3_target_transforms.json`;
- `execution_status.csv`.

После исполнения дополнительно создаются стандартные benchmark artifacts по каждой unit, единый checkpoint manifest, native/common `metrics.json` и `predictions.parquet`, а также:

- `classification_by_fold.csv`;
- `regression_by_fold.csv`;
- `classification_summary.csv`;
- `regression_summary.csv`;
- `pm_macro_summary.csv`;
- `common_cohort_comparison.csv`.

PM-macro считается после fold aggregation; окна разных PM не объединяются в одну метрику. В common comparison сохраняются только парные fold-level разности, без заявления статистической значимости по пяти folds.

## Идентичность

- Fixed-fold hash: `5927bc0a0295de1c8388d450c796a09f058b6b03b930e4a590749718fcd5629b`.
- Protocol hash: `3981d726ace6cb91bc42cc9fa0c04dea89e85d8278cf34bf511969b00d029d76`.
- Run-matrix hash: `e520dcecaffdcc0a4e341e628ddd396d7550a34020821efffc45e2cc5a34db8c`.

## Команды

Во всех командах `$dataRoot`, `$featureCache` и `$preliminaryRoot` задаются runtime-путями и не входят в tracked config.

Dry execution:

```powershell
python scripts\run_pm_confirmatory_benchmark.py --config experiments\pm_confirmatory\selected_models_5fold_v1.json --data-root $dataRoot --feature-cache-dir $featureCache --preliminary-root $preliminaryRoot --plan-only
```

Минимальный будущий smoke (не запускался):

```powershell
python scripts\run_pm_confirmatory_benchmark.py --config experiments\pm_confirmatory\selected_models_5fold_v1.json --data-root $dataRoot --feature-cache-dir $featureCache --preliminary-root $preliminaryRoot --execute --fold 1 --model random_forest --target-id pm_attention_q3_fold_local --resume
```

Полный будущий confirmatory run (не запускался):

```powershell
python scripts\run_pm_confirmatory_benchmark.py --config experiments\pm_confirmatory\selected_models_5fold_v1.json --data-root $dataRoot --feature-cache-dir $featureCache --preliminary-root $preliminaryRoot --execute --resume
```

## Оценка стоимости

Суммарное preliminary training time для 49 поддерживаемых выбранных fold-1 units составило около 1 892.9 s: ShallowConvNet 1 005.1 s, LSTM 36.1 s, Random Forest 650.3 s, XGBoost 201.4 s. При грубой линейной экстраполяции и reuse 21 Torch units ожидаемая сумма нового training wall time — около 8 423 s (2.34 h), без учёта чтения данных, common inference и конкуренции за CPU/GPU. Это ориентир по одному fold, а не гарантия длительности.
