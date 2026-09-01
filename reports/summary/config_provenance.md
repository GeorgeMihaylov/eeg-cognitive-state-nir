# Provenance конфигураций экспериментов

## 1. Канонический PM-regression baseline

Исходный вопрос: завершённый пятифолдовый Random Forest baseline был отражён
в реестре и отчёте, но не имел отдельного source YAML.

Evidence:

- resolved config и manifest:
  `benchmark_results/pm_regression_baseline_5fold/20260724_121853/`;
- итоговый отчёт: `reports/integration/pm_multioutput_regression.md`;
- commit завершённого результата: `733a85c`;
- текущий SHA-256 входного Parquet:
  `26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8`.

Принятое решение: создан
`experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml`. Он
воспроизводит сохранённый протокол: 43 174 complete-case окна, 53 субъекта,
448 EEG+POW признаков, семь targets в каноническом порядке, mean comparator и
Random Forest (`n_estimators=20`, `max_depth=8`, `random_state=42`,
`n_jobs=-1`), пять GroupKFold folds по `subject_id`, без target scaling.
Явное `preprocessing: none` только документирует отсутствие дополнительной
обработки и не меняет протокол.

Ограничение: historical manifest не содержит dataset hash. Указанный hash
вычислен для текущего файла, поэтому он подтверждает текущий dry-load, но не
является независимым доказательством байтовой идентичности файла в момент
historical run.

Затронуты новый PM YAML, curation, experiment registry и сводные отчёты.

## 2. LSTM/BiLSTM и gap-aware варианты

Исходный вопрос: обычные
`groupkfold_torch_lstm_label_q5.yaml` и
`groupkfold_torch_bilstm_label_q5.yaml` оставались `needs_evidence`.

Evidence:

- LSTM runtime:
  `benchmark_results/groupkfold_torch_lstm_label_q5/20260714_150259/`;
- BiLSTM runtime:
  `benchmark_results/groupkfold_torch_bilstm_label_q5/20260714_150400/`;
- оба каталога содержат полные five-fold summaries, predictions и fold
  artifacts;
- source configs введены commit `50d35ac`.

Принятое решение: оба обычных конфига — отдельные завершённые historical
baselines (`keep_as_legacy`, `canonical_status: historical`). Они не
объявлены superseded, поскольку это было бы более сильным утверждением, чем
позволяет evidence.

Gap-aware LSTM/BiLSTM остаются научно более безопасными reference-вариантами:
они явно проверяют временные разрывы и используют group-record inner
validation. Обычные и gap-aware результаты нельзя смешивать как один
протокол.

## 3. Transformer seeds

Исходный вопрос: primary YAML фиксирует seed 42, тогда как реестр указывает
`[7, 42, 123]`.

Evidence:

| seed | run | evidence |
|---:|---|---|
| 7 | `20260716_191618` | `benchmark_results/groupkfold_torch_transformer_label_q5_seed7/20260716_191618/run_manifest.json` |
| 42 | `20260716_191246` | `benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246/run_manifest.json` |
| 123 | `20260716_191837` | `benchmark_results/groupkfold_torch_transformer_label_q5_seed123/20260716_191837/run_manifest.json` |

Resolved configs подтверждают одинаковый протокол и изменение model,
validation, evaluation и task random state. Итоговые команды и результаты
зафиксированы в `reports/transformer_benchmark_report.md`.

Принятое решение: provenance имеет статус `documented`; отдельные seed YAML
не создаются. External orchestration и три runtime manifests теперь явно
указаны в curation и experiment registry.

## 4. Preprocessing ablation seeds

Исходный вопрос: matrix YAML фиксирует seed 42, а реестр содержит
`[7, 42, 123]`.

Evidence:

- `reports/preprocessing_selected_trials_multiseed.md` сохраняет команды
  `--experiment-matrix ... --seed 7` и `--seed 123`;
- `cli.py` принимает `--seed`;
- `bench/experiments/preprocessing_ablation.py` переносит override в model,
  validation, evaluation и task random state;
- representative runtime manifests существуют для seed 7 и 123;
- seed 42 использует исходную matrix и legacy full A–H runtime.

Принятое решение: provenance имеет статус `documented`. Seed 7 и 123
считаются внешней orchestration поверх неизменённого matrix YAML; создавать
два дополнительных source YAML не требуется. Полный A–H factorial остаётся
выполненным только для seed 42; multi-seed сравнение ограничено trials A/B/E.

## 5. Канонические конфиги без tracked-отчётных ссылок

Пять запрошенных конфигов проверены по runtime и tracked evidence:

| config | canonical status | linked evidence |
|---|---|---|
| `experiments/statistical_analysis.yaml` | completed | `reports/statistical_model_comparison.md`, `benchmark_results/analysis/` |
| `experiments/automl_transformer_label_q5.yaml` | completed | `reports/automl_transformer_pilot.md`, AutoML pilot runtime |
| `configs/groupkfold_rf_label_q5.yaml` | completed | `reports/transformer_benchmark_report.md`, RF runtime |
| `experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml` | completed | final personalization report and runtime |
| `configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml` | completed | raw-CNN report, seed-42 runtime and sibling seed configs |

`completed` означает наличие завершённого результата и evidence, а не
обязательство повторно обучать модель. AutoML остаётся завершённым
диагностическим pilot, а не финальным nested model-selection результатом.

## 6. Изменения experiment registry

- обе PM baseline entries теперь ссылаются на канонический full config;
- для PM явно сохранён порядок семи targets;
- Transformer seed metadata содержит manifests seeds 7/42/123;
- preprocessing metadata содержит механизм CLI override и representative
  runtimes;
- ShallowConvNet metadata связывает primary config с sibling seed configs.

Научные метрики, статусы результатов и параметры завершённых запусков не
изменялись.

## 7. Оставшиеся ограничения

- historical PM manifest не содержит dataset hash;
- обычные LSTM/BiLSTM не имеют отдельного tracked итогового отчёта, поэтому
  их provenance опирается на runtime artifacts и Git history;
- preprocessing A–H не является полным трёхсидовым factorial: seeds 7/123
  покрывают только выбранные A/B/E;
- runtime artifacts остаются ignored и не предназначены для Git.

## 8. Итоговый статус конфигурационной системы

Конкретные пробелы этапа 10Б.2Б закрыты без перемещения существующих YAML и
без унификации loader contracts. Явно классифицированы:

- completed canonical: PM regression, statistical analysis, AutoML pilot,
  RF label_q5, classification personalization, ShallowConvNet;
- active canonical: нет новых назначений на этом этапе;
- planned canonical: нет;
- historical canonical: обычные LSTM и BiLSTM.

Остаются два явно зафиксированных historical ограничения PM run: исходная
CLI-команда и dataset hash не были сохранены в runtime manifest. Resolved
config, artifacts и текущий dataset hash достаточны для канонической точки
повторного запуска, но эти два значения не реконструируются предположением.
