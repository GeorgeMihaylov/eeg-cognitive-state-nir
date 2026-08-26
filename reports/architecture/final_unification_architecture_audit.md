# Аудит архитектуры итоговой ветки

Дата аудита: 2026-08-26  
Ветка: `integration/final-unification-20260826`  
Исходный HEAD: `ecce69fd26470ef685d08b5ef229521ed5daef22`

## Границы аудита

Аудит выполнен до изменения production-кода. Проверены деревья `bench/`,
`cogstate/`, `model_zoo/`, `automl/`, `apps/`, `scripts/` и `src/`, импорты,
тестовые call sites и документированные исторические команды. Научные
контракты, конфигурации экспериментов, protocol/plan hashes и содержимое
`benchmark_results/` не изменяются этой работой.

## Слои до рефакторинга

| Слой | Фактическая ответственность | Проблема |
|---|---|---|
| `model_zoo/` | Канонические sklearn/PyTorch модели benchmark | Общие модели почти полностью скопированы в `cogstate/model_zoo/` |
| `cogstate/model_zoo/` | Streaming/application factory и multitask-модель | Вместе с нужным multitask-кодом содержит копии adapter, EEGNet, MLP, LSTM и ShallowConvNet |
| `bench/automl/` | Fold-safe scientific optimization, study artifacts и trial resolution | Назначение не отделено документально от top-level `automl/` |
| `automl/` | Application portfolio/orchestration и staged search | Имеет собственные application split/search abstractions без явного архитектурного контракта |
| `bench/validation/` | Канонические scientific splits и метрики | Метрики и GroupKFold-подобный цикл повторены в application-слое |
| `cogstate/evaluation/` | Application evaluation/latency | Содержит вторые classification metrics и второй cross-subject training loop |
| `bench/meta/` | Каноническая episodic/FOMAML infrastructure | Ранний независимый meta-learning prototype остаётся в task mixins |
| `model_zoo/DL/dann.py`, DANN experiments | Каноническая DANN/GRL infrastructure | Ранний DANN prototype остаётся в task mixins |
| `model_zoo/DL/contrastive.py`, contrastive experiments | Каноническая contrastive infrastructure | Ранний contrastive prototype остаётся в task mixins |
| personalization experiments и shared adapter | Каноническая leakage-safe personalization | В `cogstate/adaptation/` есть неиспользуемые упрощённые prototypes |
| `cogstate/features/` | Канонические target-free generic EEG features (371) | Исторический большой feature builder находится в `src/` |
| `bench/features/` | Dataset/cache-specific feature materialization | Допустимый второй слой, но его границы ранее не были явно зафиксированы |
| `cogstate/preprocessing/` | Канонические reusable filters/windowing/FASTER/MNE-FASTER | В `bench/preprocessing/` остались старые generic дубли |
| `bench/preprocessing/` | Fold/cache/experiment orchestration | Наряду с orchestration содержит старые `filters/features/artifacts/pipeline` |
| `cogstate/ingestion/` | Generic parsing и canonical records | Большие исторические parsers/builders остаются в `src/` |
| `bench/datasets/` | Benchmark datasets, materialization и caches | Граница с историческими builders в `src/` размыта |
| `src/` | Смешанный набор ingestion, QC, builders и reporting | 19 самостоятельных скриптов, многие содержат 300–2800 строк reusable logic |

## Решение о канонических реализациях

| Потенциальный дубль | Каноническая реализация | Классификация второго варианта | Планируемое действие |
|---|---|---|---|
| Общие neural/sklearn models | `model_zoo/` | Реальный duplicate в `cogstate/model_zoo/` | Заменить общие файлы thin re-export; оставить только application multitask extension |
| Scientific AutoML | `bench/automl/` | Отличается по ответственности от application AutoML | Сохранить оба слоя, явно документировать границу и запрет outer-test selection |
| Classification metrics и group CV | `bench/validation/` | Реальный duplicate в `cogstate/evaluation/` | Делегировать canonical metrics/split helpers; оставить latency/application facade |
| FOMAML | `bench/meta/` | Historical prototype mixin | Удалить неиспользуемый prototype |
| DANN/GRL | Shared encoder + `model_zoo/DL/dann.py` + experiments | Historical prototype mixin | Удалить неиспользуемый prototype |
| Contrastive learning | `model_zoo/DL/contrastive.py` + experiments | Historical prototype mixin | Удалить неиспользуемый prototype |
| Personalization | Benchmark personalization execution + shared adapter | Три неэкспортируемых prototype-файла в `cogstate/adaptation/` | Удалить; сохранить используемый `feature_alignment.py` |
| Generic feature extraction | `cogstate/features/` | `bench/features/` dataset-specific; `src/08` historical builder | Сохранить dataset adapters, вынести legacy builder в package module, CLI сделать тонким |
| Generic preprocessing | `cogstate/preprocessing/` | Старые generic `bench/preprocessing` файлы | Удалить неиспользуемые дубли; сохранить fold/cache orchestration |
| Exact MNE-FASTER и FASTER-like cleanup | Раздельные модули `cogstate/preprocessing/` | Не дубль: разные научные семантики | Сохранить оба варианта и явно не смешивать |
| Dataset parsing/materialization | `cogstate/ingestion/` + `bench/datasets/` | Большие исторические `src/00..12` | Перенести reusable logic в package modules; оставить только thin compatibility commands |

## Статический duplication audit

Обнаружены совпадающие крупные классы/функции для PyTorch adapter, EEGNet,
ShallowConvNet, MLP, LSTM, sequence helpers, model base и sklearn builders в
двух model-zoo деревьях. Обнаружены независимые ранние реализации GRL/DANN,
contrastive loop и meta-learning loop в `bench/tasks/mixin/`. Обнаружены
повторные classification metrics и cross-subject split/training loop в
`cogstate/evaluation/`. Старые generic preprocessing modules в
`bench/preprocessing/` не имеют production call sites.

Повторяются также immutable constants: список семи PM в
`cogstate/protocol.py`, `bench/tasks/target_registry.py` и
`bench/analysis/target_registry_audit.py`. Каноническим источником принимается
`cogstate.protocol`; registry и audit должны импортировать его. Четырёхканальный
`EEG_CHANNELS` в CLARE/CL-Drive не является дублем Emotiv-14: это отдельный
dataset contract и остаётся локальным.

## Полная классификация и план миграции `src/`

Старые пути сохраняются только как тонкие compatibility entry points там, где
они входят в provenance завершённых экспериментов или вызываются тестами.
Алгоритмический код после миграции находится в importable package modules.

| Старый файл | Категория | Новый package module | Основной CLI | Планируемый статус старого пути |
|---|---|---|---|---|
| `src/00_inventory_data.py` | A/D | `bench.data_quality.data_inventory` | `scripts/data/inventory_data.py` | thin wrapper |
| `src/01_inspect_emotiv_files.py` | A/D | `bench.data_quality.emotiv_file_inspection` | `scripts/data/inspect_emotiv_files.py` | thin wrapper |
| `src/02_build_emotiv_catalog.py` | A/B | `bench.datasets.emotiv_catalog_builder` | `scripts/data/build_emotiv_catalog.py` | thin wrapper; historical provenance |
| `src/03_validate_catalog_and_columns.py` | D | `bench.data_quality.emotiv_catalog_validation` | `scripts/data/validate_emotiv_catalog.py` | thin wrapper |
| `src/04_build_windowed_pm_dataset.py` | B | `bench.datasets.emotiv_pm_window_builder` | `scripts/data/build_emotiv_pm_windows.py` | thin wrapper; historical provenance |
| `src/08_build_eeg_features.py` | C | `bench.features.legacy_emotiv_eeg_features` | `scripts/data/build_legacy_emotiv_features.py` | thin wrapper; historical provenance |
| `src/09_audit_raw_eeg.py` | D | `bench.data_quality.raw_eeg_audit` | `scripts/data/audit_raw_eeg.py` | thin wrapper; historical provenance |
| `src/10_build_raw_eeg_window_cache.py` | B/E | existing `bench.datasets.raw_eeg_window_dataset` | `scripts/data/build_raw_eeg_window_cache.py` | thin wrapper; historical provenance |
| `src/11_audit_logical_recordings.py` | D | `bench.data_quality.logical_recording_audit` | `scripts/data/audit_logical_recordings.py` | thin wrapper |
| `src/12_audit_raw_eeg_artifacts.py` | D | `bench.data_quality.raw_eeg_artifact_audit` | `scripts/data/audit_raw_eeg_artifacts.py` | thin wrapper |
| `src/13_run_preprocessing_ablation.py` | E | existing `bench.experiments.preprocessing_ablation` | `scripts/run_preprocessing_ablation.py` | thin wrapper; historical provenance |
| `src/14_audit_robust_feature_scaling.py` | D/F | `bench.analysis.robust_feature_scaling_audit` | `scripts/analysis/audit_robust_feature_scaling.py` | thin wrapper |
| `src/15_build_experiment_summary.py` | F | `bench.analysis.experiment_summary` | `scripts/analysis/build_experiment_summary.py` | thin wrapper; tests/provenance |
| `src/16_audit_experiment_configs.py` | D/F | `bench.analysis.experiment_config_audit` | `scripts/analysis/audit_experiment_configs.py` | thin wrapper; tests/provenance |
| `src/17_build_requirements_coverage.py` | F | `bench.analysis.requirements_coverage` | `scripts/analysis/build_requirements_coverage.py` | thin wrapper; tests/provenance |
| `src/18_build_colleague_metrics_package.py` | F | `bench.analysis.colleague_metrics_package` | `scripts/analysis/build_colleague_metrics_package.py` | thin wrapper; tests/provenance |
| `src/19_build_project_final_package.py` | F | existing `bench.analysis.final_project_package` | `scripts/analysis/build_project_final_package.py` | thin wrapper |
| `src/20_build_pm_union_raw_cache.py` | B | `bench.datasets.pm_union_raw_materialization` | `scripts/data/build_pm_union_raw_cache.py` | thin wrapper |
| `src/21_run_preliminary_streaming_handoff.py` | E | existing `bench.experiments.preliminary_streaming_handoff` | `scripts/run_preliminary_streaming_handoff.py` | thin wrapper |

## Проверенные call sites и compatibility risks

- `src/02`, `src/04` и `src/08` записаны в target-provenance reports/configs;
  удаление путей нарушило бы воспроизводимость, поэтому нужны wrappers.
- `src/15`–`src/18` импортируются тестами как исторические executable modules;
  wrappers должны продолжать экспортировать их публичные функции, а не только
  `main()`.
- Streaming worker импортирует `cogstate.model_zoo.factory`, multitask model и
  weights; application facade обязателен.
- `cogstate.adaptation.feature_alignment` используется robust-shrinkage
  experiments и остаётся неизменным.
- `bench/features` содержит COG-BCI spectral/cache semantics и не должен быть
  механически перенесён в generic feature pipeline.
- `bench/preprocessing/fold_artifact_transform.py`, cache builders и COG-BCI
  preprocessing являются orchestration/dataset code, а не generic duplicates.
- Замена import paths внутри runtime configs или изменение serialized class
  names может повлиять на checkpoint compatibility; facade paths сохраняются.

## Предполагаемый набор production-изменений

1. Thin model-zoo facade и выделенный application multitask extension.
2. Удаление четырёх неиспользуемых task mixins и трёх adaptation prototypes.
3. Delegating application evaluation facade поверх `bench.validation`.
4. Явные docstrings/контракты для двух AutoML слоёв.
5. Удаление неиспользуемых generic preprocessing duplicates.
6. Миграция reusable кода из всех 19 `src/` файлов с сохранением thin wrappers.
7. Импорт общих PM constants из `cogstate.protocol`.
8. Обновление README только для структуры и entry points.

## Неизменяемые инварианты

- семь PM и их порядок;
- outer-train-only Q3 low/medium/high;
- fixed participant-disjoint folds и target-specific masks;
- `record_group_id`-aware inner validation и participant-macro metrics;
- raw shape `[B, 1, 14, 2560]`, 256 Hz, 10 s;
- target-free 371-feature schema;
- protocol/plan/condition/run/checkpoint identities;
- существующие scientific/application artifacts и численные результаты.

## Результат реализации

- Общие PyTorch/sklearn классы в `cogstate.model_zoo` заменены re-export
  facades; единственный локальный model extension — multitask ShallowConvNet и
  его masked-label adapter для streaming.
- Scientific metrics и group splitting удалены из `cogstate.evaluation`.
  Application package сохраняет только latency и external-fold guards;
  канонический scientific слой — `bench.validation`.
- Четыре historical task mixins, три неиспользуемых adaptation prototypes и
  четыре generic preprocessing duplicates удалены после проверки call sites.
- `bench.automl` и `automl` сохранены как разные слои: nested scientific
  optimization и application portfolio orchestration соответственно.
- Все 19 исторических `src/*.py` сохранены как wrappers длиной 13–22 строки.
  Алгоритмы размещены в указанных выше package modules, а актуальные CLI — в
  `scripts/data`, `scripts/analysis` и `scripts/`.
- Список семи PM теперь импортируется registry и target audit из
  `cogstate.protocol`; статический поиск показывает одно определение.
- Scientific configs, experiment IDs/hashes, данные, checkpoints и
  `benchmark_results` не изменялись.

После каждого блока должны быть выполнены targeted tests; итоговая проверка
включает `compileall`, import/model/CLI smoke и `git diff --check`.
