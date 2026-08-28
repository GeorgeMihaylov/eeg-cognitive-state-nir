# Финальная унификация Python-пакетов

## 1. Архитектура до и после

Исходная точка: ветка `refactor/final-package-unification-20260828`, коммит
`3b50f35d45522cb6faa1c40da10409b32089b081`. До миграции канонические модели
находились в корневом `model_zoo/`, `cogstate.model_zoo` был facade с несколькими
application-расширениями, scientific AutoML находился в `bench.automl`, а
personalized portfolio — в отдельном корневом `automl/`.

Итоговые границы:

```text
apps ─────────> cogstate
scripts ──────> bench / apps
bench ────────> cogstate
cogstate ──X──> bench
```

`cogstate` теперь владеет моделями и переиспользуемыми алгоритмическими
примитивами. `bench` владеет datasets, tasks, splits, scientific execution,
analysis, artifact generation и двумя явно разделёнными видами AutoML.
Крупная reusable-логика десяти исторических scripts перенесена в `bench`; на
старых script-путях оставлены только тонкие CLI-entrypoints.

### Текущее состояние Git

На финальной проверке `git status --porcelain=v1` содержит 137 modified,
44 deleted и 29 компактных untracked entries. Последнее число относится к
отображению каталогов в status, а не к числу файлов: каноническая команда
`git ls-files --others --exclude-standard` возвращает ровно **46 untracked
files**. Staging пуст. Diff охватывает 180 tracked paths; актуальная статистика
приведена в разделе 11.

## 2. Отображение старых и новых пакетов

| Старый путь | Новый путь | Семантика |
|---|---|---|
| `model_zoo.base` | `cogstate.model_zoo.base` | model adapter contract |
| `model_zoo.factory` | `cogstate.model_zoo.factory` | единственная model factory |
| `model_zoo.DL.*` | `cogstate.model_zoo.DL.*` | Torch, DANN, contrastive, ordinal, regression, fusion |
| `model_zoo.ML.*` | `cogstate.model_zoo.ML.*` | sklearn, multitask, XGBoost personalization |
| `bench.meta.buffers` | `cogstate.adaptation.meta_learning.buffers` | functional state и buffer policy |
| `bench.meta.protocol` | `cogstate.adaptation.meta_learning.protocol` | model-independent FOMAML protocol types |
| `bench.meta.fomaml` | `cogstate.adaptation.meta_learning.fomaml` | FirstOrderMAML core |
| PM calibration functions в experiment | `cogstate.adaptation.regression_calibration` | bias/affine calibration primitives |
| `bench.automl.*` | `bench.automl.scientific.*` | nested scientific optimization |
| `automl.*` | `bench.automl.personalized.*` | per-user portfolio/adaptation |

## 3. Отображение символов

`build_model`, `TORCH_MODEL_NAMES`, `SKLEARN_MODEL_NAMES`, Torch adapters и все
model classes импортируются только через `cogstate.model_zoo`. Сохранены имена
`torch_mlp`, `torch_lstm`, `torch_bilstm`, `torch_eegnet`,
`torch_shallow_convnet`, `torch_shallow_convnet_multitask`,
`torch_shallow_fusion`, `torch_transformer` и все sklearn names.

Существовавшие application-расширения не потеряны: masked multitask adapter
находится в `DL.multitask_adapter`, ShallowConvNet multitask extension — в
`DL.shallow_multitask`, streaming adapters и weight loader остаются локальными
частями `cogstate.model_zoo`. Дублирующих реализаций классов не осталось.

## 4. Dependency graph

AST-аудит всех Python-файлов подтверждает:

- абсолютных импортов корневых `model_zoo`, `automl` и `src` нет;
- executable/import-time зависимости `cogstate -> bench` нет;
- streaming worker использует `cogstate.model_zoo`;
- dynamic Python module paths и активные YAML/JSON/TOML manifests не ссылаются
  на старые package names;
- все 64 Python-файла в `scripts/` имеют не более 100 строк и не определяют
  классов; reusable-логика находится в `bench`.

Эти условия зафиксированы в `tests/test_architecture_boundaries.py`.

## 5. Удалённые корневые пакеты

Полностью удалены `model_zoo/` и `automl/`, включая stale bytecode. Корневой
`src/` отсутствовал до задачи и не создавался. Compatibility shim-пакеты на
старых путях не добавлялись.

Новые 46 файлов распределены так:

- 26 файлов `bench`: три analysis-модуля, scientific AutoML (6), personalized
  AutoML (10), два data-quality, два dataset CLI-core, два feature CLI-core и
  target-contract builder;
- 5 файлов `cogstate.adaptation`: четыре meta-learning core и regression
  calibration;
- 12 файлов `cogstate.model_zoo`: десять DL и два ML модуля;
- два architecture audit report;
- один personalized AutoML test.

Среди них нет `__pycache__`, `*.pyc`, runtime outputs, временных логов,
predictions, benchmark results или cache artifacts. Имена
`cog_bci_window_cache_cli.py` и `raw_eeg_window_cache_cli.py` обозначают
исходный CLI-код, а не файлы кэша.

## 6. FOMAML и adaptation

Dataset-independent functional state, BatchNorm buffer policies,
`FOMAMLConfig`, `FirstOrderMAML` и model compatibility logic перенесены в
`cogstate.adaptation.meta_learning`. Episode construction, materialization,
dataset views, production/synthetic orchestration и audits остаются в
`bench.meta`.

Из PM personalization experiment вынесены только `AffineCalibration`,
fit/apply bias correction, fit/apply affine calibration и минимальная array
validation. Fold planning, chronology, metrics и artifact logic не переносились.
Численные regression-calibration тесты и FOMAML synthetic/buffer tests проходят.

## 7. Scientific и personalized AutoML

`bench.automl.scientific` содержит прежние search space, objective, trial
resolver, study runner и artifacts. `bench.automl.personalized` содержит
отдельные application split, registry, portfolio, staged search и adaptation
semantics. Общий `bench.automl.__init__` экспортирует только два namespace и не
смешивает objectives или split policies.

Stale personalized bindings исправлены на локальный registry и канонические
`cogstate.model_zoo` builders. CLI и dynamic config-audit loader используют
`bench.automl.scientific.study_runner`. Отдельные тесты подтверждают разделение
namespace, EEG candidate registry и хронологический disjoint inner split.

## 8. Metadata и hash impact

PyTorch state-dict/checkpoint payloads и model parameters/defaults не менялись.
FOMAML `architecture_schema_signature` исторически включал Python class path.
После физического переноса helper `stable_model_class_path` нормализует только
одобренный переход `cogstate.model_zoo.* -> model_zoo.*` внутри этого legacy
architecture identity. Regression-тест строит прежний payload и подтверждает
побайтово тот же SHA-256. Import shim при этом отсутствует.

Module/source paths входят в некоторые implementation/protocol fingerprints.
Их возможное изменение для будущего plan-build после физического переноса не
маскируется: алгоритм, targets, folds и inputs прежние, но implementation
location является новым фактом. Уже существующие manifests, checkpoints,
predictions и benchmark results не переписывались.

## 9. Изменённые активные module paths

- DANN raw protocol implementation file list: `model_zoo.DL.*` заменён на
  `cogstate.model_zoo.DL.*`;
- PM all-targets feature baseline model-factory descriptor теперь указывает на
  `cogstate.model_zoo.build_model`;
- config-audit dynamic loader указывает на
  `bench.automl.scientific.study_runner`;
- active `reports/summary/config_curation.yaml` и
  `reports/summary/requirements_registry.yaml` evidence-ссылки переведены на
  существующие новые code paths.

Experiment YAML, fold manifests, target definitions и scientific parameters не
изменялись.

## 10. Намеренно сохранённый historical provenance

Старые Markdown-отчёты и исторические JSON summaries, описывающие фактический
путь реализации во время завершённых экспериментов, не переписывались. Поэтому
в них могут оставаться упоминания `model_zoo/...`; это historical provenance, а
не активный import/config path. Runtime artifacts под `benchmark_results/` и
данные не открывались для миграционной записи.

## 11. Проверки

- `compileall bench cogstate apps scripts tests`: успешно;
- architecture boundaries: 15 passed; scientific/personalized AutoML вместе с
  boundary-проверками до двух финальных dependency guards: 32 passed;
- model/adapter/EEGNet/MLP/LSTM/ShallowConvNet/Transformer/ordinal/regression/
  fusion/DANN/contrastive/streaming/FOMAML/calibration: 324 passed, 2 skipped;
- COG-BCI inventory после переноса library consumer: 33 passed;
- перенесённые target/channel/PM-quality helpers: 91 passed, один отказ из-за
  отсутствующего `data/processed/windowed_eeg_pm_dataset_w10.parquet`;
- полный baseline до рефакторинга: 1540 passed, 11 skipped, 80 failed,
  74 errors;
- сохранённый ранее промежуточный summary `1549 passed, 11 skipped, 81 failed,
  74 errors` содержал один неповторившийся failure:
  `tests.test_feature_group_regression::test_runner_executes_groupkfold_regression_and_standard_artifacts`.
  Traceback завершался `StopIteration` в проверке буквального имени родительского
  каталога unified `predictions.parquet`; это path-layout assertion, чувствительный
  к runtime placement/portable-path fallback, а не ошибка импорта, API или
  научного расчёта;
- этот node проходит отдельно и в итоговом JUnit full-run. Итоговый результат:
  **1552 passed, 11 skipped, 80 failed, 74 errors** из 1717 tests. Относительно
  baseline добавлены 12 новых architecture/personalized-AutoML тестов, и все 12
  проходят;
- строгое сравнение сохранённых baseline/final JUnit node IDs даёт:
  `failures only in baseline = 0`, `failures only in final = 0`,
  `errors only in baseline = 0`, `errors only in final = 0`;
- обязательный дешёвый финальный набор
  `test_architecture_boundaries.py + test_model_factory.py + test_personalized_automl.py`:
  **26 passed**; повторный `compileall bench cogstate apps scripts tests` успешен;
- UTF-8 чтение `bench/automl/personalized/__init__.py` подтверждает корректный
  символ em dash (`U+2014`): подстроки `вЂ` и replacement character `U+FFFD` в
  файле нет, наблюдавшееся искажение было только отображением консоли PowerShell;
- оставшиеся общие failures/errors относятся к
  отсутствующим local data, caches, checkpoints, generated CSV/reports,
  external archives и известным byte-exact LF/CRLF checks; новых import/API
  errors нет.

`git diff --check` не выявил whitespace errors.

Финальный `git diff --shortstat`: **180 files changed, 4102 insertions(+),
17914 deletions(-)**.

## 12. Ограничения

Этот worktree не содержит полного локального `data/`, `benchmark_results/`,
external archives, generated report tables, `AGENTS.md` и `PROJECT_CONTEXT.md`,
которые требуются части repository-wide tests. Поэтому full suite не может быть
полностью зелёным без восстановления runtime/project artifacts. Рефакторинг не
создаёт и не пересчитывает эти артефакты. Scientific training и smoke training
не запускались, поскольку задача меняет package layout, а не модель или протокол.
