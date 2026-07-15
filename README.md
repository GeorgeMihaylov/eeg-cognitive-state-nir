# EEG Cognitive State Benchmark

Воспроизводимый benchmark для классификации когнитивных состояний по агрегированным EEG/POW-признакам и сырым EEG-сигналам.

Текущая исследовательская ветка:

```text
feature/model-zoo-dl
```

Состояние документации: **15 июля 2026 года**.

## Содержание

- [Цели проекта](#цели-проекта)
- [Основные возможности](#основные-возможности)
- [Данные](#данные)
- [Протокол оценки](#протокол-оценки)
- [Модели](#модели)
- [Структура проекта](#структура-проекта)
- [Подготовка данных](#подготовка-данных)
- [Запуск экспериментов](#запуск-экспериментов)
- [Текущие результаты](#текущие-результаты)
- [Артефакты](#артефакты)
- [Тестирование](#тестирование)
- [Ограничения](#ограничения)
- [Дальнейшая работа](#дальнейшая-работа)

## Цели проекта

Проект решает задачу прогнозирования когнитивных состояний человека по EEG и связанным Performance Metrics.

Основные цели:

1. создать единый воспроизводимый benchmark;
2. сравнивать классические ML- и DL-модели на одинаковых splits;
3. исключить субъектную и сессионную утечку;
4. поддержать два представления данных:
   - агрегированные EEG/POW-признаки;
   - сырые 10-секундные EEG-окна;
5. сохранять конфиги, метрики, предсказания и веса моделей;
6. подготовить основу для cross-source evaluation, user calibration и AutoML.

Основная классификационная задача:

```text
target: label_q5
number of classes: 5
```

Также в обработанном датасете доступна регрессионная цель:

```text
target_focus
```

Полная интеграция регрессионного benchmark остаётся отдельной задачей.

## Основные возможности

В текущей ветке реализованы:

- единый CLI для запуска benchmark;
- dataset registry;
- task registry;
- model factory;
- классические sklearn-модели;
- PyTorch adapter со sklearn-подобным интерфейсом;
- train-only normalization;
- inner validation только внутри outer train;
- early stopping и восстановление лучшего checkpoint;
- случайный оконный split для технического smoke-test;
- честный пятифолдовый `GroupKFold` по `subject_id`;
- сохранение fold-level и агрегированных метрик;
- сохранение предсказаний и DL-артефактов;
- feature-based MLP;
- feature-based LSTM/BiLSTM;
- raw EEG loader и memory-mapped cache;
- EEGNet;
- ShallowConvNet;
- logical-record audit и дедупликация одинаковых сессий между источниками;
- raw EEG preprocessing и cache invalidation;
- synthetic, integration и leakage-тесты.

## Данные

### Исходные источники

Проект использует два собственных источника:

```text
data/raw/gpn_data
data/raw/Old_EEG
```

Оба источника содержат EEG, Performance Metrics и служебные данные, но отличаются структурой каталогов и форматом экспорта.

Исходные данные не хранятся в Git.

### Обработанный feature-based dataset

Основной файл:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

Характеристики:

| Параметр | Значение |
|---|---:|
| Строки | 51 308 |
| Колонки | 508 |
| Субъекты | 55 |
| Записи | 120 |
| `gpn_data` | 27 021 окон |
| `Old_EEG` | 24 287 окон |
| EEG-признаки | 168 |
| POW-признаки | 280 |
| Всего модельных признаков | 448 |
| Размеченные `label_q5` строки | 45 384 |
| Размеченные `target_focus` строки | 45 384 |
| EEG coverage | 100% |
| Полностью пустые EEG-колонки | 0 |
| Infinite values | 0 |
| Duplicate columns | 0 |

Feature-based dataset подходит для:

- Random Forest;
- Logistic Regression;
- SVM;
- XGBoost;
- sklearn MLP;
- PyTorch MLP;
- feature-based LSTM/BiLSTM.

### Raw EEG dataset

Raw pipeline строит 10-секундные EEG-окна:

```text
input shape: [1, 14, 2560]
channels: 14
target sampling rate: 256 Hz
window duration: 10 s
```

Raw-QC:

| Параметр | Значение |
|---|---:|
| Supervised-окон до raw-QC | 45 384 |
| Принято | 45 326 |
| Вне raw-диапазона | 38 |
| Missing fraction > 2% | 20 |

Raw EEG cache хранится в `data/interim` и не добавляется в Git.

### Logical recordings и дедупликация

Аудит источников показал:

| Параметр | Значение |
|---|---:|
| Source-specific records | 119 |
| Logical recordings | 86 |
| Сессии, представленные в обоих источниках | 33 |
| Нарушения outer-fold assignment | 0 |
| Inner train/validation logical overlap | 0 |

Для 33 общих сессий совпадают:

- временные диапазоны;
- labels;
- кэшированные float32 EEG-тензоры на общих supervised-окнах.

После дедупликации остаётся:

```text
30 958 raw EEG windows
54 subjects
86 logical recordings
```

Удаляется 14 368 повторно взвешенных окон.

## Протокол оценки

### Основной научный протокол

```yaml
evaluation:
  protocol: group_kfold_subject
  n_splits: 5
  group_column: subject_id
  random_state: 42
```

Используется:

```python
GroupKFold(n_splits=5)
```

Свойства:

- один субъект не встречается одновременно в train и test;
- каждый supervised sample попадает в test ровно один раз;
- модель создаётся заново для каждого fold;
- train/test subject IDs сохраняются;
- fold indices одинаковы для сравниваемых моделей;
- пересечение субъектов проверяется до обучения;
- нормализация рассчитывается только на train текущего fold;
- inner validation не использует outer test.

Для raw EEG inner validation разделяется по source-independent `record_group_id`, поэтому одна logical recording не попадает одновременно в inner train и validation.

### Random-window sanity split

В проекте сохранён технический протокол:

```text
random_window_stratified_kfold_first_fold
```

Он используется только для smoke-test и проверки pipeline. Субъекты в нём не изолированы, поэтому его метрики нельзя интерпретировать как качество переноса на нового пользователя.

## Модели

### Model factory

Единая точка создания моделей:

```python
from model_zoo import build_model

model = build_model(
    model_name="torch_shallow_convnet",
    task_type="classification",
    input_shape=(1, 14, 2560),
    num_outputs=5,
    params={...},
)
```

Минимальный runtime-интерфейс:

```python
fit(X_train, y_train)
predict(X_test)
predict_proba(X_test)
save(path)
```

### Классические модели

Factory поддерживает классификацию:

```text
random_forest
svm
logistic_regression
mlp
xgboost
```

Имя `mlp` сохранено за sklearn `MLPClassifier` для обратной совместимости.

### PyTorch MLP

Тип:

```text
torch_mlp
```

Базовая архитектура:

```text
448
→ Linear(448, 256) → ReLU → Dropout(0.3)
→ Linear(256, 128) → ReLU → Dropout(0.3)
→ Linear(128, 5)
```

Вход:

```text
[batch, 448]
```

### LSTM/BiLSTM

Тип:

```text
torch_lstm
```

Вход:

```text
[batch, sequence_length, 448]
```

Последовательности формируются только внутри одной записи и не переходят через границы:

```text
source + subject_id + record_id
```

Поддерживаются однонаправленная и двунаправленная конфигурации.

### EEGNet

Тип:

```text
torch_eegnet
```

Вход:

```text
[batch, 1, 14, 2560]
```

EEGNet использует temporal, depthwise spatial и separable convolutions.

### ShallowConvNet

Тип:

```text
torch_shallow_convnet
```

Архитектура:

```text
Temporal Conv2d
→ Depthwise Spatial Conv2d
→ BatchNorm
→ Square
→ Average Pooling
→ Safe Log
→ Dropout
→ Adaptive Average Pooling
→ Linear classifier
```

Характеристики текущей конфигурации:

```text
input shape: [1, 14, 2560]
outputs: 5
trainable parameters: 1 925
```

Модель использует существующий `TorchClassificationAdapter`; отдельный training loop не создавался.

## Структура проекта

```text
.
├── bench/
│   ├── core/
│   ├── datasets/
│   │   ├── emotiv_loader.py
│   │   ├── raw_eeg_window_dataset.py
│   │   ├── raw_preprocessing.py
│   │   └── logical_recordings.py
│   ├── tasks/
│   ├── validation/
│   └── bench_runner.py
├── configs/
├── model_zoo/
│   ├── base.py
│   ├── factory.py
│   ├── ML/
│   └── DL/
│       ├── adapter.py
│       ├── mlp.py
│       ├── lstm.py
│       ├── sequence_utils.py
│       ├── eegnet.py
│       └── shallow_convnet.py
├── src/
│   ├── 00_inventory_data.py
│   ├── 01_inspect_emotiv_files.py
│   ├── 02_build_emotiv_catalog.py
│   ├── 03_validate_catalog_and_columns.py
│   ├── 04_build_windowed_pm_dataset.py
│   ├── 08_build_eeg_features.py
│   ├── 09_audit_raw_eeg.py
│   ├── 10_build_raw_eeg_window_cache.py
│   ├── 11_audit_logical_recordings.py
│   └── 12_audit_raw_eeg_artifacts.py
├── tests/
├── reports/
├── cli.py
└── README.md
```

## Подготовка данных

### 1. Инвентаризация

```powershell
python src/00_inventory_data.py --root .
```

### 2. Проверка Emotiv-файлов

```powershell
python src/01_inspect_emotiv_files.py --root .
```

### 3. Построение каталога

```powershell
python src/02_build_emotiv_catalog.py --root .
```

Основной результат:

```text
data/interim/emotiv_record_catalog.csv
```

### 4. Валидация колонок

```powershell
python src/03_validate_catalog_and_columns.py --root .
```

### 5. PM/POW dataset

```powershell
python src/04_build_windowed_pm_dataset.py --root . --window-s 10
```

### 6. Добавление агрегированных EEG-признаков

```powershell
python src/08_build_eeg_features.py `
  --root . `
  --catalog data/interim/emotiv_record_catalog.csv `
  --pm-dataset data/processed/windowed_pm_dataset_w10.parquet `
  --source all `
  --window-s 10 `
  --output-name windowed_eeg_pm_dataset_w10
```

### 7. Raw EEG audit

```powershell
python src/09_audit_raw_eeg.py
```

Отчёт:

```text
reports/raw_eeg_audit.md
```

### 8. Raw EEG cache

```powershell
python src/10_build_raw_eeg_window_cache.py
```

При совпадении config hash существующие record-level shards переиспользуются.

### 9. Logical-record audit

```powershell
python src/11_audit_logical_recordings.py
```

Отчёт:

```text
reports/logical_recording_audit.md
```

### 10. Artifact audit

```powershell
python src/12_audit_raw_eeg_artifacts.py
```

Отчёт:

```text
reports/raw_eeg_artifact_audit.md
```

## Запуск экспериментов

Активировать окружение:

```powershell
conda activate eeg_benchmark
```

### Random Forest, GroupKFold

```powershell
python cli.py `
  --config configs/groupkfold_rf_label_q5.yaml `
  --models random_forest
```

### PyTorch MLP, GroupKFold

```powershell
python cli.py `
  --config configs/groupkfold_torch_mlp_label_q5.yaml `
  --models torch_mlp
```

### EEGNet, raw deduplicated EEG

```powershell
python cli.py `
  --config configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml `
  --verbose
```

Дополнительные seeds:

```powershell
python cli.py --config configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml --verbose
python cli.py --config configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml --verbose
```

### ShallowConvNet, raw deduplicated EEG

```powershell
python cli.py `
  --config configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml `
  --verbose
```

Дополнительные seeds:

```powershell
python cli.py --config configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml --verbose
python cli.py --config configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml --verbose
```

## Текущие результаты

Все основные результаты ниже получены с межсубъектным `GroupKFold`.

### Feature-based модели

Используются 45 384 supervised-окна и 448 EEG+POW-признаков.

| Модель | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 0.3021 ± 0.0241 | **0.3059 ± 0.0255** | **0.2955 ± 0.0217** |
| PyTorch MLP | 0.2786 ± 0.0147 | 0.2822 ± 0.0168 | 0.2740 ± 0.0126 |

### Влияние random-window split

| Модель | Протокол | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Random Forest | Random-window sanity | 0.5772 | 0.5772 | 0.5748 |
| Random Forest | GroupKFold subject | 0.3021 | 0.3059 | 0.2955 |
| PyTorch MLP | Random-window sanity | 0.4048 | 0.4048 | 0.3909 |
| PyTorch MLP | GroupKFold subject | 0.2786 | 0.2822 | 0.2740 |

Разрыв подтверждает сильную субъектную зависимость. Random-window split нельзя использовать как основную научную оценку.

### EEGNet: all records, deduplication и preprocessing

| Режим | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Все source records, raw | 0.2403 ± 0.0220 | 0.2451 ± 0.0256 | 0.2184 ± 0.0151 |
| Deduplicated, raw | **0.2596 ± 0.0152** | **0.2586 ± 0.0188** | **0.2284 ± 0.0232** |
| Deduplicated, BP + notch + CAR | 0.2308 ± 0.0103 | 0.2412 ± 0.0138 | 0.1949 ± 0.0261 |

Дедупликация улучшила EEGNet, несмотря на уменьшение выборки.

Комбинация:

```text
band-pass 1–45 Hz
+ notch 50 Hz
+ common-average reference
```

ухудшила результат. Для определения причины требуется отдельная component-wise ablation.

### Raw EEG CNN, три seeds

Сравнение выполнено на одинаковых 30 958 deduplicated raw EEG-окнах, пяти outer folds и seeds:

```text
42, 7, 123
```

| Модель | Accuracy | Balanced accuracy | Macro F1 | Kappa | AUC |
|---|---:|---:|---:|---:|---:|
| EEGNet | 0.2519 | 0.2525 | 0.2236 | 0.0646 | 0.5753 |
| ShallowConvNet | **0.2825** | **0.2839** | **0.2647** | **0.1037** | **0.6047** |
| Разность | +0.0306 | +0.0313 | +0.0411 | +0.0391 | +0.0294 |

Variability:

| Модель | Межseed SD accuracy | Fold-level SD accuracy |
|---|---:|---:|
| EEGNet | 0.0061 | 0.0230 |
| ShallowConvNet | **0.0023** | **0.0134** |

ShallowConvNet показывает более высокое среднее качество и меньшую вариативность, однако эти различия пока не объявляются статистически значимыми.

### ShallowConvNet, seed 42

| Fold | N | Accuracy | Balanced accuracy | Macro F1 | Kappa | AUC | Epochs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 931 | 0.2790 | 0.2783 | 0.2637 | 0.1002 | 0.5965 | 6 |
| 2 | 6 192 | 0.2955 | 0.2925 | 0.2676 | 0.1185 | 0.6259 | 13 |
| 3 | 6 037 | 0.2508 | 0.2529 | 0.2309 | 0.0680 | 0.5884 | 8 |
| 4 | 5 776 | 0.2952 | 0.3035 | 0.2742 | 0.1207 | 0.6209 | 11 |
| 5 | 6 022 | 0.2760 | 0.2847 | 0.2634 | 0.0975 | 0.5837 | 15 |
| **Mean ± SD** | **30 958** | **0.2793 ± 0.0164** | **0.2824 ± 0.0170** | **0.2599 ± 0.0150** | **0.1010 ± 0.0189** | **0.6031 ± 0.0172** | **10.6 ± 3.3** |

Общее время обучения seed 42:

```text
713.5 s
```

## Артефакты

Корень результатов:

```text
benchmark_results/
```

Для каждого запуска сохраняются:

```text
benchmark_results_*.json
summary_*.csv
```

Для каждого DL fold:

```text
model.pt
metrics.json
training_log.csv
predictions.parquet
validation_split.json
normalization_stats.json
preprocessing_metadata.json
selected_logical_records.parquet
rejected_windows.parquet
```

Unified predictions содержат:

```text
protocol
fold
sample_id
subject_id
record_id
record_group_id
y_true
y_pred
proba_0 ... proba_4
```

Проверяется:

- отсутствие duplicate `sample_id`;
- соответствие folds между моделями;
- отсутствие outer subject overlap;
- отсутствие inner logical-record overlap;
- конечность вероятностей;
- сумма вероятностей около 1.

Отчёты:

```text
reports/raw_eeg_audit.md
reports/logical_recording_audit.md
reports/raw_eeg_artifact_audit.md
reports/raw_eegnet_label_q5_report.md
reports/raw_eeg_logical_dedup_preprocessing_ablation.md
reports/raw_eeg_cnn_model_comparison.md
```

## Тестирование

Полный test suite:

```powershell
python -m pytest -q
```

Текущее состояние:

```text
80 passed, 1 warning
```

Предупреждение sklearn связано с synthetic-тестом, в test partition которого отсутствует часть классов.

Дополнительная проверка diff:

```powershell
git diff --check
git diff --cached --check
```

## Воспроизводимость и защита от утечек

В benchmark зафиксированы следующие ограничения:

- outer test отделяется по `subject_id`;
- inner validation строится только внутри outer train;
- raw inner validation разделяется по `record_group_id`;
- normalization fit выполняется только на train;
- model state не переиспользуется между folds;
- sklearn и PyTorch-модели используют одинаковые outer fold indices;
- raw cache hash учитывает:
  - preprocessing;
  - artifact thresholds;
  - resampling;
  - channel order;
  - loader version;
  - размер и modification time исходного файла;
- данные, кэш, результаты и веса не добавляются в Git.

## Ограничения

Текущие ограничения:

1. `label_q5` является слабой PM-derived разметкой, а не прямой экспертной аннотацией когнитивного состояния.
2. Абсолютное межсубъектное качество остаётся умеренным.
3. `GroupKFold` балансирует размер групп, но не гарантирует стратификацию классов.
4. Полная preprocessing component-ablation ещё не завершена.
5. Artifact rejection пока намеренно не включён в основные сравнения.
6. Raw CNN используют независимые 10-секундные окна и не моделируют межоконный контекст.
7. Регрессионный benchmark для `target_focus` ещё не полностью интегрирован.
8. Cross-source и external-dataset evaluation остаются будущими этапами.
9. Результаты сравнения CNN по трём seeds описательные; статистическая значимость не подтверждена.
10. `sample_id` устойчив для текущей версии processed-файла, но может измениться после его физической регенерации или переупорядочивания.

## Дальнейшая работа

Ближайшие задачи:

1. завершить component-wise preprocessing ablation:
   - raw;
   - band-pass;
   - band-pass + notch;
   - band-pass + notch + CAR;
2. провести subject-level bootstrap и парные сравнения CNN;
3. выполнить error analysis по классам и субъектам;
4. добавить межоконный raw EEG контекст;
5. интегрировать регрессию `target_focus`;
6. провести cross-source evaluation;
7. добавить external EEG/wearable benchmark tracks;
8. рассмотреть user calibration и transfer learning;
9. подключать AutoML только после формирования устойчивого набора benchmark-треков.

## Git policy

Не добавлять в репозиторий:

```text
data/raw/
data/interim/
data/processed/
benchmark_results/
*.parquet
*.csv
*.pt
локальные абсолютные пути
временные логи
```

Перед commit:

```powershell
python -m pytest -q
git diff --check
git status
```

---

Основной текущий вывод: честная межсубъектная оценка существенно сложнее random-window split. Feature-based Random Forest остаётся сильным baseline, а среди raw EEG CNN ShallowConvNet показывает более высокое и более устойчивое качество, чем EEGNet.