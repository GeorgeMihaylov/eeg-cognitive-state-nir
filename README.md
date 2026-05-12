# EEG Cognitive State NIR

Проект посвящен построению, сравнению и интерпретации моделей машинного обучения для предсказания когнитивных / аффективных состояний по EEG-сигналам. В качестве слабой разметки используются PM-метрики Emotiv, синхронизированные с оконными EEG/POW-признаками.

Основная исследовательская гипотеза текущей ветки:

> PM-метрики имеют локальную временную инерцию, поэтому учет соседних EEG-окон должен улучшать качество предсказания по сравнению с однооконным tabular baseline.

После серии экспериментов гипотеза уточнена:

> Локальный временной контекст действительно улучшает качество, однако основной прирост дает не сам multi-head attention, а факт использования соседних окон. Простые context-tabular модели с `seq_len=5` часто превосходят или не уступают MHA-моделям.

---

## 1. Текущий статус ветки

В текущей ветке реализован расширенный экспериментальный pipeline:

1. Первичный осмотр исходных EEG/PM-данных.
2. Построение каталога записей.
3. Анализ частоты обновления PM-метрик.
4. Формирование оконного датасета с окном 10 секунд.
5. EDA оконного PM/POW-датасета.
6. Tabular baseline для одного PM-таргета `Focus`.
7. Объединение POW-признаков с time-domain EEG-признаками.
8. Tabular baseline для разных feature sets: `pow`, `eeg`, `pow_plus_eeg`.
9. Multi-PM baseline для всех PM-метрик.
10. Multi-head attention baseline с локальным временным контекстом.
11. Визуализация результатов MHA-прогонов.
12. Context-tabular baseline с локальным контекстом `seq_len=3` и `seq_len=5`.
13. Сравнение `tabular`, `context-tabular`, `MHA seq_len=3`, `MHA seq_len=5`.
14. Подготовлен анализ литературы и roadmap дальнейших экспериментов.

Основной финальный результат текущей ветки:

> `context-tabular len=5 + LGBM/HGB` является наиболее сильным и устойчивым baseline для большинства PM-метрик. MHA дает небольшой выигрыш только для части targets (`attention`, частично `excitement`), но не является универсально лучшей моделью.

---

## 2. Данные

Используются два источника данных:

```text
D:\PycharmProjects\eeg-cognitive-state-nir\data\raw\gpn_data
D:\PycharmProjects\eeg-cognitive-state-nir\data\raw\Old_EEG
```

Основной датасет после предобработки:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

Ключевые характеристики итогового датасета:

```text
Rows:    51 308
Columns: 508
Records: 120
Subjects: 55
Sources:
  gpn_data: 27 021 windows
  Old_EEG:  24 287 windows
```

Состав признаков:

```text
POW features: 280
EEG features: 168
Total pow_plus_eeg features: 448
```

Используемые PM-таргеты:

```text
PM.Attention.Scaled__mean
PM.Engagement.Scaled__mean
PM.Excitement.Scaled__mean
PM.Stress.Scaled__mean
PM.Relaxation.Scaled__mean
PM.Interest.Scaled__mean
PM.Focus.Scaled__mean
```

PM-метрики используются как `target`, но не используются как входные признаки.

---

## 3. Структура проекта

```text
eeg-cognitive-state-nir/
│
├── data/
│   ├── raw/
│   │   ├── gpn_data/
│   │   └── Old_EEG/
│   │
│   ├── interim/
│   │   ├── emotiv_record_catalog.csv
│   │   ├── validated_columns.json
│   │   ├── pm_sampling_record_stats.csv
│   │   ├── pm_sampling_metric_stats.csv
│   │   ├── pm_sampling_window_recommendations.csv
│   │   └── *_record_report.csv
│   │
│   └── processed/
│       ├── windowed_pm_dataset_w10.parquet
│       ├── windowed_pm_dataset_w10.csv
│       ├── windowed_eeg_pm_dataset_w10.parquet
│       ├── baseline_*_metrics.csv
│       ├── baseline_*_metrics_agg.csv
│       └── baseline_*_predictions.parquet
│
├── reports/
│   ├── figures/
│   ├── runs/
│   │   └── <run_id>/
│   │       ├── config.json
│   │       ├── train.log
│   │       ├── report.md
│   │       ├── all_targets_summary.csv
│   │       ├── all_targets_fold_metrics.csv
│   │       ├── all_targets_aggregated_metrics.csv
│   │       ├── targets/
│   │       │   └── <target>/
│   │       │       ├── sequence_metadata.csv
│   │       │       ├── epoch_history.csv
│   │       │       ├── fold_metrics.csv
│   │       │       ├── aggregated_metrics.csv
│   │       │       ├── predictions.parquet
│   │       │       ├── report.md
│   │       │       ├── checkpoints/
│   │       │       └── figures/
│   │       └── visualizations/
│   │           ├── global/
│   │           └── per_target/
│   │
│   └── comparison/
│       ├── final_pm_experiment_comparison/
│       └── final_pm_experiment_comparison_context_len5/
│           ├── normalized_all_experiments.csv
│           ├── best_models_by_target.csv
│           ├── final_experiment_comparison.csv
│           ├── final_experiment_comparison.md
│           ├── report.md
│           ├── source_files.json
│           └── figures/
│
├── src/
│   ├── 04_build_windowed_pm_dataset.py
│   ├── 05_analyze_pm_sampling.py
│   ├── 06_eda_windowed_dataset.py
│   ├── 07_train_baselines.py
│   ├── 08_build_windowed_eeg_features.py
│   ├── 09_train_multi_pm_baselines.py
│   ├── 10_describe_multi_pm_baseline.py
│   ├── 11_train_multihead_attention_short.py
│   ├── 12_visualize_mha_all_pm_run.py
│   ├── 13_train_context_tabular_baselines.py
│   └── 14_compare_experiments.py
│
├── github_issues/
│   └── *.md
│
└── README.md
```

---

## 4. Окружение

Рекомендуемое окружение `miniconda`:

```powershell
conda create -n eeg_nir python=3.10 -y
conda activate eeg_nir
```

Минимальный набор пакетов:

```powershell
pip install numpy pandas scipy scikit-learn matplotlib pyarrow fastparquet torch lightgbm tabulate
```

Проверка окружения:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe --version
D:\miniconda3\envs\eeg_nir\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Для GPU-регима на локальной машине можно использовать конфигурацию PyTorch, аналогичную окружению `aiice_env`:

```text
torch = 2.5.1+cu124
cuda available = True
cuda = 12.4
device = NVIDIA GeForce RTX 2070
```

---

## 5. Этапы pipeline

### 5.1. Построение оконного PM/POW-датасета

Скрипт:

```text
src/04_build_windowed_pm_dataset.py
```

Назначение:

- читает каталог Emotiv-записей;
- выбирает PM/POW-колонки;
- строит 10-секундные окна;
- агрегирует PM и POW внутри окна;
- формирует regression-target и quantile labels;
- сохраняет parquet/csv и markdown-отчет.

Ключевой запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\04_build_windowed_pm_dataset.py `
  --window-size 10 `
  --output-name windowed_pm_dataset_w10
```

Результат:

```text
data/processed/windowed_pm_dataset_w10.parquet
reports/windowed_pm_dataset_w10_report.md
```

Итоговая сводка:

```text
Rows/windows: 51 308
Columns: 340
Records: 120
Subjects: 55
Sources:
  gpn_data: 27 021
  Old_EEG:  24 287
```

---

### 5.2. Анализ частоты PM-метрик

Скрипт:

```text
src/05_analyze_pm_sampling.py
```

Назначение:

- анализирует интервалы обновления PM-метрик;
- считает median/p75/p90/p95;
- рекомендует размер окна.

Ключевой запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\05_analyze_pm_sampling.py --max-records 5
```

Результат:

```text
PM interval median ≈ 9.99 s
recommended window ≈ 10 s
```

На основании этого выбран основной window size:

```text
window_size = 10 seconds
```

---

### 5.3. EDA оконного датасета

Скрипт:

```text
src/06_eda_windowed_dataset.py
```

Назначение:

- описательная статистика оконного датасета;
- распределения PM-таргетов;
- распределения по источникам, субъектам и записям;
- проверка пропусков;
- анализ корреляций;
- сохранение таблиц и графиков.

Результаты сохраняются в:

```text
reports/
reports/figures/
data/interim/
```

---

### 5.4. Tabular baseline для одного target

Скрипт:

```text
src/07_train_baselines.py
```

Назначение:

- обучает классические ML-модели для regression/classification;
- поддерживает feature sets: `pow`, `eeg`, `pow_plus_eeg`;
- поддерживает `raw_pow`, `log_pow`, `raw_plus_log_pow`;
- использует несколько схем валидации.

Основные схемы валидации:

```text
random_split
GroupKFold by subject_id
cross_source
cross_source_no_overlap
```

Пример запуска для EEG-only:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\07_train_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --feature-set eeg `
  --max-rows 10000 `
  --fast `
  --enable-cross-source-no-overlap `
  --output-prefix baseline_eeg_w10_test
```

Пример запуска для POW+EEG:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\07_train_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --enable-cross-source-no-overlap `
  --output-prefix baseline_pow_plus_eeg_w10_log
```

Основной вывод:

```text
POW+EEG лучше, чем POW-only и EEG-only, особенно в regression-постановке.
```

---

### 5.5. Построение EEG feature dataset

Скрипт:

```text
src/08_build_windowed_eeg_features.py
```

Назначение:

- строит time-domain EEG-признаки;
- синхронизирует их с оконным PM/POW-датасетом;
- формирует объединенный датасет.

Результат:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

Итоговая сводка:

```text
EEG feature rows: 51 308
EEG feature columns: 177
Merged rows: 51 308
Merged columns: 508
EEG columns in merged dataset: 168
```

---

### 5.6. Multi-PM tabular baseline

Скрипт:

```text
src/09_train_multi_pm_baselines.py
```

Назначение:

- обучает tabular regression baseline отдельно для всех PM-метрик;
- сравнивает предсказуемость `attention`, `engagement`, `excitement`, `stress`, `relaxation`, `interest`, `focus`;
- сохраняет summary по таргетам.

Быстрый запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --models hgb,lgbm `
  --validation groupkfold `
  --max-rows 10000 `
  --run-name multi_pm_test_pow_plus_eeg_log_pow
```

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --models hgb,lgbm `
  --validation groupkfold `
  --run-name multi_pm_full_pow_plus_eeg
```

Вывод тестового multi-PM baseline:

```text
Focus не является самым предсказуемым таргетом.
Лучше предсказывались: Excitement, Relaxation, Engagement.
```

---

### 5.7. Описательная статистика multi-PM baseline

Скрипт:

```text
src/10_describe_multi_pm_baseline.py
```

Назначение:

- читает папку multi-PM запуска;
- агрегирует target availability;
- агрегирует метрики по target/model;
- строит рейтинг target-ов;
- формирует markdown-отчет.

Пример запуска:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\10_describe_multi_pm_baseline.py `
  --run-dir reports\runs\<multi_pm_run_id>
```

Результаты:

```text
reports/runs/<run_id>/descriptive_stats/target_availability_stats.csv
reports/runs/<run_id>/descriptive_stats/metrics_by_target_model.csv
reports/runs/<run_id>/descriptive_stats/target_ranking.csv
reports/runs/<run_id>/descriptive_stats/global_metric_descriptive_stats.csv
reports/runs/<run_id>/descriptive_stats/report_descriptive.md
```

---

## 6. Multi-head self-attention baseline

### 6.1. Скрипт

```text
src/11_train_multihead_attention_short.py
```

Несмотря на слово `short` в имени, текущая версия поддерживает как короткие, так и полные запуски.

Назначение:

- строит последовательности соседних окон внутри одного `record_id`;
- использует `original_row_idx`, чтобы не рассинхронизировать признаки и таргеты после сортировки по времени;
- обучает TransformerEncoder-based regressor;
- поддерживает один target или все PM-таргеты через `--pm-target all`;
- сохраняет результаты по каждому target в отдельную подпапку.

Архитектура:

```text
Input:
  [X_{t-k}, ..., X_t, ..., X_{t+k}]

Feature projection:
  Linear(n_features -> d_model)
  LayerNorm
  GELU
  Dropout

Temporal encoder:
  TransformerEncoderLayer
  Multi-head self-attention

Pooling:
  center token pooling by default

Head:
  MLP regression head
```

Основная конфигурация:

```text
feature_set = pow_plus_eeg
feature_mode = log_pow
seq_len = 3 or 5
d_model = 64
nhead = 4
num_layers = 1
batch_size = 128
epochs = 12
validation = GroupKFold by subject_id
```

---

### 6.2. Полный MHA-прогон по всем PM, seq_len=3

Команда:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\11_train_multihead_attention_short.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 3 `
  --fold-limit 0 `
  --epochs 12 `
  --batch-size 128 `
  --d-model 64 `
  --nhead 4 `
  --num-layers 1 `
  --run-name mha_all_pm_full
```

Результаты сохраняются в:

```text
reports/runs/<timestamp>_mha_all_pm_full_all_pow_plus_eeg_len3/
```

Главные файлы:

```text
all_targets_summary.csv
all_targets_fold_metrics.csv
all_targets_aggregated_metrics.csv
report.md
```

---

### 6.3. Короткое сравнение MHA seq_len=3 и seq_len=5

Проверялась гипотеза, что более широкий контекст из пяти окон может улучшить качество:

```text
seq_len=3:
  [X_{t-1}, X_t, X_{t+1}]

seq_len=5:
  [X_{t-2}, X_{t-1}, X_t, X_{t+1}, X_{t+2}]
```

Оба коротких прогона выполнялись в одинаковом режиме:

```text
--pm-target all
--feature-set pow_plus_eeg
--feature-mode log_pow
--max-samples 10000
--fold-limit 2
--epochs 12
--batch-size 128
--d-model 64
--nhead 4
--num-layers 1
```

Сравнение коротких MHA-прогонов:

| Target | R2 len3 | R2 len5 | ΔR2 | Spearman len3 | Spearman len5 | ΔSpearman |
|---|---:|---:|---:|---:|---:|---:|
| excitement | 0.581 | 0.427 | -0.155 | 0.709 | 0.628 | -0.081 |
| relaxation | 0.381 | 0.249 | -0.133 | 0.654 | 0.512 | -0.142 |
| engagement | 0.274 | 0.256 | -0.018 | 0.480 | 0.432 | -0.048 |
| attention | 0.209 | 0.193 | -0.017 | 0.491 | 0.443 | -0.047 |
| stress | 0.179 | 0.245 | +0.066 | 0.365 | 0.488 | +0.124 |
| focus | 0.169 | 0.307 | +0.138 | 0.452 | 0.537 | +0.085 |
| interest | 0.128 | 0.123 | -0.005 | 0.356 | 0.340 | -0.017 |

Вывод:

```text
seq_len=3 остается более универсальной MHA-конфигурацией.
seq_len=5 улучшает Focus и Stress, но ухудшает Excitement и Relaxation.
```

---

## 7. Context-tabular baseline

### 7.1. Скрипт

```text
src/13_train_context_tabular_baselines.py
```

Назначение:

- строит локальный контекст соседних окон внутри одного `record_id`;
- формирует плоское табличное представление через конкатенацию окон;
- обучает классические регрессионные модели;
- поддерживает один PM-target или все PM-targets;
- сохраняет отдельные fold metrics и общий summary.

Форматы признаков:

```text
seq_len=3:
  concat(X_{t-1}, X_t, X_{t+1})
  448 × 3 = 1344 features

seq_len=5:
  concat(X_{t-2}, X_{t-1}, X_t, X_{t+1}, X_{t+2})
  448 × 5 = 2240 features
```

### 7.2. Context-tabular seq_len=3

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\13_train_context_tabular_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 3 `
  --fold-limit 0 `
  --fast `
  --models lgbm_reg,hgb_reg `
  --save-predictions false `
  --no-plots `
  --run-name context_tabular_len3_fast
```

Run directory:

```text
reports/runs/20260512_144503_context_tabular_len3_fast_all_pow_plus_eeg_len3
```

Результаты:

| Target | Best R2 | Best Spearman |
|---|---:|---:|
| attention | 0.181 | 0.462 |
| engagement | 0.322 | 0.525 |
| excitement | 0.570 | 0.717 |
| stress | 0.354 | 0.530 |
| relaxation | 0.425 | 0.640 |
| interest | 0.226 | 0.452 |
| focus | 0.261 | 0.489 |

### 7.3. Context-tabular seq_len=5

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\13_train_context_tabular_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 5 `
  --fold-limit 0 `
  --fast `
  --models lgbm_reg,hgb_reg `
  --save-predictions false `
  --no-plots `
  --run-name context_tabular_len5_fast
```

Run directory:

```text
reports/runs/20260512_152527_context_tabular_len5_fast_all_pow_plus_eeg_len5
```

Результаты:

| Target | Best model | Best R2 | Best Spearman | RMSE | MAE |
|---|---|---:|---:|---:|---:|
| attention | hgb/lgbm close | 0.203 | 0.467 | 0.113 | 0.089 |
| engagement | hgb_reg | 0.306 | 0.513 | 0.108 | 0.086 |
| excitement | lgbm_reg | 0.579 | 0.718 | 0.151 | 0.112 |
| stress | hgb_reg | 0.347 | 0.502 | 0.110 | 0.080 |
| relaxation | hgb_reg | 0.426 | 0.642 | 0.125 | 0.098 |
| interest | hgb_reg | 0.274 | 0.465 | 0.082 | 0.060 |
| focus | lgbm_reg | 0.345 | 0.568 | 0.100 | 0.077 |

### 7.4. Context-tabular len=3 vs len=5

| Target | Context len=3 R2 | Context len=5 R2 | ΔR2 |
|---|---:|---:|---:|
| attention | 0.181 | 0.203 | +0.022 |
| engagement | 0.322 | 0.306 | -0.016 |
| excitement | 0.570 | 0.579 | +0.009 |
| stress | 0.354 | 0.347 | -0.007 |
| relaxation | 0.425 | 0.426 | +0.001 |
| interest | 0.226 | 0.274 | +0.048 |
| focus | 0.261 | 0.345 | +0.084 |

Вывод:

```text
Оптимальная длина локального контекста зависит от target.
Focus, Interest и Attention заметно выигрывают от seq_len=5.
Engagement и Stress немного лучше при seq_len=3.
Relaxation и Excitement устойчиво хорошо предсказываются при обоих вариантах.
```

---

## 8. Сравнение экспериментов

### 8.1. Скрипт

```text
src/14_compare_experiments.py
```

Назначение:

- собирает summary-файлы из разных run directories;
- нормализует форматы результатов;
- выбирает лучшую модель по каждому target;
- строит итоговую таблицу сравнения;
- считает deltas между подходами;
- строит графики.

Поддерживаемые входы:

```text
--tabular
--context
--mha-len3
--mha-len5
```

### 8.2. Сравнение с context len=3

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\14_compare_experiments.py `
  --tabular data\processed\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv `
  --context reports\runs\20260512_144503_context_tabular_len3_fast_all_pow_plus_eeg_len3 `
  --mha-len3 reports\runs\20260508_172632_mha_all_pm_short_len3_control_all_pow_plus_eeg_len3 `
  --mha-len5 reports\runs\20260508_171708_mha_all_pm_short_len5_all_pow_plus_eeg_len5 `
  --output-dir reports\comparison\final_pm_experiment_comparison
```

Результат:

```text
reports/comparison/final_pm_experiment_comparison/
```

### 8.3. Сравнение с context len=5

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\14_compare_experiments.py `
  --tabular data\processed\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv `
  --context reports\runs\20260512_152527_context_tabular_len5_fast_all_pow_plus_eeg_len5 `
  --mha-len3 reports\runs\20260508_172632_mha_all_pm_short_len3_control_all_pow_plus_eeg_len3 `
  --mha-len5 reports\runs\20260508_171708_mha_all_pm_short_len5_all_pow_plus_eeg_len5 `
  --output-dir reports\comparison\final_pm_experiment_comparison_context_len5
```

Результат:

```text
reports/comparison/final_pm_experiment_comparison_context_len5/
```

Главные файлы:

```text
normalized_all_experiments.csv
best_models_by_target.csv
final_experiment_comparison.csv
final_experiment_comparison.md
report.md
source_files.json
figures/
```

### 8.4. Итоговое сравнение с context len=5

| Target | Tabular R2 | Context len=5 R2 | MHA len=3 R2 | MHA len=5 R2 | Best approach |
|---|---:|---:|---:|---:|---|
| attention | NaN | 0.203 | 0.209 | 0.193 | MHA len=3, small margin |
| engagement | NaN | 0.306 | 0.274 | 0.256 | context len=5 |
| excitement | NaN | 0.579 | 0.581 | 0.427 | MHA len=3 / context close |
| stress | NaN | 0.347 | 0.179 | 0.245 | context len=5 |
| relaxation | NaN | 0.426 | 0.381 | 0.249 | context len=5 |
| interest | NaN | 0.274 | 0.128 | 0.123 | context len=5 |
| focus | 0.145 | 0.345 | 0.169 | 0.307 | context len=5 |

Главные deltas:

```text
Focus:
  tabular X_t R2     = 0.145
  context len=5 R2   = 0.345
  MHA len=5 R2       = 0.307

Excitement:
  context len=5 R2   = 0.579
  MHA len=3 R2       = 0.581

Attention:
  context len=5 R2   = 0.203
  MHA len=3 R2       = 0.209

Interest:
  context len=5 R2   = 0.274
  MHA len=3 R2       = 0.128
  MHA len=5 R2       = 0.123
```

Главный вывод:

```text
На текущих агрегированных признаках POW+EEG MHA не дает устойчивого преимущества над простой context-tabular моделью.
Основной источник прироста — локальный временной контекст, а не сам механизм attention.
```

---

## 9. Визуализация MHA-прогона

Скрипт:

```text
src/12_visualize_mha_all_pm_run.py
```

Назначение:

- строит global dashboard по всем PM-метрикам;
- строит bar plots для R2, Spearman, Pearson, RMSE, MAE;
- строит heatmap `target x metric`;
- строит fold-level heatmaps;
- строит boxplot-распределения метрик по folds;
- строит scatter `y_true vs y_pred`;
- строит residual plots;
- строит histograms residuals;
- строит binned calibration plots;
- строит loss curves;
- формирует markdown-отчет со списком всех графиков.

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\12_visualize_mha_all_pm_run.py `
  --run-dir reports\runs\<mha_all_pm_full_run_id>
```

Результаты:

```text
reports/runs/<run_id>/visualizations/
├── visualization_report.md
├── visualization_summary.csv
├── visualization_fold_metrics.csv
├── global/
│   ├── dashboard_main_metrics.png
│   ├── target_metric_heatmap.png
│   ├── r2_bar.png
│   ├── spearman_bar.png
│   ├── pearson_bar.png
│   ├── rmse_bar.png
│   ├── mae_bar.png
│   ├── fold_heatmap_r2.png
│   ├── fold_heatmap_spearman.png
│   └── fold_boxplot_*.png
└── per_target/
    └── <target>/
        ├── <target>_scatter_y_true_y_pred.png
        ├── <target>_residual_vs_true.png
        ├── <target>_residual_hist.png
        ├── <target>_residual_by_fold.png
        ├── <target>_calibration_binned.png
        ├── <target>_target_distribution.png
        └── <target>_loss_*.png
```

---

## 10. Анализ литературы и исследовательский roadmap

Подготовлен общий обзор статей по EEG foundation models, temporal context, event detection, graph/connectivity representations, multi-task EEG и benchmark methodology.

Общий вывод по литературе:

```text
Современная литература поддерживает стратегию:
1. сначала строить сильные baseline и строгий benchmark;
2. затем проводить frequency/channel/context/session ablation;
3. только после этого усложнять нейросетевую архитектуру.
```

Ключевые идеи из литературы:

| Направление | Практическая идея для проекта |
|---|---|
| NeuralBench | единый benchmark registry и стандартизированные метрики |
| MPNet | multi-rhythm features, covariance/connectivity, compact pooling |
| CLEF | session-level context и spectral representation |
| DANCE | PM-state event detection и transition prediction |
| CORTEG | subject calibration и parameter-efficient adaptation |
| SIMON | saliency / feature / channel importance |
| CFSPMNet | Fourier context, prototype consistency, cross-subject adaptation |
| MTEEG | multi-task PM model с target-specific heads/adapters |

Файл с обзором:

```text
eeg_pm_articles_summary_and_project_recommendations.md
```

---

## 11. Методологические ограничения

1. PM-метрики Emotiv являются слабой разметкой, а не экспертным ground truth.
2. Модель предсказывает значения PM-метрик, а не когнитивное состояние напрямую.
3. Основная честная оценка качества — `GroupKFold` по `subject_id`.
4. `Random split` может завышать качество из-за leakage между окнами одного субъекта.
5. Источники `gpn_data` и `Old_EEG` имеют доменный сдвиг.
6. Не все PM-метрики одинаково предсказуемы.
7. `Attention` и `Interest` сложнее, чем `Excitement` и `Relaxation`.
8. `seq_len=5` не дает универсального улучшения для всех моделей и targets.
9. MHA-подход требует честного сравнения с сильным context-tabular baseline.
10. Текущие признаки агрегированы по окнам; raw temporal dynamics пока используются ограниченно.

---

## 12. Основные выводы текущей ветки

1. Окно 10 секунд обосновано частотой обновления PM-метрик.
2. POW+EEG признаки лучше, чем использование только POW или только EEG.
3. `Focus` не является самым сильным PM-таргетом.
4. Наиболее предсказуемые PM-метрики: `Excitement`, `Relaxation`, далее `Stress` и `Focus`.
5. Локальный временной контекст соседних EEG-окон заметно улучшает предсказание PM-метрик.
6. `context-tabular len=5` является наиболее сильным текущим baseline для большинства targets.
7. MHA полезен только для части PM-метрик и не превосходит context-tabular устойчиво.
8. Оптимальная длина контекста зависит от target.
9. Следующий научно обоснованный шаг — frequency-band ablation.
10. Следующий инфраструктурный шаг — master benchmark registry.

---

## 13. Рекомендуемый следующий план

### 13.1. Инфраструктурный шаг

Создать единый benchmark registry:

```text
src/20_build_pm_benchmark_registry.py
```

Цель:

```text
собрать все completed runs в один master table:
target, experiment, model, feature_set, seq_len, validation, folds,
r2, spearman, rmse, mae, n_samples, n_subjects, run_dir
```

Выход:

```text
reports/comparison/master_pm_benchmark/master_pm_benchmark.csv
reports/comparison/master_pm_benchmark/report.md
reports/comparison/master_pm_benchmark/figures/
```

### 13.2. Главный следующий ML-эксперимент

Frequency-band ablation:

```text
src/15_train_band_ablation_baselines.py
```

Feature groups:

```text
theta
alpha
beta
gamma/high
low = theta + alpha
high = beta + gamma
low_high_concat
all_pow
eeg_only
pow_plus_eeg
```

Setup:

```text
seq_len = 5
validation = GroupKFold by subject_id
models = lgbm_reg,hgb_reg
targets = all PM metrics
```

### 13.3. Дальнейшие эксперименты

```text
src/16_train_context_pooling_baselines.py
src/16_train_session_context_baselines.py
src/17_build_connectivity_features.py
src/18_train_connectivity_augmented_baselines.py
src/17_build_pm_state_events.py
src/18_train_pm_event_baselines.py
src/17_subject_calibration_experiment.py
src/21_analyze_pm_target_relationships.py
src/21_train_multitask_pm_model.py
```

---

## 14. Минимальная последовательность воспроизведения

```powershell
cd D:\PycharmProjects\eeg-cognitive-state-nir
```

Построить 10-секундный PM/POW dataset:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\04_build_windowed_pm_dataset.py `
  --window-size 10 `
  --output-name windowed_pm_dataset_w10
```

Построить EEG+PM dataset:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\08_build_windowed_eeg_features.py
```

Запустить multi-PM tabular baseline:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --models hgb,lgbm `
  --validation groupkfold `
  --run-name multi_pm_full_pow_plus_eeg
```

Запустить context-tabular baseline `seq_len=5`:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\13_train_context_tabular_baselines.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 5 `
  --fold-limit 0 `
  --fast `
  --models lgbm_reg,hgb_reg `
  --save-predictions false `
  --no-plots `
  --run-name context_tabular_len5_fast
```

Запустить MHA all-PM baseline:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\11_train_multihead_attention_short.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 3 `
  --fold-limit 0 `
  --epochs 12 `
  --batch-size 128 `
  --d-model 64 `
  --nhead 4 `
  --num-layers 1 `
  --run-name mha_all_pm_full
```

Построить сравнение экспериментов:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\14_compare_experiments.py `
  --tabular data\processed\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv `
  --context reports\runs\20260512_152527_context_tabular_len5_fast_all_pow_plus_eeg_len5 `
  --mha-len3 reports\runs\20260508_172632_mha_all_pm_short_len3_control_all_pow_plus_eeg_len3 `
  --mha-len5 reports\runs\20260508_171708_mha_all_pm_short_len5_all_pow_plus_eeg_len5 `
  --output-dir reports\comparison\final_pm_experiment_comparison_context_len5
```

Построить визуализации MHA-прогона:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\12_visualize_mha_all_pm_run.py `
  --run-dir reports\runs\<mha_all_pm_full_run_id>
```

---

## 15. Ключевой результат для отчета

```text
В рамках текущей ветки реализован pipeline предсказания PM-метрик Emotiv по EEG/POW-признакам. После построения 10-секундного оконного датасета были обучены однооконные tabular baselines, temporal multi-head self-attention models и context-tabular baselines. Эксперименты показали, что локальный временной контекст соседних EEG-окон существенно улучшает качество предсказания PM-метрик. Однако механизм multi-head attention не дал устойчивого преимущества над простым context-tabular baseline. Наиболее сильный текущий подход — context-tabular len=5 + LGBM/HGB. Для Focus достигнуто R2≈0.345 и Spearman≈0.568, для Excitement — R2≈0.579 и Spearman≈0.718, для Relaxation — R2≈0.426 и Spearman≈0.642. Это подтверждает полезность временного контекста, но указывает, что дальнейшее развитие должно быть направлено на frequency-band ablation, compact context pooling, session-level context и channel/connectivity features, а не на слепое усложнение attention-модели.
```
