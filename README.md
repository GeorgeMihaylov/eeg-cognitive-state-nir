# EEG Cognitive State NIR

Актуализировано: 2026-05-15.

Проект посвящен построению, сравнению и интерпретации моделей машинного обучения для предсказания когнитивных / аффективных состояний по EEG-сигналам. В качестве слабой разметки используются PM-метрики Emotiv, синхронизированные с оконными EEG/POW-признаками. Дополнительно в текущей ветке начата внешняя линия проверки wearable-подхода: на датасете WESAD исследуется, могут ли физиологические сигналы носимых устройств выступать proxy-источником для stress/arousal-related состояний.

Основная исследовательская гипотеза текущей ветки:

> PM-метрики имеют локальную временную инерцию, поэтому учет соседних EEG-окон должен улучшать качество предсказания по сравнению с однооконным tabular baseline.

После серии экспериментов гипотеза уточнена:

> Локальный временной контекст действительно улучшает качество, однако основной прирост дает не сам multi-head attention, а факт использования соседних окон. Простые context-tabular модели с `seq_len=5` часто превосходят или не уступают MHA-моделям.

Дополнительная рабочая гипотеза wearable-ветки:

> Для `PM.Stress` и близких arousal-состояний полезны не только EEG/POW-признаки, но и физиологические сигналы носимых устройств: BVP/PPG, EDA, TEMP и ACC. При этом ACC должен рассматриваться как контроль движения и протокольных артефактов, а не как чистый физиологический маркер.

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
15. Подключен внешний wearable benchmark WESAD.
16. Построен WESAD windowed stress dataset с окнами 60 секунд и шагом 10 секунд.
17. Обучены WESAD stress baselines и проведен threshold analysis.
18. Проведен feature-group ablation для `EDA`, `BVP`, `TEMP`, `ACC` и их комбинаций.
19. Проведен protocol-control experiment: `all`, `no_acc`, `acc_only`, `bvp_only`, `bvp_temp`, `eda_bvp_temp`.
20. Сформирован итоговый WESAD summary report.
21. COLET временно отложен: данные доступны, но формат MATLAB v7.3 с object references требует отдельной MATLAB-конвертации.

Основной финальный результат текущей EEG/PM-линии:

> `context-tabular len=5 + LGBM/HGB` является наиболее сильным и устойчивым baseline для большинства PM-метрик. MHA дает небольшой выигрыш только для части targets (`attention`, частично `excitement`), но не является универсально лучшей моделью.

Основной финальный результат wearable-линии:

> На WESAD стресс-состояние предсказывается по wrist physiology при subject-aware validation. Лучший компактный физиологический вариант — `BVP + TEMP + logistic_robust`, balanced accuracy ≈ 0.843. Однако `ACC-only` также дает высокий результат, поэтому в дальнейшем ACC нужно использовать как контроль движения/protocol confounding, а основной wearable proxy для `PM.Stress` строить на BVP/PPG, EDA и TEMP.

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

Дополнительно подключены внешние датасеты для анализа связи PM-метрик с wearable / eye-tracking источниками:

```text
data/external/WESAD/
data/external/COLET/
```

Текущий статус внешних источников:

```text
WESAD: обработан, построен windowed stress dataset, обучены baseline-модели, проведены ablation и protocol-control эксперименты.
COLET: скачан и проинспектирован на уровне HDF5-структуры, но временно отложен из-за MATLAB v7.3 object references; требуется MATLAB-конвертация .mat -> CSV/Parquet.
```

Ключевой WESAD dataset:

```text
data/processed/wesad_windowed_stress_dataset.parquet
Rows: 4 214
Columns: 138
Subjects: 15
Feature columns: 116
Target: stress_binary
```

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
│   ├── comparison/
│   │   ├── final_pm_experiment_comparison/
│   │   └── final_pm_experiment_comparison_context_len5/
│   │       ├── normalized_all_experiments.csv
│   │       ├── best_models_by_target.csv
│   │       ├── final_experiment_comparison.csv
│   │       ├── final_experiment_comparison.md
│   │       ├── report.md
│   │       ├── source_files.json
│   │       └── figures/
│   │
│   └── wearable_pm_alignment/
│       ├── wesad_inventory.csv
│       ├── wesad_windowed_stress_dataset_report.md
│       ├── runs/
│       │   └── <wesad_run_id>/
│       └── wesad_final_summary/
│           ├── wesad_final_summary.md
│           ├── wesad_key_metrics.csv
│           ├── wesad_protocol_conclusions.csv
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
│   ├── 14_compare_experiments.py
│   ├── 16_inspect_wesad_dataset.py
│   ├── 17_prepare_wesad_windowed_dataset.py
│   ├── 18_train_wesad_stress_baseline.py
│   ├── 19_analyze_wesad_stress_results.py
│   ├── 20_train_wesad_feature_group_ablation.py
│   ├── 21_train_wesad_protocol_control.py
│   ├── 22_inspect_colet_dataset.py
│   ├── 22_inspect_colet_dataset_light.py
│   ├── 23_probe_colet_minimal.py
│   ├── 24_probe_colet_task_leafs.py
│   ├── 25_check_colet_matlab_outputs.py
│   └── 26_build_wesad_summary_report.py
│
├── tools/
│   └── 25_convert_colet_mat_to_tables.m
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
pip install numpy pandas scipy scikit-learn matplotlib pyarrow fastparquet torch lightgbm tabulate h5py
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

Для COLET требуется отдельная MATLAB-конвертация, если планируется продолжать работу с этим датасетом:

```text
COLET .mat files are MATLAB v7.3 / HDF5 files with object references.
Python/h5py opens the files, but dereferencing MATLAB object references is too slow for practical extraction.
Recommended path: MATLAB -> intermediate CSV/Parquet -> Python feature engineering.
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


## 10. Wearable benchmark: WESAD

### 10.1. Мотивация

После созвона была выделена отдельная исследовательская линия: проверить, можно ли соотносить PM-метрики Emotiv, прежде всего `PM.Stress`, с сигналами носимых устройств. Для первичной проверки выбран WESAD, так как он содержит wrist-physiology сигналы и разметку stress / non-stress.

WESAD используется как внешний benchmark. Он не является прямым датасетом для предсказания Emotiv PM-метрик, но проверяет более общую гипотезу:

```text
wearable physiology can provide useful proxy signals for stress/arousal-related cognitive-state estimation
```

### 10.2. Использованные данные WESAD

Сырые данные:

```text
data/external/WESAD/WESAD/
```

Подготовленный датасет:

```text
data/processed/wesad_windowed_stress_dataset.parquet
```

Параметры подготовки:

```text
window_size_sec = 60
step_size_sec   = 10
validation      = GroupKFold by subject_id
```

Итоговая сводка:

| Показатель | Значение |
|---|---:|
| Rows | 4 214 |
| Columns | 138 |
| Subjects | 15 |
| Feature columns | 116 |
| non-stress windows | 3 275 |
| stress windows | 939 |

### 10.3. Реализованные WESAD-скрипты

| Скрипт | Назначение |
|---|---|
| `src/16_inspect_wesad_dataset.py` | Инвентаризация WESAD-файлов и структуры `.pkl` / E4 data. |
| `src/17_prepare_wesad_windowed_dataset.py` | Формирование 60-секундного оконного stress dataset. |
| `src/18_train_wesad_stress_baseline.py` | Обучение baseline-классификаторов stress / non-stress. |
| `src/19_analyze_wesad_stress_results.py` | Threshold analysis, per-subject metrics, errors, feature importance. |
| `src/20_train_wesad_feature_group_ablation.py` | Ablation по группам признаков: EDA, BVP, TEMP, ACC. |
| `src/21_train_wesad_protocol_control.py` | Проверка protocol/movement confounding: `all`, `no_acc`, `acc_only`, `bvp_only`, `bvp_temp`. |
| `src/26_build_wesad_summary_report.py` | Сборка итогового WESAD summary report. |

### 10.4. Baseline stress classification

Лучший default-threshold baseline:

| Model | Balanced Accuracy | Macro-F1 | ROC-AUC | Average Precision | F1 stress |
|---|---:|---:|---:|---:|---:|
| `logistic_robust` | 0.803 | 0.793 | 0.878 | 0.804 | 0.694 |

Threshold tuning показал, что `rf_clf` можно улучшить подбором порога:

| Model | Best threshold | Best balanced accuracy |
|---|---:|---:|
| `rf_clf` | 0.25 | 0.809 |
| `logistic_robust` | 0.45 | 0.804 |
| `lgbm_clf` | 0.05 | 0.798 |

### 10.5. Feature-group ablation

Главный результат ablation:

| Feature group | Model | Features | Balanced Accuracy | ROC-AUC | F1 stress |
|---|---|---:|---:|---:|---:|
| `bvp_only` | `logistic_robust` | 18 | 0.835 | 0.900 | 0.699 |
| `acc_only` | `logistic_robust` | 64 | 0.830 | 0.915 | 0.709 |
| `eda_bvp_temp` | `logistic_robust` | 52 | 0.813 | 0.883 | 0.682 |
| `all` | `logistic_robust` | 116 | 0.803 | 0.878 | 0.694 |
| `eda_only` | `logistic_robust` | 18 | 0.758 | 0.831 | 0.611 |

Вывод:

```text
BVP/PPG оказался самым сильным компактным физиологическим proxy.
EDA полезна, но с текущими простыми статистическими признаками слабее BVP.
ACC-only слишком силен, поэтому его нельзя трактовать как чистую физиологию.
```

### 10.6. Protocol-control experiment

Protocol-control был нужен, чтобы проверить, не объясняется ли качество только движением или структурой протокола.

Лучшие default-threshold результаты:

| Feature group | Model | Balanced Accuracy | ROC-AUC | F1 stress |
|---|---|---:|---:|---:|
| `bvp_temp` | `logistic_robust` | 0.843 | 0.909 | 0.703 |
| `bvp_only` | `logistic_robust` | 0.835 | 0.900 | 0.699 |
| `acc_only` | `logistic_robust` | 0.830 | 0.915 | 0.709 |
| `no_acc` | `logistic_robust` | 0.813 | 0.883 | 0.682 |
| `no_acc` | `lgbm_clf` | 0.807 | 0.920 | 0.714 |

Threshold-optimized результаты:

| Feature group | Model | Best threshold | Best balanced accuracy |
|---|---|---:|---:|
| `acc_only` | `logistic_robust` | 0.65 | 0.845 |
| `no_acc` | `lgbm_clf` | 0.07 | 0.845 |
| `bvp_temp` | `logistic_robust` | 0.48 | 0.844 |
| `bvp_only` | `logistic_robust` | 0.49 | 0.836 |

Итоговая интерпретация:

```text
WESAD подтверждает перспективность wearable stress detection, но высокий результат ACC-only указывает на movement/protocol confounding. Для PM.Stress alignment ACC следует использовать как контрольный канал, а основной физиологический wearable proxy строить на BVP/PPG, EDA и TEMP.
```

### 10.7. Итоговые файлы WESAD

```text
reports/wearable_pm_alignment/wesad_final_summary/
  wesad_final_summary.md
  wesad_key_metrics.csv
  wesad_baseline_summary.csv
  wesad_threshold_summary.csv
  wesad_feature_group_summary.csv
  wesad_protocol_control_summary.csv
  wesad_protocol_conclusions.csv
  source_files.json
  figures/
```

---

## 11. COLET: статус и причина остановки

COLET был выбран как потенциальный eye-tracking benchmark для cognitive workload / attention / focus линии. Данные скачаны в:

```text
data/external/COLET/
```

Обнаруженная структура:

```text
COLET_v0: images + readme only
COLET_v1: data.mat
COLET_v2: data.mat
COLET_v3: data_v3.mat
```

Инспекция показала, что `data.mat` / `data_v3.mat` — это MATLAB v7.3 / HDF5 файлы со структурой:

```text
/Data/subject_info  shape=(47, 1), dtype=object
/Data/task          shape=(47, 1), dtype=object
```

Для первого task-объекта найдены поля:

```text
annotation
blinks
gaze
pupil
```

Проблема:

```text
Python/h5py открывает файл быстро, но разыменование MATLAB object references работает слишком медленно.
Даже минимальное обращение к Data.task[1] заняло около 175 секунд.
```

Решение:

```text
COLET временно отложен.
Для продолжения нужна MATLAB-конвертация .mat -> CSV/Parquet.
После конвертации Python будет использовать только промежуточные таблицы.
```

Подготовленные вспомогательные файлы:

| Файл | Статус |
|---|---|
| `src/22_inspect_colet_dataset.py` | Полный инспектор, оказался слишком тяжелым для v7.3 references. |
| `src/22_inspect_colet_dataset_light.py` | Легкая HDF5-инспекция верхнего уровня. |
| `src/23_probe_colet_minimal.py` | Минимальный probe object references. |
| `src/24_probe_colet_task_leafs.py` | Probe task leaf references, также слишком медленный для практического чтения. |
| `tools/25_convert_colet_mat_to_tables.m` | MATLAB-конвертер для будущего этапа. |

---

## 12. Анализ литературы и исследовательский roadmap

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

## 13. Методологические ограничения

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
11. WESAD является внешним wearable benchmark, а не прямым датасетом для Emotiv PM prediction.
12. Высокий результат `ACC-only` на WESAD указывает на protocol/movement confounding.
13. COLET пока не используется в моделировании: требуется MATLAB-конвертация из MATLAB v7.3 object references.

---

## 14. Основные выводы текущей ветки

1. Окно 10 секунд обосновано частотой обновления PM-метрик.
2. POW+EEG признаки лучше, чем использование только POW или только EEG.
3. `Focus` не является самым сильным PM-таргетом.
4. Наиболее предсказуемые PM-метрики: `Excitement`, `Relaxation`, далее `Stress` и `Focus`.
5. Локальный временной контекст соседних EEG-окон заметно улучшает предсказание PM-метрик.
6. `context-tabular len=5` является наиболее сильным текущим baseline для большинства targets.
7. MHA полезен только для части PM-метрик и не превосходит context-tabular устойчиво.
8. Оптимальная длина контекста зависит от target.
9. WESAD подтвердил, что wearable physiology может быть полезным внешним proxy для stress/arousal-related состояний.
10. BVP/PPG и BVP+TEMP являются наиболее сильными компактными физиологическими группами WESAD.
11. ACC-only на WESAD слишком силен, поэтому ACC следует использовать как контроль movement/protocol confounding.
12. COLET отложен до MATLAB-конвертации.
13. Следующий основной ML-шаг — subject calibration для Emotiv PM targets.
14. Следующий инфраструктурный шаг — master benchmark registry.

---

## 15. Рекомендуемый следующий план

### 15.1. Главный следующий ML-эксперимент

Subject calibration для Emotiv PM targets:

```text
src/27_train_pm_subject_calibration.py
```

Мотивация:

```text
1. В EEG/PM экспериментах есть межсубъектная нестабильность.
2. WESAD подтвердил, что физиологические сигналы также сильно зависят от субъекта.
3. Усложнение модели само по себе не гарантирует прироста.
4. Калибровка пользователя ближе к реальному сценарию: короткий индивидуальный warm-up -> персонализированное предсказание состояния.
```

Варианты постановки:

```text
zero-calibration: train on other subjects -> test on new subject
few-window calibration: добавить первые N окон субъекта в calibration set
linear residual calibration: global model + subject-specific correction
personal normalization: z-score / robust scaling within subject/session
adapter calibration: small calibration head on top of frozen/global features
```

Ожидаемые выходы:

```text
reports/calibration/pm_subject_calibration/
  calibration_summary.csv
  calibration_by_target.csv
  calibration_by_subject.csv
  report.md
  figures/
```

### 15.2. Инфраструктурный шаг

Создать единый benchmark registry:

```text
src/28_build_pm_benchmark_registry.py
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

### 15.3. Следующие EEG/PM эксперименты

```text
src/29_train_band_ablation_baselines.py
src/30_train_context_pooling_baselines.py
src/31_train_session_context_baselines.py
src/32_build_connectivity_features.py
src/33_train_connectivity_augmented_baselines.py
src/34_build_pm_state_events.py
src/35_train_pm_event_baselines.py
src/36_analyze_pm_target_relationships.py
src/37_train_multitask_pm_model.py
```

Рекомендуемый порядок:

```text
1. subject calibration
2. master benchmark registry
3. frequency-band ablation
4. compact context pooling
5. session-level context
6. connectivity/channel features
7. PM event detection
8. multi-task PM model
```

### 15.4. Wearable / external datasets

```text
WESAD: считать текущий этап завершенным и использовать как внешний аргумент для wearable stress proxy.
COLET: держать в backlog до появления MATLAB или готовой MATLAB-конвертации.
```

---

## 16. Минимальная последовательность воспроизведения

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

Воспроизвести WESAD wearable benchmark:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\16_inspect_wesad_dataset.py

D:\miniconda3\envs\eeg_nir\python.exe src\17_prepare_wesad_windowed_dataset.py `
  --output-name wesad_windowed_stress_dataset

D:\miniconda3\envs\eeg_nir\python.exe src\18_train_wesad_stress_baseline.py `
  --dataset data\processed\wesad_windowed_stress_dataset.parquet `
  --fast `
  --run-name wesad_stress_full

D:\miniconda3\envs\eeg_nir\python.exe src\19_analyze_wesad_stress_results.py `
  --run-dir reports\wearable_pm_alignment\runs\<wesad_stress_full_run_id>

D:\miniconda3\envs\eeg_nir\python.exe src\20_train_wesad_feature_group_ablation.py `
  --dataset data\processed\wesad_windowed_stress_dataset.parquet `
  --fast `
  --run-name wesad_feature_group_ablation

D:\miniconda3\envs\eeg_nir\python.exe src\21_train_wesad_protocol_control.py `
  --dataset data\processed\wesad_windowed_stress_dataset.parquet `
  --fast `
  --models logistic_robust,lgbm_clf `
  --run-name wesad_protocol_control

D:\miniconda3\envs\eeg_nir\python.exe src\26_build_wesad_summary_report.py
```

---

## 17. Ключевой результат для отчета

```text
В рамках текущей ветки реализован pipeline предсказания PM-метрик Emotiv по EEG/POW-признакам. После построения 10-секундного оконного датасета были обучены однооконные tabular baselines, temporal multi-head self-attention models и context-tabular baselines. Эксперименты показали, что локальный временной контекст соседних EEG-окон существенно улучшает качество предсказания PM-метрик. Однако механизм multi-head attention не дал устойчивого преимущества над простым context-tabular baseline. Наиболее сильный текущий подход — context-tabular len=5 + LGBM/HGB. Для Focus достигнуто R2≈0.345 и Spearman≈0.568, для Excitement — R2≈0.579 и Spearman≈0.718, для Relaxation — R2≈0.426 и Spearman≈0.642.

Дополнительно завершена внешняя wearable-линия на WESAD. Был построен 60-секундный оконный stress dataset, обучены stress baselines, проведены threshold analysis, feature-group ablation и protocol-control experiment. Лучший компактный default-вариант — BVP+TEMP + logistic regression, balanced accuracy≈0.843. BVP-only также силен: balanced accuracy≈0.835. При этом ACC-only дает сопоставимое качество, что указывает на movement/protocol confounding. Поэтому для будущего PM.Stress alignment ACC следует использовать как контрольный канал, а основной физиологический wearable proxy строить на BVP/PPG, EDA и TEMP.

COLET скачан и проинспектирован, но временно отложен: MATLAB v7.3 object references слишком медленно читаются через Python/h5py. Для продолжения нужна MATLAB-конвертация .mat -> CSV/Parquet. Следующий основной шаг проекта — subject calibration для Emotiv PM targets, потому что и EEG/PM, и wearable-эксперименты показывают значимую межсубъектную вариативность.
```
