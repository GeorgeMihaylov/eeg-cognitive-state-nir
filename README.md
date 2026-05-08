# EEG Cognitive State NIR

Проект посвящен построению и сравнению моделей машинного обучения для предсказания когнитивных / аффективных состояний по EEG-сигналам. В качестве слабой разметки используются PM-метрики Emotiv, синхронизированные с оконными EEG/POW-признаками.

Основная исследовательская гипотеза текущей ветки:

> Локальный временной контекст соседних EEG-окон улучшает предсказание PM-метрик по сравнению с табличным baseline, использующим только одно окно.

---

## 1. Статус ветки

В текущей ветке реализован полный экспериментальный pipeline:

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
11. Визуализация результатов MHA-прогона.
12. Короткое сравнение attention-window `seq_len=3` и `seq_len=5`.

Основной финальный результат текущей ветки: **multi-head self-attention model с `seq_len=3` улучшает большинство PM-метрик относительно tabular baseline**.

---

## 2. Данные

Используются два источника данных:

```text
D:\PycharmProjects\eeg-cognitive-state-nir\data\raw\gpn_data
D:\PycharmProjects\eeg-cognitive-state-nir\data\raw\Old_EEG
```

После предобработки основной датасет:

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

PM-метрики используются только как `target`, но не как входные признаки.

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
├── src/
│   ├── 04_build_windowed_pm_dataset.py
│   ├── 05_analyze_pm_sampling.py
│   ├── 06_eda_windowed_dataset.py
│   ├── 07_train_baselines.py
│   ├── 08_build_windowed_eeg_features.py
│   ├── 09_train_multi_pm_baselines.py
│   ├── 10_describe_multi_pm_baseline.py
│   ├── 11_train_multihead_attention_short.py
│   └── 12_visualize_mha_all_pm_run.py
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

На текущей машине MHA-эксперименты запускались на CPU. Для ускорения можно использовать CUDA, если PyTorch установлен с поддержкой GPU.

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
seq_len = 3
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

По каждому target:

```text
targets/<target>/fold_metrics.csv
targets/<target>/aggregated_metrics.csv
targets/<target>/predictions.parquet
targets/<target>/epoch_history.csv
targets/<target>/report.md
```

### 6.3. Основные результаты MHA seq_len=3 full

| Rank | Target | R2 mean | Spearman mean | RMSE mean |
|---:|---|---:|---:|---:|
| 1 | excitement | 0.555 | 0.698 | 0.154 |
| 2 | relaxation | 0.365 | 0.628 | 0.132 |
| 3 | stress | 0.274 | 0.469 | 0.116 |
| 4 | engagement | 0.216 | 0.476 | 0.115 |
| 5 | interest | 0.213 | 0.431 | 0.086 |
| 6 | focus | 0.198 | 0.470 | 0.111 |
| 7 | attention | 0.008 | 0.374 | 0.125 |

Главный вывод:

```text
MHA с локальным контекстом из трех окон улучшает качество большинства PM-таргетов относительно tabular baseline.
```

---

### 6.4. Короткое сравнение seq_len=3 и seq_len=5

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

Короткий прогон `seq_len=5`:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\11_train_multihead_attention_short.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 5 `
  --max-samples 10000 `
  --fold-limit 2 `
  --epochs 12 `
  --batch-size 128 `
  --d-model 64 `
  --nhead 4 `
  --num-layers 1 `
  --run-name mha_all_pm_short_len5
```

Контрольный короткий прогон `seq_len=3`:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\11_train_multihead_attention_short.py `
  --dataset data\processed\windowed_eeg_pm_dataset_w10.parquet `
  --pm-target all `
  --feature-set pow_plus_eeg `
  --feature-mode log_pow `
  --seq-len 3 `
  --max-samples 10000 `
  --fold-limit 2 `
  --epochs 12 `
  --batch-size 128 `
  --d-model 64 `
  --nhead 4 `
  --num-layers 1 `
  --run-name mha_all_pm_short_len3_control
```

Сравнение коротких прогонов:

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
seq_len=3 остается основной универсальной MHA-конфигурацией.
seq_len=5 улучшает Focus и Stress, но ухудшает Excitement и Relaxation.
```

---

## 7. Визуализация MHA-прогона

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

Графики, которые рекомендуется включить в отчет:

```text
global/dashboard_main_metrics.png
global/target_metric_heatmap.png
global/r2_bar.png
global/spearman_bar.png
global/fold_heatmap_r2.png
global/fold_heatmap_spearman.png
per_target/excitement/excitement_scatter_y_true_y_pred.png
per_target/relaxation/relaxation_scatter_y_true_y_pred.png
per_target/focus/focus_scatter_y_true_y_pred.png
per_target/excitement/excitement_residual_hist.png
per_target/relaxation/relaxation_calibration_binned.png
```

---

## 8. Таблицы сравнения

### 8.1. Tabular baseline vs MHA seq_len=3

Ключевой результат:

| Target | Tabular R2 | MHA R2 | ΔR2 | Tabular Spearman | MHA Spearman | ΔSpearman |
|---|---:|---:|---:|---:|---:|---:|
| excitement | ~0.336 | 0.555 | +0.219 | ~0.518 | 0.698 | +0.180 |
| relaxation | ~0.203 | 0.365 | +0.162 | ~0.471 | 0.628 | +0.157 |
| engagement | ~0.229 | 0.216 | -0.013 | ~0.427 | 0.476 | +0.049 |
| stress | ~0.157 | 0.274 | +0.117 | ~0.345 | 0.469 | +0.124 |
| interest | ~0.149 | 0.213 | +0.064 | ~0.358 | 0.431 | +0.073 |
| focus | ~0.110 | 0.198 | +0.088 | ~0.342 | 0.470 | +0.128 |
| attention | ~0.104 | 0.008 | -0.096 | ~0.393 | 0.374 | -0.019 |

Вывод:

```text
MHA seq_len=3 улучшает большинство PM-метрик, особенно Excitement, Relaxation, Stress и Focus.
```

### 8.2. MHA seq_len=3 vs seq_len=5 short

См. раздел 6.4.

Вывод:

```text
seq_len=5 не является универсально лучшим. Он улучшает Focus и Stress, но ухудшает наиболее сильные таргеты Excitement и Relaxation.
```

---

## 9. Методологические ограничения

1. PM-метрики Emotiv являются слабой разметкой, а не экспертным ground truth.
2. Модель предсказывает значения PM-метрик, а не когнитивное состояние напрямую.
3. Основная честная оценка качества — `GroupKFold` по `subject_id`.
4. `Random split` может завышать качество из-за leakage между окнами одного субъекта.
5. Источники `gpn_data` и `Old_EEG` имеют доменный сдвиг.
6. Не все PM-метрики одинаково предсказуемы.
7. `Attention` оказался наиболее проблемным PM-таргетом.
8. `seq_len=5` не дает универсального улучшения и требует отдельной настройки под target.

---

## 10. Основные выводы текущей ветки

1. Окно 10 секунд обосновано частотой обновления PM-метрик.
2. POW+EEG признаки лучше, чем использование только POW или только EEG.
3. `Focus` не является самым сильным PM-таргетом.
4. Наиболее предсказуемые PM-метрики: `Excitement`, `Relaxation`.
5. Multi-head self-attention с локальным контекстом `seq_len=3` улучшает большинство PM-метрик.
6. Улучшение наблюдается на subject-aware `GroupKFold`, то есть при переносе на новых субъектов.
7. `seq_len=3` выбран как основная универсальная MHA-конфигурация.
8. `seq_len=5` полезен для `Focus` и `Stress`, но ухудшает сильные таргеты `Excitement` и `Relaxation`.

---

## 11. Рекомендуемый следующий план

1. Зафиксировать финальные таблицы сравнения:
   - tabular baseline vs MHA seq_len=3;
   - MHA seq_len=3 vs MHA seq_len=5 short.
2. Включить в отчет ключевые visualizations:
   - dashboard;
   - heatmap метрик;
   - fold heatmaps;
   - scatter plots для `excitement`, `relaxation`, `focus`;
   - residual plots.
3. Подготовить итоговый технический отчет:
   - данные;
   - признаки;
   - pipeline;
   - baseline;
   - MHA;
   - сравнение;
   - ограничения;
   - дальнейшие работы.
4. В дальнейшем рассмотреть:
   - target-specific `seq_len`;
   - multi-task MHA для всех PM одновременно;
   - domain adaptation между `gpn_data` и `Old_EEG`;
   - синхронизацию annotation-based behavioral targets из `Old_EEG`;
   - анализ артефактов и качества контакта электродов.

---

## 12. Минимальная последовательность воспроизведения

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

Запустить MHA all-PM full baseline:

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

Построить визуализации:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\12_visualize_mha_all_pm_run.py `
  --run-dir reports\runs\<mha_all_pm_full_run_id>
```

---

## 13. Ключевой результат для отчета

```text
В рамках текущей ветки реализован pipeline предсказания PM-метрик Emotiv по EEG/POW-признакам. После построения 10-секундного оконного датасета были обучены tabular baselines и temporal multi-head self-attention baseline. Наиболее сильный результат показала MHA-модель с локальным контекстом из трех окон. Для PM.Excitement.Scaled достигнуто R2≈0.555 и Spearman≈0.698, для PM.Relaxation.Scaled — R2≈0.365 и Spearman≈0.628, для PM.Focus.Scaled — R2≈0.198 и Spearman≈0.470. Это подтверждает гипотезу о полезности локального временного контекста соседних EEG-окон для предсказания PM-состояний.
```
