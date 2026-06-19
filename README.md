# EEG Cognitive State NIR

## Тема НИР

**Разработка и валидация методов моделирования латентных proxy-состояний человека по данным электроэнцефалографии с учетом межсубъектной вариабельности**

## Краткое описание

Проект посвящен разработке и валидации методов анализа состояний человека по данным электроэнцефалографии. Вместо прямого предсказания отдельных Performance Metrics по независимым EEG-окнам в проекте используется более устойчивая постановка: моделирование временной траектории пользователя в латентном пространстве proxy-состояний.

Латентные proxy-состояния строятся на основе сглаженных PM-метрик и затем предсказываются по последовательностям EEG/POW-признаков. Основной акцент сделан на межсубъектной вариабельности: модель проверяется на новых субъектах через subject-wise разбиение и дополняется персональной head-only калибровкой.

Важно: используемые состояния являются **PM-derived proxy-состояниями**, а не прямыми объективными измерениями когнитивно-аффективного состояния человека.

---

## Основная идея

Исходная задача прямого предсказания отдельных PM-метрик по окнам EEG/POW показала ограниченную устойчивость при переносе на новых пользователей. Поэтому была сформулирована новая постановка:

```text
EEG/POW sequence → Transformer → latent proxy-state trajectory → personal calibration
```

Финальный подход:

1. унифицировать EEG/PM-данные из нескольких источников;
2. построить slow-компоненты PM-метрик;
3. выделить латентные proxy-оси с помощью PCA;
4. обучить temporal Transformer для предсказания `slow_pca_1..3`;
5. провести персональную head-only калибровку для новых субъектов;
6. проверить качество на feature, temporal, split-seed и naive baselines.

---

## Данные

В проекте используются два основных корпуса EEG-записей:

```text
data/raw/gpn_data/
data/raw/Old_EEG/
```

### gpn_data

Основной корпус проекта. Содержит EEG-записи, PM-метрики, маркеры и аннотации.

### Old_EEG

Ранее собранный корпус, близкий по типу данных, оборудованию и условиям записи.

Оба корпуса объединяются в единый обработанный датасет. При этом переносимость на другие устройства, другие протоколы и другие типы EEG-систем в текущей работе не утверждается.

Основной обработанный датасет:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

Датасет латентных slow-состояний:

```text
reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet
```

Используемые PM-метрики:

```text
Attention
Engagement
Excitement
Stress
Relaxation
Interest
Focus
```

PM-метрики используются как weak/proxy-разметка и не используются как входные признаки модели.

---

## Латентные proxy-состояния

Для построения целевого пространства использовались slow-компоненты PM-метрик. На них была применена PCA-декомпозиция.

Финальные латентные оси:

| Ось          | Интерпретация                               | Статус                         |
| ------------ | ------------------------------------------- | ------------------------------ |
| `slow_pca_1` | Stress / Arousal / общая активация          | используется                   |
| `slow_pca_2` | Recovery / Fatigue / Relaxation             | используется                   |
| `slow_pca_3` | Workload / Attention / когнитивный контроль | используется                   |
| `slow_pca_4` | Engagement / Involvement                    | исключена как менее стабильная |

Финальная модель предсказывает:

```text
slow_pca_1
slow_pca_2
slow_pca_3
```

---

## Архитектура решения

Общий pipeline:

```text
raw EEG/PM data
    ↓
data inventory and validation
    ↓
windowing and synchronization
    ↓
EEG/POW feature extraction
    ↓
PM slow-component construction
    ↓
PCA latent proxy-state space
    ↓
sequence dataset construction
    ↓
TransformerEncoder
    ↓
zero-shot prediction
    ↓
personal head-only calibration
    ↓
evaluation and baseline reports
```

Финальная temporal-модель:

```text
EEG/POW sequence
    → input projection
    → positional encoding
    → TransformerEncoder
    → pooling
    → regression head
    → slow_pca_1..3
```

---

## Контроль утечки данных

В проекте используется строгая subject-wise схема оценки:

* train, validation и test разделяются по субъектам;
* окна одного субъекта не попадают одновременно в разные выборки;
* preprocessing, imputation и scaling обучаются только на train;
* validation используется для выбора протокола;
* test используется только для финальной оценки;
* персональная калибровка использует только начальный фрагмент held-out субъекта, а оценка проводится на оставшейся части последовательности.

---

## Baseline v1

Ветка `main` фиксирует интегрированный baseline v1:

```text
Baseline v1: personal calibration of latent EEG proxy-state trajectories
```

Финальная конфигурация:

| Параметр           | Значение                                 |
| ------------------ | ---------------------------------------- |
| `feature_set`      | `pow_plus_eeg`                           |
| `seq_len`          | `8`                                      |
| `targets`          | `slow_pca_1`, `slow_pca_2`, `slow_pca_3` |
| `calibration_lr`   | `0.0001`                                 |
| `calibration_frac` | `0.20`                                   |
| split              | subject-wise                             |
| split seeds        | `42, 123, 2024, 3407, 777`               |

Основной отчет baseline v1:

```text
reports/baseline_v1/baseline_v1_report.md
```

Сводная матрица гипотез:

```text
reports/baseline_v1/hypothesis_baseline_matrix.csv
```

JSON-сводка:

```text
reports/baseline_v1/baseline_v1_summary.json
```

---

## Основные результаты

### Feature ablation

| Feature set    | Test R² | Test Spearman |
| -------------- | ------: | ------------: |
| `pow_plus_eeg` |  0.2398 |        0.5804 |
| `eeg`          |  0.1915 |        0.5433 |
| `pow`          |  0.1410 |        0.5243 |

Финальным выбран объединенный набор признаков:

```text
pow_plus_eeg
```

### Персональная head-only калибровка

| Режим      | Mean R² | Mean Spearman |
| ---------- | ------: | ------------: |
| Zero-shot  | -0.0530 |        0.5478 |
| Calibrated |  0.2398 |        0.5804 |
| Gain       | +0.2928 |       +0.0326 |

Калибровка проводится для held-out субъекта: первые 20% последовательностей используются для дообучения только regression head, а оставшиеся 80% — для оценки.

### Temporal baselines

| Модель          |   Test R² | Test Spearman |
| --------------- | --------: | ------------: |
| Transformer     |    0.2614 |        0.6094 |
| GRU             |    0.0442 |        0.5231 |
| mean_pool_mlp   |   -8.2988 |        0.4731 |
| last_window_mlp | -122.3203 |        0.4877 |

Transformer показал лучший zero-shot результат среди temporal-моделей.

### Split-seed robustness

Финальный протокол был проверен на 5 subject-wise split seeds:

| Метрика       | Zero-shot | Calibrated |    Gain |
| ------------- | --------: | ---------: | ------: |
| Mean R²       |   -0.0299 |     0.2085 | +0.2384 |
| Mean Spearman |    0.5388 |     0.5804 | +0.0416 |

Эффект калибровки остается положительным в среднем по разным subject-wise разбиениям, но величина эффекта зависит от состава test-субъектов.

### Naive baselines

| Baseline                   | Использует историю target | Test mean R² | Интерпретация                                  |
| -------------------------- | ------------------------: | -----------: | ---------------------------------------------- |
| `previous_state`           |                        да |       0.9381 | sanity-check на временную инерцию              |
| `train_mean`               |                       нет | около -0.020 | простая константа не объясняет результат       |
| `subject_calibration_mean` |                       нет | около -0.074 | среднего по calibration-фрагменту недостаточно |
| `subject_calibration_last` |                       нет | около -0.330 | последнее calibration-значение недостаточно    |

`previous_state` показывает очень высокий R², но использует истинное предыдущее значение целевой переменной. Поэтому это не deployable EEG-only baseline, а sanity-check на гладкость и автокорреляцию slow-состояний.

---

## Структура репозитория

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
│   │   └── ...
│   │
│   └── processed/
│       ├── windowed_pm_dataset_w10.parquet
│       ├── windowed_eeg_pm_dataset_w10.parquet
│       └── ...
│
├── src/
│   ├── 31_build_pm_latent_states.py
│   ├── 32_build_pm_state_dynamics.py
│   ├── 33_train_pm_dynamics_baselines.py
│   ├── 35_build_and_train_slow_latent_states.py
│   ├── 36_user_calibration_experiments.py
│   ├── 37_build_device_metric_alignment_report.py
│   ├── 44_run_seq_len_sensitivity.py
│   ├── 46_run_reliable_axes_calibration_val_test.py
│   ├── 48_train_temporal_baselines.py
│   ├── 49_summarize_final_experiments.py
│   ├── 50_run_split_seed_robustness.py
│   ├── 51_run_naive_hypothesis_baselines.py
│   └── 52_run_integrated_baseline_v1.py
│
├── reports/
│   ├── slow_latent_states/
│   ├── feature_ablation_v2/
│   ├── temporal_baselines/
│   ├── split_seed_robustness/
│   ├── naive_hypothesis_baselines/
│   ├── final_experiment_summary/
│   └── baseline_v1/
│
└── README.md
```

---

## Установка окружения

Рекомендуется использовать отдельное conda-окружение:

```powershell
conda create -n eeg_nir python=3.10
conda activate eeg_nir
```

Установка зависимостей:

```powershell
pip install numpy pandas scikit-learn pyarrow matplotlib torch lightgbm
```

Если в репозитории используется `requirements.txt`, предпочтительно установить зависимости так:

```powershell
pip install -r requirements.txt
```

---

## Быстрый запуск baseline v1

Сбор уже готовых результатов без переобучения:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\52_run_integrated_baseline_v1.py `
  --root . `
  --mode summarize
```

Ожидаемые выходные файлы:

```text
reports/baseline_v1/baseline_v1_report.md
reports/baseline_v1/baseline_v1_summary.json
reports/baseline_v1/hypothesis_baseline_matrix.csv
reports/baseline_v1/artifact_index.csv
reports/baseline_v1/README.md
```

---

## Воспроизведение основных экспериментов

### 1. Temporal baselines

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\48_train_temporal_baselines.py `
  --root . `
  --dataset reports\slow_latent_states\pm_w10\slow_pm_latent_states_w10.parquet `
  --output-dir reports\temporal_baselines\pow_plus_eeg_seq8_pca123 `
  --models last_window_mlp,mean_pool_mlp,gru,transformer `
  --feature-set pow_plus_eeg `
  --targets slow_pca_1,slow_pca_2,slow_pca_3 `
  --seq-len 8 `
  --calibration-lr 0.0001 `
  --calibration-frac 0.20 `
  --device cuda
```

### 2. Split-seed robustness

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\50_run_split_seed_robustness.py `
  --root . `
  --dataset reports\slow_latent_states\pm_w10\slow_pm_latent_states_w10.parquet `
  --output-dir reports\split_seed_robustness\pow_plus_eeg_seq8_pca123 `
  --seeds 42,123,2024,3407,777 `
  --feature-set pow_plus_eeg `
  --targets slow_pca_1,slow_pca_2,slow_pca_3 `
  --seq-len 8 `
  --calibration-lr 0.0001 `
  --calibration-frac 0.20 `
  --device cuda
```

### 3. Naive baselines

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\51_run_naive_hypothesis_baselines.py `
  --root . `
  --dataset reports\slow_latent_states\pm_w10\slow_pm_latent_states_w10.parquet `
  --output-dir reports\naive_hypothesis_baselines\pow_plus_eeg_seq8_pca123 `
  --seeds 42,123,2024,3407,777 `
  --feature-set pow_plus_eeg `
  --targets slow_pca_1,slow_pca_2,slow_pca_3 `
  --seq-len 8 `
  --calibration-frac 0.20
```

### 4. Integrated baseline summary

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\52_run_integrated_baseline_v1.py `
  --root . `
  --mode summarize
```

---

## Полное воспроизведение baseline-блоков

Если нужно пересобрать основные baseline-блоки через интегрирующий runner:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\52_run_integrated_baseline_v1.py `
  --root . `
  --mode reproduce `
  --skip-existing `
  --run-temporal-baselines `
  --run-split-seeds `
  --run-final-summary `
  --run-naive `
  --device cuda
```

---

## Основные артефакты

| Артефакт                                                              | Назначение                                 |
| --------------------------------------------------------------------- | ------------------------------------------ |
| `reports/baseline_v1/baseline_v1_report.md`                           | итоговый интегрированный отчет baseline v1 |
| `reports/baseline_v1/baseline_v1_summary.json`                        | машинно-читаемая сводка результатов        |
| `reports/baseline_v1/hypothesis_baseline_matrix.csv`                  | матрица гипотез и проверок                 |
| `reports/final_experiment_summary/final_experiment_summary_report.md` | итоговый технический summary-отчет         |
| `reports/temporal_baselines/pow_plus_eeg_seq8_pca123/`                | сравнение temporal-моделей                 |
| `reports/split_seed_robustness/pow_plus_eeg_seq8_pca123/`             | устойчивость к subject-wise split seeds    |
| `reports/naive_hypothesis_baselines/pow_plus_eeg_seq8_pca123/`        | простые statistical/persistence baselines  |

---

## Ограничения

1. Латентные состояния являются PM-derived proxy-состояниями, а не прямыми объективными измерениями состояния человека.
2. Используются два близких корпуса EEG-данных; переносимость на другие устройства и протоколы не доказана.
3. Персональная калибровка требует начального фрагмента данных нового субъекта.
4. `previous_state` baseline использует историю target и не является deployable EEG-only моделью.
5. Качество EEG-сигнала и артефакты пока не учитываются как отдельный reliability-модуль.
6. Ось `slow_pca_4` исключена из финального протокола из-за меньшей устойчивости.

---

## Дальнейшая работа

Планируемые направления:

1. Проверить чувствительность к доле калибровочных данных: 5%, 10%, 15%, 20%, 30%.
2. Провести source-holdout проверку: `gpn_data → Old_EEG` и `Old_EEG → gpn_data`.
3. Добавить анализ автокорреляции latent targets.
4. Проверить дополнительные режимы персональной калибровки:

   * `bias_only`;
   * `head_only`;
   * `last_block + head`;
   * `full_finetune`.
5. Добавить явный модуль оценки качества EEG-сигнала и артефактов.

---

## Итоговый вывод

В проекте разработан и валидирован pipeline моделирования латентных proxy-состояний человека по EEG/POW-данным. Показано, что представление состояния в виде временной траектории в латентном PM-derived пространстве является более устойчивой постановкой, чем прямое предсказание отдельных PM-метрик по независимым окнам.

Финальный baseline v1 подтверждает, что Transformer-модель с объединенными EEG/POW-признаками и персональной head-only калибровкой дает положительный прирост качества на held-out субъектах и сохраняет эффект при проверке на нескольких subject-wise разбиениях.
