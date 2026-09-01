# Raw EEG → EEGNet baseline для `label_q5`

## Итог

В ветке `feature/model-zoo-dl` реализован отдельный pipeline настоящих raw EEG окон. EEGNet получает временные тензоры `[batch, 1, 14, 2560]`; агрегированные EEG/POW признаки из processed Parquet в модель не подаются. Parquet используется только как источник metadata, границ 10-секундных окон и `label_q5`.

Финальный протокол — 5-fold `GroupKFold` по `subject_id`. Списки test subjects совпадают с прежними grouped benchmarks. Inner validation разделена по source-independent `record_group_id`, поэтому одна и та же запись из `Old_EEG` и `gpn_data` не может оказаться одновременно в inner train и validation.

## Raw-аудит

- Каталог: 120 raw-файлов; 119 source-specific records имеют supervised окна.
- Источники: 71 `gpn_data` bzip2-файл и 49 несжатых `Old_EEG` файлов.
- Каналы: общая упорядоченная схема из 14 сигналов — `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`.
- Время: `Timestamp` в Unix seconds; выборка выполняется по абсолютным timestamp boundaries, а не по номерам строк.
- Частоты: 116 файлов имеют nominal 256 Hz, 4 файла — nominal 128 Hz. Только 128 Hz записи проходят `scipy.signal.resample_poly` до 256 Hz.
- Дубликаты/обратное время: 0/0. Найдено 23 разрыва больше 1.5 nominal intervals и 14 NaN-ячеек в одном `gpn_data` файле.
- Амплитуды сохраняются в numeric units экспорта; физическая калибровка единиц в исходных metadata не подтверждена.
- 119 source-specific records соответствуют 86 logical recordings; 33 записи представлены в обоих источниках.

Подробный аудит: `reports/raw_eeg_audit.md`. Машинная схема: `data/interim/raw_eeg_schema.json` (игнорируется Git).

## Индекс, QC и кэш

Исходная supervised-выборка после удаления строк без target: **45 384** окна.

| Статус | Окон |
|---|---:|
| accepted | 45 326 |
| rejected: вне raw-диапазона | 38 |
| rejected: missing fraction > 2% | 20 |

Accepted class distribution: `0: 9058`, `1: 9067`, `2: 9069`, `3: 9067`, `4: 9065`.

Индекс: `data/interim/raw_eeg_window_index_w10.parquet`. Кэш: 119 `.npy` shards и 119 JSON manifests, 6.06 GiB. Shards читаются через memory mapping; config hash включает raw size/mtime, канал-схему, sfreq, QC threshold и границы окон. Повторная проверка переиспользовала 119/119 shards.

## Модель

EEGNet baseline:

1. temporal convolution, kernel `0.5 s = 128 samples`;
2. depthwise spatial convolution через все 14 каналов;
3. BatchNorm, ELU, average pooling, dropout;
4. depthwise-separable temporal convolution, kernel `0.125 s = 32 samples`;
5. linear 5-class head.

Общий PyTorch adapter предоставляет `fit`, `predict`, `predict_proba`, `save`; использует DataLoader, AdamW, CrossEntropyLoss, seed 42, `device: auto`, early stopping и восстановление лучшего state. Channel-wise normalization вычисляется только на inner-train каждого fold и сохраняется отдельно.

## Команды

```powershell
C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe scripts\data\audit_raw_eeg.py
C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe scripts\data\build_raw_eeg_window_cache.py
C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe cli.py --config configs\smoke_torch_eegnet_label_q5.yaml --verbose
C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe cli.py --config configs\groupkfold_torch_eegnet_label_q5.yaml --verbose
C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe -m pytest -q
```

## Smoke-run

- 1 прежний outer fold; 1000 детерминированно выбранных окон; все 54 subjects сохранены в подвыборке.
- Train/test: 803/197; test subjects: 11.
- CUDA: NVIDIA GeForce RTX 5060 Ti.
- 3 эпохи; best epoch 1; best validation loss 1.7135.
- Accuracy 0.1675; balanced accuracy 0.1669; macro-F1 0.1516; weighted-F1 0.1518.

Smoke — только техническая проверка и не является главным научным результатом.

## Финальный 5-fold результат

| Метрика | mean | std |
|---|---:|---:|
| accuracy | 0.2403 | 0.0220 |
| balanced accuracy | 0.2451 | 0.0256 |
| macro-F1 | 0.2184 | 0.0151 |
| weighted-F1 | 0.2165 | 0.0151 |
| Cohen's kappa | 0.0543 | 0.0302 |
| weighted OVR AUC | 0.5684 | 0.0307 |

| Fold | train/test | epochs / best | best val loss | accuracy | balanced acc. | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 36213 / 9113 | 8 / 4 | 1.5977 | 0.2683 | 0.2861 | 0.2292 |
| 2 | 36349 / 8977 | 5 / 1 | 1.6035 | 0.2056 | 0.2125 | 0.1904 |
| 3 | 36232 / 9094 | 11 / 7 | 1.5564 | 0.2451 | 0.2524 | 0.2337 |
| 4 | 36278 / 9048 | 11 / 7 | 1.5981 | 0.2556 | 0.2509 | 0.2203 |
| 5 | 36232 / 9094 | 6 / 2 | 1.5435 | 0.2266 | 0.2237 | 0.2184 |

Среднее число эпох: 8.2; средний best epoch: 4.2; средний best validation loss: 1.5798. Суммарное training time: 815.4 s.

## Сравнение с grouped baselines

| Модель / вход | Accuracy | Balanced acc. | Macro-F1 |
|---|---:|---:|---:|
| RF, EEG+POW features | 0.3021 | 0.3059 | 0.2955 |
| PyTorch MLP, EEG+POW features | 0.2786 | 0.2822 | 0.2740 |
| gap-aware LSTM, feature sequences | 0.3673 | 0.3697 | 0.3555 |
| gap-aware BiLSTM, feature sequences | 0.3653 | 0.3681 | 0.3570 |
| EEGNet, raw 10 s windows | 0.2403 | 0.2451 | 0.2184 |

Это не полностью одинаковые observation units: EEGNet классифицирует отдельное raw-окно, а LSTM/BiLSTM используют последовательности feature windows. Результат показывает, что минимальный raw EEGNet pipeline работает, но текущий baseline уступает engineered-feature моделям.

## Артефакты

- JSON: `benchmark_results/groupkfold_torch_eegnet_label_q5/benchmark_results_20260714_164638.json`
- Summary CSV: `benchmark_results/groupkfold_torch_eegnet_label_q5/summary_20260714_164638.csv`
- Consolidated predictions: `benchmark_results/groupkfold_torch_eegnet_label_q5/20260714_164638/emotiv_raw_eeg/cognitive_load_5class/torch_eegnet/group_kfold_subject/predictions.parquet`
- Fold artifacts находятся в `.../group_kfold_subject/fold_01` … `fold_05`: `model.pt`, `training_log.csv`, `predictions.parquet`, `validation_split.json`, `raw_eeg_stats.json`, `normalization_stats.json`.

Consolidated predictions: 45 326 строк, 0 duplicate `sample_id`, `proba_0…proba_4` присутствуют и суммируются примерно в 1. Все fold validation artifacts имеют нулевые logical-record/group/outer-test overlaps.

## Тесты и ограничения

`pytest -q`: **69 passed**, одно предупреждение sklearn в synthetic test, где test partition намеренно не содержит все классы. `git diff --check` не обнаружил ошибок.

Оставшиеся ограничения:

- `label_q5` — weak target, производный от performance metrics;
- отсутствуют band-pass/notch и отдельный artifact-rejection этап;
- upsampling 128→256 Hz не добавляет новой информации;
- physical units экспортированного EEG не подтверждены;
- raw EEGNet не моделирует межоконный контекст;
- одинаковые recordings из двух sources остаются двумя test observations и могут перевзвешивать некоторые sessions, хотя они совместно группируются во всех splits;
- гиперпараметры EEGNet не оптимизировались.

Данные, shards и веса не добавлялись в Git. Commit/push/merge/rebase не выполнялись.
