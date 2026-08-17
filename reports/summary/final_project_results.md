# Итоговый пакет результатов EEG-бенчмарка

Дата актуализации: 2026-08-17. Документ синхронизирован с кодом, tracked
конфигурациями и доступными runtime CSV/JSON. В ходе синхронизации обучение и
перестроение кэшей не выполнялись.

## Научный и программный контур

Основной benchmark использует `gpn_data` и `Old_EEG`, явные target contracts и
outer `GroupKFold` по `subject_id`. Любые обучаемые преобразования —
нормализация, clipping, Q3-пороги и feature selection — fitted только на
train-части. Канонический feature parquet содержит 51 308 окон и 448 EEG+POW
признаков; `label_q5` доступен для 45 384 окон и 54 участников, complete-case
когорта семи PM — для 43 174 окон и 53 участников. Raw-deduplicated контур
содержит 30 958 окон формы `[1, 14, 2560]`.

Современная основная цель проекта — семь PM: Attention, Engagement,
Excitement, Stress, Relaxation, Interest и Focus. `label_q5` сохраняется как
исторический Focus-specific benchmark.

## Результаты

### PM и временная обработка

Полный Random Forest sensitivity-анализ охватывает 7 PM × 4 target variants ×
classification/regression × 5 folds = 280 запусков. Для raw PM средние по 35
PM×fold: classification Macro F1 0.473036 и Balanced Accuracy 0.479122;
regression MAE 0.098373, R² 0.185193, Pearson 0.445013. Causal median, EMA и
Hampel не дали универсального улучшения: classification ухудшилась, а снижение
MAE отдельных сглаженных целей сопровождалось снижением R² и корреляций. Raw PM
остаётся каноническим вариантом. Подробности:
[`pm_temporal_quality_v1.md`](../pm_quality/pm_temporal_quality_v1.md).

### Признаки и LightGBM

Реализованы спектральные, статистические, entropy и connectivity признаки;
канонический EEG+POW baseline использует 448 исходных колонок. Fold-local
selector и LightGBM завершили 140/140 запусков без ошибок:
7 PM × 2 tasks × 2 feature regimes × 5 folds. Переход 448 → 50 уменьшает
размерность на 88.84% и ускоряет downstream fit примерно в 6.78 раза, но
немного ухудшает participant-macro качество: classification Macro F1
0.418690 → 0.411554; regression MAE 0.098432 → 0.099252. Это полезный
вычислительный профиль, а не способ повысить точность.

### Модели

Model zoo содержит Random Forest, LightGBM, XGBoost, MLP, LSTM, BiLSTM,
Transformer, EEGNet, ShallowConvNet и ordinal heads. Исторические полноценные
`label_q5` benchmarks дают Macro F1 0.3570 для BiLSTM, 0.3568 для Transformer,
0.3555 для LSTM, 0.2955 для RF, 0.2647 для ShallowConvNet и 0.2236 для EEGNet.
Preliminary seven-PM model comparison выполнен только на fold 1; его нельзя
называть confirmatory. Полный selected-model seven-PM protocol подготовлен,
но не выполнен: 245 поддерживаемых training units, из них 224 требуют нового
обучения. Его формальная обязательность не устанавливается без исходного ТЗ.

### Предобработка EEG

Полная A–H ablation band-pass/notch/CAR выполнена для seed 42; raw,
band-pass и band-pass+notch дополнительно проверены на seeds 7/42/123. Средний
CAR-контраст по Balanced Accuracy отрицателен (−0.0285); универсального
преимущества фильтрации нет, raw остаётся reference.

Fold-safe FASTER-like/ICA инфраструктура интегрирована с outer-train-only ICA,
mean-channel interpolation и train-only normalization. Четырёхвариантный
smoke (`raw`, `faster`, `ica`, `faster_ica`) завершён. Полная 7 PM × 4
variants × 5 folds = 140 run quantitative ablation не выполнялась, поэтому
вывод о влиянии FASTER-like/ICA на качество отсутствует.

### Персонализация и перенос

Трёхсидовая chronological 20% personalization семи PM даёт небольшой, но
воспроизводимый full-model эффект относительно zero-shot: MAE −0.002685,
RMSE −0.002411 и Spearman +0.011985. Для исторической `label_q5` средняя
Accuracy изменилась примерно с 0.2967 до 0.3138; максимум 0.634921, и 0/53
участников достигли 0.75. Порог Accuracy ≥75% проверен и не достигнут.

Confirmatory DANN `Old_EEG → gpn_data` на folds 1–5 и seeds 123/2026 дал
ΔMacro F1 +0.008048 и ΔBalanced Accuracy +0.008332; participant bootstrap CI
для ΔMacro F1 [−0.001672; 0.017882] включает ноль. Статус —
`partially_confirmed`. Contrastive transfer не дал устойчивого downstream
улучшения; FOMAML получил `do_not_proceed` (ΔMacro F1 −0.046338 против
supervised full-model).

### Мультимодальность

MEFAR, CL-Drive и CLARE проверены в participant-disjoint folds. XGBoost
fusion − EEG-only по Macro F1: +0.113961, +0.011120 и −0.037978
соответственно. Shallow fusion: −0.070163 на CL-Drive и −0.112538 на CLARE.
На MEFAR wearable-only (0.577597) лучше fusion (0.511133). Универсальное
улучшение 5–10% не подтверждено; эффект dataset- и model-specific.

### Streaming и demo

Scientific replay использует 10-секундное окно и обновление раз в 1 секунду.
Full 399-feature profile имеет Total P95 3052.311 ms и не укладывается в
1-секундный бюджет. Lightweight 336-feature profile имеет Feature P95
6.573 ms, Model P95 5.624 ms и Total P95 12.215 ms (realtime factor 63.825×).
Streaming worker, replay, LSL source, quality checks, postprocessing, FastAPI
endpoints и WebSocket реализованы. Это подтверждает software/computational
real-time, но не физическую задержку `сенсор → API/UI`; live EEG-устройство не
проверялось.

## Статус требований

Каноническая матрица находится в
[`final_requirement_coverage.md`](../requirements/final_requirement_coverage.md).
Главные незакрытые пункты: физический end-to-end тест; решение руководителей о
нормативной обязательности selected-model и FASTER-like/ICA full benchmarks;
финальная презентация/текст. Отрицательные результаты не считаются TODO.

## Источники ключевых чисел

- `benchmark_results/lightgbm_feature_selection_v1/execution_manifest.json` и
  `pm_macro_summary.csv`;
- `benchmark_results/streaming_scientific_v1/run_summary.json` и
  `benchmark_results/streaming_scientific_lightweight_v1/run_summary.json`;
- `benchmark_results/mefar_multimodal_xgboost_v1/summary_xgboost.csv`;
- `benchmark_results/cl_drive_multimodal_v1/summary_*.csv`;
- `benchmark_results/clare_multimodal_v1/summary_*.csv`;
- [`personalization_multiseed_20pct.md`](../integration/personalization_multiseed_20pct.md);
- [`pm_regression_personalization_multiseed_20pct.md`](../integration/pm_regression_personalization_multiseed_20pct.md);
- [`dann_label_q5_confirmatory_v2.md`](../integration/dann_label_q5_confirmatory_v2.md);
- [`preprocessing_selected_trials_multiseed.md`](../preprocessing_selected_trials_multiseed.md).
