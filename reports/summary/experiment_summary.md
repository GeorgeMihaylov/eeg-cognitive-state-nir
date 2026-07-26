# Сводка экспериментов

Ручной состав и статусы задаются в `experiment_registry.yaml`; числовые значения извлекаются только из явно указанных источников.

## Итоговые результаты

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **ShallowConvNet на deduplicated raw EEG**<br>Лучший из двух проверенных raw-CNN; macro F1 0.2647 по трём seeds. Ограничения: Сравнение архитектур описательное, без утверждения значимости. | cognitive_load_5class | torch_shallow_convnet | 5-fold GroupKFold by subject_id | 7,42,123 | 54 | macro_f1 = 0.2647134257 | balanced_accuracy=0.2838635014; accuracy=0.2825338117 | final | [отчёт](../../reports/raw_eeg_cnn_model_comparison.md) |
| **Transformer для label_q5**<br>Устойчивый temporal encoder, близкий к LSTM/BiLSTM без явного общего превосходства. Ограничения: Sequence identities отличаются от recurrent baseline. | cognitive_load_5class | torch_transformer | 5-fold GroupKFold by subject_id | 7,42,123 | 54 | macro_f1 = 0.3568413722 | balanced_accuracy=0.3654730138; accuracy=0.3629381429 | final | [отчёт](../../reports/transformer_benchmark_report.md) |

## Базовые результаты

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **Random Forest baseline для label_q5**<br>Оконный классический baseline с macro F1 0.2955. Ограничения: Не моделирует временной контекст между окнами. | cognitive_load_5class | random_forest | 5-fold GroupKFold by subject_id | 42 | 54 | macro_f1 = 0.29545516 | balanced_accuracy=0.3059459905; accuracy=0.302064436 | baseline | [отчёт](../../reports/transformer_benchmark_report.md) |
| **Gap-aware BiLSTM для label_q5**<br>Recurrent baseline с macro F1 0.3570. Ограничения: Выполнен только seed 42. | cognitive_load_5class | torch_bilstm_gapaware | 5-fold GroupKFold by subject_id | 42 | 54 | macro_f1 = 0.3570419693 | balanced_accuracy=0.3681302205; accuracy=0.3652619104 | baseline | [отчёт](../../reports/transformer_benchmark_report.md) |
| **EEGNet на deduplicated raw EEG**<br>Валидный raw-EEG baseline, уступающий ShallowConvNet и feature sequences. Ограничения: Гиперпараметры EEGNet не оптимизировались. | cognitive_load_5class | torch_eegnet | 5-fold GroupKFold by subject_id | 7,42,123 | 54 | macro_f1 = 0.2236312311 | balanced_accuracy=0.2525412545; accuracy=0.2518951282 | baseline | [отчёт](../../reports/raw_eeg_cnn_model_comparison.md) |
| **Gap-aware LSTM для label_q5**<br>Сильный recurrent baseline; macro F1 0.3555. Ограничения: Последовательности длины 10 не выровнены один-к-одному с Transformer. | cognitive_load_5class | torch_lstm_gapaware | 5-fold GroupKFold by subject_id | 42 | 54 | macro_f1 = 0.3554619934 | balanced_accuracy=0.3697380654; accuracy=0.3673394838 | baseline | [отчёт](../../reports/transformer_benchmark_report.md) |
| **Torch MLP baseline для label_q5**<br>Базовый feature MLP уступил RF и временным моделям. Ограничения: Использует отдельные окна без межоконного контекста. | cognitive_load_5class | torch_mlp | 5-fold GroupKFold by subject_id | 42 | 54 | macro_f1 = 0.2739963828 | balanced_accuracy=0.2822211487; accuracy=0.2786039326 | baseline | [отчёт](../../reports/transformer_benchmark_report.md) |
| **Mean baseline многовыходной PM-регрессии**<br>Наивная нижняя граница качества для семи PM targets. Ограничения: Не использует EEG+POW признаки. | performance_metrics_regression | mean_regressor | 5-fold GroupKFold by subject_id | 42 | 53 | macro_mae = 0.1109889428 | macro_rmse=0.1442927509; macro_r2=-0.007826335213 | baseline | [отчёт](../../reports/integration/pm_multioutput_regression.md) |
| **Random Forest baseline многовыходной PM-регрессии**<br>RF устойчиво превзошёл mean baseline по всем пяти folds. Ограничения: Классический оконный baseline без персонализации. | performance_metrics_regression | random_forest | 5-fold GroupKFold by subject_id | 42 | 53 | macro_mae = 0.1002780125 | macro_rmse=0.1314386269; macro_r2=0.1442974577; macro_spearman=0.3314982822 | baseline | [отчёт](../../reports/integration/pm_multioutput_regression.md) |

## Предобработка и диагностика

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **Абляция предобработки raw EEG A–H**<br>Band-pass и notch дали малые нестабильные BA-изменения; CAR ухудшал seed-42 contrasts, значимость не заявлялась. Ограничения: Полный A–H factorial выполнен только для seed 42. | cognitive_load_5class | torch_shallow_convnet | 5-fold GroupKFold by subject_id | 7,42,123 | 54 | raw_macro_f1 = 0.2647134256 | raw_balanced_accuracy=0.2838635014 | diagnostic | [отчёт](../../reports/preprocessing_selected_trials_multiseed.md) |
| **Leakage-safe standard_clip для EEG+POW**<br>Train-only clipping q0.5/q99.5 стабилизировало feature MLP и стало production preprocessing. Ограничения: Диагностический fold 1, не самостоятельный финальный benchmark. | performance_metrics_regression | torch_mlp | GroupKFold fold 1 diagnostic | 42 | 53 | macro_mae = 0.1049 | macro_rmse=0.1399; macro_r2=0.0471 | diagnostic | [отчёт](../../reports/integration/robust_feature_scaling_audit.md) |

## Персонализация

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **Трёхсидовая персонализация label_q5**<br>Full-model дал положительный macro F1 gain с subject-bootstrap CI выше нуля. Ограничения: Проверены только MLP, три seeds и бюджет 20%. | cognitive_load_5class | torch_mlp | 5-fold GroupKFold plus chronological user calibration | 7,42,2026 | 53 | macro_f1_gain = 0.0065691825 | accuracy_gain=0.01711659105; balanced_accuracy_gain=0.004350084654 | final | [отчёт](../../reports/integration/personalization_multiseed_20pct.md) |
| **Трёхсидовая персонализация PM-регрессии**<br>Full-model устойчиво снижает macro MAE во всех трёх seeds. Ограничения: Проверены только MLP, три seeds и бюджет 20%. | performance_metrics_regression | torch_mlp | 5-fold GroupKFold plus chronological user calibration | 7,42,2026 | 53 | macro_mae_gain = 0.002684712851 | macro_rmse_gain=0.002411094174; macro_spearman_gain=0.01198536362 | final | [отчёт](../../reports/integration/pm_regression_personalization_multiseed_20pct.md) |

## Mixins

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **Аудит Contrastive learning mixin**<br>Synthetic smoke выполнен, но contrastive pipeline не интегрирован. Ограничения: Псевдо-raw reshaping агрегированных признаков методологически неприемлем. | mixin_audit | historical_contrastive_mixin | isolated synthetic encoder smoke | 42 |  | audit_decision |  | diagnostic | [отчёт](../../reports/integration/feature_benchmarking_mixins_audit.md) |
| **Аудит Domain adaptation mixin**<br>DANN не интегрирован и требует нового encoder/domain contract. Ограничения: gpn_data и Old_EEG нельзя автоматически трактовать как разные устройства. | mixin_audit | historical_dann_mixin | isolated contract audit | 42 |  | audit_decision |  | diagnostic | [отчёт](../../reports/integration/feature_benchmarking_mixins_audit.md) |
| **Аудит Meta-learning mixin**<br>MAML отложен; validated fine-tuning остаётся более обоснованным. Ограничения: Опциональная зависимость и runner path отсутствуют. | mixin_audit | historical_maml_mixin | dependency and episodic-contract audit | 42 |  | audit_decision |  | diagnostic | [отчёт](../../reports/integration/feature_benchmarking_mixins_audit.md) |
| **Аудит Transfer learning mixin**<br>Исторический класс не production-ready, функциональность интегрирована заново. Ограничения: Нельзя использовать старый calibration path как доказательство transfer gain. | mixin_audit | historical_transfer_mixin | isolated smoke plus production-path comparison | 42 |  | audit_decision |  | diagnostic | [отчёт](../../reports/integration/feature_benchmarking_mixins_audit.md) |

## Невалидные и заменённые запуски

| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |
|---|---|---|---|---:|---:|---|---|---|---|
| **Первый GroupKFold run robust scaling без model-level config**<br>Исключён, потому что model-level scaling config не дошёл до GroupKFold model build. Ограничения: Метрики нельзя интерпретировать как результат robust scaling. | performance_metrics_regression | torch_mlp | GroupKFold fold 1 diagnostic | 42 | 53 | validation_status |  | invalidated | [отчёт](../../reports/integration/robust_feature_scaling_audit.md) |

## Неразрешённые записи

| Experiment ID | Отсутствует | Проверено | Рекомендуемое действие |
|---|---|---|---|
| label_q5_lightgbm_baseline | structured final result and primary metric | reports,benchmark_results,configs,experiments | Выполнить или импортировать канонический 5-fold GroupKFold run и tracked report. |
| label_q5_histgradientboosting_baseline | structured final result and primary metric | reports,benchmark_results,configs,experiments | Добавить канонический config и 5-fold GroupKFold baseline. |
| label_q5_logistic_regression_baseline | EEG+POW GroupKFold result | reports,benchmark_results,configs,experiments | Не использовать temporal diagnostic D1–D3; выполнить отдельный EEG+POW baseline. |
| pm_regression_torch_mlp_5fold | comparable five-fold final result | reports/integration/pm_multioutput_regression.md,benchmark_results/pm_regression_smoke,experiments/pm_regression | Зарегистрировать после отдельного полного five-fold run; текущий Torch MLP является fold-1 smoke. |
