# Закрытые отрицательные и количественно не достигнутые результаты

Эти результаты являются исходами выполненных проверок, а не ошибками
реализации и не списком незапущенных экспериментов.

| Направление | Фактический результат | Решение | Evidence |
|---|---|---|---|
| PM smoothing | Median/EMA/Hampel не дали универсального downstream выигрыша; raw лучше по classification, R² и correlations. | Сохранить raw PM как канонический target. | [PM temporal quality](../pm_quality/pm_temporal_quality_v1.md) |
| LightGBM 448 → 50 | Размерность −88.84% и fit ×6.78 быстрее, но Macro F1 0.418690 → 0.411554 и MAE 0.098432 → 0.099252. | Использовать top-50 для экономии, не как улучшение качества. | `benchmark_results/lightgbm_feature_selection_v1/final_report.md` |
| ShallowConvNet CAR | Средний описательный CAR-эффект по Balanced Accuracy −0.0285. | Не использовать CAR как default. | [Preprocessing ablation](../preprocessing_factorial_ablation.md) |
| Ordinal Transformer | Ordinal objectives снижают ordinal errors, но не дают устойчивого Balanced Accuracy gain. | Categorical Transformer — primary reference. | [Ordinal statistics](../ordinal_transformer_multiseed_statistics.md) |
| Classification personalization, Accuracy ≥75% | После full-model средняя Accuracy около 0.3138, максимум 0.634921; 0/53 участников достигли 0.75. | Количественный порог не достигнут; не объявлять эксперимент незавершённым. | [Personalization](personalization_multiseed_20pct.md) |
| Classification personalization, full vs head | Full-model не универсально лучше head-only по всем участникам/метрикам. | Показывать оба режима и heterogeneity. | [Personalization](personalization_multiseed_20pct.md) |
| Raw-deduplicated FOMAML | ΔMacro F1 −0.046338 и Δordinal MAE +0.449093 против supervised full-model. | `do_not_proceed`. | [FOMAML](fomaml_label_q5_raw_diagnostic.md) |
| COG-BCI CNN | EEGNet/ShallowConvNet близки к chance в N-Back diagnostic. | Закрыть CNN-only exploration. | [COG-BCI CNN](cog_bci_nback_baseline.md) |
| COG-BCI preprocessing | Фильтрация подавила nuisance contamination, но не улучшила inner criterion. | Не расширять preprocessing search. | [COG-BCI preprocessing](cog_bci_nback_preprocessing_ablation.md) |
| COG-BCI 62 channels | ΔBalanced Accuracy +0.0077, ниже decision threshold +0.03. | `retain_14_channel_cache`. | [Spectral benchmark](cog_bci_nback_spectral_benchmark.md) |
| Contrastive transfer | Shape-only и time-aligned pretraining не превзошли random initialization downstream. | `close_transfer_track`. | [Shape-only](cog_bci_contrastive_transfer_screening.md), [time-aligned](cog_bci_time_aligned_transfer_screening.md) |
| Multimodal improvement 5–10% | Знак и величина эффекта зависят от dataset/model; Shallow fusion отрицателен на CL-Drive и CLARE. | Не заявлять универсальный эффект; применять dataset-specific evaluation. | [Multimodal summary](../external_datasets/multimodal_external_dataset_recommendation.md) |
| Full 399-feature online profile | Total P95 3052.311 ms превышает 1-секундный update budget. | Для online использовать lightweight; full оставить offline/для оптимизации. | `benchmark_results/streaming_scientific_v1/run_summary.json` |

DANN не включён в таблицу отрицательных результатов: его статус
`partially_confirmed`, а не `confirmed` и не `closed_negative`. Полная
FASTER-like/ICA ablation и selected-model seven-PM benchmark также не включены:
они не выполнены и поэтому не имеют ни положительного, ни отрицательного
научного результата.
