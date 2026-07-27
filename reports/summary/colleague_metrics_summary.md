# Единая сводка экспериментальных результатов EEG-проекта

## Краткое резюме

1. Честное cross-subject качество заметно ниже диагностического random-window split; эти протоколы не объединяются.
2. Лучший macro F1 для `label_q5` показала модель BiLSTM: 0.3570.
3. Лучший balanced accuracy показала модель LSTM: 0.3697.
4. Random Forest остаётся сильным и воспроизводимым feature-based baseline.
5. Raw-EEG CNN уступают sequence-моделям на текущем deduplicated наборе.
6. PM Random Forest превосходит mean baseline: macro MAE 0.10028 против 0.11099, macro R² 0.14430.
7. Full-model персонализация классификации даёт небольшой macro F1 gain +0.00657; порог accuracy 0.75 не достигнут (наблюдаемый максимум 0.6349206349).
8. Full-model PM-персонализация снижает macro MAE на 0.002685, устойчиво минимум в двух seeds у 66.04% испытуемых.
9. `full_model` лучше `head_only`, но размер дополнительного выигрыша невелик.
10. Различия preprocessing относительно малы; статистическая значимость не заявлялась.
11. CAR дал отрицательный описательный эффект по balanced accuracy (-0.0285).
12. `standard_clip` устраняет экстремальные outlier failures; transfer-функциональность интегрирована через переработанный leakage-safe pipeline.

## 1. Задачи и данные

Пакет разделяет пяти-классовую классификацию `label_q5`, семивыходную PM-регрессию, две задачи персонализации и диагностическую предобработку. Feature windows, feature sequences и deduplicated raw EEG маркируются явно.

## 2. Правила сопоставления результатов

В основные таблицы входят только `final` и `baseline`. `diagnostic` вынесен отдельно, `smoke` и `invalidated` исключены. Основной научный протокол — cross-subject 5-fold GroupKFold по `subject_id`; random-window, single-seed и multi-seed, raw и feature-based результаты не смешиваются без маркировки.

## 3. Классификация label_q5

| Модель | Вход | Seeds | Macro F1 | Balanced accuracy | Accuracy | Статус |
|---|---|---|---|---|---|---|
| BiLSTM | feature_sequence | 42 | 0.3570 ± 0.0168 | 0.3681 ± 0.0194 | 0.3653 | baseline |
| Transformer | feature_sequence | 7|42|123 | 0.3568 ± 0.0253 | 0.3655 ± 0.0237 | 0.3629 | final |
| LSTM | feature_sequence | 42 | 0.3555 ± 0.0273 | 0.3697 ± 0.0239 | 0.3673 | baseline |
| Random Forest | feature_window | 42 | 0.2955 ± 0.0217 | 0.3059 ± 0.0255 | 0.3021 | baseline |
| Torch MLP | feature_window | 42 | 0.2740 ± 0.0126 | 0.2822 ± 0.0168 | 0.2786 | baseline |
| ShallowConvNet | raw_eeg_window | 7|42|123 | 0.2647 ± 0.0132 | 0.2839 ± 0.0139 | 0.2825 | final |
| EEGNet | raw_eeg_window | 7|42|123 | 0.2236 ± 0.0272 | 0.2525 ± 0.0227 | 0.2519 | baseline |

Sequence models дают лучшие macro F1 и balanced accuracy, однако абсолютное cross-subject качество остаётся умеренным. Ordinal Transformer хранится как отдельный диагностический/неканонический разрез и не включён в этот рейтинг.

## 4. Многовыходная PM-регрессия

| Модель | Macro MAE | Macro RMSE | Macro R² | Pearson | Spearman |
|---|---|---|---|---|---|
| random_forest | 0.100278 ± 0.003863 | 0.131439 ± 0.004632 | 0.144297 ± 0.030150 | 0.38380692510451 | 0.331498282212052 |
| mean_regressor | 0.110989 ± 0.004564 | 0.144293 ± 0.004644 | -0.007826 ± 0.002623 |  |  |

Random Forest превосходит средний baseline и даёт положительный macro R². Per-target значения агрегированы только из существующих `per_target_metrics.csv`; отсутствующий absolute bias оставлен пустым.

### Per-target: `pm_regression_mean_baseline_5fold`

| Target | MAE | RMSE | R² | Pearson | Spearman | Abs bias |
|---|---|---|---|---|---|---|
| target_attention | 0.0996785568373318 | 0.127413246261064 | -0.00775724699166946 |  |  |  |
| target_engagement | 0.101122572675292 | 0.130267693313428 | -0.0152321765694068 |  |  |  |
| target_excitement | 0.178373600499657 | 0.225423569101905 | -0.00798002785370424 |  |  |  |
| target_stress | 0.0974964185381043 | 0.13965914982317 | -0.00197731051331278 |  |  |  |
| target_relaxation | 0.136080639343851 | 0.167048249212335 | -0.00670754570769668 |  |  |  |
| target_interest | 0.0669645592855258 | 0.0956122384731632 | -0.00416013555693978 |  |  |  |
| target_focus | 0.0972062524141475 | 0.124625110412088 | -0.0109699032999675 |  |  |  |

### Per-target: `pm_regression_random_forest_5fold`

| Target | MAE | RMSE | R² | Pearson | Spearman | Abs bias |
|---|---|---|---|---|---|---|
| target_attention | 0.0983764488718219 | 0.125676028372744 | 0.0186909412922368 | 0.170739536021906 | 0.143787325123128 |  |
| target_engagement | 0.0926634959833021 | 0.117458188840372 | 0.174704274328676 | 0.435250360179516 | 0.318448682323438 |  |
| target_excitement | 0.14583028601589 | 0.190412256704333 | 0.276293389331264 | 0.533444985097875 | 0.493421847019361 |  |
| target_stress | 0.0910270653269493 | 0.131431178900986 | 0.113081571751846 | 0.365529351316661 | 0.289724966155782 |  |
| target_relaxation | 0.118888960268414 | 0.148041744300173 | 0.207382702662674 | 0.480220465543855 | 0.451948215316935 |  |
| target_interest | 0.063493886674244 | 0.0890204427559602 | 0.127023174558651 | 0.380522593406091 | 0.338040828874709 |  |
| target_focus | 0.0916659446880699 | 0.118030548247635 | 0.0929061499618896 | 0.320941184165664 | 0.285116110671013 |  |

## 5. Персонализация классификации

| Метод | Macro F1 before | After | Gain | 95% CI | Improved ≥2/3 |
|---|---|---|---|---|---|
| full_model | 0.22344187064328 | 0.230011053142805 | 0.00656918249952442 | [0.001499, 0.012117] | 64.15% |
| head_only | 0.22344187064328 | 0.227764450975047 | 0.0043225803317665 | [0.000578, 0.008337] | 60.38% |
| zero_shot | 0.22344187064328 | 0.22344187064328 | 0 | [0.000000, 0.000000] | 0.00% |

Full-model macro F1 gain равен +0.006569; head-only — +0.004323. Full-vs-head разности сохранены в `secondary_metrics_json`. Статистически положительный средний gain не означает достижение абсолютного порога: accuracy 0.75 не достигнута, наблюдаемый максимум 0.6349206349.

## 6. Персонализация PM-регрессии

| Метод | Macro MAE before | After | Reduction | 95% CI | Improved ≥2/3 | All 3 |
|---|---|---|---|---|---|---|
| full_model | 0.105088341855941 | 0.102403629004948 | 0.0026847128509924 | [0.001506, 0.003980] | 66.04% | 52.83% |
| head_only | 0.105088341855941 | 0.103203692666928 | 0.0018846491890124 | [0.000774, 0.003101] | 64.15% | 47.17% |
| zero_shot | 0.105088341855941 | 0.105088341855941 | 0 | [0.000000, 0.000000] | 0.00% | 0.00% |

Full-model против head-only: преимущество по MAE 0.000800, RMSE 0.001174 и Spearman 0.009128. Fine-tuning устойчиво улучшает средние метрики, но не устраняет межсубъектную вариативность.

### Per-target full-model PM personalization

| Target | MAE gain | 95% CI | Improved ≥2/3 |
|---|---|---|---|
| target_attention | 0.00128220674075005 | [-0.000605, 0.003346] | 54.72% |
| target_engagement | 0.00187537456610299 | [0.000278, 0.003667] | 62.26% |
| target_excitement | 0.0072430451442675 | [0.004463, 0.010646] | 75.47% |
| target_stress | 0.00186852872678803 | [-0.000749, 0.004562] | 54.72% |
| target_relaxation | 0.00247493556375509 | [0.000877, 0.004119] | 69.81% |
| target_interest | 0.000898715724422902 | [-0.001394, 0.003201] | 52.83% |
| target_focus | 0.00315018349086031 | [0.000749, 0.006043] | 60.38% |

Наиболее устойчивы excitement, engagement, relaxation и focus; interest имеет минимальный средний эффект.

## 7. Предобработка EEG

| Rank | Trial | Шаги | Balanced accuracy | Macro F1 |
|---|---|---|---|---|
| 1 | E | band-pass + notch | 0.2889 ± 0.0148 | 0.2659 |
| 2 | B | band-pass | 0.2873 ± 0.0156 | 0.2653 |
| 3 | C | notch | 0.2833 ± 0.0170 | 0.2611 |
| 4 | A | raw | 0.2824 ± 0.0170 | 0.2599 |
| 5 | G | notch + CAR | 0.2634 ± 0.0072 | 0.2433 |
| 6 | D | CAR | 0.2620 ± 0.0115 | 0.2396 |
| 7 | F | band-pass + CAR | 0.2514 ± 0.0243 | 0.2315 |
| 8 | H | band-pass + notch + CAR | 0.2510 ± 0.0246 | 0.2315 |

Описательные факторные эффекты balanced accuracy: CAR -0.02854 ± 0.01496; band-pass -0.00309 ± 0.01158; notch +0.00085 ± 0.00340. Различия не объявлялись статистически значимыми.

## 8. Robust scaling

`standard_clip` — diagnostic: maximum train-relative validation z-score 41472.62 → 11.93; outlier-subject MSE 2564.3972 → 0.02872. В one-fold outer test MAE улучшился примерно на 4.9%, RMSE — на 8.2%; это не финальное сравнение моделей.

## 9. Transfer learning и mixins

| Метод | Проверен | Интеграция | Решение |
|---|---|---|---|
| Transfer learning | Да | integrated_as_reimplemented_pipeline | keep |
| Domain adaptation | Да | not_integrated | defer |
| Meta-learning | Да | not_integrated | defer |
| Contrastive learning | Да | not_integrated | defer |

Старый transfer prototype сбрасывал pretrained weights и напрямую не переносился; его назначение реализовано leakage-safe pipeline с `head_only` и `full_model`. DANN не имел корректного source/target contract, MAML — runnable production path, contrastive encoder не был подключён downstream. Prototype smoke metrics не являются научным результатом.

## 10. Основные научные выводы

- Классификация: sequence models лидируют, но cross-subject качество остаётся умеренным.
- PM-регрессия: Random Forest лучше mean baseline и показывает положительный macro R².
- Персонализация: gains малы, но устойчивы; full-model в среднем лучше head-only.
- Предобработка: band-pass и notch дают небольшие различия, CAR в текущем протоколе ухудшает качество.
- Архитектура: платформа поддерживает воспроизводимые GroupKFold-эксперименты, multi-output regression и leakage-safe personalization.

## 11. Ограничения

Наборы входов и единицы наблюдения различаются: feature windows, sequences и raw windows нельзя считать одним однородным рейтингом. Single-seed и multi-seed оценки явно помечены. One-fold diagnostics, smoke и invalidated runs не используются в научных выводах. Новые статистические тесты в рамках сборки не выполнялись.

## 12. Отсутствующие сопоставимые результаты

| Experiment | Config | Report | Metrics | Причина исключения |
|---|---|---|---|---|
| label_q5_lightgbm_baseline | не найден | не найден | не найдены | No structured final 5-fold GroupKFold result and primary metric. |
| label_q5_histgradientboosting_baseline | не найден | не найден | не найдены | No canonical config or structured final 5-fold GroupKFold result. |
| label_q5_logistic_regression_baseline | найден | найден | найдены | Existing temporal diagnostic metrics do not use the comparable EEG+POW GroupKFold input. |
| pm_regression_torch_mlp_5fold | найден | найден | найдены | Available Torch MLP metrics are a one-fold smoke, not a comparable five-fold baseline. |

## 13. Источники и воспроизводимость

Каждая строка таблицы связана с `metrics_provenance.yaml`. Structured CSV используются напрямую; единственные явно зафиксированные tracked-report значения относятся к factorial effects и `standard_clip`. Генератор не читает predictions и не обучает модели.

Команда:

```powershell
python src\18_build_colleague_metrics_package.py --experiment-registry reports\summary\experiment_registry.yaml --config-registry reports\summary\config_registry.yaml --output-dir reports\summary --strict
```
