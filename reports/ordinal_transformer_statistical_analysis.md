# Статистический анализ порядкового Transformer

## 1. Цель

Строгий парный анализ шести завершённых Transformer-вариантов. Независимая единица — `subject_id`; окна, последовательности, folds и источники не используются как независимые наблюдения.

## 2. Анализируемые runs

- `categorical_eeg_only`: `benchmark_results\feature_group_transformer_ablation\runs\transformer_classification_eeg_only\20260718_124023`
- `categorical_eeg_pow`: `benchmark_results\feature_group_transformer_ablation\runs\transformer_classification_eeg_pow\20260718_124412`
- `coral_eeg_only`: `benchmark_results\ordinal_transformer_full_seed42\runs\coral_eeg_only\20260718_160001`
- `coral_eeg_pow`: `benchmark_results\ordinal_transformer_full_seed42\runs\coral_eeg_pow\20260718_160242`
- `corn_eeg_only`: `benchmark_results\ordinal_transformer_full_seed42\runs\corn_eeg_only\20260718_160527`
- `corn_eeg_pow`: `benchmark_results\ordinal_transformer_full_seed42\runs\corn_eeg_pow\20260718_160719`

Все runs: seed 42, sequence length 8, 44 142 последовательности, 53 испытуемых, 5 folds. Smoke-runs исключены.

## 3. Exact alignment

Совпали `sequence_id, fold, subject_id, record_id, source, y_true`: 44,142 строк, 0 расхождений, 0 дубликатов.

## 4. Повторный расчёт метрик

| Method | BA | Macro F1 | QWK | Ordinal MAE | Severe error | Expected-rank MAE | Expected-rank ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| categorical_eeg_only | 0.3304 | 0.2889 | 0.4280 | 1.0528 | 0.2799 | 0.9993 | 0.5286 |
| categorical_eeg_pow | 0.3514 | 0.3135 | 0.4670 | 0.9546 | 0.2401 | 0.9414 | 0.5649 |
| coral_eeg_only | 0.3169 | 0.2946 | 0.4435 | 0.9914 | 0.2573 | 0.9879 | 0.5079 |
| coral_eeg_pow | 0.3389 | 0.3070 | 0.4617 | 0.9739 | 0.2465 | 0.9744 | 0.5422 |
| corn_eeg_only | 0.3164 | 0.2816 | 0.4395 | 0.9936 | 0.2465 | 0.9952 | 0.5184 |
| corn_eeg_pow | 0.3271 | 0.3029 | 0.4591 | 0.9558 | 0.2363 | 0.9637 | 0.5526 |

Fold-метрики пересчитаны из unified predictions и совпали с сохранёнными fold reports в пределах машинной точности.

## 5. Заранее заданные основные гипотезы

В каждой feature group отдельная семья из CORAL/CORN × ordinal MAE/severe error; Holm применён только к Wilcoxon p-values.

## 6. Основные статистические результаты

| Group | Candidate | Metric | Reference mean | Candidate mean | Raw Δ | Improvement [95% CI] | Better/worse/tie | Wilcoxon p | Holm p | Sign p | Rank-biserial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eeg_only | coral_eeg_only | ordinal_mae | 1.0528 | 0.9914 | -0.0614 | 0.0614 [0.0190, 0.1078] | 32/21/0 | 0.0124 | 0.0248 | 0.1690 | 0.3948 |
| eeg_only | coral_eeg_only | severe_error_rate | 0.2799 | 0.2573 | -0.0226 | 0.0226 [0.0043, 0.0417] | 31/22/0 | 0.0449 | 0.0449 | 0.2717 | 0.3166 |
| eeg_only | corn_eeg_only | ordinal_mae | 1.0528 | 0.9936 | -0.0593 | 0.0593 [0.0088, 0.1109] | 38/14/1 | 0.0063 | 0.0189 | 0.0012 | 0.4354 |
| eeg_only | corn_eeg_only | severe_error_rate | 0.2799 | 0.2465 | -0.0335 | 0.0335 [0.0119, 0.0541] | 37/15/1 | 0.0011 | 0.0045 | 0.0032 | 0.5196 |
| eeg_pow | coral_eeg_pow | ordinal_mae | 0.9546 | 0.9739 | 0.0194 | -0.0194 [-0.0534, 0.0140] | 26/27/0 | 0.4544 | 1.0000 | 1.0000 | -0.1181 |
| eeg_pow | coral_eeg_pow | severe_error_rate | 0.2401 | 0.2465 | 0.0065 | -0.0065 [-0.0225, 0.0094] | 26/27/0 | 0.4334 | 1.0000 | 1.0000 | -0.1237 |
| eeg_pow | corn_eeg_pow | ordinal_mae | 0.9546 | 0.9558 | 0.0013 | -0.0013 [-0.0401, 0.0360] | 28/25/0 | 0.7533 | 1.0000 | 0.7838 | 0.0496 |
| eeg_pow | corn_eeg_pow | severe_error_rate | 0.2401 | 0.2363 | -0.0038 | 0.0038 [-0.0136, 0.0208] | 27/25/1 | 0.6165 | 1.0000 | 0.8899 | 0.0798 |

Конвенция Wilcoxon: two-sided scipy.stats.wilcoxon; zero_method='wilcox' discards exact zero differences before ranking; method='auto'; all-zero pairs are explicitly undefined.

## 7. Вторичные результаты

| Group | Candidate | Metric | Reference mean | Candidate mean | Raw Δ | Improvement [95% CI] | Better/worse/tie | Wilcoxon p | Holm p | Sign p | Rank-biserial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eeg_only | coral_eeg_only | balanced_accuracy | 0.3304 | 0.3169 | -0.0135 | -0.0135 [-0.0295, 0.0010] | 23/30/0 | 0.1659 | 1.0000 | 0.4101 | -0.2187 |
| eeg_only | coral_eeg_only | macro_f1 | 0.2889 | 0.2946 | 0.0057 | 0.0057 [-0.0077, 0.0186] | 29/24/0 | 0.2590 | 1.0000 | 0.5831 | 0.1782 |
| eeg_only | coral_eeg_only | quadratic_weighted_kappa | 0.4280 | 0.4435 | 0.0155 | 0.0155 [-0.0086, 0.0382] | 36/17/0 | 0.0819 | 1.0000 | 0.0127 | 0.2746 |
| eeg_only | coral_eeg_only | adjacent_accuracy | 0.7201 | 0.7427 | 0.0226 | 0.0226 [0.0043, 0.0417] | 31/22/0 | 0.0449 | 0.7192 | 0.2717 | 0.3166 |
| eeg_only | coral_eeg_only | expected_rank_mae | 0.9993 | 0.9879 | -0.0114 | 0.0114 [-0.0199, 0.0444] | 26/27/0 | 0.8077 | 1.0000 | 1.0000 | 0.0384 |
| eeg_only | coral_eeg_only | expected_rank_spearman | 0.5286 | 0.5079 | -0.0207 | -0.0207 [-0.0418, -0.0017] | 24/29/0 | 0.1267 | 1.0000 | 0.5831 | -0.2411 |
| eeg_only | corn_eeg_only | balanced_accuracy | 0.3304 | 0.3164 | -0.0140 | -0.0140 [-0.0277, -0.0005] | 22/31/0 | 0.0952 | 1.0000 | 0.2717 | -0.2635 |
| eeg_only | corn_eeg_only | macro_f1 | 0.2889 | 0.2816 | -0.0073 | -0.0073 [-0.0198, 0.0054] | 23/30/0 | 0.3324 | 1.0000 | 0.4101 | -0.1530 |
| eeg_only | corn_eeg_only | quadratic_weighted_kappa | 0.4280 | 0.4395 | 0.0116 | 0.0116 [-0.0129, 0.0341] | 33/20/0 | 0.1579 | 1.0000 | 0.0984 | 0.2229 |
| eeg_only | corn_eeg_only | adjacent_accuracy | 0.7201 | 0.7535 | 0.0335 | 0.0335 [0.0119, 0.0541] | 37/15/1 | 0.0011 | 0.0200 | 0.0032 | 0.5196 |
| eeg_only | corn_eeg_only | expected_rank_mae | 0.9993 | 0.9952 | -0.0041 | 0.0041 [-0.0351, 0.0415] | 28/25/0 | 0.5922 | 1.0000 | 0.7838 | 0.0846 |
| eeg_only | corn_eeg_only | expected_rank_spearman | 0.5286 | 0.5184 | -0.0102 | -0.0102 [-0.0343, 0.0114] | 27/26/0 | 0.8560 | 1.0000 | 1.0000 | -0.0287 |
| eeg_only | coral_eeg_only | balanced_accuracy | 0.3164 | 0.3169 | 0.0004 | 0.0004 [-0.0137, 0.0138] | 30/23/0 | 0.5325 | 1.0000 | 0.4101 | 0.0985 |
| eeg_only | coral_eeg_only | macro_f1 | 0.2816 | 0.2946 | 0.0130 | 0.0130 [0.0006, 0.0247] | 38/15/0 | 0.0216 | 0.3672 | 0.0022 | 0.3627 |
| eeg_only | coral_eeg_only | quadratic_weighted_kappa | 0.4395 | 0.4435 | 0.0040 | 0.0040 [-0.0199, 0.0274] | 32/21/0 | 0.5325 | 1.0000 | 0.1690 | 0.0985 |
| eeg_only | coral_eeg_only | adjacent_accuracy | 0.7535 | 0.7427 | -0.0109 | -0.0109 [-0.0317, 0.0107] | 21/29/3 | 0.2567 | 1.0000 | 0.3222 | -0.1843 |
| eeg_only | coral_eeg_only | expected_rank_mae | 0.9952 | 0.9879 | -0.0073 | 0.0073 [-0.0273, 0.0445] | 24/29/0 | 0.9965 | 1.0000 | 0.5831 | -0.0007 |
| eeg_only | coral_eeg_only | expected_rank_spearman | 0.5184 | 0.5079 | -0.0105 | -0.0105 [-0.0290, 0.0077] | 24/29/0 | 0.1857 | 1.0000 | 0.5831 | -0.2089 |
| eeg_pow | coral_eeg_pow | balanced_accuracy | 0.3514 | 0.3389 | -0.0125 | -0.0125 [-0.0286, 0.0021] | 27/26/0 | 0.2590 | 1.0000 | 1.0000 | -0.1782 |
| eeg_pow | coral_eeg_pow | macro_f1 | 0.3135 | 0.3070 | -0.0065 | -0.0065 [-0.0204, 0.0073] | 25/28/0 | 0.5153 | 1.0000 | 0.7838 | -0.1027 |
| eeg_pow | coral_eeg_pow | quadratic_weighted_kappa | 0.4670 | 0.4617 | -0.0052 | -0.0052 [-0.0287, 0.0168] | 27/26/0 | 0.9119 | 1.0000 | 1.0000 | -0.0175 |
| eeg_pow | coral_eeg_pow | adjacent_accuracy | 0.7599 | 0.7535 | -0.0065 | -0.0065 [-0.0225, 0.0094] | 26/27/0 | 0.4334 | 1.0000 | 1.0000 | -0.1237 |
| eeg_pow | coral_eeg_pow | expected_rank_mae | 0.9414 | 0.9744 | 0.0330 | -0.0330 [-0.0644, -0.0029] | 22/31/0 | 0.0987 | 1.0000 | 0.2717 | -0.2607 |
| eeg_pow | coral_eeg_pow | expected_rank_spearman | 0.5649 | 0.5422 | -0.0227 | -0.0227 [-0.0416, -0.0055] | 23/30/0 | 0.0449 | 0.7641 | 0.4101 | -0.3166 |
| eeg_pow | corn_eeg_pow | balanced_accuracy | 0.3514 | 0.3271 | -0.0244 | -0.0244 [-0.0413, -0.0091] | 23/30/0 | 0.0151 | 0.2717 | 0.4101 | -0.3836 |
| eeg_pow | corn_eeg_pow | macro_f1 | 0.3135 | 0.3029 | -0.0106 | -0.0106 [-0.0245, 0.0031] | 25/28/0 | 0.4028 | 1.0000 | 0.7838 | -0.1321 |
| eeg_pow | corn_eeg_pow | quadratic_weighted_kappa | 0.4670 | 0.4591 | -0.0079 | -0.0079 [-0.0310, 0.0148] | 27/26/0 | 0.7872 | 1.0000 | 1.0000 | -0.0426 |
| eeg_pow | corn_eeg_pow | adjacent_accuracy | 0.7599 | 0.7637 | 0.0038 | 0.0038 [-0.0136, 0.0208] | 27/25/1 | 0.6165 | 1.0000 | 0.8899 | 0.0798 |
| eeg_pow | corn_eeg_pow | expected_rank_mae | 0.9414 | 0.9637 | 0.0224 | -0.0224 [-0.0568, 0.0106] | 25/28/0 | 0.2982 | 1.0000 | 0.7838 | -0.1642 |
| eeg_pow | corn_eeg_pow | expected_rank_spearman | 0.5649 | 0.5526 | -0.0123 | -0.0123 [-0.0309, 0.0051] | 24/29/0 | 0.5922 | 1.0000 | 0.5831 | -0.0846 |
| eeg_pow | coral_eeg_pow | balanced_accuracy | 0.3271 | 0.3389 | 0.0118 | 0.0118 [-0.0019, 0.0270] | 33/20/0 | 0.0987 | 1.0000 | 0.0984 | 0.2607 |
| eeg_pow | coral_eeg_pow | macro_f1 | 0.3029 | 0.3070 | 0.0041 | 0.0041 [-0.0120, 0.0214] | 23/30/0 | 0.9894 | 1.0000 | 0.4101 | 0.0021 |
| eeg_pow | coral_eeg_pow | quadratic_weighted_kappa | 0.4591 | 0.4617 | 0.0026 | 0.0026 [-0.0177, 0.0234] | 29/24/0 | 0.5922 | 1.0000 | 0.5831 | 0.0846 |
| eeg_pow | coral_eeg_pow | adjacent_accuracy | 0.7637 | 0.7535 | -0.0102 | -0.0102 [-0.0271, 0.0080] | 22/31/0 | 0.1081 | 1.0000 | 0.2717 | -0.2537 |
| eeg_pow | coral_eeg_pow | expected_rank_mae | 0.9637 | 0.9744 | 0.0106 | -0.0106 [-0.0444, 0.0240] | 21/32/0 | 0.4491 | 1.0000 | 0.1690 | -0.1195 |
| eeg_pow | coral_eeg_pow | expected_rank_spearman | 0.5526 | 0.5422 | -0.0104 | -0.0104 [-0.0291, 0.0076] | 27/26/0 | 0.4028 | 1.0000 | 1.0000 | -0.1321 |

## 8. Влияние группы признаков

| Group | Candidate | Metric | Reference mean | Candidate mean | Raw Δ | Improvement [95% CI] | Better/worse/tie | Wilcoxon p | Holm p | Sign p | Rank-biserial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eeg_pow | categorical_eeg_pow | balanced_accuracy | 0.3304 | 0.3514 | 0.0210 | 0.0210 [0.0066, 0.0351] | 35/18/0 | 0.0091 | 0.1187 | 0.0270 | 0.4116 |
| eeg_pow | categorical_eeg_pow | macro_f1 | 0.2889 | 0.3135 | 0.0246 | 0.0246 [0.0069, 0.0423] | 33/20/0 | 0.0158 | 0.1585 | 0.0984 | 0.3809 |
| eeg_pow | categorical_eeg_pow | quadratic_weighted_kappa | 0.4280 | 0.4670 | 0.0390 | 0.0390 [0.0096, 0.0678] | 36/17/0 | 0.0099 | 0.1187 | 0.0127 | 0.4074 |
| eeg_pow | categorical_eeg_pow | ordinal_mae | 1.0528 | 0.9546 | -0.0982 | 0.0982 [0.0347, 0.1647] | 34/19/0 | 0.0104 | 0.1187 | 0.0534 | 0.4046 |
| eeg_pow | categorical_eeg_pow | severe_error_rate | 0.2799 | 0.2401 | -0.0399 | 0.0399 [0.0137, 0.0668] | 32/20/1 | 0.0080 | 0.1141 | 0.1263 | 0.4224 |
| eeg_pow | coral_eeg_pow | balanced_accuracy | 0.3169 | 0.3389 | 0.0221 | 0.0221 [0.0066, 0.0375] | 36/17/0 | 0.0175 | 0.1585 | 0.0127 | 0.3753 |
| eeg_pow | coral_eeg_pow | macro_f1 | 0.2946 | 0.3070 | 0.0124 | 0.0124 [-0.0051, 0.0294] | 32/21/0 | 0.1659 | 0.8573 | 0.1690 | 0.2187 |
| eeg_pow | coral_eeg_pow | quadratic_weighted_kappa | 0.4435 | 0.4617 | 0.0182 | 0.0182 [-0.0092, 0.0446] | 31/22/0 | 0.1043 | 0.7299 | 0.2717 | 0.2565 |
| eeg_pow | coral_eeg_pow | ordinal_mae | 0.9914 | 0.9739 | -0.0175 | 0.0175 [-0.0331, 0.0666] | 29/24/0 | 0.4282 | 1.0000 | 0.5831 | 0.1251 |
| eeg_pow | coral_eeg_pow | severe_error_rate | 0.2573 | 0.2465 | -0.0108 | 0.0108 [-0.0126, 0.0338] | 28/24/1 | 0.3970 | 1.0000 | 0.6778 | 0.1350 |
| eeg_pow | corn_eeg_pow | balanced_accuracy | 0.3164 | 0.3271 | 0.0107 | 0.0107 [-0.0041, 0.0248] | 32/21/0 | 0.0835 | 0.6680 | 0.1690 | 0.2732 |
| eeg_pow | corn_eeg_pow | macro_f1 | 0.2816 | 0.3029 | 0.0213 | 0.0213 [0.0072, 0.0358] | 35/18/0 | 0.0076 | 0.1141 | 0.0270 | 0.4214 |
| eeg_pow | corn_eeg_pow | quadratic_weighted_kappa | 0.4395 | 0.4591 | 0.0196 | 0.0196 [-0.0062, 0.0460] | 34/19/0 | 0.1429 | 0.8573 | 0.0534 | 0.2313 |
| eeg_pow | corn_eeg_pow | ordinal_mae | 0.9936 | 0.9558 | -0.0377 | 0.0377 [-0.0102, 0.0890] | 30/23/0 | 0.1659 | 0.8573 | 0.4101 | 0.2187 |
| eeg_pow | corn_eeg_pow | severe_error_rate | 0.2465 | 0.2363 | -0.0102 | 0.0102 [-0.0118, 0.0332] | 30/23/0 | 0.4816 | 1.0000 | 0.4101 | 0.1111 |

## 9. Неоднородность эффекта по испытуемым

Полные minimum/q10/q25/median/q75/q90/maximum, SD и стандартизованные парные эффекты сохранены в `paired_comparisons.parquet`; индивидуальные типы — в `subject_effect_types.parquet`.

- `eeg_only / coral / both_ordinal_degraded`: 12 subjects
- `eeg_only / coral / both_ordinal_degraded+ba_gain_ordinal_tradeoff`: 6 subjects
- `eeg_only / coral / both_ordinal_improved`: 15 subjects
- `eeg_only / coral / both_ordinal_improved+ba_tradeoff`: 13 subjects
- `eeg_only / coral / mixed_ordinal_effect+ba_gain_ordinal_tradeoff`: 2 subjects
- `eeg_only / coral / mixed_ordinal_effect+ba_tradeoff`: 5 subjects
- `eeg_only / corn / both_ordinal_degraded`: 9 subjects
- `eeg_only / corn / both_ordinal_degraded+ba_gain_ordinal_tradeoff`: 2 subjects
- `eeg_only / corn / both_ordinal_improved`: 18 subjects
- `eeg_only / corn / both_ordinal_improved+ba_tradeoff`: 16 subjects
- `eeg_only / corn / mixed_ordinal_effect+ba_gain_ordinal_tradeoff`: 2 subjects
- `eeg_only / corn / mixed_ordinal_effect+ba_tradeoff`: 5 subjects
- `eeg_only / corn / ordinal_tie`: 1 subjects
- `eeg_pow / coral / both_ordinal_degraded`: 18 subjects
- `eeg_pow / coral / both_ordinal_degraded+ba_gain_ordinal_tradeoff`: 8 subjects
- `eeg_pow / coral / both_ordinal_improved`: 18 subjects
- `eeg_pow / coral / both_ordinal_improved+ba_tradeoff`: 7 subjects
- `eeg_pow / coral / mixed_ordinal_effect+ba_gain_ordinal_tradeoff`: 1 subjects
- `eeg_pow / coral / mixed_ordinal_effect+ba_tradeoff`: 1 subjects
- `eeg_pow / corn / both_ordinal_degraded`: 17 subjects
- `eeg_pow / corn / both_ordinal_degraded+ba_gain_ordinal_tradeoff`: 4 subjects
- `eeg_pow / corn / both_ordinal_improved`: 15 subjects
- `eeg_pow / corn / both_ordinal_improved+ba_tradeoff`: 8 subjects
- `eeg_pow / corn / mixed_ordinal_effect`: 1 subjects
- `eeg_pow / corn / mixed_ordinal_effect+ba_gain_ordinal_tradeoff`: 3 subjects
- `eeg_pow / corn / mixed_ordinal_effect+ba_tradeoff`: 5 subjects

## 10. Результаты трудных испытуемых

| Group | Candidate | Baseline quartile | Subjects | Ordinal-MAE improvement | Severe-error improvement | Improved fraction |
|---|---|---|---:|---:|---:|---:|
| eeg_only | coral | worst_quartile | 14 | 0.1800 | 0.0566 | 0.7857 |
| eeg_only | coral | best_quartile | 14 | -0.0092 | -0.0039 | 0.5000 |
| eeg_only | corn | worst_quartile | 14 | 0.1689 | 0.0657 | 0.8571 |
| eeg_only | corn | best_quartile | 14 | -0.0502 | -0.0084 | 0.5000 |
| eeg_pow | coral | worst_quartile | 14 | 0.0282 | 0.0164 | 0.7143 |
| eeg_pow | coral | best_quartile | 14 | -0.0512 | -0.0212 | 0.4286 |
| eeg_pow | corn | worst_quartile | 14 | 0.0532 | 0.0168 | 0.6429 |
| eeg_pow | corn | best_quartile | 14 | -0.0606 | -0.0193 | 0.2857 |

## 11. Расстояние ошибки

Распределения расстояний 0–4 и матрицы переходов candidate-vs-categorical сохранены в `error_distance_transitions.parquet`. Последовательностные переходы описательны; статистический вывод остаётся subject-level.

| Group | Candidate | Transition | Count | Fraction |
|---|---|---|---:|---:|
| eeg_only | coral | severe_to_adjacent | 3457 | 0.0783 |
| eeg_only | coral | adjacent_to_exact | 4632 | 0.1049 |
| eeg_only | coral | exact_to_error | 5782 | 0.1310 |
| eeg_only | coral | became_more_severe | 9303 | 0.2108 |
| eeg_only | coral | net_severe_error_change | -592 | -0.0134 |
| eeg_only | corn | severe_to_adjacent | 3676 | 0.0833 |
| eeg_only | corn | adjacent_to_exact | 4501 | 0.1020 |
| eeg_only | corn | exact_to_error | 5759 | 0.1305 |
| eeg_only | corn | became_more_severe | 8864 | 0.2008 |
| eeg_only | corn | net_severe_error_change | -1176 | -0.0266 |
| eeg_pow | coral | severe_to_adjacent | 2474 | 0.0560 |
| eeg_pow | coral | adjacent_to_exact | 4280 | 0.0970 |
| eeg_pow | coral | exact_to_error | 5471 | 0.1239 |
| eeg_pow | coral | became_more_severe | 9221 | 0.2089 |
| eeg_pow | coral | net_severe_error_change | 271 | 0.0061 |
| eeg_pow | corn | severe_to_adjacent | 2794 | 0.0633 |
| eeg_pow | corn | adjacent_to_exact | 4472 | 0.1013 |
| eeg_pow | corn | exact_to_error | 5328 | 0.1207 |
| eeg_pow | corn | became_more_severe | 8243 | 0.1867 |
| eeg_pow | corn | net_severe_error_change | -552 | -0.0125 |

## 12. Результаты по классам

Recall, precision, F1, ordinal MAE, severe-error rate и распределения прогнозов для классов 0–4 сохранены в `class_error_analysis.parquet`. Ниже показаны заранее выделенные средние классы 1–3.

| Method | True class | Recall | Precision | F1 | Ordinal MAE | Severe error |
|---|---:|---:|---:|---:|---:|---:|
| categorical_eeg_only | 1 | 0.2678 | 0.3007 | 0.2833 | 1.0143 | 0.2239 |
| categorical_eeg_only | 2 | 0.2648 | 0.2578 | 0.2613 | 1.0361 | 0.3009 |
| categorical_eeg_only | 3 | 0.3166 | 0.2840 | 0.2994 | 1.0764 | 0.2566 |
| categorical_eeg_pow | 1 | 0.3102 | 0.3018 | 0.3059 | 0.9624 | 0.2216 |
| categorical_eeg_pow | 2 | 0.2843 | 0.2761 | 0.2802 | 0.9644 | 0.2487 |
| categorical_eeg_pow | 3 | 0.3255 | 0.2944 | 0.3092 | 1.0158 | 0.2413 |
| coral_eeg_only | 1 | 0.3561 | 0.2777 | 0.3120 | 0.8382 | 0.1537 |
| coral_eeg_only | 2 | 0.3011 | 0.2373 | 0.2654 | 0.9433 | 0.2444 |
| coral_eeg_only | 3 | 0.2452 | 0.3011 | 0.2703 | 1.1170 | 0.2827 |
| coral_eeg_pow | 1 | 0.2598 | 0.2940 | 0.2758 | 0.9884 | 0.1910 |
| coral_eeg_pow | 2 | 0.3241 | 0.2564 | 0.2863 | 0.9453 | 0.2695 |
| coral_eeg_pow | 3 | 0.2632 | 0.2892 | 0.2756 | 1.0786 | 0.2300 |
| corn_eeg_only | 1 | 0.4479 | 0.2902 | 0.3522 | 0.7565 | 0.1611 |
| corn_eeg_only | 2 | 0.2786 | 0.2412 | 0.2586 | 0.9143 | 0.1929 |
| corn_eeg_only | 3 | 0.2492 | 0.2816 | 0.2644 | 1.1199 | 0.2886 |
| corn_eeg_pow | 1 | 0.3359 | 0.3174 | 0.3264 | 0.9128 | 0.2035 |
| corn_eeg_pow | 2 | 0.3127 | 0.2543 | 0.2805 | 0.8985 | 0.2112 |
| corn_eeg_pow | 3 | 0.3105 | 0.2941 | 0.3020 | 0.9761 | 0.2022 |

## 13. Смещение прогнозов

- `categorical_eeg_only`: mean(y_pred−y_true)=0.0109, over=0.3310, under=0.3244, expected-rank bias=-0.0442.
- `categorical_eeg_pow`: mean(y_pred−y_true)=0.0483, over=0.3355, under=0.2981, expected-rank bias=0.0071.
- `coral_eeg_only`: mean(y_pred−y_true)=-0.0821, over=0.3082, under=0.3535, expected-rank bias=-0.0564.
- `coral_eeg_pow`: mean(y_pred−y_true)=0.0019, over=0.3263, under=0.3201, expected-rank bias=-0.0037.
- `corn_eeg_only`: mean(y_pred−y_true)=-0.0469, over=0.3272, under=0.3420, expected-rank bias=-0.0700.
- `corn_eeg_pow`: mean(y_pred−y_true)=0.0675, over=0.3413, under=0.2985, expected-rank bias=0.0391.

## 14. Threshold-rule против argmax

Основным остаётся заранее заданное `count(q >= 0.5)`. Диагностический argmax не меняет основной прогноз.

| Group | Method | Rule | Disagreements | BA | Macro F1 | QWK | Ordinal MAE | Severe error |
|---|---|---|---:|---:|---:|---:|---:|---:|
| eeg_only | coral | threshold_0.5 | 0 | 0.3386 | 0.3434 | 0.4941 | 1.0058 | 0.2649 |
| eeg_only | coral | class_probability_argmax | 8075 | 0.3441 | 0.3360 | 0.4825 | 1.0643 | 0.2968 |
| eeg_pow | coral | threshold_0.5 | 0 | 0.3541 | 0.3550 | 0.5026 | 1.0001 | 0.2575 |
| eeg_pow | coral | class_probability_argmax | 3830 | 0.3538 | 0.3467 | 0.4990 | 1.0255 | 0.2728 |
| eeg_only | corn | threshold_0.5 | 0 | 0.3307 | 0.3308 | 0.4728 | 1.0096 | 0.2517 |
| eeg_only | corn | class_probability_argmax | 7983 | 0.3414 | 0.3411 | 0.4592 | 1.0525 | 0.2704 |
| eeg_pow | corn | threshold_0.5 | 0 | 0.3605 | 0.3648 | 0.5148 | 0.9628 | 0.2388 |
| eeg_pow | corn | class_probability_argmax | 4667 | 0.3669 | 0.3684 | 0.5031 | 0.9854 | 0.2487 |

## 15. Результаты по источникам

Old_EEG и gpn_data рассчитаны описательно. Они не считаются независимыми группами, потому что часть людей присутствует в обоих источниках.

| Method | Source | Subjects* | BA | Macro F1 | QWK | Ordinal MAE | Severe error |
|---|---|---:|---:|---:|---:|---:|---:|
| categorical_eeg_only | Old_EEG | 42 | 0.3463 | 0.3478 | 0.4533 | 1.0583 | 0.2787 |
| categorical_eeg_only | gpn_data | 41 | 0.3439 | 0.3447 | 0.4582 | 1.0548 | 0.2780 |
| categorical_eeg_pow | Old_EEG | 42 | 0.3680 | 0.3694 | 0.5055 | 0.9854 | 0.2532 |
| categorical_eeg_pow | gpn_data | 41 | 0.3651 | 0.3679 | 0.5049 | 0.9826 | 0.2496 |
| coral_eeg_only | Old_EEG | 42 | 0.3425 | 0.3470 | 0.5000 | 1.0013 | 0.2638 |
| coral_eeg_only | gpn_data | 41 | 0.3348 | 0.3399 | 0.4883 | 1.0098 | 0.2660 |
| coral_eeg_pow | Old_EEG | 42 | 0.3597 | 0.3598 | 0.5158 | 0.9890 | 0.2541 |
| coral_eeg_pow | gpn_data | 41 | 0.3485 | 0.3501 | 0.4898 | 1.0101 | 0.2606 |
| corn_eeg_only | Old_EEG | 42 | 0.3365 | 0.3336 | 0.4776 | 1.0075 | 0.2507 |
| corn_eeg_only | gpn_data | 41 | 0.3250 | 0.3275 | 0.4674 | 1.0115 | 0.2527 |
| corn_eeg_pow | Old_EEG | 42 | 0.3666 | 0.3702 | 0.5260 | 0.9524 | 0.2362 |
| corn_eeg_pow | gpn_data | 41 | 0.3544 | 0.3593 | 0.5037 | 0.9721 | 0.2412 |

`*` Числа людей по источникам перекрываются и не суммируются как независимые выборки.

## 16. Стабильность между folds

Fold means/std/min/max и все candidate−categorical дельты сохранены в summary JSON и `fold_deltas.parquet`; folds не передавались в Wilcoxon.

| Method | Metric | Fold mean ± SD | Min | Max |
|---|---|---:|---:|---:|
| categorical_eeg_only | balanced_accuracy | 0.3456 ± 0.0232 | 0.3160 | 0.3764 |
| categorical_eeg_only | macro_f1 | 0.3403 ± 0.0210 | 0.3123 | 0.3702 |
| categorical_eeg_only | ordinal_mae | 1.0565 ± 0.1171 | 0.8813 | 1.1874 |
| categorical_eeg_only | quadratic_weighted_kappa | 0.4648 ± 0.0601 | 0.3886 | 0.5447 |
| categorical_eeg_only | severe_error_rate | 0.2784 ± 0.0508 | 0.2005 | 0.3461 |
| categorical_eeg_pow | balanced_accuracy | 0.3687 ± 0.0189 | 0.3342 | 0.3874 |
| categorical_eeg_pow | macro_f1 | 0.3615 ± 0.0169 | 0.3359 | 0.3856 |
| categorical_eeg_pow | ordinal_mae | 0.9838 ± 0.0784 | 0.8974 | 1.1150 |
| categorical_eeg_pow | quadratic_weighted_kappa | 0.5126 ± 0.0392 | 0.4381 | 0.5509 |
| categorical_eeg_pow | severe_error_rate | 0.2513 ± 0.0324 | 0.2086 | 0.2960 |
| coral_eeg_only | balanced_accuracy | 0.3367 ± 0.0225 | 0.3084 | 0.3710 |
| coral_eeg_only | macro_f1 | 0.3396 ± 0.0229 | 0.3105 | 0.3720 |
| coral_eeg_only | ordinal_mae | 1.0058 ± 0.0838 | 0.8766 | 1.0885 |
| coral_eeg_only | quadratic_weighted_kappa | 0.4966 ± 0.0368 | 0.4541 | 0.5634 |
| coral_eeg_only | severe_error_rate | 0.2650 ± 0.0416 | 0.2005 | 0.3115 |
| coral_eeg_pow | balanced_accuracy | 0.3560 ± 0.0199 | 0.3175 | 0.3715 |
| coral_eeg_pow | macro_f1 | 0.3504 ± 0.0185 | 0.3167 | 0.3733 |
| coral_eeg_pow | ordinal_mae | 1.0000 ± 0.0775 | 0.9233 | 1.1445 |
| coral_eeg_pow | quadratic_weighted_kappa | 0.5072 ± 0.0421 | 0.4347 | 0.5646 |
| coral_eeg_pow | severe_error_rate | 0.2574 ± 0.0323 | 0.2205 | 0.3102 |
| corn_eeg_only | balanced_accuracy | 0.3305 ± 0.0191 | 0.3057 | 0.3605 |
| corn_eeg_only | macro_f1 | 0.3269 ± 0.0165 | 0.3062 | 0.3521 |
| corn_eeg_only | ordinal_mae | 1.0095 ± 0.0820 | 0.9262 | 1.1311 |
| corn_eeg_only | quadratic_weighted_kappa | 0.4753 ± 0.0548 | 0.4118 | 0.5544 |
| corn_eeg_only | severe_error_rate | 0.2517 ± 0.0348 | 0.2148 | 0.2984 |
| corn_eeg_pow | balanced_accuracy | 0.3609 ± 0.0243 | 0.3251 | 0.3997 |
| corn_eeg_pow | macro_f1 | 0.3597 ± 0.0283 | 0.3242 | 0.4028 |
| corn_eeg_pow | ordinal_mae | 0.9626 ± 0.1012 | 0.8705 | 1.1481 |
| corn_eeg_pow | quadratic_weighted_kappa | 0.5224 ± 0.0518 | 0.4252 | 0.5741 |
| corn_eeg_pow | severe_error_rate | 0.2388 ± 0.0393 | 0.2000 | 0.3092 |

## 17. Ограничения анализа seed 42

Оценён один initial state. Парный subject-level протокол измеряет устойчивость между людьми, но не между seeds; source/fold/class analyses описательны.

## 18. Статистически допустимые выводы

По заранее заданному правилу выбрано решение 3: `continue_with_both_ordinal_heads`.

## 19. Пока недопустимые утверждения

Нельзя заявлять seed-устойчивое или причинное преимущество, считать folds/источники независимыми репликациями либо менять основное threshold decoding по результатам диагностического argmax.
