# Финальный вывод по временному согласованию EEG и PM

Дата консолидации: 2026-08-29. Новое обучение для этого документа не
выполнялось: он объединяет завершённый exploratory sweep и два независимых
confirmatory сравнения.

## Финальное решение

Для всех семи continuous PM в будущих core-экспериментах использовать
фиксированное согласование **`EEG(t−10s) → PM(t)`** — EEG, предшествующее
timestamp PM на одно 10-секундное окно. Это экспериментальный контракт
согласования данных, а не preprocessing filter.

Не выполнять target-specific lag selection, не использовать отдельный
`Focus −20 s` и не запускать дополнительный lag search без новой заранее
утверждённой гипотезы.

## Цепочка доказательств

### 1. Exploratory sweep

Отдельный гипотезообразующий sweep сравнил `0, −10, −20, −30, −40 s`.
`−10 s` был широким лучшим кандидатом для шести из семи PM; у Focus локальный
максимум был при `−20 s` (ΔMacro-F1 +0.05255 относительно lag 0). Чтобы не
вносить target-specific post-hoc selection, до confirmatory regression был
зафиксирован единый кандидат `−10 s` для всех PM.

### 2. Независимое классификационное подтверждение

- Experiment: `pm_eeg_lag_confirmatory_371_xgboost_v1`.
- Protocol hash: `064fe752a541e753f53a1463d2749823b37c16045d559316ceaa05a0d5ab283e`.
- Execution commit: `f1c6e2dc209ba769a6dc3eb793aadd9d5aa6ece2`.
- Fold-local Q3 fit только на outer-train; 5 fixed subject-disjoint folds,
  371 признак, XGBoost seed 42; matched cohort
  30806 окон и
  54 участника.
- 35/35 fold×PM сравнений положительны по Macro-F1 и balanced accuracy;
  7/7 PM имеют благоприятный средний эффект.
- Pooled paired ΔMacro-F1
  +0.053003; Δbalanced accuracy
  +0.056273.

### 3. Continuous-regression подтверждение

- Experiment: `pm_eeg_lag_regression_confirmatory_371_xgboost_v1`.
- Protocol hash: `96b99b28533af365aa15b1a0464ce151ddbc34a51bac45645e4103acecfeb026`.
- Execution-code HEAD: `156ef5e833062caa00c87e44ae2c7fa467233927`.
- Все семь continuous PM, 5 fixed subject-disjoint folds, 371 признак,
  XGBRegressor seed 42; основная единица анализа — participant macro.
- В 35 fold×PM сравнениях MAE уменьшилась с
  0.104731 до
  0.092238: ΔMAE
  -0.012493, относительное снижение
  11.93%.
  Благоприятны 32/35 сравнений и
  7/7 PM means.
- Pearson вырос с 0.394526 до
  0.603319: ΔPearson
  +0.208793; благоприятны 35/35 сравнений
  и 7/7 PM means.

| PM | MAE reduction | delta Pearson | MAE favorable folds | Pearson favorable folds |
|---|---|---|---|---|
| Attention | 3.47% | +0.1299 | 4/5 | 5/5 |
| Engagement | 5.54% | +0.1960 | 4/5 | 5/5 |
| Excitement | 20.60% | +0.2951 | 5/5 | 5/5 |
| Stress | 11.29% | +0.2179 | 5/5 | 5/5 |
| Relaxation | 17.50% | +0.2427 | 5/5 | 5/5 |
| Interest | 6.22% | +0.1543 | 4/5 | 5/5 |
| Focus | 10.25% | +0.2257 | 5/5 | 5/5 |

## Инварианты и отсутствие leakage

- Общий fixed-fold hash:
  `2a176aad988fb814175c9edbf1e08266809f5c0a4ed4bf0dfd987ebf8c43c5dd`.
- Условия используют одинаковые target sample IDs, subject IDs, fold
  membership и train/test counts.
- Cross-subject, cross-record и cross-fold pairs: 0.
- Pairing строго record-local по точному `t_start` с шагом 10 s: первое окно
  каждого record теряется; окно после gap также исключается. Предыдущее
  доступное окно никогда не подставляется вместо отсутствующего точного
  predecessor.
- Target labels test-участников не используются для fitting или выбора lag;
  regression не выполняет target-specific lag selection.

## R2 и ограничения

R2 благоприятен в 30/35 paired
сравнений; median paired ΔR2
+0.197662, а per-PM median положителен
для 7/7 PM. Pooled arithmetic
mean R2 не используется: participant-level R2 неустойчив при малом числе
окон и почти постоянной цели. В частности, у participant `9192c107` для
некоторых PM остаются только два релевантных окна, что порождает экстремально
отрицательные значения. NaN не заменялись нулями, участники post hoc не
исключались.

Результат сильно подтверждён MAE и Pearson; R2 даёт поддерживающее, но
неоднородное и нестабильное свидетельство.

## Допустимая интерпретация

Корректная формулировка: **фиксированное причинное согласование предыдущего
окна EEG**, или **temporal alignment correction**, где EEG предшествует PM
timestamp на одно 10-секундное окно.

Нельзя заключать, что доказан физиологический лаг, что Emotiv всегда
использует ровно предыдущие 10 секунд, что известен внутренний proprietary
algorithm или что найден универсальный физиологический delay. Algorithmic
latency, proprietary aggregation, internal history и timestamp semantics
остаются только возможными объяснениями наблюдаемого dataset-level эффекта.
