# Аудит воспроизводимости итогового пакета

## Контракт

- Все пути в tracked-материалах относительные.
- Seeds и outer/inner protocols указаны в инвентаризации.
- Runtime predictions/checkpoints/caches остаются вне Git.
- Основные доказательства допускаются только при полном provenance.
- Метрики разных задач и уровней (`window`, `record`, `subject`) не
  агрегируются в одну величину.

## Результат

Полный provenance: **41/47**. Неполные записи
сохраняются в инвентаризации как supporting-only и не используются как
основное доказательство.

| experiment_id | missing | evidence_role |
|---|---|---|
| dann_label_q5_old_eeg_to_gpn_confirmatory_v2_execution | split_hash | supporting_only |
| dann_label_q5_old_eeg_to_gpn_diagnostic_v1 | split_hash | supporting_only |
| label_q5_auxiliary_corn_policy | split_hash | supporting_only |
| label_q5_random_forest_groupkfold | split_hash | supporting_only |
| label_q5_torch_mlp_groupkfold | split_hash | supporting_only |
| pm_robust_scaling_unwired_groupkfold | resolved_config | supporting_only |

## Аудит временного согласования EEG→PM

- Оба confirmatory протокола используют общий fixed-fold hash
  `2a176aad988fb814175c9edbf1e08266809f5c0a4ed4bf0dfd987ebf8c43c5dd`, 371 признаков и seed 42.
- Classification protocol hash:
  `064fe752a541e753f53a1463d2749823b37c16045d559316ceaa05a0d5ab283e`; regression protocol hash:
  `96b99b28533af365aa15b1a0464ce151ddbc34a51bac45645e4103acecfeb026`.
- Между `lag=0` и `lag=−10 s` сохранены одинаковые target sample IDs,
  участники, folds и train/test counts; cross-subject, cross-record и
  cross-fold pairs равны нулю.
- Пары строятся строго внутри logical record по точному шагу 10 s. Потеряны
  86 первых окон и
  66 окон после разрывов; разрыв
  никогда не заменяется предыдущим доступным окном.
- R2 не сворачивается в pooled arithmetic mean: используются paired median
  ΔR2 +0.197662, favorable count
  30/35 и знак per-PM median
  (7/7 положительных).

Известные ограничения перечислены в
`reports/summary/final_result_tables/reproducibility_limitations.csv`.
