# Аудит воспроизводимости итогового пакета

## Контракт

- Все пути в tracked-материалах относительные.
- Seeds и outer/inner protocols указаны в инвентаризации.
- Runtime predictions/checkpoints/caches остаются вне Git.
- Основные доказательства допускаются только при полном provenance.
- Метрики разных задач и уровней (`window`, `record`, `subject`) не
  агрегируются в одну величину.

## Результат

Полный provenance: **39/45**. Неполные записи
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

Известные ограничения перечислены в
`reports/summary/final_result_tables/reproducibility_limitations.csv`.
