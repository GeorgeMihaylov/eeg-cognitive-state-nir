# Итоговое состояние проекта

Дата консолидации: 2026-08-29. Пакет построен только из существующих
артефактов; обучение, новые folds/seeds и перестроение кэшей не выполнялись.

## Масштаб

- Экспериментов и инфраструктурных этапов: **47**.
- Completed: **18**.
- Diagnostic: **4**.
- Closed negative: **7**.
- Infrastructure only: **16**.
- Полный provenance по автоматической проверке: **41/47**.

Основные таблицы находятся в `reports/summary/final_result_tables/`, рисунки —
в `reports/summary/final_result_tables/figures/`, а канонический индекс —
`reports/summary/final_experiment_inventory.csv`.

## Зафиксированные решения

1. Основной научный протокол — outer GroupKFold по `subject_id` с
   group-aware inner validation.
2. Для `label_q5` наиболее сильные feature-sequence модели находятся около
   macro F1 0.36; случайный уровень 0.20 не используется как единственный
   критерий качества.
3. Семь PM targets оцениваются отдельно и macro-агрегируются только внутри
   одной регрессионной задачи.
4. Для всех семи PM принят фиксированный контракт
   `EEG(t−10s) → PM(t)`: в continuous regression participant-macro MAE
   снизилась с 0.104731 до
   0.092238 (11.93%),
   а Pearson вырос с 0.394526 до
   0.603319; классификационное
   подтверждение независимо дало ΔMacro-F1
   +0.053003.
5. COG-BCI нативный и transfer screening завершены как diagnostic/negative
   evidence.
6. Решения `retain_14_channel_cache` и `close_transfer_track` закрывают
   расширение 62-channel cache и contrastive transfer без новой гипотезы.
7. Raw-deduplicated FOMAML diagnostic получил `do_not_proceed`: Δmacro F1
   −0.046338 против supervised full-model при одном fold, одном seed и пяти
   участниках.
8. Confirmatory DANN дал небольшой положительный participant-level эффект
   (Δmacro F1 +0.008048; Δbalanced accuracy +0.008332; Δordinal MAE −0.034008),
   но имеет статус `partially_confirmed`, не `confirmed`.

Экспериментальная работа **не объявляется полностью завершённой или
замороженной**: пакет фиксирует только текущее состояние evidence.

## Неполный provenance

| experiment_id | status | missing | evidence_role |
|---|---|---|---|
| dann_label_q5_old_eeg_to_gpn_confirmatory_v2_execution | completed | split_hash | supporting_only |
| dann_label_q5_old_eeg_to_gpn_diagnostic_v1 | diagnostic | split_hash | supporting_only |
| label_q5_auxiliary_corn_policy | closed_negative | split_hash | supporting_only |
| label_q5_random_forest_groupkfold | completed | split_hash | supporting_only |
| label_q5_torch_mlp_groupkfold | completed | split_hash | supporting_only |
| pm_robust_scaling_unwired_groupkfold | superseded | resolved_config | supporting_only |
