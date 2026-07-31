# Итоговое состояние проекта

Дата консолидации: 2026-07-29. Пакет построен только из существующих
артефактов; обучение, новые folds/seeds и перестроение кэшей не выполнялись.

## Масштаб

- Экспериментов и инфраструктурных этапов: **33**.
- Completed: **14**.
- Diagnostic: **3**.
- Closed negative: **6**.
- Infrastructure only: **9**.
- Полный provenance по автоматической проверке: **29/33**.

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
4. COG-BCI нативный и transfer screening завершены как diagnostic/negative
   evidence.
5. Решения `retain_14_channel_cache` и `close_transfer_track` закрывают
   расширение 62-channel cache и contrastive transfer без новой гипотезы.

## Неполный provenance

| experiment_id | status | missing | evidence_role |
|---|---|---|---|
| label_q5_auxiliary_corn_policy | closed_negative | split_hash | supporting_only |
| label_q5_random_forest_groupkfold | completed | split_hash | supporting_only |
| label_q5_torch_mlp_groupkfold | completed | split_hash | supporting_only |
| pm_robust_scaling_unwired_groupkfold | superseded | resolved_config | supporting_only |
