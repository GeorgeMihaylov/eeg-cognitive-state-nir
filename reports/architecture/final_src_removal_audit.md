# Final legacy `src/` removal audit

Дата: 2026-08-26  
Ветка: `integration/final-unification-20260826`  
Исходный HEAD этапа: `b463a345230b1f49e685dce7ca310e0f67c42ec2`

## Результат

Legacy-каталог `src/` удалён полностью. Его 19 файлов перед удалением были
проверены как thin wrappers над уже существующими package modules и CLI.
Вычислительная логика не переносилась повторно и не дублировалась.

## Mapping

| Удалённый entry point | Канонический API | Актуальный CLI |
|---|---|---|
| `src/00_inventory_data.py` | `bench.data_quality.data_inventory` | `scripts/data/inventory_data.py` |
| `src/01_inspect_emotiv_files.py` | `bench.data_quality.emotiv_file_inspection` | `scripts/data/inspect_emotiv_files.py` |
| `src/02_build_emotiv_catalog.py` | `bench.datasets.emotiv_catalog_builder` | `scripts/data/build_emotiv_catalog.py` |
| `src/03_validate_catalog_and_columns.py` | `bench.data_quality.emotiv_catalog_validation` | `scripts/data/validate_emotiv_catalog.py` |
| `src/04_build_windowed_pm_dataset.py` | `bench.datasets.emotiv_pm_window_builder` | `scripts/data/build_emotiv_pm_windows.py` |
| `src/08_build_eeg_features.py` | `bench.features.legacy_emotiv_eeg_features` | `scripts/data/build_legacy_emotiv_features.py` |
| `src/09_audit_raw_eeg.py` | `bench.data_quality.raw_eeg_audit` | `scripts/data/audit_raw_eeg.py` |
| `src/10_build_raw_eeg_window_cache.py` | `bench.datasets.raw_eeg_window_dataset` | `scripts/data/build_raw_eeg_window_cache.py` |
| `src/11_audit_logical_recordings.py` | `bench.data_quality.logical_recording_audit` | `scripts/data/audit_logical_recordings.py` |
| `src/12_audit_raw_eeg_artifacts.py` | `bench.data_quality.raw_eeg_artifact_audit` | `scripts/data/audit_raw_eeg_artifacts.py` |
| `src/13_run_preprocessing_ablation.py` | `bench.experiments.preprocessing_ablation` | `scripts/run_preprocessing_ablation.py` |
| `src/14_audit_robust_feature_scaling.py` | `bench.analysis.robust_feature_scaling_audit` | `scripts/analysis/audit_robust_feature_scaling.py` |
| `src/15_build_experiment_summary.py` | `bench.analysis.experiment_summary` | `scripts/analysis/build_experiment_summary.py` |
| `src/16_audit_experiment_configs.py` | `bench.analysis.experiment_config_audit` | `scripts/analysis/audit_experiment_configs.py` |
| `src/17_build_requirements_coverage.py` | `bench.analysis.requirements_coverage` | `scripts/analysis/build_requirements_coverage.py` |
| `src/18_build_colleague_metrics_package.py` | `bench.analysis.colleague_metrics_package` | `scripts/analysis/build_colleague_metrics_package.py` |
| `src/19_build_project_final_package.py` | `bench.analysis.project_final_package` | `scripts/analysis/build_project_final_package.py` |
| `src/20_build_pm_union_raw_cache.py` | `bench.datasets.pm_union_raw_materialization` | `scripts/data/build_pm_union_raw_cache.py` |
| `src/21_run_preliminary_streaming_handoff.py` | `bench.experiments.preliminary_streaming_handoff` | `scripts/run_preliminary_streaming_handoff.py` |

## Исправленные активные зависимости

- Analysis/config/reporting tests теперь импортируют
  `bench.analysis.experiment_summary`,
  `bench.analysis.experiment_config_audit`,
  `bench.analysis.requirements_coverage` и
  `bench.analysis.colleague_metrics_package` напрямую.
- Будущие target/PM audit reports указывают на package modules под
  `bench/datasets` и `bench/features`.
- README и актуальные команды raw-cache, preprocessing-ablation, config audit,
  summary generation и colleague package переведены на `scripts/`.
- Architecture guard требует полного отсутствия каталога `src/`, запрещает
  Python imports из `src` и проверяет наличие всех 19 заменяющих CLI.

## Намеренно сохранённые исторические упоминания

Старые пути остаются только в immutable/historical provenance:

- completed label-target audit config и его generated target registry/CSV;
- Transformer legacy reports, ссылающиеся на файлы старой research-ветки;
- исходный architecture audit, помеченный как исторический этап;
- старые target/preprocessing reports, где рядом добавлено пояснение о
  текущем canonical module/CLI.

Эти строки не импортируются, не исполняются и не являются актуальными
командами. Три старых пути в `experiments/label_target_audit.yaml` сохранены
побайтно как provenance завершённого аудита, а не runtime entry points.

## Scientific integrity

Experiment configs не менялись, включая их байтовое содержимое. Данные,
`benchmark_results`, checkpoints, predictions, protocol hashes и scientific
metrics не изменялись.

## Проверки

- Architecture boundary: `6 passed`.
- Основной targeted regression suite: `255 passed, 1 deselected`.
- Все 19 канонических CLI: `--help` завершился с кодом 0.
- Model facade: adapter, EEGNet, LSTM и ShallowConvNet являются теми же
  объектами классов, что и канонические реализации в `model_zoo`.
- `compileall` для `bench`, `cogstate`, `model_zoo`, `automl`, `apps` и
  `scripts`: успешно.
- Расширенный analysis/reporting suite: `144 passed`; оставшиеся
  `10 failed, 29 errors` требуют отсутствующие в этом worktree data/runtime
  и generated report artifacts и не связаны с удалением `src`.
- `git diff --check`: чисто; unmerged и staged files отсутствуют.
