# Аудит конфигураций экспериментов

Найдено **69** tracked experiment YAML/YML; CLI-loadable: **65**; base: **11**, full: **19**, smoke: **16**, diagnostic: **13**, legacy: **6**, unknown: **0**. Конфигов с ошибками: **0**, с предупреждениями: **57**; exact duplicate groups: **0**, protocol duplicate groups: **4**; с experiment registry связано: **12**.

## 1. Область аудита

Обследованы tracked-конфиги: `configs.yaml`, `configs/`, `experiments/`, а также YAML/YML в `benchmark/`, `bench/` и `model_zoo/`, если они существуют. Исключены `data/`, `benchmark_results/`, `.git`, окружения, кэши и runtime-конфиги.

Каталоги верхнего уровня: `configs`, `configs.yaml`, `experiments`.

## 2. Фактические загрузчики конфигураций

- benchmark_config: cli.load_config + cli.validate_config; no base merge.
- preprocessing_ablation: load_experiment_spec; trial configs are resolved in memory.
- automl_study: load_automl_study_spec; base_config.path is loaded separately.
- calibration/personalization: specialized experiment classes load base_run/base_template separately.
- ordinal/CORN: dedicated load_*_spec functions selected through --ordinal-transformer-experiment.
- analysis-only specs: dedicated analysis loaders; no model training is needed for audit.

| loader_type | configs |
| --- | --- |
| automl_study | 1 |
| auxiliary_corn_lambda_selection_setup | 1 |
| auxiliary_corn_nested_lambda | 1 |
| auxiliary_corn_nested_lambda_finalize | 1 |
| auxiliary_corn_policy_statistics | 1 |
| auxiliary_corn_transformer_smoke | 1 |
| benchmark_config | 36 |
| completed_run_subject_statistics | 1 |
| cross_source_experiment | 1 |
| feature_group_experiment | 2 |
| label_definition_sensitivity | 1 |
| label_target_audit | 1 |
| ordinal_transformer_full | 1 |
| ordinal_transformer_multiseed | 1 |
| ordinal_transformer_multiseed_statistics | 1 |
| ordinal_transformer_smoke | 1 |
| pm_regression_personalization | 2 |
| pm_regression_personalization_multiseed | 2 |
| preprocessing_ablation | 1 |
| raw_preprocessing_fragment | 4 |
| statistical_analysis | 1 |
| temporal_target_audit | 1 |
| user_calibration | 4 |
| user_calibration_multiseed | 2 |

Единого production-наследования нет. Обычный `--config` выполняет только `yaml.safe_load` и `cli.validate_config`. AutoML, calibration и personalization загружают указанный base отдельно; preprocessing ablation строит trial-конфиги в памяти. Аудитор сохраняет эти ссылки, но не меняет их семантику.

## 3. Общая статистика

| role | count |
| --- | --- |
| root | 1 |
| base | 11 |
| full | 19 |
| smoke | 16 |
| diagnostic | 13 |
| ablation | 3 |
| legacy | 6 |
| unknown | 0 |

| status | count |
| --- | --- |
| baseline | 19 |
| diagnostic | 29 |
| final | 6 |
| invalidated | 0 |
| smoke | 15 |
| unclassified | 0 |

## 4. Конфигурационные семейства

| family | count | base | full | smoke | diagnostic/ablation | registry |
| --- | --- | --- | --- | --- | --- | --- |
| analysis | 4 | 0 | 0 | 0 | 4 | 0 |
| classification | 5 | 0 | 2 | 2 | 0 | 2 |
| classification_personalization | 11 | 5 | 2 | 3 | 0 | 1 |
| feature_groups | 1 | 0 | 0 | 0 | 1 | 0 |
| ordinal_transformer | 10 | 0 | 2 | 2 | 6 | 0 |
| pm_regression | 4 | 0 | 1 | 3 | 0 | 2 |
| pm_regression_personalization | 6 | 2 | 2 | 2 | 0 | 1 |
| preprocessing_ablation | 5 | 4 | 0 | 0 | 1 | 1 |
| raw_eeg | 12 | 0 | 6 | 3 | 2 | 2 |
| sequence_models | 11 | 0 | 4 | 1 | 2 | 3 |

Семейства используют фактические схемы своих загрузчиков. На следующем этапе допустимо выделять общие base-конфиги только внутри одного loader family.

## 5. Связь с реестром экспериментов

### Experiment registry consistency

| experiment_id | config_path | field | registry_value | config_value | severity | recommended_action |
| --- | --- | --- | --- | --- | --- | --- |
| label_q5_eegnet_raw_dedup_multiseed | configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml | seeds | [7,42,123] | [42] | warning | Document companion seed configs or the external multi-seed orchestration. |
| label_q5_shallowconvnet_raw_dedup_multiseed | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml | seeds | [7,42,123] | [42] | warning | Document companion seed configs or the external multi-seed orchestration. |
| label_q5_transformer_multiseed | configs/groupkfold_torch_transformer_label_q5.yaml | seeds | [7,42,123] | [42] | warning | Document companion seed configs or the external multi-seed orchestration. |
| shallowconvnet_preprocessing_ablation | experiments/preprocessing_ablation_shallowconvnet.yaml | seeds | [7,42,123] | [42] | warning | Document companion seed configs or the external multi-seed orchestration. |

Отсутствующие config_path из registry:

_Не обнаружено._

## 6. Базовые конфиги и наследование

| child | base | depth | exists |
| --- | --- | --- | --- |
| experiments/calibration/label_q5_finetuning_full_subjects.yaml | experiments/calibration/label_q5_finetuning_full_base.yaml | 1 | True |
| experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml | experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | 1 | True |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | 1 | True |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | 1 | True |
| experiments/calibration/label_q5_finetuning_smoke.yaml | experiments/calibration/label_q5_finetuning_base_smoke.yaml | 1 | True |
| experiments/calibration/pm_regression_personalization_20pct.yaml | experiments/calibration/pm_regression_personalization_20pct_base.yaml | 1 | True |
| experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml | experiments/calibration/pm_regression_personalization_20pct_base.yaml | 1 | True |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | 1 | True |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | experiments/calibration/pm_regression_personalization_20pct_base.yaml | 1 | True |
| experiments/automl_transformer_label_q5.yaml | configs/groupkfold_torch_transformer_label_q5.yaml | 1 | True |

Отсутствующие base-конфиги:

_Не обнаружено._

Циклы:

_Не обнаружено._

## 7. CLI-совместимость

| config | loader | CLI argument | loadable | schema |
| --- | --- | --- | --- | --- |
| experiments/label_definition_sensitivity.yaml | label_definition_sensitivity | --label-definition-sensitivity | True | True |
| experiments/label_target_audit.yaml | label_target_audit | --label-target-audit | True | True |
| experiments/statistical_analysis.yaml | statistical_analysis | --statistical-analysis | True | True |
| experiments/temporal_target_audit.yaml | temporal_target_audit | --temporal-target-audit | True | True |
| configs.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_rf_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_mlp_label_q5.yaml | benchmark_config | --config | True | True |
| configs/smoke_rf_label_q5.yaml | benchmark_config | --config | True | True |
| configs/smoke_torch_mlp_label_q5.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_full_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/label_q5_finetuning_full_subjects.yaml | user_calibration | --calibration-experiment | True | True |
| experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml | user_calibration_multiseed | --calibration-experiment | True | True |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | user_calibration | --calibration-experiment | True | True |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | user_calibration_multiseed | --calibration-experiment | True | True |
| experiments/calibration/label_q5_finetuning_smoke.yaml | user_calibration | --calibration-experiment | True | True |
| experiments/user_calibration_transformer_label_q5.yaml | user_calibration | --calibration-experiment | True | True |
| experiments/feature_group_rf_ablation.yaml | feature_group_experiment | --feature-group-experiment | True | True |
| experiments/ordinal_transformer_full_seed42.yaml | ordinal_transformer_full | --ordinal-transformer-experiment | True | True |
| experiments/ordinal_transformer_multiseed.yaml | ordinal_transformer_multiseed | --ordinal-transformer-experiment | True | True |
| experiments/auxiliary_corn_transformer_smoke.yaml | auxiliary_corn_transformer_smoke | --ordinal-transformer-experiment | True | True |
| experiments/ordinal_transformer_smoke.yaml | ordinal_transformer_smoke | --ordinal-transformer-experiment | True | True |
| experiments/auxiliary_corn_lambda_selection_setup.yaml | auxiliary_corn_lambda_selection_setup | --ordinal-transformer-experiment | True | True |
| experiments/auxiliary_corn_nested_lambda.yaml | auxiliary_corn_nested_lambda | --ordinal-transformer-experiment | True | True |
| experiments/auxiliary_corn_nested_lambda_finalize.yaml | auxiliary_corn_nested_lambda_finalize | --ordinal-transformer-experiment | True | True |
| experiments/auxiliary_corn_policy_statistics.yaml | auxiliary_corn_policy_statistics | --ordinal-transformer-analysis | True | True |
| experiments/ordinal_transformer_multiseed_statistics.yaml | ordinal_transformer_multiseed_statistics | --ordinal-transformer-analysis | True | True |
| experiments/ordinal_transformer_statistics.yaml | completed_run_subject_statistics | --ordinal-transformer-analysis | True | True |
| experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml | benchmark_config | --config | True | True |
| experiments/pm_regression/pm_regression_group_validation_smoke.yaml | benchmark_config | --config | True | True |
| experiments/pm_regression/pm_regression_robust_scaling_smoke.yaml | benchmark_config | --config | True | True |
| experiments/pm_regression/pm_regression_smoke.yaml | benchmark_config | --config | True | True |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | benchmark_config | --config | True | True |
| experiments/calibration/pm_regression_personalization_20pct.yaml | pm_regression_personalization | --calibration-experiment | True | True |
| experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml | pm_regression_personalization_multiseed | --calibration-experiment | True | True |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | pm_regression_personalization | --calibration-experiment | True | True |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | pm_regression_personalization_multiseed | --calibration-experiment | True | True |
| configs/raw_preprocessing_a_raw.yaml | raw_preprocessing_fragment | — | False | True |
| configs/raw_preprocessing_b_bandpass.yaml | raw_preprocessing_fragment | — | False | True |
| configs/raw_preprocessing_c_bandpass_notch.yaml | raw_preprocessing_fragment | — | False | True |
| configs/raw_preprocessing_d_bandpass_notch_car.yaml | raw_preprocessing_fragment | — | False | True |
| experiments/preprocessing_ablation_shallowconvnet.yaml | preprocessing_ablation | --experiment-matrix | True | True |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | benchmark_config | --config | True | True |
| configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml | benchmark_config | --config | True | True |
| configs/smoke_torch_eegnet_label_q5.yaml | benchmark_config | --config | True | True |
| configs/smoke_torch_shallow_convnet_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_eegnet_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_transformer_label_q5.yaml | benchmark_config | --config | True | True |
| experiments/cross_source_generalization.yaml | cross_source_experiment | --cross-source-experiment | True | True |
| configs/smoke_torch_lstm_label_q5.yaml | benchmark_config | --config | True | True |
| experiments/automl_transformer_label_q5.yaml | automl_study | --automl-study | True | True |
| experiments/feature_group_transformer_ablation.yaml | feature_group_experiment | --feature-group-experiment | True | True |
| configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_bilstm_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_lstm_gapaware_label_q5.yaml | benchmark_config | --config | True | True |
| configs/groupkfold_torch_lstm_label_q5.yaml | benchmark_config | --config | True | True |

## 8. Ошибки научного протокола

_Не обнаружено._

Ошибки зафиксированы для последующей ручной проверки; в рамках 10Б.1 конфиги не исправлялись.

## 9. Устаревшие и неизвестные поля

Фактические контракты используют несколько контекстных имён, которые нельзя механически переименовывать: `target_col`/`target_cols` принадлежат benchmark dataset, `target`/`targets` — специализированным specs; `n_classes`, `num_classes`, `n_outputs` относятся к разным слоям; `random_state`, `split_seed`, `model_seed`, `model_seeds` имеют различную семантику; `validation` — фактическая секция inner validation benchmark, а `feature_scaling`, `raw_preprocessing`, `preprocessing` описывают разные преобразования. `output_dir` также встречается на root, `experiment`, `analysis` и `audit` уровнях. Эти варианты не считаются взаимозаменяемыми автоматически.

_Не обнаружено._

## 10. Абсолютные пути

_Не обнаружено._

## 11. Дублирующиеся конфигурации

| group | kind | configs | recommendation |
| --- | --- | --- | --- |
| D001 | same_protocol_different_output | experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | keep separate when outputs represent distinct completed runs |
| D002 | same_protocol_different_output | experiments/calibration/label_q5_finetuning_full_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | keep separate when outputs represent distinct completed runs |
| D003 | same_protocol_different_seed | configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml, configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml, configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | replace with a seed-aware base config on the consolidation stage |
| D004 | same_protocol_different_seed | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml, configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml, configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | replace with a seed-aware base config on the consolidation stage |

## 12. Невостребованные конфигурации

Ниже перечислены конфиги, для которых не найдена точная ссылка ни в tracked-отчётах, ни в experiment registry. Это не доказательство ненужности.

| config |
| --- |
| experiments/statistical_analysis.yaml |
| configs/smoke_rf_label_q5.yaml |
| configs/smoke_torch_mlp_label_q5.yaml |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml |
| experiments/calibration/label_q5_finetuning_full_base.yaml |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml |
| experiments/user_calibration_transformer_label_q5.yaml |
| experiments/ordinal_transformer_full_seed42.yaml |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml |
| experiments/calibration/pm_regression_personalization_20pct.yaml |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml |
| configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml |
| configs/smoke_torch_shallow_convnet_label_q5.yaml |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml |
| configs/smoke_torch_lstm_label_q5.yaml |
| experiments/automl_transformer_label_q5.yaml |
| configs/groupkfold_torch_bilstm_label_q5.yaml |
| configs/groupkfold_torch_lstm_label_q5.yaml |

## 13. Legacy и invalidated

| config | role | status | superseded_by | reason |
| --- | --- | --- | --- | --- |
| configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml | legacy | baseline | — | Tracked statistical_analysis.yaml identifies this completed length-10 representation as legacy; retain as a baseline. |
| configs/groupkfold_torch_lstm_gapaware_label_q5.yaml | legacy | baseline | — | Tracked statistical_analysis.yaml identifies this completed length-10 representation as legacy; retain as a baseline. |

## 14. Рекомендуемая целевая структура

Не переносить файлы автоматически. На этапе 10Б.2 сохранить тематические каталоги `calibration/` и `pm_regression/`, а для многочисленных root-level конфигов рассмотреть минимальное разделение на `classification`, `raw_eeg`, `sequence_models`, `analysis` и `preprocessing_ablation` с подкаталогами `base`, `smoke`, `full`, только если CLI и ссылки получают совместимый alias/deprecation-период.

## 15. План безопасной консолидации

1. Зафиксировать вручную роль/status для `unknown` и `unclassified` записей.
2. Исправить только подтверждённые ошибки отдельными маленькими patches с тестами.
3. Выбрать одну duplicate group за раз и проверить resolved config hash и существующие артефакты.
4. Добавлять base-конфиги только внутри одного loader family; не вводить общий merge для специализированных specs.
5. Сначала добавить совместимые ссылки/aliases, затем обновить отчёты и experiment registry, и лишь после этого обсуждать перемещение.
6. Повторить этот аудитор и полный pytest после каждого пакета.

## 16. Что нельзя менять автоматически

Нельзя автоматически менять config_path, output_dir, порядок PM targets, split/model seeds, raw preprocessing, sequence grouping, validation strategy, calibration budgets/methods, ссылки на completed runs и конфиги со статусом final/baseline. Также нельзя удалять exact/protocol duplicates без проверки их исторических артефактов.

## 17. Ручной curation layer

Применён `config_curation.yaml`: решений **69**, reviewed **65**, needs_evidence **0**, not_applicable **4**.

| decision | count |
| --- | --- |
| keep | 16 |
| keep_as_base | 11 |
| keep_as_diagnostic | 21 |
| keep_as_legacy | 6 |
| keep_as_smoke | 15 |

Curation влияет только на отчётные role/status/decision поля; исходные experiment YAML и production loaders не изменяются.
