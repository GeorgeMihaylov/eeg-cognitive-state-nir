# Курирование конфигураций экспериментов

## 1. Итог

Курировано **69** конфигов; ранее unclassified рассмотрено **31**. Reviewed: **65**, needs_evidence: **0**, not_applicable: **4**. Canonical family decisions: **14**; safe_to_move=true: **0**, safe_to_edit=true: **0**.

| decision | count |
| --- | --- |
| keep | 16 |
| keep_as_base | 11 |
| keep_as_diagnostic | 21 |
| keep_as_legacy | 6 |
| keep_as_smoke | 15 |

## 2. Решения по 31 unclassified конфигурации

| config | review_status | decision | role | result_status | canonical | reason |
| --- | --- | --- | --- | --- | --- | --- |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml | reviewed | keep_as_base | base | diagnostic | experiments/calibration/label_q5_finetuning_base_smoke.yaml | Base run для single-seed calibration smoke. |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | reviewed | keep_as_base | base | diagnostic | experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | Base run для CUDA integration smoke. |
| experiments/calibration/label_q5_finetuning_full_base.yaml | reviewed | keep_as_base | base | baseline | experiments/calibration/label_q5_finetuning_full_base.yaml | Global base run для full-subject single-seed personalization. |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | reviewed | keep_as_base | base | baseline | experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | Template base для финальной multiseed personalization. |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | reviewed | keep_as_base | base | diagnostic | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | Template base для multiseed CUDA smoke. |
| experiments/calibration/label_q5_finetuning_full_subjects.yaml | reviewed | keep | full | baseline | experiments/calibration/label_q5_finetuning_full_subjects.yaml | Завершённый single-seed full-subject personalization baseline. |
| experiments/user_calibration_transformer_label_q5.yaml | reviewed | keep_as_legacy | legacy | baseline | experiments/user_calibration_transformer_label_q5.yaml | Завершённый Transformer calibration protocol предшествует текущему MLP fine-tuning и нужен для воспроизводимости статистического inventory.  |
| experiments/ordinal_transformer_full_seed42.yaml | reviewed | keep | full | baseline | experiments/ordinal_transformer_full_seed42.yaml | Полный seed-42 ordinal reference, используемый multiseed analysis. |
| experiments/ordinal_transformer_multiseed.yaml | reviewed | keep | full | baseline | experiments/ordinal_transformer_multiseed.yaml | Полная multiseed ordinal Transformer matrix. |
| experiments/auxiliary_corn_lambda_selection_setup.yaml | reviewed | keep_as_diagnostic | diagnostic | diagnostic | experiments/auxiliary_corn_lambda_selection_setup.yaml | Infrastructure/setup для leakage-safe CORN lambda selection. |
| experiments/auxiliary_corn_nested_lambda.yaml | reviewed | keep_as_diagnostic | diagnostic | diagnostic | experiments/auxiliary_corn_nested_lambda.yaml | Nested CORN lambda diagnostic matrix. |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml | reviewed | keep_as_base | base | baseline | experiments/calibration/pm_regression_personalization_20pct_base.yaml | Global base/template для PM personalization. |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | reviewed | keep_as_base | base | diagnostic | experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | Global base для PM CUDA smoke. |
| experiments/calibration/pm_regression_personalization_20pct.yaml | reviewed | keep | full | baseline | experiments/calibration/pm_regression_personalization_20pct.yaml | Завершённый single-seed PM personalization baseline. |
| configs/raw_preprocessing_a_raw.yaml | not_applicable | keep_as_base | base | diagnostic | configs/raw_preprocessing_a_raw.yaml | Raw preprocessing fragment A; не standalone run. |
| configs/raw_preprocessing_b_bandpass.yaml | not_applicable | keep_as_base | base | diagnostic | configs/raw_preprocessing_b_bandpass.yaml | Band-pass preprocessing fragment B; не standalone run. |
| configs/raw_preprocessing_c_bandpass_notch.yaml | not_applicable | keep_as_base | base | diagnostic | configs/raw_preprocessing_c_bandpass_notch.yaml | Band-pass+notch fragment C; не standalone run. |
| configs/raw_preprocessing_d_bandpass_notch_car.yaml | not_applicable | keep_as_base | base | diagnostic | configs/raw_preprocessing_d_bandpass_notch_car.yaml | Full-filter fragment D; не standalone run. |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | reviewed | keep | full | baseline | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | Отдельный seed-123 EEGNet config; runner не принимает seed list. |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | reviewed | keep | full | baseline | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | Отдельный seed-7 EEGNet config; runner не принимает seed list. |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | reviewed | keep | full | final | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | Отдельный seed-123 ShallowConvNet final-family config. |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | reviewed | keep | full | final | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | Отдельный seed-7 ShallowConvNet final-family config. |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | reviewed | keep_as_diagnostic | diagnostic | diagnostic | configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | Filtered deduplicated raw-EEG preprocessing comparison. |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | reviewed | keep_as_diagnostic | diagnostic | diagnostic | configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | Raw-all logical-duplicate sensitivity comparison. |
| configs/groupkfold_torch_eegnet_label_q5.yaml | reviewed | keep_as_legacy | legacy | baseline | configs/groupkfold_torch_eegnet_label_q5.yaml | Завершённый исходный EEGNet run до logical-record dedup; сохраняется рядом с актуальным raw-deduplicated baseline.  |
| configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml | reviewed | keep_as_diagnostic | full | diagnostic | configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml | In-domain gpn_data reference для cross-source protocol. |
| configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | reviewed | keep_as_diagnostic | full | diagnostic | configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | In-domain Old_EEG reference для cross-source protocol. |
| experiments/cross_source_generalization.yaml | reviewed | keep_as_diagnostic | full | diagnostic | experiments/cross_source_generalization.yaml | Завершённая проверка по data subsets/protocol organization; не интерпретировать как cross-device final result.  |
| experiments/automl_transformer_label_q5.yaml | reviewed | keep_as_diagnostic | diagnostic | diagnostic | experiments/automl_transformer_label_q5.yaml | Nested AutoML pilot, не финальный model-selection protocol. |
| configs/groupkfold_torch_bilstm_label_q5.yaml | reviewed | keep_as_legacy | legacy | baseline | configs/groupkfold_torch_bilstm_label_q5.yaml | Завершённый исторический five-fold BiLSTM baseline без gap-aware continuity checks и явного group-record inner validation; сохраняется отдельно и не объявляется заменённым другим протоколом.  |
| configs/groupkfold_torch_lstm_label_q5.yaml | reviewed | keep_as_legacy | legacy | baseline | configs/groupkfold_torch_lstm_label_q5.yaml | Завершённый исторический five-fold LSTM baseline без gap-aware continuity checks и явного group-record inner validation; сохраняется отдельно и не объявляется заменённым другим протоколом.  |

## 3. Канонические конфиги по семействам

| family | canonical | canonical status | canonical smoke | base | legacy | reason |
| --- | --- | --- | --- | --- | --- | --- |
| analysis | experiments/statistical_analysis.yaml | completed | — | — | — | Общий статистический анализ завершённых запусков является основной точкой входа семейства; target-аудиты сохранены отдельными семействами.  |
| automl | experiments/automl_transformer_label_q5.yaml | completed | — | configs/groupkfold_torch_transformer_label_q5.yaml | — | Единственный текущий AutoML study использует канонический Transformer как отдельно загружаемый base и остаётся диагностическим pilot.  |
| classification | configs/groupkfold_rf_label_q5.yaml | completed | configs/smoke_rf_label_q5.yaml | — | — | Random Forest — канонический классический baseline label_q5; Torch MLP сохраняется как отдельный neural feature baseline.  |
| classification_personalization | experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml | completed | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | experiments/calibration/label_q5_finetuning_base_smoke.yaml, experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml, experiments/calibration/label_q5_finetuning_full_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | experiments/user_calibration_transformer_label_q5.yaml | Multiseed 20% MLP fine-tuning является текущим финальным протоколом; single-seed и Transformer calibration сохраняются для provenance.  |
| cross_source | experiments/cross_source_generalization.yaml |  | — | configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml, configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | — | Матрица cross-source и два in-domain reference-конфига образуют один специализированный диагностический протокол и не должны смешиваться с обычным GroupKFold.  |
| feature_groups | experiments/feature_group_rf_ablation.yaml |  | — | — | — | RF и Transformer ablation имеют разные model families; RF выбран семейным entry point, Transformer остаётся отдельным подсемейством.  |
| ordinal_transformer | experiments/ordinal_transformer_multiseed.yaml |  | experiments/ordinal_transformer_smoke.yaml | — | — | Multiseed matrix является наиболее полным воспроизводимым ordinal протоколом; seed-42, CORN selection и analyses остаются отдельными.  |
| pm_personalization | experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml |  | experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | experiments/calibration/pm_regression_personalization_20pct_base.yaml, experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | — | Multiseed 20% PM personalization является финальным протоколом; single-seed run остаётся baseline provenance.  |
| pm_regression | experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml | completed | experiments/pm_regression/pm_regression_smoke.yaml | — | — | Новый канонический full YAML дословно фиксирует семантический протокол завершённого пятифолдового mean/RF baseline; существующий smoke остаётся отдельной технической точкой входа.  |
| preprocessing_ablation | experiments/preprocessing_ablation_shallowconvnet.yaml |  | — | configs/raw_preprocessing_a_raw.yaml, configs/raw_preprocessing_b_bandpass.yaml, configs/raw_preprocessing_c_bandpass_notch.yaml, configs/raw_preprocessing_d_bandpass_notch_car.yaml | — | Factorial matrix является каноническим entry point; четыре raw fragments являются компонентами cache/preprocessing, а не запусками.  |
| raw_eeg | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml | completed | configs/smoke_torch_shallow_convnet_label_q5.yaml | — | configs/groupkfold_torch_eegnet_label_q5.yaml | ShallowConvNet raw-deduplicated — финальный CNN entry point; EEGNet raw-deduplicated сохраняется отдельным baseline и имеет собственные seed configs. Raw-all/filtered варианты остаются diagnostic provenance.  |
| sensitivity | experiments/label_definition_sensitivity.yaml |  | — | — | — | Единственный leakage-safe sensitivity analysis для определения label_q5.  |
| sequence_models | configs/groupkfold_torch_transformer_label_q5.yaml |  | configs/smoke_torch_lstm_label_q5.yaml | — | configs/groupkfold_torch_lstm_label_q5.yaml, configs/groupkfold_torch_bilstm_label_q5.yaml, configs/groupkfold_torch_lstm_gapaware_label_q5.yaml, configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml | Transformer length-8 является текущим финальным sequence entry point; обычные и gap-aware length-10 LSTM/BiLSTM сохраняются как отдельные завершённые historical baselines, причём gap-aware варианты являются научно более безопасными reference-конфигами.  |
| target_audit | experiments/label_target_audit.yaml |  | — | — | — | Provenance audit является основной точкой входа; temporal audit — отдельное дополнение без EEG model training.  |

## 4. Base и template configs

| config | decision | used by | evidence |
| --- | --- | --- | --- |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml | keep_as_base | referenced as base/template | experiments/calibration/label_q5_finetuning_smoke.yaml, commit:3c47ce3 |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml, commit:275fadb |
| experiments/calibration/label_q5_finetuning_full_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/label_q5_finetuning_full_subjects.yaml, reports/integration/personalization_full_subjects.md |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml, commit:255d802 |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml, commit:255d802 |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/pm_regression_personalization_20pct.yaml, experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml, commit:b1f47f4 |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | keep_as_base | referenced as base/template | experiments/calibration/pm_regression_personalization_cuda_smoke.yaml, commit:b1f47f4 |
| configs/raw_preprocessing_a_raw.yaml | keep_as_base | referenced as base/template | reports/preprocessing_ablation_implementation_audit.md, commit:50d35ac |
| configs/raw_preprocessing_b_bandpass.yaml | keep_as_base | referenced as base/template | reports/preprocessing_ablation_implementation_audit.md, commit:50d35ac |
| configs/raw_preprocessing_c_bandpass_notch.yaml | keep_as_base | referenced as base/template | reports/preprocessing_ablation_implementation_audit.md, commit:50d35ac |
| configs/raw_preprocessing_d_bandpass_notch_car.yaml | keep_as_base | referenced as base/template | reports/preprocessing_ablation_implementation_audit.md, commit:50d35ac |

## 5. Smoke и diagnostic configs

| config | decision | status |
| --- | --- | --- |
| experiments/label_definition_sensitivity.yaml | keep_as_diagnostic | diagnostic |
| experiments/label_target_audit.yaml | keep_as_diagnostic | diagnostic |
| experiments/statistical_analysis.yaml | keep_as_diagnostic | diagnostic |
| experiments/temporal_target_audit.yaml | keep_as_diagnostic | diagnostic |
| configs.yaml | keep_as_diagnostic | diagnostic |
| configs/smoke_rf_label_q5.yaml | keep_as_smoke | smoke |
| configs/smoke_torch_mlp_label_q5.yaml | keep_as_smoke | smoke |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | keep_as_smoke | smoke |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | keep_as_smoke | smoke |
| experiments/calibration/label_q5_finetuning_smoke.yaml | keep_as_smoke | smoke |
| experiments/feature_group_rf_ablation.yaml | keep_as_diagnostic | diagnostic |
| experiments/auxiliary_corn_transformer_smoke.yaml | keep_as_smoke | smoke |
| experiments/ordinal_transformer_smoke.yaml | keep_as_smoke | smoke |
| experiments/auxiliary_corn_lambda_selection_setup.yaml | keep_as_diagnostic | diagnostic |
| experiments/auxiliary_corn_nested_lambda.yaml | keep_as_diagnostic | diagnostic |
| experiments/auxiliary_corn_nested_lambda_finalize.yaml | keep_as_diagnostic | diagnostic |
| experiments/auxiliary_corn_policy_statistics.yaml | keep_as_diagnostic | diagnostic |
| experiments/ordinal_transformer_multiseed_statistics.yaml | keep_as_diagnostic | diagnostic |
| experiments/ordinal_transformer_statistics.yaml | keep_as_diagnostic | diagnostic |
| experiments/pm_regression/pm_regression_group_validation_smoke.yaml | keep_as_smoke | smoke |
| experiments/pm_regression/pm_regression_robust_scaling_smoke.yaml | keep_as_diagnostic | diagnostic |
| experiments/pm_regression/pm_regression_smoke.yaml | keep_as_smoke | smoke |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | keep_as_smoke | smoke |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | keep_as_smoke | smoke |
| experiments/preprocessing_ablation_shallowconvnet.yaml | keep_as_diagnostic | diagnostic |
| configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml | keep_as_smoke | smoke |
| configs/smoke_torch_eegnet_label_q5.yaml | keep_as_smoke | smoke |
| configs/smoke_torch_shallow_convnet_label_q5.yaml | keep_as_smoke | smoke |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | keep_as_diagnostic | diagnostic |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | keep_as_diagnostic | diagnostic |
| configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml | keep_as_diagnostic | diagnostic |
| configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | keep_as_diagnostic | diagnostic |
| experiments/cross_source_generalization.yaml | keep_as_diagnostic | diagnostic |
| configs/smoke_torch_lstm_label_q5.yaml | keep_as_smoke | smoke |
| experiments/automl_transformer_label_q5.yaml | keep_as_diagnostic | diagnostic |
| experiments/feature_group_transformer_ablation.yaml | keep_as_diagnostic | diagnostic |

## 6. Legacy configs

| config | decision | result status | reason |
| --- | --- | --- | --- |
| experiments/user_calibration_transformer_label_q5.yaml | keep_as_legacy | baseline | Завершённый Transformer calibration protocol предшествует текущему MLP fine-tuning и нужен для воспроизводимости статистического inventory.  |
| configs/groupkfold_torch_eegnet_label_q5.yaml | keep_as_legacy | baseline | Завершённый исходный EEGNet run до logical-record dedup; сохраняется рядом с актуальным raw-deduplicated baseline.  |
| configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml | keep_as_legacy | baseline | Completed length-10 gap-aware BiLSTM baseline. |
| configs/groupkfold_torch_bilstm_label_q5.yaml | keep_as_legacy | baseline | Завершённый исторический five-fold BiLSTM baseline без gap-aware continuity checks и явного group-record inner validation; сохраняется отдельно и не объявляется заменённым другим протоколом.  |
| configs/groupkfold_torch_lstm_gapaware_label_q5.yaml | keep_as_legacy | baseline | Completed length-10 gap-aware LSTM baseline. |
| configs/groupkfold_torch_lstm_label_q5.yaml | keep_as_legacy | baseline | Завершённый исторический five-fold LSTM baseline без gap-aware continuity checks и явного group-record inner validation; сохраняется отдельно и не объявляется заменённым другим протоколом.  |

## 7. Несвязанные конфиги

| config | category | decision | evidence | canonical | safe_to_move |
| --- | --- | --- | --- | --- | --- |
| experiments/statistical_analysis.yaml | diagnostic utility | keep_as_diagnostic | reports/statistical_model_comparison.md, commit:02334e6 | experiments/statistical_analysis.yaml | False |
| configs/smoke_rf_label_q5.yaml | smoke support | keep_as_smoke | cli.py, commit:50d35ac | configs/smoke_rf_label_q5.yaml | False |
| configs/smoke_torch_mlp_label_q5.yaml | smoke support | keep_as_smoke | tests/test_bench_runner.py, commit:50d35ac | configs/smoke_torch_mlp_label_q5.yaml | False |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml | base dependency | keep_as_base | experiments/calibration/label_q5_finetuning_smoke.yaml, commit:3c47ce3 | experiments/calibration/label_q5_finetuning_base_smoke.yaml | False |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | base dependency | keep_as_base | experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml, commit:275fadb | experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | False |
| experiments/calibration/label_q5_finetuning_full_base.yaml | base dependency | keep_as_base | experiments/calibration/label_q5_finetuning_full_subjects.yaml, reports/integration/personalization_full_subjects.md | experiments/calibration/label_q5_finetuning_full_base.yaml | False |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | base dependency | keep_as_base | experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml, commit:255d802 | experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | False |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | base dependency | keep_as_base | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml, commit:255d802 | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | False |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | smoke support | keep_as_smoke | tests/test_personalization_finetuning.py, commit:275fadb | experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | False |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | smoke support | keep_as_smoke | tests/test_personalization_multiseed.py, commit:255d802 | experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | False |
| experiments/user_calibration_transformer_label_q5.yaml | legacy reproducibility | keep_as_legacy | reports/user_calibration_report.md, experiments/statistical_analysis.yaml, commit:43ee533 | experiments/user_calibration_transformer_label_q5.yaml | False |
| experiments/ordinal_transformer_full_seed42.yaml | active but undocumented | keep | reports/ordinal_transformer_full_seed42.md, reports/ordinal_transformer_statistical_analysis.md, commit:ad3365a | experiments/ordinal_transformer_full_seed42.yaml | False |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml | base dependency | keep_as_base | experiments/calibration/pm_regression_personalization_20pct.yaml, experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml, commit:b1f47f4 | experiments/calibration/pm_regression_personalization_20pct_base.yaml | False |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | base dependency | keep_as_base | experiments/calibration/pm_regression_personalization_cuda_smoke.yaml, commit:b1f47f4 | experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | False |
| experiments/calibration/pm_regression_personalization_20pct.yaml | active but undocumented | keep | reports/integration/pm_regression_personalization_20pct.md, commit:b1f47f4 | experiments/calibration/pm_regression_personalization_20pct.yaml | False |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | smoke support | keep_as_smoke | tests/test_pm_regression_personalization.py, commit:b1f47f4 | experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | False |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | smoke support | keep_as_smoke | tests/test_pm_regression_personalization_multiseed.py, commit:6c0b77f | experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | False |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | active but undocumented | keep | reports/raw_eeg_cnn_model_comparison.md, experiments/statistical_analysis.yaml, commit:50d35ac | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | False |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | active but undocumented | keep | reports/raw_eeg_cnn_model_comparison.md, experiments/statistical_analysis.yaml, commit:50d35ac | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | False |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | active but undocumented | keep | reports/raw_eeg_cnn_model_comparison.md, experiments/statistical_analysis.yaml, commit:50d35ac | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | False |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | active but undocumented | keep | reports/raw_eeg_cnn_model_comparison.md, experiments/statistical_analysis.yaml, commit:50d35ac | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | False |
| configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml | smoke support | keep_as_smoke | tests/test_torch_eegnet.py, commit:50d35ac | configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml | False |
| configs/smoke_torch_shallow_convnet_label_q5.yaml | smoke support | keep_as_smoke | tests/test_torch_shallow_convnet.py, commit:50d35ac | configs/smoke_torch_shallow_convnet_label_q5.yaml | False |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | diagnostic utility | keep_as_diagnostic | reports/raw_eeg_logical_dedup_preprocessing_ablation.md, commit:50d35ac | configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | False |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | diagnostic utility | keep_as_diagnostic | reports/raw_eeg_logical_dedup_preprocessing_ablation.md, commit:50d35ac | configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | False |
| configs/smoke_torch_lstm_label_q5.yaml | smoke support | keep_as_smoke | tests/test_torch_lstm.py, commit:50d35ac | configs/smoke_torch_lstm_label_q5.yaml | False |
| experiments/automl_transformer_label_q5.yaml | diagnostic utility | keep_as_diagnostic | reports/automl_transformer_pilot.md, commit:54c98fc | experiments/automl_transformer_label_q5.yaml | False |
| configs/groupkfold_torch_bilstm_label_q5.yaml | historical runtime provenance | keep_as_legacy | commit:50d35ac | configs/groupkfold_torch_bilstm_label_q5.yaml | False |
| configs/groupkfold_torch_lstm_label_q5.yaml | historical runtime provenance | keep_as_legacy | commit:50d35ac | configs/groupkfold_torch_lstm_label_q5.yaml | False |

## 8. Scientific protocol duplicate groups

| group | classification | configs | keep separate | canonical template | relationship | future candidate |
| --- | --- | --- | --- | --- | --- | --- |
| classification_personalization_smoke_base | same_protocol_different_output | experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | True | — | Single-seed и multiseed CUDA smoke bases. | True |
| classification_personalization_full_base | same_protocol_different_output | experiments/calibration/label_q5_finetuning_full_base.yaml, experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | True | — | Single-seed global base и multiseed base template. | True |
| eegnet_raw_multiseed | same_protocol_different_seed | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml, configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml, configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | True | configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml | Один протокол, три model/split seeds. | True |
| shallowconvnet_raw_multiseed | same_protocol_different_seed | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml, configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml, configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | True | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml | Один протокол, три model/split seeds. | True |

## 9. Seed provenance

| model/experiment | registry seeds | primary config seeds | siblings | external orchestration | status | recommended metadata fix |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | [7,42,123] | [42] | — | Seeds 7/123 запускались тем же benchmark config с runtime seed overrides; отдельные tracked source YAML не создавались.  | documented | Registry теперь явно хранит runtime orchestration для всех трёх seeds; source YAML не меняется.  |
| EEGNet | [7,42,123] | [42] | configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml, configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | Отдельные tracked sibling configs. | documented | В registry перечислить primary и sibling config paths как provenance.  |
| ShallowConvNet | [7,42,123] | [42] | configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml, configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | Отдельные tracked sibling configs. | documented | В registry перечислить primary и sibling config paths как provenance.  |
| preprocessing_ablation | [7,42,123] | [42] | — | Seeds 7/123 передавались через CLI --experiment-matrix --seed поверх fixed matrix; seed 42 взят из исходной matrix и legacy full runs.  | documented | Registry теперь хранит external seed orchestration; fixed split seed исходной matrix не меняется.  |

## 10. Защищённые конфиги

| config | protected fields |
| --- | --- |
| experiments/label_definition_sensitivity.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/label_target_audit.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/statistical_analysis.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/temporal_target_audit.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_rf_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_mlp_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_rf_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_torch_mlp_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_base_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_full_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_full_subjects.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/label_q5_finetuning_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/user_calibration_transformer_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/feature_group_rf_ablation.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/ordinal_transformer_full_seed42.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/ordinal_transformer_multiseed.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/auxiliary_corn_transformer_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/ordinal_transformer_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/auxiliary_corn_lambda_selection_setup.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/auxiliary_corn_nested_lambda.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/auxiliary_corn_nested_lambda_finalize.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/auxiliary_corn_policy_statistics.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/ordinal_transformer_multiseed_statistics.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/ordinal_transformer_statistics.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/pm_regression/pm_regression_group_validation_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/pm_regression/pm_regression_robust_scaling_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/pm_regression/pm_regression_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_20pct_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_cuda_smoke_base.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_20pct.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_multiseed_20pct.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_cuda_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/calibration/pm_regression_personalization_multiseed_cuda_smoke.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/raw_preprocessing_a_raw.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/raw_preprocessing_b_bandpass.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/raw_preprocessing_c_bandpass_notch.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/raw_preprocessing_d_bandpass_notch_car.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/preprocessing_ablation_shallowconvnet.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_torch_eegnet_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_torch_shallow_convnet_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_eegnet_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_transformer_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/cross_source_generalization.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/smoke_torch_lstm_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/automl_transformer_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| experiments/feature_group_transformer_ablation.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_bilstm_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_lstm_gapaware_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |
| configs/groupkfold_torch_lstm_label_q5.yaml | config_path, output_dir, dataset, task, target, targets, preprocessing, evaluation, validation, split_seed, model_seed, model_seeds |

## 11. Конфиги с недостаточным evidence

_Не обнаружено._

## 12. Кандидаты на минимальную нормализацию

Кандидатами считаются только metadata/registry изменения и четыре явно описанные protocol groups. Source YAML не меняются на этом этапе.

## 13. Конфиги, которые нельзя менять автоматически

Все перечисленные конфиги имеют `safe_to_move=false` и `safe_to_edit=false`, если отдельное решение явно не говорит обратного. Это сохраняет ссылки CLI, reports, base/template и completed-run provenance.

## 14. План 10Б.2Б

### Пакет 1. Метаданные

- Затрагиваемые файлы: reports/summary/config_curation.yaml, reports/summary/config_registry.yaml, reports/summary/experiment_registry.yaml
- Риск: Низкий, если изменяется только отчётный metadata layer.
- Тесты: tests/test_config_curation.py, tests/test_config_audit.py
- Dry-load: scripts/analysis/audit_experiment_configs.py --strict
- Rollback: Удалить только metadata patch; source experiment YAML не затрагиваются.

### Пакет 2. Base/template consistency

- Затрагиваемые файлы: experiments/calibration/
- Риск: Средний из-за config_path references и completed-run hashes.
- Тесты: tests/test_user_finetuning.py, tests/test_personalization_multiseed.py, tests/test_pm_regression_personalization_multiseed.py
- Dry-load: --calibration-experiment ... --plan-only
- Rollback: Откатить один loader-family patch и восстановить прежние paths byte-for-byte.

### Пакет 3. Smoke naming and output isolation

- Затрагиваемые файлы: configs/smoke_*.yaml, experiments/calibration/*smoke*.yaml
- Риск: Средний; пути могут использоваться документацией и smoke automation.
- Тесты: tests/test_bench_runner.py, tests/test_config_audit.py
- Dry-load: cli.py --config <smoke-config> без execute
- Rollback: Вернуть имена и output_dir каждого smoke-конфига отдельно.

### Пакет 4. Registry/report links

- Затрагиваемые файлы: reports/summary/experiment_registry.yaml, reports/**/*.md
- Риск: Низкий для runtime, средний для provenance.
- Тесты: tests/test_experiment_summary.py, tests/test_config_curation.py
- Dry-load: scripts/analysis/build_experiment_summary.py --strict, scripts/analysis/audit_experiment_configs.py --strict
- Rollback: Откатить только ссылки report/registry; experiment configs не менять.

### Пакет 5. Физическое перемещение

- Затрагиваемые файлы: определить после review
- Риск: Высокий; в текущем curation safe_to_move=true отсутствует.
- Тесты: полный pytest, поиск всех config_path references
- Dry-load: каждый фактический loader type
- Rollback: Не выполнять пакет без отдельной migration map и совместимых aliases.
