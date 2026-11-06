# Current protocol documentation audit — 2026-11-06

Tracked .md/.yaml/.yml/.json/.csv/.py; before baseline commit, after working tree plus declared new task files. Audit artifacts/script excluded to avoid self-counting. Q3 excludes Q33; counts are occurrences, not matching lines. Native external semantics and synthetic fixtures retained. No global replacement.

Starting branch: analysis/pm-temporal-predictability-audit; starting HEAD fbd100f. Documentation and metadata only: no training, feature-cache rebuild, target/fold changes, or scientific metric recomputation.

## Search totals

| Pattern | Before occurrences | After occurrences |
|---|---:|---:|
| Q3 | 1300 | 1373 |
| label_q5 | 2086 | 2203 |
| cognitive_load_5class | 1290 | 1295 |
| 3-class | 35 | 54 |
| temporal_status | 16 | 8 |
| other_status | 12 | 14 |

## Semantic classification (files, not occurrences)

| Classification | Before | After |
|---|---:|---:|
| current_correct | 61 | 78 |
| stale_needs_update | 17 | 0 |
| legacy_valid | 258 | 222 |
| historical_snapshot | 12 | 57 |
| native_three_class_keep | 11 | 11 |
| synthetic_test_keep | 35 | 35 |
| generated_from_registry | 30 | 26 |
| unclear_manual_review | 0 | 0 |

A file can contain several target contracts. The table records its dominant reviewed role; native external classes are never converted, and historical metrics retain their labels. The JSON inventory includes exact matching line numbers per pattern. Preflight manifests and immutable namespace identifiers are intentionally retained, even when an experiment later completed.

## Authoritative current and historical hierarchy

Current: [README](../../README.md), [PROJECT_CONTEXT](../../PROJECT_CONTEXT.md), [AGENTS](../../AGENTS.md), current LOW/HIGH and temporal reports, [experiment registry](../summary/experiment_registry.yaml). Execution status and scientific_status are separate: current_primary, current_supporting, historical, legacy, diagnostic, closed_negative, infrastructure_only.
Historical final snapshots retain the original 2026-08-29 date. The previous article/conclusions are preserved in separate legacy documents. Internal Q3/label_q5 reports receive banners; valid COG-BCI/CL-Drive/CLARE three-class labels and synthetic fixtures are untouched.

## Corrections and generator ownership

- Corrected future/smoke-only temporal status to 175/175 matched and 25/25 order control; 5/7 history gains and 4/5 conditional order effects; stress inconclusive.
- Corrected exact LOW <= Q33 / HIGH >= Q67 and before-history threshold fitting, canonical 371 features, nested AutoML completion and LOW/HIGH personalization.
- Rebuilt current experiment/colleague/requirements summaries from metadata. Historical execution statuses, metric sources and numerical tables are retained.
- experiment_summary separates current primary and legacy sections; colleague generator prepends curated current evidence. Requirement generator owns both summary and final coverage.
- config auditor emits scientific roles from config_curation.yaml; dedicated JSON protocols are explicitly metadata_only, not falsely resolved by the YAML auditor.
- Historical report_builder/project_final_package outputs cannot overwrite current article/conclusions/requirement coverage. No completed legacy package regeneration or result replacement was performed. Two existing tests entered the legacy generator but stopped at metadata mapping before writes.

## File-by-file after audit

| Path | Matches | Classification | Action |
|---|---|---|---|
| `AGENTS.md` | Q3: 2, label_q5: 2, 3-class: 3, other_status: 1, temporal_status: 1 | current_correct | current primary separated from retained historical evidence |
| `PROJECT_CONTEXT.md` | temporal_status: 1, label_q5: 3, Q3: 3, 3-class: 5 | current_correct | current primary separated from retained historical evidence |
| `README.md` | Q3: 24, label_q5: 2, temporal_status: 1, other_status: 1, 3-class: 1 | current_correct | current primary separated from retained historical evidence |
| `apps/streaming_worker/model_bundle.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `apps/streaming_worker/scientific.py` | Q3: 18 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `artifacts/pm_xgboost_streaming_lightweight_v1/manifest.json` | Q3: 3 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `artifacts/pm_xgboost_streaming_v1/manifest.json` | Q3: 3 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `bench/analysis/colleague_metrics_package.py` | label_q5: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/diagnostic_baselines.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/experiment_config_audit.py` | label_q5: 8, Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/interchannel_lag_diagnostics.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/label_definition_sensitivity.py` | label_q5: 15 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/label_target_audit.py` | label_q5: 13 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_eeg_lag_sweep.py` | Q3: 1, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_frozen_raw_preflight.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_low_high_confirmatory_posthoc.py` | Q3: 2, 3-class: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/analysis/pm_quality_downstream.py` | Q3: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_subject_structure.py` | Q3: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_target_validity_audit.py` | Q3: 8, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_target_validity_cli.py` | Q3: 9 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_target_validity_postprocess.py` | Q3: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_temporal_quality.py` | Q3: 16, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/pm_temporal_structure.py` | Q3: 12 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/project_final_package.py` | label_q5: 15, 3-class: 2, Q3: 3 | historical_snapshot | dated engineering/protocol inventory, not current primary claims |
| `bench/analysis/report_builder.py` | label_q5: 4, other_status: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/svg.py` | Q3: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/target_registry_audit.py` | label_q5: 40, cognitive_load_5class: 1, Q3: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/temporal_target_structure.py` | label_q5: 23 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/xgboost_feature_alignment_diagnostics.py` | 3-class: 1, Q3: 12 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/analysis/xgboost_robust_shrinkage_final_analysis.py` | Q3: 3, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/automl/personalized/bindings.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/automl/scientific/nested_extremes.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/automl/scientific/study_runner.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/data_quality/feature_outlier_audit.py` | Q3: 11 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/data_quality/logical_recording_audit.py` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/datasets/base_eeg_data_loader.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/datasets/logical_recordings.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/datasets/raw_eeg_window_cache_cli.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/datasets/raw_eeg_window_dataset.py` | label_q5: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/__init__.py` | label_q5: 21 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/artifact_removal_ablation.py` | Q3: 12, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/artifact_removal_ablation_v2.py` | Q3: 10 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/cog_bci_contrastive_transfer.py` | label_q5: 6 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `bench/experiments/cog_bci_time_aligned_transfer.py` | label_q5: 2 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `bench/experiments/cross_dataset_clare_cldrive_dann.py` | 3-class: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `bench/experiments/cross_source_generalization.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/dann_label_q5_confirmatory_protocol.py` | label_q5: 11 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/dann_label_q5_confirmatory_v2.py` | label_q5: 10 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/dann_label_q5_confirmatory_v2_protocol.py` | label_q5: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/dann_label_q5_raw_diagnostic.py` | label_q5: 8 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/dann_label_q5_raw_protocol.py` | label_q5: 14 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/feature_group_ablation.py` | label_q5: 13, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/fomaml_label_q5_diagnostic.py` | label_q5: 9 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/fomaml_label_q5_raw_diagnostic.py` | label_q5: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/fomaml_label_q5_raw_protocol.py` | label_q5: 20, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/lightgbm_feature_selection.py` | Q3: 13, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/mefar_external_xgboost.py` | 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/ordinal_transformer.py` | label_q5: 6, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/ordinal_transformer_full.py` | label_q5: 6, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/personalization_calibration.py` | Q3: 12, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/personalization_calibration_execution.py` | Q3: 11 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_confirmatory_benchmark.py` | Q3: 28 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_eeg_absolute_lag.py` | Q3: 1, temporal_status: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_eeg_lag_confirmatory.py` | Q3: 9, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_eeg_lag_regression_confirmatory.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_f3_f4_channel_drop.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_low_high_model_robustness.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_nested_automl.py` | Q3: 3 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_neural_robustness.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalization_duration_feasibility.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalization_duration_response.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalization_feasibility.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalization_long_duration_feasibility.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalization_long_duration_response.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalized_probability_calibration.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalized_threshold.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_personalized_weighted_augmentation.py` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_q3_extremes_confirmatory.py` | Q3: 2, label_q5: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_low_high_temporal_context_matched.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/pm_temporal_order_control.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/pm_temporal_predictability.py` | Q3: 1, temporal_status: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `bench/experiments/preliminary_model_zoo_comparison.py` | Q3: 27, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/preliminary_streaming_handoff.py` | Q3: 18, 3-class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/preprocessing_factorial_q3_all_pm.py` | Q3: 25, label_q5: 2, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/user_calibration.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/user_calibration_multiseed.py` | cognitive_load_5class: 1, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/xgboost_regional_montage_transfer.py` | Q3: 15, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/experiments/xgboost_robust_shrinkage_personalization.py` | Q3: 6, 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/features/cogstate_feature_cache.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/features/legacy_emotiv_eeg_features.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/meta/meta_validation.py` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/meta/synthetic.py` | 3-class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/tasks/cognitive_load.py` | cognitive_load_5class: 1, Q3: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/tasks/target_contract_builder.py` | label_q5: 4, Q3: 4, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/tasks/target_registry.py` | Q3: 3, label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `bench/tasks/tasks_registry.py` | Q3: 5, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `cli.py` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs.yaml` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `configs/groupkfold_rf_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_source_gpn_rf_transformer_label_q5.yaml` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_source_old_eeg_rf_transformer_label_q5.yaml` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_bilstm_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_preprocessed_dedup_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_raw_all_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_raw_dedup_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_lstm_gapaware_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_lstm_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_mlp_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/groupkfold_torch_transformer_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_rf_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_torch_eegnet_dedup_preprocessed_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_torch_eegnet_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_torch_lstm_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_torch_mlp_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `configs/smoke_torch_shallow_convnet_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/automl_transformer_label_q5.yaml` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/auxiliary_corn_lambda_selection_setup.yaml` | label_q5: 1, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/auxiliary_corn_nested_lambda.yaml` | label_q5: 1, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/auxiliary_corn_transformer_smoke.yaml` | label_q5: 1, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_base_smoke.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_cuda_integration_smoke.yaml` | label_q5: 4, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_cuda_smoke_base.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_full_base.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_full_subjects.yaml` | label_q5: 4, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml` | label_q5: 4, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_multiseed_20pct_base.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke.yaml` | label_q5: 3, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_multiseed_cuda_smoke_base.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/label_q5_finetuning_smoke.yaml` | label_q5: 3, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/personalization_calibration_v1.json` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/personalization_calibration_xgboost_v1.json` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/calibration/xgboost_robust_shrinkage_personalization_v1.json` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/cog_bci/contrastive_eegnet_transfer_screening.json` | label_q5: 4, cognitive_load_5class: 2 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `experiments/cog_bci/time_aligned_eegnet_transfer_screening.json` | label_q5: 2, cognitive_load_5class: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `experiments/cross_montage/xgboost_regional_montage_transfer_v1.json` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/cross_montage/xgboost_regional_montage_transfer_v2.json` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/cross_source_generalization.yaml` | label_q5: 4, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/domain_adaptation/dann_label_q5_confirmatory_protocol.json` | label_q5: 5, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/domain_adaptation/dann_label_q5_confirmatory_v2_execution.json` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/domain_adaptation/dann_label_q5_confirmatory_v2_protocol.json` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/domain_adaptation/dann_label_q5_raw_diagnostic.json` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/domain_adaptation/dann_label_q5_raw_protocol.json` | label_q5: 5, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/feature_group_rf_ablation.yaml` | label_q5: 2, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/feature_group_transformer_ablation.yaml` | label_q5: 3, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/label_definition_sensitivity.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/label_target_audit.yaml` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/episode_infrastructure_smoke.json` | cognitive_load_5class: 2, label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/fomaml_label_q5_diagnostic.json` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/fomaml_label_q5_raw_diagnostic.json` | label_q5: 10 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/fomaml_label_q5_raw_protocol.json` | label_q5: 5, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/fomaml_production_contract.json` | label_q5: 6, cognitive_load_5class: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/meta_learning/fomaml_synthetic_smoke.json` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/ordinal_transformer_full_seed42.yaml` | label_q5: 1, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/ordinal_transformer_multiseed.yaml` | label_q5: 4, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/ordinal_transformer_smoke.yaml` | label_q5: 1, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/pm_diagnostics/channel_quality_v1.json` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_model_robustness_v1.json` | Q3: 3 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_neural_robustness_v1.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_personalization_duration_response_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_personalization_feasibility_v1.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_personalization_long_duration_response_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_personalized_probability_calibration_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_personalized_threshold_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json` | Q3: 3 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_low_high_temporal_context_matched_v1.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_diagnostics/pm_temporal_predictability_v1.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `experiments/pm_quality/pm_temporal_quality_rf_final_v1.json` | label_q5: 1, cognitive_load_5class: 1, Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/pm_quality/pm_temporal_quality_v1.json` | label_q5: 1, cognitive_load_5class: 1, Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/pm_regression/pm_all_targets_feature_baseline.yaml` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/preprocessing/preprocessing_factorial_q3_all_pm_v1.json` | Q3: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/preprocessing_ablation_shallowconvnet.yaml` | cognitive_load_5class: 1, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/statistical_analysis.yaml` | label_q5: 30 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/temporal_target_audit.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `experiments/user_calibration_transformer_label_q5.yaml` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `reports/architecture/final_unification_architecture_audit.md` | Q3: 1, 3-class: 1 | historical_snapshot | dated engineering/protocol inventory, not current primary claims |
| `reports/architecture/pm_audit_final_unification_reconciliation.md` | Q3: 1 | historical_snapshot | dated engineering/protocol inventory, not current primary claims |
| `reports/article_results_summary.md` | label_q5: 6, Q3: 4, 3-class: 5 | current_correct | current primary separated from retained historical evidence |
| `reports/article_results_summary_legacy_label_q5.md` | label_q5: 1, other_status: 3 | historical_snapshot | preserved previous document; no numerical rewriting |
| `reports/automl_architecture_audit.md` | label_q5: 3, cognitive_load_5class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/automl_transformer_pilot.md` | Q3: 1, label_q5: 11, other_status: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/auxiliary_corn_lambda_selection_setup_summary.json` | cognitive_load_5class: 30 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/auxiliary_corn_nested_lambda_finalized_results.json` | cognitive_load_5class: 10 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/auxiliary_corn_transformer_experiment_design.json` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/auxiliary_corn_transformer_experiment_design.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/cross_source_data_audit.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/cross_source_data_inventory.json` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/cross_source_experiment_plan.json` | label_q5: 8, cognitive_load_5class: 8 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/cross_source_summary.json` | cognitive_load_5class: 38, label_q5: 2 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/channel_quality_v1/protocol.json` | Q3: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/channel_quality_v1/run_summary.json` | Q3: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/pm_eeg_absolute_lag_v1/README.md` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_eeg_absolute_lag_v1/protocol.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_eeg_lag_confirmatory_v1/q3_target_transforms.json` | Q3: 35 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/pm_eeg_lag_final_conclusion.md` | Q3: 2, 3-class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/diagnostics/pm_f3_f4_channel_drop_v1/run_manifest.json` | Q3: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/pm_low_high_model_robustness_v1/protocol.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_nested_automl_v1/reference_audit.csv` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_duration_response_v1/prediction_source_audit.csv` | Q3: 105 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_duration_response_v1/prediction_source_matrix.csv` | Q3: 35 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_duration_response_v1/protocol.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_long_duration_response_v1/prediction_source_audit.csv` | Q3: 105 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_long_duration_response_v1/prediction_source_matrix.csv` | Q3: 35 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalization_long_duration_response_v1/protocol.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_probability_calibration_v1/prediction_source_audit.csv` | Q3: 105 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_probability_calibration_v1/prediction_source_matrix.csv` | Q3: 35 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_probability_calibration_v1/protocol.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_threshold_v1/prediction_source_audit.csv` | Q3: 105 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_threshold_v1/prediction_source_matrix.csv` | Q3: 35 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_threshold_v1/protocol.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/reference_audit.csv` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/README.md` | Q3: 3, 3-class: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/dry_run_summary.json` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/posthoc_robustness_audit.md` | Q3: 2, 3-class: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/protocol.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_qc_channel_drop_confirmatory_v1/run_manifest.json` | Q3: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/diagnostics/pm_temporal_predictability_audit_v1/README.md` | temporal_status: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/diagnostics/pm_temporal_predictability_v1/order_control_h120/protocol.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_temporal_predictability_v1/protocol.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/diagnostics/pm_temporal_predictability_v1/smoke/protocol.json` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `reports/external_datasets/cl_drive_multimodal_protocol.md` | 3-class: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `reports/feature_group_rf_ablation.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/feature_group_rf_summary.json` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/feature_group_transformer_ablation.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/feature_group_transformer_summary.json` | label_q5: 2, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/benchmark_core_audit.md` | cognitive_load_5class: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/benchmarking_branch_audit.md` | cognitive_load_5class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/canonical_label_q5_smoke.md` | Q3: 1, label_q5: 5 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/canonical_target_contract.md` | label_q5: 3, Q3: 2, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/cog_bci_contrastive_transfer_screening.md` | label_q5: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `reports/integration/cog_bci_nback_baseline.md` | 3-class: 3 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `reports/integration/cog_bci_task_and_split_protocol.md` | label_q5: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `reports/integration/config_validation_patch.md` | 3-class: 1, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/dann_label_q5_confirmatory_protocol.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/dann_label_q5_confirmatory_v2.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/dann_label_q5_raw_diagnostic.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/dann_label_q5_raw_protocol.md` | Q3: 1, label_q5: 6 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/feature_benchmarking_mixins_audit.md` | label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/fomaml_label_q5_diagnostic.md` | Q3: 1, label_q5: 4 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/fomaml_label_q5_raw_diagnostic.md` | Q3: 1, label_q5: 6 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/fomaml_label_q5_raw_protocol.md` | Q3: 1, label_q5: 4 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/fomaml_production_contract.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/full_target_registry_audit.md` | label_q5: 8, Q3: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/meta_learning_infrastructure.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/personalization_finetuning_smoke.md` | Q3: 1, label_q5: 4 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/personalization_full_subjects.md` | Q3: 1, label_q5: 7 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/personalization_multiseed_20pct.md` | Q3: 1, label_q5: 7 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/pm_multioutput_regression.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/pm_regression_personalization_20pct.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/pm_regression_personalization_multiseed_20pct.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/project_final_state.md` | label_q5: 6 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/project_negative_results.md` | label_q5: 1, 3-class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/project_reproducibility_audit.md` | label_q5: 5 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/project_scientific_conclusions.md` | Q3: 1, label_q5: 1, 3-class: 2 | current_correct | current primary separated from retained historical evidence |
| `reports/integration/project_scientific_conclusions_legacy_20260829.md` | label_q5: 1, Q3: 1 | historical_snapshot | preserved previous document; no numerical rewriting |
| `reports/integration/shared_encoder_interface.md` | 3-class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/integration/target_pipeline_audit.md` | label_q5: 21, cognitive_load_5class: 5 | historical_snapshot | banner; original results and provenance retained |
| `reports/integration/target_pipeline_resolution.md` | label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/label_definition_sensitivity.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/label_definition_sensitivity_summary.json` | label_q5: 12, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/label_target_audit.md` | Q3: 1, label_q5: 7 | historical_snapshot | banner; original results and provenance retained |
| `reports/label_target_audit_summary.json` | label_q5: 2 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/ordinal_transformer_architecture_audit.md` | Q3: 1, label_q5: 3, cognitive_load_5class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/ordinal_transformer_experiment_design.json` | cognitive_load_5class: 1, label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/ordinal_transformer_experiment_design.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/ordinal_transformer_full_seed42_summary.json` | cognitive_load_5class: 302 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/ordinal_transformer_implementation.md` | Q3: 3, label_q5: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/ordinal_transformer_implementation_summary.json` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/ordinal_transformer_multiseed_runs.md` | Q3: 1, label_q5: 4 | historical_snapshot | banner; original results and provenance retained |
| `reports/ordinal_transformer_multiseed_runs_summary.json` | label_q5: 3, cognitive_load_5class: 600 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/ordinal_transformer_statistical_analysis_summary.json` | cognitive_load_5class: 36 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/personalization_calibration_v1_technical_report.md` | Q3: 17, label_q5: 3, 3-class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/pm_confirmatory_benchmark_plan.md` | Q3: 7, label_q5: 1, 3-class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/pm_quality/pm_temporal_quality_v1.md` | Q3: 5, label_q5: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/preprocessing_ablation_cache_validation.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/preprocessing_factorial_ablation.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/preprocessing_features_integration_20260818.md` | Q3: 2, label_q5: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/preprocessing_selected_trials_multiseed.md` | Q3: 1, label_q5: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/raw_eeg_cnn_model_comparison.md` | Q3: 1, label_q5: 8 | historical_snapshot | banner; original results and provenance retained |
| `reports/raw_eeg_logical_dedup_preprocessing_ablation.md` | Q3: 1, label_q5: 6 | historical_snapshot | banner; original results and provenance retained |
| `reports/raw_eegnet_label_q5_report.md` | Q3: 1, label_q5: 9, cognitive_load_5class: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/requirements/final_requirement_coverage.md` | label_q5: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/statistical_analysis_run_inventory.json` | label_q5: 54, cognitive_load_5class: 50 | historical_snapshot | dated engineering/protocol inventory, not current primary claims |
| `reports/statistical_analysis_run_inventory.md` | label_q5: 54, cognitive_load_5class: 50 | historical_snapshot | banner; original results and provenance retained |
| `reports/summary/colleague_metrics_summary.md` | label_q5: 8 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/config_audit.md` | label_q5: 88 | historical_snapshot | banner; original results and provenance retained |
| `reports/summary/config_curation.md` | label_q5: 194, other_status: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/summary/config_curation.yaml` | label_q5: 211, other_status: 1, Q3: 4 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/config_provenance.md` | label_q5: 12, other_status: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/summary/config_registry.yaml` | label_q5: 167, Q3: 3, other_status: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/experiment_registry.yaml` | label_q5: 131, cognitive_load_5class: 14, Q3: 8, 3-class: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/experiment_summary.md` | Q3: 4, 3-class: 1, label_q5: 26, cognitive_load_5class: 14 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/final_experiment_inventory.csv` | label_q5: 84, cognitive_load_5class: 14, Q3: 3 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_project_results.md` | label_q5: 1 | historical_snapshot | banner; original results and provenance retained |
| `reports/summary/final_result_tables/classification_results.csv` | label_q5: 14 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/dataset_characteristics.csv` | label_q5: 4 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/external_dataset_results.csv` | label_q5: 6 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/final_domain_adaptation_results.csv` | label_q5: 4 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/final_experiment_statuses.csv` | label_q5: 11 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/final_meta_learning_results.csv` | label_q5: 6 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/negative_result_summary.csv` | label_q5: 1, 3-class: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/personalization_results.csv` | label_q5: 9 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/provenance_audit.csv` | label_q5: 20 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/reproducibility_limitations.csv` | label_q5: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/final_result_tables/requirement_coverage.csv` | label_q5: 29 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/metrics_glossary.md` | label_q5: 2 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/metrics_provenance.yaml` | label_q5: 52, Q3: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/project_remaining_work.md` | Q3: 1, label_q5: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/requirements_coverage.md` | label_q5: 1 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/requirements_registry.yaml` | Q3: 3, label_q5: 21 | current_correct | current primary separated from retained historical evidence |
| `reports/summary/target_alias_audit.csv` | label_q5: 2 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_availability_by_source.csv` | label_q5: 6, Q3: 21 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_cohort_counts.csv` | label_q5: 8 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_contract/candidate_targets.csv` | Q3: 28 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_contract/derived_transform_contracts.json` | Q3: 7 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_contract/executable_targets.csv` | label_q5: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_contract/legacy_aliases.csv` | label_q5: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_contract/target_raw_compatibility.csv` | Q3: 7 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_derivation_audit.csv` | label_q5: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_inventory.csv` | label_q5: 2, Q3: 7 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_leakage_risk.csv` | label_q5: 4, Q3: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_proxy_candidates.csv` | Q3: 7 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_registry.yaml` | label_q5: 4, cognitive_load_5class: 1, Q3: 7 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/summary/target_task_coverage.csv` | label_q5: 6, Q3: 1 | generated_from_registry | legacy/current role supplied by metadata; historical tables remain unchanged |
| `reports/temporal_target_summary.json` | label_q5: 3, cognitive_load_5class: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `reports/transformer_benchmark_report.md` | Q3: 1, label_q5: 6 | historical_snapshot | banner; original results and provenance retained |
| `reports/transformer_legacy_audit.md` | Q3: 1, label_q5: 3 | historical_snapshot | banner; original results and provenance retained |
| `reports/user_calibration_architecture_audit.md` | Q3: 1, label_q5: 2 | historical_snapshot | banner; original results and provenance retained |
| `reports/user_calibration_report.md` | Q3: 1, label_q5: 4 | historical_snapshot | banner; original results and provenance retained |
| `scripts/analysis/analyze_pm_low_high_confirmatory_posthoc.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `scripts/analysis/audit_pm_temporal_acf.py` | temporal_status: 1, Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/analysis/audit_pm_temporal_history_support.py` | Q3: 2, temporal_status: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/analysis/audit_pm_temporal_mi.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/analysis/audit_pm_temporal_permutation_entropy.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/analysis/audit_pm_temporal_ssa.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/analysis/run_pm_low_high_q3_extremes_confirmatory.py` | Q3: 3 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `scripts/audit_target_registry.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/build_dann_label_q5_confirmatory_protocol.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/build_dann_label_q5_confirmatory_v2_protocol.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/build_dann_label_q5_raw_protocol.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/build_fomaml_label_q5_raw_protocol.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/run_dann_label_q5_confirmatory_v2.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/run_dann_label_q5_raw_diagnostic.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/run_fomaml_label_q5_diagnostic.py` | label_q5: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/run_fomaml_label_q5_raw_diagnostic.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `scripts/run_preprocessing_factorial_q3_all_pm.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `stew_integration_instructions.md` | label_q5: 1 | legacy_valid | historical/supporting result or provenance; not promoted to current primary |
| `tests/test_artifact_removal_ablation.py` | Q3: 6 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_artifact_removal_ablation_v2.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_automl_objective.py` | label_q5: 1, cognitive_load_5class: 2 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_automl_study_runner.py` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_automl_trial_resolver.py` | label_q5: 5, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_auxiliary_corn_artifacts.py` | cognitive_load_5class: 2 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_auxiliary_corn_nested_lambda.py` | cognitive_load_5class: 2 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_bench_runner.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_benchmark_config_validation.py` | cognitive_load_5class: 2, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_canonical_target_contract.py` | Q3: 3, label_q5: 8 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_channel_quality.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_cog_bci_contrastive_transfer.py` | label_q5: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_cog_bci_dataset.py` | label_q5: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `tests/test_cog_bci_time_aligned_transfer.py` | label_q5: 1 | native_three_class_keep | native external task labels retained; any internal label_q5 transfer endpoint remains legacy |
| `tests/test_cogstate_feature_materializer.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_config_audit.py` | label_q5: 5, cognitive_load_5class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_config_curation.py` | label_q5: 4 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_config_provenance.py` | label_q5: 17 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_cross_source_experiment.py` | label_q5: 3, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_cross_source_leakage.py` | cognitive_load_5class: 3 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_current_protocol_reporting.py` | Q3: 5, label_q5: 2, 3-class: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_dann_label_q5_confirmatory_protocol.py` | label_q5: 7 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_dann_label_q5_confirmatory_v2.py` | label_q5: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_dann_label_q5_confirmatory_v2_protocol.py` | label_q5: 6 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_dann_label_q5_raw_diagnostic.py` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_dann_label_q5_raw_protocol.py` | label_q5: 8 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_diagnostic_baselines.py` | label_q5: 6 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_emotiv_target_contract.py` | label_q5: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_experiment_summary.py` | cognitive_load_5class: 1, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_feature_group_ablation.py` | label_q5: 1, cognitive_load_5class: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_feature_group_regression.py` | label_q5: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_fold_local_q3_execution.py` | Q3: 16 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_fomaml_label_q5_diagnostic.py` | label_q5: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_fomaml_label_q5_raw_diagnostic.py` | label_q5: 6 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_fomaml_label_q5_raw_protocol.py` | label_q5: 6 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_interchannel_lag_diagnostics.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_label_definition_sensitivity.py` | label_q5: 7 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_label_target_audit.py` | label_q5: 9 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_lightgbm_feature_selection.py` | Q3: 5 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_ordinal_adapter.py` | cognitive_load_5class: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_ordinal_checkpoint_compatibility.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_ordinal_transformer_full_experiment.py` | cognitive_load_5class: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_personalization_calibration_execution.py` | Q3: 20 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_personalization_calibration_protocol.py` | Q3: 5 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_personalization_multiseed.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_pm_all_targets_feature_baseline.py` | label_q5: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_confirmatory_benchmark.py` | Q3: 6, label_q5: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_eeg_absolute_lag.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_eeg_lag_confirmatory.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_eeg_lag_regression_confirmatory.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_f3_f4_channel_drop.py` | Q3: 6 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_frozen_raw_preflight.py` | Q3: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_pm_low_high_model_robustness.py` | Q3: 3 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_low_high_nested_automl.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_low_high_neural_robustness.py` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_low_high_personalization_feasibility.py` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_low_high_q3_extremes_confirmatory.py` | Q3: 3, label_q5: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_low_high_temporal_context_matched.py` | Q3: 2 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_multioutput_regression.py` | cognitive_load_5class: 1, label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_pm_raw_shallow_smoke.py` | Q3: 3 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_temporal_predictability.py` | Q3: 1 | current_correct | q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained |
| `tests/test_pm_temporal_quality.py` | Q3: 4 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_pm_union_raw_contract.py` | label_q5: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_preliminary_model_zoo_comparison.py` | Q3: 9 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_preprocessing_factorial_q3_all_pm.py` | Q3: 6, label_q5: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_project_final_package.py` | label_q5: 11, cognitive_load_5class: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_raw_eeg_window_dataset.py` | label_q5: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_requirements_coverage.py` | label_q5: 1 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_streaming_scientific.py` | Q3: 3 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_target_registry_audit.py` | label_q5: 6 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_temporal_target_structure.py` | label_q5: 5 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_torch_eegnet.py` | label_q5: 1, cognitive_load_5class: 2 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_torch_transformer.py` | cognitive_load_5class: 2, label_q5: 2 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
| `tests/test_xgboost_regional_montage_transfer.py` | Q3: 2 | legacy_valid | historical execution/target code or compatible fixtures; no target/config change |
| `tests/test_xgboost_robust_shrinkage_final_analysis.py` | Q3: 1 | synthetic_test_keep | small test fixture or validation contract, not internal PM target |
