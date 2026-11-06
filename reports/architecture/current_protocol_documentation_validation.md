# Documentation reconciliation: validation and handoff

Branch: `analysis/pm-temporal-predictability-audit`. Starting HEAD: `fbd100fbd20e9e2e5e132547b03bf12a7a942e36`. The final audit commit follows the three commits below; `git log` is authoritative for its hash.

- `a894c41` — docs: align canonical project context with low-high protocol
- `6e898b1` — docs: mark historical q3 and label-q5 reports
- `840d6e9` — docs: refresh experiment registries and project summaries

## Validation

- Targeted reporting suite: **147 passed, 2 deselected**. The two deselected legacy tests rewrite a complete historical package in the repository; no completed historical regeneration was performed. Five current reporting/audit tests additionally passed, including target-role separation and protection of current documents from the historical renderer.
- First test pass: 144 passed, four failures from the new not_applicable metadata status and an old expectation that the label_q5 criterion defines current acceptance. The mapper/test were corrected. Both attempted legacy generations failed before writes. No model training occurred.
- Relative Markdown links across changed documents: **280 checked, 0 missing**, including this audit appendix.
- Historical body preservation: **51 documents** retain their exact original body after the banner. Previous article and conclusions are archived separately.
- Six generated current documents/metadata are byte-identical after a second generation. Config registry matches the safe auditor output; 79 YAML configs, zero errors, 67 metadata warnings retained; five dedicated JSON protocols are metadata_only. Existing resolved config hashes unchanged.
- `git diff --check`: passed. Git CRLF notices are line-ending normalization notices, not scientific warnings.
- Scientific CSV/JSON results, native external configs, target definitions, folds, caches and experiment code are unchanged. The pre-existing modified nested AutoML protocol.json was not touched or staged.
- No training, AutoML, order-control rerun, Transformer run, feature-cache rebuild or new inferential analysis was executed. Reporting only consumes existing artifacts.

## Safe reporting generators executed

```text
python scripts/analysis/build_experiment_summary.py --strict
python scripts/analysis/build_colleague_metrics_package.py --experiment-registry reports/summary/experiment_registry.yaml --config-registry reports/summary/config_registry.yaml --output-dir reports/summary --strict
python scripts/analysis/build_requirements_coverage.py --strict
python scripts/analysis/audit_experiment_configs.py --root . --output-dir benchmark_results/documentation_audit/config_regenerated --curation reports/summary/config_curation.yaml --strict
python scripts/analysis/audit_current_protocol_documentation.py
```

Existing optional mixin runtime folders and unresolved legacy label_q5 baselines remain recorded as warnings/missing evidence. They were not fabricated or filled by new training. The old 0.75 label_q5 acceptance number and its observed result remain in provenance, with not_applicable for current binary acceptance.

## Authority and deliberate retention

Authoritative current: README, PROJECT_CONTEXT, AGENTS, current LOW/HIGH/temporal experiment reports and experiment registry. Historical: 2026-08-29 final states, archived article/conclusions, Q3/label_q5 benchmark/calibration/preprocessing reports. Native COG-BCI N-Back/MATB and CL-Drive/CLARE three-class labels and synthetic fixtures retain their semantics. Immutable q3_extremes namespaces are current binary tertile-extremes protocols, not obsolete three-class labels.
Counts increased for some historical terms because explicit banners and archived copies now document their scope. Counts are occurrences; semantic classifications count files. The [full audit](current_protocol_documentation_audit.md) and [machine-readable inventory](current_protocol_documentation_audit.json) contain all paths and line references.

## Scientific sanity check

| Check | Result |
|---|---|
| README current primary = seven-PM LOW/HIGH | YES |
| Internal Q3 explicitly historical | YES |
| label_q5 explicitly historical | YES |
| Temporal matched complete 175/175 | YES |
| H120 BH gain in 5/7 PM | YES |
| Order control complete 25/25 | YES |
| Order effect in 4/5 gated PM | YES |
| Stress order result inconclusive | YES |
| Nested AutoML not only pilot | YES |
| Current canonical 371 includes entropy/connectivity | YES |
| No physiological causality claim | YES |
| Native COG-BCI / external three-class semantics preserved | YES |
| Future bounded sequence experiment planned, not completed | YES |

## Full changed tracked file list (86 including this audit package)

- [AGENTS.md](../../AGENTS.md)
- [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md)
- [README.md](../../README.md)
- [bench/analysis/colleague_metrics_package.py](../../bench/analysis/colleague_metrics_package.py)
- [bench/analysis/experiment_config_audit.py](../../bench/analysis/experiment_config_audit.py)
- [bench/analysis/experiment_summary.py](../../bench/analysis/experiment_summary.py)
- [bench/analysis/project_final_package.py](../../bench/analysis/project_final_package.py)
- [bench/analysis/report_builder.py](../../bench/analysis/report_builder.py)
- [bench/analysis/requirements_coverage.py](../../bench/analysis/requirements_coverage.py)
- [reports/architecture/current_protocol_documentation_audit.json](../../reports/architecture/current_protocol_documentation_audit.json)
- [reports/architecture/current_protocol_documentation_audit.md](../../reports/architecture/current_protocol_documentation_audit.md)
- [reports/architecture/current_protocol_documentation_validation.md](../../reports/architecture/current_protocol_documentation_validation.md)
- [reports/article_results_summary.md](../../reports/article_results_summary.md)
- [reports/article_results_summary_legacy_label_q5.md](../../reports/article_results_summary_legacy_label_q5.md)
- [reports/automl_architecture_audit.md](../../reports/automl_architecture_audit.md)
- [reports/automl_transformer_pilot.md](../../reports/automl_transformer_pilot.md)
- [reports/auxiliary_corn_transformer_experiment_design.md](../../reports/auxiliary_corn_transformer_experiment_design.md)
- [reports/diagnostics/pm_eeg_lag_final_conclusion.md](../../reports/diagnostics/pm_eeg_lag_final_conclusion.md)
- [reports/diagnostics/pm_low_high_nested_automl_v1/README.md](../../reports/diagnostics/pm_low_high_nested_automl_v1/README.md)
- [reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/README.md](../../reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/README.md)
- [reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/README.md](../../reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/README.md)
- [reports/diagnostics/pm_temporal_predictability_audit_v1/README.md](../../reports/diagnostics/pm_temporal_predictability_audit_v1/README.md)
- [reports/diagnostics/pm_temporal_predictability_v1/README.md](../../reports/diagnostics/pm_temporal_predictability_v1/README.md)
- [reports/feature_group_rf_ablation.md](../../reports/feature_group_rf_ablation.md)
- [reports/feature_group_transformer_ablation.md](../../reports/feature_group_transformer_ablation.md)
- [reports/integration/benchmark_core_audit.md](../../reports/integration/benchmark_core_audit.md)
- [reports/integration/benchmarking_branch_audit.md](../../reports/integration/benchmarking_branch_audit.md)
- [reports/integration/canonical_label_q5_smoke.md](../../reports/integration/canonical_label_q5_smoke.md)
- [reports/integration/dann_label_q5_confirmatory_protocol.md](../../reports/integration/dann_label_q5_confirmatory_protocol.md)
- [reports/integration/dann_label_q5_confirmatory_v2.md](../../reports/integration/dann_label_q5_confirmatory_v2.md)
- [reports/integration/dann_label_q5_raw_diagnostic.md](../../reports/integration/dann_label_q5_raw_diagnostic.md)
- [reports/integration/dann_label_q5_raw_protocol.md](../../reports/integration/dann_label_q5_raw_protocol.md)
- [reports/integration/feature_benchmarking_mixins_audit.md](../../reports/integration/feature_benchmarking_mixins_audit.md)
- [reports/integration/fomaml_label_q5_diagnostic.md](../../reports/integration/fomaml_label_q5_diagnostic.md)
- [reports/integration/fomaml_label_q5_raw_diagnostic.md](../../reports/integration/fomaml_label_q5_raw_diagnostic.md)
- [reports/integration/fomaml_label_q5_raw_protocol.md](../../reports/integration/fomaml_label_q5_raw_protocol.md)
- [reports/integration/full_target_registry_audit.md](../../reports/integration/full_target_registry_audit.md)
- [reports/integration/personalization_finetuning_smoke.md](../../reports/integration/personalization_finetuning_smoke.md)
- [reports/integration/personalization_full_subjects.md](../../reports/integration/personalization_full_subjects.md)
- [reports/integration/personalization_multiseed_20pct.md](../../reports/integration/personalization_multiseed_20pct.md)
- [reports/integration/project_final_state.md](../../reports/integration/project_final_state.md)
- [reports/integration/project_negative_results.md](../../reports/integration/project_negative_results.md)
- [reports/integration/project_reproducibility_audit.md](../../reports/integration/project_reproducibility_audit.md)
- [reports/integration/project_scientific_conclusions.md](../../reports/integration/project_scientific_conclusions.md)
- [reports/integration/project_scientific_conclusions_legacy_20260829.md](../../reports/integration/project_scientific_conclusions_legacy_20260829.md)
- [reports/integration/target_pipeline_audit.md](../../reports/integration/target_pipeline_audit.md)
- [reports/integration/target_pipeline_resolution.md](../../reports/integration/target_pipeline_resolution.md)
- [reports/label_definition_sensitivity.md](../../reports/label_definition_sensitivity.md)
- [reports/label_target_audit.md](../../reports/label_target_audit.md)
- [reports/ordinal_transformer_architecture_audit.md](../../reports/ordinal_transformer_architecture_audit.md)
- [reports/ordinal_transformer_experiment_design.md](../../reports/ordinal_transformer_experiment_design.md)
- [reports/ordinal_transformer_implementation.md](../../reports/ordinal_transformer_implementation.md)
- [reports/ordinal_transformer_multiseed_runs.md](../../reports/ordinal_transformer_multiseed_runs.md)
- [reports/personalization_calibration_v1_technical_report.md](../../reports/personalization_calibration_v1_technical_report.md)
- [reports/pm_confirmatory_benchmark_plan.md](../../reports/pm_confirmatory_benchmark_plan.md)
- [reports/pm_quality/pm_temporal_quality_v1.md](../../reports/pm_quality/pm_temporal_quality_v1.md)
- [reports/preprocessing_ablation_cache_validation.md](../../reports/preprocessing_ablation_cache_validation.md)
- [reports/preprocessing_factorial_ablation.md](../../reports/preprocessing_factorial_ablation.md)
- [reports/preprocessing_features_integration_20260818.md](../../reports/preprocessing_features_integration_20260818.md)
- [reports/preprocessing_selected_trials_multiseed.md](../../reports/preprocessing_selected_trials_multiseed.md)
- [reports/raw_eeg_cnn_model_comparison.md](../../reports/raw_eeg_cnn_model_comparison.md)
- [reports/raw_eeg_logical_dedup_preprocessing_ablation.md](../../reports/raw_eeg_logical_dedup_preprocessing_ablation.md)
- [reports/raw_eegnet_label_q5_report.md](../../reports/raw_eegnet_label_q5_report.md)
- [reports/requirements/final_requirement_coverage.md](../../reports/requirements/final_requirement_coverage.md)
- [reports/statistical_analysis_run_inventory.md](../../reports/statistical_analysis_run_inventory.md)
- [reports/summary/colleague_metrics_summary.md](../../reports/summary/colleague_metrics_summary.md)
- [reports/summary/config_audit.md](../../reports/summary/config_audit.md)
- [reports/summary/config_curation.md](../../reports/summary/config_curation.md)
- [reports/summary/config_curation.yaml](../../reports/summary/config_curation.yaml)
- [reports/summary/config_provenance.md](../../reports/summary/config_provenance.md)
- [reports/summary/config_registry.yaml](../../reports/summary/config_registry.yaml)
- [reports/summary/experiment_registry.yaml](../../reports/summary/experiment_registry.yaml)
- [reports/summary/experiment_summary.md](../../reports/summary/experiment_summary.md)
- [reports/summary/final_project_results.md](../../reports/summary/final_project_results.md)
- [reports/summary/metrics_glossary.md](../../reports/summary/metrics_glossary.md)
- [reports/summary/metrics_provenance.yaml](../../reports/summary/metrics_provenance.yaml)
- [reports/summary/project_remaining_work.md](../../reports/summary/project_remaining_work.md)
- [reports/summary/requirements_coverage.md](../../reports/summary/requirements_coverage.md)
- [reports/summary/requirements_registry.yaml](../../reports/summary/requirements_registry.yaml)
- [reports/transformer_benchmark_report.md](../../reports/transformer_benchmark_report.md)
- [reports/transformer_legacy_audit.md](../../reports/transformer_legacy_audit.md)
- [reports/user_calibration_architecture_audit.md](../../reports/user_calibration_architecture_audit.md)
- [reports/user_calibration_report.md](../../reports/user_calibration_report.md)
- [scripts/analysis/audit_current_protocol_documentation.py](../../scripts/analysis/audit_current_protocol_documentation.py)
- [tests/test_current_protocol_reporting.py](../../tests/test_current_protocol_reporting.py)
- [tests/test_requirements_coverage.py](../../tests/test_requirements_coverage.py)

## Unrelated working-tree state preserved

The following is exactly the starting unrelated `git status --short` set and remains after task files are committed. Existing permission-denied warnings concern old pytest_tmp subdirectories; none were removed or accessed for new tests. New tests used dedicated ignored benchmark_results directories.

```text
 M reports/diagnostics/pm_low_high_nested_automl_v1/protocol.json
?? .codex_cmf_fix.patch
?? .codex_cmf_fold_manifests.patch
?? .codex_cmf_hash_fix.patch
?? .codex_cmf_hash_fix2.patch
?? .codex_cmf_new/
?? .codex_cmf_new2/
?? .codex_cmf_new3/
?? .codex_cmf_new4/
?? .codex_cmf_old/
?? .codex_cmf_old2/
?? .codex_cmf_old3/
?? .codex_cmf_old4/
?? .codex_cmf_raw_audit_fix.patch
?? .codex_cmf_raw_audit_fix2.patch
?? .codex_cmf_runtime_audit.patch
?? .codex_cmf_temp.py
?? .codex_cmf_temp2.py
?? .codex_cmf_temp3.py
?? .codex_cmf_temp4.py
?? .codex_cmf_test_temp.py
?? .codex_crossval_clean.py
?? .codex_crossval_temp.py
?? .codex_model_family_existing.patch
?? .codex_model_family_old/
?? .codex_model_family_stage/
?? .codex_stage_marker
?? .codex_tmp_faster_audit/
?? .codex_tmp_mne_faster/
?? bench/experiments/pm_qc_channel_drop.py
?? collect_proxy_stage_bundle.ps1
?? collect_target_audit_bundle.ps1
?? "ersGeorgeAppDataLocalProgramsPythonPython312python.exe `"
?? pytest_tmp/
?? reconcile_final_package.ps1
?? reports/diagnostics/channel_quality_smoke_v1/
?? reports/diagnostics/interchannel_lag_pilot_v1/
?? reports/diagnostics/pm_eeg_absolute_lag_v1/runs/
?? reports/diagnostics/pm_eeg_absolute_lag_v1/smoke/
?? reports/diagnostics/pm_eeg_lag_confirmatory_v1/runs/
?? reports/diagnostics/pm_eeg_lag_regression_confirmatory_v1/runs/
?? reports/diagnostics/pm_low_high_model_robustness_v1/runs/
?? reports/diagnostics/pm_low_high_nested_automl_v1/final_runs/
?? reports/diagnostics/pm_low_high_nested_automl_v1/inner_runs/
?? reports/diagnostics/pm_low_high_neural_robustness_v1/runs/
?? reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/runs/
?? reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/runs/
?? reports/diagnostics/pm_low_high_temporal_context_matched_v1/runs/
?? reports/diagnostics/pm_target_validity_audit_fast/
?? reports/diagnostics/pm_target_validity_audit_fast_v3/
?? reports/diagnostics/pm_target_validity_audit_full_v4/
?? reports/diagnostics/pm_target_validity_audit_smoke/
?? reports/diagnostics/pm_temporal_predictability_v1/full_matched/protocol.json
?? scripts/analysis/report_pm_temporal_predictability.py
?? scripts/audit_raw_isactive.py
?? scripts/audit_raw_isactive_v2.py
?? tests/test_pm_qc_channel_drop.py
?? "ults\357\201\274.codex_\357\201\274__pycache__\357\201\274.pyc$\357\201\274features.npy'"
```
