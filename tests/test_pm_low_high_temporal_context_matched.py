from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bench.experiments.pm_low_high_temporal_context_matched import (
    EXPECTED_MODELS,
    MODEL_ORDER,
    TEMPORAL_TABULAR_PAIRS,
    load_config,
)


def _config():
    return {
        "schema_version": "pm-low-high-temporal-context-matched-v1",
        "experiment_id": "pm_low_high_temporal_context_matched_v1",
        "result_status": "preregistered_candidate",
        "references": {
            "low_high": {
                "config": "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json",
                "output_dir": "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1",
                "protocol_hash": "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431",
            },
            "neural_robustness": {
                "output_dir": "reports/diagnostics/pm_low_high_neural_robustness_v1",
                "protocol_hash": "e902f4dbe8f317be4ac6ed5061104cf5a3399eea5125414a675add88f4105a8d",
                "reused_model": "torch_lstm",
            },
        },
        "scientific_contract": {
            "pm_names": [
                "attention", "engagement", "excitement", "stress",
                "relaxation", "interest", "focus",
            ],
            "alignment": "causal EEG history ending at t-10s -> PM(t)",
            "lag_seconds": -10,
            "common_cohort": "10_contiguous_feature_windows_ending_at_t_minus_10s",
            "target_transform": "outer_train_q33_q67_extremes",
            "middle_policy": "exclude",
            "outer_group": "subject_id",
            "folds": [1,2,3,4,5],
            "seed": 42,
        },
        "models": deepcopy(EXPECTED_MODELS),
        "validation": {
            "strategy": "group_record",
            "group_column": "record_group_id",
            "validation_size": 0.15,
            "random_state": 42,
        },
        "evaluation": {
            "primary_metric": "participant_macro_balanced_accuracy",
            "secondary_metrics": [
                "participant_macro_f1",
                "participant_macro_roc_auc",
                "participant_macro_pr_auc",
                "participant_macro_low_recall",
                "participant_macro_high_recall",
                "participant_macro_precision",
                "participant_macro_accuracy",
            ],
            "paired_temporal_vs_tabular": [
                list(pair) for pair in TEMPORAL_TABULAR_PAIRS
            ],
            "participant_cluster_bootstrap": {
                "replicates": 10000,
                "seed": 42,
                "cluster": "subject_id",
                "metrics": ["balanced_accuracy", "f1", "roc_auc"],
            },
        },
        "model_selection_for_personalization": {
            "ranking_metric": "participant_macro_balanced_accuracy",
            "practical_equivalence_margin": 0.01,
            "maximum_models_advanced": 2,
            "rule": "advance_best_only_unless_second_best_is_within_0.01_balanced_accuracy",
            "tie_breakers": [
                "participant_macro_roc_auc",
                "lower_balanced_accuracy_std",
                "fixed_model_order",
            ],
            "fixed_model_order": list(MODEL_ORDER),
        },
        "forbidden": {
            "hyperparameter_search": True,
            "lag_search": True,
            "sequence_length_search": True,
            "target_specific_models": True,
            "focus_specific_logic": True,
            "feature_selection": True,
            "class_reweighting": True,
            "oversampling": True,
            "test_threshold_fitting": True,
            "random_window_validation": True,
        },
        "matrix_cells": 140,
        "planned_new_fits": 105,
        "planned_reused_reference_fits": 35,
        "output_dir": "reports/diagnostics/pm_low_high_temporal_context_matched_v1",
    }


def test_config_accepts_frozen_contract(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["matrix_cells"] == 140
    assert loaded["planned_new_fits"] == 105
    assert loaded["planned_reused_reference_fits"] == 35


def test_config_rejects_transformer_change(tmp_path):
    config = _config()
    config["models"]["torch_transformer"]["params"]["d_model"] = 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="historical hyperparameters"):
        load_config(path)


def test_config_rejects_common_cohort_change(tmp_path):
    config = _config()
    config["scientific_contract"]["common_cohort"] = "8_window"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Common matched cohort"):
        load_config(path)


def test_config_rejects_lstm_retraining_count(tmp_path):
    config = _config()
    config["planned_new_fits"] = 140
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="105 new fits"):
        load_config(path)


def test_config_rejects_pair_change(tmp_path):
    config = _config()
    config["evaluation"]["paired_temporal_vs_tabular"] = [
        ["torch_lstm", "lightgbm"]
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Paired temporal-vs-tabular"):
        load_config(path)
