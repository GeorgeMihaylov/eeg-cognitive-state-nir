from __future__ import annotations

import json

import pandas as pd
import pytest

from bench.experiments.pm_low_high_neural_robustness import (
    EXPECTED_MODELS,
    exact_history_endpoint_ids,
    load_config,
)

from copy import deepcopy

def _config():
    return {
        "schema_version": "pm-low-high-neural-robustness-v1",
        "experiment_id": "pm_low_high_neural_robustness_v1",
        "result_status": "preregistered_candidate",
        "reference": {
            "config": "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json",
            "output_dir": "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1",
            "protocol_hash": "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431",
        },
        "data": {
            "raw_manifest": "data/interim/raw_eeg_window_index_w10_pm_union_composite_v1.parquet",
        },
        "scientific_contract": {
            "pm_names": [
                "attention", "engagement", "excitement", "stress",
                "relaxation", "interest", "focus",
            ],
            "alignment": "EEG history ending at t-10s -> PM(t)",
            "lag_seconds": -10,
            "target_transform": "outer_train_q33_q67_extremes",
            "middle_policy": "exclude",
            "outer_group": "subject_id",
            "folds": [1,2,3,4,5],
            "seed": 42,
        },
        "raw_input": {
            "shape": [1,14,2560],
            "sampling_rate_hz": 256,
            "window_seconds": 10,
            "expected_preprocessing_hash": "2251ca950a467267dcccc1c5b83157f26e02768f46c6073d33f5dc16225bda84",
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
            "screening_only": True,
            "direct_cross_architecture_ranking_forbidden": True,
            "matched_cohort_followup_if_sequence_competitive": True,
        },
        "forbidden": {
            "hyperparameter_search": True,
            "lag_search": True,
            "target_specific_models": True,
            "focus_specific_logic": True,
            "feature_selection": True,
            "class_reweighting": True,
            "oversampling": True,
            "test_threshold_fitting": True,
            "random_window_validation": True,
        },
        "planned_fits": 105,
        "output_dir": "reports/diagnostics/pm_low_high_neural_robustness_v1",
    }


def test_config_accepts_frozen_models(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["planned_fits"] == 105
    assert list(loaded["models"]) == [
        "torch_shallow_convnet", "torch_lstm", "torch_transformer"
    ]


def test_config_rejects_transformer_change(tmp_path):
    config = _config()
    config["models"]["torch_transformer"]["params"]["d_model"] = 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="historical hyperparameters"):
        load_config(path)


def test_exact_history_endpoint_ids_requires_contiguous_steps():
    metadata = pd.DataFrame({
        "sample_id": [1,2,3,4,5,6],
        "source": ["s"]*6,
        "subject_id": ["p"]*6,
        "record_id": ["r"]*6,
        "record_group_id": ["g"]*6,
        "t_start": [0.0,10.0,20.0,40.0,50.0,60.0],
    })
    assert exact_history_endpoint_ids(metadata, length=3) == {3, 6}


def test_exact_history_does_not_cross_record_group():
    metadata = pd.DataFrame({
        "sample_id": [1,2,3,4],
        "source": ["s"]*4,
        "subject_id": ["p"]*4,
        "record_id": ["r1","r1","r2","r2"],
        "record_group_id": ["g1","g1","g2","g2"],
        "t_start": [0.0,10.0,20.0,30.0],
    })
    assert exact_history_endpoint_ids(metadata, length=3) == set()
