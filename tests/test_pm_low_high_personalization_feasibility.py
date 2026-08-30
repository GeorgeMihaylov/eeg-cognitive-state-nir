from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_low_high_personalization_feasibility import (
    BUDGETS_SECONDS,
    _as_list,
    _categorize,
    load_config,
)


def _config():
    return {
        "schema_version": "pm-low-high-personalization-feasibility-v1",
        "experiment_id": "pm_low_high_personalization_feasibility_v1",
        "result_status": "preregistered_candidate",
        "references": {
            "low_high": {
                "config": "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json",
                "output_dir": "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1",
                "protocol_hash": "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431",
            },
            "matched_model_selection": {
                "output_dir": "reports/diagnostics/pm_low_high_temporal_context_matched_v1",
                "protocol_hash": "e09f28dab2b37321dd665cc55653cfc08a5a29afc38927ee26bc2d2c6cc988e7",
                "advanced_models": ["xgboost", "lightgbm"],
            },
        },
        "data": {
            "logical_recording_map": "data/interim/logical_recording_map.parquet",
        },
        "scientific_contract": {
            "pm_names": [
                "attention", "engagement", "excitement", "stress",
                "relaxation", "interest", "focus",
            ],
            "alignment": "EEG(t-10s) -> PM(t)",
            "lag_seconds": -10,
            "calibration_budgets_seconds": [0,30,60,120,300],
            "maximum_calibration_budget_seconds": 300,
            "calibration_record_policy":
                "earliest_logical_record_by_selected_record_start_utc",
            "calibration_cross_record_policy": "forbidden",
            "calibration_time_origin": "start_of_earliest_logical_record",
            "calibration_interval_rule":
                "0 < target_relative_seconds <= budget_seconds",
            "budget_measurement":
                "elapsed_recording_time_not_extreme_sample_count",
            "fixed_evaluation_policy":
                "all_exact_lag_targets_strictly_after_max_budget_utc_boundary",
            "fixed_evaluation_boundary_rule":
                "absolute_target_utc > earliest_record_start_utc + 300s",
            "reserved_interval_policy":
                "targets_after_current_budget_and_not_after_300s_are_unused",
            "target_transform": "outer_train_q33_q67_extremes",
            "threshold_fit_scope":
                "outer_train_continuous_complete_cases",
            "middle_policy":
                "exclude_from_binary_training_and_evaluation_but_count_in_feasibility",
            "missing_pm_policy": "count_as_missing_not_middle",
            "outer_group": "subject_id",
            "folds": [1,2,3,4,5],
        },
        "feasibility_criteria": {
            "report_any_extreme": True,
            "report_both_extreme_classes": True,
            "report_minimum_per_class": [1,2,3],
            "minimum_fixed_evaluation_extremes_descriptive": 20,
            "minimum_fixed_evaluation_both_classes": True,
            "criteria_role":
                "descriptive_only_final_personalization_rules_frozen_after_feasibility_audit",
        },
        "forbidden": {
            "model_training": True,
            "model_inference": True,
            "performance_metric_use": True,
            "threshold_refitting_on_test_subject": True,
            "search_until_both_classes_observed": True,
            "extend_budget_to_collect_extremes": True,
            "cross_record_calibration_stitching": True,
            "target_specific_budget_change": True,
            "focus_specific_budget_change": True,
        },
        "output_dir":
            "reports/diagnostics/pm_low_high_personalization_feasibility_v1",
    }


def test_config_accepts_frozen_contract(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_config(path)
    assert tuple(
        loaded["scientific_contract"]["calibration_budgets_seconds"]
    ) == BUDGETS_SECONDS


def test_config_rejects_budget_change(tmp_path):
    config = _config()
    config["scientific_contract"]["calibration_budgets_seconds"] = [
        0, 30, 60, 180, 300
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Calibration budgets"):
        load_config(path)


def test_config_rejects_scan_until_classes(tmp_path):
    config = _config()
    config["forbidden"]["search_until_both_classes_observed"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_config(path)


def test_as_list_handles_numpy_and_json():
    assert _as_list(np.asarray(["a", "b"])) == ["a", "b"]
    assert _as_list('["a", "b"]') == ["a", "b"]


def test_categorize_keeps_missing_separate_from_middle():
    values = np.asarray([0.1, 0.5, 0.9, np.nan])
    result = _categorize(values, q_low=0.3, q_high=0.7)
    assert result.tolist() == ["low", "middle", "high", "missing"]


def test_boundary_semantics_are_three_windows_at_30s():
    relative = np.asarray([10.0, 20.0, 30.0, 40.0])
    selected = relative[(relative > 0.0) & (relative <= 30.0)]
    assert selected.tolist() == [10.0, 20.0, 30.0]
