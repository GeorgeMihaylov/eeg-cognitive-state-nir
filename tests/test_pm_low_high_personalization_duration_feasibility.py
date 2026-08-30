from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_low_high_personalization_duration_feasibility import (
    _calibration_mask,
    _evaluation_mask,
    _state_counts,
    load_config,
)


def _config() -> dict:
    return {
        "schema_version": "pm-low-high-personalization-duration-feasibility-v1",
        "experiment_id":
            "pm_low_high_personalization_duration_feasibility_v1",
        "reference_feasibility": {
            "config": "x",
            "output_dir": "x",
            "protocol_hash":
                "94c568d7e41344478c0550f573b0abf8893783831f6c7241b92c8e4fdd25c9cd",
        },
        "scientific_contract": {
            "pm_names": [
                "attention", "engagement", "excitement", "stress",
                "relaxation", "interest", "focus",
            ],
            "alignment": "EEG(t-10s) -> PM(t)",
            "lag_seconds": -10,
            "calibration_budgets_seconds": [300, 600, 900],
            "maximum_calibration_budget_seconds": 900,
            "calibration_record_policy":
                "earliest_logical_record_by_selected_record_start_utc",
            "calibration_cross_record_policy": "forbidden",
            "calibration_time_origin":
                "start_of_earliest_logical_record",
            "calibration_interval_rule":
                "0 < target_relative_seconds <= budget_seconds",
            "budget_measurement":
                "elapsed_recording_time_not_extreme_sample_count",
            "fixed_evaluation_policy":
                "all_exact_lag_targets_strictly_after_max_budget_utc_boundary",
            "fixed_evaluation_boundary_rule":
                "absolute_target_utc > earliest_record_start_utc + 900s",
            "reserved_interval_policy":
                "targets_after_current_budget_and_not_after_900s_are_unused",
            "target_transform": "outer_train_q33_q67_extremes",
            "threshold_fit_scope":
                "outer_train_continuous_complete_cases",
            "middle_policy":
                "exclude_from_binary_training_and_evaluation_but_count_in_feasibility",
            "missing_pm_policy": "count_as_missing_not_middle",
            "outer_group": "subject_id",
            "folds": [1, 2, 3, 4, 5],
            "cross_record_overlap_policy":
                "earlier_record_precedence_trim_later_overlapping_prefix_by_feature_grid_utc",
            "future_personalization_models": ["xgboost", "lightgbm"],
            "future_threshold_strategy": "median_midpoint",
            "minimum_calibration_per_class_for_future_run": 2,
            "future_ineligible_policy":
                "zero_shot_fallback_no_budget_extension",
        },
        "feasibility_criteria": {
            "report_any_extreme": True,
            "report_both_extreme_classes": True,
            "report_minimum_per_class": [1, 2, 3, 5],
            "minimum_fixed_evaluation_extremes_descriptive": 20,
            "minimum_fixed_evaluation_both_classes": True,
            "report_joint_min2_and_fixed_evaluation_ready": True,
            "criteria_role":
                "descriptive_prerun_gate_for_duration_dose_response",
        },
        "planned_duration_comparison": {
            "control_budget_seconds": 300,
            "intermediate_budget_seconds": 600,
            "primary_budget_seconds": 900,
            "primary_contrast": "900s_minus_300s",
            "secondary_contrast": "600s_minus_300s",
            "common_evaluation_boundary_seconds": 900,
            "performance_run_not_executed_by_this_audit": True,
        },
        "forbidden": {
            "model_training": True,
            "model_inference": True,
            "scan_forward_until_classes": True,
            "extend_budget_until_classes": True,
            "calibration_cross_record_stitching": True,
            "target_specific_budget": True,
            "focus_specific_logic": True,
            "lag_search": True,
            "threshold_method_search": True,
            "use_evaluation_labels_for_calibration": True,
            "change_outer_train_q33_q67_thresholds": True,
            "change_common_evaluation_boundary_by_budget": True,
        },
        "output_dir": "x",
    }


def test_config_accepts_frozen_contract(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["scientific_contract"]["calibration_budgets_seconds"] == [
        300, 600, 900
    ]


def test_config_rejects_budget_search(tmp_path):
    cfg = _config()
    cfg["scientific_contract"]["calibration_budgets_seconds"] = [
        300, 600, 900, 1200
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific contract"):
        load_config(path)


def test_calibration_budget_includes_exact_endpoint():
    relative = np.asarray([0.0, 10.0, 300.0, 300.001, 600.0])
    mask = _calibration_mask(relative, 300)
    assert relative[mask].tolist() == [10.0, 300.0]


def test_common_evaluation_is_strictly_after_900_seconds():
    start = 1000.0
    absolute = np.asarray([1899.999, 1900.0, 1900.001, 2000.0])
    mask = _evaluation_mask(absolute, start)
    assert absolute[mask].tolist() == [1900.001, 2000.0]


def test_state_counts_reports_min5_each():
    frame = pd.DataFrame({
        "state": (
            ["low"] * 5
            + ["high"] * 5
            + ["middle"] * 2
            + ["missing"]
        )
    })
    counts = _state_counts(frame, "calibration")
    assert counts["calibration_min2_each"] is True
    assert counts["calibration_min3_each"] is True
    assert counts["calibration_min5_each"] is True
    assert counts["calibration_extreme"] == 10
    assert counts["calibration_middle"] == 2
    assert counts["calibration_missing_pm"] == 1
