from __future__ import annotations

import json
import numpy as np
import pytest

from bench.experiments.pm_low_high_personalized_threshold import (
    _empirical_ba_threshold,
    _median_midpoint,
    load_config,
)


def test_median_midpoint_adapts_when_medians_ordered():
    p = np.asarray([0.1, 0.2, 0.8, 0.9])
    y = np.asarray([0, 0, 1, 1])
    threshold, applied, reason, extra = _median_midpoint(p, y)
    assert applied is True
    assert reason == "adapted"
    assert threshold == pytest.approx(0.5)
    assert extra["median_probability_low"] == pytest.approx(0.15)
    assert extra["median_probability_high"] == pytest.approx(0.85)


def test_median_midpoint_falls_back_when_medians_reversed():
    p = np.asarray([0.8, 0.9, 0.1, 0.2])
    y = np.asarray([0, 0, 1, 1])
    threshold, applied, reason, _ = _median_midpoint(p, y)
    assert threshold == 0.5
    assert applied is False
    assert reason == "nonseparated_medians"


def test_empirical_threshold_is_deterministic_and_prefers_near_half():
    p = np.asarray([0.1, 0.2, 0.8, 0.9])
    y = np.asarray([0, 0, 1, 1])
    first = _empirical_ba_threshold(p, y)
    second = _empirical_ba_threshold(p, y)
    assert first == second
    assert first[1] is True
    assert 0.2 < first[0] < 0.8


def test_config_rejects_30_second_execution(tmp_path):
    # Build from the real-shaped minimum fixture.
    cfg = {
      "schema_version":"pm-low-high-personalized-threshold-v1",
      "references":{
        "feasibility":{"config":"x","output_dir":"x","protocol_hash":"94c568d7e41344478c0550f573b0abf8893783831f6c7241b92c8e4fdd25c9cd"},
        "xgboost":{"output_dir":"x","protocol_hash":"ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"},
        "lightgbm":{"output_dir":"x","protocol_hash":"a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e","model":"lightgbm"}
      },
      "scientific_contract":{
        "pm_names":["attention","engagement","excitement","stress","relaxation","interest","focus"],
        "models":["xgboost","lightgbm"],"alignment":"EEG(t-10s) -> PM(t)","lag_seconds":-10,
        "target_transform":"outer_train_q33_q67_extremes","threshold_fit_scope":"outer_train_continuous_complete_cases",
        "middle_policy":"exclude","calibration_record_policy":"earliest_logical_record_by_selected_record_start_utc",
        "cross_record_overlap_policy":"earlier_record_precedence_trim_later_overlapping_prefix_by_feature_grid_utc",
        "fixed_evaluation_boundary_seconds":300,"fixed_evaluation_policy":"strictly_after_earliest_record_start_plus_300s",
        "budgets_seconds":[30,60,120,300],"budget_roles":{"60":"exploratory","120":"secondary","300":"primary"},
        "excluded_budget_30s_reason":"feasibility_min2_each_zero_of_378","minimum_calibration_per_class":2,
        "minimum_fixed_evaluation_extremes":20,"require_both_evaluation_classes":True,
        "ineligible_policy":"zero_shot_fallback_no_budget_extension",
        "probability_source":"stored_predict_proba_high","base_decision_threshold":0.5
      },
      "strategies":{
        "median_midpoint":{"role":"primary","rule":"midpoint_between_median_LOW_and_median_HIGH_probability","nonseparated_medians_policy":"zero_shot_fallback"},
        "empirical_balanced_accuracy":{"role":"sensitivity","rule":"maximize_calibration_balanced_accuracy_over_probability_midpoints","tie_break":["closest_to_0.5","lower_threshold"]}
      },
      "evaluation":{
        "primary_metric":"balanced_accuracy","secondary_metrics":["macro_f1","low_recall","high_recall","precision","accuracy"],
        "unchanged_ranking_metrics":["roc_auc","pr_auc"],"aggregation":"mean_pm_within_participant_then_mean_participants",
        "primary_estimand":"operational_all_fixed_evaluation_ready_with_zero_shot_fallback",
        "secondary_estimand":"adaptation_applied_only","bootstrap_replicates":10000,"bootstrap_seed":42,"bootstrap_unit":"subject_id"
      },
      "forbidden":{
        "base_model_retraining":True,"base_model_inference":True,"feature_refitting":True,"lag_search":True,
        "budget_search":True,"threshold_method_search":True,"scan_forward_until_classes":True,
        "extend_calibration_budget":True,"use_evaluation_labels_for_calibration":True,
        "target_specific_budget":True,"focus_specific_logic":True
      },
      "output_dir":"x"
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific contract"):
        load_config(path)
