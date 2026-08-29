from __future__ import annotations
import json
import copy
import pandas as pd
import pytest
from bench.experiments.pm_low_high_model_robustness import ALL_MODEL_ORDER, EXPECTED_MODELS, load_config, select_models_for_personalization

def _valid_config():
    return {
        "schema_version":"pm-low-high-model-robustness-v1","experiment_id":"pm_low_high_model_robustness_371_v1","result_status":"preregistered_candidate",
        "reference":{"config":"experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json","output_dir":"reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1","experiment_id":"pm_low_high_q3_extremes_confirmatory_371_xgboost_v1","protocol_hash":"ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431","model":"xgboost"},
        "scientific_contract":{"pm_names":["attention","engagement","excitement","stress","relaxation","interest","focus"],"alignment":"EEG(t-10s) -> PM(t)","lag_seconds":-10,"feature_count":371,"target_transform":"outer_train_q33_q67_extremes","middle_policy":"exclude","outer_group":"subject_id","folds":[1,2,3,4,5],"seed":42},
        "candidate_models":copy.deepcopy(EXPECTED_MODELS),
        "evaluation":{"primary_metric":"participant_macro_balanced_accuracy","secondary_metrics":["participant_macro_f1","participant_macro_roc_auc","participant_macro_pr_auc","participant_macro_low_recall","participant_macro_high_recall","participant_macro_accuracy"],"probability_source":"predict_proba_high","single_class_auc_policy":"undefined_exclude_metric_only"},
        "model_selection_for_personalization":{"ranking_metric":"participant_macro_balanced_accuracy","practical_equivalence_margin":0.01,"maximum_models_advanced":2,"rule":"advance_best_only_unless_second_best_is_within_0.01_balanced_accuracy","tie_breakers":["participant_macro_roc_auc","lower_balanced_accuracy_std","fixed_model_order"],"fixed_model_order":list(ALL_MODEL_ORDER)},
        "forbidden":{"hyperparameter_search":True,"lag_search":True,"target_specific_models":True,"focus_specific_logic":True,"feature_selection":True,"class_reweighting":True,"oversampling":True,"test_threshold_fitting":True},
        "planned_new_fits":70,"output_dir":"reports/diagnostics/pm_low_high_model_robustness_v1"}

def test_load_config_accepts_contract(tmp_path):
    p=tmp_path/"c.json"; p.write_text(json.dumps(_valid_config()),encoding="utf-8")
    assert load_config(p)["planned_new_fits"]==70

def test_load_config_rejects_model_change(tmp_path):
    c=_valid_config(); c["candidate_models"]["random_forest"]["params"]["n_estimators"]=201
    p=tmp_path/"c.json"; p.write_text(json.dumps(c),encoding="utf-8")
    with pytest.raises(ValueError,match="hyperparameters"): load_config(p)

def _summary(xgb,lgbm,rf):
    vals={"xgboost":xgb,"lightgbm":lgbm,"random_forest":rf}
    return pd.DataFrame([{"model":m,"participant_macro_balanced_accuracy_mean":v,"participant_macro_roc_auc_mean":0.8,"participant_macro_balanced_accuracy_std":0.05} for m,v in vals.items()])

def test_selection_two_within_margin():
    assert select_models_for_personalization(_summary(.750,.745,.720))["advanced_models"]==["xgboost","lightgbm"]

def test_selection_one_outside_margin():
    assert select_models_for_personalization(_summary(.750,.730,.720))["advanced_models"]==["xgboost"]

def test_candidate_can_win():
    assert select_models_for_personalization(_summary(.750,.770,.730))["advanced_models"]==["lightgbm"]
