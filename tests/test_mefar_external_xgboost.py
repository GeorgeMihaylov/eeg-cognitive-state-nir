from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.external_multimodal_protocol import (
    load_config as load_external_config,
    run_model_family,
)
from bench.experiments.mefar_external_xgboost import (
    EXPECTED_FOLD_MANIFEST_HASH,
    EXPECTED_SAMPLE_IDS_HASH,
    build_plan,
    file_sha256,
    fit_outer_train_median,
    plan_experiment,
    write_plan_artifacts,
)
from bench.experiments.mefar_multimodal import feature_names


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "external_datasets" / "mefar_multimodal_xgboost_v1.json"
RF_RESULTS = ROOT / "benchmark_results" / "mefar_multimodal_v1"


def _protected_rf_hashes() -> dict[str, str]:
    paths = [
        RF_RESULTS / "summary.csv",
        RF_RESULTS / "protocol_manifest.json",
        RF_RESULTS / "fold_manifest.json",
        *sorted(RF_RESULTS.glob("*/metrics.json")),
    ]
    return {path.relative_to(RF_RESULTS).as_posix(): file_sha256(path) for path in paths}


def test_external_protocol_recognizes_fixed_mefar_contract() -> None:
    config = load_external_config(CONFIG)
    assert config["dataset"]["name"] == "mefar"
    assert config["target"]["target_id"] == "mefar_cfs_fatigue_binary"
    assert config["target"]["threshold"] == {"operator": ">=", "value": 12}
    assert config["model"]["name"] == "xgboost"
    assert config["model"]["hyperparameter_search"] is False


def test_plan_reuses_exact_rf_cohort_folds_and_sample_ids() -> None:
    plan = build_plan(CONFIG)
    protocol = plan["protocol"]
    assert protocol["participants"] == 23
    assert protocol["sessions"] == 46
    assert protocol["class_distribution"] == {"0": 22, "1": 24}
    assert protocol["fold_manifest_hash"] == EXPECTED_FOLD_MANIFEST_HASH
    assert protocol["sample_ids_hash"] == EXPECTED_SAMPLE_IDS_HASH
    assert protocol["source_rf_protocol_hash"] != protocol["protocol_hash"]
    folds = plan["source"]["folds"]["folds"]
    seen: set[str] = set()
    for fold in folds:
        assert not set(fold["train_participants"]) & set(fold["test_participants"])
        assert seen.isdisjoint(fold["test_sample_ids"])
        seen.update(fold["test_sample_ids"])
    assert len(seen) == 46


def test_feature_schemas_and_mode_mapping_remain_exact() -> None:
    plan = build_plan(CONFIG)
    assert {mode: value["feature_count"] for mode, value in plan["protocol"]["modes"].items()} == {
        "eeg_only": 56,
        "wearable_only": 57,
        "eeg_wearable": 113,
    }
    assert plan["protocol"]["modes"]["wearable_only"]["semantic_mode"] == "peripheral_only"
    assert plan["protocol"]["modes"]["eeg_wearable"]["semantic_mode"] == "eeg_peripheral"
    forbidden = ("target", "class", "cfs", "attention", "meditation", "derived")
    assert not any(
        token in column.lower()
        for column in feature_names("eeg_wearable")
        for token in forbidden
    )


def test_run_matrix_has_15_units_and_same_fold_evaluation_ids() -> None:
    matrix = build_plan(CONFIG)["matrix"]
    assert len(matrix) == 15
    assert set(matrix["model"]) == {"xgboost"}
    for _, fold_rows in matrix.groupby("fold"):
        assert set(fold_rows["mode"]) == {"eeg_only", "wearable_only", "eeg_wearable"}
        assert fold_rows["evaluation_sample_ids_hash"].nunique() == 1


def test_plan_artifacts_copy_folds_without_modifying_rf_namespace(tmp_path: Path) -> None:
    before = _protected_rf_hashes()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["output_dir"] = (tmp_path / "mefar_xgb").as_posix()
    temporary_config = tmp_path / "config.json"
    temporary_config.write_text(json.dumps(config), encoding="utf-8")
    written = write_plan_artifacts(temporary_config)
    first = plan_experiment(temporary_config)
    second = plan_experiment(temporary_config)
    assert first == second
    assert written["models_trained"] == first["models_trained"] == 0
    assert first["writes_performed"] is False
    assert (tmp_path / "mefar_xgb" / "fold_manifest.json").read_bytes() == (
        RF_RESULTS / "fold_manifest.json"
    ).read_bytes()
    assert _protected_rf_hashes() == before


def test_leakage_guards_and_shallow_rejection_are_explicit() -> None:
    protocol = build_plan(CONFIG)["protocol"]
    guards = protocol["leakage_guards"]
    assert guards == {
        "participant_disjoint_outer_folds": True,
        "same_evaluation_sample_ids": True,
        "train_only_median_imputation": True,
        "global_scaler": False,
        "oversampling": False,
        "target_columns_excluded": True,
        "attention_meditation_derived_excluded": True,
    }
    assert protocol["shallow_supported"] is False
    with pytest.raises(ValueError, match="no suitable raw multichannel EEG"):
        run_model_family(CONFIG, "shallow")


def test_median_imputation_fits_outer_train_only_without_scaling() -> None:
    train = np.asarray([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    test = np.asarray([[1000.0, np.nan], [np.nan, 1000.0]])
    transformed_train, transformed_test, medians = fit_outer_train_median(train, test)
    np.testing.assert_allclose(medians, [3.0, 20.0])
    np.testing.assert_allclose(transformed_test, [[1000.0, 20.0], [3.0, 1000.0]])
    assert transformed_train[0, 0] == 1.0
