from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.datasets.datasets_registry import DATASET_REGISTRY, get_dataset
from bench.datasets.mefar_dataset import MEFARDataset
from bench.experiments.mefar_multimodal import (
    EXPECTED_ARCHIVE_SHA256,
    build_fold_manifest,
    build_inventory,
    build_protocol,
    build_run_matrix,
    feature_names,
    file_sha256,
    load_config,
    plan_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "external_datasets" / "mefar_multimodal_v1.json"
RESULTS = ROOT / "benchmark_results" / "mefar_multimodal_v1"


def _sessions() -> pd.DataFrame:
    config = load_config(CONFIG)
    return build_inventory(config)["sessions"]


def test_mefar_dataset_is_registered_as_lazy_record_dataset() -> None:
    assert DATASET_REGISTRY["mefar"] is MEFARDataset
    dataset = get_dataset("mefar", {"data_path": "data/raw/mefar/extracted"})
    assert isinstance(dataset, MEFARDataset)
    records = list(dataset.iter_records())
    assert len(records) == 46
    assert len({record.participant_id for record in records}) == 23


def test_archive_is_immutable_and_matches_validated_digest() -> None:
    config = load_config(CONFIG)
    archive = ROOT / config["dataset"]["archive"]
    before = file_sha256(archive)
    assert before == EXPECTED_ARCHIVE_SHA256
    assert file_sha256(archive) == before


def test_target_is_verified_session_level_cfs_binary() -> None:
    sessions = _sessions()
    assert set(sessions["target_id"]) == {"mefar_cfs_fatigue_binary"}
    assert set(sessions["target"]) == {0, 1}
    assert (sessions["cfs_threshold"] == 12).all()
    assert (
        sessions["target"] == sessions["cfs_likert_score"].ge(12).astype(int)
    ).all()
    assert sessions["score_mapping_verified"].all()
    assert (
        sessions["source_subject_list_score"]
        == sessions["source_response_sheet_score"]
    ).all()
    assert (
        sessions["source_subject_list_score"]
        == sessions["source_reported_sheet_score"]
    ).all()
    assert (sessions["source_response_count"] == 11).all()
    assert set(sessions["session_time_proxy_role"]) == {"diagnostic_metadata_only"}
    assert sessions["target"].value_counts().sort_index().to_dict() == {0: 22, 1: 24}
    assert (
        sessions.groupby("session_label")["target"]
        .value_counts()
        .sort_index()
        .to_dict()
        == {("evening", 0): 6, ("evening", 1): 17, ("morning", 0): 16, ("morning", 1): 7}
    )


def test_fold_assignment_is_deterministic_and_participant_disjoint() -> None:
    sessions = _sessions()
    first = build_fold_manifest(sessions)
    second = build_fold_manifest(sessions)
    assert first == second
    test_ids = []
    for fold in first["folds"]:
        assert not set(fold["train_participants"]) & set(fold["test_participants"])
        test_ids.extend(fold["test_sample_ids"])
    assert len(test_ids) == len(set(test_ids)) == 46
    assert [fold["train_class_counts"] for fold in first["folds"]] == [
        {"0": 18, "1": 18},
        {"0": 18, "1": 18},
        {"0": 16, "1": 20},
        {"0": 17, "1": 21},
        {"0": 19, "1": 19},
    ]
    assert [fold["test_class_counts"] for fold in first["folds"]] == [
        {"0": 4, "1": 6},
        {"0": 4, "1": 6},
        {"0": 6, "1": 4},
        {"0": 5, "1": 3},
        {"0": 3, "1": 5},
    ]
    assert all(len(fold["train_class_counts"]) == 2 for fold in first["folds"])
    assert all(len(fold["test_class_counts"]) == 2 for fold in first["folds"])


def test_modes_share_identical_evaluation_cohort() -> None:
    config = load_config(CONFIG)
    sessions = _sessions()
    folds = build_fold_manifest(sessions)
    matrix = build_run_matrix(config, folds)
    assert len(matrix) == 15
    for _, group in matrix.groupby("fold"):
        assert set(group["mode"]) == {"eeg_only", "wearable_only", "eeg_wearable"}
        assert group["evaluation_sample_ids_hash"].nunique() == 1
        assert group["n_test_samples"].nunique() == 1


def test_feature_contracts_are_stable_and_exclude_leakage_columns() -> None:
    eeg = feature_names("eeg_only")
    wearable = feature_names("wearable_only")
    fused = feature_names("eeg_wearable")
    assert len(eeg) == 56
    assert len(wearable) == 57
    assert fused == eeg + wearable
    assert len(fused) == len(set(fused)) == 113
    forbidden = ("target", "class", "attention", "meditation", "session_label", "cfs")
    assert not any(token in name.lower() for name in fused for token in forbidden)


def test_protocol_forbids_global_scaling_and_oversampling() -> None:
    config = load_config(CONFIG)
    sessions = _sessions()
    inventory = {
        "archive_before": EXPECTED_ARCHIVE_SHA256,
        "sessions": sessions,
    }
    protocol = build_protocol(config, inventory)["protocol"]
    assert protocol["leakage_guards"]["participant_disjoint_outer_folds"] is True
    assert protocol["leakage_guards"]["train_only_imputation"] is True
    assert protocol["leakage_guards"]["global_scaler"] is False
    assert protocol["leakage_guards"]["oversampling"] is False
    assert protocol["leakage_guards"]["processed_down_mid_up_used"] is False


def test_synchronization_contract_does_not_claim_window_alignment() -> None:
    synchronization = pd.read_csv(RESULTS / "synchronization_audit.csv")
    assert set(synchronization["safe_fusion_level"]) == {"participant_session_summary"}
    assert not synchronization["common_absolute_clock"].any()
    assert int(synchronization["explicit_sync_marker"].sum()) == 2


def test_plan_only_is_read_only_and_trains_no_models() -> None:
    manifest = RESULTS / "protocol_manifest.json"
    before = file_sha256(manifest)
    first = plan_experiment(CONFIG)
    second = plan_experiment(CONFIG)
    assert first == second
    assert first["models_trained"] == 0
    assert first["writes_performed"] is False
    assert first["class_distribution"] == {"0": 22, "1": 24}
    assert first["participants_changing_class"] == 12
    assert first["participants_same_class"] == 11
    assert first["existing_group_kfold_usable"] is True
    assert file_sha256(manifest) == before
