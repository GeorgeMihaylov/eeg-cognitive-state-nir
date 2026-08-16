from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bench.datasets.clare_cldrive_dataset import CLAREDataset, CLDriveDataset
from bench.datasets.datasets_registry import DATASET_REGISTRY, get_dataset
from bench.experiments.external_multimodal_protocol import (
    EXPECTED_SHA256,
    build_folds,
    build_run_matrix,
    compatibility_matrix,
    feature_names,
    file_sha256,
    load_config,
    plan_experiment,
)
from model_zoo.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "clare": ROOT / "experiments" / "external_datasets" / "clare_multimodal_v1.json",
    "cl_drive": ROOT / "experiments" / "external_datasets" / "cl_drive_multimodal_v1.json",
}


def test_archives_match_validated_sha256_and_are_unchanged() -> None:
    for dataset, config_path in CONFIGS.items():
        config = load_config(config_path)
        archive = ROOT / config["dataset"]["archive"]
        before = file_sha256(archive)
        assert before == EXPECTED_SHA256[dataset]
        assert file_sha256(archive) == before


def test_record_loaders_are_registered_and_report_real_counts() -> None:
    assert DATASET_REGISTRY["clare"] is CLAREDataset
    assert DATASET_REGISTRY["cl_drive"] is CLDriveDataset
    expected = {"clare": (20, 79), "cl_drive": (21, 189)}
    for dataset, config_path in CONFIGS.items():
        config = load_config(config_path)
        instance = get_dataset(dataset, {"data_path": config["dataset"]["extracted_root"]})
        records = list(instance.iter_records())
        participants, record_count = expected[dataset]
        assert len(records) == record_count
        assert len({record.participant_id for record in records}) == participants


def test_feature_contracts_are_stable_and_target_free() -> None:
    assert len(feature_names("eeg_only")) == 52
    assert len(feature_names("peripheral_only")) == 28
    assert len(feature_names("eeg_peripheral")) == 80
    forbidden = ("target", "label", "class", "gaze")
    assert not any(
        token in column.lower()
        for column in feature_names("eeg_peripheral")
        for token in forbidden
    )


def test_xgboost_uses_existing_factory_without_training() -> None:
    config = load_config(CONFIGS["clare"])
    model = build_model(
        "xgboost", "classification", (52,), 3, config["models"]["xgboost"]
    )
    assert model.__class__.__name__ == "XGBClassifier"


def test_compatibility_matrix_separates_gaze_and_raw_eeg() -> None:
    matrix = compatibility_matrix()
    supported = set(
        matrix.loc[matrix["supported"], ["model", "mode"]].itertuples(index=False, name=None)
    )
    assert ("xgboost", "peripheral_only") in supported
    assert ("torch_shallow_convnet", "eeg_only") in supported
    assert ("torch_shallow_fusion", "eeg_peripheral") in supported
    assert ("torch_shallow_convnet", "peripheral_only") not in supported
    assert not matrix.loc[matrix["mode"].eq("gaze_only"), "supported"].any()


def test_real_fold_manifests_are_disjoint_class_complete_and_deterministic() -> None:
    for dataset, config_path in CONFIGS.items():
        config = load_config(config_path)
        output = ROOT / config["output_dir"]
        cohort = pd.read_csv(output / "cohort_inventory.csv")
        first = build_folds(config, cohort)
        second = build_folds(config, cohort)
        assert first == second
        seen: set[str] = set()
        for fold in first["folds"]:
            assert fold["participant_overlap"] == 0
            assert set(fold["train_participants"]).isdisjoint(fold["test_participants"])
            assert set(fold["train_class_counts"]) == {"0", "1", "2"}
            assert set(fold["test_class_counts"]) == {"0", "1", "2"}
            assert seen.isdisjoint(fold["test_sample_ids"])
            seen.update(fold["test_sample_ids"])
        assert seen == set(cohort["sample_id"].astype(str))
        matrix = build_run_matrix(config, first)
        assert len(matrix) == 25
        for _, group in matrix.groupby("fold"):
            assert group["evaluation_sample_ids_hash"].nunique() == 1


def test_plan_only_is_deterministic_read_only_and_training_free() -> None:
    for config_path in CONFIGS.values():
        config = load_config(config_path)
        manifest = ROOT / config["output_dir"] / "protocol_manifest.json"
        before = file_sha256(manifest)
        first = plan_experiment(config_path)
        second = plan_experiment(config_path)
        assert first == second
        assert first["models_trained"] == 0
        assert first["writes_performed"] is False
        assert first["run_count"] == 25
        assert first["evaluation_units"] == 25
        assert first["usable_common_cohort_participants"] == first["participants"]
        assert set(first["modality_record_counts"]) == {"ECG", "EDA", "EEG", "Gaze"}
        assert "runs/<run_id>/predictions.parquet" in first["expected_training_artifacts"]
        assert file_sha256(manifest) == before


def test_protocol_does_not_claim_unproven_clock_or_oversampling() -> None:
    for config_path in CONFIGS.values():
        config = load_config(config_path)
        protocol = json.loads(
            (ROOT / config["output_dir"] / "protocol_manifest.json").read_text(encoding="utf-8")
        )
        assert protocol["nearest_neighbour_merge"] is False
        assert protocol["leakage_guards"]["oversampling_before_split"] is False
        assert protocol["leakage_guards"]["train_only_imputer_scaler"] is True
        assert protocol["leakage_guards"]["dataset_mixing"] is False
