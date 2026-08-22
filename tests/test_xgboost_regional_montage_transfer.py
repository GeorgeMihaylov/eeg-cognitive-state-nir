from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bench.experiments.xgboost_regional_montage_transfer import (
    MONTAGE_PROFILES,
    PROFILE_ORDER,
    _pipeline,
    audit_prediction_identity,
    build_run_matrix,
    feature_cache_identity,
    fit_full_and_evaluate_profiles,
    load_config,
    profile_registry_manifest,
    protocol_plan,
    resumable_summary,
    stable_hash,
    validate_nested_profiles,
)


CONFIG_PATH = Path(
    "experiments/cross_montage/xgboost_regional_montage_transfer_v1.json"
)
CONFIG_PATH_V2 = Path(
    "experiments/cross_montage/xgboost_regional_montage_transfer_v2.json"
)


def test_profile_registry_is_locked_deterministic_and_nested() -> None:
    config = load_config(CONFIG_PATH)
    validate_nested_profiles(config["profiles"])
    assert tuple(config["profiles"]) == PROFILE_ORDER
    assert [len(MONTAGE_PROFILES[name]) for name in PROFILE_ORDER] == [14, 12, 10, 8, 6]
    assert MONTAGE_PROFILES["full_14"] == (
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8",
        "T8", "FC6", "F4", "F8", "AF4",
    )
    for larger, smaller in zip(PROFILE_ORDER, PROFILE_ORDER[1:]):
        assert set(MONTAGE_PROFILES[smaller]) < set(MONTAGE_PROFILES[larger])


def test_profiles_share_schema_and_have_distinct_montage_hashes() -> None:
    config = load_config(CONFIG_PATH)
    manifest = profile_registry_manifest(_pipeline(config), config["profiles"])
    rows = manifest["profiles"]
    assert manifest == profile_registry_manifest(_pipeline(config), config["profiles"])
    assert manifest["feature_width"] == 728
    assert len({row["schema_hash"] for row in rows}) == 1
    assert len({row["montage_hash"] for row in rows}) == 5
    assert all(
        {"frontal_midline", "central_midline", "parietal_midline", "occipital_midline"}
        <= set(row["regions_absent"])
        for row in rows
    )
    by_name = {row["profile"]: row for row in rows}
    assert by_name["full_14"]["constant_missing_region_feature_count"] == 208
    assert by_name["coverage_8"]["constant_missing_region_feature_count"] == 312
    assert by_name["coverage_6"]["constant_missing_region_feature_count"] == 416


def test_v2_profiles_keep_registry_and_montage_metadata_with_new_schema() -> None:
    v1 = load_config(CONFIG_PATH)
    v2 = load_config(CONFIG_PATH_V2)
    v1_manifest = profile_registry_manifest(_pipeline(v1), v1["profiles"])
    v2_manifest = profile_registry_manifest(_pipeline(v2), v2["profiles"])
    assert v1["profiles"] == v2["profiles"]
    assert v1["output_dir"] != v2["output_dir"]
    assert v1_manifest["feature_width"] == 728
    assert v2_manifest["feature_width"] == 364
    assert v1_manifest["regional_feature_schema_hash"] == (
        "10d3beb20bfa2829374916fbbcd1878b851ccccd962ce6fbcf93a343592a2575"
    )
    assert v2_manifest["regional_feature_schema_hash"] == (
        "5db7a691b3f505bb16ca89680ed4080972b7fd9ab07e8b03cdaf6a17cbe7f96e"
    )
    v1_rows = {row["profile"]: row for row in v1_manifest["profiles"]}
    v2_rows = {row["profile"]: row for row in v2_manifest["profiles"]}
    assert {
        name: row["montage_hash"] for name, row in v1_rows.items()
    } == {
        name: row["montage_hash"] for name, row in v2_rows.items()
    }
    assert v2_rows["full_14"]["constant_missing_region_feature_count"] == 104
    assert v2_rows["coverage_8"]["constant_missing_region_feature_count"] == 156
    assert v2_rows["coverage_6"]["constant_missing_region_feature_count"] == 208
    assert v2_rows["full_14"]["montage_manifest"]["input_channel_count"] == 14
    assert v2_rows["coverage_6"]["montage_manifest"]["input_channel_count"] == 6


def test_dry_run_matrix_has_35_trainings_and_175_evaluations() -> None:
    config = load_config(CONFIG_PATH)
    specs = build_run_matrix(config)
    assert len(specs) == 35
    assert len({spec.run_id for spec in specs}) == 35
    assert {(spec.fold, spec.pm) for spec in specs} == {
        (fold, pm) for fold in range(1, 6) for pm in config["targets"]
    }
    assert len(specs) * len(PROFILE_ORDER) == 175
    assert all("profile" not in vars(spec) for spec in specs)


def test_v2_dry_and_smoke_counts_train_full_profile_only() -> None:
    config = load_config(CONFIG_PATH_V2)
    specs = build_run_matrix(config)
    smoke_specs = [
        spec
        for spec in specs
        if spec.fold in config["smoke"]["folds"]
        and spec.pm in config["smoke"]["targets"]
    ]
    assert len(specs) == 35
    assert len(specs) * len(PROFILE_ORDER) == 175
    assert len(smoke_specs) == 7
    assert len(smoke_specs) * len(PROFILE_ORDER) == 35
    assert all("profile" not in vars(spec) for spec in specs)


def test_cache_identity_is_deterministic_and_target_free() -> None:
    config = load_config(CONFIG_PATH)
    index = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "preprocessing_hash": ["raw-hash", "raw-hash"],
        }
    )
    universe = SimpleNamespace(
        manifest=index,
        source_contract_hash="source-hash",
    )
    registry = profile_registry_manifest(_pipeline(config), config["profiles"])
    plan = {"profile_registry": registry}
    first = feature_cache_identity(universe, config, plan)
    second = feature_cache_identity(universe, config, plan)
    assert first == second
    assert first["matrix_shape"] == [2, 5, 728]
    assert first["target_columns_present"] is False
    assert first["cache_identity_hash"] == second["cache_identity_hash"]


def test_v2_cache_identity_is_separate_target_free_and_364_wide() -> None:
    config = load_config(CONFIG_PATH_V2)
    index = pd.DataFrame(
        {"sample_id": [1, 2], "preprocessing_hash": ["raw-hash", "raw-hash"]}
    )
    universe = SimpleNamespace(manifest=index, source_contract_hash="source-hash")
    registry = profile_registry_manifest(_pipeline(config), config["profiles"])
    identity = feature_cache_identity(
        universe, config, {"profile_registry": registry}
    )
    assert identity["matrix_shape"] == [2, 5, 364]
    assert identity["target_columns_present"] is False
    assert identity["regional_schema_hash"] == (
        "5db7a691b3f505bb16ca89680ed4080972b7fd9ab07e8b03cdaf6a17cbe7f96e"
    )


class _FakeBooster:
    def save_raw(self, raw_format: str) -> bytes:
        assert raw_format == "ubj"
        return b"unchanged-fake-booster"


class _FakeModel:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.fit_shape: tuple[int, ...] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeModel":
        self.fit_calls += 1
        self.fit_shape = X.shape
        assert set(np.unique(y)) <= {0, 1, 2}
        return self

    def get_booster(self) -> _FakeBooster:
        return _FakeBooster()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.arange(len(X)) % 3).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 3), 1.0 / 3.0)


def test_one_model_is_fit_only_on_full_and_reused_for_all_profiles() -> None:
    model = _FakeModel()
    X_train_full = np.full((12, 728), 14.0, dtype=np.float32)
    y_train = np.arange(12) % 3
    test_views = {
        profile: np.full((4, 728), len(MONTAGE_PROFILES[profile]), dtype=np.float32)
        for profile in PROFILE_ORDER
    }
    predictions, booster_hash = fit_full_and_evaluate_profiles(
        model, X_train_full, y_train, test_views
    )
    assert model.fit_calls == 1
    assert model.fit_shape == X_train_full.shape
    assert set(predictions) == set(PROFILE_ORDER)
    assert len({stable_hash(values[1].tolist()) for values in predictions.values()}) == 1
    assert booster_hash
    assert all(np.isfinite(proba).all() for _, proba in predictions.values())


def _prediction_frame(profile: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [1, 2],
            "subject_id": ["s1", "s2"],
            "record_group_id": ["r1", "r2"],
            "outer_fold": [1, 1],
            "y_true": [0, 2],
            "y_pred": [0, 1],
            "profile": profile,
        }
    )


def test_exact_sample_target_and_fold_identity_across_profiles() -> None:
    frames = {profile: _prediction_frame(profile) for profile in PROFILE_ORDER}
    audit = audit_prediction_identity(frames)
    assert audit["exact_identity"] is True
    assert audit["rows_per_profile"] == 2
    assert audit["columns"] == [
        "sample_id", "subject_id", "record_group_id", "outer_fold", "y_true"
    ]


def test_resume_requires_current_hash_and_all_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"test")
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "specification_hash": "current",
                "required_artifacts": [str(artifact)],
            }
        ),
        encoding="utf-8",
    )
    assert resumable_summary(summary_path, specification_hash="current") is not None
    assert resumable_summary(summary_path, specification_hash="stale") is None
    artifact.unlink()
    assert resumable_summary(summary_path, specification_hash="current") is None


def test_real_dry_plan_uses_shared_sample_and_fold_identity() -> None:
    plan = protocol_plan(CONFIG_PATH)
    assert plan["expected_xgboost_trainings"] == 35
    assert plan["expected_prediction_evaluations"] == 175
    assert plan["feature_width"] == 728
    assert plan["sample_identity_audit"]["shared_index_for_all_profiles"] is True
    assert plan["sample_identity_audit"]["target_and_fold_identity_shared"] is True
    assert plan["sample_identity_audit"]["sample_ids_unique"] is True
    assert plan["fold_audit"]["reference_assignments_match"] is True
    assert all(
        not fold["subject_overlap"] and not fold["record_group_overlap"]
        for fold in plan["fold_audit"]["folds"].values()
    )
    assert len(plan["target_fold_audit"]) == 35


def test_v1_protocol_plan_and_cache_identity_remain_exactly_unchanged() -> None:
    plan = protocol_plan(CONFIG_PATH)
    config = load_config(CONFIG_PATH)
    from bench.experiments.artifact_removal_ablation_v2 import load_signal_universe

    universe = load_signal_universe(config)
    identity = feature_cache_identity(universe, config, plan)
    assert plan["protocol_hash"] == (
        "abd354693f3d51bc1ae8781aaf05f09a15536767cdaa32d0ca2655b539e66762"
    )
    assert plan["plan_hash"] == (
        "5df8fa8b33c222e8d644d435e8de60182c7c621726b2d78a3c35f23f0ea044ee"
    )
    assert plan["profile_registry"]["profile_registry_hash"] == (
        "18435f13e2ebff09b4392448236e1a647c2bf827aeca61efe07305944c882ef7"
    )
    assert identity["cache_identity_hash"] == (
        "c114d1d151f1fba3132d5bc0b2e72d08e11627b452ff0c2888fde810982e8c79"
    )


def test_v2_real_plan_preserves_v1_sample_fold_and_target_identities() -> None:
    v1 = protocol_plan(CONFIG_PATH)
    v2 = protocol_plan(CONFIG_PATH_V2)
    assert v2["feature_width"] == 364
    assert v2["expected_xgboost_trainings"] == 35
    assert v2["expected_prediction_evaluations"] == 175
    assert v2["expected_smoke_trainings"] == 7
    assert v2["expected_smoke_evaluations"] == 35
    assert v2["sample_identity_audit"] == v1["sample_identity_audit"]
    assert v2["fold_audit"] == v1["fold_audit"]
    assert v2["target_fold_audit"] == v1["target_fold_audit"]
    assert v2["q3_transforms"] == v1["q3_transforms"]
