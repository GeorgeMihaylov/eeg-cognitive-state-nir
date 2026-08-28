from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_confirmatory_benchmark import (
    MODELS,
    audit_preliminary_checkpoints,
    benchmark_run_config,
    build_evaluation_matrix,
    build_metadata_plan,
    build_training_matrix,
    checkpoint_identity,
    sequence_endpoint_ids,
    stable_hash,
    validate_checkpoint_identity,
    validate_feature_cache_identity,
    write_plan,
)
from cogstate.model_zoo import build_model
from cogstate.model_zoo.DL.sequence_utils import build_sequences


def _config() -> dict:
    return json.loads(
        Path("experiments/pm_confirmatory/selected_models_5fold_v1.json")
        .read_text(encoding="utf-8")
    )


def _synthetic_index() -> pd.DataFrame:
    rows = []
    sample_id = 0
    for fold in range(1, 6):
        for subject_offset in range(2):
            subject = f"s{fold}-{subject_offset}"
            for window in range(12):
                rows.append({
                    "sample_id": sample_id,
                    "source": "synthetic",
                    "subject_id": subject,
                    "record_id": f"r-{subject}",
                    "record_group_id": f"g-{subject}",
                    "t_start": float(window * 10),
                    "t_end": float((window + 1) * 10),
                    "outer_fold": fold,
                })
                sample_id += 1
    return pd.DataFrame(rows)


def _synthetic_targets(index: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = index.loc[:, ["sample_id", "subject_id", "record_id"]].copy()
    for offset, pm in enumerate(config["pm_names"]):
        frame[f"target_{pm}"] = (
            np.arange(len(frame), dtype=float) + offset
        ) / len(frame)
    return frame


def _fake_cache(tmp_path: Path, config: dict) -> tuple[Path, pd.DataFrame]:
    cache = tmp_path / "cache"
    cache.mkdir()
    index = _synthetic_index()
    np.save(cache / "features.npy", np.zeros((len(index), 2), dtype=np.float32))
    index.to_parquet(cache / "feature_index.parquet", index=False)
    (cache / "feature_names.json").write_text(
        json.dumps({"feature_names": ["a", "b"]}), encoding="utf-8"
    )
    identity = {
        "cache_schema_version": "cogstate-feature-cache-v1",
        "cache_identity_hash": "cache",
        "feature_hash": "features",
        "sample_id_universe_hash": "samples",
        "raw_preprocessing_hash": "raw",
        "rows": len(index),
        "n_features": 2,
        "dtype": "float32",
    }
    (cache / "feature_materialization_manifest.json").write_text(
        json.dumps({"identity": identity}), encoding="utf-8"
    )
    config["feature_cache_identity"].update(identity)
    config["sequence"]["input_shape"] = [10, 2]
    return cache, index


def test_final_matrix_size_and_unsupported_lstm_regression() -> None:
    matrix = build_training_matrix(_config())
    assert len(matrix) == 5 * 7 * 2 * 4 == 280
    assert matrix.supported.sum() == 245
    assert (~matrix.supported).sum() == 35
    unsupported = matrix.loc[~matrix.supported]
    assert set(unsupported.model) == {"torch_lstm"}
    assert set(unsupported.task_type) == {"regression"}
    with pytest.raises(ValueError, match="classification only"):
        build_model("torch_lstm", "regression", (10, 371), 1, {})


def test_fixed_folds_q3_outer_train_and_common_cohorts() -> None:
    config = _config()
    index = _synthetic_index()
    targets = _synthetic_targets(index, config)
    before = targets.copy(deep=True)
    folds, cohorts, transforms, common = build_metadata_plan(config, index, targets)
    pd.testing.assert_frame_equal(targets, before)
    assert len(folds) == 5
    assert folds.subject_overlap.eq(0).all()
    assert folds.record_group_overlap.eq(0).all()
    assert len(cohorts) == 35
    assert len(transforms) == 35
    for payload in transforms.values():
        assert payload["fit_scope"] == "outer_train_only"
        assert payload["fit_sample_count"] == 96
        assert payload["actual_class_count"] == 3
    assert cohorts.native_test_count.eq(24).all()
    assert cohorts.common_test_count.eq(6).all()
    assert cohorts.sequence_exclusions.eq(18).all()
    assert common.groupby(["fold", "pm"]).size().eq(6).all()


def test_sequence_endpoint_contract_matches_shared_builder() -> None:
    metadata = _synthetic_index().loc[lambda x: x.outer_fold.eq(1)].reset_index(drop=True)
    expected, stats = sequence_endpoint_ids(
        metadata,
        metadata.sample_id,
        length=10,
        stride=1,
        max_gap_seconds=10.01,
    )
    built = build_sequences(
        np.ones((len(metadata), 2), dtype=np.float32),
        np.arange(len(metadata)),
        metadata,
        sequence_length=10,
        stride=1,
        expected_step_seconds=10.0,
        max_gap_seconds=10.01,
    )
    assert expected.tolist() == sorted(built.metadata.target_sample_id.tolist())
    assert stats["sequence_endpoint_count"] == len(built.X)
    for _, group in built.metadata.groupby("sequence_id"):
        assert group.subject_id.nunique() == 1
        assert group.record_id.nunique() == 1


def test_sequence_endpoint_rejects_cross_record_group() -> None:
    metadata = _synthetic_index().iloc[:12].copy()
    metadata.loc[6:, "record_id"] = "second-record"
    with pytest.raises(ValueError, match="record boundary"):
        sequence_endpoint_ids(
            metadata, metadata.sample_id, length=3, stride=1, max_gap_seconds=10.01
        )


def test_feature_cache_identity_and_raw_contract(tmp_path: Path) -> None:
    config = _config()
    cache, _ = _fake_cache(tmp_path, config)
    identity = validate_feature_cache_identity(cache, config["feature_cache_identity"])
    assert identity["n_features"] == 2
    config["feature_cache_identity"]["feature_hash"] = "wrong"
    with pytest.raises(ValueError, match="identity gate"):
        validate_feature_cache_identity(cache, config["feature_cache_identity"])
    original = _config()
    assert original["raw_input"] == {
        "dtype": "float32",
        "channels": 14,
        "sampling_rate_hz": 256,
        "window_seconds": 10,
        "input_shape": [1, 14, 2560],
    }


def test_evaluation_views_share_common_ids_without_second_checkpoint() -> None:
    config = _config()
    training = build_training_matrix(config)
    cohorts = pd.DataFrame([
        {
            "fold": fold,
            "pm": pm,
            "native_test_count": 20,
            "common_test_count": 11,
            "common_sample_id_hash": f"hash-{fold}-{pm}",
        }
        for fold in config["folds"] for pm in config["pm_names"]
    ])
    evaluations = build_evaluation_matrix(training, cohorts)
    assert len(evaluations) == 490
    assert evaluations.requires_new_checkpoint.eq(False).all()
    common = evaluations.loc[evaluations.cohort_type.eq("common_sequence_eligible")]
    assert common.groupby(["fold", "pm", "task_type"])["sample_id_hash"].nunique().eq(1).all()
    lstm_native = evaluations.loc[
        evaluations.model.eq("torch_lstm") & evaluations.cohort_type.eq("native")
    ]
    assert lstm_native.n_samples.eq(11).all()


def test_benchmark_config_keeps_fixed_outer_and_group_inner_validation(tmp_path: Path) -> None:
    config = _config()
    unit = build_training_matrix(config).loc[
        lambda x: x.model.eq("torch_lstm") & x.task_type.eq("classification")
    ].iloc[0].to_dict()
    run = benchmark_run_config(
        config,
        unit,
        data_root=tmp_path,
        feature_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    assert run["evaluation"]["precomputed_fold_column"] == "outer_fold"
    assert run["evaluation"]["folds"] == [1]
    assert run["validation"]["strategy"] == "group_record"
    assert run["validation"]["group_column"] == "record_group_id"
    assert run["sequence"]["max_gap_seconds"] == 10.01


def test_checkpoint_identity_rejects_any_protocol_change(tmp_path: Path) -> None:
    config = _config()
    checkpoint = tmp_path / "model.joblib"
    checkpoint.write_bytes(b"checkpoint")
    unit = build_training_matrix(config).loc[
        lambda x: x.model.eq("random_forest") & x.task_type.eq("classification")
    ].iloc[0].to_dict()
    unit["q3_transform_hash"] = "q3"
    identity = checkpoint_identity(
        config, unit, protocol_hash="protocol", checkpoint=checkpoint
    )
    validate_checkpoint_identity(identity, dict(identity))
    changed = dict(identity)
    changed["q3_transform_hash"] = "other"
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        validate_checkpoint_identity(identity, changed)


def test_plan_is_deterministic_preserves_resume_and_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    cache, index = _fake_cache(tmp_path, config)
    data_root = tmp_path / "data-root"
    target_path = data_root / config["data"]["processed_targets"]
    target_path.parent.mkdir(parents=True)
    targets = _synthetic_targets(index, config).set_index("sample_id")
    targets.to_parquet(target_path)
    preliminary = tmp_path / "preliminary"
    preliminary.mkdir()
    monkeypatch.setattr(
        "bench.experiments.pm_confirmatory_benchmark.BenchmarkRunner.run",
        lambda self: (_ for _ in ()).throw(AssertionError("training called")),
    )
    out = tmp_path / "plan"
    first = write_plan(
        config,
        data_root=data_root,
        feature_cache_dir=cache,
        preliminary_root=preliminary,
        output_dir=out,
    )
    status = pd.read_csv(out / "execution_status.csv")
    status.loc[0, "training_status"] = "completed"
    status.to_csv(out / "execution_status.csv", index=False)
    second = write_plan(
        config,
        data_root=data_root,
        feature_cache_dir=cache,
        preliminary_root=preliminary,
        output_dir=out,
    )
    assert first["protocol_hash"] == second["protocol_hash"]
    assert first["run_matrix_hash"] == second["run_matrix_hash"]
    assert pd.read_csv(out / "execution_status.csv").loc[0, "training_status"] == "completed"
    assert first["training_executed"] is False
    assert first["matrix_cells"] == 280
    assert first["supported_training_units"] == 245


def test_preliminary_sklearn_metrics_without_checkpoint_are_not_reusable(tmp_path: Path) -> None:
    config = _config()
    training = build_training_matrix(config)
    training["q3_transform_hash"] = "q3"
    root = tmp_path / "preliminary"
    root.mkdir()
    audit = audit_preliminary_checkpoints(config, training, root)
    sklearn = audit.loc[audit.model.isin(["random_forest", "xgboost"])]
    assert len(sklearn) == 28
    assert sklearn.reusable.eq(False).all()
    assert sklearn.reason.str.contains("no serialized sklearn checkpoint").all()


def test_tracked_config_contains_no_absolute_local_path() -> None:
    text = Path(
        "experiments/pm_confirmatory/selected_models_5fold_v1.json"
    ).read_text(encoding="utf-8")
    assert "F:\\" not in text and "F:/" not in text
    assert tuple(_config()["models"]) == MODELS
    assert "label_q5" not in text
    assert stable_hash(_config()) == stable_hash(json.loads(text))
