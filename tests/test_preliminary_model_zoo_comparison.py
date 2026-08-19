from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import time

from bench.experiments.preliminary_model_zoo_comparison import (
    aggregate_model_summary,
    benchmark_run_config,
    build_run_status_matrix,
    build_streaming_ranking,
    compatibility_matrix,
    factory_model_names,
    latency_percentiles,
    model_input_family,
    comparison_protocol_hash,
    ResourceSampler,
    validate_shallow_reuse,
)
from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_task import TaskSplit
from bench.features.cogstate_feature_cache import sample_id_universe_hash
from model_zoo import build_model
from model_zoo.factory import TORCH_MODEL_NAMES
from model_zoo.ML.sklearn_models import SKLEARN_MODEL_NAMES


def _config() -> dict:
    return {
        "output_dir": "benchmark_results/preliminary_model_zoo_comparison_fold1",
        "data": {
            "raw_manifest": "data/interim/raw.parquet",
            "logical_recording_map": "data/interim/logical.parquet",
            "processed_targets": "data/processed/targets.parquet",
        },
        "raw_preprocessing": {"resample_hz": 256},
        "sequence": {"length": 10, "stride": 1, "target_position": "last"},
        "model_params": {},
    }


def test_factory_enumeration_and_compatibility_are_current() -> None:
    names = factory_model_names()
    assert names == tuple(sorted(SKLEARN_MODEL_NAMES | TORCH_MODEL_NAMES))
    frame = compatibility_matrix()
    assert frame["model_id"].tolist() == list(names)
    assert frame["model_id"].is_unique
    assert set(frame["input_family"]) == {"raw", "sequence", "features"}


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("torch_eegnet", "raw"),
        ("torch_shallow_convnet", "raw"),
        ("torch_lstm", "sequence"),
        ("torch_bilstm", "sequence"),
        ("torch_transformer", "sequence"),
        ("torch_mlp", "features"),
        ("random_forest", "features"),
    ],
)
def test_input_family_routing(model_id: str, family: str) -> None:
    assert model_input_family(model_id) == family


def test_each_input_family_builds_through_factory() -> None:
    raw = build_model(
        "torch_eegnet", "classification", (1, 2, 128), 3,
        {"sampling_rate": 128, "channel_names": ["C1", "C2"], "device": "cpu"},
    )
    sequence = build_model(
        "torch_lstm", "classification", (3, 8), 3,
        {"hidden_size": 4, "classifier_hidden": 3, "device": "cpu"},
    )
    features = build_model(
        "torch_mlp", "classification", (8,), 3,
        {"hidden_dims": [4], "device": "cpu"},
    )
    assert raw.predict_proba
    assert sequence.predict_proba
    assert features.predict_proba


def test_unsupported_runs_are_isolated_in_status_matrix() -> None:
    status = build_run_status_matrix()
    assert len(status) == len(factory_model_names()) * 14
    assert set(status["status"]) == {"blocked", "unsupported"}
    logistic_regression = status.loc[status.model.eq("logistic_regression")]
    assert logistic_regression.loc[
        logistic_regression.task_type.eq("regression"), "status"
    ].eq("unsupported").all()
    assert status.loc[
        status.model.eq("torch_eegnet") & status.task_type.eq("regression"), "status"
    ].eq("blocked").all()


def test_benchmark_configs_route_raw_sequence_and_features(tmp_path: Path) -> None:
    config = _config()
    raw = benchmark_run_config(
        config, model_id="torch_eegnet", target_id="pm_focus_q3_fold_local",
        output_dir=tmp_path, data_root=tmp_path,
    )
    sequence = benchmark_run_config(
        config, model_id="torch_lstm", target_id="pm_focus_q3_fold_local",
        output_dir=tmp_path, data_root=tmp_path,
    )
    feature = benchmark_run_config(
        config, model_id="random_forest", target_id="pm_focus_q3_fold_local",
        output_dir=tmp_path, data_root=tmp_path,
    )
    assert set(raw["datasets"]) == {"emotiv_raw_eeg"}
    assert set(sequence["datasets"]) == {"cogstate_features"}
    assert "sequence" in sequence and "sequence" not in feature
    assert raw["evaluation"]["folds"] == [1]
    assert raw["validation"]["group_column"] == "record_group_id"


def test_latency_and_summary_aggregation() -> None:
    latency = latency_percentiles([1.0, 2.0, 3.0, 4.0])
    assert latency["p50_ms"] == pytest.approx(2.5)
    assert latency["p95_ms"] > latency["p50_ms"]
    rows = pd.DataFrame(
        [
            {"model": "a", "input_family": "features", "status": "completed", "macro_f1": 0.4, "balanced_accuracy": 0.42, "model_latency_p95_ms": 2.0, "end_to_end_latency_p95_ms": 240.0},
            {"model": "a", "input_family": "features", "status": "completed", "macro_f1": 0.5, "balanced_accuracy": 0.52, "model_latency_p95_ms": 3.0, "end_to_end_latency_p95_ms": 241.0},
            {"model": "b", "input_family": "raw", "status": "failed", "macro_f1": np.nan, "balanced_accuracy": np.nan},
        ]
    )
    summary = aggregate_model_summary(rows, task_type="classification")
    a = summary.set_index("model").loc["a"]
    assert a["mean_macro_f1"] == pytest.approx(0.45)
    assert a["completed_targets"] == 2
    ranking = build_streaming_ranking(summary)
    assert set(["rank_f1", "rank_model_latency", "rank_end_to_end_latency"]).issubset(ranking)


def test_shallow_reuse_requires_identity_and_leakage_checks(tmp_path: Path) -> None:
    root = tmp_path / "shallow"
    root.mkdir()
    manifest = {
        "result_status": "preliminary",
        "evaluation": {"folds": [1], "random_state": 42},
        "composite_audit": {"preprocessing_hash": "raw-hash"},
        "model": {"type": "torch_shallow_convnet"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    targets = [
        *(f"pm_{name}_regression" for name in ("attention", "engagement", "excitement", "stress", "relaxation", "interest", "focus")),
        *(f"pm_{name}_q3_fold_local" for name in ("attention", "engagement", "excitement", "stress", "relaxation", "interest", "focus")),
    ]
    pd.DataFrame(
        {"target_id": targets, "status": "completed", "subject_overlap": 0, "inner_group_overlap": 0}
    ).to_csv(root / "summary.csv", index=False)
    pd.DataFrame([{"target_id": targets[0], "p95_ms": 1.0}]).to_csv(root / "latency.csv", index=False)
    summary, _ = validate_shallow_reuse(root, raw_preprocessing_hash="raw-hash")
    assert len(summary) == 14
    with pytest.raises(ValueError, match="preprocessing_hash"):
        validate_shallow_reuse(root, raw_preprocessing_hash="other")


def test_tracked_comparison_config_contains_no_absolute_local_path() -> None:
    path = Path("experiments/model_zoo/preliminary_model_zoo_comparison_fold1.json")
    text = path.read_text(encoding="utf-8")
    assert "F:/" not in text and "F:\\" not in text


def test_protocol_hash_is_deterministic_and_scientific_config_sensitive() -> None:
    first = _config()
    second = json.loads(json.dumps(first))
    assert comparison_protocol_hash(first) == comparison_protocol_hash(second)
    second["sequence"]["length"] = 11
    assert comparison_protocol_hash(first) != comparison_protocol_hash(second)


def test_resource_sampler_reports_cpu_and_windows_ram() -> None:
    with ResourceSampler(interval_seconds=0.001) as sampler:
        payload = np.ones(2_000_000, dtype=np.float64)
        _ = float(payload.sum())
        time.sleep(0.01)
    result = sampler.result()
    assert result["training_wall_time_s"] > 0
    assert result["process_cpu_time_s"] is not None
    if result["baseline_ram_mb"] is not None:
        assert result["peak_ram_mb"] >= result["baseline_ram_mb"]
        assert result["peak_ram_delta_mb"] >= 0


def test_failure_isolation_records_error_and_keeps_matrix(tmp_path: Path, monkeypatch) -> None:
    import bench.experiments.preliminary_model_zoo_comparison as comparison

    executor = comparison.PreliminaryComparisonExecutor.__new__(
        comparison.PreliminaryComparisonExecutor
    )
    executor.config = _config()
    executor.data_root = tmp_path
    executor.output = tmp_path
    executor.resume = True
    executor.retry_failed = True
    executor.protocol_hash = comparison_protocol_hash(executor.config)
    executor.status = build_run_status_matrix()
    executor.rows = []
    executor.latency_rows = []
    executor.resource_rows = []
    executor.cohort_rows = []
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_hash": executor.protocol_hash}), encoding="utf-8"
    )

    class FailingRunner:
        def __init__(self, config):
            raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(comparison, "BenchmarkRunner", FailingRunner)
    result = executor.run_one(
        "logistic_regression", "pm_excitement_q3_fold_local"
    )

    assert result is None
    row = executor.status.loc[
        executor.status.model.eq("logistic_regression")
        & executor.status.target.eq("pm_excitement_q3_fold_local")
    ].iloc[0]
    assert row.status == "failed"
    assert row.error_type == "RuntimeError"
    assert "synthetic training failure" in row.error_message
    assert len(executor.status) == len(factory_model_names()) * 14


def test_cli_plan_and_execute_modes_are_explicit(tmp_path: Path, monkeypatch) -> None:
    from scripts.run_preliminary_model_zoo_comparison import main

    config = _config()
    config.update(
        schema_version="preliminary-model-zoo-comparison-v1",
        experiment_id="test-plan",
        feature_profile="experiments/features/preliminary_model_zoo_features_v1.json",
        output_dir=str(tmp_path / "out"),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["comparison", "--config", str(config_path), "--plan-only"]
    )
    assert main() == 0
    assert (tmp_path / "out" / "run_status.csv").is_file()
    status_path = tmp_path / "out" / "run_status.csv"
    status = pd.read_csv(status_path)
    status.loc[0, "status"] = "completed"
    status.to_csv(status_path, index=False)
    assert main() == 0
    assert pd.read_csv(status_path).loc[0, "status"] == "completed"

    monkeypatch.setattr("sys.argv", ["comparison", "--config", str(config_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_runner_sequence_context_uses_endpoint_targets_only(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    values = []
    for subject_index, subject in enumerate(("train", "test")):
        for offset in range(12):
            sample_id = subject_index * 100 + offset
            rows.append({
                "sample_id": sample_id,
                "source": "source",
                "subject_id": subject,
                "record_id": f"record-{subject}",
                "record_group_id": f"logical-{subject}",
                "t_start": float(offset * 10),
                "t_end": float((offset + 1) * 10),
                "outer_fold": subject_index + 1,
            })
            values.append([sample_id, offset])
    index = pd.DataFrame(rows)
    np.save(cache / "features.npy", np.asarray(values, dtype=np.float32))
    index.to_parquet(cache / "feature_index.parquet", index=False)
    (cache / "feature_names.json").write_text(
        json.dumps({"feature_names": ["sample", "offset"]}), encoding="utf-8"
    )
    identity = {
        "rows": len(index),
        "n_features": 2,
        "dtype": "float32",
        "sample_id_universe_hash": sample_id_universe_hash(index["sample_id"]),
    }
    (cache / "feature_materialization_manifest.json").write_text(
        json.dumps({
            "schema_version": "cogstate-feature-cache-v1",
            "status": "complete",
            "identity": identity,
        }),
        encoding="utf-8",
    )
    split = TaskSplit(
        X_train=np.asarray([[9, 9], [10, 10], [11, 11]], dtype=np.float32),
        y_train=np.asarray([0, 1, 2]),
        X_test=np.asarray([[109, 9], [110, 10], [111, 11]], dtype=np.float32),
        y_test=np.asarray([2, 1, 0]),
        subject_train=np.asarray(["train"] * 3),
        subject_test=np.asarray(["test"] * 3),
        sample_id_train=np.asarray([9, 10, 11]),
        sample_id_test=np.asarray([109, 110, 111]),
        record_id_train=np.asarray(["record-train"] * 3),
        record_id_test=np.asarray(["record-test"] * 3),
        row_metadata_train={"record_group_id": np.asarray(["logical-train"] * 3)},
        row_metadata_test={"record_group_id": np.asarray(["logical-test"] * 3)},
        feature_names=["sample", "offset"],
        metadata={
            "dataset_metadata": {"feature_cache_path": str(cache)},
            "target_transform_hash": "shared-q3-hash",
            "target_transform": {"transform_hash": "shared-q3-hash"},
            "subject_overlap": [],
        },
    )
    runner = BenchmarkRunner({
        "output_dir": str(tmp_path / "out"),
        "models": {},
        "sequence": {
            "length": 10,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.01,
        },
    })

    sequence = runner._build_sequence_split(split)

    assert sequence.X_train.shape == (3, 10, 2)
    assert sequence.y_train.tolist() == [0, 1, 2]
    assert sequence.X_train[0, :, 0].astype(int).tolist() == list(range(10))
    assert sequence.metadata["target_transform_hash"] == "shared-q3-hash"
    assert sequence.metadata["record_group_overlap"] == []
    assert sequence.metadata["sequence_stats"]["train"]["full_target_count"] == 3
    assert sequence.metadata["sequence_stats"]["train"]["dropped_no_history"] == 0
