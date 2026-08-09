from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.preliminary_model_zoo_comparison import (
    aggregate_model_summary,
    benchmark_run_config,
    build_run_status_matrix,
    build_streaming_ranking,
    compatibility_matrix,
    factory_model_names,
    latency_percentiles,
    model_input_family,
    validate_shallow_reuse,
)
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
    ].eq("unsupported").all()


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
