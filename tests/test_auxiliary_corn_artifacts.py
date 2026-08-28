from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_task import TaskSplit
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    labels = np.tile(np.arange(5), 12).astype(np.int64)
    features = rng.normal(size=(len(labels), 8, 6)).astype(np.float32)
    return features, labels


def _adapter():
    return build_model(
        "torch_transformer",
        "classification",
        (8, 6),
        5,
        {
            "head_type": "categorical_corn",
            "auxiliary_weight": 0.5,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 64,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )


def test_writer_keeps_primary_probabilities_and_adds_auxiliary_fields(
    tmp_path: Path,
) -> None:
    features, labels = _data()
    adapter = _adapter().fit(features, labels)
    n_test = 10
    split = TaskSplit(
        X_train=features[n_test:],
        y_train=labels[n_test:],
        X_test=features[:n_test],
        y_test=labels[:n_test],
        subject_train=np.asarray(["train"] * (len(labels) - n_test)),
        subject_test=np.asarray(["test"] * n_test),
        feature_names=[f"f{i}" for i in range(6)],
        sample_id_train=np.arange(n_test, len(labels)),
        sample_id_test=np.arange(n_test),
        record_id_train=np.asarray(["r-train"] * (len(labels) - n_test)),
        record_id_test=np.asarray(["r-test"] * n_test),
        row_metadata_test={"source": np.asarray(["synthetic"] * n_test)},
        metadata={"split_type": "synthetic"},
    )
    detailed = adapter.predict_detailed(split.X_test)
    metrics = MetricsCalculator.calculate_all_metrics(
        split.y_test,
        detailed["y_pred"],
        detailed["class_probabilities"],
        expected_rank=detailed["categorical_expected_rank"],
    )
    runner = BenchmarkRunner({"output_dir": str(tmp_path), "models": {}})
    artifacts = runner._save_split_artifacts(
        model=adapter,
        split=split,
        y_pred=detailed["y_pred"],
        y_proba=detailed["class_probabilities"],
        dataset_name="synthetic",
        task_name="cognitive_load_5class",
        model_name="transformer",
        artifact_split_name="test",
        metrics=metrics,
        detailed_predictions=detailed,
    )
    frame = pd.read_parquet(artifacts["predictions"])
    required = {
        "head_type",
        "categorical_expected_rank",
        "aux_expected_rank",
        "aux_ordinal_prediction",
        "aux_ordinal_argmax",
        "auxiliary_weight",
        *{f"class_probability_{i}" for i in range(5)},
        *{f"aux_class_probability_{i}" for i in range(5)},
        *{f"aux_threshold_probability_{i}" for i in range(4)},
        *{f"aux_threshold_logit_{i}" for i in range(4)},
    }
    assert required <= set(frame)
    assert frame["head_type"].eq("categorical_corn").all()
    for index in range(5):
        np.testing.assert_allclose(
            frame[f"proba_{index}"],
            frame[f"class_probability_{index}"],
            atol=0,
            rtol=0,
        )
    metadata = json.loads(
        Path(artifacts["auxiliary_corn_metadata"]).read_text(encoding="utf-8")
    )
    assert metadata["head_type"] == "categorical_corn"
    assert metadata["auxiliary_weight"] == 0.5
    assert metadata["maximum_primary_probability_row_sum_error"] <= 1e-6
    assert metadata["maximum_auxiliary_probability_row_sum_error"] <= 1e-6
    assert metadata["maximum_auxiliary_monotonicity_violation"] == 0.0


def test_runner_metrics_use_primary_head_and_prefix_auxiliary_metrics(
    tmp_path: Path,
) -> None:
    features, labels = _data()
    adapter = _adapter()
    split = TaskSplit(
        X_train=features[10:],
        y_train=labels[10:],
        X_test=features[:10],
        y_test=labels[:10],
        subject_train=np.asarray(["train"] * (len(labels) - 10)),
        subject_test=np.asarray(["test"] * 10),
        feature_names=[f"f{i}" for i in range(6)],
        sample_id_train=np.arange(10, len(labels)),
        sample_id_test=np.arange(10),
        record_id_train=np.asarray(["r-train"] * (len(labels) - 10)),
        record_id_test=np.asarray(["r-test"] * 10),
        row_metadata_test={"source": np.asarray(["synthetic"] * 10)},
        metadata={"split_type": "synthetic"},
    )
    runner = BenchmarkRunner({"output_dir": str(tmp_path), "models": {}})
    result = runner._evaluate_split(
        adapter,
        split,
        "transformer",
        dataset_name="synthetic",
        task_name="cognitive_load_5class",
        artifact_split_name="test",
    )
    assert "balanced_accuracy" in result["metrics"]
    assert "aux_balanced_accuracy" in result["metrics"]
    assert "categorical_aux_prediction_agreement" in result["metrics"]
