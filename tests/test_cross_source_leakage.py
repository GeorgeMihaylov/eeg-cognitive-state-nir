import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from bench.bench_runner import BenchmarkRunner
from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter
from tests.test_cross_source_split import make_cross_source_data


def transformer_config(output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "datasets": {"synthetic": {"data_path": "unused.parquet"}},
        "tasks": ["cognitive_load_5class"],
        "sequence": {
            "length": 2,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.5,
        },
        "validation": {
            "strategy": "group_record",
            "group_column": "subject_id",
            "validation_size": 0.2,
            "random_state": 42,
        },
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "sequence_length": 2,
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "dropout": 0.0,
                    "activation": "relu",
                    "pooling": "last",
                    "positional_encoding": "learned",
                    "batch_size": 32,
                    "max_epochs": 1,
                    "learning_rate": 0.002,
                    "validation_size": 0.2,
                    "early_stopping_patience": 1,
                    "device": "cpu",
                    "random_state": 42,
                    "standardize": True,
                },
            }
        },
        "evaluation": {
            "protocol": "cross_source_holdout",
            "train_source": "gpn_data",
            "test_source": "Old_EEG",
            "subject_mode": "source_exclusive",
            "remove_logical_duplicates": True,
            "thresholds": {
                "minimum_train_subjects": 5,
                "minimum_test_subjects": 3,
                "minimum_train_classes": 5,
                "minimum_test_classes": 2,
                "minimum_predictions_per_test_subject": 20,
            },
        },
        "task_config": {"random_state": 42},
        "run_within_subject": False,
        "run_loso": False,
    }


@patch("bench.bench_runner.get_dataset")
def test_target_source_never_reaches_normalization_or_early_stopping(
    get_dataset_mock,
    tmp_path,
    monkeypatch,
):
    data = make_cross_source_data()
    dataset = Mock()
    dataset.load.return_value = data
    get_dataset_mock.return_value = dataset
    captured_ids = []
    original = TorchClassificationAdapter._fit_standardizer

    def capture(self, features):
        captured_ids.extend(np.asarray(features)[..., 0].ravel().astype(int).tolist())
        return original(self, features)

    monkeypatch.setattr(TorchClassificationAdapter, "_fit_standardizer", capture)
    runner = BenchmarkRunner(transformer_config(tmp_path))
    runner.run()

    test_ids = set(data.sample_ids[np.asarray(data.row_metadata["source"]) == "Old_EEG"])
    assert set(captured_ids).isdisjoint(test_ids)
    result = runner.results["synthetic"]["models"]["cognitive_load_5class"][
        "torch_transformer"
    ]["cross_source_holdout"]
    split_result = next(iter(result["splits"].values()))
    validation_path = Path(split_result["artifacts"]["validation_split"])
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert set(validation["inner_train_subject_ids"]).isdisjoint(
        validation["inner_validation_subject_ids"]
    )
    assert set(validation["inner_train_subject_ids"]).isdisjoint({"O0", "O1", "O2"})
    assert set(validation["inner_validation_subject_ids"]).isdisjoint({"O0", "O1", "O2"})


@patch("bench.bench_runner.get_dataset")
def test_runner_uses_source_pure_sequences_and_standard_artifacts(
    get_dataset_mock,
    tmp_path,
):
    dataset = Mock()
    dataset.load.return_value = make_cross_source_data()
    get_dataset_mock.return_value = dataset
    runner = BenchmarkRunner(transformer_config(tmp_path))
    summary = runner.run()

    assert len(summary) == 1
    result = runner.results["synthetic"]["models"]["cognitive_load_5class"][
        "torch_transformer"
    ]["cross_source_holdout"]
    predictions = pd.read_parquet(result["artifacts"]["predictions"])
    assert set(predictions["source"]) == {"Old_EEG"}
    assert predictions["sequence_id"].is_unique
    assert result["split_metadata"]["logical_record_overlap"] == []
    artifacts = next(iter(result["splits"].values()))["artifacts"]
    required = {
        "predictions", "metrics", "model", "training_log",
        "validation_split", "normalization_stats", "cross_source_split",
        "excluded_subjects", "excluded_logical_recordings",
        "source_distribution",
    }
    assert required.issubset(artifacts)
    assert all(Path(artifacts[name]).is_file() for name in required)


def test_direction_changes_benchmark_config_hash(tmp_path):
    forward = transformer_config(tmp_path / "forward")
    reverse = transformer_config(tmp_path / "reverse")
    reverse["evaluation"].update({
        "train_source": "Old_EEG",
        "test_source": "gpn_data",
    })

    assert BenchmarkRunner.config_hash_for(forward) != BenchmarkRunner.config_hash_for(reverse)


def test_cross_source_metrics_include_severe_error_rate():
    from bench.validation.metrics import MetricsCalculator

    metrics = MetricsCalculator.calculate_all_metrics(
        np.asarray([0, 1, 4]),
        np.asarray([3, 1, 0]),
    )

    assert metrics["severe_error_rate"] == 2 / 3
