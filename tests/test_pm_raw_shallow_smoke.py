from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_dataset import EEGData


def _raw_pm_data(target_id: str) -> EEGData:
    n_subjects = 10
    rows_per_subject = 12
    n_rows = n_subjects * rows_per_subject
    subjects = np.repeat(
        [f"S{index:02d}" for index in range(n_subjects)], rows_per_subject
    )
    subject_index = np.repeat(np.arange(n_subjects), rows_per_subject)
    position = np.tile(np.arange(rows_per_subject), n_subjects)
    records = np.asarray(
        [f"{subject}_R{index // 6}" for subject, index in zip(subjects, position)]
    )
    base = np.tile(np.linspace(0.0, 1.0, 6), n_subjects * 2)
    labels = (base + subject_index * 0.01).astype(np.float32)
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(n_rows, 1, 2, 64)).astype(np.float32)
    windows[:, 0, 0] += labels[:, None] * 0.05
    source_target = target_id.replace("pm_", "target_").split("_q3")[0]
    if target_id.endswith("_regression"):
        source_target = target_id.removeprefix("pm_").removesuffix("_regression")
        source_target = f"target_{source_target}"
    return EEGData(
        data=windows,
        labels=labels,
        subject_ids=subjects,
        sample_ids=np.arange(20_000, 20_000 + n_rows),
        record_ids=records,
        row_metadata={
            "record_group_id": records,
            "outer_fold": subject_index // 2 + 1,
            "source": np.full(n_rows, "synthetic", dtype=object),
        },
        metadata={
            "target_id": target_id,
            "target_cols": [source_target],
            "task_type": (
                "classification" if "_q3_" in target_id else "regression"
            ),
            "observation_unit": "raw_window",
        },
    )


@pytest.mark.parametrize(
    ("metric", "task_type"),
    [
        ("attention", "regression"),
        ("attention", "classification"),
        ("stress", "regression"),
        ("stress", "classification"),
    ],
)
@patch("bench.bench_runner.get_dataset")
def test_one_fold_pm_shallow_execution_path(
    mock_get_dataset: Mock,
    metric: str,
    task_type: str,
    tmp_path: Path,
) -> None:
    target_id = (
        f"pm_{metric}_regression"
        if task_type == "regression"
        else f"pm_{metric}_q3_fold_local"
    )
    dataset = Mock()
    dataset.load.return_value = _raw_pm_data(target_id)
    mock_get_dataset.return_value = dataset
    output_dir = tmp_path / f"{metric}_{task_type}"
    runner = BenchmarkRunner(
        {
            "output_dir": str(output_dir),
            "datasets": {"synthetic_raw_pm": {"data_path": "unused.parquet"}},
            "tasks": [target_id],
            "task_config": {"target_id": target_id},
            "models": {
                "torch_shallow_convnet": {
                    "type": "torch_shallow_convnet",
                    "task_type": task_type,
                    "params": {
                        "n_filters": 2,
                        "temporal_kernel_samples": 5,
                        "pool_size": 8,
                        "pool_stride": 4,
                        "dropout": 0.1,
                        "batch_size": 16,
                        "max_epochs": 1,
                        "validation_size": 0.25,
                        "early_stopping_patience": 1,
                        "device": "cpu",
                        "random_state": 42,
                    },
                }
            },
            "validation": {
                "strategy": "group_record",
                "group_column": "record_group_id",
                "fraction": 0.25,
                "random_state": 42,
            },
            "evaluation": {
                "protocol": "group_kfold_subject",
                "n_splits": 5,
                "group_column": "subject_id",
                "precomputed_fold_column": "outer_fold",
                "folds": [1],
                "random_state": 42,
            },
        }
    )
    runner.run()
    fold = runner.results["synthetic_raw_pm"]["models"][target_id][
        "torch_shallow_convnet"
    ]["group_kfold_subject"]["folds"]["fold_01"]
    predictions = pd.read_parquet(fold["artifacts"]["predictions"])
    assert len(predictions) == 24
    assert np.isfinite(predictions["y_pred"]).all()
    assert fold["split_metadata"]["subject_overlap"] == []
    assert fold["training"]["task_type"] == task_type
    assert fold["artifacts"]["model"].endswith("model.pt")
    if task_type == "classification":
        assert set(predictions["y_true"]).issubset({0, 1, 2})
        assert Path(fold["artifacts"]["target_transform"]).is_file()
        assert {"proba_0", "proba_1", "proba_2"}.issubset(predictions.columns)
    else:
        assert "target_transform" not in fold["artifacts"]
