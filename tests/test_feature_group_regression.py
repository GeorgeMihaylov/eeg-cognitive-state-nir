from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from bench.bench_runner import BenchmarkRunner
from bench.datasets.base_eeg_data_loader import feature_list_sha256
from bench.experiments.feature_group_ablation import (
    GLOBAL_LABEL_THRESHOLDS,
    prediction_alignment,
    quantize_regression_predictions,
)
from bench.tasks.tasks_registry import get_task
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model


def test_factory_builds_random_forest_regressor() -> None:
    model = build_model(
        model_name="random_forest",
        task_type="regression",
        input_shape=(4,),
        num_outputs=1,
        params={"n_estimators": 3, "random_state": 42},
    )
    assert isinstance(model, RandomForestRegressor)


def test_regression_metrics_preserve_undefined_values() -> None:
    metrics = MetricsCalculator.calculate_all_metrics(
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        task_type="regression",
    )
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["pearson"])
    assert np.isnan(metrics["spearman"])


def test_fixed_regression_quantization_uses_right_closed_global_edges() -> None:
    q20, q40, q60, q80 = GLOBAL_LABEL_THRESHOLDS
    values = np.array([-100.0, q20, np.nextafter(q20, np.inf), q40, q60, q80, 100.0])
    expected = [0, 0, 1, 1, 2, 3, 4]
    assert quantize_regression_predictions(values).tolist() == expected
    shifted_test_distribution = np.concatenate([values, np.full(100, 1000.0)])
    assert quantize_regression_predictions(shifted_test_distribution)[: len(values)].tolist() == expected


def test_cross_task_alignment_ignores_different_target_scales() -> None:
    base = pd.DataFrame({
        "sample_id": [1, 2],
        "fold": [1, 2],
        "subject_id": ["s1", "s2"],
        "record_id": ["r1", "r2"],
        "source": ["a", "b"],
        "y_true": [0, 4],
    })
    regression = base.copy()
    regression["y_true"] = [0.2, 0.8]
    assert prediction_alignment(base, regression, compare_target=False)["exact_match"]


def test_runner_executes_groupkfold_regression_and_standard_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(42)
    subjects = np.repeat([f"s{i}" for i in range(6)], 20)
    target = rng.uniform(0.05, 0.95, len(subjects))
    data = pd.DataFrame({
        "subject_id": subjects,
        "record_id": [f"r-{subject}" for subject in subjects],
        "source": np.where(np.arange(len(subjects)) % 2, "Old_EEG", "gpn_data"),
        "t_start": np.tile(np.arange(20) * 10.0, 6),
        "t_end": np.tile(np.arange(1, 21) * 10.0, 6),
        "target_focus": target,
        "label_q5": np.searchsorted(GLOBAL_LABEL_THRESHOLDS, target, side="left"),
        "EEG.AF3__mean": target + rng.normal(0, 0.05, len(target)),
        "EEG.AF3__std": rng.normal(size=len(target)),
        "POW.AF3.Alpha__mean": target + rng.normal(0, 0.08, len(target)),
        "POW.AF3.Alpha__std": rng.normal(size=len(target)),
    })
    data_path = tmp_path / "synthetic.parquet"
    data.to_parquet(data_path, index=False)
    features = [
        "EEG.AF3__mean", "EEG.AF3__std",
        "POW.AF3.Alpha__mean", "POW.AF3.Alpha__std",
    ]
    output = tmp_path / "results"
    config = {
        "output_dir": str(output),
        "datasets": {
            "emotiv_cognitive": {
                "data_path": str(data_path),
                "feature_set": "eeg_pow",
                "target_col": "target_focus",
                "subject_col": "subject_id",
                "discretize": False,
                "max_features": 4,
                "expected_feature_count": 4,
                "feature_list_sha256": feature_list_sha256(features),
            }
        },
        "tasks": ["focus_regression"],
        "models": {
            "random_forest_regressor": {
                "type": "random_forest",
                "task_type": "regression",
                "params": {
                    "n_estimators": 5,
                    "max_depth": 4,
                    "random_state": 42,
                    "n_jobs": 1,
                },
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": 3,
            "group_column": "subject_id",
            "random_state": 42,
        },
        "run_within_subject": False,
        "run_loso": False,
    }
    runner = BenchmarkRunner(config)
    summary = runner.run()
    completed = runner.completed_run()

    assert summary.loc[0, "mae"] >= 0
    assert completed.manifest_file.is_file()
    predictions = list(output.rglob("predictions.parquet"))
    assert len(predictions) == 4  # three folds plus one unified artifact
    unified = pd.read_parquet(
        next(path for path in predictions if path.parent.name == "group_kfold_subject")
    )
    assert len(unified) == len(data)
    assert unified["sample_id"].is_unique
    assert np.issubdtype(unified["y_pred"].dtype, np.floating)
    importance = list(output.rglob("feature_importance.parquet"))
    manifests = list(output.rglob("feature_manifest.json"))
    assert len(importance) == 3
    assert len(manifests) == 3
    assert all(len(pd.read_parquet(path)) == 4 for path in importance)
    assert all(json.loads(path.read_text())["feature_list_sha256"] == feature_list_sha256(features) for path in manifests)


def test_regression_task_registry_accepts_continuous_targets() -> None:
    from bench.core.abstract_dataset import EEGData

    data = EEGData(
        data=np.arange(20, dtype=float).reshape(10, 2),
        labels=np.linspace(0.1, 0.9, 10),
        subject_ids=np.repeat(["a", "b"], 5),
    )
    task = get_task("focus_regression", data, {"random_state": 42})
    assert task.task_type == "regression"
    assert task.name == "focus_regression"
