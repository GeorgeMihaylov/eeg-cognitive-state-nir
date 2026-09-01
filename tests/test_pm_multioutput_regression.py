from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from bench.bench_runner import BenchmarkRunner
from bench.datasets.emotiv_loader import EmotivDataset
from bench.tasks.tasks_registry import get_task
from cli import validate_config
from cogstate.model_zoo import build_model


TARGET_NAMES = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]


def _write_dataset(path: Path, *, n_rows: int = 30) -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "record_id": [f"r{index // 3}" for index in range(n_rows)],
        "source": ["gpn_data"] * n_rows,
        "subject_id": [f"s{index // 6}" for index in range(n_rows)],
        "EEG.AF3__mean": rng.normal(size=n_rows),
        "EEG.AF4__std": rng.normal(size=n_rows),
        "POW.AF3.Alpha__mean": rng.normal(size=n_rows),
        "POW.AF4.BetaL__std": rng.normal(size=n_rows),
        "PM.Focus.Scaled__mean": rng.uniform(size=n_rows),
        **{
            target_name: rng.uniform(size=n_rows)
            for target_name in TARGET_NAMES
        },
    })
    frame.loc[0, "target_stress"] = np.nan
    frame.to_parquet(path, index=False)


def _dataset_config(path: Path) -> dict:
    return {
        "data_path": str(path),
        "feature_set": "pow_plus_eeg",
        "target_cols": TARGET_NAMES,
        "n_outputs": 7,
        "task_type": "regression",
        "discretize": False,
        "max_features": 500,
    }


def test_loader_preserves_seven_targets_and_complete_cases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pm.parquet"
    _write_dataset(path)

    data = EmotivDataset(_dataset_config(path)).load()

    assert data.labels.shape == (29, 7)
    assert data.metadata["target_cols"] == TARGET_NAMES
    assert data.metadata["n_samples_before_target_filter"] == 30
    assert data.metadata["n_samples_after_target_filter"] == 29
    assert data.metadata["dropped_target_rows"] == 1
    assert data.feature_names == [
        "EEG.AF3__mean",
        "EEG.AF4__std",
        "POW.AF3.Alpha__mean",
        "POW.AF4.BetaL__std",
    ]
    assert not any(
        name.startswith(("PM.", "target_")) for name in data.feature_names
    )


def test_loader_rejects_ambiguous_or_missing_targets(tmp_path: Path) -> None:
    path = tmp_path / "pm.parquet"
    _write_dataset(path)
    ambiguous = {
        **_dataset_config(path),
        "target_col": "target_focus",
    }
    with pytest.raises(ValueError, match="either 'target_col' or 'target_cols'"):
        EmotivDataset(ambiguous)

    missing = _dataset_config(path)
    missing["target_cols"] = [*TARGET_NAMES[:-1], "target_missing"]
    with pytest.raises(ValueError, match="absent from the dataset"):
        EmotivDataset(missing).load()


def test_task_and_group_split_preserve_multioutput_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pm.parquet"
    _write_dataset(path)
    data = EmotivDataset(_dataset_config(path)).load()

    task = get_task(
        "performance_metrics_regression",
        data,
        {"n_outputs": 7, "random_state": 42, "n_splits": 5},
    )
    from bench.validation.cross_val import CrossValidator

    split = CrossValidator(task).run_group_kfold(
        group_column="subject_id",
        n_splits=5,
    )["fold_01"]

    assert task.n_outputs == 7
    assert split.y_train.ndim == split.y_test.ndim == 2
    assert split.y_train.shape[1] == split.y_test.shape[1] == 7
    assert not set(split.subject_train) & set(split.subject_test)
    assert split.metadata["target_names"] == TARGET_NAMES


def test_regression_models_support_seven_outputs() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 8))
    y = rng.normal(size=(60, 7))

    mean = build_model("mean_regressor", "regression", (8,), 7, {})
    forest = build_model(
        "random_forest",
        "regression",
        (8,),
        7,
        {"n_estimators": 3, "random_state": 42},
    )
    assert isinstance(mean, DummyRegressor)
    assert isinstance(forest, RandomForestRegressor)
    assert mean.fit(X, y).predict(X[:4]).shape == (4, 7)
    assert forest.fit(X, y).predict(X[:4]).shape == (4, 7)


def test_torch_mlp_multioutput_regression_and_classification_compatibility() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    y_regression = rng.normal(size=(50, 7)).astype(np.float32)
    regression = build_model(
        "torch_mlp",
        "regression",
        (8,),
        7,
        {
            "hidden_dims": [12],
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "validation_size": 0.2,
        },
    )
    regression.fit(X, y_regression)
    assert regression.predict(X[:5]).shape == (5, 7)
    with pytest.raises(AttributeError, match="unavailable for regression"):
        regression.predict_proba(X[:5])

    classification = build_model(
        "torch_mlp",
        "classification",
        (8,),
        3,
        {
            "hidden_dims": [12],
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "validation_size": 0.2,
        },
    )
    labels = np.tile(np.arange(3), 20)[:50]
    classification.fit(X, labels)
    assert classification.predict(X[:5]).shape == (5,)
    assert classification.predict_proba(X[:5]).shape == (5, 3)


def test_small_runner_writes_multioutput_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "pm.parquet"
    _write_dataset(path)
    config = {
        "output_dir": str(tmp_path / "results"),
        "datasets": {"emotiv_pm_regression": _dataset_config(path)},
        "tasks": ["performance_metrics_regression"],
        "models": {
            "mean": {
                "type": "mean_regressor",
                "task_type": "regression",
                "params": {},
            }
        },
        "task_config": {"n_outputs": 7, "n_splits": 5, "random_state": 42},
        "evaluation": {
            "protocol": "group_kfold_subject",
            "group_column": "subject_id",
            "n_splits": 5,
            "folds": [1],
            "random_state": 42,
        },
    }

    runner = BenchmarkRunner(config)
    summary = runner.run()
    protocol = runner.results["emotiv_pm_regression"]["models"][
        "performance_metrics_regression"
    ]["mean"]["group_kfold_subject"]
    fold = protocol["folds"]["fold_01"]

    assert summary.loc[0, "n_outputs"] == 7
    assert Path(fold["artifacts"]["per_target_metrics"]).is_file()
    assert Path(fold["artifacts"]["subject_target_predictions"]).is_file()
    predictions = pd.read_parquet(fold["artifacts"]["predictions"])
    assert predictions.filter(like="y_true_").shape[1] == 7
    assert predictions.filter(like="y_pred_").shape[1] == 7
    assert "accuracy" not in fold["metrics"]


def test_pm_smoke_config_validates_and_root_config_stays_classification() -> None:
    smoke = yaml.safe_load(
        Path("experiments/pm_regression/pm_regression_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    canonical = yaml.safe_load(Path("configs.yaml").read_text(encoding="utf-8"))

    assert validate_config(smoke)
    assert canonical["tasks"] == ["cognitive_load_5class"]
    assert canonical["datasets"]["emotiv_cognitive"]["target_col"] == "label_q5"
