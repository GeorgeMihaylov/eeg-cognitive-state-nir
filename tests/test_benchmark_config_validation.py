from __future__ import annotations

from pathlib import Path

import pytest

from bench.bench_runner import BenchmarkRunner
from cli import create_default_config, validate_config


def _valid_config(tmp_path: Path) -> dict:
    data_path = tmp_path / "dataset.parquet"
    data_path.touch()
    return {
        "output_dir": str(tmp_path / "results"),
        "datasets": {
            "emotiv_cognitive": {
                "data_path": str(data_path),
                "feature_set": "pow_plus_eeg",
                "n_classes": 5,
            }
        },
        "tasks": ["cognitive_load_5class"],
        "models": {
            "rf": {
                "type": "random_forest",
                "task_type": "classification",
                "params": {"random_state": 42},
            }
        },
        "task_config": {"random_state": 42, "n_splits": 5},
    }


def test_default_config_uses_explicit_five_class_task() -> None:
    config = create_default_config()

    assert config["tasks"] == ["cognitive_load_5class"]
    assert config["datasets"]["emotiv_cognitive"]["n_classes"] == 5
    assert (
        config["datasets"]["emotiv_cognitive"]["data_path"]
        == "./data/processed/windowed_eeg_pm_dataset_w10.parquet"
    )


def test_validate_config_accepts_registered_task_and_model(
    tmp_path: Path,
) -> None:
    assert validate_config(_valid_config(tmp_path))


def test_validate_config_rejects_unknown_task(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    config["tasks"] = ["unknown_task"]

    assert not validate_config(config)


def test_validate_config_rejects_unknown_model(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    config["models"]["rf"]["type"] = "unknown_model"

    assert not validate_config(config)


def test_validate_config_rejects_task_model_type_mismatch(
    tmp_path: Path,
) -> None:
    config = _valid_config(tmp_path)
    config["models"]["rf"]["task_type"] = "regression"

    assert not validate_config(config)


def test_validate_config_rejects_torch_regression_until_supported(
    tmp_path: Path,
) -> None:
    config = _valid_config(tmp_path)
    config["tasks"] = ["focus_regression"]
    config["models"] = {
        "transformer": {
            "type": "torch_transformer",
            "task_type": "regression",
            "params": {},
        }
    }

    assert not validate_config(config)


def test_runner_requires_explicit_tasks() -> None:
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.config = {}
    runner.timestamp = "test"
    runner.load_dataset = lambda _: object()

    with pytest.raises(
        ValueError,
        match="must define at least one explicit task",
    ):
        runner.run_for_dataset("dummy")
