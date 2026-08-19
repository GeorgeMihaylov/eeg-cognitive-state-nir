from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bench.bench_runner import BenchmarkRunner
from model_zoo import build_model
from model_zoo.DL.eegnet import TorchEEGNetClassifier


def _small_eegnet() -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=4,
        n_times=64,
        num_classes=5,
        temporal_kernel_samples=16,
        separable_kernel_samples=8,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.1,
    )


def test_eegnet_forward_pass_and_shape_errors():
    model = _small_eegnet()
    logits = model(torch.zeros(3, 1, 4, 64))
    assert logits.shape == (3, 5)


def test_factory_builds_torch_eegnet_and_fits_numpy_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 1, 4, 64)).astype(np.float32)
    y = np.tile(np.arange(5), 6)
    model = build_model(
        model_name="torch_eegnet",
        task_type="classification",
        input_shape=(1, 4, 64),
        num_outputs=5,
        params={
            "sampling_rate": 64,
            "temporal_kernel_seconds": 0.25,
            "separable_kernel_seconds": 0.125,
            "f1": 2,
            "depth_multiplier": 1,
            "f2": 2,
            "pool1": 2,
            "pool2": 2,
            "dropout": 0.1,
            "batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "validation_size": 0.2,
            "device": "cpu",
            "random_state": 42,
        },
    )
    model.fit(X, y)
    probabilities = model.predict_proba(X[:7])

    assert probabilities.shape == (7, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.get_training_summary()["input_shape"] == [1, 4, 64]


def test_factory_builds_torch_eegnet_for_scalar_regression():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(24, 1, 4, 64)).astype(np.float32)
    y = rng.normal(size=24).astype(np.float32)
    model = build_model(
        model_name="torch_eegnet",
        task_type="regression",
        input_shape=(1, 4, 64),
        num_outputs=1,
        params={
            "sampling_rate": 64,
            "temporal_kernel_seconds": 0.25,
            "separable_kernel_seconds": 0.125,
            "f1": 2,
            "depth_multiplier": 1,
            "f2": 2,
            "pool1": 2,
            "pool2": 2,
            "dropout": 0.1,
            "batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "validation_size": 0.2,
            "device": "cpu",
            "random_state": 42,
        },
    )
    model.fit(X, y)
    prediction = model.predict(X[:7])

    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert model.task_type == "regression"


def test_runner_executes_small_lazy_raw_eeg_smoke(tmp_path):
    rng = np.random.default_rng(7)
    n_subjects = 20
    windows_per_subject = 2
    n_windows = n_subjects * windows_per_subject
    array = rng.normal(size=(n_windows, 4, 64)).astype(np.float32)
    cache_path = tmp_path / "raw_windows.npy"
    np.save(cache_path, array)
    subject_ids = np.repeat([f"S{index:02d}" for index in range(n_subjects)], 2)
    labels = np.repeat(np.arange(n_subjects) % 5, 2)
    outer_fold = np.repeat(np.arange(n_subjects) % 5 + 1, 2)
    manifest = pd.DataFrame(
        {
            "sample_id": np.arange(n_windows),
            "source": "synthetic",
            "subject_id": subject_ids,
            "record_id": np.repeat([f"R{index:02d}" for index in range(n_subjects)], 2),
            "record_group_id": np.repeat(
                [f"R{index:02d}" for index in range(n_subjects)], 2
            ),
            "raw_file_path": "synthetic",
            "t_start": np.tile([0.0, 1.0], n_subjects),
            "t_end": np.tile([1.0, 2.0], n_subjects),
            "label_q5": labels,
            "sfreq_original": 64.0,
            "sfreq_target": 64.0,
            "n_channels": 4,
            "n_samples_expected": 64,
            "outer_fold": outer_fold,
            "status": "ok",
            "rejection_reason": "",
            "cache_file": str(cache_path),
            "cache_offset": np.arange(n_windows),
            "missing_fraction": 0.0,
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    output_dir = tmp_path / "results"
    config = {
        "output_dir": str(output_dir),
        "datasets": {
            "emotiv_raw_eeg": {"data_path": str(manifest_path)}
        },
        "tasks": ["cognitive_load_5class"],
        "models": {
            "tiny_eegnet": {
                "type": "torch_eegnet",
                "task_type": "classification",
                "params": {
                    "temporal_kernel_seconds": 0.25,
                    "separable_kernel_seconds": 0.125,
                    "f1": 2,
                    "depth_multiplier": 1,
                    "f2": 2,
                    "pool1": 2,
                    "pool2": 2,
                    "dropout": 0.1,
                    "batch_size": 8,
                    "max_epochs": 1,
                    "early_stopping_patience": 1,
                    "validation_size": 0.2,
                    "device": "cpu",
                    "random_state": 42,
                },
            }
        },
        "validation": {
            "strategy": "group_record",
            "group_column": "record_group_id",
            "validation_size": 0.2,
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
        "task_config": {"random_state": 42},
        "run_within_subject": False,
        "run_loso": False,
    }

    runner = BenchmarkRunner(config)
    summary = runner.run()

    assert len(summary) == 1
    assert summary.iloc[0]["n_folds"] == 1
    result = runner.results["emotiv_raw_eeg"]["models"]["cognitive_load_5class"]["tiny_eegnet"]["group_kfold_subject"]
    artifacts = result["folds"]["fold_01"]["artifacts"]
    for name in (
        "model", "training_log", "predictions", "validation_split",
        "raw_eeg_stats", "normalization_stats",
    ):
        assert Path(artifacts[name]).exists()
