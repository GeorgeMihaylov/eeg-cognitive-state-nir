from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_task import TaskSplit
from bench.experiments.user_calibration import UserCalibrationExperiment
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model


def _synthetic_sequences() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(5), 10).astype(np.int64)
    features = rng.normal(size=(len(labels), 8, 6)).astype(np.float32)
    features[:, :, 0] += labels[:, None] * 0.5
    return features, labels


def _adapter(head_type: str):
    return build_model(
        "torch_transformer",
        "classification",
        input_shape=(8, 6),
        num_outputs=5,
        params={
            "head_type": head_type,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            # Forty training rows after the inner split fit in one optimizer step.
            "batch_size": 64,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )


@pytest.mark.parametrize("head_type", ["categorical", "coral", "corn"])
def test_adapter_synthetic_fit_predict_and_probabilities(head_type: str) -> None:
    features, labels = _synthetic_sequences()
    adapter = _adapter(head_type)
    adapter.fit(features, labels)
    detailed = adapter.predict_detailed(features[:12])
    probabilities = adapter.predict_proba(features[:12])
    predictions = adapter.predict(features[:12])

    assert probabilities.shape == (12, 5)
    assert predictions.shape == (12,)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(predictions).all()
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(predictions, detailed["y_pred"])
    np.testing.assert_allclose(
        probabilities,
        detailed["class_probabilities"],
        atol=0,
        rtol=0,
    )
    assert adapter.training_log_[0]["validation_loss"] >= 0
    if head_type == "categorical":
        np.testing.assert_array_equal(predictions, probabilities.argmax(axis=1))
        assert "threshold_probabilities" not in detailed
    else:
        assert detailed["threshold_probabilities"].shape == (12, 4)
        assert detailed["expected_rank"].shape == (12,)
        assert detailed["ordinal_argmax"].shape == (12,)
        expected_predictions = (
            detailed["threshold_probabilities"] >= 0.5
        ).sum(axis=1)
        np.testing.assert_array_equal(predictions, expected_predictions)


def test_ordinal_checkpoint_rejects_categorical_loader(tmp_path: Path) -> None:
    features, labels = _synthetic_sequences()
    ordinal = _adapter("coral").fit(features, labels)
    checkpoint = tmp_path / "ordinal.pt"
    ordinal.save(checkpoint)
    categorical = _adapter("categorical")
    with pytest.raises(ValueError, match="Checkpoint head_type"):
        categorical.load(checkpoint)


def test_categorical_checkpoint_rejects_ordinal_loader(tmp_path: Path) -> None:
    features, labels = _synthetic_sequences()
    categorical = _adapter("categorical").fit(features, labels)
    checkpoint = tmp_path / "categorical.pt"
    categorical.save(checkpoint)
    ordinal = _adapter("corn")
    with pytest.raises(ValueError, match="Checkpoint head_type"):
        ordinal.load(checkpoint)


def test_ordinal_checkpoint_records_objective_metadata(tmp_path: Path) -> None:
    import torch

    features, labels = _synthetic_sequences()
    ordinal = _adapter("coral").fit(features, labels)
    checkpoint = tmp_path / "ordinal.pt"
    ordinal.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["head_type"] == "coral"
    assert payload["training_config"]["head_type"] == "coral"
    assert payload["objective"]["head_type"] == "coral"
    assert payload["objective"]["num_thresholds"] == 4


def _split(features: np.ndarray, labels: np.ndarray) -> TaskSplit:
    test_rows = 10
    return TaskSplit(
        X_train=features[test_rows:],
        y_train=labels[test_rows:],
        X_test=features[:test_rows],
        y_test=labels[:test_rows],
        subject_train=np.asarray(["train"] * (len(labels) - test_rows)),
        subject_test=np.asarray(["test"] * test_rows),
        feature_names=[f"f{index}" for index in range(features.shape[-1])],
        sample_id_train=np.arange(test_rows, len(labels)),
        sample_id_test=np.arange(test_rows),
        record_id_train=np.asarray(["train-record"] * (len(labels) - test_rows)),
        record_id_test=np.asarray(["test-record"] * test_rows),
        row_metadata_test={
            "source": np.asarray(["synthetic"] * test_rows),
            "sequence_id": np.asarray([f"seq-{index}" for index in range(test_rows)]),
        },
        metadata={"split_type": "synthetic"},
    )


@pytest.mark.parametrize("head_type", ["categorical", "corn"])
def test_writer_preserves_categorical_and_adds_ordinal_columns(
    head_type: str,
    tmp_path: Path,
) -> None:
    features, labels = _synthetic_sequences()
    adapter = _adapter(head_type).fit(features, labels)
    split = _split(features, labels)
    detailed = adapter.predict_detailed(split.X_test)
    probabilities = detailed["class_probabilities"]
    predictions = detailed["y_pred"]
    metrics = MetricsCalculator.calculate_all_metrics(
        split.y_test,
        predictions,
        probabilities,
        expected_rank=detailed.get("expected_rank"),
    )
    runner = BenchmarkRunner({"output_dir": str(tmp_path), "models": {}})
    artifacts = runner._save_split_artifacts(
        model=adapter,
        split=split,
        y_pred=predictions,
        y_proba=probabilities,
        dataset_name="synthetic",
        task_name="cognitive_load_5class",
        model_name="transformer",
        artifact_split_name="test",
        metrics=metrics,
        detailed_predictions=detailed,
    )
    frame = pd.read_parquet(artifacts["predictions"])
    assert {f"proba_{index}" for index in range(5)} <= set(frame)
    if head_type == "categorical":
        assert "head_type" not in frame
        assert "threshold_probability_0" not in frame
        assert "ordinal_metadata" not in artifacts
    else:
        required = {
            "head_type",
            "expected_rank",
            "ordinal_argmax",
            "y_pred_argmax",
            *{f"threshold_logit_{index}" for index in range(4)},
            *{f"threshold_probability_{index}" for index in range(4)},
            *{f"conditional_probability_{index}" for index in range(4)},
            *{f"class_probability_{index}" for index in range(5)},
        }
        assert required <= set(frame)
        assert frame["head_type"].eq("corn").all()
        for index in range(5):
            np.testing.assert_allclose(
                frame[f"proba_{index}"],
                frame[f"class_probability_{index}"],
                atol=0,
                rtol=0,
            )
        metadata = json.loads(Path(artifacts["ordinal_metadata"]).read_text())
        assert metadata["head_type"] == "corn"
        assert metadata["round_off_correction_count"] >= 0
        assert metadata["monotonicity_within_tolerance"] is True
        assert metadata["maximum_class_probability_row_sum_error"] <= 1e-6


def test_metrics_include_qwk_and_expected_rank_diagnostics() -> None:
    truth = np.arange(5, dtype=np.int64)
    prediction = np.asarray([0, 1, 2, 4, 4])
    ranks = np.asarray([0.1, 1.0, 2.1, 3.4, 3.9])
    metrics = MetricsCalculator.calculate_all_metrics(
        truth,
        prediction,
        expected_rank=ranks,
    )
    assert np.isfinite(metrics["quadratic_weighted_kappa"])
    assert metrics["expected_rank_mae"] == pytest.approx(
        np.abs(truth - ranks).mean()
    )
    assert np.isfinite(metrics["expected_rank_spearman"])


def test_expected_rank_metrics_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricsCalculator.calculate_all_metrics(
            np.arange(5),
            np.arange(5),
            expected_rank=np.asarray([0.0, 1.0, np.nan, 3.0, 4.0]),
        )


def test_user_calibration_rejects_ordinal_base_before_loading_manifest(
    tmp_path: Path,
) -> None:
    base_run = tmp_path / "base"
    base_run.mkdir()
    (base_run / "config.yaml").write_text(
        yaml.safe_dump({
            "models": {
                "transformer": {
                    "type": "torch_transformer",
                    "params": {"head_type": "coral"},
                }
            }
        }),
        encoding="utf-8",
    )
    experiment = tmp_path / "calibration.yaml"
    experiment.write_text(
        yaml.safe_dump({
            "experiment": {"type": "user_calibration"},
            "base_run": {
                "run_directory": str(base_run),
                "model": "transformer",
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="Ordinal calibration"):
        UserCalibrationExperiment(experiment)


def test_adapter_fine_tune_rejects_ordinal_calibration() -> None:
    features, labels = _synthetic_sequences()
    adapter = _adapter("corn").fit(features, labels)
    with pytest.raises(NotImplementedError, match="Ordinal calibration"):
        adapter.fine_tune(features[:10], labels[:10], max_epochs=1)
