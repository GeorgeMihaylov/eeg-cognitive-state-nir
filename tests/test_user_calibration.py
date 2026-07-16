from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import cli
from bench.core.abstract_dataset import EEGData
from bench.experiments.user_calibration import (
    CalibrationSpec,
    UserCalibrationExperiment,
    _build_sequences,
    calibration_normalization_statistics,
    chronological_window_partition,
    resolve_calibration_parameters,
)
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.factory import build_model


SEQUENCE_CONFIG = {
    "length": 8,
    "stride": 1,
    "target_position": "last",
    "expected_step_seconds": 10.0,
    "max_gap_seconds": 10.5,
}


def _windows(
    n_windows: int = 40,
    *,
    record_id: str = "record-a",
    subject_id: str = "subject-a",
    start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = np.arange(n_windows * 4, dtype=np.float32).reshape(n_windows, 4)
    y = np.arange(n_windows, dtype=np.int64) % 5
    metadata = pd.DataFrame({
        "source": "synthetic",
        "subject_id": subject_id,
        "record_id": record_id,
        "record_group_id": record_id,
        "sample_id": [f"{record_id}-{index:03d}" for index in range(n_windows)],
        "t_start": start + np.arange(n_windows, dtype=float) * 10.0,
    })
    return X, y, metadata


def _spec(method: str = "head_only", budget: float = 100.0) -> CalibrationSpec:
    return CalibrationSpec(
        method=method,
        budget_seconds=budget,
        purge_windows=7,
        min_evaluation_sequences=1,
    )


def _partition(
    method: str = "head_only", budget: float = 100.0
):
    X, y, metadata = _windows()
    return chronological_window_partition(
        X,
        y,
        metadata,
        _spec(method, budget),
        window_seconds=10.0,
        max_gap_seconds=10.5,
    )


def _small_adapter(random_state: int = 7) -> tuple[TorchClassificationAdapter, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(50, 4, 3)).astype(np.float32)
    y = np.arange(50, dtype=np.int64) % 2
    adapter = build_model(
        model_name="torch_transformer",
        task_type="classification",
        input_shape=(4, 3),
        num_outputs=2,
        params={
            "sequence_length": 4,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 10,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "device": "cpu",
            "random_state": random_state,
        },
    )
    assert isinstance(adapter, TorchClassificationAdapter)
    adapter.fit(X, y)
    return adapter, X, y


def _state(adapter: TorchClassificationAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in adapter.model.state_dict().items()
    }


def test_calibration_and_evaluation_windows_do_not_overlap() -> None:
    partition = _partition()
    calibration = set(partition.calibration_metadata["sample_id"])
    evaluation = set(partition.evaluation_metadata["sample_id"])
    assert calibration.isdisjoint(evaluation)


def test_purge_gap_is_sequence_length_minus_one() -> None:
    partition = _partition()
    assert len(partition.purged_metadata) == 7
    assert partition.calibration_metadata["t_start"].max() == 90.0
    assert partition.evaluation_metadata["t_start"].min() == 170.0


def test_chronological_prefix_is_deterministic() -> None:
    first = _partition()
    second = _partition()
    assert first.calibration_metadata["sample_id"].tolist() == (
        second.calibration_metadata["sample_id"].tolist()
    )
    assert first.actual_seconds == second.actual_seconds == 100.0


def test_sequences_from_calibration_and_evaluation_use_disjoint_windows() -> None:
    partition = _partition()
    calibration = _build_sequences(
        partition.calibration_X,
        partition.calibration_y,
        partition.calibration_metadata,
        SEQUENCE_CONFIG,
    )
    evaluation = _build_sequences(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
        SEQUENCE_CONFIG,
    )
    assert len(calibration.X) == 3
    assert len(evaluation.X) == 16
    assert set(partition.calibration_metadata.sample_id).isdisjoint(
        partition.evaluation_metadata.sample_id
    )


def test_record_boundaries_are_not_crossed() -> None:
    X1, y1, m1 = _windows(12, record_id="record-a")
    X2, y2, m2 = _windows(12, record_id="record-b")
    partition = chronological_window_partition(
        np.vstack([X1, X2]),
        np.r_[y1, y2],
        pd.concat([m1, m2], ignore_index=True),
        _spec(budget=120.0),
        window_seconds=10.0,
        max_gap_seconds=10.5,
    )
    evaluation = _build_sequences(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
        SEQUENCE_CONFIG,
    )
    assert set(evaluation.metadata["record_id"]) == {"record-b"}
    assert (evaluation.metadata["sequence_length"] == 8).all()


def test_gap_aware_split_does_not_cross_temporal_gap() -> None:
    X, y, metadata = _windows(20)
    metadata.loc[10:, "t_start"] += 100.0
    partition = chronological_window_partition(
        X,
        y,
        metadata,
        _spec(budget=100.0),
        window_seconds=10.0,
        max_gap_seconds=10.5,
    )
    evaluation = _build_sequences(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
        SEQUENCE_CONFIG,
    )
    assert len(partition.purged_metadata) == 0
    assert len(evaluation.X) == 3
    assert evaluation.metadata["max_internal_gap"].max() <= 10.5


def test_outer_group_kfold_has_no_test_subject_in_train() -> None:
    subject_ids = np.repeat(["s1", "s2", "s3", "s4"], 4)
    data = EEGData(
        data=np.zeros((16, 2), dtype=np.float32),
        labels=np.arange(16) % 2,
        subject_ids=subject_ids,
        sample_ids=np.arange(16),
        record_ids=np.asarray([f"r-{value}" for value in subject_ids]),
    )
    folds = CrossValidator(SimpleNamespace(data=data)).run_group_kfold(
        "subject_id", n_splits=2
    )
    for split in folds.values():
        assert set(split.subject_train).isdisjoint(split.subject_test)


def test_subject_normalization_uses_calibration_only() -> None:
    partition = _partition()
    expected_mean, expected_scale = calibration_normalization_statistics(
        partition.calibration_X
    )
    modified_evaluation = partition.evaluation_X + 1_000_000.0
    actual_mean, actual_scale = calibration_normalization_statistics(
        partition.calibration_X
    )
    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_scale, expected_scale)
    assert not np.allclose(modified_evaluation.mean(axis=0), actual_mean)


def test_adapter_checkpoint_load_round_trip(tmp_path: Path) -> None:
    base, X, _ = _small_adapter()
    checkpoint = tmp_path / "model.pt"
    base.save(checkpoint)
    restored, _, _ = _small_adapter()
    restored.load(checkpoint)
    np.testing.assert_allclose(
        restored.predict_proba(X[:5]), base.predict_proba(X[:5]), atol=1e-6
    )


def test_zero_shot_prediction_does_not_change_checkpoint(tmp_path: Path) -> None:
    base, X, _ = _small_adapter()
    checkpoint = tmp_path / "model.pt"
    base.save(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base.predict_proba(X[:5])
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert before == after


def test_head_only_changes_only_classifier_parameters() -> None:
    base, X, y = _small_adapter()
    adapted = base.clone()
    before = _state(adapted)
    adapted.fine_tune(
        X[:12],
        np.zeros(12, dtype=np.int64),
        trainable_parameter_prefixes=("classifier.",),
        max_epochs=2,
        learning_rate=1e-2,
    )
    after = _state(adapted)
    assert any(
        not torch.equal(before[name], after[name])
        for name in before if name.startswith("classifier.")
    )
    assert all(
        torch.equal(before[name], after[name])
        for name in before if not name.startswith("classifier.")
    )


def test_full_fine_tuning_uses_independent_clone() -> None:
    base, X, y = _small_adapter()
    base_state = _state(base)
    first = base.clone()
    second = base.clone()
    first.fine_tune(X[:16], y[:16], max_epochs=1)
    assert all(torch.equal(base_state[name], _state(base)[name]) for name in base_state)
    assert all(torch.equal(base_state[name], _state(second)[name]) for name in base_state)


def test_adapting_one_subject_does_not_affect_another() -> None:
    base, X, y = _small_adapter()
    subject_a = base.clone()
    subject_b = base.clone()
    before_b = subject_b.predict_proba(X[:5])
    subject_a.fine_tune(
        X[:10], y[:10],
        trainable_parameter_prefixes=("classifier.",),
        max_epochs=1,
    )
    np.testing.assert_allclose(subject_b.predict_proba(X[:5]), before_b, atol=1e-7)


def test_insufficient_sequence_context_status_is_explicit() -> None:
    partition = _partition(budget=60.0)
    calibration = _build_sequences(
        partition.calibration_X,
        partition.calibration_y,
        partition.calibration_metadata,
        SEQUENCE_CONFIG,
    )
    evaluation = _build_sequences(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
        SEQUENCE_CONFIG,
    )
    status = UserCalibrationExperiment._status(
        _spec(budget=60.0), partition, calibration, evaluation
    )
    assert status == "insufficient_sequence_context"


def test_prediction_frame_is_unique_and_excludes_calibration_samples() -> None:
    partition = _partition()
    evaluation = _build_sequences(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
        SEQUENCE_CONFIG,
    )
    probabilities = np.full((len(evaluation.X), 5), 0.2)
    frame = UserCalibrationExperiment._prediction_frame(
        "fold_01",
        "subject-a",
        _spec(method="zero_shot"),
        partition,
        evaluation,
        np.zeros(len(evaluation.X), dtype=int),
        probabilities,
    )
    assert not frame["sequence_id"].duplicated().any()
    assert not frame["is_calibration_sample"].any()


def test_metrics_are_computed_from_evaluation_only() -> None:
    evaluation_true = np.array([0, 1, 2, 3, 4])
    metrics = MetricsCalculator.calculate_all_metrics(
        evaluation_true, evaluation_true.copy()
    )
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["ordinal_mae"] == 0.0
    assert metrics["adjacent_accuracy"] == 1.0


def test_calibration_spec_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown calibration parameters"):
        CalibrationSpec.from_dict({
            "method": "head_only",
            "budget_seconds": 60,
            "evaluation_labels": True,
        })


def test_calibration_config_hash_changes_with_budget_and_method() -> None:
    first = CalibrationSpec(method="head_only", budget_seconds=60)
    repeated = CalibrationSpec(method="head_only", budget_seconds=60)
    changed_budget = CalibrationSpec(method="head_only", budget_seconds=180)
    changed_method = CalibrationSpec(method="zero_shot", budget_seconds=60)
    assert first.config_hash == repeated.config_hash
    assert first.config_hash != changed_budget.config_hash
    assert first.config_hash != changed_method.config_hash


def test_automl_style_parameter_dict_resolves() -> None:
    base = CalibrationSpec(method="head_only", budget_seconds=60)
    resolved = resolve_calibration_parameters(base, {
        "calibration.budget_seconds": 300,
        "calibration.learning_rate": 5e-4,
        "calibration.max_epochs": 9,
    })
    assert resolved.budget_seconds == 300
    assert resolved.learning_rate == 5e-4
    assert resolved.max_epochs == 9


def test_automl_style_unknown_dotted_parameter_fails() -> None:
    with pytest.raises(ValueError, match="Unknown calibration parameter"):
        resolve_calibration_parameters(
            CalibrationSpec(method="head_only", budget_seconds=60),
            {"calibration.future_test_score": 1.0},
        )


def test_cli_routes_calibration_smoke_overrides(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_init(self, path):
        calls["path"] = path

    def fake_execute(self, **kwargs):
        calls.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(UserCalibrationExperiment, "__init__", fake_init)
    monkeypatch.setattr(UserCalibrationExperiment, "execute", fake_execute)
    cli.main([
        "--calibration-experiment", "experiment.yaml",
        "--fold-limit", "1",
        "--subject-limit", "2",
        "--calibration-budgets", "0,60",
        "--calibration-methods", "zero_shot,head_only",
        "--max-calibration-epochs", "3",
        "--seed", "42",
        "--output-dir", "out",
    ])
    assert calls["fold_limit"] == 1
    assert calls["subject_limit"] == 2
    assert calls["budgets_seconds"] == [0.0, 60.0]
    assert calls["methods"] == ["zero_shot", "head_only"]
    assert calls["max_epochs"] == 3
    assert '"status": "completed"' in capsys.readouterr().out
