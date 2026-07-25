from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bench.experiments.user_calibration import (
    CalibrationSpec,
    UserCalibrationExperiment,
    _as_window_observations,
    _parameter_audit,
    _state_digest,
    _use_reference_evaluation,
    chronological_window_partition,
)
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.factory import build_model


def _windows(
    rows_per_record: int = 100,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows = rows_per_record * 2
    X = rng.normal(size=(rows, 6)).astype(np.float32)
    y = np.arange(rows, dtype=np.int64) % 5
    record = np.repeat(["record-a", "record-b"], rows_per_record)
    offset = np.tile(np.arange(rows_per_record), 2)
    metadata = pd.DataFrame({
        "source": np.repeat(["Old_EEG", "gpn_data"], rows_per_record),
        "subject_id": "target",
        "record_id": record,
        "record_group_id": record,
        "sample_id": [f"sample-{index:04d}" for index in range(rows)],
        "t_start": offset.astype(float) * 10.0,
    })
    return X, y, metadata


def _fraction_spec(
    fraction: float,
    method: str = "head_only_finetuning",
) -> CalibrationSpec:
    return CalibrationSpec.from_dict({
        "method": method,
        "budget_fraction": fraction if fraction else None,
        "budget_seconds": 0 if fraction == 0 else None,
        "purge_windows": 0,
        "minimum_calibration_samples": 1,
        "minimum_evaluation_samples": 1,
        "min_calibration_sequences": 1,
        "min_evaluation_sequences": 1,
        "max_epochs": 1,
        "fallback_fixed_epochs": 1,
        "early_stopping_patience": 1,
        "random_state": 42,
    })


def _partition(fraction: float):
    X, y, metadata = _windows()
    return chronological_window_partition(
        X,
        y,
        metadata,
        _fraction_spec(fraction),
        window_seconds=10.0,
        max_gap_seconds=10.5,
    )


def _adapter(seed: int = 42) -> tuple[
    TorchClassificationAdapter, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    y = np.arange(80, dtype=np.int64) % 5
    adapter = build_model(
        model_name="torch_mlp",
        task_type="classification",
        input_shape=(6,),
        num_outputs=5,
        params={
            "hidden_dims": [12, 8],
            "dropout": 0.0,
            "batch_size": 16,
            "max_epochs": 2,
            "validation_size": 0.2,
            "early_stopping_patience": 2,
            "device": "cpu",
            "random_state": seed,
            "feature_scaling": {
                "strategy": "standard_clip",
                "clip_percentiles": [0.5, 99.5],
            },
            "feature_names": [f"EEG.feature_{index}" for index in range(6)],
        },
    )
    assert isinstance(adapter, TorchClassificationAdapter)
    adapter.fit(X, y)
    return adapter, X, y


def _parameters(adapter: TorchClassificationAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in adapter.model.named_parameters()
    }


def test_fractional_prefix_is_per_record_deterministic_and_disjoint() -> None:
    first = _partition(0.10)
    second = _partition(0.10)
    assert len(first.calibration_X) == 20
    assert first.actual_fraction == 0.10
    assert first.calibration_metadata.groupby("record_id").size().tolist() == [10, 10]
    assert first.calibration_metadata.sample_id.tolist() == (
        second.calibration_metadata.sample_id.tolist()
    )
    assert set(first.calibration_metadata.sample_id).isdisjoint(
        first.evaluation_metadata.sample_id
    )


def test_smaller_budgets_share_fixed_final_evaluation() -> None:
    reference = _partition(0.20)
    small = _use_reference_evaluation(_partition(0.01), reference)
    assert small.evaluation_metadata.sample_id.tolist() == (
        reference.evaluation_metadata.sample_id.tolist()
    )
    assert len(small.reserved_metadata) == 38
    all_ids = (
        set(small.calibration_metadata.sample_id)
        | set(small.reserved_metadata.sample_id)
        | set(small.evaluation_metadata.sample_id)
    )
    assert len(all_ids) == 200


def test_zero_budget_is_valid_without_calibration() -> None:
    partition = _use_reference_evaluation(_partition(0), _partition(0.20))
    calibration = _as_window_observations(
        partition.calibration_X,
        partition.calibration_y,
        partition.calibration_metadata,
    )
    evaluation = _as_window_observations(
        partition.evaluation_X,
        partition.evaluation_y,
        partition.evaluation_metadata,
    )
    assert UserCalibrationExperiment._status(
        _fraction_spec(0), partition, calibration, evaluation
    ) == "valid"


def test_mlp_head_only_finetuning_preserves_body_and_preprocessor() -> None:
    base, X, y = _adapter()
    adapted = base.clone()
    preprocessor_state = adapted.get_feature_preprocessing_state()
    before = _parameters(adapted)
    trainable, frozen, _, _ = _parameter_audit(adapted, "head_only")
    adapted.fine_tune(
        X[:30],
        np.zeros(30, dtype=np.int64),
        mode="head_only_finetuning",
        max_epochs=2,
        learning_rate=0.02,
    )
    after = _parameters(adapted)
    assert trainable
    assert any(not torch.equal(before[name], after[name]) for name in trainable)
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    assert adapted.get_feature_preprocessing_state() == preprocessor_state


def test_full_finetuning_starts_from_checkpoint_and_keeps_base_unchanged() -> None:
    base, X, y = _adapter()
    global_hash = _state_digest(base)
    adapted = base.clone()
    assert _state_digest(adapted) == global_hash
    adapted.fine_tune(
        X[:40],
        y[:40],
        mode="full_finetuning",
        max_epochs=1,
        learning_rate=0.01,
    )
    assert _state_digest(base) == global_hash
    assert _state_digest(adapted) != global_hash


def test_zero_adaptation_predictions_match_global_exactly() -> None:
    base, X, _ = _adapter()
    clone = base.clone()
    assert _state_digest(clone) == _state_digest(base)
    np.testing.assert_array_equal(
        clone.predict_proba(X[:12]), base.predict_proba(X[:12])
    )


def test_each_budget_clone_starts_from_same_global_checkpoint() -> None:
    base, X, y = _adapter()
    digest = _state_digest(base)
    final_hashes = []
    for size in (12, 24):
        adapted = base.clone()
        assert _state_digest(adapted) == digest
        adapted.fine_tune(
            X[:size],
            y[:size],
            mode="full_finetuning",
            max_epochs=1,
        )
        final_hashes.append(_state_digest(adapted))
    assert _state_digest(base) == digest
    assert all(value != digest for value in final_hashes)


def test_class_incomplete_calibration_produces_five_class_probabilities() -> None:
    base, X, _ = _adapter()
    adapted = base.clone()
    adapted.fine_tune(
        X[:20],
        np.zeros(20, dtype=np.int64),
        mode="head_only",
        max_epochs=1,
    )
    predictions = adapted.predict(X[20:30])
    probabilities = adapted.predict_proba(X[20:30])
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 5)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_finetuned_checkpoint_reproduces_predictions(tmp_path: Path) -> None:
    base, X, y = _adapter()
    adapted = base.clone()
    adapted.fine_tune(
        X[:30], y[:30], mode="head_only", max_epochs=1
    )
    expected = adapted.predict_proba(X[30:40])
    checkpoint = tmp_path / "model.pt"
    adapted.save(checkpoint)
    restored = base.clone().load(checkpoint)
    np.testing.assert_allclose(
        restored.predict_proba(X[30:40]), expected, atol=1e-7
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()
