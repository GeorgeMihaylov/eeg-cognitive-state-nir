from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.ordinal_transformer import (
    audit_prediction_probabilities,
    prediction_alignment,
)
from model_zoo import build_model


def _ordinal_predictions() -> pd.DataFrame:
    cumulative = np.asarray([
        [0.9, 0.7, 0.4, 0.1],
        [0.8, 0.6, 0.3, 0.2],
    ])
    probabilities = np.column_stack([
        1 - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2] - cumulative[:, 3],
        cumulative[:, 3],
    ])
    frame = pd.DataFrame({
        "sequence_id": ["s0", "s1"],
        "fold": [1, 1],
        "subject_id": ["a", "b"],
        "record_id": ["ra", "rb"],
        "source": ["x", "y"],
        "target_sample_id": [10, 20],
        "target_time": [70.0, 80.0],
        "y_true": [2, 3],
        "split": ["group_kfold_subject", "group_kfold_subject"],
        "y_pred": (cumulative >= 0.5).sum(axis=1),
        "expected_rank": cumulative.sum(axis=1),
        "ordinal_argmax": probabilities.argmax(axis=1),
    })
    for index in range(4):
        frame[f"threshold_probability_{index}"] = cumulative[:, index]
    for index in range(5):
        frame[f"class_probability_{index}"] = probabilities[:, index]
        frame[f"proba_{index}"] = probabilities[:, index]
    return frame


def test_probability_audit_accepts_valid_ordinal_artifacts() -> None:
    result = audit_prediction_probabilities(_ordinal_predictions(), "coral")
    assert result["class_probability_shape"] == [2, 5]
    assert result["y_pred_recomputation_mismatches"] == 0
    assert result["maximum_monotonicity_violation"] == 0


def test_probability_audit_detects_material_monotonicity_violation() -> None:
    frame = _ordinal_predictions()
    frame.loc[0, "threshold_probability_2"] = 0.8
    with pytest.raises(ValueError, match="monotonicity"):
        audit_prediction_probabilities(frame, "coral")


def test_categorical_probability_audit_needs_no_ordinal_columns() -> None:
    frame = _ordinal_predictions().loc[:, [
        "sequence_id", *[f"proba_{index}" for index in range(5)]
    ]]
    result = audit_prediction_probabilities(frame, "categorical")
    assert result["class_probability_shape"] == [2, 5]


def test_prediction_parquet_reload_preserves_identity_and_numeric_types(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.parquet"
    expected = _ordinal_predictions()
    expected.to_parquet(path, index=False)
    observed = pd.read_parquet(path)
    assert observed["sequence_id"].dtype == expected["sequence_id"].dtype
    assert np.issubdtype(observed["y_pred"].dtype, np.integer)
    assert np.issubdtype(observed["class_probability_0"].dtype, np.floating)
    assert prediction_alignment(expected, observed)["exact_match"] is True


@pytest.mark.parametrize("head_type", ["coral", "corn"])
def test_training_log_contains_head_specific_diagnostics(head_type: str) -> None:
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(5), 6).astype(np.int64)
    features = rng.normal(size=(len(labels), 8, 6)).astype(np.float32)
    adapter = build_model(
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
            "batch_size": 64,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    adapter.fit(features, labels)
    row = adapter.training_log_[0]
    assert np.isfinite(row["learning_rate"])
    assert adapter.get_training_summary()["stopping_reason"] == "max_epochs"
    if head_type == "coral":
        cutpoints = [row[f"cutpoint_{index}"] for index in range(4)]
        assert np.all(np.diff(cutpoints) > 0)
        assert row["cutpoint_min_gap"] > 0
    else:
        counts = [row[f"risk_count_{index}"] for index in range(4)]
        assert counts[0] > 0
        assert np.all(np.diff(counts) <= 0)
