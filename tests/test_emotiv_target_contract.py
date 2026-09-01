from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.datasets.base_eeg_data_loader import BaseEEGDataset
from bench.datasets.emotiv_loader import EmotivDataset


class _TestDataset(BaseEEGDataset):
    def load(self):
        raise NotImplementedError


def _write_placeholder(path: Path) -> None:
    pd.DataFrame({"x": [1]}).to_parquet(path)


def test_discretization_preserves_missing_targets(tmp_path: Path) -> None:
    data_path = tmp_path / "placeholder.parquet"
    _write_placeholder(data_path)

    dataset = _TestDataset(
        {
            "data_path": str(data_path),
            "n_classes": 5,
            "discretize": True,
        }
    )
    values = np.asarray(
        [0.01, 0.11, np.nan, 0.24, 0.39, 0.51, 0.66, 0.78, 0.88, 0.99],
        dtype=float,
    )

    labels = dataset._discretize_target(values)

    assert labels.shape == values.shape
    assert np.isnan(labels[2])
    assert np.isfinite(labels[np.arange(len(labels)) != 2]).all()


def test_discretization_preserves_existing_integer_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "placeholder.parquet"
    _write_placeholder(data_path)

    dataset = _TestDataset(
        {
            "data_path": str(data_path),
            "n_classes": 3,
            "discretize": True,
        }
    )
    values = np.asarray([0, 1, 2, 0, 1, 2, np.nan], dtype=float)
    monkeypatch.setattr(
        pd,
        "qcut",
        lambda *args, **kwargs: pytest.fail(
            "Complete stored classes must not be passed to qcut"
        ),
    )

    labels = dataset._discretize_target(values)

    assert labels[:6].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    assert np.isnan(labels[6])


def test_incomplete_integer_class_subset_is_discretized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "placeholder.parquet"
    _write_placeholder(data_path)
    dataset = _TestDataset(
        {
            "data_path": str(data_path),
            "n_classes": 3,
            "discretize": True,
        }
    )
    values = np.asarray([0, 0, 1, 1, np.nan], dtype=float)
    qcut_calls = []

    def _fake_qcut(values, *, q, labels, duplicates):
        qcut_calls.append((values.copy(), q, labels, duplicates))
        return np.asarray([0, 0, 2, 2])

    monkeypatch.setattr(pd, "qcut", _fake_qcut)

    result = dataset._discretize_target(values)

    assert len(qcut_calls) == 1
    assert qcut_calls[0][1:] == (3, False, "drop")
    assert result[:4].tolist() == [0.0, 0.0, 2.0, 2.0]
    assert np.isnan(result[4])


def test_explicit_label_q5_contract_drops_unlabeled_rows(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "record_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "sample_id": np.arange(6),
            "EEG.AF3.mean": np.arange(6, dtype=float),
            "POW.AF3.alpha": np.arange(10, 16, dtype=float),
            "target_main": np.linspace(0.0, 1.0, 6),
            "label_q5": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
        }
    )
    frame.to_parquet(data_path)

    dataset = EmotivDataset(
        {
            "data_path": str(data_path),
            "feature_set": "pow_plus_eeg",
            "target_col": "label_q5",
            "n_classes": 5,
            "discretize": False,
            "max_features": None,
        }
    ).load()

    assert dataset.n_samples == 5
    assert dataset.n_subjects == 3
    assert dataset.metadata["target_col"] == "label_q5"
    assert dataset.metadata["discretize"] is False
    assert dataset.labels.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_explicit_target_does_not_fall_back_to_target_main(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "subject_id": ["s1", "s1"],
            "record_id": ["r1", "r1"],
            "sample_id": [0, 1],
            "EEG.AF3.mean": [0.0, 1.0],
            "POW.AF3.alpha": [2.0, 3.0],
            "target_main": [0.1, 0.9],
        }
    )
    frame.to_parquet(data_path)

    dataset = EmotivDataset(
        {
            "data_path": str(data_path),
            "feature_set": "pow_plus_eeg",
            "target_col": "label_q5",
            "n_classes": 5,
            "discretize": False,
        }
    )

    with pytest.raises(ValueError, match="Configured target column"):
        dataset.load()
