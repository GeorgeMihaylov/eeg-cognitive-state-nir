"""Cohort-manifest alignment for feature-based benchmark models."""

from pathlib import Path

import numpy as np
import pandas as pd

from bench.datasets.emotiv_loader import EmotivDataset


def test_feature_loader_uses_exact_sample_universe_and_outer_folds(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({
        "subject_id": ["a", "a", "b", "b"],
        "record_id": ["r1", "r1", "r2", "r2"],
        "source": ["x"] * 4,
        "target_focus": [0.1, 0.2, 0.3, 0.4],
        "EEG.AF3.mean": [1.0, 2.0, 3.0, 4.0],
        "POW.AF3.theta": [4.0, 3.0, 2.0, 1.0],
    })
    data_path = tmp_path / "features.parquet"
    frame.to_parquet(data_path, index=False)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "sample_id": [1, 3], "outer_fold": [1, 2],
        "subject_id": ["a", "b"], "record_id": ["r1", "r2"],
    }).to_parquet(cohort_path, index=False)
    dataset = EmotivDataset({
        "data_path": str(data_path),
        "target_id": "pm_focus_regression",
        "feature_set": "pow_plus_eeg",
        "cohort_manifest_path": str(cohort_path),
    }).load()
    assert dataset.sample_ids.tolist() == [1, 3]
    assert dataset.get_row_values("outer_fold").tolist() == [1, 2]
    assert dataset.subject_ids.tolist() == ["a", "b"]
    assert dataset.data.shape == (2, 2)
    assert np.allclose(dataset.labels, [0.2, 0.4])
