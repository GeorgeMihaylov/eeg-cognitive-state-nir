from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bench.automl.search_space import SearchSpaceSpec
from bench.automl.trial_resolver import resolve_automl_trial_config
from bench.datasets.emotiv_loader import EmotivDataset
from bench.experiments.preprocessing_ablation import resolve_trial_config


def _base_config(data_path: str = "features.parquet") -> dict:
    return {
        "output_dir": "results",
        "datasets": {
            "emotiv_cognitive": {
                "data_path": data_path,
                "feature_set": "pow_plus_eeg",
                "target_col": "label_q5",
                "subject_col": "subject_id",
                "n_classes": 5,
                "discretize": False,
            }
        },
        "tasks": ["cognitive_load_5class"],
        "sequence": {
            "length": 8,
            "pooling": "last",
            "positional_encoding": "learned",
        },
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "sequence_length": 8,
                    "pooling": "last",
                    "positional_encoding": "learned",
                },
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "group_column": "subject_id",
            "n_splits": 5,
        },
    }


def _space() -> SearchSpaceSpec:
    return SearchSpaceSpec.from_dict({
        "model.params.d_model": {"type": "categorical", "choices": [128]},
        "model.params.nhead": {"type": "categorical", "choices": [4]},
        "model.params.num_layers": {"type": "categorical", "choices": [2]},
        "model.params.dim_feedforward": {
            "type": "categorical",
            "choices": [256],
        },
        "model.params.dropout": {"type": "categorical", "choices": [0.2]},
        "training.learning_rate": {
            "type": "categorical",
            "choices": [0.0005],
        },
        "training.weight_decay": {
            "type": "categorical",
            "choices": [0.0001],
        },
        "training.batch_size": {"type": "categorical", "choices": [128]},
    })


def _params() -> dict:
    return {
        parameter.path: parameter.choices[0]
        for parameter in _space().parameters
    }


def test_shared_resolver_accepts_transformer_paths_without_raw_preprocessing() -> None:
    resolved = resolve_trial_config(_base_config(), {
        "model.name": "torch_transformer",
        **_params(),
        "training.random_state": 42,
        "training.max_epochs": 3,
    })
    params = resolved["models"]["torch_transformer"]["params"]
    assert params["d_model"] == 128
    assert params["learning_rate"] == 0.0005
    assert params["max_epochs"] == 3


def test_trial_resolution_excludes_outer_test_and_hashes_protocol(tmp_path) -> None:
    first, first_hash = resolve_automl_trial_config(
        base_config=_base_config(),
        trial_parameters=_params(),
        search_space=_space(),
        outer_fold=1,
        outer_train_subjects=["S1", "S2", "S3"],
        outer_test_subjects=["S4"],
        inner_splits=2,
        random_state=42,
        benchmark_runs_root=tmp_path,
        max_epochs=3,
        max_windows=300,
    )
    second, second_hash = resolve_automl_trial_config(
        base_config=_base_config(),
        trial_parameters=_params(),
        search_space=_space(),
        outer_fold=2,
        outer_train_subjects=["S1", "S2", "S4"],
        outer_test_subjects=["S3"],
        inner_splits=2,
        random_state=42,
        benchmark_runs_root=tmp_path,
        max_epochs=3,
        max_windows=300,
    )
    dataset = first["datasets"]["emotiv_cognitive"]
    assert dataset["include_subject_ids"] == ["S1", "S2", "S3"]
    assert "S4" not in dataset["include_subject_ids"]
    assert first["evaluation"]["role"] == "inner_search"
    assert first["evaluation"]["n_splits"] == 2
    assert first_hash != second_hash
    assert first["output_dir"].endswith(first_hash[:20])
    assert second["output_dir"].endswith(second_hash[:20])


def test_subject_filter_is_applied_before_window_limit(tmp_path) -> None:
    frame = pd.DataFrame({
        "subject_id": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4,
        "record_id": [f"R{i}" for i in range(12)],
        "sample_id": np.arange(12),
        "EEG.feature": np.arange(12, dtype=float),
        "POW.feature": np.arange(12, dtype=float) + 1.0,
        "label_q5": np.arange(12) % 5,
    })
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    data = EmotivDataset({
        "data_path": path,
        "feature_set": "pow_plus_eeg",
        "target_col": "label_q5",
        "discretize": False,
        "include_subject_ids": ["S1", "S2"],
        "max_windows": 6,
    }).load()
    assert data.n_samples == 6
    assert set(data.subject_ids) == {"S1", "S2"}
    assert data.metadata["include_subject_ids"] == ["S1", "S2"]


def test_subject_filter_rejects_unknown_subject(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    pd.DataFrame({
        "subject_id": ["S1"],
        "record_id": ["R1"],
        "EEG.feature": [0.0],
        "label_q5": [0],
    }).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="absent"):
        EmotivDataset({
            "data_path": path,
            "target_col": "label_q5",
            "discretize": False,
            "include_subject_ids": ["S2"],
        }).load()
