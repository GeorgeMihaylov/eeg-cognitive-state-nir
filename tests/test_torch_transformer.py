from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from bench.bench_runner import BenchmarkRunner, benchmark_config_hash
from bench.core.abstract_dataset import EEGData
from bench.datasets.emotiv_loader import EmotivDataset
from cli import override_config_with_args
from model_zoo import build_model
from model_zoo.DL import (
    TorchClassificationAdapter,
    TorchFeatureTransformerClassifier,
)
from model_zoo.DL.adapter import seed_torch
from model_zoo.DL.sequence_utils import build_sequences


def _module(pooling: str = "last") -> TorchFeatureTransformerClassifier:
    return TorchFeatureTransformerClassifier(
        input_size=6,
        num_classes=5,
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        pooling=pooling,
        positional_encoding="learned",
    )


def _classification_sequences() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(5), 16).astype(np.int64)
    X = rng.normal(size=(len(y), 4, 6)).astype(np.float32)
    X[:, :, 0] += y[:, None]
    return X, y


def _sequence_metadata(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["gpn_data"] * len(times),
        "subject_id": ["S1"] * len(times),
        "record_id": ["R1"] * len(times),
        "sample_id": np.arange(100, 100 + len(times)),
        "t_start": times,
    })


@pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
def test_transformer_forward_output_shape(pooling: str) -> None:
    model = _module(pooling)
    logits = model(torch.randn(7, 4, 6))

    assert logits.shape == (7, 5)
    assert torch.isfinite(logits).all()


def test_factory_builds_transformer_and_passes_parameters() -> None:
    adapter = build_model(
        "torch_transformer",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "sequence_length": 4,
            "d_model": 12,
            "nhead": 3,
            "num_layers": 2,
            "dim_feedforward": 24,
            "dropout": 0.2,
            "activation": "relu",
            "pooling": "mean",
            "positional_encoding": "sinusoidal",
            "max_epochs": 1,
            "device": "cpu",
        },
    )

    assert isinstance(adapter, TorchClassificationAdapter)
    assert adapter.input_shape == (4, 6)
    assert adapter.num_classes == 5
    assert adapter.model.d_model == 12
    assert adapter.model.nhead == 3
    assert adapter.model.num_layers == 2
    assert adapter.model.dim_feedforward == 24
    assert adapter.model.activation == "relu"
    assert adapter.model.pooling == "mean"
    assert adapter.model.positional_encoding_kind == "sinusoidal"
    assert adapter.model_metadata["parameter_count"] == sum(
        parameter.numel()
        for parameter in adapter.model.parameters()
        if parameter.requires_grad
    )


def test_factory_rejects_invalid_or_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        build_model(
            "torch_transformer",
            "classification",
            (4, 6),
            5,
            {"d_model": 10, "nhead": 3, "device": "cpu"},
        )
    with pytest.raises(ValueError, match="Unsupported torch_transformer"):
        build_model(
            "torch_transformer",
            "classification",
            (4, 6),
            5,
            {"unknown_parameter": 1, "device": "cpu"},
        )
    with pytest.raises(ValueError, match="must match sequences"):
        build_model(
            "torch_transformer",
            "classification",
            (4, 6),
            5,
            {"sequence_length": 8, "device": "cpu"},
        )


def test_learned_positional_encoding_is_seed_deterministic() -> None:
    X = torch.randn(3, 4, 6)
    seed_torch(42)
    first = _module().eval()
    seed_torch(42)
    second = _module().eval()

    torch.testing.assert_close(
        first.positional_encoding.encoding,
        second.positional_encoding.encoding,
    )
    torch.testing.assert_close(first(X), second(X))


@pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
def test_padding_mask_excludes_padded_tokens(pooling: str) -> None:
    seed_torch(42)
    model = _module(pooling).eval()
    X = torch.randn(2, 4, 6)
    padding_mask = torch.tensor([
        [False, False, True, True],
        [False, False, False, True],
    ])
    changed = X.clone()
    changed[padding_mask] = 10_000.0

    with torch.no_grad():
        reference = model(X, padding_mask=padding_mask)
        actual = model(changed, padding_mask=padding_mask)

    torch.testing.assert_close(reference, actual, atol=1e-6, rtol=1e-6)


def test_sequence_builder_respects_records_and_time_gaps() -> None:
    first = _sequence_metadata([0.0, 10.0, 20.0, 100.0, 110.0, 120.0])
    second = first.copy()
    second["record_id"] = "R2"
    second["sample_id"] += 100
    metadata = pd.concat([first, second], ignore_index=True)
    X = np.column_stack([
        metadata["sample_id"].to_numpy(),
        metadata["t_start"].to_numpy(),
    ]).astype(np.float32)
    y = np.arange(len(metadata), dtype=np.int64) % 5

    result = build_sequences(
        X,
        y,
        metadata,
        sequence_length=3,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )

    assert len(result.X) == 4
    assert result.stats["records_total"] == 2
    assert result.stats["gaps_detected"] == 2
    for sequence, row in zip(result.X, result.metadata.itertuples(index=False)):
        sample_ids = sequence[:, 0].astype(np.int64)
        source_rows = metadata.set_index("sample_id").loc[sample_ids]
        assert source_rows["record_id"].nunique() == 1
        assert source_rows["record_id"].iloc[0] == row.record_id
        assert np.diff(sequence[:, 1]).max() <= 10.5


def test_adapter_fit_predict_probabilities_and_checkpoint(tmp_path) -> None:
    X, y = _classification_sequences()
    adapter = build_model(
        "torch_transformer",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 20,
            "max_epochs": 1,
            "validation_size": 0.25,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    adapter.fit(X, y)
    predictions = adapter.predict(X[:11])
    probabilities = adapter.predict_proba(X[:11])
    checkpoint = tmp_path / "model.pt"
    adapter.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")

    restored = build_model(
        "torch_transformer",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_epochs": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    restored.model.load_state_dict(payload["model_state_dict"])

    assert predictions.shape == (11,)
    assert probabilities.shape == (11, 5)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert payload["input_shape"] == (4, 6)
    assert payload["model_metadata"]["model_type"] == "torch_transformer"
    for name, parameter in restored.model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), payload["model_state_dict"][name])


def _runner_data() -> EEGData:
    rng = np.random.default_rng(123)
    rows = []
    for subject_index in range(6):
        for record_index in range(2):
            for window_index in range(10):
                rows.append((subject_index, record_index, window_index))
    n_rows = len(rows)
    subjects = np.asarray([f"S{row[0]}" for row in rows])
    records = np.asarray([f"S{row[0]}_R{row[1]}" for row in rows])
    times = np.asarray([row[2] * 10.0 for row in rows])
    labels = np.asarray([row[2] % 5 for row in rows], dtype=np.int64)
    features = rng.normal(size=(n_rows, 6)).astype(np.float32)
    features[:, 0] += labels
    return EEGData(
        data=features,
        labels=labels,
        subject_ids=subjects,
        feature_names=[f"f{index}" for index in range(features.shape[1])],
        sample_ids=np.arange(n_rows, dtype=np.int64),
        record_ids=records,
        row_metadata={
            "source": np.asarray(["synthetic"] * n_rows),
            "t_start": times,
            "record_group_id": records.copy(),
        },
    )


def test_runner_builds_fresh_transformer_per_fold_and_infers_shape(
    tmp_path, monkeypatch
) -> None:
    dataset = Mock()
    dataset.load.return_value = _runner_data()
    monkeypatch.setattr(
        "bench.bench_runner.get_dataset", lambda *args, **kwargs: dataset
    )
    config = {
        "output_dir": str(tmp_path),
        "datasets": {"synthetic": {"data_path": "unused.parquet"}},
        "tasks": ["cognitive_load_5class"],
        "sequence": {
            "length": 3,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.5,
        },
        "validation": {
            "strategy": "group_record",
            "group_column": "record_group_id",
            "validation_size": 0.25,
            "random_state": 42,
        },
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "dropout": 0.0,
                    "batch_size": 16,
                    "max_epochs": 1,
                    "early_stopping_patience": 1,
                    "device": "cpu",
                    "random_state": 42,
                },
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": 2,
            "group_column": "subject_id",
            "random_state": 42,
        },
        "task_config": {"random_state": 42},
        "run_within_subject": False,
        "run_loso": False,
    }
    runner = BenchmarkRunner(config)
    created = []
    original_create = runner._create_model

    def recording_create(*args, **kwargs):
        model = original_create(*args, **kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(runner, "_create_model", recording_create)
    summary = runner.run()

    assert len(summary) == 1
    assert len(created) == 2
    assert len({id(model) for model in created}) == 2
    assert all(model.input_shape == (3, 6) for model in created)
    assert len(list(tmp_path.rglob("run_manifest.json"))) == 1
    assert len(list(tmp_path.rglob("model.pt"))) == 2
    assert len(list(tmp_path.rglob("training_log.csv"))) == 2
    assert len(list(tmp_path.rglob("normalization_stats.json"))) == 2
    assert len(list(tmp_path.rglob("class_metrics.json"))) == 2
    assert len(list(tmp_path.rglob("sequence_index_manifest.json"))) == 2


def test_main_cli_smoke_overrides_are_canonical() -> None:
    config = {
        "output_dir": "results",
        "datasets": {"data": {"data_path": "dataset.parquet"}},
        "tasks": ["cognitive_load_5class"],
        "sequence": {"length": 8},
        "models": {
            "transformer": {
                "type": "torch_transformer",
                "params": {"max_epochs": 15, "random_state": 42},
            }
        },
        "evaluation": {"protocol": "group_kfold_subject", "n_splits": 5},
        "validation": {"random_state": 42},
        "task_config": {"random_state": 42},
    }
    args = SimpleNamespace(
        output_dir=None,
        dataset=None,
        models=None,
        task=None,
        no_loso=False,
        no_within=False,
        feature_set=None,
        seed=7,
        fold_limit=1,
        max_windows=1000,
        max_epochs=3,
    )

    resolved = override_config_with_args(config, args)

    assert resolved["evaluation"]["folds"] == [1]
    assert resolved["evaluation"]["random_state"] == 7
    assert resolved["datasets"]["data"]["max_windows"] == 1000
    assert resolved["models"]["transformer"]["params"]["max_epochs"] == 3
    assert resolved["models"]["transformer"]["params"]["random_state"] == 7


def test_transformer_config_hash_is_parameter_deterministic() -> None:
    config = {
        "models": {
            "transformer": {
                "type": "torch_transformer",
                "params": {
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 2,
                },
            }
        },
        "sequence": {"length": 8},
    }
    repeated = deepcopy(config)
    changed = deepcopy(config)
    changed["models"]["transformer"]["params"]["d_model"] = 64

    assert benchmark_config_hash(config) == benchmark_config_hash(repeated)
    assert benchmark_config_hash(config) != benchmark_config_hash(changed)


def test_feature_dataset_max_windows_preserves_record_groups(tmp_path) -> None:
    rows = []
    for subject_index in range(5):
        for window_index in range(10):
            rows.append({
                "sample_id": subject_index * 10 + window_index,
                "source": "synthetic",
                "subject_id": f"S{subject_index}",
                "record_id": f"R{subject_index}",
                "t_start": float(window_index * 10),
                "label_q5": window_index % 5,
                "EEG.AF3__mean": float(window_index),
            })
    path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    dataset = EmotivDataset({
        "data_path": str(path),
        "feature_set": "pow_plus_eeg",
        "target_col": "label_q5",
        "n_classes": 5,
        "discretize": False,
        "max_windows": 20,
    })

    data = dataset.load()

    assert data.n_samples == 20
    assert data.n_subjects == 5
    np.testing.assert_array_equal(
        data.row_metadata["record_group_id"], data.record_ids
    )
