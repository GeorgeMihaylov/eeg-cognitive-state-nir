"""Synthetic checks for the native COG-BCI N-Back baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bench.datasets.cog_bci_baseline_dataset import (
    COGBCINBackWindowDataset,
)
from bench.experiments.cog_bci_nback_baseline import (
    BaselineRunOptions,
    COGBCINBackBaselineRunner,
    aggregate_record_predictions,
    calculate_record_subject_metrics,
    classification_metrics,
)
from model_zoo.factory import build_model


CACHE_HASH = "cache-hash"
PROTOCOL_HASH = "protocol-hash"
CHANNELS = [
    "EEG.AF3",
    "EEG.F7",
    "EEG.F3",
    "EEG.FC5",
    "EEG.T7",
    "EEG.P7",
    "EEG.O1",
    "EEG.O2",
    "EEG.P8",
    "EEG.T8",
    "EEG.FC6",
    "EEG.F4",
    "EEG.F8",
    "EEG.AF4",
]
VARIANTS = ("zero_back", "one_back", "two_back")


def _stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode()).hexdigest()[:24]


@pytest.fixture()
def synthetic_repository(tmp_path: Path) -> tuple[Path, dict]:
    cache = tmp_path / "cache"
    protocol = tmp_path / "protocol"
    shards = cache / "shards"
    shards.mkdir(parents=True)
    protocol.mkdir()
    target_rows = []
    cache_rows = []
    outer_rows = []
    rng = np.random.default_rng(12)
    for subject_index in range(10):
        subject = f"sub-{subject_index + 1:02d}"
        fold = subject_index % 5 + 1
        for target, variant in enumerate(VARIANTS):
            record = f"cog_bci::{subject}::ses-01::{variant}::run-na"
            array = rng.normal(
                loc=float(target), scale=0.1, size=(2, 14, 2560)
            ).astype(np.float32)
            stem = _stem(record)
            np.save(shards / f"{stem}.npy", array)
            (shards / f"{stem}.json").write_text(
                json.dumps({
                    "record_id": record,
                    "config_hash": CACHE_HASH,
                    "array_shape": [2, 14, 2560],
                }),
                encoding="utf-8",
            )
            for window in range(2):
                sample = f"{record}::w{window}"
                base = {
                    "sample_id": sample,
                    "subject_id": subject,
                    "session_id": "ses-01",
                    "record_id": record,
                    "record_group_id": record,
                    "window_index": window,
                }
                target_rows.append({
                    **base,
                    "dataset": "cog_bci",
                    "source": "COG-BCI",
                    "task_family": "n_back",
                    "task_variant": variant,
                    "start_sample": window * 2560,
                    "stop_sample": (window + 1) * 2560,
                    "start_time_seconds": window * 5.12,
                    "stop_time_seconds": (window + 1) * 5.12,
                    "window_duration_seconds": 5.12,
                    "status": "accepted",
                    "included_for_supervised": True,
                    "target": target,
                    "class_name": variant,
                    "target_name": "n_back_level",
                    "task_id": "cog_bci_nback_3class",
                    "target_level": "record",
                })
                cache_rows.append({
                    "sample_id": sample,
                    "cache_offset": window,
                    "channel_order": json.dumps(CHANNELS, separators=(",", ":")),
                    "sampling_rate_hz": 500.0,
                    "preprocessing_name": "none",
                })
                outer_rows.append({"sample_id": sample, "fold": fold})
    target = pd.DataFrame(target_rows)
    target.to_parquet(protocol / "target_index.parquet", index=False)
    pd.DataFrame(cache_rows).to_parquet(
        cache / "window_index.parquet", index=False
    )
    pd.DataFrame(outer_rows).to_parquet(
        protocol / "outer_assignments.parquet", index=False
    )
    inner_rows = []
    for fold in range(1, 6):
        test_subjects = {
            f"sub-{index + 1:02d}" for index in range(10) if index % 5 + 1 == fold
        }
        outer_train_subjects = sorted(
            set(target["subject_id"]) - test_subjects
        )
        validation_subjects = set(outer_train_subjects[:2])
        for row in target.itertuples(index=False):
            if row.subject_id in test_subjects:
                partition = "outer_test_excluded"
            elif row.subject_id in validation_subjects:
                partition = "inner_validation"
            else:
                partition = "inner_train"
            inner_rows.append({
                "outer_fold": fold,
                "sample_id": row.sample_id,
                "partition": partition,
            })
    pd.DataFrame(inner_rows).to_parquet(
        protocol / "inner_assignments.parquet", index=False
    )
    (cache / "dataset_manifest.json").write_text(
        json.dumps({
            "config_hash": CACHE_HASH,
            "channel_order": CHANNELS,
            "channel_count": 14,
            "samples_per_window": 2560,
            "sampling_rate_hz": 500.0,
            "channel_mapping_hash": "mapping-hash",
            "source_filter_status": "unknown_eeglab_processing_history",
        }),
        encoding="utf-8",
    )
    (protocol / "protocol_summary.json").write_text(
        json.dumps({
            "task_id": "cog_bci_nback_3class",
            "target_name": "n_back_level",
            "protocol_hash": PROTOCOL_HASH,
            "outer_split_hash": "outer-hash",
            "inner_split_hash": "inner-hash",
            "target_schema_hash": "schema-hash",
            "target_index_hash": "target-hash",
        }),
        encoding="utf-8",
    )
    config = {
        "dataset": "cog_bci_nback_raw",
        "window_cache": "cache",
        "task_protocol": "protocol",
        "model": {
            "type": "torch_eegnet",
            "params": {
                "temporal_kernel_seconds": 0.02,
                "separable_kernel_seconds": 0.01,
                "f1": 2,
                "depth_multiplier": 1,
                "f2": 2,
                "pool1": 8,
                "pool2": 8,
                "dropout": 0.1,
            },
        },
        "seed": 42,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 8,
        "num_workers": 0,
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "early_stopping": {
            "monitor": "validation_record_macro_f1",
            "mode": "max",
            "patience": 1,
        },
        "input_scale": {
            "input_unit_before": "volt_after_mne_reader",
            "source_physical_unit_status": "not_exposed_by_reader",
            "scale_factor": 1.0,
            "normalization": "inner_train_channel_standardization",
            "input_unit_after": "standardized_dimensionless",
        },
        "metrics": {"levels": ["window", "record"]},
        "smoke": {"max_train_batches_per_epoch": 1},
        "hashes": {
            "window_cache_config_hash": CACHE_HASH,
            "task_protocol_hash": PROTOCOL_HASH,
            "outer_split_hash": "outer-hash",
            "inner_split_hash": "inner-hash",
            "channel_mapping_hash": "mapping-hash",
        },
        "output_dir": "outputs/eegnet",
    }
    return tmp_path, config


def test_loader_contract_and_excludes_non_nback(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, config = synthetic_repository
    dataset = COGBCINBackWindowDataset({
        "data_path": root / config["window_cache"],
        "task_protocol_path": root / config["task_protocol"],
        "window_cache_config_hash": CACHE_HASH,
        "task_protocol_hash": PROTOCOL_HASH,
    })
    data = dataset.load()
    assert data.data.shape == (60, 1, 14, 2560)
    assert data.labels.shape == (60,)
    assert sorted(np.unique(data.labels).tolist()) == [0, 1, 2]
    assert data.n_subjects == 10
    assert len(np.unique(data.record_ids)) == 30
    assert "ECG1" not in data.feature_names
    assert data.feature_names == CHANNELS
    assert set(data.metadata["frame"]["task_family"]) == {"n_back"}
    assert data.metadata["frame"]["status"].eq("accepted").all()


def test_loader_does_not_modify_cache(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, config = synthetic_repository
    index = root / "cache" / "window_index.parquet"
    before = index.read_bytes()
    COGBCINBackWindowDataset({
        "data_path": root / config["window_cache"],
        "task_protocol_path": root / config["task_protocol"],
        "window_cache_config_hash": CACHE_HASH,
        "task_protocol_hash": PROTOCOL_HASH,
    }).load()
    assert index.read_bytes() == before


def test_precomputed_splits_are_used_without_overlap(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, config = synthetic_repository
    runner = COGBCINBackBaselineRunner(config, repository_root=root)
    data = runner._dataset()
    outer_train, outer_test, inner_train, inner_validation = (
        runner._split_indices(data, 1)
    )
    assert set(data.subject_ids[outer_train]).isdisjoint(
        data.subject_ids[outer_test]
    )
    assert set(data.subject_ids[outer_train[inner_train]]).isdisjoint(
        data.subject_ids[outer_train[inner_validation]]
    )
    assert set(data.sample_ids[outer_train]) | set(data.sample_ids[outer_test]) == set(
        data.sample_ids
    )


@pytest.mark.parametrize(
    "model_name", ["torch_eegnet", "torch_shallow_convnet"]
)
def test_raw_models_preserve_encoder_and_three_class_head(
    model_name: str,
) -> None:
    params = {
        "sampling_rate": 500.0,
        "channel_names": CHANNELS,
        "batch_size": 2,
        "max_epochs": 1,
        "device": "cpu",
        "standardize": False,
    }
    if model_name == "torch_eegnet":
        params.update({
            "temporal_kernel_seconds": 0.02,
            "separable_kernel_seconds": 0.01,
            "f1": 2,
            "depth_multiplier": 1,
            "f2": 2,
            "pool1": 8,
            "pool2": 8,
        })
    model = build_model(
        model_name,
        "classification",
        (1, 14, 2560),
        3,
        params,
    )
    X = torch.zeros(2, 1, 14, 2560)
    encoded = model.model.encode(X)
    output = model.model.forward_head(encoded)
    assert output.shape == (2, 3)
    assert encoded.shape[0] == 2


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["a0", "a1", "b0", "b1", "c0", "c1"],
        "record_id": ["a", "a", "b", "b", "c", "c"],
        "subject_id": ["s1"] * 6,
        "session_id": ["ses-01"] * 6,
        "true_class": [0, 0, 1, 1, 2, 2],
        "predicted_class": [0, 1, 1, 1, 2, 2],
        "probability_class_0": [0.8, 0.4, 0.1, 0.1, 0.1, 0.1],
        "probability_class_1": [0.1, 0.5, 0.8, 0.8, 0.2, 0.2],
        "probability_class_2": [0.1, 0.1, 0.1, 0.1, 0.7, 0.7],
        "fold_id": [1] * 6,
        "model": ["model"] * 6,
        "seed": [42] * 6,
    })


def test_record_aggregation_means_probabilities_without_mixing() -> None:
    records = aggregate_record_predictions(_prediction_frame())
    assert records["record_id"].is_unique
    assert records.set_index("record_id").loc["a", "window_count"] == 2
    assert records.set_index("record_id").loc["a", "predicted_class"] == 0
    np.testing.assert_allclose(
        records.set_index("record_id").loc[
            "a",
            [
                "mean_probability_class_0",
                "mean_probability_class_1",
                "mean_probability_class_2",
            ],
        ].to_numpy(dtype=float),
        [0.6, 0.3, 0.1],
    )


def test_ordinal_metrics_and_subject_record_summary() -> None:
    truth = np.asarray([0, 1, 2])
    prediction = np.asarray([1, 1, 0])
    probability = np.eye(3)[prediction]
    metrics = classification_metrics(truth, prediction, probability)
    assert metrics["ordinal_mae"] == pytest.approx(1.0)
    assert metrics["within_one_class_accuracy"] == pytest.approx(2 / 3)
    assert np.isfinite(metrics["quadratic_weighted_kappa"])
    records = aggregate_record_predictions(_prediction_frame())
    subject, summary = calculate_record_subject_metrics(records)
    assert len(subject) == 1
    assert subject.iloc[0]["records"] == 3
    assert set(summary) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ordinal_mae",
    }


def test_explicit_validation_record_monitor_and_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(18, 6)).astype(np.float32)
    y = np.tile(np.arange(3), 6).astype(np.int64)
    subjects = np.repeat([f"s{i}" for i in range(6)], 3)
    records = np.asarray(
        [f"{subjects[index]}-r{y[index]}" for index in range(len(y))]
    )
    model = build_model(
        "torch_mlp",
        "classification",
        (6,),
        3,
        {
            "hidden_dims": [8],
            "batch_size": 6,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_monitor": "validation_record_macro_f1",
            "device": "cpu",
            "random_state": 42,
        },
    )
    train = np.arange(12)
    validation = np.arange(12, 18)
    model.set_validation_indices(
        train,
        validation,
        subject_ids=subjects,
        record_ids=records,
        group_ids=subjects,
        outer_test_group_ids=np.asarray(["outer"]),
    )
    model.fit(X, y)
    assert model.inner_validation_indices_.tolist() == validation.tolist()
    assert model.early_stopping_monitor_ == "validation_record_macro_f1"
    assert model.best_monitor_value_ is not None
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_state_dict"] is not None
    assert payload["training_summary"]["best_monitor_value"] is not None
    loaded = build_model(
        "torch_mlp",
        "classification",
        (6,),
        3,
        {
            "hidden_dims": [8],
            "batch_size": 6,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_monitor": "validation_record_macro_f1",
            "device": "cpu",
            "random_state": 42,
        },
    )
    loaded.load(checkpoint)
    np.testing.assert_allclose(
        model.predict_proba(X), loaded.predict_proba(X), atol=1e-7
    )


def test_config_rejects_double_scaling(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, config = synthetic_repository
    config["input_scale"]["scale_factor"] = 1_000_000.0
    with pytest.raises(ValueError, match="unit-preserving"):
        COGBCINBackBaselineRunner(config, repository_root=root)


def test_synthetic_end_to_end_smoke_writes_unique_predictions(
    synthetic_repository: tuple[Path, dict],
) -> None:
    root, config = synthetic_repository
    runner = COGBCINBackBaselineRunner(
        config,
        repository_root=root,
        options=BaselineRunOptions(smoke=True, fold=1),
    )
    summary = runner.run()
    output = root / "outputs" / "eegnet" / "smoke"
    windows = pd.read_parquet(output / "window_predictions.parquet")
    records = pd.read_parquet(output / "record_predictions.parquet")
    assert summary["folds_completed"] == 1
    assert windows["sample_id"].is_unique
    assert records["record_id"].is_unique
    assert np.allclose(
        windows[
            [
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ]
        ].sum(axis=1),
        1.0,
    )
    assert (output / "checkpoints" / "fold_01.pt").is_file()
    assert json.loads(
        (output / "leakage_audit.json").read_text(encoding="utf-8")
    )["all_folds_leakage_safe"]
