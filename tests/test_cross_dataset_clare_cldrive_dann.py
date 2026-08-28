from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments import cross_dataset_clare_cldrive_dann as protocol
from cogstate.model_zoo.DL.dann import DANNFoldData, DANNPartition


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/external_datasets/cross_dataset_clare_cldrive_dann_v1.json"


def _frame(dataset: str, *, participants: int = 10) -> pd.DataFrame:
    rows = []
    for participant_index in range(participants):
        source_id = f"{1000 + participant_index}"
        participant = f"sub-{source_id}"
        for class_id in range(3):
            for repeat in range(2):
                rows.append({
                    "sample_id": f"{dataset}-{participant}-{class_id}-{repeat}",
                    "dataset": dataset,
                    "participant_id": participant,
                    "dataset_participant_id": f"{dataset}::{participant}",
                    "source_participant_id": source_id,
                    "cross_dataset_person_key": (
                        participant if participant_index < 2 else f"{dataset}::{participant}"
                    ),
                    "record_id": f"{dataset}-{participant}-record",
                    "target": class_id,
                })
    return pd.DataFrame(rows)


def test_fixed_target_mapping() -> None:
    assert [protocol.fixed_target(value) for value in range(1, 10)] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2
    ]
    for value in (0, 10, 1.5, np.nan):
        with pytest.raises(ValueError):
            protocol.fixed_target(value)


def test_eeg_only_inclusion_is_independent_of_other_modalities() -> None:
    timestamps = np.arange(2560, dtype=float) / 256.0
    values = np.ones((2560, 4), dtype=np.float64)
    window, reason = protocol.eeg_only_window(
        timestamps, values, start_seconds=0.0, end_seconds=10.0
    )
    assert reason == "accepted"
    assert window is not None and window.shape == (1, 4, 2560)
    assert window.dtype == np.float32
    bad = values.copy()
    bad[4, 2] = np.nan
    assert protocol.eeg_only_window(
        timestamps, bad, start_seconds=0.0, end_seconds=10.0
    )[1] == "nonfinite_eeg"
    assert protocol.eeg_only_window(
        timestamps[:-1], values[:-1], start_seconds=0.0, end_seconds=10.0
    )[1] == "wrong_sample_count"


def test_config_and_cache_identity_are_deterministic() -> None:
    config = protocol.load_config(CONFIG)
    assert config["model"]["name"] == "torch_shallow_convnet"
    payload = {
        "shape": [3086, 1, 4, 2560],
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "target_free": True,
    }
    assert protocol.stable_hash(payload) == protocol.stable_hash(dict(reversed(list(payload.items()))))


def test_folds_are_deterministic_disjoint_and_exclude_overlapping_people() -> None:
    config = protocol.load_config(CONFIG)
    indices = {"cl_drive": _frame("cl_drive"), "clare": _frame("clare")}
    first = protocol.build_cross_dataset_folds(config, indices)
    second = protocol.build_cross_dataset_folds(config, indices)
    assert protocol.stable_hash(first) == protocol.stable_hash(second)
    assert len(first["folds"]) == 10
    audit = protocol.validate_protocol(config, first, indices)
    assert audit["leakage_status"] == "clean"
    for fold in first["folds"]:
        assert not (
            set(fold["target_adaptation_participants"])
            & set(fold["target_test_participants"])
        )
        source = indices[fold["source_dataset"]]
        selected = source[source["sample_id"].isin(fold["source_train_sample_ids"])]
        assert not (
            set(selected["cross_dataset_person_key"])
            & set(fold["target_test_cross_dataset_person_keys"])
        )


def test_run_matrix_has_30_paired_runs_and_stable_hashes() -> None:
    config = protocol.load_config(CONFIG)
    indices = {"cl_drive": _frame("cl_drive"), "clare": _frame("clare")}
    folds = protocol.build_cross_dataset_folds(config, indices)
    matrix = protocol.build_run_matrix(config, folds, "protocol-hash")
    assert len(matrix) == 30
    assert matrix["run_id"].nunique() == 30
    assert set(matrix["mode"]) == set(protocol.MODES)
    grouped = matrix.groupby(["direction", "fold"])
    assert grouped["test_sample_ids_hash"].nunique().eq(1).all()
    assert grouped.size().eq(3).all()


def test_dann_adaptation_loader_has_no_target_task_labels() -> None:
    shape = (3, 1, 4, 32)
    source = DANNPartition(
        "source", np.ones(shape, np.float32), np.zeros(3, np.int64),
        [f"source-{i}" for i in range(3)], [f"source-record-{i}" for i in range(3)],
        [f"source-subject-{i}" for i in range(3)], np.asarray([0, 1, 2]),
    )
    target = DANNPartition(
        "target", np.ones(shape, np.float32), np.ones(3, np.int64),
        [f"target-{i}" for i in range(3)], [f"target-record-{i}" for i in range(3)],
        [f"target-subject-{i}" for i in range(3)], None,
    )
    validation = DANNPartition(
        "validation", np.ones(shape, np.float32), np.zeros(3, np.int64),
        [f"validation-{i}" for i in range(3)], [f"validation-record-{i}" for i in range(3)],
        [f"validation-subject-{i}" for i in range(3)], np.asarray([0, 1, 2]),
    )
    test = DANNPartition(
        "test", np.ones(shape, np.float32), np.ones(3, np.int64),
        [f"test-{i}" for i in range(3)], [f"test-record-{i}" for i in range(3)],
        [f"test-subject-{i}" for i in range(3)], np.asarray([0, 1, 2]),
    )
    batch = next(iter(DANNFoldData(source, target, validation, test).training_loader(
        batch_size=2, shuffle=False
    )))
    assert hasattr(batch, "source_task_labels")
    assert not hasattr(batch, "target_task_labels")


def test_plan_only_writes_protocol_but_never_trains(monkeypatch, tmp_path: Path) -> None:
    config = protocol.load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    indices = {"cl_drive": _frame("cl_drive"), "clare": _frame("clare")}
    manifests = {
        name: {
            "cache_identity_hash": f"{name}-cache",
            "shape": [len(frame), 1, 4, 2560],
            "rows": len(frame),
        }
        for name, frame in indices.items()
    }
    monkeypatch.setattr(protocol, "_repository_root", lambda _: tmp_path)
    monkeypatch.setattr(
        protocol,
        "participant_identity_audit",
        lambda *_: {
            "overlapping_source_ids": ["sub-1000", "sub-1001"],
            "count": 2,
            "evidence_inspected": [],
            "conclusion": "synthetic",
            "policy_used_for_protocol": "conservative",
        },
    )
    monkeypatch.setattr(
        protocol,
        "_load_cache",
        lambda _config, _root, dataset: (
            np.empty((len(indices[dataset]), 1, 4, 2560), dtype=np.float32),
            indices[dataset],
            manifests[dataset],
        ),
    )
    monkeypatch.setattr(
        protocol,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail("plan-only attempted to build a model"),
    )
    first = protocol.plan_experiment(config_path)
    second = protocol.plan_experiment(config_path)
    assert first == second
    assert first["planned_runs"] == 30
    assert first["models_trained"] == 0
    assert first["training_status"] == "training_not_started"
    assert (tmp_path / "output/protocol_manifest.json").is_file()


def test_execute_is_confirmation_gated() -> None:
    with pytest.raises(PermissionError):
        protocol.execute(CONFIG)


def test_resume_requires_complete_matching_specification(tmp_path: Path) -> None:
    summary = tmp_path / "run_summary.json"
    summary.write_text(
        json.dumps({"status": "complete", "specification_hash": "current"}),
        encoding="utf-8",
    )
    assert protocol._resumable_summary(summary, "current") is not None
    assert protocol._resumable_summary(summary, "stale") is None
    summary.write_text(
        json.dumps({"status": "failed", "specification_hash": "current"}),
        encoding="utf-8",
    )
    assert protocol._resumable_summary(summary, "current") is None
