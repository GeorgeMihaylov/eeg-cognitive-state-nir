from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_task import TaskSplit
from cli import validate_config
from cogstate.model_zoo import build_model


def _classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    subjects = np.repeat([f"subject_{index:02d}" for index in range(10)], 6)
    labels = np.tile(np.asarray([0, 1, 2, 0, 1, 2]), 10)
    features = rng.normal(size=(len(labels), 4)).astype(np.float32)
    features[:, 0] += labels
    records = np.asarray([
        f"{subject}_record_{row % 2}"
        for row, subject in enumerate(subjects)
    ])
    return features, labels, subjects, records


def _torch_mlp(
    *,
    task_type: str = "classification",
    num_outputs: int = 3,
    random_state: int = 42,
):
    return build_model(
        "torch_mlp",
        task_type,
        input_shape=(4,),
        num_outputs=num_outputs,
        params={
            "hidden_dims": [8],
            "dropout": 0.0,
            "batch_size": 16,
            "max_epochs": 1,
            "learning_rate": 0.002,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": random_state,
        },
    )


def _configure_group_holdout(
    model,
    subjects: np.ndarray,
    records: np.ndarray,
    *,
    random_state: int = 42,
) -> None:
    model.set_validation_groups(
        subjects,
        subject_ids=subjects,
        record_ids=records,
        outer_test_group_ids=np.asarray(["outer_00", "outer_01"]),
        strategy="group_holdout",
        group_column="subject_id",
        validation_size=0.2,
        random_state=random_state,
    )


def test_group_holdout_is_subject_disjoint_and_deterministic() -> None:
    _, labels, subjects, records = _classification_data()
    first = _torch_mlp()
    second = _torch_mlp()
    changed_seed = _torch_mlp(random_state=7)
    _configure_group_holdout(first, subjects, records)
    _configure_group_holdout(second, subjects, records)
    _configure_group_holdout(
        changed_seed, subjects, records, random_state=7
    )

    train_idx, validation_idx = first._validation_indices(labels)
    train_idx_again, validation_idx_again = second._validation_indices(labels)
    _, validation_idx_changed = changed_seed._validation_indices(labels)

    np.testing.assert_array_equal(train_idx, train_idx_again)
    np.testing.assert_array_equal(validation_idx, validation_idx_again)
    assert set(subjects[train_idx]).isdisjoint(subjects[validation_idx])
    assert set(subjects[train_idx]) | set(subjects[validation_idx]) == set(subjects)
    assert len(np.unique(subjects[train_idx])) >= 1
    assert len(np.unique(subjects[validation_idx])) >= 1
    for subject in np.unique(subjects):
        partitions = {
            "train" if index in set(train_idx) else "validation"
            for index in np.flatnonzero(subjects == subject)
        }
        assert len(partitions) == 1
    assert not np.array_equal(validation_idx, validation_idx_changed)


def test_group_holdout_rejects_one_group_and_impossible_class_coverage() -> None:
    one_group = _torch_mlp()
    subjects = np.repeat("subject_00", 12)
    records = np.asarray([f"record_{index}" for index in range(12)])
    _configure_group_holdout(one_group, subjects, records)
    with pytest.raises(ValueError, match="at least two unique"):
        one_group._validation_indices(np.tile(np.arange(3), 4))

    impossible = _torch_mlp()
    isolated_subjects = np.repeat(["s0", "s1", "s2"], 4)
    isolated_labels = np.repeat([0, 1, 2], 4)
    isolated_records = np.asarray([
        f"r{index}" for index in range(len(isolated_labels))
    ])
    _configure_group_holdout(
        impossible, isolated_subjects, isolated_records
    )
    with pytest.raises(
        ValueError,
        match="class-complete group_holdout.*class_distribution_by_group",
    ):
        impossible._validation_indices(isolated_labels)


def test_grouped_classification_preserves_target_shape_and_fits_scaler_on_train() -> None:
    features, labels, subjects, records = _classification_data()
    model = _torch_mlp()
    _configure_group_holdout(model, subjects, records)

    model.fit(features, labels)

    assert model.predict(features[:5]).shape == (5,)
    expected_mean = features[model.inner_train_indices_].mean(
        axis=0, dtype=np.float64
    )
    np.testing.assert_allclose(model.feature_mean_, expected_mean, atol=1e-6)
    assert not np.allclose(model.feature_mean_, features.mean(axis=0))
    metadata = model.validation_split_
    assert metadata["validation_strategy"] == "group_holdout"
    assert metadata["validation_group_column"] == "subject_id"
    assert metadata["validation_fraction"] == pytest.approx(0.2)
    assert metadata["validation_random_state"] == 42
    assert metadata["inner_group_overlap"] == 0
    assert metadata["outer_test_group_overlap"] == 0
    assert metadata["train_outer_test_overlap_count"] == 0
    assert metadata["validation_outer_test_overlap_count"] == 0
    assert set(metadata["inner_train_class_distribution"]) == {"0", "1", "2"}
    assert set(metadata["inner_val_class_distribution"]) == {"0", "1", "2"}


@pytest.mark.parametrize(
    ("num_outputs", "target_shape", "expected_shape"),
    [
        (1, (60,), (5,)),
        (7, (60, 7), (5, 7)),
    ],
)
def test_grouped_regression_preserves_scalar_and_multioutput_shapes(
    num_outputs: int,
    target_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(60, 4)).astype(np.float32)
    targets = rng.normal(size=target_shape).astype(np.float32)
    subjects = np.repeat([f"subject_{index:02d}" for index in range(10)], 6)
    records = np.asarray([f"record_{index // 3}" for index in range(60)])
    model = _torch_mlp(task_type="regression", num_outputs=num_outputs)
    _configure_group_holdout(model, subjects, records)

    model.fit(features, targets)

    assert model.predict(features[:5]).shape == expected_shape
    assert model.validation_split_["inner_group_overlap"] == 0


def test_explicit_random_holdout_remains_available() -> None:
    _, labels, subjects, records = _classification_data()
    model = _torch_mlp()
    model.set_random_validation(
        subject_ids=subjects,
        record_ids=records,
        validation_size=0.2,
        random_state=42,
    )

    train_idx, validation_idx = model._validation_indices(labels)

    assert model.validation_strategy_ == "random_holdout"
    assert set(subjects[train_idx]) & set(subjects[validation_idx])


def test_runner_saves_zero_overlap_inner_validation_audit(
    tmp_path: Path,
) -> None:
    features, labels, subjects, records = _classification_data()
    train_mask = ~np.isin(subjects, ["subject_08", "subject_09"])
    test_mask = ~train_mask
    split = TaskSplit(
        X_train=features[train_mask],
        y_train=labels[train_mask],
        X_test=features[test_mask],
        y_test=labels[test_mask],
        subject_train=subjects[train_mask],
        subject_test=subjects[test_mask],
        record_id_train=records[train_mask],
        record_id_test=records[test_mask],
        sample_id_train=np.flatnonzero(train_mask),
        sample_id_test=np.flatnonzero(test_mask),
        feature_names=[f"feature_{index}" for index in range(4)],
        metadata={
            "protocol": "group_kfold_subject",
            "fold": 1,
            "fold_name": "fold_01",
            "split_type": "group_kfold_subject",
        },
    )
    runner = BenchmarkRunner({
        "output_dir": str(tmp_path),
        "datasets": {},
        "tasks": [],
        "models": {},
        "validation": {
            "strategy": "group_holdout",
            "group_column": "subject_id",
            "fraction": 0.2,
            "random_state": 42,
        },
    })
    model = _torch_mlp()

    result = runner._evaluate_split(
        model,
        split,
        "torch_mlp",
        dataset_name="synthetic",
        task_name="classification",
        artifact_split_name="fold_01",
    )

    audit_path = Path(result["artifacts"]["inner_validation_audit"])
    audit = pd.read_csv(audit_path)
    assert audit_path.is_file()
    assert audit.loc[0, "strategy"] == "group_holdout"
    assert audit.loc[0, "group_column"] == "subject_id"
    assert audit.loc[0, "inner_overlap_count"] == 0
    assert audit.loc[0, "train_outer_test_overlap_count"] == 0
    assert audit.loc[0, "validation_outer_test_overlap_count"] == 0
    assert audit.loc[0, "n_inner_train_groups"] >= 1
    assert audit.loc[0, "n_inner_validation_groups"] >= 1


def test_pm_group_validation_smoke_config_is_valid() -> None:
    config = yaml.safe_load(
        Path(
            "experiments/pm_regression/"
            "pm_regression_group_validation_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    assert validate_config(config)
    assert list(config["models"]) == ["torch_mlp"]
    assert config["validation"] == {
        "strategy": "group_holdout",
        "group_column": "subject_id",
        "fraction": 0.15,
        "random_state": 42,
    }
    assert "max_windows" not in config["datasets"]["emotiv_pm_regression"]
