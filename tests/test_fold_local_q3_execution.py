from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.core.abstract_dataset import EEGData
from bench.bench_runner import BenchmarkRunner
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from bench.tasks.target_transforms import validate_target_transform_manifest
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator


def _q3_data(target_id: str = "pm_attention_q3_fold_local") -> EEGData:
    n_subjects = 10
    rows_per_subject = 12
    n_rows = n_subjects * rows_per_subject
    subjects = np.repeat(
        [f"S{index:02d}" for index in range(n_subjects)], rows_per_subject
    )
    subject_index = np.repeat(np.arange(n_subjects), rows_per_subject)
    within_subject = np.tile(np.arange(rows_per_subject), n_subjects)
    values = (
        subject_index * 0.07
        + within_subject / (rows_per_subject - 1)
        + np.sin(within_subject) * 0.003
    ).astype(np.float32)
    outer_fold = subject_index // 2 + 1
    record_groups = np.asarray(
        [f"{subject}_R{position // 4}" for subject, position in zip(subjects, within_subject)]
    )
    rng = np.random.default_rng(42)
    return EEGData(
        data=rng.normal(size=(n_rows, 5)).astype(np.float32),
        labels=values,
        subject_ids=subjects,
        sample_ids=np.arange(10_000, 10_000 + n_rows),
        record_ids=record_groups.copy(),
        row_metadata={
            "outer_fold": outer_fold,
            "record_group_id": record_groups,
        },
        metadata={
            "target_id": target_id,
            "target_cols": [get_target_spec(target_id).processed_columns[0]],
            "task_type": "classification",
        },
    )


def _splits(data: EEGData):
    target_id = str(data.metadata["target_id"])
    task = get_task(target_id, data, {"target_id": target_id})
    return CrossValidator(task).run_group_kfold(
        group_column="subject_id",
        n_splits=5,
        precomputed_fold_column="outer_fold",
    )


def test_all_q3_targets_are_executable_and_use_the_correct_pm() -> None:
    for metric in PM_METRICS:
        spec = get_target_spec(f"pm_{metric}_q3_fold_local")
        assert spec.is_executable
        assert spec.processed_columns == (f"target_{metric}",)
        assert spec.requires_fold_local_transform
        assert spec.output_dim == 1
    assert not get_target_spec(
        "pm_focus_q5_fold_local", require_executable=False
    ).is_executable


def test_q3_fits_outer_train_once_and_saves_independent_fold_manifests() -> None:
    splits = _splits(_q3_data())
    assert len(splits) == 5
    hashes = []
    all_test_ids = []
    for fold_index, split in enumerate(splits.values(), start=1):
        assert set(np.unique(split.y_train)) == {0, 1, 2}
        assert set(np.unique(split.y_test)).issubset({0, 1, 2})
        assert set(split.subject_train).isdisjoint(set(split.subject_test))
        manifest = split.metadata["target_transform"]
        hashes.append(validate_target_transform_manifest(manifest))
        assert manifest["outer_fold"] == fold_index
        assert manifest["fit_scope"] == "outer_train_only"
        assert manifest["fit_sample_count"] == len(split.y_train)
        assert len(manifest["boundaries"]) == 4
        all_test_ids.extend(split.sample_id_test.tolist())
    assert len(set(hashes)) == 5
    assert len(all_test_ids) == len(set(all_test_ids)) == 120


def test_outer_test_values_do_not_change_q3_boundaries_or_hash() -> None:
    original = _q3_data()
    changed = copy.deepcopy(original)
    changed.labels = changed.labels.copy()
    changed.labels[changed.row_metadata["outer_fold"] == 1] += 1000.0
    original_manifest = _splits(original)["fold_01"].metadata["target_transform"]
    changed_manifest = _splits(changed)["fold_01"].metadata["target_transform"]
    assert original_manifest == changed_manifest


def test_outer_train_values_change_q3_boundaries_and_hash() -> None:
    original = _q3_data()
    changed = copy.deepcopy(original)
    changed.labels = changed.labels.copy()
    train_mask = changed.row_metadata["outer_fold"] != 1
    changed.labels[train_mask] = changed.labels[train_mask] ** 2
    original_manifest = _splits(original)["fold_01"].metadata["target_transform"]
    changed_manifest = _splits(changed)["fold_01"].metadata["target_transform"]
    assert original_manifest["boundaries"] != changed_manifest["boundaries"]
    assert original_manifest["transform_hash"] != changed_manifest["transform_hash"]


def test_q3_collapsed_boundaries_fail_without_fallback() -> None:
    data = _q3_data()
    train_mask = data.row_metadata["outer_fold"] != 1
    data.labels[train_mask] = 0.0
    with pytest.raises(ValueError, match="classes instead of 3"):
        _splits(data)


def test_resume_rejects_incompatible_target_transform_hash() -> None:
    manifest = _splits(_q3_data())["fold_01"].metadata["target_transform"]
    with pytest.raises(ValueError, match="Incompatible target transform"):
        validate_target_transform_manifest(
            manifest,
            expected_hash="0" * 64,
        )


def test_runner_rejects_incompatible_existing_fold_transform(tmp_path: Path) -> None:
    class NoopModel:
        pass

    splits = _splits(_q3_data())
    fold_one = splits["fold_01"]
    runner = BenchmarkRunner({"output_dir": str(tmp_path), "models": {}})
    artifacts = runner._save_split_artifacts(
        model=NoopModel(),
        split=fold_one,
        y_pred=fold_one.y_test,
        y_proba=np.eye(3, dtype=float)[fold_one.y_test],
        dataset_name="synthetic",
        task_name="pm_attention_q3_fold_local",
        model_name="noop",
        artifact_split_name="fold_01",
        metrics={},
    )
    transform_path = Path(artifacts["target_transform"])
    transform_path.write_text(
        json.dumps(splits["fold_02"].metadata["target_transform"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Incompatible target transform"):
        runner._save_split_artifacts(
            model=NoopModel(),
            split=fold_one,
            y_pred=fold_one.y_test,
            y_proba=np.eye(3, dtype=float)[fold_one.y_test],
            dataset_name="synthetic",
            task_name="pm_attention_q3_fold_local",
            model_name="noop",
            artifact_split_name="fold_01",
            metrics={},
        )
