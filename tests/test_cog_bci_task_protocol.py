"""Synthetic contract tests for native COG-BCI targets and splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.tasks.cog_bci_tasks import (
    COG_BCI_TASK_DEFINITIONS,
    COGBCITaskDefinition,
    build_cog_bci_target_index,
    get_cog_bci_task_definition,
    require_relative_path,
)
from bench.validation.cog_bci_protocol import (
    COGBCIProtocolConfig,
    build_cog_bci_protocol,
    materialize_cog_bci_protocol,
)
from bench.validation.cross_val import deterministic_group_kfold_indices


TASK_VARIANTS = {
    "n_back": ("zero_back", "one_back", "two_back"),
    "matb": ("matb_easy", "matb_medium", "matb_difficult"),
}


def _windows(
    *,
    subjects: int = 10,
    windows_per_record: int = 2,
    include_other_tasks: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index in range(subjects):
        subject = f"sub-{subject_index + 1:02d}"
        for family, variants in TASK_VARIANTS.items():
            for target, variant in enumerate(variants):
                record = f"{subject}_ses-01_{variant}"
                for window in range(windows_per_record):
                    rows.append(
                        {
                            "sample_id": f"{record}_w{window}",
                            "dataset": "cog_bci",
                            "source": "cog_bci",
                            "subject_id": subject,
                            "session_id": "ses-01",
                            "record_id": record,
                            "record_group_id": record,
                            "task_family": family,
                            "task_variant": variant,
                            "window_index": window,
                            "start_sample": window * 2560,
                            "stop_sample": (window + 1) * 2560,
                            "start_time_seconds": window * 5.12,
                            "stop_time_seconds": (window + 1) * 5.12,
                            "status": "accepted",
                            "_expected_target": target,
                        }
                    )
                rows.append(
                    {
                        "sample_id": f"{record}_tail",
                        "dataset": "cog_bci",
                        "source": "cog_bci",
                        "subject_id": subject,
                        "session_id": "ses-01",
                        "record_id": record,
                        "record_group_id": record,
                        "task_family": family,
                        "task_variant": variant,
                        "window_index": windows_per_record,
                        "start_sample": windows_per_record * 2560,
                        "stop_sample": windows_per_record * 2560 + 100,
                        "start_time_seconds": windows_per_record * 5.12,
                        "stop_time_seconds": windows_per_record * 5.12 + 0.2,
                        "status": "rejected_incomplete",
                        "_expected_target": target,
                    }
                )
    if include_other_tasks:
        rows.append(
            {
                "sample_id": "pvt_sample",
                "dataset": "cog_bci",
                "source": "cog_bci",
                "subject_id": "sub-01",
                "session_id": "ses-01",
                "record_id": "pvt_record",
                "record_group_id": "pvt_record",
                "task_family": "pvt",
                "task_variant": "pvt",
                "window_index": 0,
                "start_sample": 0,
                "stop_sample": 2560,
                "start_time_seconds": 0.0,
                "stop_time_seconds": 5.12,
                "status": "accepted",
                "_expected_target": -1,
            }
        )
    return pd.DataFrame(rows)


def _config(task_id: str, *, root: str = "cache") -> dict[str, object]:
    definition = get_cog_bci_task_definition(task_id)
    suffix = "nback" if "nback" in task_id else "matb"
    return {
        "dataset": "cog_bci",
        "window_cache": {"path": root, "config_hash": "cache-hash"},
        "task_id": task_id,
        "target_name": definition.target_name,
        "splitter": {
            "name": "group_kfold",
            "group_column": "subject_id",
            "n_splits": 5,
            "shuffle": False,
        },
        "inner_validation": {
            "name": "group_kfold_first_fold",
            "group_column": "subject_id",
            "n_splits": 5,
            "shuffle": False,
            "outer_train_only": True,
        },
        "loso": {"enabled": True, "group_column": "subject_id"},
        "output_dir": f"outputs/{suffix}",
    }


@pytest.mark.parametrize(
    ("task_id", "expected_variants", "target_name"),
    [
        (
            "cog_bci_nback_3class",
            ("zero_back", "one_back", "two_back"),
            "n_back_level",
        ),
        (
            "cog_bci_matb_3class",
            ("matb_easy", "matb_medium", "matb_difficult"),
            "matb_difficulty",
        ),
    ],
)
def test_task_definitions_are_ordered_ordinal_contracts(
    task_id: str,
    expected_variants: tuple[str, ...],
    target_name: str,
) -> None:
    definition = get_cog_bci_task_definition(task_id)
    assert definition.included_task_variants == expected_variants
    assert definition.class_to_index == {
        variant: index for index, variant in enumerate(expected_variants)
    }
    assert definition.target_name == target_name
    assert definition.target_type == "ordinal_classification"
    assert definition.ordered_classes is True
    assert definition.target_level == "record"


@pytest.mark.parametrize("task_id", sorted(COG_BCI_TASK_DEFINITIONS))
def test_target_is_inherited_by_all_windows_of_its_record(
    task_id: str,
) -> None:
    index = build_cog_bci_target_index(_windows(), task_id)
    assert index.frame.groupby("record_id")["target"].nunique().max() == 1
    assert index.frame.groupby("record_id")["task_variant"].nunique().max() == 1
    definition = get_cog_bci_task_definition(task_id)
    assert (
        index.frame["target"]
        == index.frame["task_variant"].map(definition.class_to_index)
    ).all()


@pytest.mark.parametrize(
    ("task_id", "excluded"),
    [
        ("cog_bci_nback_3class", {"pvt", "flanker", "matb_easy"}),
        ("cog_bci_matb_3class", {"pvt", "flanker", "zero_back"}),
    ],
)
def test_unrelated_variants_receive_no_target(
    task_id: str, excluded: set[str]
) -> None:
    index = build_cog_bci_target_index(_windows(), task_id)
    assert excluded.isdisjoint(set(index.frame["task_variant"]))


@pytest.mark.parametrize("task_id", sorted(COG_BCI_TASK_DEFINITIONS))
def test_rejected_tail_is_labelled_but_not_supervised(task_id: str) -> None:
    index = build_cog_bci_target_index(_windows(), task_id)
    rejected = index.frame.loc[index.frame["status"].eq("rejected_incomplete")]
    assert not rejected.empty
    assert rejected["target"].notna().all()
    assert not rejected["included_for_supervised"].any()


def test_record_cannot_receive_two_task_labels() -> None:
    windows = _windows()
    selected = windows["task_variant"].eq("one_back")
    windows.loc[selected, "record_id"] = "sub-01_ses-01_zero_back"
    with pytest.raises(ValueError, match="two labels"):
        build_cog_bci_target_index(windows, "cog_bci_nback_3class")


@pytest.mark.parametrize("column", ["record_id", "record_group_id"])
def test_missing_record_identity_is_rejected(column: str) -> None:
    windows = _windows()
    windows.loc[0, column] = ""
    with pytest.raises(ValueError, match=column):
        build_cog_bci_target_index(windows, "cog_bci_nback_3class")


def test_duplicate_sample_id_is_rejected() -> None:
    windows = _windows()
    windows.loc[1, "sample_id"] = windows.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_cog_bci_target_index(windows, "cog_bci_nback_3class")


def test_target_index_hash_is_stable_under_input_row_order() -> None:
    windows = _windows()
    forward = build_cog_bci_target_index(
        windows, "cog_bci_nback_3class"
    )
    shuffled = build_cog_bci_target_index(
        windows.sample(frac=1.0, random_state=9),
        "cog_bci_nback_3class",
    )
    assert forward.target_index_hash == shuffled.target_index_hash


def test_changed_target_schema_invalidates_hash() -> None:
    original = get_cog_bci_task_definition("cog_bci_nback_3class")
    changed = COGBCITaskDefinition(
        task_id=original.task_id,
        task_family=original.task_family,
        target_name=original.target_name,
        target_type=original.target_type,
        class_names=original.class_names,
        class_to_index_items=(
            ("zero_back", 0),
            ("two_back", 1),
            ("one_back", 2),
        ),
        ordered_classes=original.ordered_classes,
        included_task_variants=("zero_back", "two_back", "one_back"),
        excluded_task_variants=original.excluded_task_variants,
    )
    first = build_cog_bci_target_index(
        _windows(), "cog_bci_nback_3class"
    )
    second = type(first).from_window_index(_windows(), changed)
    assert first.target_index_hash != second.target_index_hash
    assert first.definition.schema_hash != second.definition.schema_hash


@pytest.mark.parametrize("value", ["F:/EEG/cache", r"C:\cache", "../cache"])
def test_machine_specific_or_escaping_paths_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        require_relative_path(value, label="cache")


def test_protocol_config_rejects_training_fields() -> None:
    config = _config("cog_bci_nback_3class")
    config["model"] = "torch_eegnet"
    with pytest.raises(ValueError, match="training fields"):
        COGBCIProtocolConfig.from_mapping(config)


def test_deterministic_group_kfold_helper_is_stable_and_disjoint() -> None:
    groups = np.repeat([f"s{i}" for i in range(10)], 3)
    first = deterministic_group_kfold_indices(groups, n_splits=5)
    second = deterministic_group_kfold_indices(groups, n_splits=5)
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)
        assert set(groups[train_a]).isdisjoint(groups[test_a])


@pytest.mark.parametrize("task_id", sorted(COG_BCI_TASK_DEFINITIONS))
def test_protocol_builds_outer_inner_and_loso_manifests(task_id: str) -> None:
    target = build_cog_bci_target_index(_windows(), task_id)
    result = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-hash",
        window_index_sha256="index-hash",
    )
    assert len(result.outer_folds["folds"]) == 5
    assert len(result.inner_folds["folds"]) == 5
    assert len(result.loso_folds["folds"]) == 10
    assert result.protocol_summary["all_outer_folds_leakage_safe"]
    assert result.protocol_summary["all_inner_folds_leakage_safe"]
    assert result.protocol_summary["all_loso_folds_leakage_safe"]
    assert result.protocol_summary["training_performed"] is False
    assert result.protocol_summary["scaler_fitted"] is False
    assert result.protocol_summary["sampler_used"] is False


def test_each_subject_and_sample_has_exactly_one_outer_test_fold() -> None:
    target = build_cog_bci_target_index(
        _windows(), "cog_bci_nback_3class"
    )
    result = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-hash",
        window_index_sha256="index-hash",
    )
    assignments = result.outer_assignments
    assert not assignments["sample_id"].duplicated().any()
    assert assignments.groupby("subject_id")["fold"].nunique().max() == 1
    assert assignments.groupby("record_id")["fold"].nunique().max() == 1
    assert set(assignments["sample_id"]) == set(target.accepted["sample_id"])


def test_inner_split_excludes_outer_test_and_is_subject_disjoint() -> None:
    target = build_cog_bci_target_index(
        _windows(), "cog_bci_nback_3class"
    )
    result = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-hash",
        window_index_sha256="index-hash",
    )
    assignments = result.inner_assignments
    for _, fold in assignments.groupby("outer_fold"):
        train = fold.loc[fold["partition"].eq("inner_train")]
        validation = fold.loc[fold["partition"].eq("inner_validation")]
        test = fold.loc[fold["partition"].eq("outer_test_excluded")]
        assert set(train["subject_id"]).isdisjoint(validation["subject_id"])
        assert set(train["record_group_id"]).isdisjoint(
            validation["record_group_id"]
        )
        assert set(train["sample_id"]).isdisjoint(validation["sample_id"])
        assert set(test["subject_id"]).isdisjoint(
            set(train["subject_id"]) | set(validation["subject_id"])
        )


def test_split_hash_is_stable_and_depends_on_cache_hash() -> None:
    target = build_cog_bci_target_index(
        _windows(), "cog_bci_nback_3class"
    )
    first = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-a",
        window_index_sha256="index-hash",
    )
    repeat = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-a",
        window_index_sha256="index-hash",
    )
    changed = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-b",
        window_index_sha256="index-hash",
    )
    assert (
        first.protocol_summary["protocol_hash"]
        == repeat.protocol_summary["protocol_hash"]
    )
    assert (
        first.protocol_summary["protocol_hash"]
        != changed.protocol_summary["protocol_hash"]
    )


def test_class_balance_is_computed_at_record_and_window_levels() -> None:
    target = build_cog_bci_target_index(
        _windows(windows_per_record=3), "cog_bci_nback_3class"
    )
    result = build_cog_bci_protocol(
        target,
        window_cache_config_hash="cache-hash",
        window_index_sha256="index-hash",
    )
    assert result.class_balance["record_distribution"] == {
        "0": 10,
        "1": 10,
        "2": 10,
    }
    assert result.class_balance["accepted_window_distribution"] == {
        "0": 30,
        "1": 30,
        "2": 30,
    }


def test_materialization_writes_expected_relative_artifacts(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    windows = _windows()
    windows.to_parquet(cache / "window_index.parquet", index=False)
    (cache / "dataset_manifest.json").write_text(
        json.dumps({"config_hash": "cache-hash"}), encoding="utf-8"
    )
    config = _config("cog_bci_nback_3class")
    result = materialize_cog_bci_protocol(config, repository_root=tmp_path)
    output = tmp_path / "outputs" / "nback"
    expected = {
        "task_definition.json",
        "target_index.parquet",
        "record_target_summary.csv",
        "window_target_summary.csv",
        "class_balance.json",
        "outer_folds.json",
        "outer_assignments.parquet",
        "inner_folds.json",
        "inner_assignments.parquet",
        "loso_folds.json",
        "protocol_summary.json",
        "protocol_report.md",
        "errors.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert result.protocol_summary["model_used"] is False
    assert not any(
        str(tmp_path) in path.read_text(encoding="utf-8")
        for path in output.glob("*.json")
    )


def test_materialization_does_not_modify_window_index(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "window_index.parquet"
    _windows().to_parquet(path, index=False)
    before = path.read_bytes()
    (cache / "dataset_manifest.json").write_text(
        json.dumps({"config_hash": "cache-hash"}), encoding="utf-8"
    )
    materialize_cog_bci_protocol(
        _config("cog_bci_matb_3class"), repository_root=tmp_path
    )
    assert path.read_bytes() == before


def test_config_hash_mismatch_fails_before_artifact_write(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _windows().to_parquet(cache / "window_index.parquet", index=False)
    (cache / "dataset_manifest.json").write_text(
        json.dumps({"config_hash": "other"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="config hash mismatch"):
        materialize_cog_bci_protocol(
            _config("cog_bci_nback_3class"), repository_root=tmp_path
        )
    assert not (tmp_path / "outputs").exists()
