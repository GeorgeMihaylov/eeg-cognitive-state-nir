from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bench.analysis import project_final_package as package


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_inventory_is_deterministic() -> None:
    first = package.build_inventory(ROOT)
    second = package.build_inventory(ROOT)
    assert first == second
    assert package._csv_text(first, package.INVENTORY_COLUMNS) == package._csv_text(
        copy.deepcopy(second), package.INVENTORY_COLUMNS
    )


def test_inventory_has_no_duplicate_experiment_ids() -> None:
    rows = package.build_inventory(ROOT)
    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_primary_results_have_provenance_or_are_excluded() -> None:
    inventory = package.build_inventory(ROOT)
    audit = package.build_provenance_audit(ROOT, inventory)
    for row in audit:
        if row["evidence_role"] == "primary":
            assert row["complete"] == "true"
        if row["complete"] != "true":
            assert row["evidence_role"] == "supporting_only"


def test_different_tasks_are_not_aggregated_together() -> None:
    classification = package.build_inventory(ROOT)
    targets = {
        row["task"]: row["target"]
        for row in classification
        if row["experiment_id"] in {
            "label_q5_random_forest_groupkfold",
            "pm_regression_random_forest_5fold",
            "cog_bci_nback_cnn_baseline",
        }
    }
    assert targets["cognitive_load_5class"] == "label_q5"
    assert targets["performance_metrics_regression"] == "performance_metrics_7"
    assert targets["nback_3class"] == "nback_load"


def test_window_and_record_metrics_are_explicitly_separated() -> None:
    rows = package.build_external_results(ROOT)
    assert {row["analysis_unit"] for row in rows} == {"window", "record"}
    assert all(row["analysis_unit"] in {"window", "record"} for row in rows)


def test_absolute_paths_are_rejected() -> None:
    with pytest.raises(package.FinalPackageError, match="Absolute path"):
        package._relative(r"F:\EEG\data")


def test_requirement_statuses_belong_to_final_vocabulary() -> None:
    rows = package.build_requirement_rows(ROOT)
    assert rows
    assert {row["status"] for row in rows} <= package.REQUIREMENT_STATUSES


def test_closed_negative_tracks_are_marked() -> None:
    inventory = {
        row["experiment_id"]: row for row in package.build_inventory(ROOT)
    }
    for experiment_id in (
        "label_q5_auxiliary_corn_policy",
        "cog_bci_nback_cnn_baseline",
        "cog_bci_preprocessing_ablation",
        "cog_bci_spectral_14_vs_62",
        "cog_bci_shape_only_contrastive_transfer",
        "cog_bci_time_aligned_contrastive_transfer",
    ):
        assert inventory[experiment_id]["status"] == "closed_negative"


def test_tracked_report_links_exist() -> None:
    for row in package.build_inventory(ROOT):
        report_path = row["report_path"]
        assert report_path
        assert (ROOT / report_path).is_file(), row["experiment_id"]


def test_inventory_has_required_columns_and_statuses() -> None:
    rows = package.build_inventory(ROOT)
    assert rows
    assert set(rows[0]) == set(package.INVENTORY_COLUMNS)
    assert {row["status"] for row in rows} <= package.FINAL_STATUSES
