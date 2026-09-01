from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pandas as pd
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


NEW_EXPERIMENT_IDS = {
    "meta_learning_episode_infrastructure_v1",
    "fomaml_synthetic_contract_v1",
    "fomaml_production_buffer_contract_v1",
    "fomaml_label_q5_feature_protocol_blocked_v1",
    "fomaml_label_q5_raw_protocol_v2",
    "fomaml_label_q5_raw_deduplicated_diagnostic_v1",
    "dann_label_q5_raw_protocol_v1",
    "dann_label_q5_old_eeg_to_gpn_diagnostic_v1",
    "dann_label_q5_confirmatory_v1_protocol",
    "dann_label_q5_confirmatory_v2_protocol",
    "dann_label_q5_old_eeg_to_gpn_confirmatory_v2_execution",
}


def test_new_meta_and_domain_experiments_are_registered() -> None:
    inventory = {row["experiment_id"] for row in package.build_inventory(ROOT)}
    assert NEW_EXPERIMENT_IDS <= inventory
    statuses = package.build_experiment_statuses(ROOT)
    assert set(package.STATUS_COLUMNS) == set(statuses[0])
    assert all(row["protocol_hash"] for row in statuses)
    assert all(row["preregistration_hash"] for row in statuses)
    assert all((ROOT / row["result_artifact"]).is_dir() for row in statuses)


def test_fomaml_participant_result_and_decision() -> None:
    rows = {row["method"]: row for row in package.build_meta_learning_results(ROOT)}
    selected = rows["selected_fomaml"]
    assert selected["status"] == "do_not_proceed"
    assert selected["delta_macro_f1_vs_supervised_full_model"] == pytest.approx(-0.046338, abs=1e-6)
    assert selected["delta_balanced_accuracy_vs_supervised_full_model"] == pytest.approx(0.039053, abs=1e-6)
    assert selected["delta_ordinal_mae_vs_supervised_full_model"] == pytest.approx(0.449093, abs=1e-6)
    assert (selected["macro_f1_wins"], selected["macro_f1_losses"], selected["macro_f1_ties"]) == (1, 4, 0)


def test_dann_primary_result_is_not_replaced_by_sensitivity() -> None:
    rows = package.build_domain_adaptation_results(ROOT)
    primary = next(row for row in rows if row["analysis_group"] == "primary_confirmatory" and row["method"] == "dann")
    assert primary["status"] == "partially_confirmed"
    assert primary["delta_macro_f1"] == pytest.approx(0.008048, abs=1e-6)
    assert primary["delta_balanced_accuracy"] == pytest.approx(0.008332, abs=1e-6)
    assert primary["delta_ordinal_mae"] == pytest.approx(-0.034008, abs=1e-6)
    seeds = package.build_domain_adaptation_seed_results(ROOT)
    primary_seeds = {row["seed"] for row in seeds if row["included_in_primary_decision"]}
    sensitivity = [row for row in seeds if not row["included_in_primary_decision"]]
    assert primary_seeds == {123, 2026}
    assert len(sensitivity) == 1 and sensitivity[0]["seed"] == 42
    assert sensitivity[0]["analysis_group"] == "sensitivity"
    assert sensitivity[0]["diagnostic_fold_1_reused"] is True


def test_diagnostic_fold_one_seed_42_is_not_primary() -> None:
    seeds = package.build_domain_adaptation_seed_results(ROOT)
    assert not any(row["seed"] == 42 and row["included_in_primary_decision"] for row in seeds)
    diagnostic = [
        row for row in package.build_domain_adaptation_results(ROOT)
        if row["analysis_group"] == "diagnostic"
    ]
    assert {row["folds"] for row in diagnostic} == {"1"}
    assert {row["seeds"] for row in diagnostic} == {"42"}


def test_new_reports_exist_and_runtime_outputs_are_untracked() -> None:
    statuses = package.build_experiment_statuses(ROOT)
    registry = package._load_yaml(ROOT / "reports/summary/experiment_registry.yaml")
    by_id = {row["experiment_id"]: row for row in registry["experiments"]}
    assert all((ROOT / by_id[row["experiment_id"]]["report_path"]).is_file() for row in statuses)
    tracked = subprocess.run(
        ["git", "ls-files", "benchmark_results"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert tracked == ""


def test_generated_svg_has_lf_and_no_trailing_whitespace() -> None:
    package.generate(ROOT)
    svg_paths = sorted((ROOT / "reports/summary/final_result_tables/figures").glob("*.svg"))
    assert len(svg_paths) == 14
    for path in svg_paths:
        data = path.read_bytes()
        assert b"\r\n" not in data
        assert all(line == line.rstrip() for line in data.splitlines())


def test_final_csv_generation_is_byte_identical() -> None:
    first = package.generate(ROOT)
    table_paths = [ROOT / path for path in first["outputs"] if path.endswith(".csv")]
    before = {path: path.read_bytes() for path in table_paths}
    second = package.generate(ROOT)
    assert first == second
    assert before == {path: path.read_bytes() for path in table_paths}
    required = {
        "final_meta_learning_results.csv",
        "final_domain_adaptation_results.csv",
        "final_domain_adaptation_fold_results.csv",
        "final_domain_adaptation_seed_results.csv",
        "final_experiment_statuses.csv",
        "final_result_inventory.csv",
    }
    assert required <= {path.name for path in table_paths}
    result_inventory = pd.read_csv(
        ROOT / "reports/summary/final_result_tables/final_result_inventory.csv",
        encoding="utf-8-sig",
    )
    listed = set(result_inventory["artifact_path"])
    assert {
        "reports/summary/final_project_results.md",
        "reports/summary/final_result_tables/final_meta_learning_results.csv",
        "reports/summary/final_result_tables/final_domain_adaptation_results.csv",
        "reports/summary/final_result_tables/figures/14_evidence_status_map.svg",
    } <= listed


def test_final_tables_preserve_participant_analysis_level() -> None:
    domain = pd.DataFrame(package.build_domain_adaptation_results(ROOT))
    meta = pd.DataFrame(package.build_meta_learning_results(ROOT))
    assert set(domain["analysis_level"]) == {"participant"}
    assert set(meta["analysis_level"]) == {"participant"}
