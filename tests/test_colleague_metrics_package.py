"""Contract tests for the deterministic colleague metrics package.

These tests read existing summary artifacts only; they never start model
training or recompute metrics from predictions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from bench.analysis import colleague_metrics_package as PACKAGE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provenance():
    return PACKAGE.load_yaml(ROOT / "reports" / "summary" / "metrics_provenance.yaml")


@pytest.fixture(scope="module")
def experiment_registry():
    return PACKAGE.load_yaml(ROOT / "reports" / "summary" / "experiment_registry.yaml")


@pytest.fixture(scope="module")
def config_registry():
    return PACKAGE.load_yaml(ROOT / "reports" / "summary" / "config_registry.yaml")


@pytest.fixture(scope="module")
def tables(provenance, experiment_registry):
    return PACKAGE.build_tables(provenance, experiment_registry, ROOT)[0]


def _csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "reports" / "summary" / name).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _generate_twice(tmp_path: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    output = tmp_path / "summary"
    output.mkdir()
    source = ROOT / "reports" / "summary" / "metrics_provenance.yaml"
    (output / "metrics_provenance.yaml").write_bytes(source.read_bytes())
    kwargs = dict(
        experiment_registry_path=ROOT
        / "reports"
        / "summary"
        / "experiment_registry.yaml",
        config_registry_path=ROOT / "reports" / "summary" / "config_registry.yaml",
        output_dir=output,
        strict=True,
        repo_root=ROOT,
    )
    PACKAGE.generate_package(**kwargs)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    PACKAGE.generate_package(**kwargs)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    return first, second


def test_01_classification_and_regression_are_separate(tables):
    assert all("macro_mae_mean" not in row for row in tables["classification"])
    assert all("macro_f1_mean" not in row for row in tables["pm_regression"])


def test_02_smoke_is_excluded_from_main_tables(tables):
    rows = tables["classification"] + tables["pm_regression"]
    assert all(row["result_status"] != "smoke" for row in rows)


def test_03_invalidated_is_excluded_from_rankings(tables):
    rows = tables["classification"] + tables["pm_regression"]
    assert all(row["result_status"] != "invalidated" for row in rows)


def test_04_diagnostic_is_marked(tables):
    assert {row["result_status"] for row in tables["preprocessing"]} == {
        "diagnostic"
    }


def test_05_unknown_experiment_id_is_rejected(
    provenance, experiment_registry, config_registry
):
    bad = copy.deepcopy(provenance)
    bad["classification"][0]["experiment_id"] = "not_registered"
    with pytest.raises(PACKAGE.PackageValidationError, match="unknown experiment ID"):
        PACKAGE.validate_provenance(bad, experiment_registry, config_registry, ROOT)


def test_06_missing_metric_source_is_rejected(
    provenance, experiment_registry, config_registry
):
    bad = copy.deepcopy(provenance)
    bad["classification"][0]["metric_source_path"] = "reports/does_not_exist.csv"
    with pytest.raises(PACKAGE.PackageValidationError, match="missing metric source"):
        PACKAGE.validate_provenance(bad, experiment_registry, config_registry, ROOT)


def test_07_absolute_path_is_rejected(
    provenance, experiment_registry, config_registry
):
    bad = copy.deepcopy(provenance)
    bad["classification"][0]["report_path"] = r"F:\EEG\reports\bad.md"
    with pytest.raises(PACKAGE.PackageValidationError, match="Absolute path"):
        PACKAGE.validate_provenance(bad, experiment_registry, config_registry, ROOT)


def test_08_pm_target_order_is_checked(
    provenance, experiment_registry, config_registry
):
    bad = copy.deepcopy(provenance)
    bad["pm_target_order"] = list(reversed(bad["pm_target_order"]))
    with pytest.raises(PACKAGE.PackageValidationError, match="PM target order"):
        PACKAGE.validate_provenance(bad, experiment_registry, config_registry, ROOT)


def test_09_mae_gain_direction_is_checked(tables, provenance):
    bad = copy.deepcopy(tables)
    row = next(
        item
        for item in bad["personalization"]
        if item["task"] == "pm_regression" and item["method"] == "full_model"
    )
    row["absolute_gain"] = -row["absolute_gain"]
    with pytest.raises(PACKAGE.PackageValidationError, match="gain direction"):
        PACKAGE.validate_generated_tables(bad, provenance)


def test_10_negative_r2_is_preserved():
    row = next(item for item in _csv("pm_regression_metrics_unified.csv") if item["model"] == "mean_regressor")
    assert float(row["macro_r2_mean"]) < 0


def test_11_missing_metrics_remain_empty():
    pm = next(item for item in _csv("pm_regression_metrics_unified.csv") if item["model"] == "mean_regressor")
    prep = next(item for item in _csv("preprocessing_metrics_unified.csv") if item["trial_id"] == "standard_clip")
    assert pm["macro_pearson_mean"] == ""
    assert prep["balanced_accuracy_mean"] == ""


def test_12_single_seed_and_multiseed_are_marked(tables):
    seeds = {row["model"]: row["seeds"] for row in tables["classification"]}
    assert seeds["Random Forest"] == "42"
    assert seeds["Transformer"] == "7|42|123"


def test_13_raw_and_feature_inputs_are_distinct(tables):
    inputs = {row["model"]: row["input_type"] for row in tables["classification"]}
    assert inputs["EEGNet"] == "raw_eeg_window"
    assert inputs["Random Forest"] == "feature_window"
    assert inputs["Transformer"] == "feature_sequence"


def test_14_random_split_is_not_mixed_with_groupkfold(tables):
    protocols = {row["evaluation_protocol"] for row in tables["classification"]}
    assert protocols == {"5-fold GroupKFold by subject_id"}


def test_15_unresolved_results_have_no_invented_metrics(provenance):
    assert len(provenance["unresolved_results"]) == 4
    assert all(not any(key.endswith("_mean") for key in row) for row in provenance["unresolved_results"])


def test_16_classification_contains_seven_confirmed_models(tables):
    assert len(tables["classification"]) == 7


def test_17_pm_contains_mean_and_rf_baselines(tables):
    assert {row["model"] for row in tables["pm_regression"]} == {
        "mean_regressor",
        "random_forest",
    }


def test_18_personalization_contains_head_and_full(tables):
    for task in ("classification", "pm_regression"):
        methods = {
            row["method"]
            for row in tables["personalization"]
            if row["task"] == task
        }
        assert {"head_only", "full_model"} <= methods


def test_19_preprocessing_contains_trials_a_through_h(tables):
    trials = {row["trial_id"] for row in tables["preprocessing"]}
    assert set("ABCDEFGH") <= trials


def test_20_standard_clip_is_diagnostic(tables):
    row = next(
        item for item in tables["preprocessing"] if item["trial_id"] == "standard_clip"
    )
    assert row["result_status"] == "diagnostic"
    assert row["n_folds"] == 1


def test_21_transfer_is_reimplemented(provenance):
    row = next(
        item
        for item in provenance["mixins"]
        if item["method"] == "Transfer learning"
    )
    assert row["integrated"] == "integrated_as_reimplemented_pipeline"


def test_22_other_mixins_are_not_integrated(provenance):
    rows = [
        row
        for row in provenance["mixins"]
        if row["method"] != "Transfer learning"
    ]
    assert {row["method"] for row in rows} == {
        "Domain adaptation",
        "Meta-learning",
        "Contrastive learning",
    }
    assert all(row["integrated"] == "not_integrated" for row in rows)


def test_23_generated_markdown_is_deterministic(tmp_path):
    first, second = _generate_twice(tmp_path)
    for name in ("colleague_metrics_summary.md", "metrics_glossary.md"):
        assert first[name] == second[name]


def test_24_generated_csv_files_are_deterministic(tmp_path):
    first, second = _generate_twice(tmp_path)
    csv_names = {
        "classification_metrics_unified.csv",
        "pm_regression_metrics_unified.csv",
        "personalization_metrics_unified.csv",
        "preprocessing_metrics_unified.csv",
    }
    assert {name: hashlib.sha256(first[name]).digest() for name in csv_names} == {
        name: hashlib.sha256(second[name]).digest() for name in csv_names
    }


def test_25_provenance_yaml_is_deterministic(tmp_path):
    first, second = _generate_twice(tmp_path)
    assert first["metrics_provenance.yaml"] == second["metrics_provenance.yaml"]


def test_26_utf8_is_preserved():
    text = (ROOT / "reports" / "summary" / "colleague_metrics_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Единая сводка" in text
    assert "Персонализация" in text


def test_27_current_package_passes_strict_validation(
    provenance, experiment_registry, config_registry, tables
):
    PACKAGE.validate_provenance(
        provenance, experiment_registry, config_registry, ROOT, strict=True
    )
    PACKAGE.validate_generated_tables(tables, provenance)
