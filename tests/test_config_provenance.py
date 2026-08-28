from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.analysis import experiment_config_audit as audit
from bench.analysis import experiment_summary as summary
from bench.datasets.datasets_registry import get_dataset
from bench.tasks.tasks_registry import get_task
from cli import load_config, validate_config
from cogstate.model_zoo.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
PM_CONFIG = (
    ROOT
    / "experiments"
    / "pm_regression"
    / "pm_regression_rf_groupkfold_full.yaml"
)
CURATION_PATH = ROOT / "reports" / "summary" / "config_curation.yaml"
REGISTRY_PATH = ROOT / "reports" / "summary" / "experiment_registry.yaml"
PM_TARGETS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]


@pytest.fixture(scope="session")
def pm_config() -> dict:
    return load_config(str(PM_CONFIG))


@pytest.fixture(scope="session")
def curation() -> dict:
    return yaml.safe_load(CURATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def curation_by_path(curation: dict) -> dict[str, dict]:
    return {item["config_path"]: item for item in curation["configs"]}


@pytest.fixture(scope="session")
def registry_by_id(registry: dict) -> dict[str, dict]:
    return {item["experiment_id"]: item for item in registry["experiments"]}


@pytest.fixture(scope="session")
def project_audit():
    return audit.audit_repository(ROOT, curation_path=CURATION_PATH)


@pytest.fixture()
def synthetic_pm_data(tmp_path: Path, pm_config: dict):
    rng = np.random.default_rng(42)
    rows = 12
    frame = pd.DataFrame(
        {
            **{f"EEG.f{i:03d}": rng.normal(size=rows) for i in range(224)},
            **{f"POW.f{i:03d}": rng.normal(size=rows) for i in range(224)},
            **{target: rng.uniform(size=rows) for target in PM_TARGETS},
            "subject_id": [f"s{i // 3}" for i in range(rows)],
            "record_id": [f"r{i // 2}" for i in range(rows)],
            "sample_id": np.arange(rows),
        }
    )
    path = tmp_path / "pm.parquet"
    frame.to_parquet(path, index=False)
    config = deepcopy(pm_config)
    config["datasets"]["emotiv_pm_regression"]["data_path"] = str(path)
    dataset = get_dataset(
        "emotiv_pm_regression",
        config["datasets"]["emotiv_pm_regression"],
    )
    return config, dataset.load()


def _curation_item(path: str, **updates) -> dict:
    item = {
        "config_path": path,
        "review_status": "reviewed",
        "decision": "keep",
        "decision_reason": "test evidence",
        "canonical_config": path,
        "safe_to_move": False,
        "safe_to_edit": False,
        "evidence": ["commit:946126c"],
    }
    item.update(updates)
    return item


def _validate_one(tmp_path: Path, item: dict, *, extra_records=None) -> None:
    path = item["config_path"]
    records = {
        path: audit.ConfigRecord(
            path=path,
            document={"datasets": {}, "models": {}, "tasks": []},
            loader_type="benchmark_config",
            role="full",
            status="unclassified",
            schema_valid=True,
        )
    }
    records[path].extracted = {}
    records.update(extra_records or {})
    document = {
        "schema_version": 1,
        "families": {
            "test": {
                "canonical_config": path,
                "canonical_smoke_config": None,
                "base_configs": [],
                "legacy_configs": [],
                "diagnostic_configs": [],
                "protected_configs": [],
                "decision_reason": "test family",
                "evidence": ["commit:946126c"],
            }
        },
        "configs": [item],
        "duplicate_groups": [],
        "seed_provenance": [],
        "normalization_plan": [],
    }
    curation_path = tmp_path / "curation.yaml"
    curation_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    audit.load_and_validate_curation(ROOT, curation_path, records)


def test_01_pm_family_has_canonical_full_config(curation: dict) -> None:
    assert curation["families"]["pm_regression"]["canonical_config"] == (
        "experiments/pm_regression/pm_regression_rf_groupkfold_full.yaml"
    )


def test_02_pm_targets_use_canonical_order(pm_config: dict) -> None:
    assert pm_config["datasets"]["emotiv_pm_regression"]["target_cols"] == PM_TARGETS


def test_03_pm_config_has_seven_targets(pm_config: dict) -> None:
    assert len(pm_config["datasets"]["emotiv_pm_regression"]["target_cols"]) == 7
    assert pm_config["datasets"]["emotiv_pm_regression"]["n_outputs"] == 7


def test_04_pm_targets_are_not_features(synthetic_pm_data) -> None:
    _, data = synthetic_pm_data
    assert data.data.shape == (12, 448)
    assert not set(PM_TARGETS) & set(data.feature_names)
    assert not any(name.startswith(("target_", "PM.")) for name in data.feature_names)


def test_05_pm_uses_subject_groupkfold(pm_config: dict) -> None:
    assert pm_config["evaluation"] == {
        "protocol": "group_kfold_subject",
        "group_column": "subject_id",
        "n_splits": 5,
        "folds": [1, 2, 3, 4, 5],
        "random_state": 42,
    }


def test_06_pm_config_loads_with_current_loader(
    pm_config: dict, synthetic_pm_data
) -> None:
    config, data = synthetic_pm_data
    assert validate_config(pm_config)
    task = get_task("performance_metrics_regression", data, config["task_config"])
    assert task.task_type == "regression"
    assert data.labels.shape == (12, 7)


def test_07_pm_config_builds_random_forest(pm_config: dict) -> None:
    model = build_model(
        model_name=pm_config["models"]["random_forest"]["type"],
        task_type="regression",
        input_shape=(448,),
        num_outputs=7,
        params=pm_config["models"]["random_forest"]["params"],
    )
    assert model.n_estimators == 20
    assert model.max_depth == 8


def test_08_pm_smoke_is_not_final(curation_by_path: dict[str, dict]) -> None:
    item = curation_by_path["experiments/pm_regression/pm_regression_smoke.yaml"]
    assert item["result_status"] == "smoke"
    assert item.get("canonical_status") != "completed"


def test_09_lstm_curation_has_historical_decision(
    curation_by_path: dict[str, dict],
) -> None:
    for path in (
        "configs/groupkfold_torch_lstm_label_q5.yaml",
        "configs/groupkfold_torch_bilstm_label_q5.yaml",
    ):
        item = curation_by_path[path]
        assert item["decision"] == "keep_as_legacy"
        assert item["canonical_status"] == "historical"
        assert item["provenance_status"] == "documented"


def test_10_superseded_requires_evidence(tmp_path: Path) -> None:
    item = _curation_item(
        "configs/a.yaml",
        decision="superseded",
        superseded_by="configs/b.yaml",
        supersession_reason="replaced",
        evidence=[],
    )
    target = audit.ConfigRecord(
        path="configs/b.yaml",
        document={},
        loader_type="benchmark_config",
        role="full",
    )
    with pytest.raises(audit.CurationValidationError, match="evidence"):
        _validate_one(tmp_path, item, extra_records={"configs/b.yaml": target})


def test_11_superseded_by_requires_existing_config(tmp_path: Path) -> None:
    item = _curation_item(
        "configs/a.yaml",
        decision="superseded",
        superseded_by="configs/missing.yaml",
        supersession_reason="replaced",
    )
    with pytest.raises(audit.CurationValidationError, match="unknown superseded_by"):
        _validate_one(tmp_path, item)


def test_12_canonical_status_rejects_unknown_value(tmp_path: Path) -> None:
    item = _curation_item("configs/a.yaml", canonical_status="finished")
    with pytest.raises(audit.CurationValidationError, match="unknown canonical_status"):
        _validate_one(tmp_path, item)


def test_13_completed_canonical_requires_report_or_runtime(tmp_path: Path) -> None:
    item = _curation_item("configs/a.yaml", canonical_status="completed")
    with pytest.raises(
        audit.CurationValidationError,
        match="completed canonical requires",
    ):
        _validate_one(tmp_path, item)


def test_14_planned_canonical_is_not_completed(tmp_path: Path) -> None:
    item = _curation_item(
        "configs/a.yaml",
        canonical_status="planned",
        result_status="final",
    )
    with pytest.raises(
        audit.CurationValidationError,
        match="planned canonical cannot represent",
    ):
        _validate_one(tmp_path, item)


def test_15_transformer_provenance_has_all_seeds(curation: dict) -> None:
    item = next(value for value in curation["seed_provenance"] if value["name"] == "Transformer")
    assert item["provenance_status"] == "documented"
    assert set(item["orchestration_provenance"]["runs"]) == {7, 42, 123}


def test_16_preprocessing_provenance_has_all_seeds(curation: dict) -> None:
    item = next(
        value
        for value in curation["seed_provenance"]
        if value["name"] == "preprocessing_ablation"
    )
    assert item["provenance_status"] == "documented"
    assert set(item["orchestration_provenance"]["runs"]) == {7, 42, 123}


def test_17_external_orchestration_is_explicit(curation: dict) -> None:
    items = {
        value["name"]: value for value in curation["seed_provenance"]
    }
    assert items["Transformer"]["orchestration_provenance"]["mode"]
    assert "--experiment-matrix" in (
        items["preprocessing_ablation"]["orchestration_provenance"]["source"]
    )


def test_18_registry_seed_metadata_matches_provenance(
    curation: dict, registry_by_id: dict[str, dict]
) -> None:
    transformer = registry_by_id["label_q5_transformer_multiseed"]
    preprocessing = registry_by_id["shallowconvnet_preprocessing_ablation"]
    assert set(transformer["seed_provenance"]["runs"]) == set(transformer["seeds"])
    assert preprocessing["seed_provenance"]["external_seeds"] == [7, 123]
    assert transformer["seeds"] == preprocessing["seeds"] == [7, 42, 123]


def test_19_automl_completed_status_has_runtime(
    curation: dict,
    curation_by_path: dict[str, dict],
    project_audit,
    tmp_path: Path,
) -> None:
    item = curation_by_path["experiments/automl_transformer_label_q5.yaml"]
    assert item["canonical_status"] == "completed"
    assert (ROOT / item["linked_runtime"]).exists()
    without_runtime = deepcopy(curation)
    altered = next(
        value
        for value in without_runtime["configs"]
        if value["config_path"] == "experiments/automl_transformer_label_q5.yaml"
    )
    altered.pop("linked_runtime")
    source = tmp_path / "curation.yaml"
    source.write_text(
        yaml.safe_dump(without_runtime, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    records = {record.path: record for record in project_audit.records}
    with pytest.raises(
        audit.CurationValidationError,
        match="completed AutoML canonical requires linked_runtime",
    ):
        audit.load_and_validate_curation(ROOT, source, records)


def test_20_rf_canonical_links_completed_baseline(
    curation_by_path: dict[str, dict], registry_by_id: dict[str, dict]
) -> None:
    item = curation_by_path["configs/groupkfold_rf_label_q5.yaml"]
    assert item["canonical_status"] == "completed"
    assert item["linked_experiments"] == ["label_q5_random_forest_groupkfold"]
    assert registry_by_id[item["linked_experiments"][0]]["status"] == "baseline"


def test_21_personalization_links_final_report(
    curation_by_path: dict[str, dict]
) -> None:
    item = curation_by_path[
        "experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml"
    ]
    assert item["canonical_status"] == "completed"
    assert item["linked_report"] == (
        "reports/integration/personalization_multiseed_20pct.md"
    )


def test_22_shallowconvnet_links_sibling_seed_configs(
    registry_by_id: dict[str, dict]
) -> None:
    provenance = registry_by_id[
        "label_q5_shallowconvnet_raw_dedup_multiseed"
    ]["seed_provenance"]
    assert provenance["sibling_config_paths"] == [
        "configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7.yaml",
        "configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123.yaml",
    ]


def test_23_repeated_audit_render_is_deterministic(project_audit) -> None:
    first = (
        audit.render_inventory_csv(project_audit),
        audit.render_config_registry(project_audit),
        audit.render_markdown(project_audit),
        audit.render_curation_markdown(project_audit),
    )
    second = (
        audit.render_inventory_csv(project_audit),
        audit.render_config_registry(project_audit),
        audit.render_markdown(project_audit),
        audit.render_curation_markdown(project_audit),
    )
    assert first == second


def test_24_repeated_experiment_summary_is_deterministic(registry: dict) -> None:
    first = summary.build_summaries(registry, ROOT, strict=True)
    second = summary.build_summaries(registry, ROOT, strict=True)
    assert summary.render_experiment_markdown(first) == summary.render_experiment_markdown(second)
    assert summary.render_mixin_markdown(first) == summary.render_mixin_markdown(second)


def test_25_non_target_experiment_yamls_are_unchanged() -> None:
    paths = [
        "experiments/statistical_analysis.yaml",
        "experiments/automl_transformer_label_q5.yaml",
        "configs/groupkfold_rf_label_q5.yaml",
        "experiments/calibration/label_q5_finetuning_multiseed_20pct.yaml",
        "configs/groupkfold_torch_shallow_convnet_raw_dedup_label_q5.yaml",
        "configs/groupkfold_torch_lstm_label_q5.yaml",
        "configs/groupkfold_torch_bilstm_label_q5.yaml",
    ]
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        assert (ROOT / path).read_bytes() == committed


def test_26_strict_audit_has_no_errors(project_audit) -> None:
    assert project_audit.structural_errors == []
    assert not [record.path for record in project_audit.records if record.errors]


def test_27_strict_experiment_registry_validation(registry: dict) -> None:
    warnings = summary.validate_registry(registry, ROOT, strict=True)
    assert isinstance(warnings, list)
