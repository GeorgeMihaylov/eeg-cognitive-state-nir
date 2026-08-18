from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "16_audit_experiment_configs.py"
SPEC = importlib.util.spec_from_file_location("config_audit_script", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write_yaml(root: Path, relative: str, document: object) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def raw_fragment(**extra: object) -> dict:
    return {"raw_preprocessing": {"bandpass": {"enabled": False}}, **extra}


def make_record(
    document: dict,
    *,
    path: str = "configs/example.yaml",
    role: str = "full",
    status: str = "unclassified",
    loader_type: str = "benchmark_config",
) -> object:
    record = audit.ConfigRecord(
        path=path,
        document=document,
        role=role,
        status=status,
        loader_type=loader_type,
        schema_valid=True,
    )
    record.extracted = audit.extract_fields(document)
    return record


@pytest.fixture(scope="session")
def project_audit() -> object:
    return audit.audit_repository(ROOT)


def test_01_discovers_yaml_and_yml(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    write_yaml(tmp_path, "experiments/b.yml", raw_fragment())
    assert audit.discover_config_paths(tmp_path) == [
        "configs/a.yaml",
        "experiments/b.yml",
    ]


def test_02_excludes_benchmark_results(tmp_path: Path) -> None:
    write_yaml(tmp_path, "benchmark_results/runtime.yaml", raw_fragment())
    write_yaml(tmp_path, "configs/source.yaml", raw_fragment())
    assert audit.discover_config_paths(tmp_path) == ["configs/source.yaml"]


def test_03_excludes_data(tmp_path: Path) -> None:
    write_yaml(tmp_path, "data/runtime.yaml", raw_fragment())
    write_yaml(tmp_path, "experiments/source.yaml", raw_fragment())
    assert audit.discover_config_paths(tmp_path) == ["experiments/source.yaml"]


def test_04_preserves_relative_paths(tmp_path: Path) -> None:
    write_yaml(tmp_path, "experiments/тема/source.yaml", raw_fragment())
    paths = audit.discover_config_paths(tmp_path)
    assert paths == ["experiments/тема/source.yaml"]
    assert not Path(paths[0]).is_absolute()


def test_05_detects_local_absolute_paths() -> None:
    found = audit.find_absolute_paths(
        {"windows": "F:\\EEG\\data.parquet", "linux": "/home/user/data"}
    )
    assert [item["yaml_key"] for item in found] == ["linux", "windows"]


def test_06_url_is_not_local_absolute_path() -> None:
    assert not audit.is_local_absolute_path("https://example.org/F:/artifact")


def test_07_detects_exact_duplicates() -> None:
    records = [
        audit.ConfigRecord(path="a.yaml", exact_hash="same"),
        audit.ConfigRecord(path="b.yaml", exact_hash="same"),
    ]
    groups = audit._duplicate_groups(records)
    assert any(group["kind"] == "exact_duplicate" for group in groups)


def test_08_detects_resolved_duplicates() -> None:
    records = [
        audit.ConfigRecord(path="a.yaml", resolved_hash="same"),
        audit.ConfigRecord(path="b.yaml", resolved_hash="same"),
    ]
    groups = audit._duplicate_groups(records)
    assert any(group["kind"] == "resolved_duplicate" for group in groups)


def test_09_detects_scientific_protocol_duplicates() -> None:
    records = [
        audit.ConfigRecord(path="a.yaml", protocol_hash="same"),
        audit.ConfigRecord(path="b.yaml", protocol_hash="same"),
    ]
    groups = audit._duplicate_groups(records)
    assert any(
        group["kind"] == "same_protocol_different_output" for group in groups
    )


def test_10_output_directory_does_not_change_protocol_hash() -> None:
    left = {"model": {"type": "rf"}, "output_dir": "one"}
    right = {"model": {"type": "rf"}, "output_dir": "two"}
    assert audit.stable_hash(audit._without_keys(left)) == audit.stable_hash(
        audit._without_keys(right)
    )


def test_11_seed_changes_protocol_hash() -> None:
    left = {"model": {"params": {"random_state": 7}}}
    right = {"model": {"params": {"random_state": 42}}}
    assert audit.stable_hash(audit._without_keys(left)) != audit.stable_hash(
        audit._without_keys(right)
    )


def test_12_base_config_resolves(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/base.yaml", {"raw_preprocessing": {"a": 1}})
    write_yaml(
        tmp_path,
        "configs/child.yaml",
        {"base_config": "base.yaml", "raw_preprocessing": {"b": 2}},
    )
    result = audit.audit_repository(tmp_path)
    child = next(item for item in result.records if item.path.endswith("child.yaml"))
    assert child.base_exists
    assert child.inheritance_depth == 1
    assert child.resolved_hash


def test_13_missing_base_is_reported(tmp_path: Path) -> None:
    write_yaml(
        tmp_path,
        "configs/child.yaml",
        {"base_config": "missing.yaml", "raw_preprocessing": {}},
    )
    result = audit.audit_repository(tmp_path)
    assert result.missing_bases == [
        {"config_path": "configs/child.yaml", "base_config": "configs/missing.yaml"}
    ]
    assert result.records[0].errors


def test_14_inheritance_cycle_is_detected(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/a.yaml", {"base_config": "b.yaml"})
    write_yaml(tmp_path, "configs/b.yaml", {"base_config": "a.yaml"})
    result = audit.audit_repository(tmp_path)
    assert result.inheritance_cycles
    assert any(issue.code == "inheritance_cycle" for issue in result.records[0].errors)


def test_15_inheritance_depth_is_calculated(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    write_yaml(tmp_path, "configs/b.yaml", {"base_config": "a.yaml"})
    write_yaml(tmp_path, "configs/c.yaml", {"base_config": "b.yaml"})
    result = audit.audit_repository(tmp_path)
    depth = {record.path: record.inheritance_depth for record in result.records}
    assert depth["configs/c.yaml"] == 2


def test_16_repeated_generation_is_deterministic(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/база.yaml", raw_fragment())
    result1 = audit.audit_repository(tmp_path)
    first = {
        "csv": audit.render_inventory_csv(result1),
        "yaml": audit.render_config_registry(result1),
        "md": audit.render_markdown(result1),
    }
    result2 = audit.audit_repository(tmp_path)
    second = {
        "csv": audit.render_inventory_csv(result2),
        "yaml": audit.render_config_registry(result2),
        "md": audit.render_markdown(result2),
    }
    assert first == second


def test_17_unknown_role_stays_unknown() -> None:
    assert audit.infer_role(
        "experiments/example.yaml", {"mystery": 1}, set(), "unknown"
    ) == "unknown"


def test_18_base_config_need_not_be_cli_loadable(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/raw.yaml", raw_fragment())
    record = audit.audit_repository(tmp_path).records[0]
    assert record.role == "base"
    assert not record.cli_loadable
    assert record.schema_valid


def test_19_full_config_uses_current_loader(project_audit: object) -> None:
    record = next(
        item
        for item in project_audit.records
        if item.path == "configs/groupkfold_rf_label_q5.yaml"
    )
    assert record.loader_type == "benchmark_config"
    assert record.cli_loadable
    assert record.schema_valid


def test_20_broken_yaml_does_not_stop_audit(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/good.yaml", raw_fragment())
    bad = tmp_path / "configs" / "bad.yaml"
    bad.write_text("broken: [", encoding="utf-8")
    result = audit.audit_repository(tmp_path)
    assert len(result.records) == 2
    assert any(record.parse_error for record in result.records)


def test_21_experiment_registry_link_is_recorded(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    write_yaml(
        tmp_path,
        "reports/summary/experiment_registry.yaml",
        {
            "experiments": [
                {
                    "experiment_id": "x",
                    "config_path": "configs/a.yaml",
                    "status": "diagnostic",
                }
            ]
        },
    )
    record = audit.audit_repository(tmp_path).records[0]
    assert record.registry_ids == ["x"]


def test_22_missing_registry_config_is_recorded(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    write_yaml(
        tmp_path,
        "reports/summary/experiment_registry.yaml",
        {
            "experiments": [
                {
                    "experiment_id": "missing",
                    "config_path": "configs/missing.yaml",
                }
            ]
        },
    )
    result = audit.audit_repository(tmp_path)
    assert result.registry_missing_configs == [
        {"experiment_id": "missing", "config_path": "configs/missing.yaml"}
    ]


def test_23_registry_model_mismatch_is_recorded() -> None:
    record = make_record(
        {"models": {"rf": {"type": "random_forest"}}, "datasets": {}, "tasks": []}
    )
    record.extracted = audit.extract_fields(record.document)
    mismatches, _ = audit.check_registry_consistency(
        ROOT,
        {record.path: record},
        [
            {
                "experiment_id": "x",
                "config_path": record.path,
                "model": "torch_mlp",
            }
        ],
    )
    assert mismatches[0]["field"] == "model"


def test_24_registry_target_mismatch_is_recorded() -> None:
    record = make_record(
        {
            "datasets": {"d": {"target_col": "label_q5"}},
            "models": {"rf": {"type": "random_forest"}},
            "tasks": ["cognitive_load_5class"],
        }
    )
    record.extracted = audit.extract_fields(record.document)
    mismatches, _ = audit.check_registry_consistency(
        ROOT,
        {record.path: record},
        [
            {
                "experiment_id": "x",
                "config_path": record.path,
                "target": "target_focus",
            }
        ],
    )
    assert mismatches[0]["field"] == "target"


def test_25_pm_target_order_is_checked() -> None:
    document = {
        "datasets": {"pm": {"target_cols": list(reversed(audit.PM_TARGETS))}},
        "models": {},
        "tasks": ["performance_metrics_regression"],
    }
    record = make_record(document)
    audit.validate_scientific_protocol(record)
    assert any(issue.code == "pm_target_order" for issue in record.errors)


def test_26_label_q5_discretize_true_is_error() -> None:
    document = {
        "datasets": {"d": {"target_col": "label_q5", "discretize": True}},
        "models": {},
        "tasks": ["cognitive_load_5class"],
    }
    record = make_record(document)
    audit.validate_scientific_protocol(record)
    assert any(issue.code == "label_q5_rediscretized" for issue in record.errors)


def test_27_random_split_for_final_is_error() -> None:
    document = {
        "datasets": {},
        "models": {},
        "tasks": [],
        "evaluation": {"protocol": "random_window"},
    }
    record = make_record(document, status="final")
    audit.validate_scientific_protocol(record)
    assert any(issue.code == "final_random_window_split" for issue in record.errors)


def test_28_random_split_for_smoke_is_not_critical() -> None:
    document = {
        "datasets": {},
        "models": {},
        "tasks": [],
        "evaluation": {"protocol": "random_window"},
    }
    record = make_record(document, role="smoke", status="smoke")
    audit.validate_scientific_protocol(record)
    assert not any(issue.code == "final_random_window_split" for issue in record.errors)


def test_29_ignored_yaml_key_is_recorded() -> None:
    record = make_record(
        {
            "datasets": {},
            "models": {},
            "tasks": [],
            "unused_future_field": 1,
        }
    )
    audit._validate_known_keys(record)
    assert any(
        issue.code == "ignored_or_unknown_field" for issue in record.warnings
    )


def test_30_conflicting_key_types_are_recorded() -> None:
    left = make_record(
        {"datasets": {}, "models": {}, "tasks": [], "validation": {}},
        path="left.yaml",
    )
    right = make_record(
        {"datasets": {}, "models": {}, "tasks": [], "validation": []},
        path="right.yaml",
    )
    audit._add_type_conflict_warnings([left, right])
    assert any(issue.code == "conflicting_key_types" for issue in left.warnings)


def test_31_inventory_csv_has_stable_order(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/z.yaml", raw_fragment())
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    result = audit.audit_repository(tmp_path)
    rows = list(csv.DictReader(io.StringIO(audit.render_inventory_csv(result))))
    assert [row["config_path"] for row in rows] == [
        record.path for record in result.records
    ]


def test_32_markdown_has_stable_order(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/z.yaml", raw_fragment())
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    result = audit.audit_repository(tmp_path)
    text = audit.render_markdown(result)
    assert text == audit.render_markdown(result)
    assert text.index("configs/a.yaml") < text.index("configs/z.yaml")


def test_33_yaml_registry_has_stable_order(tmp_path: Path) -> None:
    write_yaml(tmp_path, "configs/z.yaml", raw_fragment())
    write_yaml(tmp_path, "configs/a.yaml", raw_fragment())
    result = audit.audit_repository(tmp_path)
    document = yaml.safe_load(audit.render_config_registry(result))
    assert [item["config_path"] for item in document["configs"]] == [
        record.path for record in result.records
    ]


def test_34_utf8_russian_strings_are_preserved(tmp_path: Path) -> None:
    write_yaml(
        tmp_path,
        "configs/русский.yaml",
        {"raw_preprocessing": {"notes": "Проверка воспроизводимости"}},
    )
    output = audit.render_config_registry(audit.audit_repository(tmp_path))
    assert "русский" in output
    assert "Проверка воспроизводимости" not in output  # reports metadata, not source copy
    output.encode("utf-8").decode("utf-8")


def test_35_current_project_configs_are_all_processed(project_audit: object) -> None:
    paths = audit.discover_config_paths(ROOT)
    assert len(project_audit.records) == len(paths)
    assert len(paths) == 79
    assert not [record for record in project_audit.records if record.parse_error]
