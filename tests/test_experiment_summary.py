from __future__ import annotations

import csv
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "15_build_experiment_summary.py"
SPEC = importlib.util.spec_from_file_location("experiment_summary_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def metric(
    value: float = 0.5,
    *,
    name: str = "macro_f1",
    direction: str = "higher_is_better",
) -> dict:
    return {
        "name": name,
        "direction": direction,
        "value_source": {"type": "constant", "value": value},
    }


def entry(
    experiment_id: str = "example",
    *,
    category: str = "model",
    status: str = "baseline",
    report_path: str = "report.md",
) -> dict:
    return {
        "experiment_id": experiment_id,
        "title": f"Эксперимент {experiment_id}",
        "category": category,
        "status": status,
        "task": "cognitive_load_5class",
        "target": "label_q5",
        "model": "random_forest",
        "feature_set": "pow_plus_eeg",
        "preprocessing": "none",
        "evaluation_protocol": "5-fold GroupKFold by subject_id",
        "seeds": [42],
        "n_subjects": 5,
        "primary_metric": metric(),
        "secondary_metrics": [],
        "result_summary": "Проверенный результат.",
        "report_path": report_path,
        "runtime_path": None,
        "config_path": None,
        "commit": None,
        "limitations": [],
        "tags": [],
    }


def registry(*entries: dict, unresolved: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "experiments": list(entries),
        "unresolved_entries": unresolved or [],
    }


def prepare_root(tmp_path: Path, *report_paths: str) -> None:
    for relative in report_paths or ("report.md",):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Evidence\n", encoding="utf-8")


def write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_valid_minimal_registry_loads(tmp_path):
    path = write_registry(tmp_path, registry(entry()))
    loaded = SUMMARY.load_registry(path)
    assert loaded["schema_version"] == 1
    assert loaded["experiments"][0]["experiment_id"] == "example"


def test_duplicate_experiment_id_is_rejected(tmp_path):
    prepare_root(tmp_path)
    payload = registry(entry(), entry())
    with pytest.raises(SUMMARY.RegistryError, match="Duplicate experiment_id"):
        SUMMARY.validate_registry(payload, tmp_path)


def test_unknown_status_is_rejected(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["status"] = "published"
    with pytest.raises(SUMMARY.RegistryError, match="unknown status"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_unknown_category_is_rejected(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["category"] = "neural"
    with pytest.raises(SUMMARY.RegistryError, match="unknown category"):
        SUMMARY.validate_registry(registry(item), tmp_path)


@pytest.mark.parametrize(
    "absolute_path",
    [r"F:\EEG\report.md", r"C:\Users\George\report.md", "/tmp/report.md"],
)
def test_absolute_path_is_rejected(tmp_path, absolute_path):
    item = entry(report_path=absolute_path)
    with pytest.raises(SUMMARY.RegistryError, match="must be relative"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_missing_tracked_report_is_strict_error(tmp_path):
    item = entry(report_path="missing.md")
    with pytest.raises(SUMMARY.RegistryError, match="tracked report missing"):
        SUMMARY.validate_registry(registry(item), tmp_path, strict=True)


def test_missing_runtime_path_is_warning(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["runtime_path"] = "benchmark_results/missing"
    warnings = SUMMARY.validate_registry(registry(item), tmp_path, strict=True)
    assert (
        "example: runtime path missing: benchmark_results/missing"
        in warnings
    )


def test_constant_source_is_extracted_with_explicit_provenance(tmp_path):
    value, provenance = SUMMARY.extract_value(
        {"type": "constant", "value": 0.2889},
        tmp_path,
    )
    assert value == pytest.approx(0.2889)
    assert provenance["type"] == "registry_constant"
    assert provenance["path"] == ""


def test_json_source_is_extracted(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.31}),
        encoding="utf-8",
    )
    value, provenance = SUMMARY.extract_value(
        {"type": "json", "path": "metrics.json", "key": "macro_f1"},
        tmp_path,
    )
    assert value == pytest.approx(0.31)
    assert provenance["type"] == "json"


def test_nested_json_key_is_supported(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"metrics": {"fold": {"macro_f1": 0.42}}}),
        encoding="utf-8",
    )
    value, _ = SUMMARY.extract_value(
        {
            "type": "json",
            "path": "metrics.json",
            "key": "metrics.fold.macro_f1",
        },
        tmp_path,
    )
    assert value == pytest.approx(0.42)


def test_yaml_source_supports_nested_key(tmp_path):
    (tmp_path / "manifest.yaml").write_text(
        "metrics:\n  macro_mae: 0.101\n",
        encoding="utf-8",
    )
    value, _ = SUMMARY.extract_value(
        {
            "type": "yaml",
            "path": "manifest.yaml",
            "key": "metrics.macro_mae",
        },
        tmp_path,
    )
    assert value == pytest.approx(0.101)


def test_csv_filter_selects_rows(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "method,budget,value\nzero_shot,0.2,0.20\nfull_model,0.2,0.25\n",
        encoding="utf-8",
    )
    value, _ = SUMMARY.extract_value(
        {
            "type": "csv_filter",
            "path": "metrics.csv",
            "filters": {"method": "full_model", "budget": 0.2},
            "column": "value",
        },
        tmp_path,
    )
    assert value == pytest.approx(0.25)


def test_ambiguous_csv_filter_without_aggregation_is_error(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "method,value\nfull_model,0.2\nfull_model,0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(SUMMARY.RegistryError, match="ambiguous"):
        SUMMARY.extract_value(
            {
                "type": "csv_filter",
                "path": "metrics.csv",
                "filters": {"method": "full_model"},
                "column": "value",
            },
            tmp_path,
        )


def test_csv_filter_mean_is_calculated(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "method,value\nfull_model,0.2\nfull_model,0.4\n",
        encoding="utf-8",
    )
    value, _ = SUMMARY.extract_value(
        {
            "type": "csv_filter",
            "path": "metrics.csv",
            "filters": {"method": "full_model"},
            "column": "value",
            "aggregation": "mean",
        },
        tmp_path,
    )
    assert value == pytest.approx(0.3)


def test_missing_primary_metric_is_error(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["primary_metric"] = {"name": "macro_f1"}
    with pytest.raises(SUMMARY.RegistryError, match="value_source is required"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_secondary_metrics_serialize_deterministically(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["secondary_metrics"] = [
        metric(0.2, name="accuracy"),
        metric(0.3, name="balanced_accuracy"),
    ]
    first = SUMMARY.build_summaries(registry(item), tmp_path)
    second = SUMMARY.build_summaries(registry(item), tmp_path)
    assert (
        first["experiment_rows"][0]["secondary_metrics_json"]
        == second["experiment_rows"][0]["secondary_metrics_json"]
    )
    decoded = json.loads(
        first["experiment_rows"][0]["secondary_metrics_json"]
    )
    assert [row["name"] for row in decoded] == [
        "accuracy",
        "balanced_accuracy",
    ]


def test_generated_csv_has_stable_required_order(tmp_path):
    prepare_root(tmp_path)
    late = entry("late", status="diagnostic")
    early = entry("early", status="final")
    registry_path = write_registry(tmp_path, registry(late, early))
    output = tmp_path / "out"
    SUMMARY.generate(registry_path, output, tmp_path)
    rows = list(
        csv.DictReader(
            (output / "experiment_summary.csv").open(encoding="utf-8")
        )
    )
    assert [row["experiment_id"] for row in rows] == ["early", "late"]
    assert list(rows[0]) == SUMMARY.SUMMARY_COLUMNS


def test_generated_markdown_has_stable_order(tmp_path):
    prepare_root(tmp_path)
    zulu = entry("zulu", status="baseline")
    alpha = entry("alpha", status="baseline")
    payload = SUMMARY.build_summaries(registry(zulu, alpha), tmp_path)
    markdown = SUMMARY.render_experiment_markdown(payload)
    assert markdown.index("Эксперимент alpha") < markdown.index("Эксперимент zulu")


def test_repeated_generation_is_byte_identical(tmp_path):
    prepare_root(tmp_path)
    registry_path = write_registry(tmp_path, registry(entry()))
    output = tmp_path / "out"
    SUMMARY.generate(registry_path, output, tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    SUMMARY.generate(registry_path, output, tmp_path)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first == second


def test_invalidated_entry_requires_reason(tmp_path):
    prepare_root(tmp_path)
    item = entry(status="invalidated")
    item["superseded_by"] = None
    with pytest.raises(SUMMARY.RegistryError, match="invalidation_reason"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_superseded_by_requires_known_experiment(tmp_path):
    prepare_root(tmp_path)
    item = entry(status="invalidated")
    item["invalidation_reason"] = "Broken protocol."
    item["superseded_by"] = "missing"
    with pytest.raises(SUMMARY.RegistryError, match="unknown experiment"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_superseded_by_cycle_is_rejected(tmp_path):
    prepare_root(tmp_path, "a.md", "b.md")
    first = entry("a", status="invalidated", report_path="a.md")
    first.update(invalidation_reason="old", superseded_by="b")
    second = entry("b", status="invalidated", report_path="b.md")
    second.update(invalidation_reason="old", superseded_by="a")
    with pytest.raises(SUMMARY.RegistryError, match="cycle"):
        SUMMARY.validate_registry(registry(first, second), tmp_path)


def test_mixin_entry_requires_decision(tmp_path):
    prepare_root(tmp_path)
    item = entry(category="mixin", status="diagnostic")
    item.update(
        mixin_name="Transfer learning",
        audit_status="tested",
        integration_status="not_integrated",
        decision_reason="Reason",
    )
    with pytest.raises(SUMMARY.RegistryError, match="requires decision"):
        SUMMARY.validate_registry(registry(item), tmp_path)


def test_model_summary_excludes_smoke(tmp_path):
    prepare_root(tmp_path)
    baseline = entry("baseline")
    smoke = entry("smoke", status="smoke")
    summary = SUMMARY.build_summaries(registry(smoke, baseline), tmp_path)
    assert [row["experiment_id"] for row in summary["model_rows"]] == [
        "baseline"
    ]


def test_model_summary_excludes_invalidated(tmp_path):
    prepare_root(tmp_path)
    invalid = entry("invalid", status="invalidated")
    invalid.update(invalidation_reason="broken", superseded_by=None)
    summary = SUMMARY.build_summaries(registry(invalid), tmp_path)
    assert summary["model_rows"] == []


def test_personalization_summary_contains_only_personalization(tmp_path):
    prepare_root(tmp_path)
    model = entry("model")
    personal = entry(
        "personal",
        category="personalization",
        status="final",
    )
    personal["personalization"] = {
        "method": "full_model",
        "calibration_budget": 0.2,
        "baseline_metric": 0.20,
        "personalized_metric": 0.25,
        "gain_metric": "macro_f1_gain",
        "gain_value": 0.05,
        "bootstrap_ci_low": 0.01,
        "bootstrap_ci_high": 0.08,
        "improved_subject_fraction": 0.7,
    }
    summary = SUMMARY.build_summaries(registry(model, personal), tmp_path)
    assert [row["experiment_id"] for row in summary["personalization_rows"]] == [
        "personal"
    ]


def test_runtime_absolute_paths_never_reach_outputs(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["runtime_path"] = r"F:\EEG\benchmark_results\run"
    with pytest.raises(SUMMARY.RegistryError, match="must be relative"):
        SUMMARY.build_summaries(registry(item), tmp_path)


def test_utf8_russian_strings_are_preserved(tmp_path):
    prepare_root(tmp_path)
    item = entry()
    item["title"] = "Персонализация когнитивного состояния"
    registry_path = write_registry(tmp_path, registry(item))
    output = tmp_path / "out"
    SUMMARY.generate(registry_path, output, tmp_path)
    markdown = (output / "experiment_summary.md").read_text(encoding="utf-8")
    csv_text = (output / "experiment_summary.csv").read_text(encoding="utf-8")
    assert item["title"] in markdown
    assert item["title"] in csv_text


def test_project_registry_passes_strict_validation():
    registry_path = ROOT / "reports" / "summary" / "experiment_registry.yaml"
    payload = SUMMARY.load_registry(registry_path)
    warnings = SUMMARY.validate_registry(payload, ROOT, strict=True)
    assert all("runtime path missing" in warning for warning in warnings)
    summary = SUMMARY.build_summaries(payload, ROOT, strict=True)
    assert len(summary["experiment_rows"]) == 33
    assert len(summary["unresolved"]) == 4
