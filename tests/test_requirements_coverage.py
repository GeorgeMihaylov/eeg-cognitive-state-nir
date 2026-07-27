from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "17_build_requirements_coverage.py"
SPEC = importlib.util.spec_from_file_location("requirements_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COVERAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COVERAGE
SPEC.loader.exec_module(COVERAGE)


def _prepare_root(root: Path) -> None:
    (root / "README.md").write_text("# Цель проекта\n", encoding="utf-8")


def _requirement(requirement_id: str = "R-TEST") -> dict:
    return {
        "requirement_id": requirement_id,
        "parent_id": None,
        "title": "Проверяемое требование",
        "domain": "platform",
        "requirement_kind": "subrequirement",
        "source": {
            "type": "project_plan",
            "path": "README.md",
            "section": "Цель",
            "quote": "Проверяемая цель",
        },
        "normalized_requirement": "Реализовать проверяемую возможность.",
        "requirement_type": "functional",
        "priority": "important",
        "coverage": {
            "implementation": "complete",
            "reproducible_config": "complete",
            "experimental_validation": "complete",
            "result_quality": "complete",
            "documentation": "complete",
            "integration": "complete",
            "demonstration": "not_applicable",
        },
        "overall_status": "complete",
        "evidence": [
            {
                "type": "code",
                "path": "README.md",
                "supports": "Проверяемая реализация.",
                "level": "primary",
            }
        ],
        "gaps": [],
        "minimum_closure_action": {
            "action_type": "documentation",
            "description": "Поддерживать описание.",
            "estimated_scope": "small",
        },
        "new_experiment_required": False,
        "closure_priority": "P3",
        "blocking_dependencies": [],
        "completion_criterion": "Evidence остаётся проверяемым.",
        "notes": [],
    }


def _registry(*requirements: dict) -> dict:
    return {
        "schema_version": 1,
        "project": {
            "name": "test",
            "requirements_source_status": "project_plan_only",
            "primary_source": {"path": "README.md", "type": "project_plan"},
            "source_warnings": [],
        },
        "requirements": list(requirements) or [_requirement()],
        "deferred_now": [],
    }


def _experiments() -> dict:
    return {
        "schema_version": 1,
        "experiments": [
            {"experiment_id": "baseline_exp", "status": "baseline"},
            {"experiment_id": "smoke_exp", "status": "smoke"},
            {"experiment_id": "invalid_exp", "status": "invalidated"},
        ],
    }


def _configs() -> dict:
    return {
        "schema_version": 1,
        "configs": [{"config_path": "experiments/example.yaml"}],
    }


def _validate(registry: dict, root: Path, *, strict: bool = False) -> list[str]:
    _prepare_root(root)
    return COVERAGE.validate_registry(
        registry,
        _experiments(),
        _configs(),
        root,
        strict=strict,
    )


def _project_registry() -> dict:
    return yaml.safe_load(
        (ROOT / "reports/summary/requirements_registry.yaml").read_text(
            encoding="utf-8"
        )
    )


def _project_experiments() -> dict:
    return yaml.safe_load(
        (ROOT / "reports/summary/experiment_registry.yaml").read_text(
            encoding="utf-8"
        )
    )


def _project_configs() -> dict:
    return yaml.safe_load(
        (ROOT / "reports/summary/config_registry.yaml").read_text(encoding="utf-8")
    )


def test_01_valid_registry_passes(tmp_path: Path) -> None:
    assert _validate(_registry(), tmp_path) == []


def test_02_duplicate_requirement_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="duplicate"):
        _validate(_registry(_requirement(), _requirement()), tmp_path)


def test_03_unknown_parent_is_rejected(tmp_path: Path) -> None:
    item = _requirement()
    item["parent_id"] = "R-MISSING"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="unknown parent_id"):
        _validate(_registry(item), tmp_path)


def test_04_hierarchy_cycle_is_rejected(tmp_path: Path) -> None:
    first = _requirement("R-A")
    second = _requirement("R-B")
    first["parent_id"] = "R-B"
    second["parent_id"] = "R-A"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="hierarchy cycle"):
        _validate(_registry(first, second), tmp_path)


def test_05_unknown_overall_status_is_rejected(tmp_path: Path) -> None:
    item = _requirement()
    item["overall_status"] = "done"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="unknown status"):
        _validate(_registry(item), tmp_path)


def test_06_unknown_requirement_type_is_rejected(tmp_path: Path) -> None:
    item = _requirement()
    item["requirement_type"] = "wish"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="unknown requirement type"):
        _validate(_registry(item), tmp_path)


def test_07_absolute_local_path_is_rejected(tmp_path: Path) -> None:
    item = _requirement()
    item["source"]["path"] = str((tmp_path / "README.md").resolve())
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="absolute local path"):
        _validate(_registry(item), tmp_path)


def test_08_complete_requires_evidence(tmp_path: Path) -> None:
    item = _requirement()
    item["evidence"] = []
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="complete requires evidence"):
        _validate(_registry(item), tmp_path)


def test_09_partial_with_gap_is_valid(tmp_path: Path) -> None:
    item = _requirement()
    item["overall_status"] = "partial"
    item["coverage"]["documentation"] = "partial"
    item["gaps"] = ["Не завершена документация."]
    assert _validate(_registry(item), tmp_path) == []


def test_10_not_applicable_requires_reason(tmp_path: Path) -> None:
    item = _requirement()
    item["overall_status"] = "not_applicable"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="requires reason"):
        _validate(_registry(item), tmp_path)


def test_11_failed_criterion_requires_expected_and_actual(tmp_path: Path) -> None:
    item = _requirement()
    item["overall_status"] = "failed_acceptance_criterion"
    item["acceptance_criterion"] = {"metric": "accuracy", "operator": ">="}
    with pytest.raises(
        COVERAGE.RequirementsRegistryError,
        match="requires expected and actual",
    ):
        _validate(_registry(item), tmp_path)


def test_12_unknown_experiment_evidence_is_rejected(tmp_path: Path) -> None:
    item = _requirement()
    item["evidence"] = [{
        "type": "experiment",
        "experiment_id": "missing",
        "supports": "Result.",
        "level": "primary",
    }]
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="unknown experiment ID"):
        _validate(_registry(item), tmp_path)


def test_13_smoke_cannot_be_primary_evidence(tmp_path: Path) -> None:
    item = _requirement()
    item["evidence"] = [{
        "type": "experiment",
        "experiment_id": "smoke_exp",
        "supports": "Smoke.",
        "level": "primary",
    }]
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="smoke experiment"):
        _validate(_registry(item), tmp_path)


def test_14_invalidated_cannot_be_primary_evidence(tmp_path: Path) -> None:
    item = _requirement()
    item["evidence"] = [{
        "type": "experiment",
        "experiment_id": "invalid_exp",
        "supports": "Invalid.",
        "level": "primary",
    }]
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="invalidated experiment"):
        _validate(_registry(item), tmp_path)


def test_15_strict_mode_requires_tracked_reports(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _prepare_root(tmp_path)
    report.write_text("# Report\n", encoding="utf-8")
    item = _requirement()
    item["evidence"] = [{
        "type": "report",
        "path": "report.md",
        "supports": "Report.",
        "level": "primary",
    }]
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="tracked report"):
        COVERAGE.validate_registry(
            _registry(item), _experiments(), _configs(), tmp_path, strict=True
        )


def test_16_optional_runtime_artifact_emits_warning(tmp_path: Path) -> None:
    item = _requirement()
    item["evidence"] = [{
        "type": "artifact",
        "path": "benchmark_results/missing.json",
        "supports": "Optional runtime evidence.",
        "level": "supporting",
        "optional_runtime": True,
    }]
    warnings = _validate(_registry(item), tmp_path)
    assert len(warnings) == 1
    assert "optional runtime artifact missing" in warnings[0]


def test_17_full_experiment_action_sets_experiment_flag(tmp_path: Path) -> None:
    item = _requirement()
    item["minimum_closure_action"]["action_type"] = "full_experiment"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="requires new_experiment"):
        _validate(_registry(item), tmp_path)


def test_18_documentation_action_cannot_require_experiment(tmp_path: Path) -> None:
    item = _requirement()
    item["new_experiment_required"] = True
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="must not require"):
        _validate(_registry(item), tmp_path)


def test_19_streaming_not_closed_by_batch_inference(tmp_path: Path) -> None:
    item = _requirement()
    item["domain"] = "streaming"
    item["evidence"][0]["capability"] = "batch_inference"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="streaming cannot be complete"):
        _validate(_registry(item), tmp_path)


def test_20_demo_not_closed_by_unit_test(tmp_path: Path) -> None:
    item = _requirement()
    item["domain"] = "demo"
    item["evidence"][0]["capability"] = "unit_test"
    with pytest.raises(COVERAGE.RequirementsRegistryError, match="demo cannot be complete"):
        _validate(_registry(item), tmp_path)


def test_21_domain_adaptation_not_closed_by_subset_analysis(tmp_path: Path) -> None:
    item = _requirement()
    item["domain"] = "domain_adaptation"
    item["evidence"][0]["capability"] = "subset_analysis"
    with pytest.raises(
        COVERAGE.RequirementsRegistryError,
        match="domain_adaptation cannot be complete",
    ):
        _validate(_registry(item), tmp_path)


def test_22_multioutput_regression_is_not_multimodality(tmp_path: Path) -> None:
    item = _requirement()
    item["domain"] = "multimodality"
    item["evidence"][0]["capability"] = "multi_output_regression"
    with pytest.raises(
        COVERAGE.RequirementsRegistryError,
        match="multimodality cannot be complete",
    ):
        _validate(_registry(item), tmp_path)


def test_23_personalization_implementation_is_separate_from_quality() -> None:
    items = {item["requirement_id"]: item for item in _project_registry()["requirements"]}
    assert items["R-PERS-01"]["overall_status"] == "complete"
    assert items["R-PERS-Q01"]["requirement_kind"] == "acceptance_criterion"
    assert items["R-PERS-Q01"]["overall_status"] == "failed_acceptance_criterion"


def test_24_accuracy_threshold_is_recorded_as_failed() -> None:
    items = {item["requirement_id"]: item for item in _project_registry()["requirements"]}
    criterion = items["R-PERS-Q01"]["acceptance_criterion"]
    assert criterion["operator"] == ">="
    assert criterion["expected"] == pytest.approx(0.75)
    assert criterion["actual"] == pytest.approx(0.6349206349)
    assert COVERAGE._criterion_is_failed(criterion)


def test_25_traceability_csv_is_deterministic() -> None:
    registry = _project_registry()
    assert COVERAGE.render_traceability_csv(registry) == COVERAGE.render_traceability_csv(
        copy.deepcopy(registry)
    )


def test_26_coverage_markdown_is_deterministic() -> None:
    registry = _project_registry()
    first = COVERAGE.render_coverage_markdown(registry)
    second = COVERAGE.render_coverage_markdown(copy.deepcopy(registry))
    assert first == second


def test_27_remaining_work_is_sorted_by_priority() -> None:
    rendered = COVERAGE.render_remaining_work(_project_registry())
    assert rendered.index("## P0") < rendered.index("## P1")
    assert rendered.index("## P1") < rendered.index("## P2")
    assert rendered.index("## P2") < rendered.index("## P3")


def test_28_russian_text_survives_utf8_rendering() -> None:
    rendered = COVERAGE.render_coverage_markdown(_project_registry())
    assert "Карта соответствия требованиям проекта" in rendered
    assert rendered.encode("utf-8").decode("utf-8") == rendered


def test_29_experiment_evidence_links_current_registry() -> None:
    known = {
        item["experiment_id"] for item in _project_experiments()["experiments"]
    }
    linked = {
        evidence["experiment_id"]
        for item in _project_registry()["requirements"]
        for evidence in item["evidence"]
        if evidence["type"] == "experiment"
    }
    assert linked
    assert linked <= known


def test_30_config_evidence_links_current_registry() -> None:
    known = COVERAGE._config_paths(_project_configs())
    linked = {
        evidence["path"].replace("\\", "/")
        for item in _project_registry()["requirements"]
        for evidence in item["evidence"]
        if evidence["type"] == "config"
    }
    assert linked
    assert linked <= known


def test_31_project_registry_passes_strict_validation() -> None:
    assert COVERAGE.validate_registry(
        _project_registry(),
        _project_experiments(),
        _project_configs(),
        ROOT,
        strict=True,
    ) == []
