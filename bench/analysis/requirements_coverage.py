#!/usr/bin/env python3
"""Validate and render the curated project requirements coverage map."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import yaml


COVERAGE_FIELDS = (
    "implementation",
    "reproducible_config",
    "experimental_validation",
    "result_quality",
    "documentation",
    "integration",
    "demonstration",
)
COVERAGE_STATUSES = {
    "complete",
    "partial",
    "missing",
    "not_applicable",
    "unclear",
}
OVERALL_STATUSES = {
    "complete",
    "partial",
    "not_started",
    "not_applicable",
    "needs_clarification",
    "failed_acceptance_criterion",
}
REQUIREMENT_TYPES = {
    "functional",
    "scientific",
    "data",
    "quality",
    "reproducibility",
    "performance",
    "integration",
    "documentation",
    "deliverable",
    "organizational",
}
REQUIREMENT_KINDS = {
    "parent_requirement",
    "subrequirement",
    "acceptance_criterion",
    "deliverable",
}
PRIORITIES = {"mandatory", "important", "optional", "requires_clarification"}
CLOSURE_PRIORITIES = {"P0", "P1", "P2", "P3"}
ACTION_TYPES = {
    "documentation",
    "small_code_change",
    "integration",
    "limited_experiment",
    "full_experiment",
    "data_integration",
    "streaming",
    "demo",
    "research_decision",
    "scope_clarification",
}
ESTIMATED_SCOPES = {"small", "medium", "large"}
EVIDENCE_TYPES = {
    "code",
    "config",
    "test",
    "report",
    "experiment",
    "artifact",
    "commit",
    "documentation",
    "demo",
}
EVIDENCE_LEVELS = {"primary", "supporting", "historical", "diagnostic"}
SCIENTIFIC_TYPES = {"scientific", "quality", "performance"}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
CSV_COLUMNS = [
    "requirement_id",
    "parent_id",
    "title",
    "requirement_type",
    "priority",
    "source_type",
    "source_path",
    "source_section",
    "source_quote",
    "normalized_requirement",
    "implementation_status",
    "config_status",
    "experimental_status",
    "result_quality_status",
    "documentation_status",
    "integration_status",
    "demonstration_status",
    "overall_status",
    "evidence_count",
    "primary_evidence",
    "gaps",
    "minimum_closure_action",
    "action_type",
    "estimated_scope",
    "new_experiment_required",
    "closure_priority",
    "blocking_dependencies",
    "notes",
]


class RequirementsRegistryError(ValueError):
    """Raised when requirements provenance or coverage is inconsistent."""


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequirementsRegistryError(f"{label} does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise RequirementsRegistryError(f"invalid {label} YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise RequirementsRegistryError(f"{label} root must be a mapping")
    return document


def _is_absolute(value: str) -> bool:
    text = str(value).strip()
    return (
        Path(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or bool(re.match(r"^[A-Za-z]:[\\/]", text))
        or text.startswith(("/", "\\\\"))
    )


def _find_absolute_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_absolute_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_absolute_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _is_absolute(value):
        found.append(prefix)
    return found


def _tracked(root: Path, relative_path: str) -> bool:
    try:
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_path,
            ],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _commit_exists(root: Path, commit: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _experiment_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    experiments = document.get("experiments")
    if not isinstance(experiments, list):
        raise RequirementsRegistryError("experiment registry experiments must be a list")
    return {
        str(item["experiment_id"]): item
        for item in experiments
        if isinstance(item, Mapping) and item.get("experiment_id")
    }


def _config_paths(document: Mapping[str, Any]) -> set[str]:
    configs = document.get("configs")
    if not isinstance(configs, list):
        raise RequirementsRegistryError("config registry configs must be a list")
    return {
        PurePosixPath(str(item["config_path"])).as_posix()
        for item in configs
        if isinstance(item, Mapping) and item.get("config_path")
    }


def _hierarchy_cycle(parents: Mapping[str, str]) -> list[str] | None:
    for origin in sorted(parents):
        chain: list[str] = []
        current = origin
        while current in parents:
            if current in chain:
                return chain[chain.index(current) :] + [current]
            chain.append(current)
            current = parents[current]
    return None


def _criterion_is_failed(criterion: Mapping[str, Any]) -> bool:
    expected = criterion.get("expected")
    actual = criterion.get("actual")
    operator = criterion.get("operator")
    if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
        return False
    if operator == ">=":
        return actual < expected
    if operator == "<=":
        return actual > expected
    if operator == ">":
        return actual <= expected
    if operator == "<":
        return actual >= expected
    if operator == "==":
        return actual != expected
    return False


def validate_registry(
    registry: Mapping[str, Any],
    experiment_registry: Mapping[str, Any],
    config_registry: Mapping[str, Any],
    root: Path,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate source traceability, coverage claims, and evidence links."""
    if registry.get("schema_version") != 1:
        raise RequirementsRegistryError("requirements schema_version must equal 1")
    project = registry.get("project")
    requirements = registry.get("requirements")
    if not isinstance(project, Mapping):
        raise RequirementsRegistryError("project must be a mapping")
    if not isinstance(requirements, list):
        raise RequirementsRegistryError("requirements must be a list")
    absolute = _find_absolute_paths(registry)
    if absolute:
        raise RequirementsRegistryError(
            f"absolute local path at {sorted(absolute)[0]}"
        )

    experiments = _experiment_index(experiment_registry)
    configs = _config_paths(config_registry)
    errors: list[str] = []
    warnings: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    parents: dict[str, str] = {}

    required_fields = {
        "requirement_id",
        "title",
        "source",
        "normalized_requirement",
        "requirement_kind",
        "requirement_type",
        "priority",
        "coverage",
        "overall_status",
        "evidence",
        "gaps",
        "minimum_closure_action",
        "new_experiment_required",
        "closure_priority",
        "blocking_dependencies",
        "notes",
    }
    for index, item in enumerate(requirements):
        location = f"requirements[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{location} must be a mapping")
            continue
        missing = sorted(required_fields - set(item))
        if missing:
            errors.append(f"{location} missing fields: {', '.join(missing)}")
            continue
        requirement_id = str(item["requirement_id"]).strip()
        if not requirement_id:
            errors.append(f"{location} has empty requirement_id")
            continue
        if requirement_id in by_id:
            errors.append(f"duplicate requirement_id {requirement_id}")
            continue
        by_id[requirement_id] = item
        parent = item.get("parent_id")
        if parent:
            parents[requirement_id] = str(parent)

        if item["requirement_kind"] not in REQUIREMENT_KINDS:
            errors.append(
                f"{requirement_id}: unknown requirement kind "
                f"{item['requirement_kind']!r}"
            )
        if item["requirement_type"] not in REQUIREMENT_TYPES:
            errors.append(
                f"{requirement_id}: unknown requirement type "
                f"{item['requirement_type']!r}"
            )
        if item["priority"] not in PRIORITIES:
            errors.append(f"{requirement_id}: unknown priority {item['priority']!r}")
        if item["overall_status"] not in OVERALL_STATUSES:
            errors.append(
                f"{requirement_id}: unknown status {item['overall_status']!r}"
            )
        if item["closure_priority"] not in CLOSURE_PRIORITIES:
            errors.append(
                f"{requirement_id}: unknown closure priority "
                f"{item['closure_priority']!r}"
            )
        if not str(item["normalized_requirement"]).strip():
            errors.append(f"{requirement_id}: normalized requirement is empty")

        source = item["source"]
        if not isinstance(source, Mapping):
            errors.append(f"{requirement_id}: source must be a mapping")
        else:
            source_path = str(source.get("path", "")).replace("\\", "/")
            if not source_path or not (root / source_path).is_file():
                errors.append(f"{requirement_id}: source does not exist: {source_path}")
            if not str(source.get("type", "")).strip():
                errors.append(f"{requirement_id}: source type is empty")
            if not str(source.get("section", "")).strip():
                errors.append(f"{requirement_id}: source section is empty")
            if not str(source.get("quote", "")).strip():
                errors.append(f"{requirement_id}: source quote is empty")
            if source.get("type") == "composite_project_requirement":
                composite_sources = source.get("sources")
                if not isinstance(composite_sources, list) or len(composite_sources) < 2:
                    errors.append(
                        f"{requirement_id}: composite source requires at least "
                        "two sources"
                    )
                else:
                    primary_signature = (
                        source_path,
                        str(source.get("section", "")),
                        str(source.get("quote", "")),
                    )
                    source_signatures: set[tuple[str, str, str]] = set()
                    for source_index, component in enumerate(composite_sources):
                        component_location = (
                            f"{requirement_id}.source.sources[{source_index}]"
                        )
                        if not isinstance(component, Mapping):
                            errors.append(f"{component_location} must be a mapping")
                            continue
                        component_path = str(component.get("path", "")).replace(
                            "\\", "/"
                        )
                        component_signature = (
                            component_path,
                            str(component.get("section", "")),
                            str(component.get("quote", "")),
                        )
                        source_signatures.add(component_signature)
                        if not component_path or not (root / component_path).is_file():
                            errors.append(
                                f"{component_location}: source does not exist: "
                                f"{component_path}"
                            )
                        for field in ("type", "section", "quote"):
                            if not str(component.get(field, "")).strip():
                                errors.append(
                                    f"{component_location}: {field} is empty"
                                )
                    if primary_signature not in source_signatures:
                        errors.append(
                            f"{requirement_id}: primary source must be listed "
                            "among composite sources"
                        )

        coverage = item["coverage"]
        if not isinstance(coverage, Mapping):
            errors.append(f"{requirement_id}: coverage must be a mapping")
        else:
            missing_coverage = sorted(set(COVERAGE_FIELDS) - set(coverage))
            if missing_coverage:
                errors.append(
                    f"{requirement_id}: missing coverage fields "
                    f"{', '.join(missing_coverage)}"
                )
            for field in COVERAGE_FIELDS:
                value = coverage.get(field)
                if value not in COVERAGE_STATUSES:
                    errors.append(
                        f"{requirement_id}: unknown coverage status "
                        f"{field}={value!r}"
                    )
            if item["overall_status"] == "complete" and any(
                coverage.get(field) in {"missing", "partial", "unclear"}
                for field in COVERAGE_FIELDS
            ):
                errors.append(
                    f"{requirement_id}: complete has an incomplete mandatory dimension"
                )

        evidence = item["evidence"]
        if not isinstance(evidence, list):
            errors.append(f"{requirement_id}: evidence must be a list")
            evidence = []
        if item["overall_status"] == "complete" and not evidence:
            errors.append(f"{requirement_id}: complete requires evidence")
        if item["overall_status"] == "partial" and not item.get("gaps"):
            errors.append(f"{requirement_id}: partial requires at least one gap")
        if item["overall_status"] == "not_applicable" and not str(
            item.get("not_applicable_reason", "")
        ).strip():
            errors.append(f"{requirement_id}: not_applicable requires reason")

        for evidence_index, entry in enumerate(evidence):
            evidence_location = f"{requirement_id}.evidence[{evidence_index}]"
            if not isinstance(entry, Mapping):
                errors.append(f"{evidence_location} must be a mapping")
                continue
            evidence_type = entry.get("type")
            level = entry.get("level")
            if evidence_type not in EVIDENCE_TYPES:
                errors.append(
                    f"{evidence_location}: unknown evidence type {evidence_type!r}"
                )
                continue
            if level not in EVIDENCE_LEVELS:
                errors.append(f"{evidence_location}: unknown level {level!r}")
            if not str(entry.get("supports", "")).strip():
                errors.append(f"{evidence_location}: supports is empty")
            if evidence_type == "experiment":
                experiment_id = str(entry.get("experiment_id", ""))
                experiment = experiments.get(experiment_id)
                if experiment is None:
                    errors.append(
                        f"{evidence_location}: unknown experiment ID {experiment_id}"
                    )
                elif level == "primary" and experiment.get("status") in {
                    "smoke",
                    "invalidated",
                }:
                    errors.append(
                        f"{evidence_location}: primary evidence cannot use "
                        f"{experiment.get('status')} experiment"
                    )
            elif evidence_type == "config":
                config_path = PurePosixPath(
                    str(entry.get("path", "")).replace("\\", "/")
                ).as_posix()
                if config_path not in configs:
                    errors.append(
                        f"{evidence_location}: unknown config {config_path}"
                    )
            elif evidence_type == "commit":
                commit = str(entry.get("commit", ""))
                if strict and not _commit_exists(root, commit):
                    errors.append(f"{evidence_location}: unknown commit {commit}")
            else:
                path = str(entry.get("path", "")).replace("\\", "/")
                exists = bool(path and (root / path).exists())
                if evidence_type == "artifact" and not exists:
                    if entry.get("optional_runtime"):
                        warnings.append(
                            f"{evidence_location}: optional runtime artifact missing: {path}"
                        )
                    else:
                        errors.append(
                            f"{evidence_location}: runtime artifact missing: {path}"
                        )
                elif not exists:
                    errors.append(f"{evidence_location}: evidence path missing: {path}")
                elif strict and evidence_type in {
                    "report",
                    "documentation",
                    "demo",
                } and not _tracked(root, path):
                    errors.append(
                        f"{evidence_location}: tracked report/documentation required"
                    )

        criterion = item.get("acceptance_criterion")
        if item["overall_status"] == "failed_acceptance_criterion":
            if not isinstance(criterion, Mapping):
                errors.append(
                    f"{requirement_id}: failed criterion requires expected and actual"
                )
            elif (
                not str(criterion.get("metric", "")).strip()
                or criterion.get("expected") is None
                or criterion.get("actual") is None
                or criterion.get("operator") not in {">=", "<=", ">", "<", "=="}
            ):
                errors.append(
                    f"{requirement_id}: failed criterion requires expected and actual"
                )
            elif not _criterion_is_failed(criterion):
                errors.append(
                    f"{requirement_id}: actual result does not fail the criterion"
                )

        closure = item["minimum_closure_action"]
        if not isinstance(closure, Mapping):
            errors.append(f"{requirement_id}: closure action must be a mapping")
        else:
            action_type = closure.get("action_type")
            if action_type not in ACTION_TYPES:
                errors.append(
                    f"{requirement_id}: unknown action type {action_type!r}"
                )
            if closure.get("estimated_scope") not in ESTIMATED_SCOPES:
                errors.append(
                    f"{requirement_id}: unknown estimated scope "
                    f"{closure.get('estimated_scope')!r}"
                )
            if not str(closure.get("description", "")).strip():
                errors.append(f"{requirement_id}: closure description is empty")
            if (
                action_type == "full_experiment"
                and item.get("new_experiment_required") is not True
            ):
                errors.append(
                    f"{requirement_id}: full_experiment requires "
                    "new_experiment_required=true"
                )
            if (
                action_type == "documentation"
                and item.get("new_experiment_required") is not False
            ):
                errors.append(
                    f"{requirement_id}: documentation action must not require "
                    "a new experiment"
                )

        if item["overall_status"] == "complete":
            capabilities = {
                str(entry.get("capability", ""))
                for entry in evidence
                if isinstance(entry, Mapping)
            }
            domain = item.get("domain")
            required_capability = {
                "streaming": "streaming",
                "demo": "demo",
                "domain_adaptation": "domain_adaptation",
                "multimodality": "multimodality",
            }.get(str(domain))
            if required_capability and required_capability not in capabilities:
                errors.append(
                    f"{requirement_id}: {domain} cannot be complete from "
                    "non-matching evidence"
                )

    for requirement_id, parent in parents.items():
        if parent not in by_id:
            errors.append(f"{requirement_id}: unknown parent_id {parent}")
    cycle = _hierarchy_cycle(parents)
    if cycle:
        errors.append("requirement hierarchy cycle: " + " -> ".join(cycle))

    if errors:
        raise RequirementsRegistryError("\n".join(sorted(set(errors))))
    return sorted(set(warnings))


def _hierarchy_key(item: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]):
    chain = [str(item["requirement_id"])]
    parent = item.get("parent_id")
    while parent:
        chain.append(str(parent))
        parent = by_id.get(str(parent), {}).get("parent_id")
    chain.reverse()
    return (
        tuple(chain),
        PRIORITY_ORDER[str(item["closure_priority"])],
        str(item["requirement_id"]),
    )


def sorted_requirements(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    requirements = registry["requirements"]
    by_id = {str(item["requirement_id"]): item for item in requirements}
    return sorted(requirements, key=lambda item: _hierarchy_key(item, by_id))


def _evidence_label(entry: Mapping[str, Any]) -> str:
    if entry.get("experiment_id"):
        return f"experiment:{entry['experiment_id']}"
    if entry.get("path"):
        return f"{entry['type']}:{entry['path']}"
    if entry.get("commit"):
        return f"commit:{entry['commit']}"
    return str(entry.get("type", ""))


def render_traceability_csv(registry: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in sorted_requirements(registry):
        coverage = item["coverage"]
        evidence = item["evidence"]
        primary = [
            _evidence_label(entry)
            for entry in evidence
            if entry.get("level") == "primary"
        ]
        closure = item["minimum_closure_action"]
        source = item["source"]
        source_entries = (
            source["sources"]
            if source.get("type") == "composite_project_requirement"
            else [source]
        )
        composite = len(source_entries) > 1
        writer.writerow(
            {
                "requirement_id": item["requirement_id"],
                "parent_id": item.get("parent_id") or "",
                "title": item["title"],
                "requirement_type": item["requirement_type"],
                "priority": item["priority"],
                "source_type": source["type"],
                "source_path": (
                    _stable_json([entry["path"] for entry in source_entries])
                    if composite
                    else source["path"]
                ),
                "source_section": (
                    _stable_json([entry["section"] for entry in source_entries])
                    if composite
                    else source["section"]
                ),
                "source_quote": (
                    _stable_json([entry["quote"] for entry in source_entries])
                    if composite
                    else source["quote"]
                ),
                "normalized_requirement": item["normalized_requirement"],
                "implementation_status": coverage["implementation"],
                "config_status": coverage["reproducible_config"],
                "experimental_status": coverage["experimental_validation"],
                "result_quality_status": coverage["result_quality"],
                "documentation_status": coverage["documentation"],
                "integration_status": coverage["integration"],
                "demonstration_status": coverage["demonstration"],
                "overall_status": item["overall_status"],
                "evidence_count": len(evidence),
                "primary_evidence": _stable_json(primary),
                "gaps": _stable_json(item["gaps"]),
                "minimum_closure_action": closure["description"],
                "action_type": closure["action_type"],
                "estimated_scope": closure["estimated_scope"],
                "new_experiment_required": str(
                    bool(item["new_experiment_required"])
                ).lower(),
                "closure_priority": item["closure_priority"],
                "blocking_dependencies": _stable_json(
                    item["blocking_dependencies"]
                ),
                "notes": _stable_json(item["notes"]),
            }
        )
    return stream.getvalue()


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value if value not in (None, "") else "—").replace(
            "|", "\\|"
        ).replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _requirement_rows(items: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            item["requirement_id"],
            item["title"],
            item["overall_status"],
            item["closure_priority"],
            "; ".join(item["gaps"]) or "—",
        ]
        for item in items
    ]


def render_coverage_markdown(registry: Mapping[str, Any]) -> str:
    requirements = sorted_requirements(registry)
    status_counts = Counter(item["overall_status"] for item in requirements)
    project = registry["project"]
    by_status = defaultdict(list)
    by_domain = defaultdict(list)
    for item in requirements:
        by_status[item["overall_status"]].append(item)
        by_domain[item.get("domain", "other")].append(item)
    table_headers = ["ID", "Требование", "Статус", "Приоритет закрытия", "Пробел"]
    lines = [
        "# Карта соответствия требованиям проекта",
        "",
        "## 1. Источники требований",
        "",
        (
            f"Основной источник: `{project['primary_source']['path']}` "
            f"(`{project['primary_source']['type']}`). Статус источников: "
            f"`{project['requirements_source_status']}`."
        ),
        "",
        *[f"- {warning}" for warning in project.get("source_warnings", [])],
        "",
        "## 2. Метод оценки",
        "",
        "Реестр курируется вручную; генератор проверяет источники, evidence, иерархию, семь измерений покрытия и логические противоречия статусов.",
        "",
        "## 3. Сводный статус",
        "",
        _markdown_table(
            ["Статус", "Количество"],
            [[status, status_counts.get(status, 0)] for status in sorted(OVERALL_STATUSES)],
        ),
        "",
        f"Всего требований: **{len(requirements)}**.",
        "",
        "## 4. Полностью выполненные требования",
        "",
        _markdown_table(table_headers, _requirement_rows(by_status["complete"])),
        "",
        "## 5. Частично выполненные требования",
        "",
        _markdown_table(table_headers, _requirement_rows(by_status["partial"])),
        "",
        "## 6. Невыполненные требования",
        "",
        _markdown_table(table_headers, _requirement_rows(by_status["not_started"])),
        "",
        "## 7. Недостигнутые критерии качества",
        "",
        _markdown_table(
            table_headers,
            _requirement_rows(by_status["failed_acceptance_criterion"]),
        ),
        "",
        "## 8. Требования, нуждающиеся в уточнении",
        "",
        _markdown_table(
            table_headers,
            _requirement_rows(by_status["needs_clarification"]),
        ),
    ]
    sections = [
        ("9. Данные", "data"),
        ("10. Предобработка", "preprocessing"),
        ("11. Признаки", "features"),
        ("12. Модели", "models"),
        ("13. Оценка качества", "evaluation"),
        ("14. Персонализация и перенос", "personalization"),
        ("15. Воспроизводимая платформа", "platform"),
        ("16. Потоковый режим", "streaming"),
        ("17. Демонстрация", "demo"),
        ("18. Мультимодальность", "multimodality"),
        ("19. Документация и результаты", "documentation"),
    ]
    for title, domain in sections:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                _markdown_table(table_headers, _requirement_rows(by_domain[domain])),
            ]
        )
    incomplete = [
        item
        for item in requirements
        if item["overall_status"] not in {"complete", "not_applicable"}
    ]
    no_experiment = [
        item for item in incomplete if not item["new_experiment_required"]
    ]
    ordered = sorted(
        incomplete,
        key=lambda item: (
            PRIORITY_ORDER[item["closure_priority"]],
            item["requirement_id"],
        ),
    )
    lines.extend(
        [
            "",
            "## 20. Критические пробелы",
            "",
            _markdown_table(
                table_headers,
                _requirement_rows(
                    [item for item in ordered if item["closure_priority"] == "P0"]
                ),
            ),
            "",
            "## 21. Что не требует новых экспериментов",
            "",
            _markdown_table(table_headers, _requirement_rows(no_experiment)),
            "",
            "## 22. Рекомендуемый порядок закрытия",
            "",
            _markdown_table(table_headers, _requirement_rows(ordered)),
            "",
        ]
    )
    return "\n".join(lines)


def render_remaining_work(registry: Mapping[str, Any]) -> str:
    requirements = [
        item
        for item in registry["requirements"]
        if item["overall_status"] not in {"complete", "not_applicable"}
    ]
    lines = ["# Оставшиеся работы по проекту", ""]
    for priority in ("P0", "P1", "P2", "P3"):
        lines.extend([f"## {priority}", ""])
        selected = sorted(
            (
                item
                for item in requirements
                if item["closure_priority"] == priority
                and item["overall_status"] != "needs_clarification"
            ),
            key=lambda item: item["requirement_id"],
        )
        if not selected:
            lines.extend(["_Нет задач._", ""])
            continue
        for item in selected:
            closure = item["minimum_closure_action"]
            lines.extend(
                [
                    f"### {item['requirement_id']} — {item['title']}",
                    "",
                    f"- Пробел: {'; '.join(item['gaps']) or 'не указан'}.",
                    f"- Минимальное действие: {closure['description']}",
                    f"- Новый эксперимент: {'да' if item['new_experiment_required'] else 'нет'}.",
                    (
                        "- Зависимости: "
                        + ("; ".join(item["blocking_dependencies"]) or "нет")
                        + "."
                    ),
                    f"- Критерий завершения: {item['completion_criterion']}",
                    "",
                ]
            )
    lines.extend(["## Требуется уточнение scope", ""])
    clarification = sorted(
        (
            item
            for item in requirements
            if item["overall_status"] == "needs_clarification"
        ),
        key=lambda item: item["requirement_id"],
    )
    for item in clarification:
        lines.append(
            f"- **{item['requirement_id']}**: "
            f"{item['minimum_closure_action']['description']}"
        )
    lines.extend(["", "## Не рекомендуется выполнять сейчас", ""])
    for item in registry.get("deferred_now", []):
        lines.append(f"- **{item['title']}** — {item['reason']}")
    lines.append("")
    return "\n".join(lines)


def generate(
    registry_path: Path,
    experiment_registry_path: Path,
    config_registry_path: Path,
    output_dir: Path,
    root: Path,
    *,
    strict: bool,
) -> tuple[dict[str, Any], list[str]]:
    registry = _load_yaml(registry_path, "requirements registry")
    experiments = _load_yaml(experiment_registry_path, "experiment registry")
    configs = _load_yaml(config_registry_path, "config registry")
    warnings = validate_registry(
        registry,
        experiments,
        configs,
        root,
        strict=strict,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requirements_traceability.csv").write_text(
        render_traceability_csv(registry),
        encoding="utf-8",
        newline="",
    )
    (output_dir / "requirements_coverage.md").write_text(
        render_coverage_markdown(registry),
        encoding="utf-8",
        newline="",
    )
    (output_dir / "project_remaining_work.md").write_text(
        render_remaining_work(registry),
        encoding="utf-8",
        newline="",
    )
    return registry, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="reports/summary/requirements_registry.yaml",
    )
    parser.add_argument(
        "--experiment-registry",
        default="reports/summary/experiment_registry.yaml",
    )
    parser.add_argument(
        "--config-registry",
        default="reports/summary/config_registry.yaml",
    )
    parser.add_argument("--output-dir", default="reports/summary")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    try:
        registry_path = resolve(args.registry)
        experiment_path = resolve(args.experiment_registry)
        config_path = resolve(args.config_registry)
        registry = _load_yaml(registry_path, "requirements registry")
        experiments = _load_yaml(experiment_path, "experiment registry")
        configs = _load_yaml(config_path, "config registry")
        warnings = validate_registry(
            registry,
            experiments,
            configs,
            root,
            strict=args.strict,
        )
        if not args.validate:
            _, warnings = generate(
                registry_path,
                experiment_path,
                config_path,
                resolve(args.output_dir),
                root,
                strict=args.strict,
            )
    except RequirementsRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(
        f"Requirements {'validated' if args.validate else 'generated'}: "
        f"{len(registry['requirements'])}, warnings: {len(warnings)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
