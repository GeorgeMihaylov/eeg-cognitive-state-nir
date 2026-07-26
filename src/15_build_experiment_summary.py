"""Build deterministic project experiment summaries from a curated registry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


ALLOWED_STATUSES = {
    "final",
    "baseline",
    "smoke",
    "diagnostic",
    "invalidated",
}
ALLOWED_CATEGORIES = {
    "model",
    "preprocessing",
    "regression",
    "personalization",
    "mixin",
    "infrastructure",
}
STATUS_PRIORITY = {
    "final": 0,
    "baseline": 1,
    "diagnostic": 2,
    "smoke": 3,
    "invalidated": 4,
}
AGGREGATIONS = {"first", "mean", "median", "min", "max", "sum", "count"}
REQUIRED_FIELDS = {
    "experiment_id",
    "title",
    "category",
    "status",
    "task",
    "model",
    "feature_set",
    "preprocessing",
    "evaluation_protocol",
    "seeds",
    "primary_metric",
    "result_summary",
    "report_path",
}
PATH_FIELDS = {"report_path", "runtime_path", "config_path", "path"}
SUMMARY_COLUMNS = [
    "experiment_id",
    "title",
    "category",
    "status",
    "task",
    "target",
    "model",
    "feature_set",
    "preprocessing",
    "evaluation_protocol",
    "seeds",
    "n_subjects",
    "primary_metric",
    "primary_value",
    "primary_metric_direction",
    "primary_metric_source_type",
    "primary_metric_source_path",
    "secondary_metrics_json",
    "result_summary",
    "report_path",
    "runtime_path",
    "config_path",
    "commit",
    "limitations",
    "tags",
]
MODEL_COLUMNS = [
    "task",
    "model",
    "feature_set",
    "preprocessing",
    "evaluation_protocol",
    "seeds",
    "n_subjects",
    "primary_metric",
    "primary_value",
    "secondary_metrics_json",
    "experiment_id",
    "report_path",
    "commit",
]
PERSONALIZATION_COLUMNS = [
    "task",
    "target",
    "method",
    "calibration_budget",
    "seeds",
    "n_subjects",
    "baseline_metric",
    "personalized_metric",
    "gain_metric",
    "gain_value",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "improved_subject_fraction",
    "experiment_id",
    "report_path",
    "commit",
]


class RegistryError(ValueError):
    """Raised when the registry or one of its evidence sources is invalid."""


def _is_absolute_path(value: str) -> bool:
    path = str(value).strip()
    return (
        Path(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or bool(re.match(r"^[A-Za-z]:[\\/]", path))
        or path.startswith(("/", "\\\\"))
    )


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegistryError("Summary values must be finite")
        return f"{value:.10g}"
    return str(value)


def _list_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(_display_value(item) for item in value)
    return _display_value(value)


def load_registry(path: Path | str) -> dict[str, Any]:
    """Load a YAML registry and verify its top-level shape."""
    registry_path = Path(path)
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"Registry does not exist: {registry_path}") from exc
    except yaml.YAMLError as exc:
        raise RegistryError(f"Invalid registry YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryError("Registry root must be a mapping")
    if payload.get("schema_version") != 1:
        raise RegistryError("Registry schema_version must equal 1")
    if not isinstance(payload.get("experiments"), list):
        raise RegistryError("Registry experiments must be a list")
    unresolved = payload.get("unresolved_entries", [])
    if unresolved is not None and not isinstance(unresolved, list):
        raise RegistryError("unresolved_entries must be a list")
    return payload


def _walk_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in PATH_FIELDS and child not in (None, ""):
                yield child_prefix, str(child)
            yield from _walk_paths(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_paths(child, f"{prefix}[{index}]")


def _validate_metric_spec(spec: Any, location: str) -> None:
    if not isinstance(spec, Mapping):
        raise RegistryError(f"{location} must be a mapping")
    if not str(spec.get("name", "")).strip():
        raise RegistryError(f"{location}.name is required")
    source = spec.get("value_source")
    if not isinstance(source, Mapping):
        raise RegistryError(f"{location}.value_source is required")
    source_type = source.get("type")
    if source_type not in {"constant", "json", "csv_filter", "yaml", "report_only"}:
        raise RegistryError(f"{location} has unknown value source {source_type!r}")
    if source_type == "constant" and "value" not in source:
        raise RegistryError(f"{location} constant source requires value")
    if source_type in {"json", "csv_filter", "yaml"} and not source.get("path"):
        raise RegistryError(f"{location} {source_type} source requires path")
    if source_type in {"json", "yaml"} and not source.get("key"):
        raise RegistryError(f"{location} {source_type} source requires key")
    if source_type == "csv_filter":
        if not source.get("column") and source.get("aggregation") != "count":
            raise RegistryError(f"{location} csv_filter source requires column")
        aggregation = source.get("aggregation")
        if aggregation is not None and aggregation not in AGGREGATIONS:
            raise RegistryError(
                f"{location} has unsupported aggregation {aggregation!r}"
            )


def _validate_supersession(experiments: Sequence[Mapping[str, Any]]) -> None:
    identifiers = {str(item["experiment_id"]) for item in experiments}
    links: dict[str, str] = {}
    for item in experiments:
        target = item.get("superseded_by")
        if target is None:
            continue
        if target not in identifiers:
            raise RegistryError(
                f"{item['experiment_id']} superseded_by references unknown "
                f"experiment {target!r}"
            )
        links[str(item["experiment_id"])] = str(target)
    for start in links:
        seen: set[str] = set()
        current = start
        while current in links:
            if current in seen:
                raise RegistryError(f"superseded_by cycle contains {current!r}")
            seen.add(current)
            current = links[current]


def validate_registry(
    registry: Mapping[str, Any],
    repo_root: Path | str,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate schema and evidence paths, returning non-fatal warnings."""
    root = Path(repo_root)
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise RegistryError("Registry experiments must be a list")
    warnings: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(experiments):
        location = f"experiments[{index}]"
        if not isinstance(item, Mapping):
            raise RegistryError(f"{location} must be a mapping")
        missing = sorted(field for field in REQUIRED_FIELDS if field not in item)
        if missing:
            raise RegistryError(f"{location} missing required fields: {missing}")
        experiment_id = str(item["experiment_id"])
        if experiment_id in seen:
            raise RegistryError(f"Duplicate experiment_id: {experiment_id}")
        seen.add(experiment_id)
        if item["status"] not in ALLOWED_STATUSES:
            raise RegistryError(
                f"{experiment_id} has unknown status {item['status']!r}"
            )
        if item["category"] not in ALLOWED_CATEGORIES:
            raise RegistryError(
                f"{experiment_id} has unknown category {item['category']!r}"
            )
        if item["status"] == "invalidated":
            if not str(item.get("invalidation_reason", "")).strip():
                raise RegistryError(
                    f"{experiment_id} invalidated entry requires "
                    "invalidation_reason"
                )
            if "superseded_by" not in item:
                raise RegistryError(
                    f"{experiment_id} invalidated entry requires superseded_by"
                )
        if item["category"] == "mixin":
            for field in (
                "mixin_name",
                "audit_status",
                "integration_status",
                "decision",
                "decision_reason",
            ):
                if not str(item.get(field, "")).strip():
                    raise RegistryError(
                        f"{experiment_id} mixin entry requires {field}"
                    )
        _validate_metric_spec(
            item["primary_metric"], f"{experiment_id}.primary_metric"
        )
        for metric_index, metric in enumerate(item.get("secondary_metrics") or []):
            _validate_metric_spec(
                metric,
                f"{experiment_id}.secondary_metrics[{metric_index}]",
            )
        for path_location, path_value in _walk_paths(item):
            if _is_absolute_path(path_value):
                raise RegistryError(
                    f"{experiment_id}.{path_location} must be relative: "
                    f"{path_value}"
                )
        report_path = root / str(item["report_path"])
        if not report_path.is_file():
            message = f"{experiment_id}: tracked report missing: {item['report_path']}"
            if strict:
                raise RegistryError(message)
            warnings.append(message)
        config_path = item.get("config_path")
        if config_path and not (root / str(config_path)).is_file():
            message = f"{experiment_id}: tracked config missing: {config_path}"
            if strict and item["status"] in {"final", "baseline"}:
                raise RegistryError(message)
            warnings.append(message)
        runtime_path = item.get("runtime_path")
        if runtime_path and not (root / str(runtime_path)).exists():
            warnings.append(
                f"{experiment_id}: runtime path missing: {runtime_path}"
            )
        commit = item.get("commit")
        if commit is None:
            warnings.append(f"{experiment_id}: commit is not established")
        elif not re.fullmatch(r"[0-9a-f]{7,40}", str(commit)):
            raise RegistryError(f"{experiment_id} has invalid commit {commit!r}")
    _validate_supersession(experiments)
    for index, entry in enumerate(registry.get("unresolved_entries") or []):
        if not isinstance(entry, Mapping):
            raise RegistryError(f"unresolved_entries[{index}] must be a mapping")
        for field in (
            "experiment_id",
            "missing_field",
            "checked_paths",
            "recommended_action",
        ):
            if field not in entry:
                raise RegistryError(
                    f"unresolved_entries[{index}] requires {field}"
                )
        for _, path_value in _walk_paths(entry):
            if _is_absolute_path(path_value):
                raise RegistryError(
                    "unresolved_entries may not contain absolute paths"
                )
    return sorted(set(warnings))


def _nested_value(payload: Any, dotted_key: str) -> Any:
    current = payload
    for component in str(dotted_key).split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            current = current[int(component)]
        else:
            raise RegistryError(f"Key {dotted_key!r} does not exist")
    return current


def _filter_frame(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame
    for column, expected in filters.items():
        if column not in selected.columns:
            raise RegistryError(f"CSV filter column does not exist: {column}")
        series = selected[column]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            numeric = pd.to_numeric(series, errors="coerce")
            mask = numeric.notna() & (
                (numeric.astype(float) - float(expected)).abs() <= 1e-12
            )
        else:
            mask = series.astype(str) == str(expected)
        selected = selected.loc[mask]
    return selected


def _aggregate_series(
    frame: pd.DataFrame,
    column: str | None,
    aggregation: str | None,
) -> Any:
    if frame.empty:
        raise RegistryError("CSV filter matched zero rows")
    if aggregation is None:
        if len(frame) != 1:
            raise RegistryError(
                f"CSV filter is ambiguous: matched {len(frame)} rows without "
                "an explicit aggregation"
            )
        if column not in frame.columns:
            raise RegistryError(f"CSV value column does not exist: {column}")
        return frame.iloc[0][str(column)]
    if aggregation == "count":
        return int(len(frame))
    if column not in frame.columns:
        raise RegistryError(f"CSV value column does not exist: {column}")
    series = pd.to_numeric(frame[str(column)], errors="raise")
    if aggregation == "first":
        return series.iloc[0]
    return getattr(series, aggregation)()


def _normalise_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise RegistryError("Extracted metric must be finite")
    return value


def extract_value(
    value_source: Mapping[str, Any],
    repo_root: Path | str,
) -> tuple[Any, dict[str, str]]:
    """Extract one value and its machine-readable provenance."""
    root = Path(repo_root)
    source_type = str(value_source.get("type", ""))
    provenance = {
        "type": source_type,
        "path": str(value_source.get("path") or ""),
        "selector": "",
    }
    if source_type == "constant":
        provenance["type"] = "registry_constant"
        return _normalise_scalar(value_source.get("value")), provenance
    if source_type == "report_only":
        return None, provenance
    path = root / str(value_source.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    if source_type == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = str(value_source["key"])
        provenance["selector"] = key
        return _normalise_scalar(_nested_value(payload, key)), provenance
    if source_type == "yaml":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        key = str(value_source["key"])
        provenance["selector"] = key
        return _normalise_scalar(_nested_value(payload, key)), provenance
    if source_type == "csv_filter":
        filters = value_source.get("filters") or {}
        if not isinstance(filters, Mapping):
            raise RegistryError("csv_filter filters must be a mapping")
        frame = _filter_frame(pd.read_csv(path), filters)
        aggregation = value_source.get("aggregation")
        column = value_source.get("column")
        provenance["selector"] = _stable_json(
            {
                "filters": dict(filters),
                "column": column,
                "aggregation": aggregation,
            }
        )
        value = _aggregate_series(frame, column, aggregation)
        return _normalise_scalar(value), provenance
    raise RegistryError(f"Unsupported value source type: {source_type!r}")


def _metric_result(
    spec: Mapping[str, Any],
    experiment: Mapping[str, Any],
    repo_root: Path,
    *,
    strict: bool,
    warnings: list[str],
) -> dict[str, Any]:
    try:
        value, provenance = extract_value(spec["value_source"], repo_root)
    except FileNotFoundError:
        source_path = str(spec["value_source"].get("path") or "")
        optional_runtime = bool(spec["value_source"].get("optional_runtime"))
        has_report = bool(experiment.get("report_path"))
        if optional_runtime and has_report:
            warnings.append(
                f"{experiment['experiment_id']}: optional metric source "
                f"missing: {source_path}"
            )
            value = None
            provenance = {
                "type": str(spec["value_source"].get("type") or ""),
                "path": source_path,
                "selector": "",
            }
        else:
            raise RegistryError(
                f"{experiment['experiment_id']}: metric source missing: "
                f"{source_path}"
            )
    except (RegistryError, OSError, ValueError, KeyError) as exc:
        raise RegistryError(
            f"{experiment['experiment_id']}: failed to extract "
            f"{spec.get('name')}: {exc}"
        ) from exc
    return {
        "name": str(spec["name"]),
        "direction": str(spec.get("direction") or ""),
        "value": value,
        "source_type": provenance["type"],
        "source_path": provenance["path"],
        "source_selector": provenance["selector"],
    }


def _experiment_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["category"]),
        STATUS_PRIORITY[str(item["status"])],
        str(item.get("task") or ""),
        str(item.get("model") or ""),
        str(item["experiment_id"]),
    )


def _summary_row(
    experiment: Mapping[str, Any],
    repo_root: Path,
    *,
    strict: bool,
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = _metric_result(
        experiment["primary_metric"],
        experiment,
        repo_root,
        strict=strict,
        warnings=warnings,
    )
    secondary = [
        _metric_result(
            spec,
            experiment,
            repo_root,
            strict=strict,
            warnings=warnings,
        )
        for spec in (experiment.get("secondary_metrics") or [])
    ]
    secondary_payload = [
        {
            "direction": item["direction"],
            "name": item["name"],
            "source_path": item["source_path"],
            "source_type": item["source_type"],
            "value": item["value"],
        }
        for item in secondary
    ]
    row = {
        "experiment_id": experiment["experiment_id"],
        "title": experiment["title"],
        "category": experiment["category"],
        "status": experiment["status"],
        "task": experiment["task"],
        "target": experiment.get("target"),
        "model": experiment["model"],
        "feature_set": experiment["feature_set"],
        "preprocessing": experiment["preprocessing"],
        "evaluation_protocol": experiment["evaluation_protocol"],
        "seeds": _list_value(experiment.get("seeds")),
        "n_subjects": experiment.get("n_subjects"),
        "primary_metric": primary["name"],
        "primary_value": primary["value"],
        "primary_metric_direction": primary["direction"],
        "primary_metric_source_type": primary["source_type"],
        "primary_metric_source_path": primary["source_path"],
        "secondary_metrics_json": _stable_json(secondary_payload),
        "result_summary": experiment["result_summary"],
        "report_path": experiment["report_path"],
        "runtime_path": experiment.get("runtime_path"),
        "config_path": experiment.get("config_path"),
        "commit": experiment.get("commit"),
        "limitations": _list_value(experiment.get("limitations")),
        "tags": _list_value(experiment.get("tags")),
    }
    return row, secondary


def _personalization_value(
    value: Any,
    experiment: Mapping[str, Any],
    repo_root: Path,
    warnings: list[str],
    name: str,
) -> Any:
    if isinstance(value, Mapping) and "value_source" in value:
        spec = {"name": name, **value}
        return _metric_result(
            spec,
            experiment,
            repo_root,
            strict=False,
            warnings=warnings,
        )["value"]
    return value


def build_summaries(
    registry: Mapping[str, Any],
    repo_root: Path | str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Build all deterministic in-memory summary tables."""
    root = Path(repo_root)
    warnings = validate_registry(registry, root, strict=strict)
    experiments = sorted(registry["experiments"], key=_experiment_sort_key)
    rows: list[dict[str, Any]] = []
    secondary_by_id: dict[str, list[dict[str, Any]]] = {}
    for experiment in experiments:
        row, secondary = _summary_row(
            experiment,
            root,
            strict=strict,
            warnings=warnings,
        )
        rows.append(row)
        secondary_by_id[str(experiment["experiment_id"])] = secondary
    model_rows = [
        {column: row.get(column) for column in MODEL_COLUMNS}
        for row in rows
        if row["category"] in {"model", "regression"}
        and row["status"] in {"final", "baseline"}
    ]
    personalization_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        if experiment["category"] != "personalization":
            continue
        if experiment["status"] not in {"final", "diagnostic"}:
            continue
        details = experiment.get("personalization")
        if not isinstance(details, Mapping):
            raise RegistryError(
                f"{experiment['experiment_id']} requires personalization details"
            )
        result = {
            "task": experiment["task"],
            "target": experiment.get("target"),
            "method": details.get("method"),
            "calibration_budget": details.get("calibration_budget"),
            "seeds": _list_value(experiment.get("seeds")),
            "n_subjects": experiment.get("n_subjects"),
            "baseline_metric": _personalization_value(
                details.get("baseline_metric"),
                experiment,
                root,
                warnings,
                "baseline_metric",
            ),
            "personalized_metric": _personalization_value(
                details.get("personalized_metric"),
                experiment,
                root,
                warnings,
                "personalized_metric",
            ),
            "gain_metric": details.get("gain_metric"),
            "gain_value": _personalization_value(
                details.get("gain_value"),
                experiment,
                root,
                warnings,
                "gain_value",
            ),
            "bootstrap_ci_low": _personalization_value(
                details.get("bootstrap_ci_low"),
                experiment,
                root,
                warnings,
                "bootstrap_ci_low",
            ),
            "bootstrap_ci_high": _personalization_value(
                details.get("bootstrap_ci_high"),
                experiment,
                root,
                warnings,
                "bootstrap_ci_high",
            ),
            "improved_subject_fraction": _personalization_value(
                details.get("improved_subject_fraction"),
                experiment,
                root,
                warnings,
                "improved_subject_fraction",
            ),
            "experiment_id": experiment["experiment_id"],
            "report_path": experiment["report_path"],
            "commit": experiment.get("commit"),
        }
        personalization_rows.append(result)
    return {
        "experiments": experiments,
        "experiment_rows": rows,
        "model_rows": model_rows,
        "personalization_rows": personalization_rows,
        "secondary_by_id": secondary_by_id,
        "unresolved": list(registry.get("unresolved_entries") or []),
        "warnings": sorted(set(warnings)),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(columns),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _display_value(row.get(column)) for column in columns}
            )


def _markdown_link(path: str | None) -> str:
    if not path:
        return ""
    return f"[отчёт](../../{str(path).replace(chr(92), '/')})"


def _markdown_table(
    experiments: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
    secondary_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    lines = [
        "| Эксперимент | Задача | Модель | Протокол | Seeds | Пользователи | Основная метрика | Дополнительные | Статус | Evidence |",
        "|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        row = rows[experiment_id]
        metric = str(row["primary_metric"])
        if row["primary_value"] not in (None, ""):
            metric += f" = {_display_value(row['primary_value'])}"
        secondary = "; ".join(
            f"{item['name']}={_display_value(item['value'])}"
            for item in secondary_by_id.get(experiment_id, [])[:3]
            if item.get("value") not in (None, "")
        )
        limitations = _list_value(experiment.get("limitations"))
        summary = str(experiment["result_summary"])
        if limitations:
            summary += f" Ограничения: {limitations}"
            if not summary.endswith((".", "!", "?")):
                summary += "."
        title = f"**{experiment['title']}**<br>{summary}"
        lines.append(
            "| "
            + " | ".join(
                [
                    title.replace("|", "\\|"),
                    str(experiment["task"]).replace("|", "\\|"),
                    str(experiment["model"]).replace("|", "\\|"),
                    str(experiment["evaluation_protocol"]).replace("|", "\\|"),
                    _list_value(experiment.get("seeds")),
                    _display_value(experiment.get("n_subjects")),
                    metric.replace("|", "\\|"),
                    secondary.replace("|", "\\|"),
                    str(experiment["status"]),
                    _markdown_link(str(experiment["report_path"])),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    return lines


def render_experiment_markdown(summary: Mapping[str, Any]) -> str:
    experiments = list(summary["experiments"])
    rows = {
        str(row["experiment_id"]): row for row in summary["experiment_rows"]
    }
    secondary_by_id = summary["secondary_by_id"]

    def select(predicate: Any) -> list[Mapping[str, Any]]:
        return [item for item in experiments if predicate(item)]

    final = select(
        lambda item: item["status"] == "final"
        and item["category"] not in {"personalization", "mixin"}
    )
    baseline = select(
        lambda item: item["status"] == "baseline"
        and item["category"] not in {"personalization", "mixin"}
    )
    diagnostics = select(
        lambda item: item["category"] in {"preprocessing", "infrastructure"}
        and item["status"] in {"diagnostic", "smoke"}
    )
    personalization = select(
        lambda item: item["category"] == "personalization"
    )
    mixins = select(lambda item: item["category"] == "mixin")
    invalidated = select(lambda item: item["status"] == "invalidated")
    lines = [
        "# Сводка экспериментов",
        "",
        "Ручной состав и статусы задаются в `experiment_registry.yaml`; "
        "числовые значения извлекаются только из явно указанных источников.",
        "",
    ]
    for title, selected in (
        ("Итоговые результаты", final),
        ("Базовые результаты", baseline),
        ("Предобработка и диагностика", diagnostics),
        ("Персонализация", personalization),
        ("Mixins", mixins),
        ("Невалидные и заменённые запуски", invalidated),
    ):
        lines.extend([f"## {title}", ""])
        lines.extend(_markdown_table(selected, rows, secondary_by_id))
        lines.append("")
    lines.extend(["## Неразрешённые записи", ""])
    unresolved = summary["unresolved"]
    if unresolved:
        lines.extend(
            [
                "| Experiment ID | Отсутствует | Проверено | Рекомендуемое действие |",
                "|---|---|---|---|",
            ]
        )
        for item in unresolved:
            checked = _list_value(item.get("checked_paths"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["experiment_id"]).replace("|", "\\|"),
                        str(item["missing_field"]).replace("|", "\\|"),
                        checked.replace("|", "\\|"),
                        str(item["recommended_action"]).replace("|", "\\|"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("Неразрешённых записей нет.")
    lines.append("")
    return "\n".join(lines)


def render_mixin_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Статус исторических mixin-прототипов",
        "",
        "| Mixin | Проверен | Запускается | Интегрирован | Решение | Причина |",
        "|---|---:|---:|---:|---|---|",
    ]
    mixins = [
        item for item in summary["experiments"] if item["category"] == "mixin"
    ]
    for item in mixins:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["mixin_name"]).replace("|", "\\|"),
                    str(item["audit_status"]).replace("|", "\\|"),
                    str(item.get("prototype_runnable", "no")).replace("|", "\\|"),
                    str(item["integration_status"]).replace("|", "\\|"),
                    str(item["decision"]).replace("|", "\\|"),
                    str(item["decision_reason"]).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`TransferLearningMixin` не считается production-ready: его назначение "
            "интегрировано как заново реализованный leakage-safe fine-tuning "
            "pipeline. DANN, MAML и contrastive pretraining не интегрированы.",
            "",
        ]
    )
    return "\n".join(lines)


def generate(
    registry_path: Path | str,
    output_dir: Path | str,
    repo_root: Path | str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate the registry and atomically regenerate all summary files."""
    registry = load_registry(registry_path)
    summary = build_summaries(registry, repo_root, strict=strict)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(
        destination / "experiment_summary.csv",
        SUMMARY_COLUMNS,
        summary["experiment_rows"],
    )
    _write_csv(
        destination / "model_summary.csv",
        MODEL_COLUMNS,
        summary["model_rows"],
    )
    _write_csv(
        destination / "personalization_summary.csv",
        PERSONALIZATION_COLUMNS,
        summary["personalization_rows"],
    )
    (destination / "experiment_summary.md").write_text(
        render_experiment_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    (destination / "mixin_status.md").write_text(
        render_mixin_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="reports/summary/experiment_registry.yaml",
    )
    parser.add_argument("--output-dir", default="reports/summary")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate without generating output files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat schema and tracked-evidence problems as fatal",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    try:
        if args.validate:
            registry = load_registry(registry_path)
            warnings = validate_registry(
                registry,
                repo_root,
                strict=args.strict,
            )
            summary = build_summaries(
                registry,
                repo_root,
                strict=args.strict,
            )
            warnings = sorted(set(warnings + summary["warnings"]))
        else:
            summary = generate(
                registry_path,
                output_dir,
                repo_root,
                strict=args.strict,
            )
            warnings = summary["warnings"]
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for unresolved in summary["unresolved"]:
        print(
            "WARNING: unresolved "
            f"{unresolved['experiment_id']}: {unresolved['missing_field']}",
            file=sys.stderr,
        )
    mode = "validated" if args.validate else "generated"
    print(
        f"Registry {mode}: {len(summary['experiment_rows'])} experiments, "
        f"{len(summary['unresolved'])} unresolved entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
