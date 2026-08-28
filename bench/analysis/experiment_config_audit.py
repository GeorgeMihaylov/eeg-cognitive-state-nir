#!/usr/bin/env python3
"""Read-only inventory and audit of tracked experiment configuration files.

The project intentionally has several configuration contracts.  This module
documents those contracts; it does not introduce a new runtime schema and is
not imported by the benchmark CLI.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib
import io
import json
import logging
import re
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
ROLE_ORDER = {
    "root": 0,
    "base": 1,
    "full": 2,
    "smoke": 3,
    "diagnostic": 4,
    "ablation": 5,
    "legacy": 6,
    "unknown": 7,
}
VALID_ROLES = frozenset(ROLE_ORDER)
VALID_STATUSES = frozenset(
    {"final", "baseline", "smoke", "diagnostic", "invalidated", "unclassified"}
)
VALID_REVIEW_STATUSES = frozenset(
    {"reviewed", "needs_evidence", "not_applicable"}
)
VALID_DECISIONS = frozenset(
    {
        "keep",
        "keep_as_base",
        "keep_as_smoke",
        "keep_as_diagnostic",
        "keep_as_legacy",
        "superseded",
        "review_later",
    }
)
VALID_PROVENANCE_STATUSES = frozenset(
    {"documented", "partially_documented", "unresolved"}
)
VALID_CANONICAL_STATUSES = frozenset(
    {"completed", "active", "planned", "historical", "not_canonical"}
)
PM_TARGETS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]
INVENTORY_COLUMNS = [
    "config_path",
    "file_name",
    "topic",
    "config_role",
    "result_status",
    "automatic_config_role",
    "automatic_result_status",
    "review_status",
    "decision",
    "decision_reason",
    "canonical_config",
    "canonical_status",
    "provenance_status",
    "linked_report",
    "linked_runtime",
    "supersession_reason",
    "safe_to_move",
    "safe_to_edit",
    "curation_evidence",
    "orchestration_provenance",
    "protected_fields",
    "loader_type",
    "base_config",
    "base_config_exists",
    "inheritance_depth",
    "resolved_config_hash",
    "used_by_experiment_registry",
    "registry_experiment_ids",
    "cli_loadable",
    "expected_cli_argument",
    "schema_valid",
    "scientific_protocol_valid",
    "task",
    "task_type",
    "target",
    "targets",
    "model",
    "feature_set",
    "preprocessing",
    "outer_evaluation",
    "inner_validation",
    "n_folds",
    "split_seed",
    "model_seed",
    "model_seeds",
    "device",
    "output_dir",
    "resume_enabled",
    "absolute_paths_found",
    "duplicate_group",
    "report_linked",
    "is_legacy",
    "superseded_by",
    "deprecation_reason",
    "exact_hash",
    "scientific_protocol_hash",
    "issues_count",
    "errors",
    "warnings",
    "notes",
]
CURATION_INVENTORY_COLUMNS = frozenset(
    {
        "automatic_config_role",
        "automatic_result_status",
        "review_status",
        "decision",
        "decision_reason",
        "canonical_config",
        "safe_to_move",
        "safe_to_edit",
        "curation_evidence",
        "orchestration_provenance",
        "protected_fields",
    }
)
BASE_INVENTORY_COLUMNS = [
    column for column in INVENTORY_COLUMNS if column not in CURATION_INVENTORY_COLUMNS
]
DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "benchmark_results",
        "data",
        "__pycache__",
        ".pytest_cache",
        "cache",
        "caches",
        "tmp",
        "temp",
    }
)
DEFAULT_CONFIG_PREFIXES = frozenset(
    {"configs", "experiments", "benchmark", "bench", "model_zoo"}
)
LOCAL_ABSOLUTE_RE = re.compile(
    r"^(?:[A-Za-z]:[\\/]|/(?:home|mnt|content)(?:/|$))"
)
URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
SEED_KEYS = frozenset(
    {"seed", "seeds", "random_state", "split_seed", "model_seed", "model_seeds"}
)
NON_PROTOCOL_KEYS = frozenset(
    {
        "output_dir",
        "results_dir",
        "report_path",
        "summary_path",
        "figures_dir",
        "logging",
        "verbose",
        "notes",
        "resume",
        "resume_enabled",
        "runtime_estimates_seconds",
        "search_root",
        "search_roots",
        "run_directory",
        "canonical_reference_predictions",
        "seed_42_source_run",
        "seed_42_source_root",
        "categorical_references",
        "ordinal_seed42_references",
        "categorical_search_roots",
        "baseline_validation_root",
    }
)
LEGACY_CONFIG_EVIDENCE = {
    "configs/groupkfold_torch_lstm_gapaware_label_q5.yaml": (
        "Tracked statistical_analysis.yaml identifies this completed "
        "length-10 representation as legacy; retain as a baseline."
    ),
    "configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml": (
        "Tracked statistical_analysis.yaml identifies this completed "
        "length-10 representation as legacy; retain as a baseline."
    ),
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    yaml_key: str = ""
    value: Any = None

    def display(self) -> str:
        suffix = f" [{self.yaml_key}]" if self.yaml_key else ""
        return f"{self.code}{suffix}: {self.message}"


@dataclass
class ConfigRecord:
    path: str
    document: Any = None
    referenced_document: Any = None
    parse_error: str = ""
    loader_type: str = "unknown"
    expected_cli_argument: str = ""
    cli_loadable: bool = False
    schema_valid: bool = False
    load_error: str = ""
    base_path: str = ""
    base_exists: bool = True
    inheritance_depth: int = 0
    exact_hash: str = ""
    resolved_hash: str = ""
    protocol_hash: str = ""
    seedless_protocol_hash: str = ""
    role: str = "unknown"
    status: str = "unclassified"
    automatic_role: str = "unknown"
    automatic_status: str = "unclassified"
    topic: str = "other"
    registry_ids: list[str] = field(default_factory=list)
    registry_statuses: list[str] = field(default_factory=list)
    report_links: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    duplicate_groups: list[str] = field(default_factory=list)
    extracted: dict[str, Any] = field(default_factory=dict)
    is_legacy: bool = False
    superseded_by: str = ""
    deprecation_reason: str = ""
    curation: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


@dataclass
class AuditResult:
    records: list[ConfigRecord]
    duplicate_groups: list[dict[str, Any]]
    registry_consistency: list[dict[str, str]]
    registry_missing_configs: list[dict[str, str]]
    inheritance_cycles: list[list[str]]
    missing_bases: list[dict[str, str]]
    loader_notes: list[str]
    scanned_directories: list[str]
    structural_errors: list[str] = field(default_factory=list)
    curation: dict[str, Any] = field(default_factory=dict)
    curation_warnings: list[str] = field(default_factory=list)


class CurationValidationError(ValueError):
    """Raised when the manual reporting layer violates its declared schema."""


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def stable_json(value: Any) -> str:
    return json.dumps(
        _stable_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(
        fnmatch.fnmatch(path, pattern) or candidate.match(pattern)
        for pattern in patterns
    )


def _is_default_config_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if path == "configs.yaml":
        return True
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        return False
    return bool(candidate.parts and candidate.parts[0] in DEFAULT_CONFIG_PREFIXES)


def discover_config_paths(
    root: Path,
    *,
    includes: Sequence[str] | None = None,
    excludes: Sequence[str] | None = None,
) -> list[str]:
    """Return deterministic relative paths for source experiment configs."""
    root = root.resolve()
    candidates: list[str]
    try:
        git_root = Path(
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        if git_root != root:
            raise subprocess.CalledProcessError(1, "git root differs")
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--", "*.yaml", "*.yml"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        candidates = output.splitlines()
        # Include new source configs from the current worktree as well as
        # tracked files. This keeps pre-commit audits useful without changing
        # the tracked-only requirement for curation evidence.
        candidates.extend(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
    except (OSError, subprocess.CalledProcessError):
        candidates = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        ]

    selected: list[str] = []
    for raw_path in candidates:
        path = PurePosixPath(raw_path).as_posix()
        if any(part in DEFAULT_EXCLUDED_PARTS for part in PurePosixPath(path).parts):
            continue
        if includes:
            if not _matches_any(path, includes):
                continue
        elif not _is_default_config_path(path):
            continue
        if excludes and _matches_any(path, excludes):
            continue
        selected.append(path)
    return sorted(set(selected))


def _walk_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_strings(value[key], child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value


def is_local_absolute_path(value: str) -> bool:
    return not URL_RE.match(value.strip()) and bool(
        LOCAL_ABSOLUTE_RE.match(value.strip())
    )


def find_absolute_paths(document: Any) -> list[dict[str, str]]:
    return [
        {"yaml_key": key, "value": value}
        for key, value in _walk_strings(document)
        if is_local_absolute_path(value)
    ]


def _nested(document: Any, path: str, default: Any = None) -> Any:
    current = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _first(document: Any, paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        value = _nested(document, path)
        if value is not None:
            return value
    return default


def _first_dataset(document: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    datasets = document.get("datasets")
    if isinstance(datasets, Mapping) and datasets:
        name = next(iter(datasets))
        config = datasets[name]
        return str(name), config if isinstance(config, Mapping) else {}
    dataset = document.get("dataset")
    if isinstance(dataset, Mapping):
        return str(dataset.get("name", "")), dataset
    return "", {}


def _model_entries(document: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    models = document.get("models")
    if isinstance(models, Mapping):
        return [
            (str(name), value if isinstance(value, Mapping) else {})
            for name, value in models.items()
        ]
    model = document.get("model")
    if isinstance(model, Mapping):
        name = str(model.get("name", model.get("type", "")))
        return [(name, model)]
    if isinstance(model, str):
        return [(model, {"name": model})]
    return []


def _task_values(document: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    tasks: list[str] = []
    targets: list[str] = []
    raw_tasks = document.get("tasks")
    if isinstance(raw_tasks, list):
        tasks.extend(str(value) for value in raw_tasks)
    elif isinstance(raw_tasks, Mapping):
        for value in raw_tasks.values():
            if isinstance(value, Mapping):
                task = value.get("benchmark_task", value.get("task"))
                if task:
                    tasks.append(str(task))
                target = value.get("target")
                if target:
                    targets.append(str(target))
    task = document.get("task")
    if isinstance(task, Mapping):
        value = task.get("benchmark_task", task.get("name"))
        if value:
            tasks.append(str(value))
        target = task.get("target")
        if target:
            targets.append(str(target))
    elif isinstance(task, str):
        tasks.append(task)
    _, dataset = _first_dataset(document)
    dataset_task = dataset.get("task")
    if dataset_task:
        tasks.append(str(dataset_task))
    for key in ("target_col", "target"):
        if dataset.get(key):
            targets.append(str(dataset[key]))
    target_cols = dataset.get("target_cols")
    if isinstance(target_cols, list):
        targets.extend(str(value) for value in target_cols)
    root_targets = document.get("targets")
    if isinstance(root_targets, list):
        targets.extend(str(value) for value in root_targets)
    return list(dict.fromkeys(tasks)), list(dict.fromkeys(targets))


def classify_loader(path: str, document: Any) -> tuple[str, str]:
    if not isinstance(document, Mapping):
        return "unknown", ""
    experiment_type = str(_nested(document, "experiment.type", ""))
    analysis_type = str(_nested(document, "analysis.type", ""))
    keys = set(document)
    if {"datasets", "models", "tasks"}.issubset(keys):
        return "benchmark_config", "--config"
    if keys == {"raw_preprocessing"}:
        return "raw_preprocessing_fragment", ""
    if {"study", "base_config", "search", "search_space"}.issubset(keys):
        return "automl_study", "--automl-study"
    if "search_space" in keys and {"cache", "preprocessing", "model"}.issubset(keys):
        return "preprocessing_ablation", "--experiment-matrix"
    if experiment_type in {
        "user_calibration",
        "user_calibration_multiseed",
        "pm_regression_personalization",
        "pm_regression_personalization_multiseed",
    }:
        return experiment_type, "--calibration-experiment"
    if "cross_source" in experiment_type or {"matrix", "in_domain_references"}.issubset(keys):
        return "cross_source_experiment", "--cross-source-experiment"
    if "feature_groups" in keys and "models" in keys and "tasks" in keys:
        return "feature_group_experiment", "--feature-group-experiment"
    if experiment_type in {
        "ordinal_transformer_smoke",
        "ordinal_transformer_full",
        "ordinal_transformer_multiseed",
        "auxiliary_corn_transformer_smoke",
        "auxiliary_corn_lambda_selection_setup",
        "auxiliary_corn_nested_lambda",
        "auxiliary_corn_nested_lambda_finalize",
    }:
        return experiment_type, "--ordinal-transformer-experiment"
    if analysis_type in {
        "completed_run_subject_statistics",
        "ordinal_transformer_multiseed_statistics",
        "auxiliary_corn_policy_statistics",
    }:
        return analysis_type, "--ordinal-transformer-analysis"
    if "run_rules" in keys and "comparisons" in keys:
        return "statistical_analysis", "--statistical-analysis"
    if "audit" in keys:
        audit_name = str(_nested(document, "audit.name", ""))
        if "temporal" in audit_name or "blocked_time" in keys:
            return "temporal_target_audit", "--temporal-target-audit"
        return "label_target_audit", "--label-target-audit"
    if "diagnostic_baselines" in keys and "analysis" in keys:
        return "label_definition_sensitivity", "--label-definition-sensitivity"
    return "unknown", ""


LOADER_FUNCTIONS: dict[str, tuple[str, str]] = {
    "automl_study": (
        "bench.automl.scientific.study_runner",
        "load_automl_study_spec",
    ),
    "preprocessing_ablation": (
        "bench.experiments.preprocessing_ablation",
        "load_experiment_spec",
    ),
    "feature_group_experiment": (
        "bench.experiments.feature_group_ablation",
        "load_feature_group_spec",
    ),
    "ordinal_transformer_smoke": (
        "bench.experiments.ordinal_transformer",
        "load_ordinal_transformer_spec",
    ),
    "ordinal_transformer_full": (
        "bench.experiments.ordinal_transformer_full",
        "load_ordinal_transformer_full_spec",
    ),
    "ordinal_transformer_multiseed": (
        "bench.experiments.ordinal_transformer_multiseed",
        "load_ordinal_transformer_multiseed_spec",
    ),
    "auxiliary_corn_transformer_smoke": (
        "bench.experiments.auxiliary_corn_transformer",
        "load_auxiliary_corn_smoke_spec",
    ),
    "auxiliary_corn_lambda_selection_setup": (
        "bench.experiments.auxiliary_corn_lambda_selection",
        "load_auxiliary_corn_lambda_setup_spec",
    ),
    "auxiliary_corn_nested_lambda": (
        "bench.experiments.auxiliary_corn_nested_lambda",
        "load_auxiliary_corn_nested_spec",
    ),
    "auxiliary_corn_nested_lambda_finalize": (
        "bench.experiments.auxiliary_corn_nested_lambda_finalize",
        "load_auxiliary_corn_finalize_spec",
    ),
    "user_calibration_multiseed": (
        "bench.experiments.user_calibration_multiseed",
        "load_multiseed_calibration_spec",
    ),
    "pm_regression_personalization_multiseed": (
        "bench.experiments.pm_regression_personalization_multiseed",
        "load_pm_multiseed_spec",
    ),
    "label_target_audit": (
        "bench.analysis.label_target_audit",
        "load_label_target_audit_spec",
    ),
    "temporal_target_audit": (
        "bench.analysis.temporal_target_structure",
        "load_temporal_audit_spec",
    ),
    "label_definition_sensitivity": (
        "bench.analysis.label_definition_sensitivity",
        "load_label_sensitivity_spec",
    ),
    "statistical_analysis": (
        "bench.analysis.report_builder",
        "load_statistical_analysis_spec",
    ),
}


REQUIRED_TOP_LEVEL: dict[str, set[str]] = {
    "benchmark_config": {"datasets", "models", "tasks"},
    "raw_preprocessing_fragment": {"raw_preprocessing"},
    "automl_study": {"study", "base_config", "evaluation", "search", "search_space"},
    "preprocessing_ablation": {
        "experiment",
        "cache",
        "dataset",
        "preprocessing",
        "model",
        "validation",
        "evaluation",
        "search_space",
    },
    "user_calibration": {"experiment", "base_run", "calibration"},
    "user_calibration_multiseed": {"experiment", "base_template", "calibration"},
    "pm_regression_personalization": {
        "experiment",
        "base_run",
        "calibration",
        "targets",
    },
    "pm_regression_personalization_multiseed": {
        "experiment",
        "base_template",
        "calibration",
        "targets",
    },
    "cross_source_experiment": {
        "experiment",
        "dataset",
        "models",
        "matrix",
        "evaluation",
    },
    "feature_group_experiment": {
        "experiment",
        "dataset",
        "feature_groups",
        "tasks",
        "models",
        "evaluation",
    },
    "ordinal_transformer_smoke": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
    },
    "ordinal_transformer_full": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
    },
    "ordinal_transformer_multiseed": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
        "seeds",
    },
    "auxiliary_corn_transformer_smoke": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
    },
    "auxiliary_corn_lambda_selection_setup": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
        "selection",
    },
    "auxiliary_corn_nested_lambda": {
        "experiment",
        "dataset",
        "task",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
        "selection",
    },
    "auxiliary_corn_nested_lambda_finalize": {
        "experiment",
        "source",
        "fallback",
        "audit",
        "protocol",
    },
    "statistical_analysis": {"analysis", "run_rules", "comparisons"},
    "label_target_audit": {"audit"},
    "temporal_target_audit": {"audit", "blocked_time", "diagnostic_baselines"},
    "label_definition_sensitivity": {"analysis", "diagnostic_baselines"},
}


KNOWN_OPTIONAL_TOP_LEVEL: dict[str, set[str]] = {
    "benchmark_config": {
        "output_dir",
        "run_loso",
        "run_within_subject",
        "task_config",
        "evaluation",
        "validation",
        "sequence",
        "raw_preprocessing",
        "feature_scaling",
        "preprocessing",
        "result_status",
    },
    "automl_study": {"artifacts", "constraints"},
    "cross_source_experiment": {
        "runtime_estimates_seconds",
        "sequence",
        "validation_by_subject_mode",
        "in_domain_references",
    },
    "feature_group_experiment": {"analysis", "sequence", "validation"},
    "ordinal_transformer_smoke": {"feature_group", "head_types", "seeds"},
    "ordinal_transformer_full": {
        "feature_groups",
        "feature_definitions",
        "head_types",
        "seeds",
        "categorical_references",
    },
    "ordinal_transformer_multiseed": {
        "feature_groups",
        "feature_definitions",
        "head_types",
        "categorical_references",
        "ordinal_seed42_references",
        "categorical_search_roots",
    },
    "auxiliary_corn_transformer_smoke": {
        "feature_group",
        "auxiliary_weights",
        "seeds",
    },
    "auxiliary_corn_lambda_selection_setup": {
        "feature_groups",
        "feature_definitions",
        "auxiliary_weights",
        "categorical_baseline_index",
        "seeds",
    },
    "auxiliary_corn_nested_lambda": {
        "feature_groups",
        "feature_definitions",
        "auxiliary_weights",
        "categorical_baseline_index",
        "seeds",
        "baseline_validation_root",
    },
    "pm_regression_personalization": {"statistics"},
    "pm_regression_personalization_multiseed": {"statistics"},
}


def _validate_with_current_loader(
    root: Path,
    path: str,
    document: Mapping[str, Any],
    loader_type: str,
) -> tuple[bool, bool, str]:
    """Return (cli_loadable, schema_valid, error) without executing a run."""
    expected = REQUIRED_TOP_LEVEL.get(loader_type)
    if expected:
        missing = sorted(expected - set(document))
        if missing:
            return bool(classify_loader(path, document)[1]), False, (
                "missing required top-level fields: " + ", ".join(missing)
            )
    if loader_type == "raw_preprocessing_fragment":
        return False, isinstance(document.get("raw_preprocessing"), Mapping), ""
    if loader_type == "unknown":
        return False, False, "no current CLI loader identified"
    root_text = str(root)
    remove_root = root_text not in sys.path
    if remove_root:
        sys.path.insert(0, root_text)
    try:
        if loader_type == "benchmark_config":
            cli_module = importlib.import_module("cli")
            valid = bool(cli_module.validate_config(deepcopy(dict(document))))
            return True, valid, "" if valid else "cli.validate_config returned false"
        loader = LOADER_FUNCTIONS.get(loader_type)
        if loader:
            function = getattr(importlib.import_module(loader[0]), loader[1])
            function(root / path)
        return True, True, ""
    except (Exception, SystemExit) as exc:  # config errors belong in the report
        return bool(classify_loader(path, document)[1]), False, (
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        if remove_root and root_text in sys.path:
            sys.path.remove(root_text)


def _reference_path(document: Any) -> str:
    if not isinstance(document, Mapping):
        return ""
    for path in (
        "base_run.config_path",
        "base_template.config_path",
        "base_config.path",
    ):
        value = _nested(document, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = document.get("base_config")
    return value.strip() if isinstance(value, str) else ""


def _resolve_reference(root: Path, child_path: str, raw_reference: str) -> str:
    explicitly_root_relative = raw_reference.replace("\\", "/").startswith("./")
    cleaned = raw_reference.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    root_candidate = root / cleaned
    child_candidate = root / PurePosixPath(child_path).parent / cleaned
    if explicitly_root_relative or root_candidate.exists():
        candidate = root_candidate
    else:
        candidate = child_candidate
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return cleaned


def _deep_merge(base: Any, child: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(child, Mapping):
        result = {str(key): deepcopy(value) for key, value in base.items()}
        for key, value in child.items():
            if key == "base_config":
                continue
            result[str(key)] = (
                _deep_merge(result[key], value)
                if key in result
                else deepcopy(value)
            )
        return result
    return deepcopy(child)


def resolve_documents(
    root: Path,
    records: Mapping[str, ConfigRecord],
) -> tuple[dict[str, Any], list[list[str]], list[dict[str, str]]]:
    resolved: dict[str, Any] = {}
    cycles: list[list[str]] = []
    missing: list[dict[str, str]] = []

    def visit(path: str, stack: list[str]) -> tuple[Any, int]:
        record = records[path]
        if path in resolved:
            return resolved[path], record.inheritance_depth
        if path in stack:
            cycle = stack[stack.index(path) :] + [path]
            if cycle not in cycles:
                cycles.append(cycle)
            raise ValueError("inheritance cycle: " + " -> ".join(cycle))
        raw_reference = _reference_path(record.document)
        if not raw_reference:
            resolved[path] = record.document
            record.inheritance_depth = 0
            return record.document, 0
        base_path = _resolve_reference(root, path, raw_reference)
        record.base_path = base_path
        record.base_exists = base_path in records and (root / base_path).is_file()
        if not record.base_exists:
            entry = {"config_path": path, "base_config": base_path}
            if entry not in missing:
                missing.append(entry)
            raise FileNotFoundError(base_path)
        base_document, depth = visit(base_path, stack + [path])
        record.inheritance_depth = depth + 1
        if isinstance(record.document.get("base_config"), str):
            merged = _deep_merge(base_document, record.document)
        else:
            # Current project references a separately loaded benchmark config;
            # it does not deep-merge the specialized experiment document.
            merged = {
                "experiment_spec": record.document,
                "referenced_base": base_document,
            }
        resolved[path] = merged
        return merged, record.inheritance_depth

    for path in sorted(records):
        try:
            visit(path, [])
        except (ValueError, FileNotFoundError):
            resolved[path] = records[path].document
    return resolved, cycles, missing


def _without_keys(value: Any, *, remove_seeds: bool = False) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key)
            if normalized in NON_PROTOCOL_KEYS:
                continue
            if remove_seeds and normalized in SEED_KEYS:
                continue
            result[normalized] = _without_keys(child, remove_seeds=remove_seeds)
        return result
    if isinstance(value, list):
        return [_without_keys(item, remove_seeds=remove_seeds) for item in value]
    return value


def infer_role(
    path: str,
    document: Mapping[str, Any],
    referenced_paths: set[str],
    loader_type: str,
) -> str:
    if path == "configs.yaml":
        return "root"
    if path in referenced_paths or loader_type == "raw_preprocessing_fragment":
        return "base"
    if path in LEGACY_CONFIG_EVIDENCE:
        return "legacy"
    text = stable_json(document).lower()
    name = str(_first(document, ["experiment.name", "analysis.name", "audit.name"], ""))
    folds = _nested(document, "evaluation.folds")
    n_splits = _nested(document, "evaluation.n_splits")
    technical = bool(_nested(document, "protocol.technical_only", False))
    is_short = (
        isinstance(folds, list)
        and isinstance(n_splits, int)
        and len(folds) < n_splits
    )
    if "smoke" in name.lower() or "smoke" in PurePosixPath(path).stem.lower() or is_short or technical:
        return "smoke"
    if (
        loader_type == "preprocessing_ablation"
        or "feature_group" in loader_type
        or "ablation" in path.lower()
    ):
        return "ablation"
    if (
        loader_type.endswith("_analysis")
        or "statistics" in loader_type
        or loader_type in {
            "label_target_audit",
            "temporal_target_audit",
            "label_definition_sensitivity",
            "auxiliary_corn_nested_lambda_finalize",
        }
        or "diagnostic" in text
    ):
        return "diagnostic"
    if "legacy" in path.lower() or bool(_nested(document, "experiment.legacy", False)):
        return "legacy"
    if loader_type != "unknown":
        return "full"
    return "unknown"


def infer_topic(path: str, document: Mapping[str, Any], loader_type: str) -> str:
    models = [value.get("type", name) for name, value in _model_entries(document)]
    tasks, _ = _task_values(document)
    joined = " ".join([path, loader_type, *map(str, models), *tasks]).lower()
    if "calibration" in joined or "personalization" in joined:
        return "pm_regression_personalization" if "pm_regression" in joined else "classification_personalization"
    if "preprocessing" in joined:
        return "preprocessing_ablation"
    if "raw_eeg" in joined or "eegnet" in joined or "shallow" in joined:
        return "raw_eeg"
    if "pm_regression" in joined or "performance_metrics_regression" in joined:
        return "pm_regression"
    if "ordinal" in joined or "corn" in joined:
        return "ordinal_transformer"
    if any(value in joined for value in ("lstm", "bilstm", "transformer")):
        return "sequence_models"
    if "feature_group" in joined:
        return "feature_groups"
    if "cross_source" in joined:
        return "cross_source"
    if "audit" in joined or "statistical" in joined or "sensitivity" in joined:
        return "analysis"
    if "random_forest" in joined or "torch_mlp" in joined:
        return "classification"
    return "other"


def _extract_seed(document: Mapping[str, Any], model_entries: list[tuple[str, Mapping[str, Any]]]) -> tuple[Any, Any, list[Any]]:
    split_seed = _first(
        document,
        [
            "experiment.split_seed",
            "evaluation.random_state",
            "task_config.random_state",
            "validation.random_state",
            "analysis.random_state",
        ],
    )
    model_seed = _first(document, ["experiment.model_seed"])
    model_seeds = _first(document, ["experiment.model_seeds", "seeds"], [])
    if not isinstance(model_seeds, list):
        model_seeds = [model_seeds] if model_seeds is not None else []
    if model_seed is None:
        values = [
            _nested(config, "params.random_state")
            for _, config in model_entries
            if _nested(config, "params.random_state") is not None
        ]
        if len(set(values)) == 1 and values:
            model_seed = values[0]
    return split_seed, model_seed, model_seeds


def extract_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    dataset_name, dataset = _first_dataset(document)
    tasks, targets = _task_values(document)
    models = _model_entries(document)
    model_names = [name for name, _ in models if name]
    model_types = [
        str(config.get("type", config.get("name", name)))
        for name, config in models
    ]
    task_types = [
        str(config.get("task_type"))
        for _, config in models
        if config.get("task_type")
    ]
    split_seed, model_seed, model_seeds = _extract_seed(document, models)
    devices = [
        _nested(config, "params.device")
        for _, config in models
        if _nested(config, "params.device") is not None
    ]
    feature_set = dataset.get("feature_set")
    if feature_set is None:
        feature_set = _nested(document, "feature_group")
    if feature_set is None and "feature_groups" in document:
        groups = document["feature_groups"]
        feature_set = list(groups) if isinstance(groups, Mapping) else groups
    preprocessing: Any = None
    if "raw_preprocessing" in document:
        raw = document["raw_preprocessing"]
        if isinstance(raw, Mapping):
            enabled = []
            if _nested(raw, "bandpass.enabled", False):
                enabled.append("bandpass")
            if _nested(raw, "notch.enabled", False):
                enabled.append("notch")
            reref = _nested(raw, "rereference.mode")
            if reref and reref != "none":
                enabled.append(str(reref))
            preprocessing = "+".join(enabled) if enabled else "raw"
    elif "preprocessing" in document:
        preprocessing = document["preprocessing"]
    else:
        scaling = _first(document, ["feature_scaling.strategy"])
        if scaling is None:
            model_scaling = [
                _nested(config, "feature_scaling.strategy")
                for _, config in models
                if _nested(config, "feature_scaling.strategy") is not None
            ]
            scaling = model_scaling or None
        preprocessing = scaling
    evaluation = document.get("evaluation", {})
    validation = document.get("validation", {})
    output_dir = _first(
        document,
        ["output_dir", "experiment.output_dir", "analysis.output_dir", "audit.output_dir"],
        "",
    )
    resume = _first(document, ["experiment.resume", "resume"], None)
    return {
        "dataset": dataset_name,
        "task": tasks,
        "task_type": task_types,
        "target": targets[0] if len(targets) == 1 else "",
        "targets": targets,
        "model": model_names or model_types,
        "model_types": model_types,
        "feature_set": feature_set,
        "preprocessing": preprocessing,
        "outer_evaluation": evaluation,
        "inner_validation": validation,
        "n_folds": _first(document, ["evaluation.n_splits"], ""),
        "split_seed": split_seed,
        "model_seed": model_seed,
        "model_seeds": model_seeds,
        "device": list(dict.fromkeys(devices)),
        "output_dir": output_dir,
        "resume_enabled": resume,
    }


def _append_issue(record: ConfigRecord, issue: Issue) -> None:
    if (issue.severity, issue.code, issue.message, issue.yaml_key) not in {
        (item.severity, item.code, item.message, item.yaml_key)
        for item in record.issues
    }:
        record.issues.append(issue)


def validate_scientific_protocol(record: ConfigRecord) -> None:
    document = record.document
    if not isinstance(document, Mapping):
        return
    effective = (
        record.referenced_document
        if isinstance(record.referenced_document, Mapping)
        else document
    )
    datasets = effective.get("datasets", {})
    dataset_items = datasets.items() if isinstance(datasets, Mapping) else []
    for dataset_name, value in dataset_items:
        if not isinstance(value, Mapping):
            continue
        target = value.get("target_col")
        if target == "label_q5":
            if value.get("discretize", False) is not False:
                _append_issue(
                    record,
                    Issue("error", "label_q5_rediscretized", "label_q5 must set discretize: false", f"datasets.{dataset_name}.discretize"),
                )
            if value.get("n_classes", 5) != 5:
                _append_issue(
                    record,
                    Issue("error", "label_q5_class_count", "label_q5 must use five classes", f"datasets.{dataset_name}.n_classes"),
                )
        target_cols = value.get("target_cols")
        if target_cols is not None and list(target_cols) != PM_TARGETS:
            _append_issue(
                record,
                Issue("error", "pm_target_order", "PM regression targets must use the canonical seven-target order", f"datasets.{dataset_name}.target_cols"),
            )
        feature_columns = value.get("feature_columns", [])
        if isinstance(feature_columns, list) and any(
            str(column).startswith("target_") or str(column).startswith("PM.")
            for column in feature_columns
        ):
            _append_issue(
                record,
                Issue("error", "target_feature_leakage", "target/PM columns are explicitly included as features", f"datasets.{dataset_name}.feature_columns"),
            )
    root_targets = document.get("targets")
    if root_targets is not None and (
        any(str(value).startswith("target_") for value in root_targets)
        and list(root_targets) != PM_TARGETS
    ):
        _append_issue(
            record,
            Issue("error", "pm_target_order", "PM regression targets must use the canonical seven-target order", "targets"),
        )
    protocol = str(_nested(document, "evaluation.protocol", "")).lower()
    group_column = str(_nested(document, "evaluation.group_column", "")).lower()
    if record.status == "final":
        if "random" in protocol or (
            record.loader_type == "benchmark_config"
            and not protocol
            and not bool(document.get("run_loso", False))
        ):
            _append_issue(
                record,
                Issue("error", "final_random_window_split", "final result cannot use a random window split", "evaluation.protocol"),
            )
        if protocol and protocol not in {"cross_source_holdout"} and (
            "group" not in protocol or group_column not in {"subject_id", "subject"}
        ):
            _append_issue(
                record,
                Issue("error", "final_outer_not_subject_grouped", "final outer evaluation must be subject-grouped", "evaluation"),
            )
    model_types = set(record.extracted.get("model_types", []))
    dataset_name, dataset = _first_dataset(effective)
    raw_models = {"torch_eegnet", "torch_shallow_convnet"}
    if model_types & raw_models:
        if dataset_name and dataset_name != "emotiv_raw_eeg":
            _append_issue(
                record,
                Issue("error", "raw_model_wrong_dataset", "raw EEG CNN is configured with a non-raw dataset", "datasets"),
            )
    if "torch_mlp" in model_types and dataset_name == "emotiv_raw_eeg":
        _append_issue(
            record,
            Issue("error", "mlp_raw_tensor", "Torch MLP cannot consume raw EEG tensors without an explicit adapter", "datasets"),
        )
    if model_types & {"torch_lstm", "torch_bilstm", "torch_transformer"}:
        sequence = effective.get("sequence")
        if not isinstance(sequence, Mapping):
            _append_issue(
                record,
                Issue("error" if record.status == "final" else "warning", "missing_sequence_contract", "sequence model has no explicit sequence configuration", "sequence"),
            )
        validation = effective.get(
            "validation",
            document.get("validation_by_subject_mode"),
        )
        if not isinstance(validation, Mapping):
            _append_issue(
                record,
                Issue("error" if record.status == "final" else "warning", "missing_inner_validation", "Torch sequence config has no explicit group-aware inner validation", "validation"),
            )
    if any(value.startswith("torch_") for value in model_types):
        validation = effective.get(
            "validation",
            document.get("validation_by_subject_mode"),
        )
        if not isinstance(validation, Mapping):
            _append_issue(
                record,
                Issue("warning", "missing_inner_validation", "Torch config has no explicit inner validation section", "validation"),
            )
        if record.extracted.get("preprocessing") in (None, "", []):
            _append_issue(
                record,
                Issue("warning", "implicit_preprocessing", "Torch feature preprocessing is not explicit", "feature_scaling"),
            )
    if "personalization" in record.loader_type or "calibration" in record.loader_type:
        methods = _nested(document, "calibration.methods", [])
        if record.status == "final" and "zero_shot" not in methods:
            _append_issue(
                record,
                Issue("error", "missing_zero_shot", "final personalization comparison must include zero_shot", "calibration.methods"),
            )
        split_strategy = _first(
            document,
            ["calibration.defaults.split_strategy", "calibration.split_strategy"],
            "",
        )
        if split_strategy and "chronological" not in str(split_strategy):
            _append_issue(
                record,
                Issue("warning", "calibration_split_review", "calibration/final separation is not explicitly chronological", "calibration"),
            )
    if record.role in {"full", "smoke"} and "result_status" not in document:
        _append_issue(
            record,
            Issue("warning", "implicit_result_status", "result status is inferred from reports/registry rather than declared in YAML"),
        )
    if record.role not in {"root", "base"} and "config_version" not in document:
        _append_issue(
            record,
            Issue("warning", "missing_config_version", "configuration has no explicit config version"),
        )


def _validate_known_keys(record: ConfigRecord) -> None:
    if not isinstance(record.document, Mapping):
        return
    required = REQUIRED_TOP_LEVEL.get(record.loader_type, set())
    optional = KNOWN_OPTIONAL_TOP_LEVEL.get(record.loader_type, set())
    known = required | optional
    if not known:
        if record.loader_type == "unknown":
            for key in sorted(record.document):
                _append_issue(
                    record,
                    Issue("warning", "unknown_field", "field is not associated with a current loader contract", str(key)),
                )
        return
    for key in sorted(set(record.document) - known):
        _append_issue(
            record,
            Issue("warning", "ignored_or_unknown_field", "top-level field is not part of the audited loader contract", str(key)),
        )


def _load_experiment_registry(root: Path) -> tuple[list[dict[str, Any]], str]:
    path = root / "reports/summary/experiment_registry.yaml"
    if not path.is_file():
        return [], "reports/summary/experiment_registry.yaml is missing"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        experiments = document.get("experiments")
        if not isinstance(experiments, list):
            return [], "experiment_registry.yaml experiments must be a list"
        return [value for value in experiments if isinstance(value, Mapping)], ""
    except (OSError, yaml.YAMLError) as exc:
        return [], f"cannot process experiment registry: {exc}"


def _tracked_report_texts(root: Path) -> dict[str, str]:
    try:
        files = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "reports"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        files = []
    result: dict[str, str] = {}
    for relative in sorted(files):
        path = root / relative
        if path.suffix.lower() not in {".md", ".json", ".csv", ".txt"}:
            continue
        try:
            result[relative] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return result


def _registry_matches(
    registry_value: Any,
    config_values: Sequence[Any],
) -> bool:
    expected = str(registry_value).strip().lower()
    values = {str(value).strip().lower() for value in config_values}
    aliases = {
        "performance_metrics_7": set(PM_TARGETS),
        "raw_deduplicated": {"raw", "raw_deduplicated"},
        "none": {"", "none"},
    }
    return expected in values or bool(aliases.get(expected, set()) & values)


def check_registry_consistency(
    root: Path,
    records: Mapping[str, ConfigRecord],
    experiments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    mismatches: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for experiment in sorted(experiments, key=lambda value: str(value.get("experiment_id", ""))):
        experiment_id = str(experiment.get("experiment_id", ""))
        config_path = experiment.get("config_path")
        if not config_path:
            continue
        config_path = PurePosixPath(str(config_path)).as_posix()
        if is_local_absolute_path(config_path):
            mismatches.append(
                {
                    "experiment_id": experiment_id,
                    "config_path": config_path,
                    "field": "config_path",
                    "registry_value": config_path,
                    "config_value": "",
                    "severity": "error",
                    "recommended_action": "Use a repository-relative config path in the experiment registry.",
                }
            )
            continue
        record = records.get(config_path)
        if record is None:
            missing.append({"experiment_id": experiment_id, "config_path": config_path})
            continue
        record.registry_ids.append(experiment_id)
        record.registry_statuses.append(str(experiment.get("status", "unclassified")))
        checks = {
            "task": record.extracted.get("task", []),
            "model": [
                *record.extracted.get("model", []),
                *record.extracted.get("model_types", []),
            ],
            "target": [
                record.extracted.get("target", ""),
                *record.extracted.get("targets", []),
            ],
        }
        for field_name, config_values in checks.items():
            registry_value = experiment.get(field_name)
            if registry_value in (None, ""):
                continue
            if not _registry_matches(registry_value, config_values):
                mismatches.append(
                    {
                        "experiment_id": experiment_id,
                        "config_path": config_path,
                        "field": field_name,
                        "registry_value": stable_json(registry_value),
                        "config_value": stable_json(config_values),
                        "severity": "error",
                        "recommended_action": "Review the registry link; do not change the source config automatically.",
                    }
                )
        registry_seeds = experiment.get("seeds")
        config_seeds = [
            value
            for value in [
                record.extracted.get("model_seed"),
                record.extracted.get("split_seed"),
                *record.extracted.get("model_seeds", []),
            ]
            if value is not None
        ]
        if registry_seeds and not set(registry_seeds).issubset(set(config_seeds)):
            mismatches.append(
                {
                    "experiment_id": experiment_id,
                    "config_path": config_path,
                    "field": "seeds",
                    "registry_value": stable_json(registry_seeds),
                    "config_value": stable_json(sorted(set(config_seeds))),
                    "severity": "warning",
                    "recommended_action": "Document companion seed configs or the external multi-seed orchestration.",
                }
            )
        status = str(experiment.get("status", "unclassified"))
        if status == "final" and record.role == "smoke":
            mismatches.append(
                {
                    "experiment_id": experiment_id,
                    "config_path": config_path,
                    "field": "status/role",
                    "registry_value": "final",
                    "config_value": "smoke",
                    "severity": "error",
                    "recommended_action": "Link the final experiment to its full configuration.",
                }
            )
    return mismatches, missing


def _report_links(path: str, report_texts: Mapping[str, str]) -> list[str]:
    name = PurePosixPath(path).name
    return sorted(
        report
        for report, text in report_texts.items()
        if path in text or name in text
    )


def _declared_report_links(root: Path, document: Any) -> list[str]:
    links: set[str] = set()
    for _, value in _walk_strings(document):
        normalized = value.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized.startswith("reports/"):
            continue
        candidate = root / normalized
        if candidate.is_file():
            links.add(normalized)
    return sorted(links)


def _duplicate_groups(records: Sequence[ConfigRecord]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    def add_groups(kind: str, attribute: str, recommendation: str) -> None:
        by_hash: dict[str, list[ConfigRecord]] = defaultdict(list)
        for record in records:
            digest = getattr(record, attribute)
            if digest:
                by_hash[digest].append(record)
        for digest, members in sorted(by_hash.items()):
            if len(members) < 2:
                continue
            groups.append(
                {
                    "kind": kind,
                    "hash": digest,
                    "configs": sorted(record.path for record in members),
                    "recommendation": recommendation,
                }
            )

    add_groups("exact_duplicate", "exact_hash", "review manually")
    add_groups("resolved_duplicate", "resolved_hash", "replace with base config")
    add_groups(
        "same_protocol_different_output",
        "protocol_hash",
        "keep separate when outputs represent distinct completed runs",
    )
    add_groups(
        "same_protocol_different_seed",
        "seedless_protocol_hash",
        "replace with a seed-aware base config on the consolidation stage",
    )
    groups = [
        group
        for group in groups
        if group["kind"] != "same_protocol_different_seed"
        or len(
            {
                record.protocol_hash
                for record in records
                if record.path in group["configs"]
            }
        )
        > 1
    ]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for group in groups:
        key = (group["kind"], tuple(group["configs"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(group)
    for index, group in enumerate(unique, start=1):
        group["group_id"] = f"D{index:03d}"
        for record in records:
            if record.path in group["configs"]:
                record.duplicate_groups.append(group["group_id"])
    return unique


def _add_type_conflict_warnings(records: Sequence[ConfigRecord]) -> None:
    by_loader_and_key: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    def walk(value: Any, prefix: str, destination: dict[str, set[str]], path: str) -> None:
        if isinstance(value, Mapping):
            destination[prefix or "<root>"].add("mapping")
            for key, child in value.items():
                child_key = f"{prefix}.{key}" if prefix else str(key)
                walk(child, child_key, destination, path)
        elif isinstance(value, list):
            destination[prefix].add("list")
        else:
            destination[prefix].add(type(value).__name__)

    loader_maps: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        if isinstance(record.document, Mapping):
            walk(record.document, "", loader_maps[record.loader_type], record.path)
    for loader_type, values in loader_maps.items():
        conflicts = {key for key, types in values.items() if len(types) > 1}
        if not conflicts:
            continue
        for record in records:
            if record.loader_type != loader_type:
                continue
            for key in sorted(conflicts):
                _append_issue(
                    record,
                    Issue(
                        "warning",
                        "conflicting_key_types",
                        f"key has multiple types within loader family {loader_type}",
                        key,
                    ),
                )


def _tracked_path(root: Path, value: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", value],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _validate_evidence(root: Path, evidence: Any, *, location: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(evidence, list) or not evidence:
        return [f"{location} must contain at least one evidence item"]
    for index, item in enumerate(evidence):
        item_location = f"{location}[{index}]"
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{item_location} must be a non-empty string")
            continue
        value = item.strip()
        if is_local_absolute_path(value):
            errors.append(f"{item_location} must not be an absolute local path")
            continue
        if value.startswith("commit:"):
            commit = value.split(":", 1)[1].strip()
            if not commit:
                errors.append(f"{item_location} has an empty commit reference")
                continue
            try:
                subprocess.check_output(
                    ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                errors.append(f"{item_location} references unknown commit {commit}")
            continue
        normalized = value.replace("\\", "/")
        if not (root / normalized).is_file():
            errors.append(f"{item_location} references missing file {normalized}")
        elif not _tracked_path(root, normalized):
            errors.append(f"{item_location} must reference a tracked file")
    return errors


def _curation_cycle(
    links: Mapping[str, str],
) -> list[str] | None:
    visited: set[str] = set()
    for origin in sorted(links):
        path: list[str] = []
        current = origin
        while current in links:
            if current in path:
                return path[path.index(current) :] + [current]
            if current in visited:
                break
            path.append(current)
            current = links[current]
        visited.update(path)
    return None


def load_and_validate_curation(
    root: Path,
    path: Path,
    records: Mapping[str, ConfigRecord],
) -> tuple[dict[str, Any], list[str]]:
    """Load and strictly validate the persistent manual decision layer."""
    source = path if path.is_absolute() else root / path
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CurationValidationError(f"cannot load curation YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CurationValidationError("curation root must be a mapping")
    if document.get("schema_version") != 1:
        raise CurationValidationError("curation schema_version must equal 1")
    absolute = find_absolute_paths(document)
    if absolute:
        first = absolute[0]
        raise CurationValidationError(
            f"curation contains an absolute local path at {first['yaml_key']}"
        )
    configs = document.get("configs")
    families = document.get("families")
    if not isinstance(configs, list):
        raise CurationValidationError("curation configs must be a list")
    if not isinstance(families, Mapping):
        raise CurationValidationError("curation families must be a mapping")

    errors: list[str] = []
    warnings: list[str] = []
    by_path: dict[str, Mapping[str, Any]] = {}
    superseded_links: dict[str, str] = {}
    required = {
        "config_path",
        "review_status",
        "decision",
        "decision_reason",
        "canonical_config",
        "safe_to_move",
        "safe_to_edit",
        "evidence",
    }
    referenced_as_base = {
        record.base_path for record in records.values() if record.base_path
    }
    for index, item in enumerate(configs):
        location = f"configs[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{location} must be a mapping")
            continue
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"{location} missing fields: {', '.join(missing)}")
            continue
        config_path = PurePosixPath(str(item["config_path"])).as_posix()
        if config_path in by_path:
            errors.append(f"duplicate curation decision for {config_path}")
            continue
        by_path[config_path] = item
        record = records.get(config_path)
        if record is None:
            errors.append(f"unknown config_path: {config_path}")
            continue
        review_status = str(item["review_status"])
        decision = str(item["decision"])
        if review_status not in VALID_REVIEW_STATUSES:
            errors.append(
                f"{config_path}: unknown review_status {review_status!r}"
            )
        if decision not in VALID_DECISIONS:
            errors.append(f"{config_path}: unknown decision {decision!r}")
        role = item.get("config_role")
        status = item.get("result_status")
        canonical_status = item.get("canonical_status")
        provenance_status = item.get("provenance_status")
        if role is not None and role not in VALID_ROLES:
            errors.append(f"{config_path}: unknown config_role {role!r}")
        if status is not None and status not in VALID_STATUSES:
            errors.append(f"{config_path}: unknown result_status {status!r}")
        if (
            canonical_status is not None
            and canonical_status not in VALID_CANONICAL_STATUSES
        ):
            errors.append(
                f"{config_path}: unknown canonical_status {canonical_status!r}"
            )
        if (
            provenance_status is not None
            and provenance_status not in VALID_PROVENANCE_STATUSES
        ):
            errors.append(
                f"{config_path}: unknown provenance_status {provenance_status!r}"
            )
        reason = str(item.get("decision_reason", "")).strip()
        evidence = item.get("evidence")
        if review_status == "reviewed":
            if not reason:
                errors.append(f"{config_path}: reviewed requires decision_reason")
            errors.extend(
                _validate_evidence(
                    root,
                    evidence,
                    location=f"{config_path}.evidence",
                )
            )
        elif isinstance(evidence, list) and evidence:
            errors.extend(
                _validate_evidence(
                    root,
                    evidence,
                    location=f"{config_path}.evidence",
                )
            )
        if decision == "review_later" or review_status == "needs_evidence":
            warnings.append(f"{config_path}: manual evidence remains incomplete")
        if decision == "keep" and not record.registry_ids:
            warnings.append(
                f"{config_path}: active config has no experiment registry link"
            )
        if decision == "keep_as_legacy" and not any(
            isinstance(value, str) and value.startswith("commit:")
            for value in (evidence if isinstance(evidence, list) else [])
        ):
            warnings.append(f"{config_path}: legacy config has no Git evidence")
        superseded_by = item.get("superseded_by")
        if decision == "superseded" and not superseded_by:
            errors.append(f"{config_path}: superseded requires superseded_by")
        if decision == "superseded" and not str(
            item.get("supersession_reason", "")
        ).strip():
            errors.append(f"{config_path}: superseded requires supersession_reason")
        if superseded_by:
            target = PurePosixPath(str(superseded_by)).as_posix()
            if target not in records:
                errors.append(f"{config_path}: unknown superseded_by {target}")
            else:
                superseded_links[config_path] = target
        canonical = PurePosixPath(str(item["canonical_config"])).as_posix()
        if canonical not in records:
            errors.append(f"{config_path}: unknown canonical_config {canonical}")
        elif canonical != config_path:
            target_record = records[canonical]
            if (
                target_record.loader_type != record.loader_type
                and not str(item.get("canonical_link_reason", "")).strip()
            ):
                errors.append(
                    f"{config_path}: canonical_config uses incompatible loader_type "
                    "without canonical_link_reason"
                )
        safe_to_move = item.get("safe_to_move")
        safe_to_edit = item.get("safe_to_edit")
        if not isinstance(safe_to_move, bool) or not isinstance(safe_to_edit, bool):
            errors.append(
                f"{config_path}: safe_to_move and safe_to_edit must be booleans"
            )
        if safe_to_move and (
            record.registry_ids
            or record.report_links
            or record.base_path
            or config_path in referenced_as_base
        ):
            errors.append(
                f"{config_path}: safe_to_move=true conflicts with detected references"
            )
        effective_role = str(role or record.role)
        effective_status = str(status or record.status)
        linked_report = item.get("linked_report")
        linked_runtime = item.get("linked_runtime")
        if linked_report is not None:
            report_path = str(linked_report).replace("\\", "/")
            if is_local_absolute_path(report_path):
                errors.append(f"{config_path}: linked_report must be repository-relative")
            elif not (root / report_path).is_file():
                errors.append(f"{config_path}: linked_report does not exist: {report_path}")
            elif not _tracked_path(root, report_path):
                errors.append(f"{config_path}: linked_report must be tracked")
        if linked_runtime is not None:
            runtime_path = str(linked_runtime).replace("\\", "/")
            if is_local_absolute_path(runtime_path):
                errors.append(f"{config_path}: linked_runtime must be repository-relative")
            elif not (root / runtime_path).exists():
                errors.append(
                    f"{config_path}: linked_runtime does not exist: {runtime_path}"
                )
        if canonical_status == "completed" and not (
            linked_report or linked_runtime
        ):
            errors.append(
                f"{config_path}: completed canonical requires linked_report "
                "or linked_runtime"
            )
        if (
            canonical_status == "completed"
            and record.loader_type == "automl_study"
            and not linked_runtime
        ):
            errors.append(
                f"{config_path}: completed AutoML canonical requires linked_runtime"
            )
        if canonical_status == "planned" and (
            linked_runtime or effective_status in {"final", "baseline"}
        ):
            errors.append(
                f"{config_path}: planned canonical cannot represent a completed result"
            )
        if record.loader_type == "raw_preprocessing_fragment" and (
            effective_role == "full" or effective_status == "final"
        ):
            errors.append(
                f"{config_path}: raw preprocessing fragment cannot be a full/final run"
            )
        if record.role == "base" and effective_status == "final":
            errors.append(f"{config_path}: base config cannot be a final run")
        if "targets" in item:
            automatic_targets = record.extracted.get("targets", [])
            if automatic_targets and list(item["targets"]) != list(automatic_targets):
                errors.append(
                    f"{config_path}: curation cannot change automatic target order"
                )

    cycle = _curation_cycle(superseded_links)
    if cycle:
        errors.append("superseded_by cycle: " + " -> ".join(cycle))

    for family_name, family in sorted(families.items()):
        location = f"families.{family_name}"
        if not isinstance(family, Mapping):
            errors.append(f"{location} must be a mapping")
            continue
        reason = str(family.get("decision_reason", "")).strip()
        if not reason:
            errors.append(f"{location} requires decision_reason")
        canonical = family.get("canonical_config")
        if canonical is not None:
            canonical = PurePosixPath(str(canonical)).as_posix()
            if canonical not in records:
                errors.append(f"{location} has unknown canonical_config {canonical}")
            elif not (
                records[canonical].report_links
                or by_path.get(canonical, {}).get("linked_report")
                or by_path.get(canonical, {}).get("linked_runtime")
            ):
                warnings.append(
                    f"{location}: canonical config has no exact tracked report link"
                )
        elif not str(family.get("no_canonical_reason", "")).strip():
            errors.append(
                f"{location} requires canonical_config or no_canonical_reason"
            )
        smoke = family.get("canonical_smoke_config")
        if smoke is None:
            warnings.append(f"{location}: no canonical smoke config")
        elif PurePosixPath(str(smoke)).as_posix() not in records:
            errors.append(f"{location} has unknown canonical_smoke_config {smoke}")
        errors.extend(
            _validate_evidence(
                root,
                family.get("evidence"),
                location=f"{location}.evidence",
            )
        )

    duplicate_decisions = document.get("duplicate_groups", [])
    if not isinstance(duplicate_decisions, list):
        errors.append("duplicate_groups must be a list")
    else:
        for index, group in enumerate(duplicate_decisions):
            if not isinstance(group, Mapping):
                errors.append(f"duplicate_groups[{index}] must be a mapping")
                continue
            members = group.get("configs")
            if not isinstance(members, list) or len(members) < 2:
                errors.append(f"duplicate_groups[{index}].configs must list members")
                continue
            for member in members:
                if PurePosixPath(str(member)).as_posix() not in records:
                    errors.append(
                        f"duplicate_groups[{index}] references unknown config {member}"
                    )

    provenance = document.get("seed_provenance", [])
    if not isinstance(provenance, list):
        errors.append("seed_provenance must be a list")
    else:
        for index, item in enumerate(provenance):
            if not isinstance(item, Mapping):
                errors.append(f"seed_provenance[{index}] must be a mapping")
                continue
            status = item.get("provenance_status")
            if status not in VALID_PROVENANCE_STATUSES:
                errors.append(
                    f"seed_provenance[{index}] has unknown provenance_status {status!r}"
                )
            if status == "partially_documented":
                warnings.append(
                    f"seed_provenance[{index}]: provenance is partially documented"
                )
            errors.extend(
                _validate_evidence(
                    root,
                    item.get("report_evidence"),
                    location=f"seed_provenance[{index}].report_evidence",
                )
            )

    if errors:
        raise CurationValidationError("\n".join(sorted(set(errors))))
    normalized = deepcopy(dict(document))
    normalized["configs"] = [
        deepcopy(dict(by_path[path])) for path in sorted(by_path)
    ]
    normalized["families"] = {
        key: deepcopy(dict(families[key])) for key in sorted(families)
    }
    return normalized, sorted(set(warnings))


def apply_curation(
    records: Mapping[str, ConfigRecord],
    curation: Mapping[str, Any],
) -> None:
    for item in curation.get("configs", []):
        path = PurePosixPath(str(item["config_path"])).as_posix()
        record = records[path]
        record.curation = deepcopy(dict(item))
        if item.get("config_role"):
            record.role = str(item["config_role"])
        if item.get("result_status"):
            record.status = str(item["result_status"])
        if item.get("superseded_by"):
            record.superseded_by = str(item["superseded_by"])


def audit_repository(
    root: Path,
    *,
    includes: Sequence[str] | None = None,
    excludes: Sequence[str] | None = None,
    curation_path: Path | None = None,
) -> AuditResult:
    root = root.resolve()
    paths = discover_config_paths(root, includes=includes, excludes=excludes)
    records: dict[str, ConfigRecord] = {}
    for path in paths:
        record = ConfigRecord(path=path)
        try:
            record.document = yaml.safe_load((root / path).read_text(encoding="utf-8"))
            if record.document is None:
                record.document = {}
            if not isinstance(record.document, Mapping):
                record.parse_error = "config root must be a mapping"
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            record.document = {}
            record.parse_error = f"{type(exc).__name__}: {exc}"
        record.loader_type, record.expected_cli_argument = classify_loader(
            path, record.document
        )
        records[path] = record

    resolved, cycles, missing_bases = resolve_documents(root, records)
    referenced_paths = {
        record.base_path
        for record in records.values()
        if record.base_path
        and (
            "base_run" in record.document
            or "base_template" in record.document
            or isinstance(record.document.get("base_config"), str)
        )
    }
    registry, registry_error = _load_experiment_registry(root)
    report_texts = _tracked_report_texts(root)

    for record in records.values():
        if record.parse_error:
            _append_issue(
                record,
                Issue("error", "yaml_parse_error", record.parse_error),
            )
        if record.base_path and not record.base_exists:
            _append_issue(
                record,
                Issue("error", "missing_base_config", f"base config does not exist: {record.base_path}"),
            )
        for cycle in cycles:
            if record.path in cycle:
                _append_issue(
                    record,
                    Issue("error", "inheritance_cycle", " -> ".join(cycle)),
                )
        record.exact_hash = stable_hash(record.document) if not record.parse_error else ""
        record.resolved_hash = stable_hash(resolved[record.path]) if not record.parse_error else ""
        protocol = _without_keys(resolved[record.path])
        seedless = _without_keys(resolved[record.path], remove_seeds=True)
        record.protocol_hash = stable_hash(protocol) if not record.parse_error else ""
        record.seedless_protocol_hash = stable_hash(seedless) if not record.parse_error else ""
        record.role = infer_role(
            record.path,
            record.document,
            referenced_paths,
            record.loader_type,
        )
        record.topic = infer_topic(record.path, record.document, record.loader_type)
        record.is_legacy = record.role == "legacy"
        if record.path in LEGACY_CONFIG_EVIDENCE:
            record.deprecation_reason = LEGACY_CONFIG_EVIDENCE[record.path]
        record.extracted = extract_fields(record.document)
        if record.base_path in records:
            record.referenced_document = records[record.base_path].document
            base_fields = extract_fields(records[record.base_path].document)
            for key in (
                "dataset",
                "task",
                "task_type",
                "target",
                "targets",
                "model",
                "model_types",
                "feature_set",
                "preprocessing",
                "outer_evaluation",
                "inner_validation",
                "n_folds",
                "split_seed",
                "model_seed",
                "device",
            ):
                if record.extracted.get(key) in (None, "", [], {}):
                    record.extracted[key] = deepcopy(base_fields.get(key))
        record.report_links = sorted(
            set(_report_links(record.path, report_texts))
            | set(_declared_report_links(root, record.document))
        )
        cli_loadable, schema_valid, load_error = _validate_with_current_loader(
            root,
            record.path,
            record.document,
            record.loader_type,
        )
        record.cli_loadable = cli_loadable
        record.schema_valid = schema_valid and not record.parse_error
        record.load_error = load_error
        if load_error:
            severity = "warning" if record.loader_type == "unknown" else "error"
            _append_issue(
                record,
                Issue(severity, "loader_validation", load_error),
            )
        for absolute in find_absolute_paths(record.document):
            _append_issue(
                record,
                Issue(
                    "error",
                    "absolute_local_path",
                    f"tracked config contains local absolute path: {absolute['value']}",
                    absolute["yaml_key"],
                    absolute["value"],
                ),
            )

    consistency, registry_missing = check_registry_consistency(
        root, records, registry
    )
    for record in records.values():
        if record.registry_statuses:
            statuses = sorted(set(record.registry_statuses))
            record.status = statuses[0] if len(statuses) == 1 else "unclassified"
        elif record.role == "smoke":
            record.status = "smoke"
        elif record.role in {"diagnostic", "ablation", "root"}:
            record.status = "diagnostic"
        else:
            record.status = "unclassified"
        _validate_known_keys(record)
        validate_scientific_protocol(record)
    for mismatch in consistency:
        record = records.get(mismatch["config_path"])
        if record:
            _append_issue(
                record,
                Issue(
                    mismatch["severity"],
                    "experiment_registry_mismatch",
                    f"{mismatch['experiment_id']} {mismatch['field']}: registry={mismatch['registry_value']} config={mismatch['config_value']}",
                ),
            )
    _add_type_conflict_warnings(list(records.values()))
    duplicates = _duplicate_groups(list(records.values()))
    for record in records.values():
        record.automatic_role = record.role
        record.automatic_status = record.status
    curation: dict[str, Any] = {}
    curation_warnings: list[str] = []
    if curation_path is not None:
        curation, curation_warnings = load_and_validate_curation(
            root,
            curation_path,
            records,
        )
        apply_curation(records, curation)
        for record in records.values():
            validate_scientific_protocol(record)
    loader_notes = [
        "benchmark_config: cli.load_config + cli.validate_config; no base merge.",
        "preprocessing_ablation: load_experiment_spec; trial configs are resolved in memory.",
        "automl_study: load_automl_study_spec; base_config.path is loaded separately.",
        "calibration/personalization: specialized experiment classes load base_run/base_template separately.",
        "ordinal/CORN: dedicated load_*_spec functions selected through --ordinal-transformer-experiment.",
        "analysis-only specs: dedicated analysis loaders; no model training is needed for audit.",
    ]
    if registry_error:
        loader_notes.append(registry_error)
    scanned = sorted(
        {
            PurePosixPath(path).parts[0]
            for path in paths
            if PurePosixPath(path).parts
        }
    )
    return AuditResult(
        records=sorted(
            records.values(),
            key=lambda item: (item.topic, ROLE_ORDER[item.role], item.path),
        ),
        duplicate_groups=duplicates,
        registry_consistency=consistency,
        registry_missing_configs=registry_missing,
        inheritance_cycles=cycles,
        missing_bases=missing_bases,
        loader_notes=loader_notes,
        scanned_directories=scanned,
        structural_errors=[registry_error] if registry_error else [],
        curation=curation,
        curation_warnings=curation_warnings,
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, Mapping)) or value is None:
        return stable_json(value)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def record_to_row(record: ConfigRecord) -> dict[str, Any]:
    absolute = find_absolute_paths(record.document)
    errors = sorted(issue.display() for issue in record.errors)
    warnings = sorted(issue.display() for issue in record.warnings)
    values = record.extracted
    row = {
        "config_path": record.path,
        "file_name": PurePosixPath(record.path).name,
        "topic": record.topic,
        "config_role": record.role,
        "result_status": record.status,
        "automatic_config_role": record.automatic_role,
        "automatic_result_status": record.automatic_status,
        "review_status": record.curation.get("review_status", ""),
        "decision": record.curation.get("decision", ""),
        "decision_reason": record.curation.get("decision_reason", ""),
        "canonical_config": record.curation.get("canonical_config", ""),
        "canonical_status": record.curation.get("canonical_status", ""),
        "provenance_status": record.curation.get("provenance_status", ""),
        "linked_report": record.curation.get("linked_report", ""),
        "linked_runtime": record.curation.get("linked_runtime", ""),
        "supersession_reason": record.curation.get("supersession_reason", ""),
        "safe_to_move": record.curation.get("safe_to_move"),
        "safe_to_edit": record.curation.get("safe_to_edit"),
        "curation_evidence": record.curation.get("evidence", []),
        "orchestration_provenance": record.curation.get(
            "orchestration_provenance", {}
        ),
        "protected_fields": record.curation.get("protected_fields", []),
        "loader_type": record.loader_type,
        "base_config": record.base_path,
        "base_config_exists": record.base_exists,
        "inheritance_depth": record.inheritance_depth,
        "resolved_config_hash": record.resolved_hash,
        "used_by_experiment_registry": bool(record.registry_ids),
        "registry_experiment_ids": record.registry_ids,
        "cli_loadable": record.cli_loadable,
        "expected_cli_argument": record.expected_cli_argument,
        "schema_valid": record.schema_valid,
        "scientific_protocol_valid": not record.errors,
        "task": values.get("task", []),
        "task_type": values.get("task_type", []),
        "target": values.get("target", ""),
        "targets": values.get("targets", []),
        "model": values.get("model", []),
        "feature_set": values.get("feature_set"),
        "preprocessing": values.get("preprocessing"),
        "outer_evaluation": values.get("outer_evaluation", {}),
        "inner_validation": values.get("inner_validation", {}),
        "n_folds": values.get("n_folds", ""),
        "split_seed": values.get("split_seed"),
        "model_seed": values.get("model_seed"),
        "model_seeds": values.get("model_seeds", []),
        "device": values.get("device", []),
        "output_dir": values.get("output_dir", ""),
        "resume_enabled": values.get("resume_enabled"),
        "absolute_paths_found": absolute,
        "duplicate_group": sorted(record.duplicate_groups),
        "report_linked": bool(record.report_links),
        "is_legacy": record.is_legacy,
        "superseded_by": record.superseded_by,
        "deprecation_reason": record.deprecation_reason,
        "exact_hash": record.exact_hash,
        "scientific_protocol_hash": record.protocol_hash,
        "issues_count": len(record.issues),
        "errors": errors,
        "warnings": warnings,
        "notes": record.report_links,
    }
    return {column: _csv_value(row[column]) for column in INVENTORY_COLUMNS}


def render_inventory_csv(result: AuditResult) -> str:
    stream = io.StringIO(newline="")
    columns = INVENTORY_COLUMNS if result.curation else BASE_INVENTORY_COLUMNS
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in result.records:
        row = record_to_row(record)
        writer.writerow({column: row[column] for column in columns})
    return stream.getvalue()


def render_config_registry(result: AuditResult) -> str:
    configs: list[dict[str, Any]] = []
    for record in result.records:
        item = {
            "config_path": record.path,
            "topic": record.topic,
            "role": record.role,
            "status": record.status,
            "loader_type": record.loader_type,
            "expected_cli_argument": record.expected_cli_argument or None,
            "base_config": record.base_path or None,
            "inheritance_depth": record.inheritance_depth,
            "resolved_config_hash": record.resolved_hash or None,
            "used_by": record.registry_ids,
            "report_links": record.report_links,
            "is_legacy": record.is_legacy,
            "superseded_by": record.superseded_by or None,
            "deprecation_reason": record.deprecation_reason or None,
            "audit": {
                "cli_loadable": record.cli_loadable,
                "schema_valid": record.schema_valid,
                "scientific_protocol_valid": not record.errors,
                "errors": sorted(issue.display() for issue in record.errors),
                "warnings": sorted(issue.display() for issue in record.warnings),
            },
        }
        if result.curation:
            item["automatic_role"] = record.automatic_role
            item["automatic_status"] = record.automatic_status
            item["curation"] = deepcopy(record.curation)
        configs.append(item)
    document = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Generated read-only configuration map; not a CLI configuration source."
        ),
        "configs": configs,
        "duplicate_groups": result.duplicate_groups,
    }
    return yaml.safe_dump(
        document,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_Не обнаружено._\n"
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result) + "\n"


def render_markdown(result: AuditResult) -> str:
    roles = Counter(record.role for record in result.records)
    statuses = Counter(record.status for record in result.records)
    loaders = Counter(record.loader_type for record in result.records)
    error_records = [record for record in result.records if record.errors]
    warning_records = [record for record in result.records if record.warnings]
    exact_groups = [
        group for group in result.duplicate_groups if group["kind"] == "exact_duplicate"
    ]
    protocol_groups = [
        group
        for group in result.duplicate_groups
        if group["kind"]
        in {"same_protocol_different_output", "same_protocol_different_seed"}
    ]
    registry_records = [record for record in result.records if record.registry_ids]
    orphan_records = [
        record
        for record in result.records
        if not record.registry_ids and not record.report_links
    ]
    families: dict[str, list[ConfigRecord]] = defaultdict(list)
    for record in result.records:
        families[record.topic].append(record)
    lines = [
        "# Аудит конфигураций экспериментов",
        "",
        (
            f"Найдено **{len(result.records)}** tracked experiment YAML/YML; "
            f"CLI-loadable: **{sum(record.cli_loadable for record in result.records)}**; "
            f"base: **{roles['base']}**, full: **{roles['full']}**, smoke: "
            f"**{roles['smoke']}**, diagnostic: **{roles['diagnostic']}**, "
            f"legacy: **{roles['legacy']}**, unknown: **{roles['unknown']}**. "
            f"Конфигов с ошибками: **{len(error_records)}**, с предупреждениями: "
            f"**{len(warning_records)}**; exact duplicate groups: "
            f"**{len(exact_groups)}**, protocol duplicate groups: "
            f"**{len(protocol_groups)}**; с experiment registry связано: "
            f"**{len(registry_records)}**."
        ),
        "",
        "## 1. Область аудита",
        "",
        "Обследованы tracked-конфиги: `configs.yaml`, `configs/`, `experiments/`, а также YAML/YML в `benchmark/`, `bench/` и `model_zoo/`, если они существуют. Исключены `data/`, `benchmark_results/`, `.git`, окружения, кэши и runtime-конфиги.",
        "",
        "Каталоги верхнего уровня: " + ", ".join(f"`{value}`" for value in result.scanned_directories) + ".",
        "",
        "## 2. Фактические загрузчики конфигураций",
        "",
    ]
    lines.extend(f"- {note}" for note in result.loader_notes)
    lines.extend(["", _markdown_table(["loader_type", "configs"], [[key, value] for key, value in sorted(loaders.items())]).rstrip(), ""])
    lines.extend(
        [
            "Единого production-наследования нет. Обычный `--config` выполняет только `yaml.safe_load` и `cli.validate_config`. AutoML, calibration и personalization загружают указанный base отдельно; preprocessing ablation строит trial-конфиги в памяти. Аудитор сохраняет эти ссылки, но не меняет их семантику.",
            "",
            "## 3. Общая статистика",
            "",
            _markdown_table(
                ["role", "count"],
                [[role, roles[role]] for role in ROLE_ORDER],
            ).rstrip(),
            "",
            _markdown_table(
                ["status", "count"],
                [[status, statuses[status]] for status in sorted(VALID_STATUSES)],
            ).rstrip(),
            "",
            "## 4. Конфигурационные семейства",
            "",
            _markdown_table(
                ["family", "count", "base", "full", "smoke", "diagnostic/ablation", "registry"],
                [
                    [
                        family,
                        len(records),
                        sum(item.role == "base" for item in records),
                        sum(item.role == "full" for item in records),
                        sum(item.role == "smoke" for item in records),
                        sum(item.role in {"diagnostic", "ablation"} for item in records),
                        sum(bool(item.registry_ids) for item in records),
                    ]
                    for family, records in sorted(families.items())
                ],
            ).rstrip(),
            "",
            "Семейства используют фактические схемы своих загрузчиков. На следующем этапе допустимо выделять общие base-конфиги только внутри одного loader family.",
            "",
            "## 5. Связь с реестром экспериментов",
            "",
            "### Experiment registry consistency",
            "",
            _markdown_table(
                ["experiment_id", "config_path", "field", "registry_value", "config_value", "severity", "recommended_action"],
                [
                    [
                        value["experiment_id"],
                        value["config_path"],
                        value["field"],
                        value["registry_value"],
                        value["config_value"],
                        value["severity"],
                        value["recommended_action"],
                    ]
                    for value in result.registry_consistency
                ],
            ).rstrip(),
            "",
            "Отсутствующие config_path из registry:",
            "",
            _markdown_table(
                ["experiment_id", "config_path"],
                [[value["experiment_id"], value["config_path"]] for value in result.registry_missing_configs],
            ).rstrip(),
            "",
            "## 6. Базовые конфиги и наследование",
            "",
            _markdown_table(
                ["child", "base", "depth", "exists"],
                [
                    [record.path, record.base_path, record.inheritance_depth, record.base_exists]
                    for record in result.records
                    if record.base_path
                ],
            ).rstrip(),
            "",
            "Отсутствующие base-конфиги:",
            "",
            _markdown_table(
                ["config_path", "base_config"],
                [[value["config_path"], value["base_config"]] for value in result.missing_bases],
            ).rstrip(),
            "",
            "Циклы:",
            "",
            _markdown_table(
                ["cycle"],
                [[" → ".join(cycle)] for cycle in result.inheritance_cycles],
            ).rstrip(),
            "",
            "## 7. CLI-совместимость",
            "",
            _markdown_table(
                ["config", "loader", "CLI argument", "loadable", "schema"],
                [
                    [
                        record.path,
                        record.loader_type,
                        record.expected_cli_argument or "—",
                        record.cli_loadable,
                        record.schema_valid,
                    ]
                    for record in result.records
                ],
            ).rstrip(),
            "",
            "## 8. Ошибки научного протокола",
            "",
            _markdown_table(
                ["config", "error"],
                [
                    [record.path, issue.display()]
                    for record in error_records
                    for issue in record.errors
                ],
            ).rstrip(),
            "",
            "Ошибки зафиксированы для последующей ручной проверки; в рамках 10Б.1 конфиги не исправлялись.",
            "",
            "## 9. Устаревшие и неизвестные поля",
            "",
            "Фактические контракты используют несколько контекстных имён, которые нельзя механически переименовывать: `target_col`/`target_cols` принадлежат benchmark dataset, `target`/`targets` — специализированным specs; `n_classes`, `num_classes`, `n_outputs` относятся к разным слоям; `random_state`, `split_seed`, `model_seed`, `model_seeds` имеют различную семантику; `validation` — фактическая секция inner validation benchmark, а `feature_scaling`, `raw_preprocessing`, `preprocessing` описывают разные преобразования. `output_dir` также встречается на root, `experiment`, `analysis` и `audit` уровнях. Эти варианты не считаются взаимозаменяемыми автоматически.",
            "",
            _markdown_table(
                ["config", "warning"],
                [
                    [record.path, issue.display()]
                    for record in warning_records
                    for issue in record.warnings
                    if issue.code in {"unknown_field", "ignored_or_unknown_field", "conflicting_key_types"}
                ],
            ).rstrip(),
            "",
            "## 10. Абсолютные пути",
            "",
            _markdown_table(
                ["config", "yaml_key", "value", "severity"],
                [
                    [record.path, found["yaml_key"], found["value"], "error"]
                    for record in result.records
                    for found in find_absolute_paths(record.document)
                ],
            ).rstrip(),
            "",
            "## 11. Дублирующиеся конфигурации",
            "",
            _markdown_table(
                ["group", "kind", "configs", "recommendation"],
                [
                    [
                        group["group_id"],
                        group["kind"],
                        ", ".join(group["configs"]),
                        group["recommendation"],
                    ]
                    for group in result.duplicate_groups
                ],
            ).rstrip(),
            "",
            "## 12. Невостребованные конфигурации",
            "",
            "Ниже перечислены конфиги, для которых не найдена точная ссылка ни в tracked-отчётах, ни в experiment registry. Это не доказательство ненужности.",
            "",
            _markdown_table(["config"], [[record.path] for record in orphan_records]).rstrip(),
            "",
            "## 13. Legacy и invalidated",
            "",
            _markdown_table(
                ["config", "role", "status", "superseded_by", "reason"],
                [
                    [
                        record.path,
                        record.role,
                        record.status,
                        record.superseded_by or "—",
                        record.deprecation_reason or "—",
                    ]
                    for record in result.records
                    if record.is_legacy or record.status == "invalidated"
                ],
            ).rstrip(),
            "",
            "## 14. Рекомендуемая целевая структура",
            "",
            "Не переносить файлы автоматически. На этапе 10Б.2 сохранить тематические каталоги `calibration/` и `pm_regression/`, а для многочисленных root-level конфигов рассмотреть минимальное разделение на `classification`, `raw_eeg`, `sequence_models`, `analysis` и `preprocessing_ablation` с подкаталогами `base`, `smoke`, `full`, только если CLI и ссылки получают совместимый alias/deprecation-период.",
            "",
            "## 15. План безопасной консолидации",
            "",
            "1. Зафиксировать вручную роль/status для `unknown` и `unclassified` записей.",
            "2. Исправить только подтверждённые ошибки отдельными маленькими patches с тестами.",
            "3. Выбрать одну duplicate group за раз и проверить resolved config hash и существующие артефакты.",
            "4. Добавлять base-конфиги только внутри одного loader family; не вводить общий merge для специализированных specs.",
            "5. Сначала добавить совместимые ссылки/aliases, затем обновить отчёты и experiment registry, и лишь после этого обсуждать перемещение.",
            "6. Повторить этот аудитор и полный pytest после каждого пакета.",
            "",
            "## 16. Что нельзя менять автоматически",
            "",
            "Нельзя автоматически менять config_path, output_dir, порядок PM targets, split/model seeds, raw preprocessing, sequence grouping, validation strategy, calibration budgets/methods, ссылки на completed runs и конфиги со статусом final/baseline. Также нельзя удалять exact/protocol duplicates без проверки их исторических артефактов.",
            "",
        ]
    )
    if result.curation:
        curated = [record for record in result.records if record.curation]
        reviews = Counter(
            record.curation.get("review_status", "") for record in curated
        )
        decisions = Counter(record.curation.get("decision", "") for record in curated)
        lines.extend(
            [
                "## 17. Ручной curation layer",
                "",
                (
                    f"Применён `config_curation.yaml`: решений **{len(curated)}**, "
                    f"reviewed **{reviews['reviewed']}**, needs_evidence "
                    f"**{reviews['needs_evidence']}**, not_applicable "
                    f"**{reviews['not_applicable']}**."
                ),
                "",
                _markdown_table(
                    ["decision", "count"],
                    [[key, value] for key, value in sorted(decisions.items())],
                ).rstrip(),
                "",
                "Curation влияет только на отчётные role/status/decision поля; исходные experiment YAML и production loaders не изменяются.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_curation_markdown(result: AuditResult) -> str:
    curated = [record for record in result.records if record.curation]
    curated_by_path = {record.path: record for record in curated}
    reviews = Counter(record.curation.get("review_status", "") for record in curated)
    decisions = Counter(record.curation.get("decision", "") for record in curated)
    originally_unclassified = [
        record for record in curated if record.automatic_status == "unclassified"
    ]
    orphan_records = [
        record
        for record in curated
        if not record.registry_ids and not record.report_links
    ]
    protected = [
        record
        for record in curated
        if not record.curation.get("safe_to_move", False)
        and not record.curation.get("safe_to_edit", False)
    ]
    needs = [
        record
        for record in curated
        if record.curation.get("review_status") == "needs_evidence"
        or record.curation.get("decision") == "review_later"
    ]
    families = result.curation.get("families", {})
    duplicate_groups = result.curation.get("duplicate_groups", [])
    seed_provenance = result.curation.get("seed_provenance", [])
    normalization_plan = result.curation.get("normalization_plan", [])
    lines = [
        "# Курирование конфигураций экспериментов",
        "",
        "## 1. Итог",
        "",
        (
            f"Курировано **{len(curated)}** конфигов; ранее unclassified рассмотрено "
            f"**{len(originally_unclassified)}**. Reviewed: **{reviews['reviewed']}**, "
            f"needs_evidence: **{reviews['needs_evidence']}**, not_applicable: "
            f"**{reviews['not_applicable']}**. Canonical family decisions: "
            f"**{len(families)}**; safe_to_move=true: "
            f"**{sum(bool(record.curation.get('safe_to_move')) for record in curated)}**, "
            f"safe_to_edit=true: "
            f"**{sum(bool(record.curation.get('safe_to_edit')) for record in curated)}**."
        ),
        "",
        _markdown_table(
            ["decision", "count"],
            [[key, value] for key, value in sorted(decisions.items())],
        ).rstrip(),
        "",
        "## 2. Решения по 31 unclassified конфигурации",
        "",
        _markdown_table(
            [
                "config",
                "review_status",
                "decision",
                "role",
                "result_status",
                "canonical",
                "reason",
            ],
            [
                [
                    record.path,
                    record.curation["review_status"],
                    record.curation["decision"],
                    record.role,
                    record.status,
                    record.curation["canonical_config"],
                    record.curation["decision_reason"],
                ]
                for record in originally_unclassified
            ],
        ).rstrip(),
        "",
        "## 3. Канонические конфиги по семействам",
        "",
        _markdown_table(
            [
                "family",
                "canonical",
                "canonical status",
                "canonical smoke",
                "base",
                "legacy",
                "reason",
            ],
            [
                [
                    name,
                    value.get("canonical_config") or "—",
                    (
                        curated_by_path[value["canonical_config"]]
                        .curation.get("canonical_status", "")
                        if value.get("canonical_config")
                        and value["canonical_config"] in curated_by_path
                        else ""
                    ),
                    value.get("canonical_smoke_config") or "—",
                    ", ".join(value.get("base_configs", [])) or "—",
                    ", ".join(value.get("legacy_configs", [])) or "—",
                    value.get("decision_reason", ""),
                ]
                for name, value in sorted(families.items())
            ],
        ).rstrip(),
        "",
        "## 4. Base и template configs",
        "",
        _markdown_table(
            ["config", "decision", "used by", "evidence"],
            [
                [
                    record.path,
                    record.curation["decision"],
                    record.base_path or "referenced as base/template",
                    ", ".join(record.curation.get("evidence", [])),
                ]
                for record in curated
                if record.curation.get("decision") == "keep_as_base"
            ],
        ).rstrip(),
        "",
        "## 5. Smoke и diagnostic configs",
        "",
        _markdown_table(
            ["config", "decision", "status"],
            [
                [record.path, record.curation["decision"], record.status]
                for record in curated
                if record.curation.get("decision")
                in {"keep_as_smoke", "keep_as_diagnostic"}
            ],
        ).rstrip(),
        "",
        "## 6. Legacy configs",
        "",
        _markdown_table(
            ["config", "decision", "result status", "reason"],
            [
                [
                    record.path,
                    record.curation["decision"],
                    record.status,
                    record.curation["decision_reason"],
                ]
                for record in curated
                if record.curation.get("decision") == "keep_as_legacy"
            ],
        ).rstrip(),
        "",
        "## 7. Несвязанные конфиги",
        "",
        _markdown_table(
            [
                "config",
                "category",
                "decision",
                "evidence",
                "canonical",
                "safe_to_move",
            ],
            [
                [
                    record.path,
                    record.curation.get("unlinked_category", "insufficient evidence"),
                    record.curation["decision"],
                    ", ".join(record.curation.get("evidence", [])),
                    record.curation["canonical_config"],
                    record.curation["safe_to_move"],
                ]
                for record in orphan_records
            ],
        ).rstrip(),
        "",
        "## 8. Scientific protocol duplicate groups",
        "",
        _markdown_table(
            [
                "group",
                "classification",
                "configs",
                "keep separate",
                "canonical template",
                "relationship",
                "future candidate",
            ],
            [
                [
                    value.get("group_id", ""),
                    value.get("duplicate_classification", ""),
                    ", ".join(value.get("configs", [])),
                    value.get("keep_separate", False),
                    value.get("canonical_template") or "—",
                    value.get("relationship", ""),
                    value.get("future_consolidation_candidate", False),
                ]
                for value in duplicate_groups
            ],
        ).rstrip(),
        "",
        "## 9. Seed provenance",
        "",
        _markdown_table(
            [
                "model/experiment",
                "registry seeds",
                "primary config seeds",
                "siblings",
                "external orchestration",
                "status",
                "recommended metadata fix",
            ],
            [
                [
                    value.get("name", ""),
                    stable_json(value.get("registry_seeds", [])),
                    stable_json(value.get("primary_config_seeds", [])),
                    ", ".join(value.get("sibling_seed_configs", [])) or "—",
                    value.get("external_orchestration", ""),
                    value.get("provenance_status", ""),
                    value.get("recommended_metadata_fix", ""),
                ]
                for value in seed_provenance
            ],
        ).rstrip(),
        "",
        "## 10. Защищённые конфиги",
        "",
        _markdown_table(
            ["config", "protected fields"],
            [
                [
                    record.path,
                    ", ".join(record.curation.get("protected_fields", [])) or "all protocol and path fields",
                ]
                for record in protected
            ],
        ).rstrip(),
        "",
        "## 11. Конфиги с недостаточным evidence",
        "",
        _markdown_table(
            ["config", "reason"],
            [[record.path, record.curation["decision_reason"]] for record in needs],
        ).rstrip(),
        "",
        "## 12. Кандидаты на минимальную нормализацию",
        "",
        "Кандидатами считаются только metadata/registry изменения и четыре явно описанные protocol groups. Source YAML не меняются на этом этапе.",
        "",
        "## 13. Конфиги, которые нельзя менять автоматически",
        "",
        "Все перечисленные конфиги имеют `safe_to_move=false` и `safe_to_edit=false`, если отдельное решение явно не говорит обратного. Это сохраняет ссылки CLI, reports, base/template и completed-run provenance.",
        "",
        "## 14. План 10Б.2Б",
        "",
    ]
    if normalization_plan:
        for package in normalization_plan:
            lines.extend(
                [
                    f"### {package.get('name', 'Пакет')}",
                    "",
                    f"- Затрагиваемые файлы: {', '.join(package.get('files', [])) or 'определить после review'}",
                    f"- Риск: {package.get('risk', '')}",
                    f"- Тесты: {', '.join(package.get('tests', []))}",
                    f"- Dry-load: {', '.join(package.get('dry_load', []))}",
                    f"- Rollback: {package.get('rollback', '')}",
                    "",
                ]
            )
    else:
        lines.append("_План не задан._")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(result: AuditResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "inventory": output_dir / "config_inventory.csv",
        "registry": output_dir / "config_registry.yaml",
        "report": output_dir / "config_audit.md",
    }
    outputs["inventory"].write_text(render_inventory_csv(result), encoding="utf-8", newline="")
    outputs["registry"].write_text(render_config_registry(result), encoding="utf-8", newline="")
    outputs["report"].write_text(render_markdown(result), encoding="utf-8", newline="")
    if result.curation:
        outputs["curation_report"] = output_dir / "config_curation.md"
        outputs["curation_report"].write_text(
            render_curation_markdown(result),
            encoding="utf-8",
            newline="",
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/summary"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-config-errors", action="store_true")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--curation",
        type=Path,
        help="Optional persistent manual configuration decision layer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    try:
        result = audit_repository(
            root,
            includes=args.include or None,
            excludes=args.exclude or None,
            curation_path=args.curation,
        )
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        outputs = write_outputs(result, output_dir)
    except Exception as exc:
        if args.strict:
            print(f"Configuration auditor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        raise
    error_count = sum(bool(record.errors) for record in result.records)
    warning_count = sum(bool(record.warnings) for record in result.records)
    print(
        f"Audited {len(result.records)} configs: "
        f"{error_count} with errors, {warning_count} with warnings"
    )
    for name, path in outputs.items():
        print(f"{name}: {path.relative_to(root).as_posix()}")
    if args.strict and result.structural_errors:
        for error in result.structural_errors:
            print(f"Structural audit error: {error}", file=sys.stderr)
        return 2
    if args.fail_on_config_errors and error_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
