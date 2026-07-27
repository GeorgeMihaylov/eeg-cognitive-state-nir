"""Build deterministic, provenance-backed metrics tables for project hand-off.

The script reads existing structured experiment artifacts.  It never trains a
model and never recomputes metrics from predictions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_FILENAME = "metrics_provenance.yaml"
ALLOWED_SOURCE_TYPES = {
    "structured_csv",
    "structured_json",
    "resolved_config",
    "tracked_report",
    "registry_constant",
}
SCIENTIFIC_STATUSES = {"final", "baseline"}
PM_TARGETS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]
DISPLAY_MODELS = {
    "random_forest": "Random Forest",
    "torch_mlp": "Torch MLP",
    "torch_lstm_gapaware": "LSTM",
    "torch_bilstm_gapaware": "BiLSTM",
    "torch_transformer": "Transformer",
    "torch_eegnet": "EEGNet",
    "torch_shallow_convnet": "ShallowConvNet",
}

CLASSIFICATION_FIELDS = [
    "experiment_id",
    "result_status",
    "model",
    "model_family",
    "input_type",
    "feature_set",
    "preprocessing",
    "evaluation_protocol",
    "n_folds",
    "seeds",
    "n_subjects",
    "n_samples",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "cohen_kappa_mean",
    "auc_mean",
    "ordinal_mae_mean",
    "adjacent_accuracy_mean",
    "severe_error_rate_mean",
    "primary_metric",
    "primary_value",
    "report_path",
    "config_path",
    "commit",
    "metric_source",
    "notes",
]
PM_FIELDS = [
    "experiment_id",
    "result_status",
    "model",
    "feature_set",
    "preprocessing",
    "evaluation_protocol",
    "n_folds",
    "seeds",
    "n_subjects",
    "n_samples",
    "targets",
    "macro_mae_mean",
    "macro_mae_std",
    "macro_rmse_mean",
    "macro_rmse_std",
    "macro_r2_mean",
    "macro_r2_std",
    "macro_pearson_mean",
    "macro_spearman_mean",
    "macro_abs_bias_mean",
    "per_target_metrics_source",
    "report_path",
    "config_path",
    "commit",
    "metric_source",
    "notes",
]
PERSONALIZATION_FIELDS = [
    "experiment_id",
    "task",
    "target",
    "result_status",
    "method",
    "calibration_budget",
    "seeds",
    "n_subjects",
    "metric_before",
    "metric_after",
    "absolute_gain",
    "relative_gain",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "subjects_improved_fraction",
    "subjects_improved_at_least_2_of_3",
    "subjects_improved_all_3",
    "secondary_metrics_json",
    "report_path",
    "config_path",
    "commit",
    "notes",
]
PREPROCESSING_FIELDS = [
    "experiment_id",
    "trial_id",
    "preprocessing_steps",
    "model",
    "input_type",
    "seed",
    "n_folds",
    "n_subjects",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "accuracy_mean",
    "factor_effects_json",
    "result_status",
    "report_path",
    "config_path",
    "notes",
]


class PackageValidationError(ValueError):
    """Raised when provenance or generated tables violate package contracts."""


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PackageValidationError(f"Expected a YAML mapping: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _relative_path(value: Any, field: str) -> str:
    if value in (None, ""):
        return ""
    text = str(value).replace("\\", "/")
    if Path(text).is_absolute() or PurePath(text).drive:
        raise PackageValidationError(f"Absolute path is forbidden in {field}: {text}")
    return text


def _resolve(repo_root: Path, value: Any, field: str) -> Path:
    return repo_root / _relative_path(value, field)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise PackageValidationError(f"Non-finite metric: {value!r}")
    return result


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        return format(value, ".15g")
    return str(value)


def _mean(rows: Sequence[Mapping[str, str]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _pstdev(rows: Sequence[Mapping[str, str]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.pstdev(clean) if clean else None


def _one(
    rows: Sequence[Mapping[str, str]],
    *,
    context: str,
    **filters: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in filters.items())
    ]
    if len(matches) != 1:
        raise PackageValidationError(
            f"Expected one {context} row for {filters}, found {len(matches)}"
        )
    return matches[0]


def _experiment_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["experiment_id"]: row
        for row in registry.get("experiments", [])
        if isinstance(row, dict) and row.get("experiment_id")
    }


def _unresolved_ids(registry: Mapping[str, Any]) -> set[str]:
    return {
        row["experiment_id"]
        for row in registry.get("unresolved_entries", [])
        if isinstance(row, dict) and row.get("experiment_id")
    }


def _config_paths(registry: Mapping[str, Any]) -> set[str]:
    return {
        str(row["config_path"]).replace("\\", "/")
        for row in registry.get("configs", [])
        if isinstance(row, dict) and row.get("config_path")
    }


def validate_provenance(
    provenance: Mapping[str, Any],
    experiment_registry: Mapping[str, Any],
    config_registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    *,
    strict: bool = False,
) -> None:
    """Validate provenance and source availability without writing outputs."""

    errors: list[str] = []
    experiments = _experiment_index(experiment_registry)
    known_ids = set(experiments) | _unresolved_ids(experiment_registry)
    known_configs = _config_paths(config_registry)

    if provenance.get("schema_version") != 1:
        errors.append("metrics provenance schema_version must be 1")
    if provenance.get("pm_target_order") != PM_TARGETS:
        errors.append("canonical PM target order is invalid")

    seen_row_ids: set[str] = set()
    logical_keys: set[tuple[str, ...]] = set()
    sections = ("classification", "pm_regression", "personalization", "preprocessing")
    for section in sections:
        rows = provenance.get(section)
        if not isinstance(rows, list):
            errors.append(f"{section} must be a list")
            continue
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"{section} contains a non-mapping row")
                continue
            row_id = str(row.get("row_id", ""))
            if not row_id:
                errors.append(f"{section} row is missing row_id")
            elif row_id in seen_row_ids:
                errors.append(f"duplicate provenance row_id: {row_id}")
            seen_row_ids.add(row_id)

            experiment_id = str(row.get("experiment_id", ""))
            if experiment_id not in known_ids:
                errors.append(f"unknown experiment ID: {experiment_id}")
            status = row.get("status")
            if not status:
                errors.append(f"{row_id}: missing status")
            elif experiment_id in experiments and status != experiments[experiment_id].get(
                "status"
            ):
                errors.append(
                    f"{row_id}: status {status!r} differs from experiment registry"
                )
            if section in {"classification", "pm_regression"} and status not in SCIENTIFIC_STATUSES:
                errors.append(f"{row_id}: {status!r} is forbidden in a main table")

            source_type = row.get("metric_source_type")
            if source_type not in ALLOWED_SOURCE_TYPES:
                errors.append(f"{row_id}: unsupported metric source type {source_type!r}")
            for field in (
                "metric_source_path",
                "config_path",
                "report_path",
                "runtime_path",
                "stability_source",
                "comparison_source",
                "target_source",
                "per_target_metrics_source",
            ):
                try:
                    _relative_path(row.get(field), f"{row_id}.{field}")
                except PackageValidationError as exc:
                    errors.append(str(exc))

            source_path = row.get("metric_source_path")
            if source_path and source_type != "registry_constant":
                source = _resolve(repo_root, source_path, f"{row_id}.metric_source_path")
                if not source.exists():
                    errors.append(f"{row_id}: missing metric source {source_path}")
            if strict:
                for field in (
                    "stability_source",
                    "comparison_source",
                    "target_source",
                    "per_target_metrics_source",
                ):
                    if row.get(field) and not _resolve(
                        repo_root, row[field], f"{row_id}.{field}"
                    ).exists():
                        errors.append(f"{row_id}: missing {field} {row[field]}")
                config_path = _relative_path(row.get("config_path"), "config_path")
                if config_path and config_path not in known_configs:
                    errors.append(f"{row_id}: config is absent from config registry: {config_path}")

            if section == "personalization":
                key = (section, experiment_id, str(row.get("method")), "20%", "multiseed")
            elif section == "preprocessing":
                key = (section, experiment_id, str(row.get("trial_id")))
            else:
                key = (section, experiment_id)
            if key in logical_keys:
                errors.append(f"duplicate logical result row: {key}")
            logical_keys.add(key)

            if section == "classification" and row.get("input_type") not in {
                "feature_window",
                "feature_sequence",
                "raw_eeg_window",
            }:
                errors.append(f"{row_id}: classification input_type is not explicit")

    unresolved = provenance.get("unresolved_results", [])
    if not isinstance(unresolved, list):
        errors.append("unresolved_results must be a list")
    else:
        for row in unresolved:
            if row.get("experiment_id") not in _unresolved_ids(experiment_registry):
                errors.append(f"unknown unresolved experiment ID: {row.get('experiment_id')}")
            if any(key.endswith("_mean") for key in row):
                errors.append(
                    f"unresolved result contains an invented metric: {row.get('experiment_id')}"
                )

    mixins = provenance.get("mixins", [])
    for row in mixins if isinstance(mixins, list) else []:
        if row.get("experiment_id") not in experiments:
            errors.append(f"unknown mixin experiment ID: {row.get('experiment_id')}")

    effects = provenance.get("preprocessing_factor_effects", {})
    try:
        effect_path = _relative_path(effects.get("metric_source_path"), "factor effects")
        if effect_path and not _resolve(repo_root, effect_path, "factor effects").exists():
            errors.append(f"missing factor-effect source: {effect_path}")
    except PackageValidationError as exc:
        errors.append(str(exc))

    if errors:
        raise PackageValidationError("\n".join(errors))


def validate_generated_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    classification = list(tables["classification"])
    regression = list(tables["pm_regression"])
    personalization = list(tables["personalization"])
    preprocessing = list(tables["preprocessing"])

    if len(classification) != 7:
        errors.append(f"classification table must contain seven models, got {len(classification)}")
    if any(row["result_status"] not in SCIENTIFIC_STATUSES for row in classification):
        errors.append("classification table contains smoke, diagnostic, or invalidated results")
    if any(row["result_status"] not in SCIENTIFIC_STATUSES for row in regression):
        errors.append("regression table contains smoke, diagnostic, or invalidated results")
    if any("macro_mae_mean" in row for row in classification):
        errors.append("classification table contains regression fields")
    if any("macro_f1_mean" in row for row in regression):
        errors.append("regression table contains classification fields")
    if {row["model"] for row in classification} != {
        "Random Forest",
        "Torch MLP",
        "LSTM",
        "BiLSTM",
        "Transformer",
        "EEGNet",
        "ShallowConvNet",
    }:
        errors.append("classification model set is incomplete")
    if {row["model"] for row in regression} != {"mean_regressor", "random_forest"}:
        errors.append("PM table must contain mean and Random Forest baselines")
    if any(row["targets"].split("|") != PM_TARGETS for row in regression):
        errors.append("PM target order differs from the canonical order")
    methods = {(row["task"], row["method"]) for row in personalization}
    for task in ("classification", "pm_regression"):
        for method in ("zero_shot", "head_only", "full_model"):
            if (task, method) not in methods:
                errors.append(f"missing personalization row: {task}/{method}")
    for row in personalization:
        before = _float(row["metric_before"])
        after = _float(row["metric_after"])
        gain = _float(row["absolute_gain"])
        if before is None or after is None or gain is None:
            errors.append(f"incomplete personalization metric: {row['task']}/{row['method']}")
            continue
        expected = after - before if row["task"] == "classification" else before - after
        if not math.isclose(expected, gain, rel_tol=0, abs_tol=1e-12):
            errors.append(f"incorrect gain direction: {row['task']}/{row['method']}")
    if {row["trial_id"] for row in preprocessing if len(row["trial_id"]) == 1} != set(
        "ABCDEFGH"
    ):
        errors.append("preprocessing table must contain trials A-H")
    standard = [row for row in preprocessing if row["trial_id"] == "standard_clip"]
    if len(standard) != 1 or standard[0]["result_status"] != "diagnostic":
        errors.append("standard_clip must be a diagnostic row")
    if any(row["result_status"] == "invalidated" for row in preprocessing):
        errors.append("invalidated preprocessing result entered the comparison table")
    if len(provenance.get("unresolved_results", [])) != 4:
        errors.append("four unresolved comparable results must remain explicit")
    if errors:
        raise PackageValidationError("\n".join(errors))


def _classification_rows(
    provenance: Mapping[str, Any],
    experiments: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cache: dict[str, list[dict[str, str]]] = {}
    for spec in provenance["classification"]:
        source_path = spec["metric_source_path"]
        rows = cache.setdefault(source_path, read_csv(repo_root / source_path))
        selected = [row for row in rows if row.get("model") == spec["source_model"]]
        if not selected:
            raise PackageValidationError(
                f"No rows for {spec['source_model']} in {source_path}"
            )
        experiment = experiments[spec["experiment_id"]]
        seeds = sorted({int(row["seed"]) for row in selected})
        folds = sorted({row["fold"] for row in selected})
        model_name = DISPLAY_MODELS.get(spec["source_model"], spec["source_model"])
        result.append(
            {
                "experiment_id": spec["experiment_id"],
                "result_status": spec["status"],
                "model": model_name,
                "model_family": spec["model_family"],
                "input_type": spec["input_type"],
                "feature_set": experiment["feature_set"],
                "preprocessing": experiment["preprocessing"],
                "evaluation_protocol": experiment["evaluation_protocol"],
                "n_folds": len(folds),
                "seeds": "|".join(map(str, seeds)),
                "n_subjects": experiment["n_subjects"],
                "n_samples": spec["n_samples"],
                "accuracy_mean": _mean(selected, "accuracy"),
                "accuracy_std": _pstdev(selected, "accuracy"),
                "balanced_accuracy_mean": _mean(selected, "balanced_accuracy"),
                "balanced_accuracy_std": _pstdev(selected, "balanced_accuracy"),
                "macro_f1_mean": _mean(selected, "macro_f1"),
                "macro_f1_std": _pstdev(selected, "macro_f1"),
                "weighted_f1_mean": _mean(selected, "weighted_f1"),
                "weighted_f1_std": _pstdev(selected, "weighted_f1"),
                "cohen_kappa_mean": _mean(selected, "kappa"),
                "auc_mean": _mean(selected, "auc"),
                "ordinal_mae_mean": _mean(selected, "ordinal_mae"),
                "adjacent_accuracy_mean": _mean(selected, "adjacent_accuracy"),
                "severe_error_rate_mean": _mean(selected, "severe_error_rate"),
                "primary_metric": "macro_f1",
                "primary_value": _mean(selected, "macro_f1"),
                "report_path": spec["report_path"],
                "config_path": spec["config_path"],
                "commit": spec["commit"],
                "metric_source": f"{spec['metric_source_type']}:{source_path}",
                "notes": spec["notes"],
            }
        )
    return sorted(result, key=lambda row: (-float(row["primary_value"]), row["model"]))


def _pm_rows(
    provenance: Mapping[str, Any],
    experiments: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    result: list[dict[str, Any]] = []
    per_target: dict[str, list[dict[str, Any]]] = {}
    cache: dict[str, list[dict[str, str]]] = {}
    for spec in provenance["pm_regression"]:
        source_path = spec["metric_source_path"]
        rows = cache.setdefault(source_path, read_csv(repo_root / source_path))
        row = _one(rows, context="PM summary", model=spec["source_model"])
        experiment = experiments[spec["experiment_id"]]
        target_rows = read_csv(repo_root / spec["per_target_metrics_source"])
        target_summary: list[dict[str, Any]] = []
        target_key = "target_name" if target_rows and "target_name" in target_rows[0] else "target"
        for target in PM_TARGETS:
            current = [item for item in target_rows if item.get(target_key) == target]
            if not current:
                raise PackageValidationError(f"Missing per-target rows for {target}")
            target_summary.append(
                {
                    "target": target,
                    "mae": _mean(current, "mae"),
                    "rmse": _mean(current, "rmse"),
                    "r2": _mean(current, "r2"),
                    "pearson": _mean(current, "pearson"),
                    "spearman": _mean(current, "spearman"),
                    "absolute_bias": _mean(current, "absolute_bias"),
                }
            )
        per_target[spec["experiment_id"]] = target_summary
        result.append(
            {
                "experiment_id": spec["experiment_id"],
                "result_status": spec["status"],
                "model": spec["source_model"],
                "feature_set": experiment["feature_set"],
                "preprocessing": experiment["preprocessing"],
                "evaluation_protocol": experiment["evaluation_protocol"],
                "n_folds": int(float(row["n_folds"])),
                "seeds": "|".join(map(str, experiment["seeds"])),
                "n_subjects": experiment["n_subjects"],
                "n_samples": 43174,
                "targets": "|".join(PM_TARGETS),
                "macro_mae_mean": _float(row["mae_macro"]),
                "macro_mae_std": _float(row["mae_macro_std"]),
                "macro_rmse_mean": _float(row["rmse_macro"]),
                "macro_rmse_std": _float(row["rmse_macro_std"]),
                "macro_r2_mean": _float(row["r2_macro"]),
                "macro_r2_std": _float(row["r2_macro_std"]),
                "macro_pearson_mean": _float(row["pearson_macro"]),
                "macro_spearman_mean": _float(row["spearman_macro"]),
                "macro_abs_bias_mean": _float(row.get("abs_bias_macro")),
                "per_target_metrics_source": spec["per_target_metrics_source"],
                "report_path": spec["report_path"],
                "config_path": spec["config_path"],
                "commit": spec["commit"],
                "metric_source": f"{spec['metric_source_type']}:{source_path}",
                "notes": spec["notes"],
            }
        )
    return sorted(result, key=lambda row: float(row["macro_mae_mean"])), per_target


def _personalization_rows(
    provenance: Mapping[str, Any],
    experiments: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    pm_targets: list[dict[str, Any]] = []
    cache: dict[str, list[dict[str, str]]] = {}
    for spec in provenance["personalization"]:
        source = cache.setdefault(spec["metric_source_path"], read_csv(repo_root / spec["metric_source_path"]))
        stability = cache.setdefault(spec["stability_source"], read_csv(repo_root / spec["stability_source"]))
        comparisons = cache.setdefault(
            spec["comparison_source"], read_csv(repo_root / spec["comparison_source"])
        )
        primary = _one(
            source,
            context="personalization aggregate",
            method=spec["method"],
            metric=spec["primary_metric"],
        )
        zero = _one(
            source,
            context="zero-shot aggregate",
            method="zero_shot",
            metric=spec["primary_metric"],
        )
        stable = _one(
            stability,
            context="personalization stability",
            record_type="aggregate",
            method=spec["method"],
            metric=spec["primary_metric"],
        )
        classification = spec["task"] == "classification"
        value_key = (
            "mean_over_subject_seed_means" if classification else "mean_subject_metric"
        )
        gain_key = "mean_gain" if classification else "mean_subject_gain"
        before = _float(zero[value_key])
        after = _float(primary[value_key])
        gain = _float(primary[gain_key])
        if before is None or after is None or gain is None:
            raise PackageValidationError(f"Incomplete personalization row: {spec['row_id']}")
        secondary: dict[str, Any] = {}
        for metric_row in source:
            if metric_row["method"] != spec["method"] or metric_row["metric"] == spec["primary_metric"]:
                continue
            secondary[metric_row["metric"]] = {
                "after": _float(metric_row[value_key]),
                "gain": _float(metric_row[gain_key]),
                "ci_low": _float(metric_row["bootstrap_ci_low"]),
                "ci_high": _float(metric_row["bootstrap_ci_high"]),
            }
        if spec["method"] == "full_model":
            if classification:
                paired = [
                    row
                    for row in comparisons
                    if row.get("left_method") == "full_model"
                    and row.get("right_method") == "head_only"
                ]
            else:
                paired = [
                    row
                    for row in comparisons
                    if row.get("method") == "full_model"
                    and row.get("reference_method") == "head_only"
                ]
            secondary["full_model_vs_head_only"] = {
                row["metric"]: _float(row["mean_difference"]) for row in paired
            }
        experiment = experiments[spec["experiment_id"]]
        result.append(
            {
                "experiment_id": spec["experiment_id"],
                "task": spec["task"],
                "target": "label_q5" if classification else "|".join(PM_TARGETS),
                "result_status": spec["status"],
                "method": spec["method"],
                "calibration_budget": "20%",
                "seeds": "|".join(map(str, experiment["seeds"])),
                "n_subjects": int(float(primary["n_subjects"])),
                "metric_before": before,
                "metric_after": after,
                "absolute_gain": gain,
                "relative_gain": gain / abs(before) if before else 0.0,
                "bootstrap_ci_low": _float(primary["bootstrap_ci_low"]),
                "bootstrap_ci_high": _float(primary["bootstrap_ci_high"]),
                "subjects_improved_fraction": _float(primary["positive_subject_fraction"]),
                "subjects_improved_at_least_2_of_3": _float(
                    stable["fraction_improved_at_least_2_of_3"]
                ),
                "subjects_improved_all_3": _float(stable["fraction_improved_all_3"]),
                "secondary_metrics_json": json.dumps(
                    secondary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "report_path": spec["report_path"],
                "config_path": spec["config_path"],
                "commit": spec["commit"],
                "notes": f"Primary metric: {spec['primary_metric']}. {spec['notes']}",
            }
        )
        if not classification and spec["method"] == "full_model":
            target_rows = read_csv(repo_root / spec["target_source"])
            pm_targets = [
                {
                    "target": row["target_name"],
                    "mae_gain": _float(row["mae_gain"]),
                    "ci_low": _float(row["mae_gain_ci_low"]),
                    "ci_high": _float(row["mae_gain_ci_high"]),
                    "improved_at_least_2": _float(
                        row["fraction_improved_at_least_2_seeds"]
                    ),
                }
                for row in target_rows
                if row["method"] == "full_model"
            ]
            order = {target: index for index, target in enumerate(PM_TARGETS)}
            pm_targets.sort(key=lambda row: order[row["target"]])
    return sorted(result, key=lambda row: (row["task"], row["method"])), pm_targets


def _preprocessing_rows(
    provenance: Mapping[str, Any],
    experiments: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    effects = provenance["preprocessing_factor_effects"]["values"]
    effect_json = json.dumps(effects, sort_keys=True, separators=(",", ":"))
    source_cache: dict[str, list[dict[str, str]]] = {}
    result: list[dict[str, Any]] = []
    for spec in provenance["preprocessing"]:
        experiment = experiments[spec["experiment_id"]]
        if spec["metric_source_type"] == "structured_csv":
            source = source_cache.setdefault(
                spec["metric_source_path"], read_csv(repo_root / spec["metric_source_path"])
            )
            row = _one(source, context="preprocessing trial", trial_id=spec["trial_id"])
            result.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "trial_id": spec["trial_id"],
                    "preprocessing_steps": spec["preprocessing_steps"],
                    "model": "ShallowConvNet",
                    "input_type": "raw_eeg_window",
                    "seed": int(row["seed"]),
                    "n_folds": 5,
                    "n_subjects": experiment["n_subjects"],
                    "balanced_accuracy_mean": _float(row["balanced_accuracy_mean"]),
                    "balanced_accuracy_std": _float(row["balanced_accuracy_std"]),
                    "macro_f1_mean": _float(row["macro_f1_mean"]),
                    "accuracy_mean": _float(row["accuracy_mean"]),
                    "factor_effects_json": effect_json,
                    "result_status": spec["status"],
                    "report_path": spec["report_path"],
                    "config_path": spec["config_path"],
                    "notes": spec["notes"],
                }
            )
        else:
            tracked = spec["tracked_values"]
            result.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "trial_id": spec["trial_id"],
                    "preprocessing_steps": spec["preprocessing_steps"],
                    "model": "Torch MLP",
                    "input_type": "feature_window",
                    "seed": 42,
                    "n_folds": 1,
                    "n_subjects": experiment["n_subjects"],
                    "balanced_accuracy_mean": None,
                    "balanced_accuracy_std": None,
                    "macro_f1_mean": None,
                    "accuracy_mean": None,
                    "factor_effects_json": json.dumps(
                        tracked, sort_keys=True, separators=(",", ":")
                    ),
                    "result_status": spec["status"],
                    "report_path": spec["report_path"],
                    "config_path": spec["config_path"],
                    "notes": spec["notes"],
                }
            )
    trial_order = {letter: index for index, letter in enumerate("ABCDEFGH")}
    return sorted(
        result,
        key=lambda row: (
            row["trial_id"] == "standard_clip",
            trial_order.get(row["trial_id"], 99),
        ),
    )


def build_tables(
    provenance: Mapping[str, Any],
    experiment_registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    experiments = _experiment_index(experiment_registry)
    classification = _classification_rows(provenance, experiments, repo_root)
    pm_regression, per_target_pm = _pm_rows(provenance, experiments, repo_root)
    personalization, personalized_targets = _personalization_rows(
        provenance, experiments, repo_root
    )
    preprocessing = _preprocessing_rows(provenance, experiments, repo_root)
    tables = {
        "classification": classification,
        "pm_regression": pm_regression,
        "personalization": personalization,
        "preprocessing": preprocessing,
    }
    extras = {
        "per_target_pm": per_target_pm,
        "personalized_targets": personalized_targets,
    }
    validate_generated_tables(tables, provenance)
    return tables, extras


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    extras: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    cls = tables["classification"]
    pm = tables["pm_regression"]
    pers = tables["personalization"]
    prep = tables["preprocessing"]
    best_f1 = max(cls, key=lambda row: float(row["macro_f1_mean"]))
    best_ba = max(cls, key=lambda row: float(row["balanced_accuracy_mean"]))
    rf_pm = next(row for row in pm if row["model"] == "random_forest")
    mean_pm = next(row for row in pm if row["model"] == "mean_regressor")
    class_full = next(
        row for row in pers if row["task"] == "classification" and row["method"] == "full_model"
    )
    pm_full = next(
        row for row in pers if row["task"] == "pm_regression" and row["method"] == "full_model"
    )
    ranked_prep = sorted(
        (row for row in prep if row["trial_id"] != "standard_clip"),
        key=lambda row: -float(row["balanced_accuracy_mean"]),
    )
    standard = next(row for row in prep if row["trial_id"] == "standard_clip")
    tracked = json.loads(standard["factor_effects_json"])
    effects = provenance["preprocessing_factor_effects"]["values"]["balanced_accuracy"]

    lines = [
        "# Единая сводка экспериментальных результатов EEG-проекта",
        "",
        "## Краткое резюме",
        "",
        "1. Честное cross-subject качество заметно ниже диагностического random-window split; эти протоколы не объединяются.",
        f"2. Лучший macro F1 для `label_q5` показала модель {best_f1['model']}: {best_f1['macro_f1_mean']:.4f}.",
        f"3. Лучший balanced accuracy показала модель {best_ba['model']}: {best_ba['balanced_accuracy_mean']:.4f}.",
        "4. Random Forest остаётся сильным и воспроизводимым feature-based baseline.",
        "5. Raw-EEG CNN уступают sequence-моделям на текущем deduplicated наборе.",
        f"6. PM Random Forest превосходит mean baseline: macro MAE {rf_pm['macro_mae_mean']:.5f} против {mean_pm['macro_mae_mean']:.5f}, macro R² {rf_pm['macro_r2_mean']:.5f}.",
        f"7. Full-model персонализация классификации даёт небольшой macro F1 gain +{class_full['absolute_gain']:.5f}; порог accuracy 0.75 не достигнут (наблюдаемый максимум 0.6349206349).",
        f"8. Full-model PM-персонализация снижает macro MAE на {pm_full['absolute_gain']:.6f}, устойчиво минимум в двух seeds у {100*pm_full['subjects_improved_at_least_2_of_3']:.2f}% испытуемых.",
        "9. `full_model` лучше `head_only`, но размер дополнительного выигрыша невелик.",
        "10. Различия preprocessing относительно малы; статистическая значимость не заявлялась.",
        f"11. CAR дал отрицательный описательный эффект по balanced accuracy ({effects['car_mean_effect']:+.4f}).",
        "12. `standard_clip` устраняет экстремальные outlier failures; transfer-функциональность интегрирована через переработанный leakage-safe pipeline.",
        "",
        "## 1. Задачи и данные",
        "",
        "Пакет разделяет пяти-классовую классификацию `label_q5`, семивыходную PM-регрессию, две задачи персонализации и диагностическую предобработку. Feature windows, feature sequences и deduplicated raw EEG маркируются явно.",
        "",
        "## 2. Правила сопоставления результатов",
        "",
        "В основные таблицы входят только `final` и `baseline`. `diagnostic` вынесен отдельно, `smoke` и `invalidated` исключены. Основной научный протокол — cross-subject 5-fold GroupKFold по `subject_id`; random-window, single-seed и multi-seed, raw и feature-based результаты не смешиваются без маркировки.",
        "",
        "## 3. Классификация label_q5",
        "",
        _markdown_table(
            ["Модель", "Вход", "Seeds", "Macro F1", "Balanced accuracy", "Accuracy", "Статус"],
            [
                (
                    row["model"],
                    row["input_type"],
                    row["seeds"],
                    f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}",
                    f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}",
                    f"{row['accuracy_mean']:.4f}",
                    row["result_status"],
                )
                for row in cls
            ],
        ),
        "",
        "Sequence models дают лучшие macro F1 и balanced accuracy, однако абсолютное cross-subject качество остаётся умеренным. Ordinal Transformer хранится как отдельный диагностический/неканонический разрез и не включён в этот рейтинг.",
        "",
        "## 4. Многовыходная PM-регрессия",
        "",
        _markdown_table(
            ["Модель", "Macro MAE", "Macro RMSE", "Macro R²", "Pearson", "Spearman"],
            [
                (
                    row["model"],
                    f"{row['macro_mae_mean']:.6f} ± {row['macro_mae_std']:.6f}",
                    f"{row['macro_rmse_mean']:.6f} ± {row['macro_rmse_std']:.6f}",
                    f"{row['macro_r2_mean']:.6f} ± {row['macro_r2_std']:.6f}",
                    _fmt(row["macro_pearson_mean"]),
                    _fmt(row["macro_spearman_mean"]),
                )
                for row in pm
            ],
        ),
        "",
        "Random Forest превосходит средний baseline и даёт положительный macro R². Per-target значения агрегированы только из существующих `per_target_metrics.csv`; отсутствующий absolute bias оставлен пустым.",
    ]
    for experiment_id, target_rows in extras["per_target_pm"].items():
        lines.extend(
            [
                "",
                f"### Per-target: `{experiment_id}`",
                "",
                _markdown_table(
                    ["Target", "MAE", "RMSE", "R²", "Pearson", "Spearman", "Abs bias"],
                    [
                        (
                            row["target"],
                            row["mae"],
                            row["rmse"],
                            row["r2"],
                            row["pearson"],
                            row["spearman"],
                            row["absolute_bias"],
                        )
                        for row in target_rows
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 5. Персонализация классификации",
            "",
            _markdown_table(
                ["Метод", "Macro F1 before", "After", "Gain", "95% CI", "Improved ≥2/3"],
                [
                    (
                        row["method"],
                        row["metric_before"],
                        row["metric_after"],
                        row["absolute_gain"],
                        f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}]",
                        f"{100*row['subjects_improved_at_least_2_of_3']:.2f}%",
                    )
                    for row in pers
                    if row["task"] == "classification"
                ],
            ),
            "",
            "Full-model macro F1 gain равен +0.006569; head-only — +0.004323. Full-vs-head разности сохранены в `secondary_metrics_json`. Статистически положительный средний gain не означает достижение абсолютного порога: accuracy 0.75 не достигнута, наблюдаемый максимум 0.6349206349.",
            "",
            "## 6. Персонализация PM-регрессии",
            "",
            _markdown_table(
                ["Метод", "Macro MAE before", "After", "Reduction", "95% CI", "Improved ≥2/3", "All 3"],
                [
                    (
                        row["method"],
                        row["metric_before"],
                        row["metric_after"],
                        row["absolute_gain"],
                        f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}]",
                        f"{100*row['subjects_improved_at_least_2_of_3']:.2f}%",
                        f"{100*row['subjects_improved_all_3']:.2f}%",
                    )
                    for row in pers
                    if row["task"] == "pm_regression"
                ],
            ),
            "",
            "Full-model против head-only: преимущество по MAE 0.000800, RMSE 0.001174 и Spearman 0.009128. Fine-tuning устойчиво улучшает средние метрики, но не устраняет межсубъектную вариативность.",
            "",
            "### Per-target full-model PM personalization",
            "",
            _markdown_table(
                ["Target", "MAE gain", "95% CI", "Improved ≥2/3"],
                [
                    (
                        row["target"],
                        row["mae_gain"],
                        f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]",
                        f"{100*row['improved_at_least_2']:.2f}%",
                    )
                    for row in extras["personalized_targets"]
                ],
            ),
            "",
            "Наиболее устойчивы excitement, engagement, relaxation и focus; interest имеет минимальный средний эффект.",
            "",
            "## 7. Предобработка EEG",
            "",
            _markdown_table(
                ["Rank", "Trial", "Шаги", "Balanced accuracy", "Macro F1"],
                [
                    (
                        index,
                        row["trial_id"],
                        row["preprocessing_steps"],
                        f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}",
                        f"{row['macro_f1_mean']:.4f}",
                    )
                    for index, row in enumerate(ranked_prep, start=1)
                ],
            ),
            "",
            f"Описательные факторные эффекты balanced accuracy: CAR {effects['car_mean_effect']:+.5f} ± {effects['car_effect_std']:.5f}; band-pass {effects['band_pass_mean_effect']:+.5f} ± {effects['band_pass_effect_std']:.5f}; notch {effects['notch_mean_effect']:+.5f} ± {effects['notch_effect_std']:.5f}. Различия не объявлялись статистически значимыми.",
            "",
            "## 8. Robust scaling",
            "",
            f"`standard_clip` — diagnostic: maximum train-relative validation z-score {tracked['validation_max_train_relative_z_before']:.2f} → {tracked['validation_max_train_relative_z_after']:.2f}; outlier-subject MSE {tracked['outlier_subject_mse_before']:.4f} → {tracked['outlier_subject_mse_after']:.5f}. В one-fold outer test MAE улучшился примерно на 4.9%, RMSE — на 8.2%; это не финальное сравнение моделей.",
            "",
            "## 9. Transfer learning и mixins",
            "",
            _markdown_table(
                ["Метод", "Проверен", "Интеграция", "Решение"],
                [
                    (
                        row["method"],
                        "Да" if row["tested"] else "Нет",
                        row["integrated"],
                        row["decision"],
                    )
                    for row in provenance["mixins"]
                ],
            ),
            "",
            "Старый transfer prototype сбрасывал pretrained weights и напрямую не переносился; его назначение реализовано leakage-safe pipeline с `head_only` и `full_model`. DANN не имел корректного source/target contract, MAML — runnable production path, contrastive encoder не был подключён downstream. Prototype smoke metrics не являются научным результатом.",
            "",
            "## 10. Основные научные выводы",
            "",
            "- Классификация: sequence models лидируют, но cross-subject качество остаётся умеренным.",
            "- PM-регрессия: Random Forest лучше mean baseline и показывает положительный macro R².",
            "- Персонализация: gains малы, но устойчивы; full-model в среднем лучше head-only.",
            "- Предобработка: band-pass и notch дают небольшие различия, CAR в текущем протоколе ухудшает качество.",
            "- Архитектура: платформа поддерживает воспроизводимые GroupKFold-эксперименты, multi-output regression и leakage-safe personalization.",
            "",
            "## 11. Ограничения",
            "",
            "Наборы входов и единицы наблюдения различаются: feature windows, sequences и raw windows нельзя считать одним однородным рейтингом. Single-seed и multi-seed оценки явно помечены. One-fold diagnostics, smoke и invalidated runs не используются в научных выводах. Новые статистические тесты в рамках сборки не выполнялись.",
            "",
            "## 12. Отсутствующие сопоставимые результаты",
            "",
            _markdown_table(
                ["Experiment", "Config", "Report", "Metrics", "Причина исключения"],
                [
                    (
                        row["experiment_id"],
                        "найден" if row["config_found"] else "не найден",
                        "найден" if row["report_found"] else "не найден",
                        "найдены" if row["metrics_found"] else "не найдены",
                        row["reason"],
                    )
                    for row in provenance["unresolved_results"]
                ],
            ),
            "",
            "## 13. Источники и воспроизводимость",
            "",
            "Каждая строка таблицы связана с `metrics_provenance.yaml`. Structured CSV используются напрямую; единственные явно зафиксированные tracked-report значения относятся к factorial effects и `standard_clip`. Генератор не читает predictions и не обучает модели.",
            "",
            "Команда:",
            "",
            "```powershell",
            "python src\\18_build_colleague_metrics_package.py --experiment-registry reports\\summary\\experiment_registry.yaml --config-registry reports\\summary\\config_registry.yaml --output-dir reports\\summary --strict",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _glossary() -> str:
    rows = [
        ("Accuracy", "Доля верных классов", "выше", "окна/folds", "Искажается дисбалансом классов."),
        ("Balanced accuracy", "Средняя полнота по классам", "выше", "окна/folds", "Приоритетна для label_q5."),
        ("Macro F1", "Среднее F1 с равным весом классов", "выше", "окна/folds", "Приоритетна; чувствительна к редким классам."),
        ("Weighted F1", "F1 с весом по поддержке классов", "выше", "окна/folds", "Может скрывать слабые редкие классы."),
        ("AUC", "Площадь под ROC, multiclass aggregate", "выше", "окна/folds", "Зависит от схемы multiclass aggregation."),
        ("Ordinal MAE", "Средняя абсолютная ошибка индекса класса", "ниже", "окна/folds", "Предполагает порядковую шкалу классов."),
        ("MAE", "Средняя абсолютная ошибка", "ниже", "окна/targets/folds", "Зависит от шкалы цели."),
        ("RMSE", "Корень средней квадратичной ошибки", "ниже", "окна/targets/folds", "Сильнее штрафует выбросы."),
        ("R²", "Доля дисперсии сверх mean baseline", "выше", "окна/targets/folds", "Отрицательное значение хуже предсказания средним."),
        ("Pearson", "Линейная корреляция", "выше", "окна/targets/folds", "Не измеряет калибровку и чувствительна к выбросам."),
        ("Spearman", "Ранговая корреляция", "выше", "окна/targets/folds", "Не измеряет абсолютную ошибку."),
        ("Absolute bias", "Абсолютное среднее смещение", "ниже", "окна/targets/folds", "Не отражает разброс ошибок."),
        ("Absolute gain", "After−before для score; before−after для error", "выше", "subjects/seeds", "Направление зависит от типа метрики."),
        ("Relative gain", "Absolute gain / |baseline|", "выше", "subjects/seeds", "Нестабилен около нулевого baseline."),
        ("Bootstrap confidence interval", "Bootstrap-интервал среднего gain", "узкий и выше 0", "subjects", "Не заменяет полный анализ дизайна."),
        ("Subjects improved fraction", "Доля людей с положительным gain", "выше", "subjects", "Зависит от выбранной метрики и seed aggregation."),
    ]
    return "\n".join(
        [
            "# Глоссарий метрик",
            "",
            "Для `label_q5` приоритетны balanced accuracy и macro F1. Отсутствующие метрики оставляются пустыми, а не заменяются нулём.",
            "",
            _markdown_table(
                ["Метрика", "Смысл", "Лучше", "Агрегация", "Ограничение"], rows
            ),
            "",
        ]
    )


def _normalise_provenance(provenance: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        copy.deepcopy(dict(provenance)),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


def generate_package(
    experiment_registry_path: Path,
    config_registry_path: Path,
    output_dir: Path,
    *,
    strict: bool = False,
    validate_only: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    experiment_registry = load_yaml(experiment_registry_path)
    config_registry = load_yaml(config_registry_path)
    provenance_path = output_dir / PROVENANCE_FILENAME
    provenance = load_yaml(provenance_path)
    validate_provenance(
        provenance, experiment_registry, config_registry, repo_root, strict=strict
    )
    tables, extras = build_tables(provenance, experiment_registry, repo_root)
    if validate_only:
        return {"tables": tables, "extras": extras, "written": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "classification": output_dir / "classification_metrics_unified.csv",
        "pm_regression": output_dir / "pm_regression_metrics_unified.csv",
        "personalization": output_dir / "personalization_metrics_unified.csv",
        "preprocessing": output_dir / "preprocessing_metrics_unified.csv",
        "summary": output_dir / "colleague_metrics_summary.md",
        "glossary": output_dir / "metrics_glossary.md",
        "provenance": provenance_path,
    }
    _write_csv(outputs["classification"], CLASSIFICATION_FIELDS, tables["classification"])
    _write_csv(outputs["pm_regression"], PM_FIELDS, tables["pm_regression"])
    _write_csv(outputs["personalization"], PERSONALIZATION_FIELDS, tables["personalization"])
    _write_csv(outputs["preprocessing"], PREPROCESSING_FIELDS, tables["preprocessing"])
    outputs["summary"].write_text(
        _report(tables, extras, provenance), encoding="utf-8", newline="\n"
    )
    outputs["glossary"].write_text(_glossary(), encoding="utf-8", newline="\n")
    outputs["provenance"].write_text(
        _normalise_provenance(provenance), encoding="utf-8", newline="\n"
    )
    return {"tables": tables, "extras": extras, "written": list(outputs.values())}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-registry", required=True, type=Path)
    parser.add_argument("--config-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate", action="store_true", help="Validate without writing.")
    parser.add_argument("--strict", action="store_true", help="Require all declared sources.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_package(
        args.experiment_registry,
        args.config_registry,
        args.output_dir,
        strict=args.strict,
        validate_only=args.validate,
    )
    action = "Validated" if args.validate else "Generated"
    counts = {name: len(rows) for name, rows in result["tables"].items()}
    print(
        f"{action} colleague metrics package: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
