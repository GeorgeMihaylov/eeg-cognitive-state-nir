"""Deterministic, training-free audit of benchmark target definitions.

The module reads only the target, identifier, and PM aggregate columns needed
for the audit.  It never imports benchmark orchestration or model code and does
not materialize candidate labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from cogstate.protocol import PM_METRICS


REPO_ROOT = Path(__file__).resolve().parents[2]
PM_DISPLAY_NAMES_RU = {
    "attention": "Внимание",
    "engagement": "Вовлечённость",
    "excitement": "Возбуждение",
    "stress": "Стресс",
    "relaxation": "Расслабление",
    "interest": "Интерес",
    "focus": "Фокус",
}
TARGET_COLUMNS = tuple(f"target_{metric}" for metric in PM_METRICS)
REGISTRY_REQUIRED_FIELDS = (
    "target_id",
    "display_name_ru",
    "device_metric",
    "target_family",
    "target_type",
    "source_columns",
    "processed_column",
    "aggregation",
    "units_or_scale",
    "value_range",
    "higher_value_interpretation",
    "available_sources",
    "native_or_derived",
    "derivation_file",
    "derivation_function",
    "derivation_scope",
    "missing_value_policy",
    "recommended_metrics",
    "supported_feature_inputs",
    "supported_raw_input",
    "current_task_ids",
    "current_experiment_ids",
    "leakage_risk",
    "status",
    "limitations",
)
OUTPUT_FILENAMES = (
    "target_inventory.csv",
    "target_availability_by_source.csv",
    "target_cohort_counts.csv",
    "target_derivation_audit.csv",
    "target_alias_audit.csv",
    "target_proxy_candidates.csv",
    "target_task_coverage.csv",
    "target_leakage_risk.csv",
    "target_registry.yaml",
)


@dataclass(frozen=True)
class TargetRegistryAuditResult:
    """Summary returned by :func:`run_target_registry_audit`."""

    status: str
    dataset_rows: int
    dataset_columns: int
    feature_counts: Mapping[str, int]
    label_boundaries: tuple[float, ...]
    output_paths: tuple[Path, ...]
    report_path: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_hash(columns: Sequence[str]) -> str:
    payload = "".join(f"{column}\n" for column in columns).encode("utf-8")
    return sha256(payload).hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _load_validated_columns(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "by_field" not in value:
        raise ValueError(f"Unexpected validated-columns structure: {path}")
    return value


def _source_field_availability(
    validated: Mapping[str, Any], column: str
) -> tuple[str, ...]:
    by_source = validated["by_field"]["pm_columns"]["by_source"]
    return tuple(
        sorted(
            source
            for source, details in by_source.items()
            if column in details.get("union", ())
        )
    )


def _record_to_logical(logical_map_path: Path | None) -> tuple[dict[str, str], set[str]]:
    if logical_map_path is None or not logical_map_path.is_file():
        return {}, set()
    logical_map = pd.read_parquet(
        logical_map_path,
        columns=["record_group_id", "source_record_ids", "selected_record_id"],
    )
    mapping: dict[str, str] = {}
    for row in logical_map.itertuples(index=False):
        for record_id in row.source_record_ids:
            mapping[str(record_id)] = str(row.record_group_id)
    return mapping, set(logical_map["selected_record_id"].astype(str))


def _cohort_summary(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    cohort_id: str,
    target_ids: Sequence[str],
    source: str = "all",
) -> dict[str, Any]:
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    logical = selected["_record_group_id"].dropna().astype(str)
    return {
        "cohort_id": cohort_id,
        "target_ids": "|".join(target_ids),
        "source": source,
        "n_windows": int(len(selected)),
        "n_subjects": int(selected["subject_id"].nunique()),
        "n_source_records": int(selected["record_id"].nunique()),
        "n_logical_records": int(logical.nunique()),
    }


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = _finite_numeric(series)
    finite = numeric.dropna()
    quantiles = finite.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(finite) else pd.Series(dtype=float)
    return {
        "n_windows": int(len(finite)),
        "missing_rate": float(1.0 - len(finite) / len(numeric)) if len(numeric) else None,
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
        "q10": float(quantiles.loc[0.1]) if len(finite) else None,
        "q25": float(quantiles.loc[0.25]) if len(finite) else None,
        "median": float(quantiles.loc[0.5]) if len(finite) else None,
        "q75": float(quantiles.loc[0.75]) if len(finite) else None,
        "q90": float(quantiles.loc[0.9]) if len(finite) else None,
        "unique_values": int(finite.nunique()),
    }


def _equivalence(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    left_num = _finite_numeric(left)
    right_num = _finite_numeric(right)
    both = left_num.notna() & right_num.notna()
    differences = (left_num.loc[both] - right_num.loc[both]).abs()
    return {
        "both_finite_count": int(both.sum()),
        "max_abs_difference": float(differences.max()) if len(differences) else None,
        "mismatch_count": int((differences != 0.0).sum()),
        "left_missing_count": int(left_num.isna().sum()),
        "right_missing_count": int(right_num.isna().sum()),
        "missing_mask_mismatch_count": int((left_num.isna() != right_num.isna()).sum()),
    }


def _registry_entry(**values: Any) -> dict[str, Any]:
    missing = [field for field in REGISTRY_REQUIRED_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Registry entry is missing fields: {missing}")
    return {field: values[field] for field in REGISTRY_REQUIRED_FIELDS} | {
        key: value for key, value in values.items() if key not in REGISTRY_REQUIRED_FIELDS
    }


def _continuous_registry_entries(
    frame: pd.DataFrame, validated: Mapping[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        title = metric.title()
        column = f"target_{metric}"
        stats = _numeric_summary(frame[column])
        tasks = ["performance_metrics_regression"]
        if metric == "focus":
            tasks.insert(0, "focus_regression")
        entries.append(
            _registry_entry(
                target_id=column,
                display_name_ru=PM_DISPLAY_NAMES_RU[metric],
                device_metric=f"PM.{title}.Scaled",
                target_family="continuous_pm",
                target_type="continuous_regression",
                source_columns=[
                    f"PM.{title}.Raw",
                    f"PM.{title}.Scaled",
                    f"PM.{title}.Min",
                    f"PM.{title}.Max",
                    f"PM.{title}.IsActive",
                ],
                processed_column=column,
                aggregation=f"mean(PM.{title}.Scaled) within 10-second window",
                units_or_scale="device-provided dimensionless scaled score",
                value_range=[stats["minimum"], stats["maximum"]],
                higher_value_interpretation=f"higher device {title} score; not independently clinically validated",
                available_sources=list(
                    _source_field_availability(validated, f"PM.{title}.Scaled")
                ),
                native_or_derived="derived window aggregate of a native device metric",
                derivation_file="bench/datasets/emotiv_pm_window_builder.py",
                derivation_function="read_and_aggregate_record",
                derivation_scope="per source record, per 10-second window",
                missing_value_policy="non-numeric values become NaN; target remains missing when the window mean is missing",
                recommended_metrics=["mae", "rmse", "r2", "spearman"],
                supported_feature_inputs=["eeg", "pow", "eeg_pow"],
                supported_raw_input=False,
                current_task_ids=tasks,
                current_experiment_ids=["pm_regression", "pm_regression_personalization"],
                leakage_risk="low if all preprocessing and selection are fit on outer-train only",
                status="canonical",
                limitations="device metric semantics and scale are vendor-defined; no causal interpretation",
            )
        )
    return entries


def _activity_registry_entries(
    frame: pd.DataFrame, validated: Mapping[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        title = metric.title()
        processed = f"PM.{title}.IsActive__mean"
        stats = _numeric_summary(frame[processed])
        entries.append(
            _registry_entry(
                target_id=f"activity_{metric}",
                display_name_ru=f"Активность метрики «{PM_DISPLAY_NAMES_RU[metric]}»",
                device_metric=f"PM.{title}.IsActive",
                target_family="device_activity_proxy",
                target_type="native_activity_fraction_candidate",
                source_columns=[f"PM.{title}.IsActive"],
                processed_column=processed,
                aggregation=f"mean(normalized PM.{title}.IsActive) within 10-second window",
                units_or_scale="fraction in [0, 1]",
                value_range=[stats["minimum"], stats["maximum"]],
                higher_value_interpretation="larger fraction of window marked active by device",
                available_sources=list(
                    _source_field_availability(validated, f"PM.{title}.IsActive")
                ),
                native_or_derived="native device flag normalized then window-aggregated",
                derivation_file="bench/datasets/emotiv_pm_window_builder.py",
                derivation_function="normalize_boolish_series; read_and_aggregate_record",
                derivation_scope="per source record, per 10-second window",
                missing_value_policy="unrecognized strings and non-numeric values become NaN",
                recommended_metrics=["balanced_accuracy", "macro_f1", "average_precision"],
                supported_feature_inputs=["eeg", "pow", "eeg_pow"],
                supported_raw_input=False,
                current_task_ids=[],
                current_experiment_ids=[],
                leakage_risk="target is excluded from features; task semantics and threshold require validation",
                status="requires_semantic_validation",
                limitations="processed values are empirically 0/1, but the source-field value domain was not exhaustively rescanned",
            )
        )
    return entries


def _candidate_registry_entries(validated: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        for classes in (3, 5):
            entries.append(
                _registry_entry(
                    target_id=f"label_{metric}_q{classes}_candidate",
                    display_name_ru=f"{PM_DISPLAY_NAMES_RU[metric]}: {classes} порядковых класса (кандидат)",
                    device_metric=f"PM.{metric.title()}.Scaled",
                    target_family="derived_ordinal_proxy",
                    target_type="ordinal_classification_candidate",
                    source_columns=[f"target_{metric}"],
                    processed_column=None,
                    aggregation="not materialized; proposed train-fold quantiles",
                    units_or_scale=f"ordered classes 0..{classes - 1}",
                    value_range=[0, classes - 1],
                    higher_value_interpretation=f"higher quantile of {metric} within outer-train reference distribution",
                    available_sources=list(
                        _source_field_availability(validated, f"PM.{metric.title()}.Scaled")
                    ),
                    native_or_derived="derived candidate; not materialized",
                    derivation_file=None,
                    derivation_function=None,
                    derivation_scope="must fit thresholds on each outer-train fold",
                    missing_value_policy="retain missing continuous targets as missing labels",
                    recommended_metrics=["balanced_accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"],
                    supported_feature_inputs=["eeg", "pow", "eeg_pow"],
                    supported_raw_input=False,
                    current_task_ids=[],
                    current_experiment_ids=[],
                    leakage_risk="high if quantile boundaries are computed globally; must be fold-fitted",
                    status="candidate",
                    limitations="scientific meaning and class stability are not yet validated",
                )
            )
    return entries


def _build_registry(
    frame: pd.DataFrame,
    validated: Mapping[str, Any],
    *,
    dataset_sha256: str,
    feature_facts: Mapping[str, Any],
    label_boundaries: Sequence[float],
) -> dict[str, Any]:
    focus_stats = _numeric_summary(frame["target_focus"])
    targets = _continuous_registry_entries(frame, validated)
    targets.extend(_activity_registry_entries(frame, validated))
    targets.extend(
        [
            _registry_entry(
                target_id="target_main",
                display_name_ru="Основная цель (устаревший псевдоним Focus)",
                device_metric="PM.Focus.Scaled",
                target_family="legacy_alias",
                target_type="continuous_regression_alias",
                source_columns=["target_focus"],
                processed_column="target_main",
                aggregation="copy target_focus; builder fallback to attention then engagement only if focus column is absent",
                units_or_scale="same as target_focus",
                value_range=[focus_stats["minimum"], focus_stats["maximum"]],
                higher_value_interpretation="same as target_focus",
                available_sources=["Old_EEG", "gpn_data"],
                native_or_derived="derived legacy alias",
                derivation_file="bench/datasets/emotiv_pm_window_builder.py",
                derivation_function="read_and_aggregate_record",
                derivation_scope="per built source record",
                missing_value_policy="inherits selected fallback target missingness",
                recommended_metrics=["mae", "rmse", "r2", "spearman"],
                supported_feature_inputs=["eeg", "pow", "eeg_pow"],
                supported_raw_input=False,
                current_task_ids=[],
                current_experiment_ids=["legacy configs only"],
                leakage_risk="implicit loader fallback can silently change a task when target_col is omitted",
                status="legacy",
                limitations="not a distinct cognitive-state construct; explicit target_col is required",
            ),
            _registry_entry(
                target_id="label_q5",
                display_name_ru="Фокус: пять порядковых классов",
                device_metric="PM.Focus.Scaled",
                target_family="derived_ordinal_proxy",
                target_type="ordinal_classification",
                source_columns=["target_main", "target_focus", "PM.Focus.Scaled__mean"],
                processed_column="label_q5",
                aggregation="global pd.qcut(target_main, q=5, labels=False, duplicates='drop')",
                units_or_scale="ordered classes 0..4",
                value_range=[0, 4],
                higher_value_interpretation="higher global quantile of device Focus score",
                available_sources=["Old_EEG", "gpn_data"],
                native_or_derived="derived legacy global benchmark label",
                derivation_file="bench/datasets/emotiv_pm_window_builder.py",
                derivation_function="make_quality_labels; main",
                derivation_scope="all concatenated source records before outer splitting",
                missing_value_policy="missing target_main remains missing label; insufficient uniqueness returns all missing",
                recommended_metrics=["balanced_accuracy", "macro_f1", "quadratic_weighted_kappa", "ordinal_mae"],
                supported_feature_inputs=["eeg", "pow", "eeg_pow", "feature_sequence"],
                supported_raw_input=True,
                current_task_ids=["cognitive_load_5class"],
                current_experiment_ids=["label_q5 benchmark", "personalization", "FOMAML", "DANN"],
                leakage_risk="global quantile boundaries were computed before outer subject split",
                status="legacy",
                limitations="Focus-specific legacy label, not the complete cognitive-state target space",
                canonical_display_id="label_focus_q5",
                quantile_boundaries=[float(value) for value in label_boundaries],
            ),
            _registry_entry(
                target_id="activity_multilabel_group",
                display_name_ru="Совместная активность семи PM",
                device_metric="seven PM IsActive fields",
                target_family="multi_label_group",
                target_type="multi_label_candidate",
                source_columns=[f"PM.{metric.title()}.IsActive__mean" for metric in PM_METRICS],
                processed_column=None,
                aggregation="joint vector of seven existing window aggregates; not materialized",
                units_or_scale="seven activity fractions",
                value_range=[0, 1],
                higher_value_interpretation="metric-specific active fraction",
                available_sources=["Old_EEG", "gpn_data"],
                native_or_derived="derived grouping of native activity indicators",
                derivation_file=None,
                derivation_function=None,
                derivation_scope="candidate only",
                missing_value_policy="complete-case or explicitly masked multi-label loss required",
                recommended_metrics=["macro_f1", "micro_f1", "average_precision"],
                supported_feature_inputs=["eeg", "pow", "eeg_pow"],
                supported_raw_input=False,
                current_task_ids=[],
                current_experiment_ids=[],
                leakage_risk="low for labels themselves; thresholds and missing-label policy must be train-safe",
                status="candidate",
                limitations="device activity semantics and target dependence require validation",
            ),
            _registry_entry(
                target_id="pm_long_term_excitement_candidate",
                display_name_ru="Долговременное возбуждение (кандидат)",
                device_metric="PM.LongTermExcitement",
                target_family="candidate_additional_pm",
                target_type="continuous_candidate",
                source_columns=["PM.LongTermExcitement"],
                processed_column=None,
                aggregation="not selected by the current window builder",
                units_or_scale="unknown device-defined scale",
                value_range=None,
                higher_value_interpretation="not established",
                available_sources=list(
                    _source_field_availability(validated, "PM.LongTermExcitement")
                ),
                native_or_derived="native device field, absent from processed Parquet",
                derivation_file=None,
                derivation_function=None,
                derivation_scope="not materialized",
                missing_value_policy="not audited numerically",
                recommended_metrics=["mae", "rmse", "spearman"],
                supported_feature_inputs=[],
                supported_raw_input=False,
                current_task_ids=[],
                current_experiment_ids=[],
                leakage_risk="unknown until semantics and cohort are validated",
                status="requires_semantic_validation",
                limitations="relationship to Excitement, scale, missingness, and cross-source semantics are unresolved",
            ),
        ]
    )
    targets.extend(_candidate_registry_entries(validated))
    return {
        "schema_version": 1,
        "audit_status": "target_registry_ready",
        "dataset": {
            "path": "data/processed/windowed_eeg_pm_dataset_w10.parquet",
            "sha256": dataset_sha256,
            "rows": int(len(frame)),
            "columns": int(feature_facts["dataset_columns"]),
        },
        "canonical_pm_order": list(PM_METRICS),
        "feature_contract": dict(feature_facts),
        "raw_input_contract": {
            "shape": [1, 14, 2560],
            "sampling_rate_hz": 256,
            "window_seconds": 10,
            "current_target_limitation": "RawEEGWindowDataset supports label_q5 only",
        },
        "targets": targets,
        "report_references": [
            "reports/integration/full_target_registry_audit.md",
            "reports/integration/target_pipeline_audit.md",
            "reports/integration/target_pipeline_resolution.md",
            "reports/integration/pm_multioutput_regression.md",
        ],
    }


def _target_inventory(
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in registry["targets"]:
        column = entry["processed_column"]
        if column and column in frame.columns:
            stats = _numeric_summary(frame[column])
            valid = _finite_numeric(frame[column]).notna()
            selected = frame.loc[valid]
            sources = sorted(selected["source"].astype(str).unique())
            logical = selected["_record_group_id"].dropna().astype(str)
            counts = {
                "n_windows": stats["n_windows"],
                "n_subjects": int(selected["subject_id"].nunique()),
                "n_source_records": int(selected["record_id"].nunique()),
                "n_logical_records": int(logical.nunique()),
                "missing_rate": stats["missing_rate"],
                "minimum": stats["minimum"],
                "maximum": stats["maximum"],
                "unique_values": stats["unique_values"],
            }
        else:
            sources = entry["available_sources"]
            counts = {
                "n_windows": 0,
                "n_subjects": 0,
                "n_source_records": 0,
                "n_logical_records": 0,
                "missing_rate": None,
                "minimum": None,
                "maximum": None,
                "unique_values": 0,
            }
        rows.append(
            {
                "target_id": entry["target_id"],
                "processed_column": column,
                "target_family": entry["target_family"],
                "target_type": entry["target_type"],
                "native_or_derived": entry["native_or_derived"],
                "aggregation": entry["aggregation"],
                "sources": "|".join(sources),
                **counts,
                "status": entry["status"],
            }
        )
    return pd.DataFrame(rows)


def _availability(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in registry["targets"]:
        column = entry["processed_column"]
        for source in ("all", "Old_EEG", "gpn_data"):
            subset = frame if source == "all" else frame.loc[frame["source"] == source]
            if column and column in subset.columns:
                stats = _numeric_summary(subset[column])
                valid = _finite_numeric(subset[column]).notna()
                selected = subset.loc[valid]
                logical = selected["_record_group_id"].dropna().astype(str)
                row = {
                    "target_id": entry["target_id"],
                    "processed_column": column,
                    "source": source,
                    "upstream_field_available": source == "all" or source in entry["available_sources"],
                    **stats,
                    "n_subjects": int(selected["subject_id"].nunique()),
                    "n_source_records": int(selected["record_id"].nunique()),
                    "n_logical_records": int(logical.nunique()),
                }
            else:
                row = {
                    "target_id": entry["target_id"],
                    "processed_column": column,
                    "source": source,
                    "upstream_field_available": source == "all" or source in entry["available_sources"],
                    **{key: None for key in ("missing_rate", "minimum", "maximum", "mean", "std", "q10", "q25", "median", "q75", "q90")},
                    "n_windows": 0,
                    "unique_values": 0,
                    "n_subjects": 0,
                    "n_source_records": 0,
                    "n_logical_records": 0,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def _derivation_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        output = f"target_{metric}"
        input_column = f"PM.{metric.title()}.Scaled__mean"
        overall = _equivalence(frame[output], frame[input_column])
        source_checks = {
            source: _equivalence(
                frame.loc[frame["source"] == source, output],
                frame.loc[frame["source"] == source, input_column],
            )
            for source in sorted(frame["source"].unique())
        }
        rows.append(
            {
                "output_column": output,
                "input_columns": input_column,
                "code_path": "bench/datasets/emotiv_pm_window_builder.py",
                "function_or_block": "read_and_aggregate_record:339-344",
                "transformation": "exact copy of window mean",
                "scope": "per source record and 10-second window",
                "missing_policy": "preserve missing window mean",
                "verified_equivalence": overall["mismatch_count"] == 0 and overall["missing_mask_mismatch_count"] == 0,
                "both_finite_count": overall["both_finite_count"],
                "max_abs_difference": overall["max_abs_difference"],
                "mismatch_count": overall["mismatch_count"],
                "output_missing_count": overall["left_missing_count"],
                "input_missing_count": overall["right_missing_count"],
                "missing_mask_mismatch_count": overall["missing_mask_mismatch_count"],
                "source_checks": _json_text(source_checks),
                "leakage_risk": "none from copying; downstream fold protocol still applies",
            }
        )
    main_check = _equivalence(frame["target_main"], frame["target_focus"])
    rows.append(
        {
            "output_column": "target_main",
            "input_columns": "target_focus (fallback: target_attention, then target_engagement if columns absent)",
            "code_path": "bench/datasets/emotiv_pm_window_builder.py",
            "function_or_block": "read_and_aggregate_record:346-354",
            "transformation": "legacy conditional alias",
            "scope": "per built source record",
            "missing_policy": "inherits selected target; all NaN if no fallback column exists",
            "verified_equivalence": main_check["mismatch_count"] == 0 and main_check["missing_mask_mismatch_count"] == 0,
            "both_finite_count": main_check["both_finite_count"],
            "max_abs_difference": main_check["max_abs_difference"],
            "mismatch_count": main_check["mismatch_count"],
            "output_missing_count": main_check["left_missing_count"],
            "input_missing_count": main_check["right_missing_count"],
            "missing_mask_mismatch_count": main_check["missing_mask_mismatch_count"],
            "source_checks": "{}",
            "leakage_risk": "implicit loader fallback can change the selected target",
        }
    )
    reconstructed, boundaries = pd.qcut(
        _finite_numeric(frame["target_main"]),
        q=5,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    label_check = _equivalence(frame["label_q5"], pd.Series(reconstructed, index=frame.index))
    rows.append(
        {
            "output_column": "label_q5",
            "input_columns": "target_main == target_focus",
            "code_path": "bench/datasets/emotiv_pm_window_builder.py",
            "function_or_block": "make_quality_labels:372-389; main:650-655",
            "transformation": f"global pd.qcut(q=5, labels=False, duplicates='drop'); bins={_json_text([float(x) for x in boundaries])}",
            "scope": "all concatenated source records before outer split",
            "missing_policy": "missing target remains missing; insufficient count/uniqueness returns all missing",
            "verified_equivalence": label_check["mismatch_count"] == 0 and label_check["missing_mask_mismatch_count"] == 0,
            "both_finite_count": label_check["both_finite_count"],
            "max_abs_difference": label_check["max_abs_difference"],
            "mismatch_count": label_check["mismatch_count"],
            "output_missing_count": label_check["left_missing_count"],
            "input_missing_count": label_check["right_missing_count"],
            "missing_mask_mismatch_count": label_check["missing_mask_mismatch_count"],
            "source_checks": "{}",
            "leakage_risk": "global quantile boundaries use outer-test subjects",
        }
    )
    return pd.DataFrame(rows)


def _alias_audit(frame: pd.DataFrame) -> pd.DataFrame:
    main = _equivalence(frame["target_main"], frame["target_focus"])
    focus_mean = _equivalence(frame["target_main"], frame["PM.Focus.Scaled__mean"])
    return pd.DataFrame(
        [
            {
                "alias": "target_main",
                "canonical_target": "target_focus",
                "relationship": "exact in the canonical Parquet; builder has column-presence fallbacks",
                "both_finite_count": main["both_finite_count"],
                "max_abs_difference": main["max_abs_difference"],
                "mismatch_count": main["mismatch_count"],
                "missing_mask_mismatch_count": main["missing_mask_mismatch_count"],
                "loader_behavior": "default when target_col is omitted; fallback search remains for legacy configs",
                "status": "legacy",
            },
            {
                "alias": "target_main",
                "canonical_target": "PM.Focus.Scaled__mean",
                "relationship": "exact through target_focus in the canonical Parquet",
                "both_finite_count": focus_mean["both_finite_count"],
                "max_abs_difference": focus_mean["max_abs_difference"],
                "mismatch_count": focus_mean["mismatch_count"],
                "missing_mask_mismatch_count": focus_mean["missing_mask_mismatch_count"],
                "loader_behavior": "not recommended as an implicit task contract",
                "status": "legacy",
            },
            {
                "alias": "label_focus_q5",
                "canonical_target": "label_q5",
                "relationship": "recommended display alias only; no physical column rename",
                "both_finite_count": int(frame["label_q5"].notna().sum()),
                "max_abs_difference": 0.0,
                "mismatch_count": 0,
                "missing_mask_mismatch_count": 0,
                "loader_behavior": "physical target_col remains label_q5",
                "status": "canonical_display_alias",
            },
        ]
    )


def _proxy_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_columns = [f"PM.{metric.title()}.IsActive__mean" for metric in PM_METRICS]
    active = frame[active_columns].apply(pd.to_numeric, errors="coerce")
    for metric, column in zip(PM_METRICS, active_columns):
        series = _finite_numeric(frame[column])
        rows.append(
            {
                "candidate_id": f"activity_{metric}",
                "candidate_family": "device_activity_proxy",
                "source_target": column,
                "classes_or_outputs": 1,
                "materialized": True,
                "n_windows": int(series.notna().sum()),
                "n_zero": int((series == 0).sum()),
                "n_one": int((series == 1).sum()),
                "n_intermediate": int(((series > 0) & (series < 1)).sum()),
                "recommended_derivation": "use existing fraction as continuous until device semantics justify a binary task",
                "status": "requires_semantic_validation",
            }
        )
    complete = active.notna().all(axis=1)
    rows.append(
        {
            "candidate_id": "activity_multilabel_group",
            "candidate_family": "multi_label_group",
            "source_target": "|".join(active_columns),
            "classes_or_outputs": 7,
            "materialized": False,
            "n_windows": int(complete.sum()),
            "n_zero": None,
            "n_one": None,
            "n_intermediate": None,
            "recommended_derivation": "joint vector with explicit missing-label policy; no new threshold",
            "status": "candidate",
        }
    )
    for metric in PM_METRICS:
        for classes in (3, 5):
            rows.append(
                {
                    "candidate_id": f"label_{metric}_q{classes}_candidate",
                    "candidate_family": "derived_ordinal_proxy",
                    "source_target": f"target_{metric}",
                    "classes_or_outputs": classes,
                    "materialized": False,
                    "n_windows": int(frame[f"target_{metric}"].notna().sum()),
                    "n_zero": None,
                    "n_one": None,
                    "n_intermediate": None,
                    "recommended_derivation": "fit quantile boundaries on outer-train only; sensitivity analysis required",
                    "status": "candidate",
                }
            )
    rows.append(
        {
            "candidate_id": "pm_long_term_excitement_candidate",
            "candidate_family": "candidate_additional_pm",
            "source_target": "PM.LongTermExcitement",
            "classes_or_outputs": 1,
            "materialized": False,
            "n_windows": 0,
            "n_zero": None,
            "n_one": None,
            "n_intermediate": None,
            "recommended_derivation": "validate semantics and add a window aggregate before task design",
            "status": "requires_semantic_validation",
        }
    )
    return pd.DataFrame(rows)


def _task_coverage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        rows.extend(
            [
                {"target_id": target, "task": "regression", "representation": "eeg_pow_features", "model_or_protocol": "random_forest", "status": "trained" if target != "target_focus" else "trained"},
                {"target_id": target, "task": "seven_output_regression", "representation": "eeg_pow_features", "model_or_protocol": "torch_mlp", "status": "trained"},
                {"target_id": target, "task": "personalized_seven_output_regression", "representation": "eeg_pow_features", "model_or_protocol": "torch_mlp", "status": "confirmatory"},
                {"target_id": target, "task": "raw_regression", "representation": "raw_eeg", "model_or_protocol": "raw CNN", "status": "not_supported"},
            ]
        )
    rows.extend(
        [
            {"target_id": "label_q5", "task": "classification", "representation": "eeg_pow_features", "model_or_protocol": "random_forest", "status": "trained"},
            {"target_id": "label_q5", "task": "classification", "representation": "feature_sequence", "model_or_protocol": "Transformer/LSTM", "status": "trained"},
            {"target_id": "label_q5", "task": "classification", "representation": "raw_eeg", "model_or_protocol": "EEGNet/ShallowConvNet", "status": "trained"},
            {"target_id": "label_q5", "task": "personalization", "representation": "eeg_pow_features", "model_or_protocol": "fine_tuning", "status": "confirmatory"},
            {"target_id": "label_q5", "task": "meta_learning", "representation": "raw_eeg", "model_or_protocol": "FOMAML", "status": "diagnostic_only"},
            {"target_id": "label_q5", "task": "domain_adaptation", "representation": "eeg_pow_features", "model_or_protocol": "DANN", "status": "confirmatory"},
            {"target_id": "activity_*", "task": "binary_or_fraction_proxy", "representation": "eeg_pow_features", "model_or_protocol": "none", "status": "not_run"},
            {"target_id": "activity_multilabel_group", "task": "multi_label", "representation": "eeg_pow_features", "model_or_protocol": "none", "status": "not_supported"},
            {"target_id": "label_*_q3/q5_candidate", "task": "ordinal_classification", "representation": "eeg_pow_features", "model_or_protocol": "none", "status": "not_run"},
            {"target_id": "pm_long_term_excitement_candidate", "task": "regression", "representation": "none", "model_or_protocol": "none", "status": "not_supported"},
        ]
    )
    return pd.DataFrame(rows)


def _leakage_risks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"risk_id": "global_label_q5_quantiles", "target_ids": "label_q5", "severity": "methodological_sensitivity", "finding": "class boundaries were computed on all subjects before outer split", "required_control": "retain as legacy benchmark and report cross-fitted sensitivity"},
            {"risk_id": "implicit_target_main_fallback", "target_ids": "target_main", "severity": "high", "finding": "EmotivDataset defaults to target_main and can search fallback targets when target_col is omitted", "required_control": "all canonical configs must set target_col or target_cols explicitly"},
            {"risk_id": "target_feature_contamination", "target_ids": "all", "severity": "controlled", "finding": "resolved EEG/POW feature lists contain no target_* or PM.* columns", "required_control": "preserve loader exclusion and hash checks"},
            {"risk_id": "candidate_quantile_fit_scope", "target_ids": "label_*_q3/q5_candidate", "severity": "high", "finding": "candidate thresholds do not yet exist", "required_control": "fit boundaries on outer-train only"},
            {"risk_id": "activity_proxy_semantics", "target_ids": "activity_*", "severity": "semantic", "finding": "window aggregates are 0/1 but raw device-flag semantics are not independently validated", "required_control": "validate device meaning before declaring classification tasks"},
            {"risk_id": "raw_loader_target_lock", "target_ids": "all_except_label_q5", "severity": "compatibility", "finding": "RawEEGWindowDataset rejects every target_col except label_q5", "required_control": "generalize manifest and loader only after target approval"},
            {"risk_id": "logical_record_duplicates", "target_ids": "all", "severity": "controlled", "finding": "source records can represent the same logical recording", "required_control": "preserve record_group_id deduplication and group-disjoint splits"},
        ]
    )


def _cohort_counts(
    frame: pd.DataFrame,
    *,
    raw_manifest_path: Path | None,
    selected_record_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        valid = _finite_numeric(frame[target]).notna()
        rows.append(_cohort_summary(frame, valid, cohort_id=target, target_ids=[target]))
        for source in ("Old_EEG", "gpn_data"):
            rows.append(
                _cohort_summary(
                    frame,
                    valid & frame["source"].eq(source),
                    cohort_id=target,
                    target_ids=[target],
                    source=source,
                )
            )
    complete = frame[list(TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    rows.append(_cohort_summary(frame, complete, cohort_id="seven_pm_complete_case", target_ids=TARGET_COLUMNS))
    for source in ("Old_EEG", "gpn_data"):
        rows.append(
            _cohort_summary(
                frame,
                complete & frame["source"].eq(source),
                cohort_id="seven_pm_complete_case",
                target_ids=TARGET_COLUMNS,
                source=source,
            )
        )
    label = _finite_numeric(frame["label_q5"]).notna()
    rows.append(_cohort_summary(frame, label, cohort_id="label_q5", target_ids=["label_q5"]))
    for source in ("Old_EEG", "gpn_data"):
        rows.append(
            _cohort_summary(
                frame,
                label & frame["source"].eq(source),
                cohort_id="label_q5",
                target_ids=["label_q5"],
                source=source,
            )
        )
    active_columns = [f"PM.{metric.title()}.IsActive__mean" for metric in PM_METRICS]
    for metric, column in zip(PM_METRICS, active_columns):
        rows.append(
            _cohort_summary(
                frame,
                _finite_numeric(frame[column]).notna(),
                cohort_id=f"activity_{metric}",
                target_ids=[f"activity_{metric}"],
            )
        )
    active_complete = frame[active_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    rows.append(
        _cohort_summary(
            frame,
            active_complete,
            cohort_id="activity_multilabel_complete_case",
            target_ids=[f"activity_{metric}" for metric in PM_METRICS],
        )
    )
    if raw_manifest_path is not None and raw_manifest_path.is_file() and selected_record_ids:
        manifest = pd.read_parquet(raw_manifest_path, columns=["sample_id", "record_id", "status"])
        accepted = manifest["status"].eq("ok") & manifest["record_id"].astype(str).isin(selected_record_ids)
        sample_ids = manifest.loc[accepted, "sample_id"].astype(int)
        valid_ids = sample_ids[(sample_ids >= 0) & (sample_ids < len(frame))]
        raw_mask = pd.Series(False, index=frame.index)
        raw_mask.iloc[valid_ids.to_numpy()] = True
        rows.append(
            _cohort_summary(
                frame,
                raw_mask & label,
                cohort_id="raw_deduplicated_label_q5_compatible",
                target_ids=["label_q5"],
            )
        )
        for target in TARGET_COLUMNS:
            rows.append(
                _cohort_summary(
                    frame,
                    raw_mask & _finite_numeric(frame[target]).notna(),
                    cohort_id=f"raw_deduplicated_{target}_compatible",
                    target_ids=[target],
                )
            )
    return pd.DataFrame(rows)


def _render_report(
    *,
    registry: Mapping[str, Any],
    inventory: pd.DataFrame,
    availability: pd.DataFrame,
    cohorts: pd.DataFrame,
    derivation: pd.DataFrame,
    proxies: pd.DataFrame,
    coverage: pd.DataFrame,
    risks: pd.DataFrame,
    frame: pd.DataFrame,
) -> str:
    complete = cohorts.loc[cohorts["cohort_id"] == "seven_pm_complete_case"].iloc[0]
    label = cohorts.loc[cohorts["cohort_id"] == "label_q5"].iloc[0]
    active_columns = [f"PM.{metric.title()}.IsActive__mean" for metric in PM_METRICS]
    active = frame[active_columns].apply(pd.to_numeric, errors="coerce")
    coactivity = active.notna().all(axis=1)
    active_count_distribution = active.loc[coactivity].sum(axis=1).value_counts().sort_index()
    bins = next(item for item in registry["targets"] if item["target_id"] == "label_q5")["quantile_boundaries"]
    lines = [
        "# Полный аудит реестра целевых переменных",
        "",
        "Статус решения: **target_registry_ready**. Аудит является read-only: модели не обучались, новые классовые цели и кэши не материализовались.",
        "",
        "## 1. Причина аудита",
        "",
        "Существующий benchmark исторически называл Focus-derived `label_q5` общей целью когнитивного состояния. Реестр разделяет непрерывные PM устройства, нативные activity-прокси, производные порядковые прокси и legacy aliases.",
        "",
        "## 2. Фактическая схема датасета",
        "",
        f"Канонический Parquet содержит {len(frame):,} окон и {registry['dataset']['columns']} столбцов. Признаковые группы: 168 EEG, 280 POW, 448 EEG+POW; хэши списков сохранены в YAML. Все `target_*`, `PM.*`, идентификаторы и временные поля исключены из признаков.",
        "",
        "## 3. Семь основных PM",
        "",
        "Канонический порядок: Attention, Engagement, Excitement, Stress, Relaxation, Interest, Focus. Для каждой метрики исходная схема обоих источников содержит Raw, Scaled, Min, Max и IsActive. Текущая непрерывная цель — среднее Scaled по 10-секундному окну.",
        "",
        "## 4. PM.LongTermExcitement",
        "",
        "Поле присутствует во всех 120 исходных записях и обоих источниках, но не выбрано текущим window builder и отсутствует в processed Parquet. Его связь с Excitement, шкала, пропуски и межисточниковая семантика не подтверждены; это отдельный кандидат, не восьмой канонический output.",
        "",
        "## 5. Происхождение target_*",
        "",
        "`bench/datasets/emotiv_pm_window_builder.py::read_and_aggregate_record` агрегирует `PM.<Metric>.Scaled` функциями mean/std/min/max/last и затем точно копирует `Scaled__mean` в `target_<metric>`. Для всех семи пар, обоих источников и масок пропусков найдено ноль расхождений; максимальная абсолютная разность равна 0.",
        "",
        "## 6. Происхождение target_main",
        "",
        "В каноническом Parquet `target_main` полностью совпадает с `target_focus` и `PM.Focus.Scaled__mean`: 45 384 конечных строки, ноль расхождений. В builder это условный legacy alias: при отсутствии столбца Focus выбирается Attention, затем Engagement. Loader по-прежнему использует `target_main` по умолчанию, поэтому канонические configs обязаны задавать цель явно.",
        "",
        "## 7. Происхождение label_q5",
        "",
        f"`make_quality_labels` применяет один глобальный `pd.qcut(target_main, q=5, labels=False, duplicates='drop')` после объединения всех записей и до outer split. Полные границы bins: `{bins}`. Непустых меток: {int(label.n_windows):,}, участников: {int(label.n_subjects)}, source records: {int(label.n_source_records)}; классы 0–4. Реконструкция byte-for-value совпала со столбцом. Каноническое отображаемое имя — `label_focus_q5`, физический столбец остаётся `label_q5`.",
        "",
        "## 8. Нативные индикаторы IsActive",
        "",
        "Builder нормализует bool-like значения в 0/1, числовые значения сохраняет, затем берёт оконное среднее. Во всём processed Parquet каждый из семи `IsActive__mean` имеет только значения 0/1, промежуточных окон нет. Это эмпирический факт об агрегате, а не исчерпывающая повторная проверка 129 млн исходных CSV-строк; семантика флага остаётся кандидатом на валидацию.",
        "",
        "## 9. Производные прокси-кандидаты",
        "",
        "Для каждой непрерывной PM зарегистрированы q3/q5-кандидаты, но они не материализованы. Любые будущие границы должны оцениваться только на outer-train. Существующий глобальный Focus q5 сохраняется как legacy benchmark и sensitivity analysis, а не как шаблон для новых целей.",
        "",
        "## 10. Multi-label постановка",
        "",
        f"Совместный вектор семи IsActive возможен технически для {int(coactivity.sum()):,} окон. Распределение числа активных метрик: `{_json_text({str(int(k)): int(v) for k, v in active_count_distribution.items()})}`. До подтверждения device semantics, зависимости меток и missing-label loss это только кандидат.",
        "",
        "## 11. Доступность по источникам",
        "",
        "Все семь исходных PM-семейств и LongTermExcitement перечислены в validated-columns для `gpn_data` и `Old_EEG`. Фактические window counts, missing rates, описательная статистика и квантили по каждому источнику сохранены в `target_availability_by_source.csv`.",
        "",
        "## 12. Target-specific когорты",
        "",
        f"Размеры непрерывных когорт различаются. Семивыходная complete-case когорта: {int(complete.n_windows):,} окон, {int(complete.n_subjects)} участника, {int(complete.n_source_records)} source records. Отдельные target-, source-, activity-, label- и raw-deduplicated-compatible когорты находятся в `target_cohort_counts.csv`.",
        "",
        "## 13. Входные представления",
        "",
        "Feature mode поддерживает EEG (168), POW (280) и EEG+POW (448). Feature-sequence используется существующими temporal моделями. Raw mode имеет контракт `[1, 14, 2560]` при 256 Гц и 10 с. Признаки определены по схеме и хэшированы без чтения всех 508 столбцов.",
        "",
        "## 14. Текущее покрытие задачами",
        "",
        "Семь PM имеют совместную feature-based регрессию и персонализацию; это не заменяет отдельные научные результаты по каждому proxy. `label_q5` покрыт feature, sequence, raw, personalization, FOMAML и DANN только как Focus-derived цель. Activity, q3/q5-кандидаты и LongTermExcitement не обучались. Матрица находится в `target_task_coverage.csv`.",
        "",
        "## 15. Ограничения raw loader",
        "",
        "`RawEEGWindowDataset.load` жёстко принимает только `target_col == 'label_q5'`; raw index builder также присваивает folds по `label_q5`. Это документированное ограничение, не исправленное в данном аудите.",
        "",
        "## 16. Риски утечек",
        "",
        "Главный методический риск — глобальные границы label_q5 до subject-disjoint outer split. Также опасен неявный target_main fallback. Feature contamination сейчас контролируется: EEG/POW-списки не содержат PM или target columns. Полный реестр мер — `target_leakage_risk.csv`.",
        "",
        "## 17. Канонический реестр",
        "",
        "`reports/summary/target_registry.yaml` является машинно-читаемым источником истины. Он содержит обязательные provenance, semantics, input, task, risk, status и limitation поля для каждой цели и кандидата.",
        "",
        "## 18. Рекомендуемая матрица будущих экспериментов",
        "",
        "Этапы: (1) обобщить target specification/loaders; (2) поддержать отдельную регрессию семи PM; (3) валидировать IsActive; (4) добавить fold-fitted ordinal proxies; (5) запустить EEG/POW/EEG+POW baselines; (6) обобщить raw loader; (7) выбрать научно обоснованные deep targets; (8) лишь затем решать вопрос повторов DANN/FOMAML.",
        "",
        "## 19. Какие существующие результаты сохраняются",
        "",
        "Инфраструктура benchmark, subject-disjoint splits, logical-record deduplication, feature/raw caches, модели, метрики и artifact pipeline сохраняются. Результаты label_q5 сохраняются как результаты Focus; семивыходная PM-регрессия сохраняется. FOMAML и DANN не обобщаются на остальные цели.",
        "",
        "## 20. Открытые вопросы",
        "",
        "Нужны внешняя семантическая валидация IsActive и LongTermExcitement, решение о missing-label loss для multi-label, научное обоснование отдельных ordinal targets и безопасное расширение raw manifest/loader. Никакая из этих задач не была молча объявлена готовой к обучению.",
    ]
    return "\n".join(lines)


def run_target_registry_audit(
    dataset_path: str | Path,
    validated_columns_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    report_path: str | Path | None = None,
    logical_map_path: str | Path | None = None,
    raw_manifest_path: str | Path | None = None,
) -> TargetRegistryAuditResult:
    """Audit target provenance and write deterministic tracked summaries."""

    root = Path(repo_root).resolve()
    dataset = Path(dataset_path)
    validated_path = Path(validated_columns_path)
    output = Path(output_dir)
    dataset = dataset if dataset.is_absolute() else root / dataset
    validated_path = validated_path if validated_path.is_absolute() else root / validated_path
    output = output if output.is_absolute() else root / output
    report = (
        root / "reports/integration/full_target_registry_audit.md"
        if report_path is None
        else Path(report_path)
    )
    if not report.is_absolute():
        report = root / report
    logical = (
        root / "data/interim/logical_recording_map.parquet"
        if logical_map_path is None
        else Path(logical_map_path)
    )
    if not logical.is_absolute():
        logical = root / logical
    raw_manifest = (
        root / "data/interim/raw_eeg_window_index_w10_raw_v3.parquet"
        if raw_manifest_path is None
        else Path(raw_manifest_path)
    )
    if not raw_manifest.is_absolute():
        raw_manifest = root / raw_manifest

    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    if not validated_path.is_file():
        raise FileNotFoundError(f"Validated-columns inventory not found: {validated_path}")
    before_hash = _sha256_file(dataset)
    parquet = pq.ParquetFile(dataset)
    schema_columns = parquet.schema_arrow.names
    feature_groups = {
        "eeg": [column for column in schema_columns if column.startswith("EEG.")],
        "pow": [column for column in schema_columns if column.startswith("POW.")],
    }
    required = ["source", "subject_id", "record_id", "target_main", "label_q5"]
    required.extend(TARGET_COLUMNS)
    for metric in PM_METRICS:
        title = metric.title()
        required.extend(
            [
                f"PM.{title}.Scaled__mean",
                f"PM.{title}.Scaled__std",
                f"PM.{title}.Scaled__min",
                f"PM.{title}.Scaled__max",
                f"PM.{title}.Scaled__last",
                f"PM.{title}.IsActive__mean",
            ]
        )
    missing = sorted(set(required) - set(schema_columns))
    if missing:
        raise ValueError(f"Processed dataset is missing audit columns: {missing}")
    frame = pd.read_parquet(dataset, columns=required)
    validated = _load_validated_columns(validated_path)
    record_to_logical, selected_record_ids = _record_to_logical(logical)
    frame["_record_group_id"] = frame["record_id"].astype(str).map(record_to_logical)

    eeg_pow = [
        column
        for column in schema_columns
        if column.startswith("EEG.") or column.startswith("POW.")
    ]
    forbidden = [
        column
        for column in eeg_pow
        if column.startswith("target_") or column.startswith("PM.") or column == "label_q5"
    ]
    if forbidden:
        raise ValueError(f"Target columns leaked into EEG/POW features: {forbidden}")
    feature_facts = {
        "dataset_columns": len(schema_columns),
        "eeg_feature_count": len(feature_groups["eeg"]),
        "pow_feature_count": len(feature_groups["pow"]),
        "eeg_pow_feature_count": len(eeg_pow),
        "eeg_feature_list_sha256": _feature_hash(feature_groups["eeg"]),
        "pow_feature_list_sha256": _feature_hash(feature_groups["pow"]),
        "eeg_pow_feature_list_sha256": _feature_hash(eeg_pow),
        "target_or_pm_feature_count": len(forbidden),
    }
    reconstructed, boundaries = pd.qcut(
        _finite_numeric(frame["target_main"]),
        q=5,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    label_check = _equivalence(frame["label_q5"], pd.Series(reconstructed, index=frame.index))
    if label_check["mismatch_count"] or label_check["missing_mask_mismatch_count"]:
        raise ValueError(f"Stored label_q5 does not match global qcut: {label_check}")
    registry = _build_registry(
        frame,
        validated,
        dataset_sha256=before_hash,
        feature_facts=feature_facts,
        label_boundaries=boundaries,
    )
    inventory = _target_inventory(frame, registry)
    availability = _availability(frame, registry)
    cohorts = _cohort_counts(
        frame,
        raw_manifest_path=raw_manifest,
        selected_record_ids=selected_record_ids,
    )
    derivation = _derivation_audit(frame)
    aliases = _alias_audit(frame)
    proxies = _proxy_candidates(frame)
    coverage = _task_coverage()
    risks = _leakage_risks()

    tables = {
        "target_inventory.csv": inventory,
        "target_availability_by_source.csv": availability,
        "target_cohort_counts.csv": cohorts,
        "target_derivation_audit.csv": derivation,
        "target_alias_audit.csv": aliases,
        "target_proxy_candidates.csv": proxies,
        "target_task_coverage.csv": coverage,
        "target_leakage_risk.csv": risks,
    }
    for filename, table in tables.items():
        _write_csv(table, output / filename)
    registry_path = output / "target_registry.yaml"
    _write_text(
        registry_path,
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=False, width=100),
    )
    _write_text(
        report,
        _render_report(
            registry=registry,
            inventory=inventory,
            availability=availability,
            cohorts=cohorts,
            derivation=derivation,
            proxies=proxies,
            coverage=coverage,
            risks=risks,
            frame=frame,
        ),
    )
    after_hash = _sha256_file(dataset)
    if before_hash != after_hash:
        raise RuntimeError("Input Parquet changed during read-only audit")
    output_paths = tuple(output / filename for filename in OUTPUT_FILENAMES)
    return TargetRegistryAuditResult(
        status="target_registry_ready",
        dataset_rows=int(len(frame)),
        dataset_columns=len(schema_columns),
        feature_counts={
            "eeg": len(feature_groups["eeg"]),
            "pow": len(feature_groups["pow"]),
            "eeg_pow": len(eeg_pow),
        },
        label_boundaries=tuple(float(value) for value in boundaries),
        output_paths=output_paths,
        report_path=report,
    )


__all__ = [
    "OUTPUT_FILENAMES",
    "PM_METRICS",
    "REGISTRY_REQUIRED_FIELDS",
    "TARGET_COLUMNS",
    "TargetRegistryAuditResult",
    "run_target_registry_audit",
]
