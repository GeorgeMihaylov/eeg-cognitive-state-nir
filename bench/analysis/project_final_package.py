"""Deterministic consolidation of existing project results.

This module never trains a model or rebuilds a cache.  It reads the curated
experiment/requirement registries and completed runtime artifacts, then writes
small publication-facing tables, SVG figures, and Markdown audits.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PACKAGE_DATE = "2026-08-29"
LAG_CLASSIFICATION_PROTOCOL_HASH = (
    "064fe752a541e753f53a1463d2749823b37c16045d559316ceaa05a0d5ab283e"
)
LAG_REGRESSION_PROTOCOL_HASH = (
    "96b99b28533af365aa15b1a0464ce151ddbc34a51bac45645e4103acecfeb026"
)
PM_TARGET_ORDER = (
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
)
FINAL_STATUSES = {
    "completed",
    "diagnostic",
    "infrastructure_only",
    "not_started",
    "closed_negative",
    "superseded",
}
REQUIREMENT_STATUSES = {
    "closed",
    "partially_closed",
    "infrastructure_ready",
    "not_required_for_article",
    "open",
    "blocked",
}
INVENTORY_COLUMNS = [
    "experiment_id",
    "research_question",
    "dataset",
    "task",
    "target",
    "model",
    "feature_or_channel_policy",
    "preprocessing",
    "outer_protocol",
    "inner_protocol",
    "seeds",
    "status",
    "primary_metrics",
    "artifact_path",
    "report_path",
    "commit",
    "scientific_decision",
]
PROVENANCE_COLUMNS = [
    "experiment_id",
    "status",
    "resolved_config",
    "dataset_or_cache_hash",
    "split_hash",
    "seed",
    "folds",
    "metrics",
    "predictions_if_expected",
    "report",
    "commit_or_revision",
    "leakage_audit",
    "complete",
    "missing",
    "evidence_role",
]
STATUS_COLUMNS = [
    "experiment_id",
    "task_id",
    "method",
    "dataset",
    "data_mode",
    "folds",
    "seeds",
    "analysis_level",
    "status",
    "primary_metric",
    "primary_result",
    "decision",
    "limitations",
    "protocol_hash",
    "preregistration_hash",
    "result_artifact",
]


class FinalPackageError(ValueError):
    """Raised when consolidation inputs violate the publication contract."""


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalPackageError(f"Expected mapping in {path.as_posix()}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalPackageError(f"Expected object in {path.as_posix()}")
    return value


def _relative(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).replace("\\", "/")
    if (
        Path(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or re.match(r"^[A-Za-z]:/", text)
    ):
        raise FinalPackageError(f"Absolute path is forbidden: {text}")
    return text


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return stream.getvalue()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM keeps Cyrillic labels intact in Excel/LibreOffice imports while
    # remaining readable through Python's ``utf-8-sig`` codec.
    path.write_text(
        "\ufeff" + _csv_text(rows, columns), encoding="utf-8", newline=""
    )


def _metric_lookup(repo_root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in (
        "classification_metrics_unified.csv",
        "pm_regression_metrics_unified.csv",
        "personalization_metrics_unified.csv",
    ):
        path = repo_root / "reports" / "summary" / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            experiment_id = row["experiment_id"]
            values: list[str] = []
            for key in (
                "primary_metric",
                "primary_value",
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "macro_mae_mean",
                "macro_r2_mean",
                "method",
                "metric_after",
                "absolute_gain",
            ):
                if row.get(key) not in (None, ""):
                    values.append(f"{key}={row[key]}")
            if values:
                lookup.setdefault(experiment_id, "; ".join(values))
    return lookup


def _infer_dataset(item: Mapping[str, Any]) -> str:
    joined = " ".join(
        str(item.get(key, ""))
        for key in ("experiment_id", "title", "tags", "runtime_path")
    ).lower()
    if "cog_bci" in joined or "cog-bci" in joined:
        return "COG-BCI"
    if item.get("task") in {"mixin_audit", "encoder_contract", "domain_adaptation_contract",
                            "contrastive_pretraining_contract"}:
        return "synthetic contract / not applicable"
    return "gpn_data + Old_EEG"


def _status(item: Mapping[str, Any]) -> str:
    explicit = item.get("consolidation_status")
    if explicit:
        value = str(explicit)
    else:
        value = {
            "final": "completed",
            "baseline": "completed",
            "diagnostic": "diagnostic",
            "smoke": "diagnostic",
            "invalidated": "superseded",
        }.get(str(item.get("status")), "not_started")
        if item.get("category") in {"mixin", "infrastructure"}:
            value = "infrastructure_only"
    if value not in FINAL_STATUSES:
        raise FinalPackageError(f"Unknown final status {value!r}")
    return value


def build_inventory(repo_root: Path) -> list[dict[str, Any]]:
    """Build the canonical inventory from the existing experiment registry."""
    registry = _load_yaml(repo_root / "reports/summary/experiment_registry.yaml")
    experiments = registry.get("experiments", [])
    if not isinstance(experiments, list):
        raise FinalPackageError("experiment registry experiments must be a list")
    ids = [str(item.get("experiment_id", "")) for item in experiments]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise FinalPackageError(f"Duplicate experiment IDs: {duplicates}")
    metrics = _metric_lookup(repo_root)
    rows: list[dict[str, Any]] = []
    for item in sorted(experiments, key=lambda value: str(value["experiment_id"])):
        experiment_id = str(item["experiment_id"])
        primary = item.get("primary_metric", {})
        primary_text = metrics.get(experiment_id)
        if not primary_text:
            primary_text = str(primary.get("name", "not_applicable"))
        row = {
            "experiment_id": experiment_id,
            "research_question": item.get("title", ""),
            "dataset": _infer_dataset(item),
            "task": item.get("task", ""),
            "target": item.get("target") or "not_applicable",
            "model": item.get("model", ""),
            "feature_or_channel_policy": item.get("feature_set", ""),
            "preprocessing": item.get("preprocessing", ""),
            "outer_protocol": item.get("evaluation_protocol", ""),
            "inner_protocol": item.get(
                "inner_protocol",
                "group-aware validation where applicable"
                if str(item.get("model", "")).startswith("torch")
                else "not_applicable",
            ),
            "seeds": "|".join(str(value) for value in item.get("seeds", [])),
            "status": _status(item),
            "primary_metrics": primary_text,
            "artifact_path": _relative(item.get("runtime_path")),
            "report_path": _relative(item.get("report_path")),
            "commit": item.get("commit", ""),
            "scientific_decision": item.get(
                "scientific_decision", item.get("result_summary", "")
            ),
        }
        rows.append(row)
    return rows


def _artifact_has(root: Path, patterns: Iterable[str]) -> bool:
    if not root.exists():
        return False
    for pattern in patterns:
        if any(root.glob(pattern)):
            return True
    return False


def _evidence_text(paths: Iterable[Path], *, limit: int = 80) -> str:
    selected: list[Path] = []
    patterns = (
        "*summary*.json",
        "*manifest*.json",
        "*audit*.json",
        "*decision*.json",
        "run_summary.json",
    )
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            selected.append(root)
            continue
        for pattern in patterns:
            selected.extend(sorted(root.glob(pattern)))
            selected.extend(sorted(root.glob(f"**/{pattern}")))
    chunks: list[str] = []
    for path in sorted(set(selected), key=lambda value: value.as_posix())[:limit]:
        if path.stat().st_size <= 8_000_000:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore").lower())
    return "\n".join(chunks)


def build_provenance_audit(
    repo_root: Path, inventory: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Check publication provenance without changing runtime artifacts."""
    main_hash_reports = {
        "reports/ordinal_transformer_full_seed42_summary.json",
        "reports/label_target_audit_summary.json",
        "README.md",
    }
    rows: list[dict[str, Any]] = []
    for item in inventory:
        status = str(item["status"])
        artifact = repo_root / str(item["artifact_path"]) if item["artifact_path"] else None
        report = repo_root / str(item["report_path"]) if item["report_path"] else None
        infrastructure = status == "infrastructure_only"
        config = False
        registry_item = next(
            value
            for value in _load_yaml(
                repo_root / "reports/summary/experiment_registry.yaml"
            )["experiments"]
            if value["experiment_id"] == item["experiment_id"]
        )
        config_path = registry_item.get("config_path") or registry_item.get(
            "source_config_path"
        )
        if config_path:
            config = (repo_root / _relative(config_path)).exists()
        elif infrastructure:
            config = True
        report_text = (
            report.read_text(encoding="utf-8", errors="ignore").lower()
            if report and report.exists()
            else ""
        )
        artifact_exists = bool(artifact and artifact.exists())
        supplemental: list[Path] = []
        experiment_id = str(item["experiment_id"])
        if "ordinal_transformer_multiseed" in experiment_id:
            supplemental.extend(
                [
                    repo_root / "reports/ordinal_transformer_multiseed_runs_summary.json",
                    repo_root / "reports/ordinal_transformer_multiseed_summary.json",
                ]
            )
        if experiment_id == "label_q5_auxiliary_corn_policy":
            supplemental.extend(
                [
                    repo_root
                    / "reports/auxiliary_corn_nested_lambda_finalized_results.json",
                    repo_root
                    / "reports/auxiliary_corn_policy_subject_analysis_summary.json",
                ]
            )
        cog_split = repo_root / (
            "benchmark_results/cog_bci_protocols/nback_3class/protocol_summary.json"
        )
        if str(item["dataset"]) == "COG-BCI" and cog_split.exists():
            supplemental.append(cog_split)
        evidence_text = report_text + "\n" + _evidence_text(
            [path for path in (artifact, *supplemental) if path is not None]
        )
        hash_evidence = (
            infrastructure
            or any((repo_root / path).exists() for path in main_hash_reports)
            and _infer_dataset(registry_item) == "gpn_data + Old_EEG"
            or any(
                token in evidence_text
                for token in (
                    "sha256",
                    "cache_hash",
                    "cache hash",
                    "dataset_hash",
                    "config_hash",
                    "config hash",
                    "source_parquet_sha256",
                )
            )
        )
        split_evidence = (
            infrastructure
            or any(
                token in evidence_text
                for token in (
                    "split hash",
                    "split_hash",
                    "fixed-fold hash",
                    "fixed_fold_hash",
                    "sequence_index_sha256",
                    "subject_assignment_hash",
                    "window_bounds_hash",
                    "split_manifest_sha256",
                )
            )
            or (
                artifact_exists
                and _artifact_has(
                    artifact,
                    (
                        "**/validation_split.json",
                        "**/split_manifest*.json",
                        "**/split_audit*.csv",
                        "**/run_manifest.json",
                    ),
                )
            )
        )
        metrics = (
            infrastructure
            or bool(item["primary_metrics"])
            or (
                artifact_exists
                and _artifact_has(
                    artifact,
                    ("**/metrics.json", "**/*metrics*.csv", "**/*summary*.json"),
                )
            )
        )
        predictions_expected = (
            registry_item.get("category") != "preprocessing"
            and status in {"completed", "closed_negative"}
            and any(
            token in str(item["outer_protocol"]).lower()
            for token in ("fold", "groupkfold", "downstream")
            )
        )
        prediction_roots = [artifact] if artifact else []
        if experiment_id == "label_q5_auxiliary_corn_policy":
            prediction_roots.append(
                repo_root / "benchmark_results/auxiliary_corn_nested_lambda"
            )
        predictions = (
            not predictions_expected
            or any(
                root is not None
                and root.exists()
                and _artifact_has(
                    root, ("**/predictions.parquet", "**/*predictions*.parquet")
                )
                for root in prediction_roots
            )
        )
        leakage = (
            infrastructure
            or any(
                token in evidence_text
                for token in ("overlap", "leakage", "exact_match", "leakage_safe")
            )
            or (
                artifact_exists
                and _artifact_has(
                    artifact,
                    ("**/leakage_audit.json", "**/split_audit*.csv",
                     "**/validation_split.json"),
                )
            )
        )
        checks = {
            "resolved_config": config,
            "dataset_or_cache_hash": hash_evidence,
            "split_hash": split_evidence,
            "seed": infrastructure or bool(item["seeds"]),
            "folds": infrastructure
            or any(
                token in str(item["outer_protocol"]).lower()
                for token in ("fold", "audit", "source", "screening", "contract")
            ),
            "metrics": metrics,
            "predictions_if_expected": predictions,
            "report": bool(report and report.exists()),
            "commit_or_revision": bool(item["commit"]),
            "leakage_audit": leakage,
        }
        missing = [key for key, value in checks.items() if not value]
        complete = not missing
        evidence_role = (
            "primary"
            if complete and status == "completed"
            else "negative_result"
            if complete and status == "closed_negative"
            else "supporting_only"
        )
        rows.append(
            {
                "experiment_id": item["experiment_id"],
                "status": status,
                **{key: str(value).lower() for key, value in checks.items()},
                "complete": str(complete).lower(),
                "missing": "|".join(missing),
                "evidence_role": evidence_role,
            }
        )
    return rows


def build_dataset_characteristics() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "gpn_data + Old_EEG aggregate",
            "analysis_unit": "window",
            "rows_or_windows": 51308,
            "subjects": 55,
            "records": 120,
            "channels": 14,
            "features": 448,
            "sampling_rate_hz": "",
            "window_seconds": 10,
            "target": "label_q5 / seven PM targets",
            "notes": "Before supervised target filtering; 168 EEG + 280 POW features.",
        },
        {
            "dataset": "label_q5 supervised",
            "analysis_unit": "window",
            "rows_or_windows": 45384,
            "subjects": 54,
            "records": 119,
            "channels": 14,
            "features": 448,
            "sampling_rate_hz": "",
            "window_seconds": 10,
            "target": "label_q5 (5 classes)",
            "notes": "Rows with missing target removed.",
        },
        {
            "dataset": "seven-target PM regression",
            "analysis_unit": "window",
            "rows_or_windows": 43174,
            "subjects": 53,
            "records": "",
            "channels": 14,
            "features": 448,
            "sampling_rate_hz": "",
            "window_seconds": 10,
            "target": "7 continuous PM targets",
            "notes": "Complete-case target rows.",
        },
        {
            "dataset": "raw EEG deduplicated",
            "analysis_unit": "window",
            "rows_or_windows": 30958,
            "subjects": 54,
            "records": "",
            "channels": 14,
            "features": "",
            "sampling_rate_hz": 256,
            "window_seconds": 10,
            "target": "label_q5",
            "notes": "Input [1,14,2560]; logical-record deduplication.",
        },
        {
            "dataset": "COG-BCI corpus",
            "analysis_unit": "record",
            "rows_or_windows": 1044,
            "subjects": 29,
            "records": 1044,
            "channels": "62/63",
            "features": "",
            "sampling_rate_hz": 500,
            "window_seconds": "",
            "target": "N-Back / MATB-II protocol",
            "notes": "Three sessions; ECG1 excluded from EEG policies.",
        },
        {
            "dataset": "COG-BCI emotiv_common cache",
            "analysis_unit": "window",
            "rows_or_windows": 56903,
            "subjects": 29,
            "records": 1044,
            "channels": 14,
            "features": "",
            "sampling_rate_hz": 500,
            "window_seconds": 5.12,
            "target": "protocol-derived",
            "notes": "Record-safe native-time cache.",
        },
        {
            "dataset": "COG-BCI time-aligned cache",
            "analysis_unit": "window",
            "rows_or_windows": 28910,
            "subjects": 29,
            "records": 1044,
            "channels": 14,
            "features": "",
            "sampling_rate_hz": 256,
            "window_seconds": 10,
            "target": "contrastive pretraining",
            "notes": "Whole-record polyphase 64/125 resampling.",
        },
    ]


def build_benchmark_models() -> list[dict[str, Any]]:
    models = [
        ("Random Forest", "aggregated EEG/POW", "window", "classification/regression"),
        ("sklearn MLP", "aggregated EEG/POW", "window", "classification"),
        ("Torch MLP", "aggregated EEG/POW", "window", "classification/regression"),
        ("LSTM", "aggregated EEG/POW", "sequence", "classification"),
        ("BiLSTM", "aggregated EEG/POW", "sequence", "classification"),
        ("Transformer", "aggregated EEG/POW", "sequence", "categorical/ordinal"),
        ("EEGNet", "raw EEG", "window", "classification/encoder"),
        ("ShallowConvNet", "raw EEG", "window", "classification"),
        ("Mean regressor", "target-only train mean", "window", "regression reference"),
        ("Spectral logistic/HGB", "bandpower features", "record", "COG-BCI diagnostic"),
    ]
    return [
        {
            "model": model,
            "input_representation": input_type,
            "analysis_unit": unit,
            "supported_task": task,
        }
        for model, input_type, unit, task in models
    ]


def build_lag_alignment_summary(repo_root: Path) -> dict[str, Any]:
    """Load and validate the fixed previous-window confirmatory evidence."""
    classification_dir = repo_root / "reports/diagnostics/pm_eeg_lag_confirmatory_v1"
    regression_dir = (
        repo_root / "reports/diagnostics/pm_eeg_lag_regression_confirmatory_v1"
    )
    classification_protocol = _load_json(classification_dir / "protocol.json")
    regression_protocol = _load_json(regression_dir / "protocol.json")
    classification_dry_run = _load_json(classification_dir / "dry_run_summary.json")
    regression_dry_run = _load_json(regression_dir / "dry_run_summary.json")

    expected_protocols = (
        (
            classification_protocol,
            "pm_eeg_lag_confirmatory_371_xgboost_v1",
            LAG_CLASSIFICATION_PROTOCOL_HASH,
        ),
        (
            regression_protocol,
            "pm_eeg_lag_regression_confirmatory_371_xgboost_v1",
            LAG_REGRESSION_PROTOCOL_HASH,
        ),
    )
    for protocol, experiment_id, protocol_hash in expected_protocols:
        if protocol.get("experiment_id") != experiment_id:
            raise FinalPackageError(f"Unexpected lag experiment ID: {protocol.get('experiment_id')}")
        if protocol.get("protocol_hash") != protocol_hash:
            raise FinalPackageError(f"Unexpected protocol hash for {experiment_id}")
        if protocol.get("result_status") != "confirmatory_complete":
            raise FinalPackageError(f"Lag result is not complete: {experiment_id}")
        if protocol.get("candidate_lags_seconds") != [0, -10]:
            raise FinalPackageError(f"Unexpected lag candidates for {experiment_id}")
        if protocol.get("feature_count") != 371 or protocol.get("seed") != 42:
            raise FinalPackageError(f"Unexpected feature/seed contract for {experiment_id}")

    if classification_protocol.get("fixed_fold_hash") != regression_protocol.get(
        "fixed_fold_hash"
    ):
        raise FinalPackageError("Classification and regression lag folds differ")
    if tuple(regression_protocol.get("target_ids", ())) != PM_TARGET_ORDER:
        raise FinalPackageError("Regression lag target order differs from canonical PM order")

    invariant_fields = (
        "identical_fold_membership_between_conditions",
        "identical_subject_ids_between_conditions",
        "identical_target_ids_between_conditions",
    )
    for dry_run in (classification_dry_run, regression_dry_run):
        if any(dry_run.get(field) is not True for field in invariant_fields):
            raise FinalPackageError("Lag comparison cohort invariants failed")
        if any(dry_run.get(field) != 0 for field in (
            "cross_fold_pairs", "cross_record_pairs", "cross_subject_pairs"
        )):
            raise FinalPackageError("Lag comparison contains cross-boundary pairs")
    if regression_dry_run.get("identical_train_test_counts_between_conditions") is not True:
        raise FinalPackageError("Regression train/test counts differ between lag conditions")

    classification_pooled = pd.read_csv(
        classification_dir / "pooled_summary.csv"
    ).iloc[0]
    regression_pooled = pd.read_csv(regression_dir / "pooled_summary.csv").iloc[0]
    regression_by_pm = pd.read_csv(regression_dir / "summary_by_pm.csv")
    regression_paired = pd.read_csv(regression_dir / "paired_delta_by_fold.csv")
    if len(regression_paired) != 35 or int(regression_pooled["n_fold_pm_pairs"]) != 35:
        raise FinalPackageError("Expected 35 paired fold-by-PM regression comparisons")

    lag0_mae = float(regression_paired["participant_macro_mae_lag0"].mean())
    lag10_mae = float(
        regression_paired["participant_macro_mae_lag_minus_10s"].mean()
    )
    lag0_pearson = float(
        regression_paired["participant_macro_pearson_lag0"].mean()
    )
    lag10_pearson = float(
        regression_paired["participant_macro_pearson_lag_minus_10s"].mean()
    )

    per_pm: list[dict[str, Any]] = []
    for target_id in PM_TARGET_ORDER:
        selected = regression_by_pm[regression_by_pm["target_id"] == target_id]
        if len(selected) != 1:
            raise FinalPackageError(f"Expected one lag summary row for {target_id}")
        row = selected.iloc[0]
        paired = regression_paired[regression_paired["target_id"] == target_id]
        per_pm.append(
            {
                "pm": str(row["pm"]),
                "target_id": target_id,
                "mae_relative_reduction_percent": (
                    -float(row["delta_mae_mean"]) / float(row["lag0_mae_mean"]) * 100
                ),
                "delta_pearson": float(row["delta_pearson_mean"]),
                "mae_favorable_folds": int((paired["delta_mae"] < 0).sum()),
                "pearson_favorable_folds": int((paired["delta_pearson"] > 0).sum()),
                "median_delta_r2": float(paired["delta_r2"].median()),
            }
        )

    return {
        "fixed_fold_hash": str(regression_protocol["fixed_fold_hash"]),
        "feature_count": 371,
        "seed": 42,
        "classification": {
            "experiment_id": classification_protocol["experiment_id"],
            "protocol_hash": classification_protocol["protocol_hash"],
            "execution_commit": classification_protocol["git_commit"],
            "matched_rows": int(classification_protocol["matched_cohort_count"]),
            "subjects": int(classification_dry_run["subjects"]),
            "delta_macro_f1": float(classification_pooled["mean_delta_macro_f1"]),
            "delta_balanced_accuracy": float(
                classification_pooled["mean_delta_balanced_accuracy"]
            ),
            "favorable_fold_pm_macro_f1": int(
                classification_pooled["positive_fold_pm_macro_f1"]
            ),
            "favorable_fold_pm_balanced_accuracy": int(
                classification_pooled["positive_fold_pm_balanced_accuracy"]
            ),
            "favorable_pm_macro_f1": int(
                classification_pooled["positive_pm_mean_macro_f1"]
            ),
            "favorable_pm_balanced_accuracy": int(
                classification_pooled["positive_pm_mean_balanced_accuracy"]
            ),
        },
        "regression": {
            "experiment_id": regression_protocol["experiment_id"],
            "protocol_hash": regression_protocol["protocol_hash"],
            "execution_commit": regression_protocol["git_commit"],
            "lag0_mae": lag0_mae,
            "lag_minus_10s_mae": lag10_mae,
            "delta_mae": float(regression_pooled["mean_delta_mae"]),
            "relative_mae_reduction_percent": (lag0_mae - lag10_mae) / lag0_mae * 100,
            "favorable_fold_pm_mae": int(regression_pooled["favorable_fold_pm_mae"]),
            "favorable_pm_mean_mae": int(regression_pooled["favorable_pm_mean_mae"]),
            "lag0_pearson": lag0_pearson,
            "lag_minus_10s_pearson": lag10_pearson,
            "delta_pearson": float(regression_pooled["mean_delta_pearson"]),
            "favorable_fold_pm_pearson": int(
                regression_pooled["favorable_fold_pm_pearson"]
            ),
            "favorable_pm_mean_pearson": int(
                regression_pooled["favorable_pm_mean_pearson"]
            ),
            "median_delta_r2": float(regression_pooled["median_delta_r2"]),
            "favorable_fold_pm_r2": int(regression_pooled["favorable_fold_pm_r2"]),
            "positive_pm_median_r2": sum(row["median_delta_r2"] > 0 for row in per_pm),
        },
        "per_pm": per_pm,
        "pairing": {
            "matched_rows": int(regression_dry_run["temporal_pairing"]["matched_target_rows"]),
            "subjects": int(regression_dry_run["temporal_pairing"]["subjects"]),
            "records": int(regression_dry_run["temporal_pairing"]["records"]),
            "first_window_losses": int(
                regression_dry_run["temporal_pairing"]["first_window_losses"]
            ),
            "gap_losses": int(
                regression_dry_run["temporal_pairing"]["additional_gap_losses"]
            ),
        },
    }


def build_classification_results(repo_root: Path) -> list[dict[str, Any]]:
    """Add the fixed lag comparison to the canonical classification table."""
    path = repo_root / "reports/summary/classification_metrics_unified.csv"
    rows = pd.read_csv(path).to_dict("records")
    experiment_id = "pm_eeg_lag_confirmatory_371_xgboost_v1"
    rows = [row for row in rows if row["experiment_id"] != experiment_id]
    lag = build_lag_alignment_summary(repo_root)["classification"]
    row = {column: "" for column in rows[0]}
    row.update(
        {
            "experiment_id": experiment_id,
            "result_status": "final",
            "model": "XGBoost lag comparison",
            "model_family": "classical_ml",
            "input_type": "feature_window",
            "feature_set": "canonical_cogstate_371",
            "preprocessing": "none; fixed temporal alignment contract",
            "evaluation_protocol": "fixed 5-fold subject-disjoint matched lag 0 vs -10 s",
            "n_folds": 5,
            "seeds": "42",
            "n_subjects": 54,
            "n_samples": lag["matched_rows"],
            "primary_metric": "participant_macro_f1_delta",
            "primary_value": lag["delta_macro_f1"],
            "report_path": "reports/diagnostics/pm_eeg_lag_final_conclusion.md",
            "config_path": "experiments/pm_diagnostics/pm_eeg_lag_confirmatory_v1.json",
            "commit": lag["execution_commit"],
            "metric_source": (
                "structured_csv:reports/diagnostics/pm_eeg_lag_confirmatory_v1/"
                "pooled_summary.csv"
            ),
            "notes": (
                "Confirmatory fixed previous-window comparison; 35/35 fold-PM "
                f"Macro-F1 and balanced-accuracy deltas positive; delta balanced accuracy "
                f"{lag['delta_balanced_accuracy']:+.12f}; supports EEG(t-10s)->PM(t) "
                "as an experimental alignment contract."
            ),
        }
    )
    rows.append(row)
    return rows


def build_regression_results(repo_root: Path) -> list[dict[str, Any]]:
    path = (
        repo_root
        / "benchmark_results/pm_regression_baseline_5fold/20260724_121853"
        / "emotiv_pm_regression/performance_metrics_regression/random_forest"
        / "group_kfold_subject/per_target_metrics.csv"
    )
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for target, group in frame.groupby("target_name", sort=True):
        rows.append(
            {
                "model": "Random Forest",
                "target": target,
                "analysis_unit": "window",
                "folds": 5,
                "seed": 42,
                "mae_mean": group["mae"].mean(),
                "mae_std": group["mae"].std(ddof=0),
                "r2_mean": group["r2"].mean(),
                "r2_std": group["r2"].std(ddof=0),
                "pearson_mean": group["pearson"].mean(),
                "pearson_std": group["pearson"].std(ddof=0),
            }
        )
    lag_frame = pd.read_csv(
        repo_root
        / "reports/diagnostics/pm_eeg_lag_regression_confirmatory_v1/summary_by_pm.csv"
    )
    for _, source in lag_frame.iterrows():
        for condition, label in (("lag0", "lag=0 s"), ("lag_minus_10s", "lag=-10 s")):
            rows.append(
                {
                    "model": f"XGBRegressor [{label}]",
                    "target": source["target_id"],
                    "analysis_unit": "participant_macro",
                    "folds": int(source["n_folds"]),
                    "seed": 42,
                    "mae_mean": source[f"{condition}_mae_mean"],
                    "mae_std": source[f"{condition}_mae_std"],
                    "r2_mean": source[f"{condition}_r2_mean"],
                    "r2_std": source[f"{condition}_r2_std"],
                    "pearson_mean": source[f"{condition}_pearson_mean"],
                    "pearson_std": source[f"{condition}_pearson_std"],
                }
            )
    return rows


def build_ordinal_results(repo_root: Path) -> list[dict[str, Any]]:
    pure = _load_json(repo_root / "reports/ordinal_transformer_multiseed_summary.json")
    policy = _load_json(
        repo_root / "reports/auxiliary_corn_policy_subject_analysis_summary.json"
    )
    rows: list[dict[str, Any]] = []
    for source, allowed in ((pure, {"categorical", "corn"}), (policy, {"policy"})):
        frame = pd.DataFrame(source["aggregate_metrics_by_seed"])
        frame = frame[
            (frame["feature_group"] == "eeg_pow") & frame["method"].isin(allowed)
        ]
        for method, group in frame.groupby("method", sort=True):
            rows.append(
                {
                    "method": {
                        "categorical": "categorical Transformer",
                        "corn": "pure CORN",
                        "policy": "auxiliary CORN policy",
                    }[method],
                    "feature_set": "EEG+POW sequence",
                    "analysis_unit": "window sequence",
                    "seeds": "7|42|123",
                    "subjects": 53,
                    "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                    "balanced_accuracy_seed_std": group["balanced_accuracy"].std(ddof=0),
                    "macro_f1_mean": group["macro_f1"].mean(),
                    "macro_f1_seed_std": group["macro_f1"].std(ddof=0),
                    "ordinal_mae_mean": group["ordinal_mae"].mean(),
                    "ordinal_mae_seed_std": group["ordinal_mae"].std(ddof=0),
                    "severe_error_rate_mean": group["severe_error_rate"].mean(),
                    "severe_error_rate_seed_std": group["severe_error_rate"].std(
                        ddof=0
                    ),
                    "subject_analysis": "53 subject-level paired units",
                }
            )
    order = {
        "categorical Transformer": 0,
        "pure CORN": 1,
        "auxiliary CORN policy": 2,
    }
    return sorted(rows, key=lambda row: order[row["method"]])


def build_external_results(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, folder in (
        ("EEGNet", "eegnet_seed42"),
        ("ShallowConvNet", "shallowconvnet_seed42"),
    ):
        data = _load_json(
            repo_root
            / f"benchmark_results/cog_bci_baselines/nback_3class/{folder}"
            / "aggregate_metrics.json"
        )
        metrics = data["fold_metrics_mean_std"]["record"]
        rows.append(
            {
                "comparison": "CNN N-Back",
                "method": model,
                "channel_policy": "14 channels",
                "analysis_unit": "record",
                "protocol": "5-fold GroupKFold by subject_id",
                "seed": 42,
                "accuracy": metrics["accuracy"]["mean"],
                "balanced_accuracy": metrics["balanced_accuracy"]["mean"],
                "macro_f1": metrics["macro_f1"]["mean"],
                "reference": "chance=0.333",
            }
        )
    channel = pd.read_csv(
        repo_root
        / "benchmark_results/cog_bci_spectral_benchmark/nback_3class"
        / "channel_policy_comparison.csv"
    )
    pooled = channel[
        (channel["model"] == "multinomial_logistic_regression")
        & (channel["representation"] == "channel_wise")
        & (channel["scope"] == "pooled")
    ].iloc[0]
    for label, column in (("14 channels", "balanced_accuracy_14"),
                          ("62 channels", "balanced_accuracy_62")):
        rows.append(
            {
                "comparison": "spectral channels",
                "method": "multinomial logistic regression",
                "channel_policy": label,
                "analysis_unit": "record",
                "protocol": "5-fold GroupKFold by subject_id",
                "seed": 42,
                "accuracy": "",
                "balanced_accuracy": pooled[column],
                "macro_f1": "",
                "reference": "chance=0.333",
            }
        )
    for comparison, path, modes in (
        (
            "shape-only transfer",
            "benchmark_results/cog_bci_contrastive_transfer/downstream_fold_metrics.csv",
            ("random_init", "head_only", "full_model"),
        ),
        (
            "time-aligned transfer",
            "benchmark_results/cog_bci_time_aligned_transfer/downstream_fold_metrics.csv",
            ("random_init", "shape_only", "time_aligned"),
        ),
    ):
        frame = pd.read_csv(repo_root / path)
        if "level" in frame.columns:
            frame = frame[frame["level"] == "window"]
        for mode in modes:
            row = frame[frame["mode"] == mode].iloc[0]
            rows.append(
                {
                    "comparison": comparison,
                    "method": mode,
                    "channel_policy": "14 channels",
                    "analysis_unit": "window",
                    "protocol": "one protected downstream fold",
                    "seed": 42,
                    "accuracy": row["accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "macro_f1": row["macro_f1"],
                    "reference": "label_q5 chance=0.20",
                }
            )
    return rows


def build_meta_learning_results(repo_root: Path) -> list[dict[str, Any]]:
    """Load participant-level FOMAML diagnostic results from runtime artifacts."""
    runtime = repo_root / "benchmark_results/meta_learning_fomaml_label_q5_raw_diagnostic"
    summary = _load_json(runtime / "diagnostic_summary.json")
    decision = _load_json(runtime / "decision.json")
    paired = _load_json(runtime / "paired_comparison.json")["primary"]
    aggregate = pd.read_csv(runtime / "outer_test_aggregate_metrics.csv")
    aggregate = aggregate[aggregate["aggregation"] == "subject_mean"]
    rows: list[dict[str, Any]] = []
    for _, item in aggregate.iterrows():
        rows.append(
            {
                "experiment_id": summary["experiment_id"],
                "method": item["mode"],
                "analysis_level": "participant",
                "folds": "1",
                "seeds": "42",
                "participants": int(summary["subjects"]["evaluated_outer_test"]),
                "macro_f1": item["macro_f1"],
                "balanced_accuracy": item["balanced_accuracy"],
                "ordinal_mae": item["ordinal_mae"],
                "delta_macro_f1_vs_supervised_full_model": (
                    paired["mean_delta_macro_f1"]
                    if item["mode"] == "selected_fomaml" else ""
                ),
                "delta_balanced_accuracy_vs_supervised_full_model": (
                    paired["mean_delta_balanced_accuracy"]
                    if item["mode"] == "selected_fomaml" else ""
                ),
                "delta_ordinal_mae_vs_supervised_full_model": (
                    paired["mean_delta_ordinal_mae"]
                    if item["mode"] == "selected_fomaml" else ""
                ),
                "macro_f1_wins": paired["macro_f1_wins"] if item["mode"] == "selected_fomaml" else "",
                "macro_f1_losses": paired["macro_f1_losses"] if item["mode"] == "selected_fomaml" else "",
                "macro_f1_ties": paired["macro_f1_ties"] if item["mode"] == "selected_fomaml" else "",
                "status": decision["status"],
                "protocol_hash": summary["protocol_hash"],
                "preregistration_hash": summary["preregistration_hash"],
                "result_artifact": _relative(runtime.relative_to(repo_root)),
            }
        )
    return rows


def build_domain_adaptation_results(repo_root: Path) -> list[dict[str, Any]]:
    """Load diagnostic and confirmatory participant-level DANN summaries."""
    diagnostic_root = repo_root / "benchmark_results/domain_adaptation_dann_raw_diagnostic"
    confirmatory_root = repo_root / "benchmark_results/domain_adaptation_dann_confirmatory_v2"
    diagnostic_summary = _load_json(diagnostic_root / "diagnostic_summary.json")
    diagnostic_decision = _load_json(diagnostic_root / "decision.json")
    diagnostic_paired = _load_json(diagnostic_root / "paired_comparison.json")
    diagnostic = pd.read_csv(diagnostic_root / "target_test_aggregate_metrics.csv")
    confirmatory_summary = _load_json(confirmatory_root / "confirmatory_summary.json")
    confirmatory_decision = _load_json(confirmatory_root / "primary_decision.json")
    confirmatory_paired = _load_json(confirmatory_root / "primary_paired_comparison.json")
    bootstrap = _load_json(confirmatory_root / "primary_bootstrap.json")
    participant = pd.read_csv(confirmatory_root / "primary_participant_metrics.csv")
    rows: list[dict[str, Any]] = []
    for _, item in diagnostic.iterrows():
        rows.append(
            {
                "experiment_id": diagnostic_summary["experiment_id"],
                "analysis_group": "diagnostic",
                "method": item["mode"],
                "analysis_level": "participant",
                "folds": "1",
                "seeds": "42",
                "participants": int(item["subjects"]),
                "accuracy": item["participant_mean_accuracy"],
                "balanced_accuracy": item["participant_mean_balanced_accuracy"],
                "macro_f1": item["participant_mean_macro_f1"],
                "weighted_f1": item["participant_mean_weighted_f1"],
                "kappa": item["participant_mean_kappa"],
                "ordinal_mae": item["participant_mean_ordinal_mae"],
                "quadratic_weighted_kappa": "",
                "delta_balanced_accuracy": diagnostic_paired["mean_delta_balanced_accuracy"] if item["mode"] == "dann" else "",
                "delta_macro_f1": diagnostic_paired["mean_delta_macro_f1"] if item["mode"] == "dann" else "",
                "delta_ordinal_mae": diagnostic_paired["mean_delta_ordinal_mae"] if item["mode"] == "dann" else "",
                "median_delta_macro_f1": diagnostic_paired["median_delta_macro_f1"] if item["mode"] == "dann" else "",
                "participant_win_fraction": diagnostic_paired["macro_f1_wins"] / diagnostic_paired["n_subjects"] if item["mode"] == "dann" else "",
                "bootstrap_95_ci_low": diagnostic_paired["bootstrap_macro_f1_mean_95_ci"][0] if item["mode"] == "dann" else "",
                "bootstrap_95_ci_high": diagnostic_paired["bootstrap_macro_f1_mean_95_ci"][1] if item["mode"] == "dann" else "",
                "status": f"diagnostic_{diagnostic_decision['status']}",
                "protocol_hash": diagnostic_summary["protocol_hash"],
                "preregistration_hash": diagnostic_summary["preregistration_hash"],
                "result_artifact": _relative(diagnostic_root.relative_to(repo_root)),
            }
        )
    absolute_columns = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "kappa": "kappa",
        "ordinal_mae": "ordinal_mae",
        "quadratic_weighted_kappa": "quadratic_weighted_kappa",
    }
    for mode, suffix in (("source_only_matched", "source_only_matched"), ("dann", "dann")):
        row = {
            "experiment_id": confirmatory_summary["experiment_id"],
            "analysis_group": "primary_confirmatory",
            "method": mode,
            "analysis_level": "participant",
            "folds": "1|2|3|4|5",
            "seeds": "123|2026",
            "participants": len(participant),
        }
        for output, stem in absolute_columns.items():
            row[output] = participant[f"{stem}_{suffix}"].mean()
        row.update(
            {
                "delta_balanced_accuracy": confirmatory_paired["mean_delta_balanced_accuracy"] if mode == "dann" else "",
                "delta_macro_f1": confirmatory_paired["mean_delta_macro_f1"] if mode == "dann" else "",
                "delta_ordinal_mae": confirmatory_paired["mean_delta_ordinal_mae"] if mode == "dann" else "",
                "median_delta_macro_f1": confirmatory_paired["median_delta_macro_f1"] if mode == "dann" else "",
                "participant_win_fraction": confirmatory_decision["participant_win_fraction"] if mode == "dann" else "",
                "bootstrap_95_ci_low": bootstrap["mean_95_ci"][0] if mode == "dann" else "",
                "bootstrap_95_ci_high": bootstrap["mean_95_ci"][1] if mode == "dann" else "",
                "status": confirmatory_decision["status"],
                "protocol_hash": confirmatory_summary["protocol_hash"],
                "preregistration_hash": confirmatory_summary["execution_preregistration_hash"],
                "result_artifact": _relative(confirmatory_root.relative_to(repo_root)),
            }
        )
        rows.append(row)
    return rows


def build_domain_adaptation_fold_results(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "benchmark_results/domain_adaptation_dann_confirmatory_v2/primary_fold_metrics.csv"
    rows = pd.read_csv(path).to_dict("records")
    for row in rows:
        row.update({"analysis_group": "primary_confirmatory", "seeds": "123|2026"})
    return rows


def build_domain_adaptation_seed_results(repo_root: Path) -> list[dict[str, Any]]:
    root = repo_root / "benchmark_results/domain_adaptation_dann_confirmatory_v2"
    primary = pd.read_csv(root / "primary_seed_metrics.csv")
    sensitivity = pd.read_csv(root / "secondary_seed_metrics.csv")
    rows: list[dict[str, Any]] = []
    sensitivity_only = sensitivity[sensitivity["seed"] == 42]
    for group, frame in (("primary_confirmatory", primary), ("sensitivity", sensitivity_only)):
        for row in frame.to_dict("records"):
            row["analysis_group"] = group
            row["included_in_primary_decision"] = group == "primary_confirmatory"
            row["diagnostic_fold_1_reused"] = group == "sensitivity"
            rows.append(row)
    return rows


def build_experiment_statuses(repo_root: Path) -> list[dict[str, Any]]:
    registry = _load_yaml(repo_root / "reports/summary/experiment_registry.yaml")
    rows: list[dict[str, Any]] = []
    for item in registry["experiments"]:
        if not str(item.get("task_id", "")).startswith("8"):
            continue
        rows.append(
            {
                "experiment_id": item["experiment_id"],
                "task_id": item["task_id"],
                "method": item["model"],
                "dataset": item.get("dataset", _infer_dataset(item)),
                "data_mode": item["feature_set"],
                "folds": "|".join(str(value) for value in item.get("folds", [])),
                "seeds": "|".join(str(value) for value in item.get("seeds", [])),
                "analysis_level": item["analysis_level"],
                "status": item["stage_status"],
                "primary_metric": item["primary_metric"]["name"],
                "primary_result": item.get("primary_result", ""),
                "decision": item["scientific_decision"],
                "limitations": "|".join(item.get("limitations", [])),
                "protocol_hash": item.get("protocol_hash", ""),
                "preregistration_hash": item.get("preregistration_hash", ""),
                "result_artifact": _relative(item.get("runtime_path")),
            }
        )
    return sorted(rows, key=lambda row: (row["task_id"], row["experiment_id"]))


def build_negative_results() -> list[dict[str, Any]]:
    return [
        {
            "direction": "raw-deduplicated FOMAML",
            "result": "Selected FOMAML reduced participant macro F1 by 0.046338 and increased ordinal MAE by 0.449093 versus supervised full-model adaptation.",
            "decision": "do_not_proceed",
            "status": "closed_negative",
            "report_path": "reports/integration/fomaml_label_q5_raw_diagnostic.md",
        },
        {
            "direction": "ShallowConvNet CAR",
            "result": "CAR reduced mean balanced accuracy in the factorial raw-EEG ablation.",
            "decision": "Do not adopt CAR as the default for this dataset/model.",
            "status": "closed_negative",
            "report_path": "reports/preprocessing_factorial_ablation.md",
        },
        {
            "direction": "ordinal Transformer losses",
            "result": "Ordinal objectives reduced ordinal errors but did not produce a stable balanced-accuracy gain.",
            "decision": "Keep categorical Transformer as the primary classification reference.",
            "status": "closed_negative",
            "report_path": "reports/ordinal_transformer_multiseed_statistics.md",
        },
        {
            "direction": "classification personalization",
            "result": "Full-model fine-tuning was not consistently superior to head-only tuning across subjects.",
            "decision": "Report both; avoid claiming universal full-model superiority.",
            "status": "closed_negative",
            "report_path": "reports/integration/personalization_multiseed_20pct.md",
        },
        {
            "direction": "COG-BCI CNN",
            "result": "EEGNet and ShallowConvNet only slightly exceeded the 0.333 three-class chance level.",
            "decision": "Close CNN-only N-Back exploration.",
            "status": "closed_negative",
            "report_path": "reports/integration/cog_bci_nback_baseline.md",
        },
        {
            "direction": "COG-BCI preprocessing",
            "result": "Filtering suppressed nuisance contamination but did not improve the inner criterion.",
            "decision": "Do not run a broader preprocessing search.",
            "status": "closed_negative",
            "report_path": "reports/integration/cog_bci_nback_preprocessing_ablation.md",
        },
        {
            "direction": "COG-BCI 62 channels",
            "result": "The 62-channel spectral advantage was +0.0077 balanced accuracy, below the +0.03 decision threshold.",
            "decision": "retain_14_channel_cache",
            "status": "closed_negative",
            "report_path": "reports/integration/cog_bci_nback_spectral_benchmark.md",
        },
        {
            "direction": "shape-only contrastive transfer",
            "result": "Pretraining did not outperform random initialization downstream.",
            "decision": "Do not extend shape-only transfer to more folds or seeds.",
            "status": "closed_negative",
            "report_path": "reports/integration/cog_bci_contrastive_transfer_screening.md",
        },
        {
            "direction": "time-aligned contrastive transfer",
            "result": "Physical time alignment improved representation diagnostics but not downstream macro F1.",
            "decision": "close_transfer_track",
            "status": "closed_negative",
            "report_path": "reports/integration/cog_bci_time_aligned_transfer_screening.md",
        },
    ]


def build_requirement_rows(repo_root: Path) -> list[dict[str, Any]]:
    registry = _load_yaml(repo_root / "reports/summary/requirements_registry.yaml")
    rows: list[dict[str, Any]] = []
    for item in registry["requirements"]:
        old = item["overall_status"]
        status = {
            "complete": "closed",
            "partial": "partially_closed",
            "not_started": "open",
            "needs_clarification": "blocked",
            "failed_acceptance_criterion": "partially_closed",
        }[old]
        if item["requirement_id"] in {"R-PERS-02"}:
            status = "partially_closed"
        if item["requirement_id"] in {"R-DATA-02", "R-DATA-03", "R-MULTI"}:
            status = "not_required_for_article"
        if status not in REQUIREMENT_STATUSES:
            raise FinalPackageError(f"Unknown requirement status {status!r}")
        evidence = item.get("evidence", [])
        rows.append(
            {
                "requirement_id": item["requirement_id"],
                "requirement": item["title"],
                "category": (
                    "service/demo"
                    if item["domain"] in {"streaming", "demo"}
                    else "scientific"
                    if item["domain"] in {
                        "data", "preprocessing", "features", "models",
                        "evaluation", "personalization", "multimodality",
                    }
                    else "formal",
                ),
                "status": status,
                "evidence": "|".join(
                    str(value.get("path") or value.get("experiment_id") or "")
                    for value in evidence
                ),
                "implementation": item["coverage"]["implementation"],
                "experiment": "|".join(
                    str(value["experiment_id"])
                    for value in evidence
                    if value.get("type") == "experiment"
                ),
                "report": "|".join(
                    str(value["path"])
                    for value in evidence
                    if value.get("type") == "report"
                ),
                "remaining_gap": (
                    "DANN is partially confirmed only in Old_EEG to gpn_data; "
                    "FOMAML diagnostic is do_not_proceed; reverse DANN and a "
                    "target-supervised upper bound remain untested."
                    if item["requirement_id"] == "R-PERS-02"
                    else "|".join(item.get("gaps", []))
                ),
                "recommended_closure_form": item["minimum_closure_action"][
                    "description"
                ],
            }
        )
    return rows


def build_reproducibility_limitations() -> list[dict[str, Any]]:
    return [
        {"area": "data access", "limitation": "Raw proprietary Emotiv recordings are not stored in Git.", "mitigation": "Tracked loaders, hashes, manifests and dataset statistics."},
        {"area": "runtime artifacts", "limitation": "Predictions, checkpoints and caches are intentionally ignored by Git.", "mitigation": "Relative artifact paths and checksums are recorded in reports/manifests."},
        {"area": "hardware", "limitation": "Torch runtimes were produced on an NVIDIA RTX 5060 Ti; exact timing is hardware-dependent.", "mitigation": "Device and training time are saved; CPU unit tests cover contracts."},
        {"area": "software environment", "limitation": "Exact CUDA/kernel determinism can vary across library and driver versions.", "mitigation": "Seeds, library requirements, configs and checkpoints are retained."},
        {"area": "label definition", "limitation": "Canonical label_q5 uses global quantile thresholds computed before subject splitting.", "mitigation": "Cross-fitted label sensitivity analysis shows 2.6816% changed windows."},
        {"area": "external transfer", "limitation": "Transfer screening uses one protected downstream fold.", "mitigation": "Result is explicitly diagnostic and the track is closed rather than generalized."},
        {"area": "COG-BCI preprocessing history", "limitation": "Upstream EEGLAB processing history is incompletely known.", "mitigation": "Metadata records unknown history and requires explicit opt-in for additional filters."},
        {"area": "PM temporal alignment", "limitation": "The fixed -10 s previous-window advantage does not identify a physiological delay or the proprietary Emotiv aggregation/timestamp mechanism.", "mitigation": "Treat EEG(t-10s)->PM(t) only as a dataset-level alignment contract; algorithmic latency, internal history and timestamp semantics remain hypotheses."},
        {"area": "PM lag regression R2", "limitation": "Participant-level R2 is heterogeneous and unstable for very small or near-constant target series; participant 9192c107 has only two relevant windows for some PM and produces extreme negative values.", "mitigation": "Prioritize participant-macro MAE and Pearson; report paired median R2 and favorable counts without zeroing NaN, dropping participants post hoc or using pooled arithmetic mean R2."},
        {"area": "formal specification", "limitation": "No authoritative tracked legal/acceptance specification was found.", "mitigation": "Requirement map distinguishes project-plan evidence from formal acceptance."},
    ]


def _save_bar(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    chance: float | None = None,
    horizontal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#35618f", "#6e9fbd", "#d17b49", "#6b8f71", "#9b6fa6"]
    positions = np.arange(len(labels))
    if horizontal:
        ax.barh(positions, values, color=colors[: len(labels)])
        ax.set_yticks(positions, labels)
        ax.set_xlabel(ylabel)
        ax.set_xlim(0, max(max(values) * 1.18, chance * 1.15 if chance else 0))
        if chance is not None:
            ax.axvline(chance, color="#b22222", linestyle="--", label=f"chance={chance:.3f}")
    else:
        ax.bar(positions, values, color=colors[: len(labels)])
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(values) * 1.18, chance * 1.15 if chance else 0))
        if chance is not None:
            ax.axhline(chance, color="#b22222", linestyle="--", label=f"chance={chance:.3f}")
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    if chance is not None:
        ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(path)


def _normalize_svg(path: Path) -> None:
    """Normalize generated SVG to LF and remove trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8", newline="")


def _save_flow(path: Path, *, split: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=True)
    ax.axis("off")
    if split:
        labels = [
            "All subjects",
            "Outer train subjects",
            "Inner train / validation\ngroup-disjoint",
            "Outer test subjects\nuntouched",
        ]
    else:
        labels = [
            "Datasets & caches",
            "Tasks & leakage-safe splits",
            "Model factory & shared adapters",
            "Metrics, artifacts & reports",
        ]
    xs = np.linspace(0.12, 0.88, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "fc": "#e8f1f8", "ec": "#35618f"},
        )
        if index:
            ax.annotate(
                "",
                xy=(x - 0.095, 0.52),
                xytext=(xs[index - 1] + 0.095, 0.52),
                arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.5},
            )
    ax.set_title(
        "Subject-disjoint outer/inner evaluation"
        if split
        else "Reproducible EEG benchmark platform",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(path)


def build_figures(
    output_dir: Path,
    classification: Sequence[Mapping[str, Any]],
    ordinal: Sequence[Mapping[str, Any]],
    personalization: Sequence[Mapping[str, Any]],
    preprocessing: Sequence[Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
    meta_learning: Sequence[Mapping[str, Any]],
    domain_adaptation: Sequence[Mapping[str, Any]],
    dann_folds: Sequence[Mapping[str, Any]],
    dann_seeds: Sequence[Mapping[str, Any]],
    experiment_statuses: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> None:
    figures = output_dir / "figures"
    _save_flow(figures / "01_platform_architecture.svg")
    _save_flow(figures / "02_subject_disjoint_protocol.svg", split=True)
    class_rows = [
        row for row in classification if row["model"] in {
            "Random Forest", "Torch MLP", "LSTM", "BiLSTM", "Transformer",
            "EEGNet", "ShallowConvNet"
        }
    ]
    _save_bar(
        figures / "03_main_model_comparison.svg",
        [str(row["model"]) for row in class_rows],
        [float(row["macro_f1_mean"]) for row in class_rows],
        title="Main label_q5 model comparison",
        ylabel="Macro F1 (fold/seed aggregate)",
        chance=0.20,
    )
    _save_bar(
        figures / "04_ordinal_errors.svg",
        [str(row["method"]) for row in ordinal],
        [float(row["ordinal_mae_mean"]) for row in ordinal],
        title="Ordinal error of Transformer objectives",
        ylabel="Ordinal MAE (lower is better)",
    )
    subject_path = (
        repo_root
        / "benchmark_results/calibration_label_q5_multiseed_20pct"
        / "20260725_172240/multiseed_subject_metrics.csv"
    )
    subject = pd.read_csv(subject_path)
    gains = (
        subject.groupby(["subject_id", "method"], sort=True)["macro_f1_gain"]
        .mean()
        .unstack("method")
    )
    fig, ax = plt.subplots(figsize=(9.2, 5), constrained_layout=True)
    ax.scatter(
        gains.get("head_only", pd.Series(dtype=float)),
        gains.get("full_model", pd.Series(dtype=float)),
        s=24,
        alpha=0.75,
        color="#35618f",
    )
    bounds = np.nanmax(np.abs(gains[["head_only", "full_model"]].to_numpy()))
    bounds = max(float(bounds), 0.01)
    ax.plot([-bounds, bounds], [-bounds, bounds], "--", color="#6b7280")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    ax.set(xlabel="Head-only macro F1 gain", ylabel="Full-model macro F1 gain",
           title="Personalization effect by subject (mean across seeds)")
    ax.grid(alpha=0.2)
    figure_path = figures / "05_personalization_by_subject.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)
    raw_pre = [row for row in preprocessing if row["experiment_id"] == "shallowconvnet_preprocessing_ablation"]
    _save_bar(
        figures / "06_preprocessing_ablation.svg",
        [str(row["trial_id"]) for row in raw_pre],
        [float(row["balanced_accuracy_mean"]) for row in raw_pre],
        title="ShallowConvNet preprocessing ablation",
        ylabel="Balanced accuracy",
        chance=0.20,
    )
    cnn = [row for row in external if row["comparison"] == "CNN N-Back"]
    spectral = [
        row for row in external
        if row["comparison"] == "spectral channels" and row["channel_policy"] == "14 channels"
    ]
    _save_bar(
        figures / "07_cog_bci_cnn_vs_spectral.svg",
        [str(row["method"]) for row in cnn] + ["Spectral LR (14ch)"],
        [float(row["balanced_accuracy"]) for row in cnn + spectral],
        title="COG-BCI N-Back: CNN and spectral baselines",
        ylabel="Record-level balanced accuracy",
        chance=1 / 3,
    )
    channels = [row for row in external if row["comparison"] == "spectral channels"]
    _save_bar(
        figures / "08_cog_bci_14_vs_62.svg",
        [str(row["channel_policy"]) for row in channels],
        [float(row["balanced_accuracy"]) for row in channels],
        title="COG-BCI spectral benchmark: channel policy",
        ylabel="Record-level balanced accuracy",
        chance=1 / 3,
    )
    transfer = [row for row in external if row["comparison"] == "time-aligned transfer"]
    _save_bar(
        figures / "09_contrastive_transfer.svg",
        [str(row["method"]) for row in transfer],
        [float(row["macro_f1"]) for row in transfer],
        title="Contrastive transfer screening",
        ylabel="Downstream macro F1 (protected fold 2)",
        chance=0.20,
    )

    fold_frame = pd.DataFrame(dann_folds).sort_values("fold")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#b45309" if value < 0 else "#35618f" for value in fold_frame["mean_delta_macro_f1"]]
    ax.bar(fold_frame["fold"].astype(str), fold_frame["mean_delta_macro_f1"], color=colors)
    ax.axhline(0, color="#374151", lw=1)
    ax.set(xlabel="Outer fold", ylabel="Participant-level Δ macro F1",
           title="Confirmatory DANN effect by fold (primary seeds 123/2026)")
    ax.grid(axis="y", alpha=0.25)
    figure_path = figures / "10_dann_fold_level_effect.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)

    seed_frame = pd.DataFrame(dann_seeds).sort_values("seed")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#d17b49" if group == "sensitivity" else "#35618f" for group in seed_frame["analysis_group"]]
    bars = ax.bar(seed_frame["seed"].astype(str), seed_frame["mean_delta_macro_f1"], color=colors)
    ax.axhline(0, color="#374151", lw=1)
    ax.set(xlabel="Model seed", ylabel="Participant-level Δ macro F1",
           title="DANN seed sensitivity")
    ax.legend([bars[0], bars[-1]], ["Sensitivity only", "Primary confirmatory"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    figure_path = figures / "11_dann_seed_level_effect.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)

    primary = pd.DataFrame(domain_adaptation)
    primary = primary[primary["analysis_group"] == "primary_confirmatory"].set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, metric, label, direction in zip(
        axes,
        ("macro_f1", "balanced_accuracy", "ordinal_mae"),
        ("Macro F1", "Balanced accuracy", "Ordinal MAE"),
        ("higher is better", "higher is better", "lower is better"),
    ):
        values = [primary.loc["source_only_matched", metric], primary.loc["dann", metric]]
        ax.bar(["Source-only", "DANN"], values, color=["#6e9fbd", "#35618f"])
        ax.set_title(f"{label}\n({direction})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Confirmatory DANN participant-level metrics")
    figure_path = figures / "12_dann_aggregate_metrics.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)

    meta = pd.DataFrame(meta_learning).set_index("method")
    meta_labels = ["Zero-shot", "Supervised full", "Selected FOMAML"]
    meta_modes = ["zero_shot_supervised", "supervised_full_model", "selected_fomaml"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, metric, label, direction in zip(
        axes,
        ("macro_f1", "balanced_accuracy", "ordinal_mae"),
        ("Macro F1", "Balanced accuracy", "Ordinal MAE"),
        ("higher is better", "higher is better", "lower is better"),
    ):
        ax.bar(meta_labels, [meta.loc[mode, metric] for mode in meta_modes], color=["#6e9fbd", "#6b8f71", "#d17b49"])
        ax.set_title(f"{label}\n({direction})")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("FOMAML diagnostic outer-test metrics (participant level)")
    figure_path = figures / "13_fomaml_outer_test_comparison.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)

    status_frame = pd.DataFrame(experiment_statuses)
    selected = status_frame[status_frame["task_id"].isin(["8T", "8U", "8F", "8X", "8Ц", "8Ч", "8Ш", "8Щ", "8Ю", "8Я"])]
    phase_order = ["infrastructure", "protocol", "diagnostic", "confirmatory"]
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.axis("off")
    grouped = {
        phase: selected[selected["analysis_level"] == phase]
        for phase in phase_order
    }
    x_positions = np.linspace(0.1, 0.9, len(phase_order))
    for x, phase in zip(x_positions, phase_order):
        ax.text(x, 0.96, phase.title(), ha="center", va="top", fontsize=12, fontweight="bold")
        for index, row in enumerate(grouped[phase].itertuples(index=False)):
            ax.text(
                x, 0.82 - index * 0.16,
                f"{row.task_id}: {row.status}", ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.4", "fc": "#e8f1f8", "ec": "#35618f"},
            )
    ax.set_title("Evidence-stage status map (stage labels, not quality scores)", fontsize=14)
    figure_path = figures / "14_evidence_status_map.svg"
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _normalize_svg(figure_path)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_Нет строк._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")).replace("|", "/") for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _render_reports(
    repo_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
    meta_learning: Sequence[Mapping[str, Any]],
    domain_adaptation: Sequence[Mapping[str, Any]],
    experiment_statuses: Sequence[Mapping[str, Any]],
    lag_alignment: Mapping[str, Any],
) -> dict[Path, str]:
    counts = Counter(str(row["status"]) for row in inventory)
    complete = sum(row["complete"] == "true" for row in provenance)
    incomplete = [row for row in provenance if row["complete"] != "true"]
    lag_classification = lag_alignment["classification"]
    lag_regression = lag_alignment["regression"]
    lag_pm_rows = [
        {
            "PM": str(row["pm"]).capitalize(),
            "MAE reduction": f"{float(row['mae_relative_reduction_percent']):.2f}%",
            "delta Pearson": f"{float(row['delta_pearson']):+.4f}",
            "MAE favorable folds": f"{int(row['mae_favorable_folds'])}/5",
            "Pearson favorable folds": f"{int(row['pearson_favorable_folds'])}/5",
        }
        for row in lag_alignment["per_pm"]
    ]
    final_state = f"""# Итоговое состояние проекта

Дата консолидации: {PACKAGE_DATE}. Пакет построен только из существующих
артефактов; обучение, новые folds/seeds и перестроение кэшей не выполнялись.

## Масштаб

- Экспериментов и инфраструктурных этапов: **{len(inventory)}**.
- Completed: **{counts['completed']}**.
- Diagnostic: **{counts['diagnostic']}**.
- Closed negative: **{counts['closed_negative']}**.
- Infrastructure only: **{counts['infrastructure_only']}**.
- Полный provenance по автоматической проверке: **{complete}/{len(provenance)}**.

Основные таблицы находятся в `reports/summary/final_result_tables/`, рисунки —
в `reports/summary/final_result_tables/figures/`, а канонический индекс —
`reports/summary/final_experiment_inventory.csv`.

## Зафиксированные решения

1. Основной научный протокол — outer GroupKFold по `subject_id` с
   group-aware inner validation.
2. Для `label_q5` наиболее сильные feature-sequence модели находятся около
   macro F1 0.36; случайный уровень 0.20 не используется как единственный
   критерий качества.
3. Семь PM targets оцениваются отдельно и macro-агрегируются только внутри
   одной регрессионной задачи.
4. Для всех семи PM принят фиксированный контракт
   `EEG(t−10s) → PM(t)`: в continuous regression participant-macro MAE
   снизилась с {float(lag_regression['lag0_mae']):.6f} до
   {float(lag_regression['lag_minus_10s_mae']):.6f} ({float(lag_regression['relative_mae_reduction_percent']):.2f}%),
   а Pearson вырос с {float(lag_regression['lag0_pearson']):.6f} до
   {float(lag_regression['lag_minus_10s_pearson']):.6f}; классификационное
   подтверждение независимо дало ΔMacro-F1
   {float(lag_classification['delta_macro_f1']):+.6f}.
5. COG-BCI нативный и transfer screening завершены как diagnostic/negative
   evidence.
6. Решения `retain_14_channel_cache` и `close_transfer_track` закрывают
   расширение 62-channel cache и contrastive transfer без новой гипотезы.
7. Raw-deduplicated FOMAML diagnostic получил `do_not_proceed`: Δmacro F1
   −0.046338 против supervised full-model при одном fold, одном seed и пяти
   участниках.
8. Confirmatory DANN дал небольшой положительный participant-level эффект
   (Δmacro F1 +0.008048; Δbalanced accuracy +0.008332; Δordinal MAE −0.034008),
   но имеет статус `partially_confirmed`, не `confirmed`.

Экспериментальная работа **не объявляется полностью завершённой или
замороженной**: пакет фиксирует только текущее состояние evidence.

## Неполный provenance

{_markdown_table(incomplete, ['experiment_id', 'status', 'missing', 'evidence_role'])}
"""
    conclusions = f"""# Научные выводы проекта

## Основная классификация

**Гипотеза.** EEG/POW и временной контекст позволяют предсказывать
пятиуровневый `label_q5` между испытуемыми. **Протокол.** Пятифолдовый
subject-disjoint GroupKFold, train-only preprocessing и group-aware inner
validation. **Результат.** Последовательные LSTM/BiLSTM/Transformer достигают
macro F1 около 0.36, превосходя RF/MLP и raw CNN baselines. **Решение.**
Transformer и recurrent модели остаются основными feature-based references.
**Ограничение.** Цель инерционна во времени и основана на глобальных
квантилях. **Статус для статьи:** основной результат с обязательным
sensitivity analysis разметки.

## Порядковая постановка

**Гипотеза.** Учёт порядка классов снизит тяжёлые ошибки без потери
категориального качества. **Протокол.** Три seeds, пять folds, subject-level
paired analysis; auxiliary weight выбирался только на inner validation.
**Результат.** CORN снижает ordinal MAE и severe-error rate, но balanced
accuracy не улучшается устойчиво; auxiliary policy также не поддержана.
**Решение.** Категориальный Transformer — основной baseline, CORN —
дополнительный анализ. **Ограничение.** Один набор и три seeds. **Статус для
статьи:** отрицательный/компромиссный результат.

## Регрессия и персонализация

**Гипотеза.** EEG+POW позволяют оценивать семь PM и адаптироваться к новому
пользователю. **Протокол.** Пятифолдовая RF-регрессия и leakage-safe
chronological 20% calibration. **Результат.** RF превосходит mean baseline;
full-model PM fine-tuning даёт небольшой устойчивый macro-MAE gain, но
классификационный full-model не универсально лучше head-only. **Решение.**
Для статьи показывать эффект и межсубъектную вариативность, не утверждать
универсальное превосходство полной настройки. **Ограничение.** Один бюджет
20%. **Статус:** основной результат с осторожной интерпретацией.

## Временное согласование EEG и PM

**Гипотеза.** Для PM-меток корректнее фиксированное причинное согласование
предыдущего окна `EEG(t−10s) → PM(t)`, чем `EEG(t) → PM(t)`. **Протокол.**
Сначала отдельный exploratory sweep сформировал единственную общую гипотезу
`−10 s`; затем её независимо проверили классификацией fold-local Q3 и
регрессией всех семи continuous PM на тех же пяти subject-disjoint folds,
371 признаке и XGBoost seed 42. **Результат.** Классификация дала положительные
ΔMacro-F1 и Δbalanced accuracy во всех 35/35 fold×PM сравнениях
({float(lag_classification['delta_macro_f1']):+.6f} и
{float(lag_classification['delta_balanced_accuracy']):+.6f}). В регрессии
participant-macro MAE уменьшилась на
{float(lag_regression['relative_mae_reduction_percent']):.2f}%
({int(lag_regression['favorable_fold_pm_mae'])}/35 сравнений), Pearson вырос
на {float(lag_regression['delta_pearson']):+.6f} (35/35); средние эффекты
благоприятны для 7/7 PM. **Решение.** Использовать фиксированное
`EEG(t−10s) → PM(t)` для всех семи PM в будущих core-экспериментах без
target-specific lag selection. **Ограничение.** Это коррекция временного
согласования данных, а не доказанный физиологический лаг или известное
описание внутреннего алгоритма Emotiv; R2 поддерживает вывод только как
неоднородная и нестабильная дополнительная метрика.

## COG-BCI

**Гипотеза.** Внешний N-Back корпус может подтвердить raw CNN, преимущества
62 каналов или перенос энкодера. **Протокол.** Record-safe caches,
subject-disjoint folds и заранее защищённый downstream fold. **Результат.**
CNN близки к chance; 62 канала дают только +0.0077 BA; shape-only и
time-aligned transfer не превосходят random initialization. Физическое
согласование улучшило contrastive representation diagnostics, но не
downstream. **Решение.** `retain_14_channel_cache`, `close_transfer_track`.
**Ограничение.** Transfer — screening на одном downstream fold. **Статус:**
диагностический отрицательный результат и приложение статьи.

## FOMAML и DANN

Эпизодическая инфраструктура и безопасные BatchNorm-контракты подтверждены
инженерно. В raw-deduplicated FOMAML diagnostic выбранная policy ухудшила
participant macro F1 и ordinal MAE относительно обычной supervised
full-model адаптации; решение — `do_not_proceed`. DANN в направлении
`Old_EEG → gpn_data` дал малый положительный средний эффект: четыре из пяти
folds и оба primary seeds положительны по macro F1. Статус
`partially_confirmed`: средний эффект ниже +0.01, win fraction ниже 60%, а
participant bootstrap interval включает ноль. Статистическая значимость не
установлена; source/target являются provenance-доменами, а не доказанно
разными устройствами.
"""
    negative_report = f"""# Закрытые отрицательные результаты

Отрицательные результаты ниже считаются научными исходами проверенных
гипотез, а не ошибками реализации. Они не должны смешиваться с финальными
положительными результатами.

{_markdown_table(negative, ['direction', 'result', 'decision', 'status', 'report_path'])}
"""
    repro = f"""# Аудит воспроизводимости итогового пакета

## Контракт

- Все пути в tracked-материалах относительные.
- Seeds и outer/inner protocols указаны в инвентаризации.
- Runtime predictions/checkpoints/caches остаются вне Git.
- Основные доказательства допускаются только при полном provenance.
- Метрики разных задач и уровней (`window`, `record`, `subject`) не
  агрегируются в одну величину.

## Результат

Полный provenance: **{complete}/{len(provenance)}**. Неполные записи
сохраняются в инвентаризации как supporting-only и не используются как
основное доказательство.

{_markdown_table(incomplete, ['experiment_id', 'missing', 'evidence_role'])}

## Аудит временного согласования EEG→PM

- Оба confirmatory протокола используют общий fixed-fold hash
  `{lag_alignment['fixed_fold_hash']}`, 371 признаков и seed 42.
- Classification protocol hash:
  `{lag_classification['protocol_hash']}`; regression protocol hash:
  `{lag_regression['protocol_hash']}`.
- Между `lag=0` и `lag=−10 s` сохранены одинаковые target sample IDs,
  участники, folds и train/test counts; cross-subject, cross-record и
  cross-fold pairs равны нулю.
- Пары строятся строго внутри logical record по точному шагу 10 s. Потеряны
  {int(lag_alignment['pairing']['first_window_losses'])} первых окон и
  {int(lag_alignment['pairing']['gap_losses'])} окон после разрывов; разрыв
  никогда не заменяется предыдущим доступным окном.
- R2 не сворачивается в pooled arithmetic mean: используются paired median
  ΔR2 {float(lag_regression['median_delta_r2']):+.6f}, favorable count
  {int(lag_regression['favorable_fold_pm_r2'])}/35 и знак per-PM median
  ({int(lag_regression['positive_pm_median_r2'])}/7 положительных).

Известные ограничения перечислены в
`reports/summary/final_result_tables/reproducibility_limitations.csv`.
"""
    req_counts = Counter(str(row["status"]) for row in requirements)
    req_report = f"""# Финальное покрытие требований

Карта разделяет формальное закрытие, научную ценность и сервисные/
демонстрационные пункты. Авторитетный юридический текст ТЗ в tracked-дереве
не найден, поэтому статусы являются инженерной трассировкой утверждённого
плана проекта.

## Сводка

{_markdown_table(
    [{'status': key, 'count': value} for key, value in sorted(req_counts.items())],
    ['status', 'count'],
)}

## Требования

{_markdown_table(
    requirements,
    ['requirement_id', 'requirement', 'category', 'status',
     'remaining_gap', 'recommended_closure_form'],
)}

## Оставшаяся работа

**Обязательно:** итоговая документация, description/data card,
reproducibility section, финальный отчёт, таблицы/рисунки, презентация.

**Желательно для статьи:** выбрать центральную гипотезу, зафиксировать
статистически корректные сравнения, related work, вклад, ограничения и
приложение отрицательных результатов.

**Исключить без новой гипотезы:** дальнейший DANN search, FOMAML sweep,
новый contrastive search, полный 62-канальный cache, дополнительные COG-BCI
CNN seeds, AutoML и новые внешние наборы. Уже выполненный confirmatory DANN
сохраняется как `partially_confirmed` evidence.
"""
    lag_report = f"""# Финальный вывод по временному согласованию EEG и PM

Дата консолидации: {PACKAGE_DATE}. Новое обучение для этого документа не
выполнялось: он объединяет завершённый exploratory sweep и два независимых
confirmatory сравнения.

## Финальное решение

Для всех семи continuous PM в будущих core-экспериментах использовать
фиксированное согласование **`EEG(t−10s) → PM(t)`** — EEG, предшествующее
timestamp PM на одно 10-секундное окно. Это экспериментальный контракт
согласования данных, а не preprocessing filter.

Не выполнять target-specific lag selection, не использовать отдельный
`Focus −20 s` и не запускать дополнительный lag search без новой заранее
утверждённой гипотезы.

## Цепочка доказательств

### 1. Exploratory sweep

Отдельный гипотезообразующий sweep сравнил `0, −10, −20, −30, −40 s`.
`−10 s` был широким лучшим кандидатом для шести из семи PM; у Focus локальный
максимум был при `−20 s` (ΔMacro-F1 +0.05255 относительно lag 0). Чтобы не
вносить target-specific post-hoc selection, до confirmatory regression был
зафиксирован единый кандидат `−10 s` для всех PM.

### 2. Независимое классификационное подтверждение

- Experiment: `{lag_classification['experiment_id']}`.
- Protocol hash: `{lag_classification['protocol_hash']}`.
- Execution commit: `{lag_classification['execution_commit']}`.
- Fold-local Q3 fit только на outer-train; 5 fixed subject-disjoint folds,
  371 признак, XGBoost seed 42; matched cohort
  {int(lag_classification['matched_rows'])} окон и
  {int(lag_classification['subjects'])} участника.
- 35/35 fold×PM сравнений положительны по Macro-F1 и balanced accuracy;
  7/7 PM имеют благоприятный средний эффект.
- Pooled paired ΔMacro-F1
  {float(lag_classification['delta_macro_f1']):+.6f}; Δbalanced accuracy
  {float(lag_classification['delta_balanced_accuracy']):+.6f}.

### 3. Continuous-regression подтверждение

- Experiment: `{lag_regression['experiment_id']}`.
- Protocol hash: `{lag_regression['protocol_hash']}`.
- Execution-code HEAD: `{lag_regression['execution_commit']}`.
- Все семь continuous PM, 5 fixed subject-disjoint folds, 371 признак,
  XGBRegressor seed 42; основная единица анализа — participant macro.
- В 35 fold×PM сравнениях MAE уменьшилась с
  {float(lag_regression['lag0_mae']):.6f} до
  {float(lag_regression['lag_minus_10s_mae']):.6f}: ΔMAE
  {float(lag_regression['delta_mae']):+.6f}, относительное снижение
  {float(lag_regression['relative_mae_reduction_percent']):.2f}%.
  Благоприятны {int(lag_regression['favorable_fold_pm_mae'])}/35 сравнений и
  7/7 PM means.
- Pearson вырос с {float(lag_regression['lag0_pearson']):.6f} до
  {float(lag_regression['lag_minus_10s_pearson']):.6f}: ΔPearson
  {float(lag_regression['delta_pearson']):+.6f}; благоприятны 35/35 сравнений
  и 7/7 PM means.

{_markdown_table(
    lag_pm_rows,
    ['PM', 'MAE reduction', 'delta Pearson',
     'MAE favorable folds', 'Pearson favorable folds'],
)}

## Инварианты и отсутствие leakage

- Общий fixed-fold hash:
  `{lag_alignment['fixed_fold_hash']}`.
- Условия используют одинаковые target sample IDs, subject IDs, fold
  membership и train/test counts.
- Cross-subject, cross-record и cross-fold pairs: 0.
- Pairing строго record-local по точному `t_start` с шагом 10 s: первое окно
  каждого record теряется; окно после gap также исключается. Предыдущее
  доступное окно никогда не подставляется вместо отсутствующего точного
  predecessor.
- Target labels test-участников не используются для fitting или выбора lag;
  regression не выполняет target-specific lag selection.

## R2 и ограничения

R2 благоприятен в {int(lag_regression['favorable_fold_pm_r2'])}/35 paired
сравнений; median paired ΔR2
{float(lag_regression['median_delta_r2']):+.6f}, а per-PM median положителен
для {int(lag_regression['positive_pm_median_r2'])}/7 PM. Pooled arithmetic
mean R2 не используется: participant-level R2 неустойчив при малом числе
окон и почти постоянной цели. В частности, у participant `9192c107` для
некоторых PM остаются только два релевантных окна, что порождает экстремально
отрицательные значения. NaN не заменялись нулями, участники post hoc не
исключались.

Результат сильно подтверждён MAE и Pearson; R2 даёт поддерживающее, но
неоднородное и нестабильное свидетельство.

## Допустимая интерпретация

Корректная формулировка: **фиксированное причинное согласование предыдущего
окна EEG**, или **temporal alignment correction**, где EEG предшествует PM
timestamp на одно 10-секундное окно.

Нельзя заключать, что доказан физиологический лаг, что Emotiv всегда
использует ровно предыдущие 10 секунд, что известен внутренний proprietary
algorithm или что найден универсальный физиологический delay. Algorithmic
latency, proprietary aggregation, internal history и timestamp semantics
остаются только возможными объяснениями наблюдаемого dataset-level эффекта.
"""
    meta_rows = {str(row["method"]): row for row in meta_learning}
    dann_rows = {
        (str(row["analysis_group"]), str(row["method"])): row
        for row in domain_adaptation
    }
    final_report = f"""# Итоговый пакет результатов EEG-бенчмарка

Дата консолидации: {PACKAGE_DATE}. Этот документ агрегирует только уже
существующие runtime-артефакты. Обучение, перестроение кэшей и изменение
научных decision rules не выполнялись. Работа не объявляется полностью
завершённой.

## 1. Цель проекта

Единая воспроизводимая платформа для EEG/POW задач, subject-disjoint оценки,
персонализации, transfer/meta-learning и унифицированных артефактов.

## 2. Наборы данных

Основной benchmark объединяет `gpn_data` и `Old_EEG`; COG-BCI используется
как отдельный внешний диагностический трек. Источники Emotiv считаются
provenance-доменами, а не автоматически разными устройствами.

## 3. Каноническая выборка

Классификационная supervised-выборка содержит 45 384 окна, 54 участника и
пять классов `label_q5`. Raw-deduplicated DANN universe содержит 30 958 окон,
54 участника и 86 logical records с формой `[1, 14, 2560]`.

## 4. Схема валидации

Основной outer protocol — subject-disjoint GroupKFold. Inner validation,
персонализация, meta-episodes и DANN source validation используют отдельные
group-aware partitions; target-test не участвует в выборе модели.

## 5. Базовые модели

Random Forest и MLP остаются воспроизводимыми feature-window baselines.

## 6. Глубокие модели

LSTM, BiLSTM и Transformer используют временной контекст; EEGNet и
ShallowConvNet работают с raw окнами через общий adapter/encoder contract.

## 7. Preprocessing ablation

Factorial raw-EEG ablation не поддержала CAR как default для
ShallowConvNet; исходные численные решения не пересматривались.

## 8. Персонализация

Leakage-safe calibration отделяет calibration от final evaluation. Эффект
зависит от участника; full-model tuning не объявляется универсально лучшим.

## 9. Временное согласование EEG и PM

Единый фиксированный контракт `EEG(t−10s) → PM(t)` независимо поддержан
классификацией (35/35 положительных fold×PM ΔMacro-F1, pooled delta
{float(lag_classification['delta_macro_f1']):+.6f}) и continuous regression.
В регрессии participant-macro MAE снизилась с
{float(lag_regression['lag0_mae']):.6f} до
{float(lag_regression['lag_minus_10s_mae']):.6f}
({float(lag_regression['relative_mae_reduction_percent']):.2f}%; 32/35
сравнений), а Pearson вырос с
{float(lag_regression['lag0_pearson']):.6f} до
{float(lag_regression['lag_minus_10s_pearson']):.6f} (35/35). Все 7/7 PM
имеют благоприятные средние MAE и Pearson эффекты. Это temporal alignment
correction на уровне набора данных, не доказательство физиологического или
proprietary-algorithm delay. Подробности:
`reports/diagnostics/pm_eeg_lag_final_conclusion.md`.

## 10. Контрастивное обучение

Shape-only и time-aligned screening не улучшили downstream macro F1;
решение `close_transfer_track` сохраняется.

## 11. COG-BCI

14-channel cache сохранён; 62-channel expansion отклонён по заранее заданному
правилу. CNN и spectral результаты остаются diagnostic/negative evidence.

## 12. FOMAML

Participant-level outer-test: zero-shot macro F1
{float(meta_rows['zero_shot_supervised']['macro_f1']):.6f}, supervised
full-model {float(meta_rows['supervised_full_model']['macro_f1']):.6f}, selected
FOMAML {float(meta_rows['selected_fomaml']['macro_f1']):.6f}. FOMAML против
supervised full-model: Δmacro F1
{float(meta_rows['selected_fomaml']['delta_macro_f1_vs_supervised_full_model']):+.6f},
Δbalanced accuracy
{float(meta_rows['selected_fomaml']['delta_balanced_accuracy_vs_supervised_full_model']):+.6f},
Δordinal MAE
{float(meta_rows['selected_fomaml']['delta_ordinal_mae_vs_supervised_full_model']):+.6f};
W/L/T 1/4/0. Решение `do_not_proceed`. Это один fold, seed 42, пять
участников и EEGNet; инфраструктурная готовность не означает успех метода.

## 13. DANN

Диагностический fold 1 / seed 42: Δmacro F1 +0.013364, Δbalanced accuracy
+0.019079, Δordinal MAE −0.069330, W/L/T 6/2/0. Его bootstrap interval
включает ноль, поэтому статус — diagnostic `proceed`, не подтверждение.

## 14. Подтверждающий анализ

Primary analysis использует folds 1–5 и seeds 123/2026. DANN против
source-only: Δmacro F1
{float(dann_rows[('primary_confirmatory', 'dann')]['delta_macro_f1']):+.6f},
Δbalanced accuracy
{float(dann_rows[('primary_confirmatory', 'dann')]['delta_balanced_accuracy']):+.6f},
Δordinal MAE
{float(dann_rows[('primary_confirmatory', 'dann')]['delta_ordinal_mae']):+.6f}.
Четыре из пяти folds и оба primary seeds положительны; 54.76% участников
улучшились, bootstrap 95% CI включает ноль. Решение `partially_confirmed`.
Seed 42 — sensitivity-only; fold 1 / seed 42 не переобучался и не входил в
primary decision. Всего выполнено 28 новых trainings.

## 15. Отрицательные результаты

Канонический список находится в `final_result_tables/negative_result_summary.csv`.
FOMAML `do_not_proceed` отделён от успешной episodic infrastructure.

## 16. Ограничения

Абсолютный macro F1 низок; source-validation содержит мало участников;
domain head значительно больше EEGNet; проверено только направление
`Old_EEG → gpn_data`; reverse direction и target-supervised upper bound не
выполнялись; эффекты неоднородны между участниками и seeds.

Для EEG→PM lag participant-level R2 неоднороден и неустойчив у коротких или
почти постоянных рядов; pooled arithmetic mean R2 не используется. Основной
вывод опирается на participant-macro MAE и Pearson, а R2 представлен paired
median и favorable counts без post-hoc исключений.

## 17. Требования проекта

Покрытие находится в `final_result_tables/requirement_coverage.csv` и
различает implementation, scientific evidence и незакрытые deliverables.

## 18. Воспроизводимость

Protocol/preregistration hashes, immutable unlock manifests, subject-level
splits и target-label firewall сохранены в runtime. Checkpoints,
predictions и кэши намеренно не отслеживаются Git.

## 19. Научные выводы

Проверенный FOMAML не поддержан. DANN показывает небольшой, но неоднородный
положительный эффект со статусом `partially_confirmed`; статистическая
значимость и полная доменная инвариантность не установлены. Фиксированное
согласование предыдущего EEG-окна независимо поддержано классификацией и
регрессией всех семи PM и принято как core data-alignment contract.

## 20. Открытые направления

Нужны финальная публикационная интерпретация, presentation/demo scope и,
только при новой утверждённой гипотезе, reverse DANN или target-supervised
upper bound. Автоматические DANN/FOMAML sweeps и дополнительный PM lag search
не планируются; target-specific `Focus −20 s` не используется.
"""
    return {
        repo_root / "reports/integration/project_final_state.md": final_state,
        repo_root
        / "reports/integration/project_scientific_conclusions.md": conclusions,
        repo_root
        / "reports/integration/project_negative_results.md": negative_report,
        repo_root
        / "reports/integration/project_reproducibility_audit.md": repro,
        repo_root
        / "reports/diagnostics/pm_eeg_lag_final_conclusion.md": lag_report,
        repo_root
        / "reports/requirements/final_requirement_coverage.md": req_report,
        repo_root / "reports/summary/final_project_results.md": final_report,
    }


def validate_no_absolute_paths(paths: Iterable[Path]) -> None:
    pattern = re.compile(r"(?i)(?:[A-Z]:[\\/]|file://|\\\\[A-Za-z0-9_.-]+\\)")
    violations: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".md", ".csv", ".yaml", ".yml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            violations.append(path.as_posix())
    if violations:
        raise FinalPackageError(f"Absolute paths found in: {violations}")


def generate(repo_root: Path) -> dict[str, Any]:
    """Generate the complete deterministic publication package."""
    root = repo_root.resolve()
    summary = root / "reports/summary"
    tables = summary / "final_result_tables"
    inventory = build_inventory(root)
    provenance = build_provenance_audit(root, inventory)
    datasets = build_dataset_characteristics()
    models = build_benchmark_models()
    lag_alignment = build_lag_alignment_summary(root)
    classification = build_classification_results(root)
    regression = build_regression_results(root)
    personalization = pd.read_csv(
        summary / "personalization_metrics_unified.csv"
    ).to_dict("records")
    ordinal = build_ordinal_results(root)
    preprocessing = pd.read_csv(
        summary / "preprocessing_metrics_unified.csv"
    ).to_dict("records")
    external = build_external_results(root)
    meta_learning = build_meta_learning_results(root)
    domain_adaptation = build_domain_adaptation_results(root)
    dann_folds = build_domain_adaptation_fold_results(root)
    dann_seeds = build_domain_adaptation_seed_results(root)
    experiment_statuses = build_experiment_statuses(root)
    negative = build_negative_results()
    requirements = build_requirement_rows(root)
    limitations = build_reproducibility_limitations()

    outputs: list[Path] = []
    inventory_path = summary / "final_experiment_inventory.csv"
    _write_csv(inventory_path, inventory, INVENTORY_COLUMNS)
    outputs.append(inventory_path)
    table_specs = [
        ("dataset_characteristics.csv", datasets, list(datasets[0])),
        ("benchmark_models.csv", models, list(models[0])),
        ("classification_results.csv", classification, list(classification[0])),
        ("regression_results.csv", regression, list(regression[0])),
        ("personalization_results.csv", personalization, list(personalization[0])),
        ("ordinal_results.csv", ordinal, list(ordinal[0])),
        ("preprocessing_ablation.csv", preprocessing, list(preprocessing[0])),
        ("external_dataset_results.csv", external, list(external[0])),
        ("final_meta_learning_results.csv", meta_learning, list(meta_learning[0])),
        ("final_domain_adaptation_results.csv", domain_adaptation, list(domain_adaptation[0])),
        ("final_domain_adaptation_fold_results.csv", dann_folds, list(dann_folds[0])),
        ("final_domain_adaptation_seed_results.csv", dann_seeds, list(dann_seeds[0])),
        ("final_experiment_statuses.csv", experiment_statuses, STATUS_COLUMNS),
        ("negative_result_summary.csv", negative, list(negative[0])),
        ("provenance_audit.csv", provenance, PROVENANCE_COLUMNS),
        ("requirement_coverage.csv", requirements, list(requirements[0])),
        ("reproducibility_limitations.csv", limitations, list(limitations[0])),
    ]
    for filename, rows, columns in table_specs:
        path = tables / filename
        _write_csv(path, rows, columns)
        outputs.append(path)
    build_figures(
        tables,
        classification,
        ordinal,
        personalization,
        preprocessing,
        external,
        meta_learning,
        domain_adaptation,
        dann_folds,
        dann_seeds,
        experiment_statuses,
        root,
    )
    outputs.extend(sorted((tables / "figures").glob("*.svg")))
    for path, text in _render_reports(
        root, inventory, provenance, requirements, negative,
        meta_learning, domain_adaptation, experiment_statuses, lag_alignment,
    ).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8", newline="")
        outputs.append(path)
    result_inventory_path = tables / "final_result_inventory.csv"
    inventory_outputs = sorted({*outputs, result_inventory_path})
    result_inventory = [
        {
            "artifact_path": path.relative_to(root).as_posix(),
            "artifact_type": (
                "svg_figure" if path.suffix == ".svg"
                else "csv_table" if path.suffix == ".csv"
                else "markdown_report"
            ),
            "generated_by": "bench.analysis.project_final_package.generate",
            "tracked_intent": "yes",
        }
        for path in inventory_outputs
    ]
    _write_csv(
        result_inventory_path,
        result_inventory,
        ["artifact_path", "artifact_type", "generated_by", "tracked_intent"],
    )
    outputs.append(result_inventory_path)
    validate_no_absolute_paths(outputs)
    counts = Counter(str(row["status"]) for row in inventory)
    return {
        "experiments": len(inventory),
        "status_counts": dict(sorted(counts.items())),
        "provenance_complete": sum(row["complete"] == "true" for row in provenance),
        "provenance_incomplete": sum(row["complete"] != "true" for row in provenance),
        "tables": len(table_specs) + 2,
        "figures": 14,
        "outputs": [path.relative_to(root).as_posix() for path in sorted(outputs)],
    }
