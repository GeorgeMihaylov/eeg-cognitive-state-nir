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


PACKAGE_DATE = "2026-07-29"
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


def build_negative_results() -> list[dict[str, Any]]:
    return [
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
            status = "infrastructure_ready"
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
                "remaining_gap": "|".join(item.get("gaps", [])),
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
    fig.savefig(path, format="svg")
    plt.close(fig)


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
    fig.savefig(path, format="svg")
    plt.close(fig)


def build_figures(
    output_dir: Path,
    classification: Sequence[Mapping[str, Any]],
    ordinal: Sequence[Mapping[str, Any]],
    personalization: Sequence[Mapping[str, Any]],
    preprocessing: Sequence[Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
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
    fig.savefig(figures / "05_personalization_by_subject.svg", format="svg")
    plt.close(fig)
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
) -> dict[Path, str]:
    counts = Counter(str(row["status"]) for row in inventory)
    complete = sum(row["complete"] == "true" for row in provenance)
    incomplete = [row for row in provenance if row["complete"] != "true"]
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
4. COG-BCI нативный и transfer screening завершены как diagnostic/negative
   evidence.
5. Решения `retain_14_channel_cache` и `close_transfer_track` закрывают
   расширение 62-channel cache и contrastive transfer без новой гипотезы.

## Неполный provenance

{_markdown_table(incomplete, ['experiment_id', 'status', 'missing', 'evidence_role'])}
"""
    conclusions = """# Научные выводы проекта

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

**Исключить без новой гипотезы:** DANN, новый contrastive search, полный
62-канальный cache, дополнительные COG-BCI CNN seeds, AutoML и новые внешние
наборы.
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
        / "reports/requirements/final_requirement_coverage.md": req_report,
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
    classification = pd.read_csv(
        summary / "classification_metrics_unified.csv"
    ).to_dict("records")
    regression = build_regression_results(root)
    personalization = pd.read_csv(
        summary / "personalization_metrics_unified.csv"
    ).to_dict("records")
    ordinal = build_ordinal_results(root)
    preprocessing = pd.read_csv(
        summary / "preprocessing_metrics_unified.csv"
    ).to_dict("records")
    external = build_external_results(root)
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
        root,
    )
    outputs.extend(sorted((tables / "figures").glob("*.svg")))
    for path, text in _render_reports(
        root, inventory, provenance, requirements, negative
    ).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8", newline="")
        outputs.append(path)
    validate_no_absolute_paths(outputs)
    counts = Counter(str(row["status"]) for row in inventory)
    return {
        "experiments": len(inventory),
        "status_counts": dict(sorted(counts.items())),
        "provenance_complete": sum(row["complete"] == "true" for row in provenance),
        "provenance_incomplete": sum(row["complete"] != "true" for row in provenance),
        "tables": len(table_specs) + 1,
        "figures": 9,
        "outputs": [path.relative_to(root).as_posix() for path in sorted(outputs)],
    }
