"""Leakage-safe subject split manifests for native COG-BCI tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.tasks.cog_bci_tasks import (
    COGBCITargetIndex,
    build_cog_bci_target_index,
    require_relative_path,
    task_definition_from_config,
)
from bench.validation.cross_val import deterministic_group_kfold_indices


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_BUILDER_VERSION = "cog-bci-task-protocol-v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_value(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in frame["target"].value_counts().sort_index().items()
    }


def _record_distribution(frame: pd.DataFrame) -> dict[str, int]:
    records = frame[["record_id", "target"]].drop_duplicates()
    return {
        str(int(key)): int(value)
        for key, value in records["target"].value_counts().sort_index().items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlap(left: pd.Series, right: pd.Series) -> list[str]:
    return sorted(set(left.astype(str)) & set(right.astype(str)))


def _split_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, Any]:
    subjects = _overlap(train["subject_id"], test["subject_id"])
    records = _overlap(train["record_id"], test["record_id"])
    record_groups = _overlap(
        train["record_group_id"], test["record_group_id"]
    )
    samples = _overlap(train["sample_id"], test["sample_id"])
    return {
        f"{prefix}_subject_overlap_count": len(subjects),
        f"{prefix}_record_overlap_count": len(records),
        f"{prefix}_record_group_overlap_count": len(record_groups),
        f"{prefix}_sample_overlap_count": len(samples),
        f"{prefix}_subject_overlap": subjects,
        f"{prefix}_record_overlap": records,
        f"{prefix}_record_group_overlap": record_groups,
        f"{prefix}_sample_overlap": samples,
        "leakage_safe": (
            not subjects and not records and not record_groups and not samples
        ),
    }


@dataclass(frozen=True)
class COGBCIProtocolConfig:
    """Resolved, machine-independent protocol configuration."""

    task_id: str
    target_name: str
    window_cache_dir: Path
    output_dir: Path
    outer_n_splits: int
    inner_n_splits: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "COGBCIProtocolConfig":
        definition = task_definition_from_config(config)
        dataset = config.get("dataset")
        if dataset != "cog_bci":
            raise ValueError("COG-BCI protocol dataset must be 'cog_bci'")
        cache = config.get("window_cache", {})
        if not isinstance(cache, Mapping):
            raise ValueError("window_cache must be a mapping")
        cache_dir = require_relative_path(
            cache.get("path", ""), label="window_cache.path"
        )
        output_dir = require_relative_path(
            config.get("output_dir", ""), label="output_dir"
        )
        if str(cache_dir) in {"", "."} or str(output_dir) in {"", "."}:
            raise ValueError("window_cache.path and output_dir are required")
        splitter = config.get("splitter", {})
        inner = config.get("inner_validation", {})
        expected_outer = {
            "name": "group_kfold",
            "group_column": "subject_id",
            "shuffle": False,
        }
        expected_inner = {
            "name": "group_kfold_first_fold",
            "group_column": "subject_id",
            "shuffle": False,
        }
        for key, value in expected_outer.items():
            if splitter.get(key) != value:
                raise ValueError(f"splitter.{key} must be {value!r}")
        for key, value in expected_inner.items():
            if inner.get(key) != value:
                raise ValueError(f"inner_validation.{key} must be {value!r}")
        outer_n_splits = int(splitter.get("n_splits", 0))
        inner_n_splits = int(inner.get("n_splits", 0))
        if outer_n_splits != 5:
            raise ValueError("Canonical outer GroupKFold requires 5 folds")
        if inner_n_splits < 2:
            raise ValueError("inner_validation.n_splits must be at least 2")
        forbidden = {
            "model",
            "training",
            "optimizer",
            "scaler",
            "sampler",
            "class_weights",
        }
        present = sorted(forbidden & set(config))
        if present:
            raise ValueError(
                f"Protocol config must not contain training fields: {present}"
            )
        return cls(
            task_id=definition.task_id,
            target_name=definition.target_name,
            window_cache_dir=cache_dir,
            output_dir=output_dir,
            outer_n_splits=outer_n_splits,
            inner_n_splits=inner_n_splits,
        )


@dataclass
class COGBCIProtocolResult:
    """In-memory protocol documents and assignment tables."""

    target_index: COGBCITargetIndex
    task_definition: dict[str, Any]
    record_summary: pd.DataFrame
    window_summary: pd.DataFrame
    class_balance: dict[str, Any]
    outer_folds: dict[str, Any]
    outer_assignments: pd.DataFrame
    inner_folds: dict[str, Any]
    inner_assignments: pd.DataFrame
    loso_folds: dict[str, Any]
    protocol_summary: dict[str, Any]


def _record_summary(target_index: COGBCITargetIndex) -> pd.DataFrame:
    frame = target_index.frame
    rows: list[dict[str, Any]] = []
    for record_id, group in frame.groupby("record_id", sort=True):
        accepted = group.loc[group["included_for_supervised"]]
        first = group.iloc[0]
        rows.append(
            {
                "task_id": first["task_id"],
                "target_name": first["target_name"],
                "subject_id": first["subject_id"],
                "session_id": first["session_id"],
                "record_id": record_id,
                "record_group_id": first["record_group_id"],
                "task_variant": first["task_variant"],
                "target": int(first["target"]),
                "class_name": first["class_name"],
                "accepted_windows": int(len(accepted)),
                "rejected_windows": int(len(group) - len(accepted)),
                "total_windows": int(len(group)),
                "accepted_duration_seconds": float(
                    accepted["window_duration_seconds"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("record_id", kind="stable")


def _window_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in records.groupby("target", sort=True):
        rows.append(
            {
                "target": int(target),
                "class_name": group["class_name"].iloc[0],
                "subjects": int(group["subject_id"].nunique()),
                "sessions": int(group["session_id"].nunique()),
                "records": int(len(group)),
                "accepted_windows": int(group["accepted_windows"].sum()),
                "rejected_windows": int(group["rejected_windows"].sum()),
                "accepted_duration_hours": float(
                    group["accepted_duration_seconds"].sum() / 3600.0
                ),
                "windows_per_record_min": int(
                    group["accepted_windows"].min()
                ),
                "windows_per_record_max": int(
                    group["accepted_windows"].max()
                ),
                "windows_per_record_mean": float(
                    group["accepted_windows"].mean()
                ),
                "windows_per_record_std": float(
                    group["accepted_windows"].std(ddof=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _class_balance(
    target_index: COGBCITargetIndex,
    records: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    subject_records = (
        records.groupby(["subject_id", "target"], sort=True)
        .size()
        .rename("records")
        .reset_index()
    )
    session_records = (
        records.groupby(["session_id", "target"], sort=True)
        .size()
        .rename("records")
        .reset_index()
    )
    subject_windows = (
        target_index.accepted.groupby(["subject_id", "target"], sort=True)
        .size()
        .rename("accepted_windows")
        .reset_index()
    )
    return {
        "task_id": target_index.definition.task_id,
        "target_name": target_index.definition.target_name,
        "target_schema_hash": target_index.definition.schema_hash,
        "target_index_hash": target_index.target_index_hash,
        "accepted_window_distribution": _distribution(target_index.accepted),
        "record_distribution": _record_distribution(target_index.accepted),
        "class_rows": windows.to_dict(orient="records"),
        "records_per_subject_and_class": subject_records.to_dict(
            orient="records"
        ),
        "records_per_session_and_class": session_records.to_dict(
            orient="records"
        ),
        "windows_per_subject_and_class": subject_windows.to_dict(
            orient="records"
        ),
    }


def _fold_payload(
    *,
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    audit_prefix: str,
) -> dict[str, Any]:
    payload = {
        "fold": fold,
        "train_subjects": sorted(train["subject_id"].astype(str).unique()),
        "test_subjects": sorted(test["subject_id"].astype(str).unique()),
        "train_records": sorted(train["record_id"].astype(str).unique()),
        "test_records": sorted(test["record_id"].astype(str).unique()),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "train_records_count": int(train["record_id"].nunique()),
        "test_records_count": int(test["record_id"].nunique()),
        "train_class_distribution_windows": _distribution(train),
        "test_class_distribution_windows": _distribution(test),
        "train_class_distribution_records": _record_distribution(train),
        "test_class_distribution_records": _record_distribution(test),
        "audit": _split_audit(train, test, prefix=audit_prefix),
        "train_sample_id_hash": _stable_hash(
            sorted(train["sample_id"].astype(str))
        ),
        "test_sample_id_hash": _stable_hash(
            sorted(test["sample_id"].astype(str))
        ),
    }
    payload["fold_hash"] = _stable_hash(payload)
    return payload


def build_cog_bci_protocol(
    target_index: COGBCITargetIndex,
    *,
    window_cache_config_hash: str,
    window_index_sha256: str,
    outer_n_splits: int = 5,
    inner_n_splits: int = 5,
) -> COGBCIProtocolResult:
    """Build deterministic outer, inner, and LOSO subject manifests."""
    data = target_index.accepted.reset_index(drop=True)
    if sorted(data["target"].unique().tolist()) != [0, 1, 2]:
        raise ValueError("Canonical COG-BCI task must contain classes 0, 1, 2")
    if data["subject_id"].nunique() < max(outer_n_splits, inner_n_splits + 1):
        raise ValueError("Not enough subjects for requested split protocol")

    outer_rows: list[dict[str, Any]] = []
    outer_docs: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    inner_docs: list[dict[str, Any]] = []
    groups = data["subject_id"].astype(str).to_numpy()
    outer_indices = deterministic_group_kfold_indices(
        groups, n_splits=outer_n_splits
    )
    for fold, (train_idx, test_idx) in enumerate(outer_indices, start=1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        outer_doc = _fold_payload(
            fold=fold,
            train=train,
            test=test,
            audit_prefix="outer_train_test",
        )
        if not outer_doc["audit"]["leakage_safe"]:
            raise ValueError(f"Outer fold {fold} is not leakage-safe")
        outer_docs.append(outer_doc)
        for row in test.itertuples(index=False):
            outer_rows.append(
                {
                    "task_id": target_index.definition.task_id,
                    "target_name": target_index.definition.target_name,
                    "fold": fold,
                    "sample_id": row.sample_id,
                    "subject_id": row.subject_id,
                    "record_id": row.record_id,
                    "record_group_id": row.record_group_id,
                    "target": int(row.target),
                    "class_name": row.class_name,
                    "partition": "outer_test",
                }
            )

        local_groups = train["subject_id"].astype(str).to_numpy()
        inner_train_local, inner_validation_local = (
            deterministic_group_kfold_indices(
                local_groups, n_splits=inner_n_splits
            )[0]
        )
        inner_train = train.iloc[inner_train_local]
        inner_validation = train.iloc[inner_validation_local]
        inner_doc = _fold_payload(
            fold=fold,
            train=inner_train,
            test=inner_validation,
            audit_prefix="inner_train_validation",
        )
        inner_doc["strategy"] = "group_kfold_first_fold"
        inner_doc["outer_test_subject_overlap_count"] = len(
            set(inner_train["subject_id"])
            .union(inner_validation["subject_id"])
            .intersection(test["subject_id"])
        )
        inner_doc["outer_test_record_group_overlap_count"] = len(
            set(inner_train["record_group_id"])
            .union(inner_validation["record_group_id"])
            .intersection(test["record_group_id"])
        )
        inner_doc["outer_test_record_overlap_count"] = len(
            set(inner_train["record_id"])
            .union(inner_validation["record_id"])
            .intersection(test["record_id"])
        )
        inner_doc["outer_test_sample_overlap_count"] = len(
            set(inner_train["sample_id"])
            .union(inner_validation["sample_id"])
            .intersection(test["sample_id"])
        )
        inner_doc["leakage_safe"] = (
            inner_doc["audit"]["leakage_safe"]
            and inner_doc["outer_test_subject_overlap_count"] == 0
            and inner_doc["outer_test_record_overlap_count"] == 0
            and inner_doc["outer_test_record_group_overlap_count"] == 0
            and inner_doc["outer_test_sample_overlap_count"] == 0
        )
        inner_doc["fold_hash"] = _stable_hash(
            {key: value for key, value in inner_doc.items() if key != "fold_hash"}
        )
        if not inner_doc["leakage_safe"]:
            raise ValueError(f"Inner fold {fold} is not leakage-safe")
        inner_docs.append(inner_doc)
        partitions = (
            (inner_train, "inner_train"),
            (inner_validation, "inner_validation"),
            (test, "outer_test_excluded"),
        )
        for partition, name in partitions:
            for row in partition.itertuples(index=False):
                inner_rows.append(
                    {
                        "task_id": target_index.definition.task_id,
                        "outer_fold": fold,
                        "sample_id": row.sample_id,
                        "subject_id": row.subject_id,
                        "record_id": row.record_id,
                        "record_group_id": row.record_group_id,
                        "target": int(row.target),
                        "partition": name,
                    }
                )

    outer_assignments = pd.DataFrame(outer_rows).sort_values(
        ["fold", "sample_id"], kind="stable"
    )
    if len(outer_assignments) != len(data):
        raise ValueError("Each accepted sample must have one outer test fold")
    if outer_assignments["sample_id"].duplicated().any():
        raise ValueError("Outer assignments contain duplicate sample_id")
    inner_assignments = pd.DataFrame(inner_rows).sort_values(
        ["outer_fold", "partition", "sample_id"], kind="stable"
    )
    if len(inner_assignments) != len(data) * outer_n_splits:
        raise ValueError("Each sample must have one inner assignment per fold")
    if inner_assignments.duplicated(["outer_fold", "sample_id"]).any():
        raise ValueError("Inner assignments duplicate a sample within a fold")

    loso_docs: list[dict[str, Any]] = []
    for fold, subject in enumerate(
        sorted(data["subject_id"].astype(str).unique()), start=1
    ):
        test = data.loc[data["subject_id"].astype(str).eq(subject)]
        train = data.loc[~data["subject_id"].astype(str).eq(subject)]
        doc = _fold_payload(
            fold=fold,
            train=train,
            test=test,
            audit_prefix="loso_train_test",
        )
        doc["held_out_subject"] = subject
        if not doc["audit"]["leakage_safe"]:
            raise ValueError(f"LOSO fold {fold} is not leakage-safe")
        loso_docs.append(doc)

    records = _record_summary(target_index)
    windows = _window_summary(records)
    balance = _class_balance(target_index, records, windows)
    common = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "task_id": target_index.definition.task_id,
        "target_name": target_index.definition.target_name,
        "target_schema_hash": target_index.definition.schema_hash,
        "target_index_hash": target_index.target_index_hash,
        "window_cache_config_hash": window_cache_config_hash,
        "window_index_sha256": window_index_sha256,
        "group_column": "subject_id",
        "record_group_column": "record_group_id",
        "shuffle": False,
        "random_state": None,
    }
    outer_document = {
        **common,
        "strategy": "group_kfold",
        "n_splits": outer_n_splits,
        "folds": outer_docs,
    }
    outer_document["split_hash"] = _stable_hash(outer_document)
    inner_document = {
        **common,
        "strategy": "group_kfold_first_fold",
        "n_splits": inner_n_splits,
        "outer_split_hash": outer_document["split_hash"],
        "folds": inner_docs,
    }
    inner_document["split_hash"] = _stable_hash(inner_document)
    loso_document = {
        **common,
        "strategy": "leave_one_subject_out",
        "n_splits": len(loso_docs),
        "folds": loso_docs,
    }
    loso_document["split_hash"] = _stable_hash(loso_document)
    protocol_identity = {
        **common,
        "dataset": "cog_bci",
        "subjects": sorted(data["subject_id"].astype(str).unique()),
        "records": sorted(data["record_id"].astype(str).unique()),
        "accepted_sample_id_hash": _stable_hash(
            sorted(data["sample_id"].astype(str))
        ),
        "outer_split_hash": outer_document["split_hash"],
        "inner_split_hash": inner_document["split_hash"],
        "loso_split_hash": loso_document["split_hash"],
    }
    protocol_hash = _stable_hash(protocol_identity)
    summary = {
        **protocol_identity,
        "builder_version": PROTOCOL_BUILDER_VERSION,
        "protocol_hash": protocol_hash,
        "result_status": "diagnostic",
        "target_level": "record",
        "target_inherited_by_windows": True,
        "subjects": int(data["subject_id"].nunique()),
        "sessions": int(data["session_id"].nunique()),
        "records": int(data["record_id"].nunique()),
        "accepted_windows": int(len(data)),
        "rejected_windows": int(
            (~target_index.frame["included_for_supervised"]).sum()
        ),
        "class_distribution": _distribution(data),
        "outer_folds": outer_n_splits,
        "inner_manifests": len(inner_docs),
        "loso_folds": len(loso_docs),
        "all_outer_folds_leakage_safe": all(
            fold["audit"]["leakage_safe"] for fold in outer_docs
        ),
        "all_inner_folds_leakage_safe": all(
            fold["leakage_safe"] for fold in inner_docs
        ),
        "all_loso_folds_leakage_safe": all(
            fold["audit"]["leakage_safe"] for fold in loso_docs
        ),
        "training_performed": False,
        "model_used": False,
        "scaler_fitted": False,
        "sampler_used": False,
    }
    return COGBCIProtocolResult(
        target_index=target_index,
        task_definition=target_index.definition.to_dict(),
        record_summary=records,
        window_summary=windows,
        class_balance=balance,
        outer_folds=outer_document,
        outer_assignments=outer_assignments,
        inner_folds=inner_document,
        inner_assignments=inner_assignments,
        loso_folds=loso_document,
        protocol_summary=summary,
    )


def _protocol_report(result: COGBCIProtocolResult) -> str:
    summary = result.protocol_summary
    rows = [
        "# COG-BCI task and split protocol",
        "",
        f"- Task: `{summary['task_id']}`",
        f"- Target: `{summary['target_name']}` (record-level ordinal class)",
        f"- Subjects: {summary['subjects']}",
        f"- Records: {summary['records']}",
        f"- Accepted windows: {summary['accepted_windows']}",
        f"- Class distribution: `{summary['class_distribution']}`",
        "- Outer evaluation: deterministic 5-fold GroupKFold by `subject_id`.",
        "- Inner validation: first deterministic subject GroupKFold split, "
        "constructed only from the corresponding outer-train partition.",
        f"- LOSO manifests: {summary['loso_folds']}",
        f"- Protocol hash: `{summary['protocol_hash']}`",
        "",
        "All subject, record-group, and sample overlap audits passed. "
        "No model was trained and no scaler or sampler was fitted.",
        "",
    ]
    return "\n".join(rows)


def materialize_cog_bci_protocol(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> COGBCIProtocolResult:
    """Resolve a config, build manifests, and write runtime artifacts."""
    resolved = COGBCIProtocolConfig.from_mapping(config)
    cache_dir = repository_root / resolved.window_cache_dir
    output_dir = repository_root / resolved.output_dir
    manifest_path = cache_dir / "dataset_manifest.json"
    index_path = cache_dir / "window_index.parquet"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Existing COG-BCI window cache is incomplete: {cache_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = config.get("window_cache", {}).get("config_hash")
    actual_hash = str(manifest.get("config_hash", ""))
    if not actual_hash:
        raise ValueError("Window cache manifest has no config_hash")
    if expected_hash not in (None, actual_hash):
        raise ValueError(
            f"Window cache config hash mismatch: {expected_hash} != "
            f"{actual_hash}"
        )
    windows = pd.read_parquet(index_path)
    definition = task_definition_from_config(config)
    target_index = build_cog_bci_target_index(windows, definition.task_id)
    result = build_cog_bci_protocol(
        target_index,
        window_cache_config_hash=actual_hash,
        window_index_sha256=_sha256_file(index_path),
        outer_n_splits=resolved.outer_n_splits,
        inner_n_splits=resolved.inner_n_splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "task_definition.json", result.task_definition)
    result.target_index.frame.to_parquet(
        output_dir / "target_index.parquet", index=False
    )
    result.record_summary.to_csv(
        output_dir / "record_target_summary.csv", index=False
    )
    result.window_summary.to_csv(
        output_dir / "window_target_summary.csv", index=False
    )
    _write_json(output_dir / "class_balance.json", result.class_balance)
    _write_json(output_dir / "outer_folds.json", result.outer_folds)
    result.outer_assignments.to_parquet(
        output_dir / "outer_assignments.parquet", index=False
    )
    _write_json(output_dir / "inner_folds.json", result.inner_folds)
    result.inner_assignments.to_parquet(
        output_dir / "inner_assignments.parquet", index=False
    )
    _write_json(output_dir / "loso_folds.json", result.loso_folds)
    _write_json(output_dir / "protocol_summary.json", result.protocol_summary)
    (output_dir / "protocol_report.md").write_text(
        _protocol_report(result), encoding="utf-8"
    )
    pd.DataFrame(
        columns=["task_id", "stage", "error_type", "message"]
    ).to_csv(output_dir / "errors.csv", index=False)
    return result
