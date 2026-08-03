"""Build a raw-deduplicated, leakage-safe FOMAML episode protocol.

This module is deliberately metadata-only.  It does not import a training
adapter, create an optimizer, materialize CUDA tensors, or run model code.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import posixpath
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.datasets.logical_recordings import ensure_record_group_ids
from bench.datasets.raw_eeg_window_dataset import CANONICAL_EEG_CHANNELS
from bench.meta.episodes import stable_hash
from bench.meta.validation import validate_episode


SCHEMA_VERSION = "1.0"
PROTOCOL_ID = "fomaml_label_q5_raw_deduplicated_v2"
VALID_LABELS = (0, 1, 2, 3, 4)
CLASS_POLICIES = ("none", "at_least_one_per_class", "require_all_classes")


@dataclass(frozen=True)
class RawProtocolBuildResult:
    summary: dict[str, Any]
    raw_inventory: pd.DataFrame
    eligibility: pd.DataFrame
    episode_index: pd.DataFrame


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def _relative_text(value: Any) -> str:
    text = str(value).replace("\\", "/")
    if Path(str(value)).is_absolute() or text.startswith("/"):
        raise ValueError(f"Absolute path is forbidden in protocol metadata: {value}")
    return posixpath.normpath(text)


def _directory_hashes(repository_root: Path, relative_dirs: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_dir in relative_dirs:
        directory = repository_root / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(repository_root).as_posix()
            hashes[relative] = _sha256_file(path)
    return hashes


def _contains_absolute_path(value: Any) -> bool:
    for item in _walk_values(value):
        if not isinstance(item, str):
            continue
        normalized = item.replace("\\", "/")
        if normalized.startswith("/") or (
            len(normalized) >= 3 and normalized[1:3] == ":/"
        ):
            return True
    return False


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def validate_raw_protocol_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not False:
        raise ValueError("Raw protocol creation requires execution_enabled=false")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if int(config.get("seed", -1)) != 42 or int(config.get("outer_fold", -1)) != 1:
        raise ValueError("Task 8C is restricted to outer fold 1 and seed 42")
    if config.get("selected_class_policy") not in CLASS_POLICIES:
        raise ValueError("Unknown selected_class_policy")
    if int(config["episode_contract"]["support_record_count"]) != 1:
        raise ValueError("The preregistered contract uses one complete support record")
    if config["episode_contract"].get("allow_window_level_fallback") is not False:
        raise ValueError("Window-level fallback must be disabled")
    if config["episode_contract"].get("sampling_with_replacement") is not False:
        raise ValueError("Sampling with replacement must be disabled")
    if _contains_absolute_path(config):
        raise ValueError("Tracked raw-protocol config contains an absolute path")


def _verify_tensor_references(
    frame: pd.DataFrame,
    *,
    repository_root: Path,
    expected_channels: int,
    expected_samples: int,
    finite_windows_per_shard: int,
) -> dict[str, Any]:
    shard_rows: list[dict[str, Any]] = []
    checked_windows = 0
    for cache_file, rows in frame.groupby("cache_file", sort=True):
        relative = _relative_text(cache_file)
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Raw cache shard is missing: {relative}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_tail = (expected_channels, expected_samples)
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(f"Invalid raw shard shape for {relative}: {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"Invalid raw shard dtype for {relative}: {array.dtype}")
        offsets = rows["cache_offset"].astype(int).to_numpy()
        if len(np.unique(offsets)) != len(offsets):
            raise ValueError(f"Duplicate cache offsets in {relative}")
        if offsets.min() < 0 or offsets.max() >= len(array):
            raise ValueError(f"Unresolvable cache offset in {relative}")
        selected = sorted(set(offsets[:finite_windows_per_shard].tolist() + offsets[-finite_windows_per_shard:].tolist()))
        for offset in selected:
            window = np.asarray(array[offset])
            if window.shape != expected_tail or not np.isfinite(window).all():
                raise ValueError(f"Invalid cached window {relative}@{offset}")
        checked_windows += len(selected)
        shard_rows.append({
            "cache_file": relative,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "referenced_offsets": int(len(offsets)),
            "minimum_offset": int(offsets.min()),
            "maximum_offset": int(offsets.max()),
        })
    return {
        "all_tensor_references_resolved": True,
        "all_shard_shapes_valid": True,
        "all_shard_dtypes_float32": True,
        "referenced_samples": int(len(frame)),
        "shards": len(shard_rows),
        "finite_windows_sampled": checked_windows,
        "shard_inventory_hash": stable_hash(shard_rows),
    }


def load_raw_deduplicated_universe(
    config: Mapping[str, Any], *, repository_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load canonical metadata and verify mmap references without loading all EEG."""
    dataset = config["dataset"]
    manifest_path = repository_root / str(dataset["manifest"])
    logical_map_path = repository_root / str(dataset["logical_recording_map"])
    manifest = ensure_record_group_ids(pd.read_parquet(manifest_path))
    logical_map = pd.read_parquet(logical_map_path)
    required_map = {"record_group_id", "selected_record_id"}
    if required_map - set(logical_map):
        raise ValueError("Logical recording map has an incompatible schema")
    if logical_map["record_group_id"].astype(str).duplicated().any():
        raise ValueError("Logical recording map contains duplicate record groups")
    selected_records = set(logical_map["selected_record_id"].astype(str))
    frame = manifest.loc[
        manifest["status"].astype(str).eq("ok")
        & manifest["record_id"].astype(str).isin(selected_records)
    ].copy()
    required = {
        "sample_id", "subject_id", "record_id", "record_group_id", "source",
        "label_q5", "outer_fold", "absolute_t_start", "absolute_t_end",
        "raw_timestamp_min", "cache_file", "cache_offset", "n_channels",
        "n_samples_expected", "sfreq_target", "preprocessing_hash",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Raw manifest is missing columns: {missing}")
    if len(frame) != int(dataset["expected_samples"]):
        raise ValueError(f"Expected {dataset['expected_samples']} raw samples, found {len(frame)}")
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Raw universe sample IDs are missing or duplicated")
    for column in ("subject_id", "record_id", "record_group_id"):
        if frame[column].isna().any():
            raise ValueError(f"Raw universe contains missing {column}")
        frame[column] = frame[column].astype(str)
    labels = pd.to_numeric(frame["label_q5"], errors="raise").astype(int)
    if frame["label_q5"].isna().any() or not set(labels).issubset(VALID_LABELS):
        raise ValueError("Raw universe contains missing or invalid label_q5")
    frame["label_q5"] = labels
    frame["target"] = labels
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["cache_file"] = frame["cache_file"].map(_relative_text)
    frame = frame.sort_values(
        ["subject_id", "absolute_t_start", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    expected_channels = int(dataset["channels"])
    expected_samples = int(dataset["samples_per_window"])
    if set(frame["n_channels"].astype(int)) != {expected_channels}:
        raise ValueError("Raw universe channel count changed")
    if set(frame["n_samples_expected"].astype(int)) != {expected_samples}:
        raise ValueError("Raw universe window length changed")
    if set(frame["sfreq_target"].astype(float)) != {float(dataset["sampling_rate"])}:
        raise ValueError("Raw universe sampling rate changed")
    selected_per_group = frame.drop_duplicates("record_id")["record_group_id"]
    if selected_per_group.duplicated().any():
        raise ValueError("More than one source record survived per logical recording")
    tensor_audit = _verify_tensor_references(
        frame,
        repository_root=repository_root,
        expected_channels=expected_channels,
        expected_samples=expected_samples,
        finite_windows_per_shard=int(dataset.get("finite_windows_per_shard", 1)),
    )
    inventory_columns = [
        "sample_id", "source", "subject_id", "record_id", "record_group_id",
        "label_q5", "outer_fold", "absolute_t_start", "absolute_t_end",
        "raw_timestamp_min", "cache_file", "cache_offset", "n_channels",
        "n_samples_expected", "sfreq_target", "preprocessing_hash",
    ]
    inventory = frame[inventory_columns].copy()
    sample_id_hash = stable_hash(sorted(inventory["sample_id"].tolist()))
    metadata_hash = stable_hash(inventory.to_dict("records"))
    cache_parents = sorted({posixpath.dirname(value) for value in inventory["cache_file"]})
    cache_path = posixpath.commonpath(cache_parents)
    preprocessing_hashes = sorted(inventory["preprocessing_hash"].astype(str).unique())
    universe_core = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset["dataset_id"],
        "cache_id": dataset["cache_id"],
        "cache_path": cache_path,
        "dataset_mode": "raw_deduplicated_logical_records",
        "sample_count": int(len(inventory)),
        "subject_count": int(inventory["subject_id"].nunique()),
        "record_count": int(inventory["record_id"].nunique()),
        "logical_record_count": int(inventory["record_group_id"].nunique()),
        "class_counts": {
            str(label): int(count)
            for label, count in inventory["label_q5"].value_counts().sort_index().items()
        },
        "channel_order": list(CANONICAL_EEG_CHANNELS),
        "channels": expected_channels,
        "samples_per_window": expected_samples,
        "sampling_rate": float(dataset["sampling_rate"]),
        "window_duration_seconds": float(dataset["window_duration_seconds"]),
        "deduplication_policy": (
            "logical_recording_map:accepted_fraction_desc,available_eeg_samples_desc,"
            "mean_missing_fraction_asc,source_priority_asc,record_id_lexical_asc"
        ),
        "preprocessing_hashes": preprocessing_hashes,
        "sample_id_hash": sample_id_hash,
        "metadata_hash": metadata_hash,
        "duplicate_sample_ids": 0,
        "invalid_labels": 0,
        "tensor_audit": tensor_audit,
    }
    universe_core["raw_universe_hash"] = stable_hash(universe_core)
    return inventory, universe_core


def audit_outer_fold(
    config: Mapping[str, Any], raw_inventory: pd.DataFrame, *, repository_root: Path
) -> dict[str, Any]:
    source_path = repository_root / str(config["outer_fold_assignments"])
    source_hash = _sha256_file(source_path)
    if source_hash != str(config["expected_outer_fold_artifact_sha256"]):
        raise ValueError("Existing outer-fold artifact hash changed")
    assignments = pd.read_parquet(source_path, columns=["subject_id", "fold"])
    assignments["subject_id"] = assignments["subject_id"].astype(str)
    per_subject = assignments.drop_duplicates().sort_values(["subject_id", "fold"])
    if per_subject["subject_id"].duplicated().any():
        raise ValueError("Outer-fold artifact assigns a subject to multiple folds")
    raw_subject_folds = (
        raw_inventory[["subject_id", "outer_fold"]].drop_duplicates()
        .sort_values(["subject_id", "outer_fold"])
    )
    if raw_subject_folds["subject_id"].duplicated().any():
        raise ValueError("Raw universe assigns a subject to multiple folds")
    left = dict(zip(per_subject["subject_id"], per_subject["fold"].astype(int)))
    right = dict(zip(raw_subject_folds["subject_id"], raw_subject_folds["outer_fold"].astype(int)))
    if left != right:
        raise ValueError("Raw universe is incompatible with the existing outer folds")
    old_protocol = json.loads(
        (repository_root / str(config["old_protocol_manifest"])).read_text(encoding="utf-8")
    )
    if old_protocol["protocol_hash"] != config["old_protocol_hash"]:
        raise ValueError("Blocked protocol hash changed")
    fold = int(config["outer_fold"])
    outer_test = sorted(subject for subject, value in right.items() if value == fold)
    outer_train = sorted(subject for subject, value in right.items() if value != fold)
    if outer_train != sorted(old_protocol["outer_train_subjects"]):
        raise ValueError("Outer-train subject list changed")
    if outer_test != sorted(old_protocol["outer_test_subjects"]):
        raise ValueError("Outer-test subject list changed")
    semantic_payload = {
        "protocol": "group_kfold_subject",
        "outer_fold": fold,
        "assignments": [[subject, right[subject]] for subject in sorted(right)],
    }
    return {
        "outer_fold": fold,
        "source_fold_assignments_sha256": source_hash,
        "source_hash_matches_blocked_protocol": (
            source_hash == old_protocol["source_split_hash"]
        ),
        "outer_split_hash": stable_hash(semantic_payload),
        "outer_train_subjects": outer_train,
        "outer_test_subjects": outer_test,
        "outer_train_count": len(outer_train),
        "outer_test_count": len(outer_test),
        "outer_subject_overlap": len(set(outer_train) & set(outer_test)),
        "raw_subject_universe_matches": True,
    }


def _subject_partition(
    group: pd.DataFrame, *, support_record_count: int
) -> dict[str, Any]:
    records = (
        group.groupby(["record_id", "record_group_id"], sort=False)
        .agg(
            windows=("sample_id", "size"),
            start=("absolute_t_start", "min"),
            end=("absolute_t_end", "max"),
        )
        .reset_index()
        .sort_values(["start", "record_group_id", "record_id"], kind="stable")
    )
    if len(records) < support_record_count + 1:
        return {"records": records, "support": group.iloc[0:0], "query": group.iloc[0:0]}
    support_records = records.iloc[:support_record_count]
    query_records = records.iloc[support_record_count:]
    support_ids = set(support_records["record_id"].astype(str))
    query_ids = set(query_records["record_id"].astype(str))
    support = group.loc[group["record_id"].isin(support_ids)].sort_values(
        ["absolute_t_start", "sample_id"], kind="stable"
    )
    query = group.loc[group["record_id"].isin(query_ids)].sort_values(
        ["absolute_t_start", "sample_id"], kind="stable"
    )
    return {
        "records": records,
        "support_records": support_records,
        "query_records": query_records,
        "support": support,
        "query": query,
    }


def audit_participant_eligibility(
    raw_inventory: pd.DataFrame,
    outer_audit: Mapping[str, Any],
    episode_contract: Mapping[str, Any],
) -> pd.DataFrame:
    support_record_count = int(episode_contract["support_record_count"])
    minimum_support = int(episode_contract["minimum_support_windows"])
    minimum_query = int(episode_contract["minimum_query_windows"])
    train_subjects = set(outer_audit["outer_train_subjects"])
    rows: list[dict[str, Any]] = []
    for subject, group in raw_inventory.groupby("subject_id", sort=True):
        partition = _subject_partition(group, support_record_count=support_record_count)
        records = partition["records"]
        support = partition["support"]
        query = partition["query"]
        support_classes = sorted(support["label_q5"].astype(int).unique().tolist())
        query_classes = sorted(query["label_q5"].astype(int).unique().tolist())
        support_end = float(support["absolute_t_end"].max()) if len(support) else None
        query_start = float(query["absolute_t_start"].min()) if len(query) else None
        chronology_verified = bool(
            support_end is not None and query_start is not None and support_end < query_start
        )
        reasons: list[str] = []
        if len(records) < support_record_count + 1:
            reasons.append("insufficient_independent_records")
        if len(support) < minimum_support:
            reasons.append("support_below_minimum")
        if len(query) < minimum_query:
            reasons.append("query_below_minimum")
        base_eligible = not reasons
        missing_support = sorted(set(VALID_LABELS) - set(support_classes))
        missing_query = sorted(set(VALID_LABELS) - set(query_classes))
        policy_eligible = base_eligible and not missing_support and not missing_query
        rows.append({
            "subject_id": str(subject),
            "outer_partition": "outer_train" if subject in train_subjects else "outer_test",
            "raw_sample_count": int(len(group)),
            "record_count": int(len(records)),
            "class_count": int(group["label_q5"].nunique()),
            "class_distribution": json.dumps({
                str(label): int(count)
                for label, count in group["label_q5"].value_counts().sort_index().items()
            }, sort_keys=True),
            "chronological_metadata_available": bool(
                group[["absolute_t_start", "absolute_t_end"]].notna().all().all()
            ),
            "chronology_verified": chronology_verified,
            "record_disjoint_possible": len(records) >= support_record_count + 1,
            "support_record_count": int(partition.get("support_records", pd.DataFrame()).shape[0]),
            "query_record_count": int(partition.get("query_records", pd.DataFrame()).shape[0]),
            "support_count": int(len(support)),
            "query_count": int(len(query)),
            "support_record_ids": json.dumps(
                sorted(support["record_group_id"].astype(str).unique().tolist())
            ),
            "query_record_ids": json.dumps(
                sorted(query["record_group_id"].astype(str).unique().tolist())
            ),
            "support_classes": json.dumps(support_classes),
            "query_classes": json.dumps(query_classes),
            "support_missing_classes": json.dumps(missing_support),
            "query_missing_classes": json.dumps(missing_query),
            "maximum_support_windows_whole_records": int(len(support)),
            "maximum_query_windows_whole_records": int(len(query)),
            "base_eligible": base_eligible,
            "none_eligible": base_eligible,
            "at_least_one_per_class_eligible": policy_eligible,
            "require_all_classes_eligible": policy_eligible,
            "base_reasons": ";".join(reasons),
        })
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def build_class_policy_audit(
    eligibility: pd.DataFrame, *, selected_policy: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selected_policy not in CLASS_POLICIES:
        raise ValueError(f"Unknown class policy: {selected_policy}")
    definitions = {
        "none": "no class-completeness constraint beyond valid labels",
        "at_least_one_per_class": "support and query each contain labels 0,1,2,3,4",
        "require_all_classes": "support and query each contain all five task classes",
    }
    rows = []
    for partition in ("outer_train", "outer_test"):
        scoped = eligibility.loc[eligibility["outer_partition"].eq(partition)]
        for policy in CLASS_POLICIES:
            eligible_column = f"{policy}_eligible"
            rows.append({
                "outer_partition": partition,
                "policy": policy,
                "definition": definitions[policy],
                "participants": int(len(scoped)),
                "eligible_participants": int(scoped[eligible_column].sum()),
                "ineligible_participants": int((~scoped[eligible_column]).sum()),
                "selected": policy == selected_policy,
                "selection_scope": "outer_train_only",
            })
    audit = pd.DataFrame(rows)
    train_row = audit.loc[
        audit["outer_partition"].eq("outer_train")
        & audit["policy"].eq(selected_policy)
    ].iloc[0]
    selection = {
        "selected_policy": selected_policy,
        "selection_scope": "outer_train_only",
        "outer_test_used_for_selection": False,
        "eligible_outer_train_participants": int(train_row["eligible_participants"]),
        "automatic_policy_weakening": False,
        "oversampling": False,
    }
    return audit, selection


def audit_support_budget(
    eligibility: pd.DataFrame, episode_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scoped = eligibility.loc[
        eligibility["outer_partition"].eq("outer_train")
        & eligibility["base_eligible"]
    ].copy()
    scoped = scoped[[
        "subject_id", "record_count", "support_record_count", "query_record_count",
        "support_count", "query_count", "chronology_verified",
    ]]
    scoped["selection_scope"] = "outer_train_only"
    scoped["whole_records_only"] = True
    summary = {
        "selection_scope": "outer_train_only",
        "outer_test_used": False,
        "mode": "fixed_complete_early_records",
        "support_record_count": int(episode_contract["support_record_count"]),
        "query_record_count": "all_remaining_records",
        "minimum_support_windows": int(episode_contract["minimum_support_windows"]),
        "minimum_query_windows": int(episode_contract["minimum_query_windows"]),
        "fixed_window_cap": None,
        "window_level_fallback": False,
        "sampling_with_replacement": False,
        "eligible_outer_train_participants_checked": int(len(scoped)),
        "support_windows": {
            "minimum": int(scoped["support_count"].min()),
            "median": float(scoped["support_count"].median()),
            "maximum": int(scoped["support_count"].max()),
        },
        "query_windows": {
            "minimum": int(scoped["query_count"].min()),
            "median": float(scoped["query_count"].median()),
            "maximum": int(scoped["query_count"].max()),
        },
    }
    return scoped, summary


def choose_meta_validation_subjects(
    raw_inventory: pd.DataFrame,
    eligible_outer_train_subjects: Sequence[str],
    *,
    fraction: float,
    minimum_subjects: int,
    seed: int,
) -> dict[str, Any]:
    """Select a deterministic class-balanced subset using outer-train only."""
    subjects = sorted(str(value) for value in eligible_outer_train_subjects)
    if len(subjects) <= minimum_subjects:
        raise ValueError("Too few eligible outer-train participants for meta-validation")
    requested = max(minimum_subjects, int(round(len(subjects) * fraction)))
    requested = min(requested, len(subjects) - 1)
    scoped = raw_inventory.loc[raw_inventory["subject_id"].isin(subjects)]
    subject_counts = (
        scoped.groupby(["subject_id", "label_q5"]).size().unstack(fill_value=0)
        .reindex(index=subjects, columns=VALID_LABELS, fill_value=0)
    )
    overall = subject_counts.sum(axis=0).to_numpy(dtype=float)
    overall /= overall.sum()
    best: tuple[float, float, str, tuple[str, ...]] | None = None
    for candidate in itertools.combinations(subjects, requested):
        counts = subject_counts.loc[list(candidate)].sum(axis=0).to_numpy(dtype=float)
        proportions = counts / counts.sum()
        deviations = np.abs(proportions - overall)
        key = (
            float(deviations.max()),
            float(deviations.sum()),
            stable_hash({"seed": seed, "subjects": candidate}),
            candidate,
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise RuntimeError("Meta-validation selection produced no candidate")
    validation = sorted(best[3])
    train = sorted(set(subjects) - set(validation))
    payload = {
        "seed": seed,
        "algorithm": "exhaustive_minimax_class_proportion_deviation_then_seed_hash",
        "selection_scope": "eligible_outer_train_only",
        "outer_test_used": False,
        "requested_fraction": fraction,
        "minimum_subjects": minimum_subjects,
        "meta_train_subjects": train,
        "meta_validation_subjects": validation,
        "maximum_class_proportion_deviation": best[0],
        "total_class_proportion_deviation": best[1],
    }
    payload["meta_split_hash"] = stable_hash(payload)
    return payload


def episode_identifier(payload: Mapping[str, Any]) -> str:
    return stable_hash(payload)


def _episode_rows_for_subject(
    raw_inventory: pd.DataFrame,
    *,
    subject: str,
    scope: str,
    dataset_id: str,
    cache_id: str,
    raw_universe_hash: str,
    outer_fold: int,
    meta_split_hash: str,
    episode_spec: Mapping[str, Any],
    episode_spec_hash: str,
    seed: int,
) -> dict[str, Any]:
    group = raw_inventory.loc[raw_inventory["subject_id"].eq(subject)].copy()
    partition = _subject_partition(
        group, support_record_count=int(episode_spec["support_record_count"])
    )
    support = partition["support"]
    query = partition["query"]
    support_ids = support["sample_id"].astype(str).tolist()
    query_ids = query["sample_id"].astype(str).tolist()
    support_records = support["record_group_id"].astype(str).drop_duplicates().tolist()
    query_records = query["record_group_id"].astype(str).drop_duplicates().tolist()
    identity = {
        "dataset_id": dataset_id,
        "cache_id": cache_id,
        "raw_universe_hash": raw_universe_hash,
        "outer_fold": outer_fold,
        "meta_split_hash": meta_split_hash,
        "scope": scope,
        "subject_id": subject,
        "support_sample_ids": support_ids,
        "query_sample_ids": query_ids,
        "support_record_ids": support_records,
        "query_record_ids": query_records,
        "episode_spec_hash": episode_spec_hash,
        "seed": seed,
    }
    episode_id = episode_identifier(identity)
    boundary = (
        f"absolute_time:{float(support['absolute_t_end'].max())}"
        f"<{float(query['absolute_t_start'].min())}"
    )
    episode = SimpleNamespace(
        episode_id=episode_id,
        support_sample_ids=tuple(support_ids),
        query_sample_ids=tuple(query_ids),
        support_record_ids=tuple(support_records),
        query_record_ids=tuple(query_records),
        subject_id=subject,
        session_ids=("session-unspecified",),
        split_level="record",
    )
    episode_audit = validate_episode(episode, require_record_disjoint=True)
    if not episode_audit.valid:
        raise ValueError(f"Invalid episode for {subject}: {episode_audit.errors}")
    return {
        "episode_id": episode_id,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": dataset_id,
        "cache_id": cache_id,
        "task_id": "cognitive_load_5class",
        "target_name": "label_q5",
        "fold_id": str(outer_fold),
        "scope": scope,
        "entity_id": subject,
        "subject_id": subject,
        "support_sample_ids": support_ids,
        "query_sample_ids": query_ids,
        "support_record_ids": support_records,
        "query_record_ids": query_records,
        "support_source_record_ids": support["record_id"].astype(str).drop_duplicates().tolist(),
        "query_source_record_ids": query["record_id"].astype(str).drop_duplicates().tolist(),
        "support_targets": support["label_q5"].astype(int).tolist(),
        "query_targets": query["label_q5"].astype(int).tolist(),
        "support_start": float(support["absolute_t_start"].min()),
        "support_end": float(support["absolute_t_end"].max()),
        "query_start": float(query["absolute_t_start"].min()),
        "query_end": float(query["absolute_t_end"].max()),
        "chronological_boundary": boundary,
        "chronology_verified": float(support["absolute_t_end"].max()) < float(query["absolute_t_start"].min()),
        "split_level": "record",
        "support_count": len(support_ids),
        "query_count": len(query_ids),
        "seed": seed,
        "episode_spec_hash": episode_spec_hash,
        "meta_split_hash": meta_split_hash,
    }


def build_episode_index(
    config: Mapping[str, Any],
    raw_inventory: pd.DataFrame,
    universe_manifest: Mapping[str, Any],
    outer_audit: Mapping[str, Any],
    eligibility: pd.DataFrame,
    meta_split: Mapping[str, Any],
    class_policy: Mapping[str, Any],
    support_budget: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = str(class_policy["selected_policy"])
    eligible_column = f"{policy}_eligible"
    eligible = eligibility.loc[eligibility[eligible_column]].copy()
    eligible_test = sorted(
        eligible.loc[eligible["outer_partition"].eq("outer_test"), "subject_id"]
    )
    episode_spec = {
        "episode_type": "subject_personalization",
        "task_type": "ordinal_classification",
        "target_name": "label_q5",
        "split_level": "record",
        "support_record_count": int(config["episode_contract"]["support_record_count"]),
        "query_record_count": "all_remaining_records",
        "minimum_support_windows": int(config["episode_contract"]["minimum_support_windows"]),
        "minimum_query_windows": int(config["episode_contract"]["minimum_query_windows"]),
        "class_policy": policy,
        "chronological": True,
        "window_level_fallback": False,
        "sampling_with_replacement": False,
        "oversampling": False,
        "support_budget_selection_scope": support_budget["selection_scope"],
        "seed": int(config["seed"]),
    }
    episode_spec_hash = stable_hash(episode_spec)
    rows = []
    scopes = (
        ("meta_train", meta_split["meta_train_subjects"]),
        ("meta_validation", meta_split["meta_validation_subjects"]),
        ("outer_test", eligible_test),
    )
    for scope, subjects in scopes:
        for subject in sorted(subjects):
            rows.append(_episode_rows_for_subject(
                raw_inventory,
                subject=subject,
                scope=scope,
                dataset_id=str(config["dataset"]["dataset_id"]),
                cache_id=str(config["dataset"]["cache_id"]),
                raw_universe_hash=str(universe_manifest["raw_universe_hash"]),
                outer_fold=int(config["outer_fold"]),
                meta_split_hash=str(meta_split["meta_split_hash"]),
                episode_spec=episode_spec,
                episode_spec_hash=episode_spec_hash,
                seed=int(config["seed"]),
            ))
    frame = pd.DataFrame(rows).sort_values(["scope", "subject_id"]).reset_index(drop=True)
    return frame, {**episode_spec, "episode_spec_hash": episode_spec_hash}


def audit_episode_index(
    episode_index: pd.DataFrame,
    raw_inventory: pd.DataFrame,
    outer_audit: Mapping[str, Any],
    meta_split: Mapping[str, Any],
) -> dict[str, Any]:
    universe_ids = set(raw_inventory["sample_id"].astype(str))
    all_ids: list[str] = []
    sample_overlap = 0
    record_overlap = 0
    chronology_failures = 0
    subject_mismatches = 0
    for row in episode_index.itertuples():
        support = set(map(str, row.support_sample_ids))
        query = set(map(str, row.query_sample_ids))
        sample_overlap += len(support & query)
        record_overlap += len(set(row.support_record_ids) & set(row.query_record_ids))
        chronology_failures += int(not bool(row.chronology_verified))
        all_ids.extend(support)
        all_ids.extend(query)
        scoped_subjects = set(
            raw_inventory.loc[raw_inventory["sample_id"].isin(support | query), "subject_id"]
        )
        subject_mismatches += int(scoped_subjects != {str(row.subject_id)})
    train = set(meta_split["meta_train_subjects"])
    validation = set(meta_split["meta_validation_subjects"])
    outer_test = set(outer_audit["outer_test_subjects"])
    missing = sorted(set(all_ids) - universe_ids)
    audit = {
        "valid": False,
        "episodes": int(len(episode_index)),
        "episode_ids_unique": not episode_index["episode_id"].duplicated().any(),
        "duplicate_episode_sample_references": len(all_ids) - len(set(all_ids)),
        "missing_raw_ids": len(missing),
        "support_query_sample_overlap": sample_overlap,
        "support_query_record_overlap": record_overlap,
        "episode_subject_mismatches": subject_mismatches,
        "chronology_failures": chronology_failures,
        "within_record_fallbacks": int(episode_index["split_level"].ne("record").sum()),
        "meta_train_validation_subject_overlap": len(train & validation),
        "meta_train_outer_test_subject_overlap": len(train & outer_test),
        "meta_validation_outer_test_subject_overlap": len(validation & outer_test),
        "all_tensor_references_resolved": True,
        "sampling_with_replacement": False,
        "oversampling": False,
    }
    audit["valid"] = all([
        audit["episode_ids_unique"],
        audit["duplicate_episode_sample_references"] == 0,
        audit["missing_raw_ids"] == 0,
        audit["support_query_sample_overlap"] == 0,
        audit["support_query_record_overlap"] == 0,
        audit["episode_subject_mismatches"] == 0,
        audit["chronology_failures"] == 0,
        audit["within_record_fallbacks"] == 0,
        audit["meta_train_validation_subject_overlap"] == 0,
        audit["meta_train_outer_test_subject_overlap"] == 0,
        audit["meta_validation_outer_test_subject_overlap"] == 0,
    ])
    return audit


def compute_protocol_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(payload)


def _episode_balance(episode_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode in episode_index.itertuples():
        row = {
            "episode_id": episode.episode_id,
            "scope": episode.scope,
            "subject_id": episode.subject_id,
            "support_count": episode.support_count,
            "query_count": episode.query_count,
        }
        for label in VALID_LABELS:
            row[f"support_class_{label}"] = int(np.sum(np.asarray(episode.support_targets) == label))
            row[f"query_class_{label}"] = int(np.sum(np.asarray(episode.query_targets) == label))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_old_and_new_protocols(
    old_episode_index: pd.DataFrame,
    new_episode_index: pd.DataFrame,
    raw_inventory: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    selected_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_ids = set(raw_inventory["sample_id"].astype(str))
    new_by_subject = {
        str(row.subject_id): row for row in new_episode_index.itertuples()
    }
    eligibility_by_subject = eligibility.set_index("subject_id")
    rows = []
    for old in old_episode_index.sort_values("subject_id").itertuples():
        old_ids = list(map(str, old.support_sample_ids)) + list(map(str, old.query_sample_ids))
        compatible = sum(sample_id in raw_ids for sample_id in old_ids)
        new = new_by_subject.get(str(old.subject_id))
        eligible_row = eligibility_by_subject.loc[str(old.subject_id)]
        reasons = []
        if not bool(eligible_row["base_eligible"]):
            reasons.extend(filter(None, str(eligible_row["base_reasons"]).split(";")))
        for partition in ("support", "query"):
            missing_classes = json.loads(eligible_row[f"{partition}_missing_classes"])
            if missing_classes and selected_policy != "none":
                reasons.append(f"{partition}_missing_classes={missing_classes}")
        rows.append({
            "old_feature_level_episode": str(old.episode_id),
            "old_subject": str(old.subject_id),
            "old_scope": str(old.scope),
            "old_support_count": len(old.support_sample_ids),
            "old_query_count": len(old.query_sample_ids),
            "raw_compatible_old_ids": compatible,
            "missing_old_ids": len(old_ids) - compatible,
            "old_episode_fully_raw_compatible": compatible == len(old_ids),
            "new_raw_episode_id": None if new is None else str(new.episode_id),
            "new_support_count": None if new is None else int(new.support_count),
            "new_query_count": None if new is None else int(new.query_count),
            "subject_retained": new is not None,
            "reason": "new_raw_protocol_episode" if new is not None else ";".join(reasons),
        })
    comparison = pd.DataFrame(rows)
    common = comparison.loc[comparison["subject_retained"]]
    summary = {
        "old_episodes": int(len(comparison)),
        "old_episodes_fully_raw_compatible": int(comparison["old_episode_fully_raw_compatible"].sum()),
        "new_episodes": int(len(new_episode_index)),
        "common_participants": int(len(common)),
        "changed_common_participants": int(len(common)),
        "old_episode_ids_reused": int(
            len(set(comparison["old_feature_level_episode"]) & set(new_episode_index["episode_id"]))
        ),
        "old_missing_ids": int(comparison["missing_old_ids"].sum()),
        "remapping_performed": False,
    }
    return comparison, summary


def _write_immutable_preregistration(path: Path, payload: Mapping[str, Any]) -> str:
    content = (
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError("Existing raw-protocol preregistration differs")
    if not path.exists():
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _runtime_report(summary: Mapping[str, Any]) -> str:
    counts = summary["episode_counts"]
    return (
        "# Raw-deduplicated FOMAML episode protocol\n\n"
        f"- Decision: `{summary['readiness_status']}`.\n"
        f"- Raw universe: {summary['raw_samples']:,} windows.\n"
        f"- Eligible outer-train / outer-test: "
        f"{summary['eligible_outer_train']} / {summary['eligible_outer_test']}.\n"
        f"- Meta-train / meta-validation / outer-test episodes: "
        f"{counts['meta_train']} / {counts['meta_validation']} / {counts['outer_test']}.\n"
        f"- Protocol hash: `{summary['protocol_hash']}`.\n"
        f"- Preregistration hash: `{summary['preregistration_hash']}`.\n"
        "- Execution enabled: `false`; training, optimizer, CUDA tensors, and checkpoints: absent.\n"
    )


def build_fomaml_label_q5_raw_protocol(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    output_dir: Path | None = None,
) -> RawProtocolBuildResult:
    """Materialize protocol manifests and a disabled preregistration only."""
    validate_raw_protocol_config(config)
    old_hashes_before = _directory_hashes(
        repository_root, config["immutable_runtime_directories"]
    )
    raw_inventory, universe = load_raw_deduplicated_universe(
        config, repository_root=repository_root
    )
    outer = audit_outer_fold(config, raw_inventory, repository_root=repository_root)
    eligibility = audit_participant_eligibility(
        raw_inventory, outer, config["episode_contract"]
    )
    policy_audit, policy_selection = build_class_policy_audit(
        eligibility, selected_policy=str(config["selected_class_policy"])
    )
    selected_column = f"{config['selected_class_policy']}_eligible"
    eligible_train = sorted(
        eligibility.loc[
            eligibility["outer_partition"].eq("outer_train")
            & eligibility[selected_column],
            "subject_id",
        ]
    )
    eligible_test = sorted(
        eligibility.loc[
            eligibility["outer_partition"].eq("outer_test")
            & eligibility[selected_column],
            "subject_id",
        ]
    )
    if len(eligible_train) < int(config["meta_validation"]["minimum_subjects"]) + 1:
        raise ValueError("Selected class policy leaves too few outer-train participants")
    support_rows, support_summary = audit_support_budget(
        eligibility, config["episode_contract"]
    )
    meta_split = choose_meta_validation_subjects(
        raw_inventory,
        eligible_train,
        fraction=float(config["meta_validation"]["fraction"]),
        minimum_subjects=int(config["meta_validation"]["minimum_subjects"]),
        seed=int(config["seed"]),
    )
    episode_index, episode_spec = build_episode_index(
        config, raw_inventory, universe, outer, eligibility, meta_split,
        policy_selection, support_summary,
    )
    leakage = audit_episode_index(episode_index, raw_inventory, outer, meta_split)
    old_episode_path = repository_root / str(config["old_episode_index"])
    old_episodes = pd.read_parquet(old_episode_path)
    comparison, comparison_summary = compare_old_and_new_protocols(
        old_episodes, episode_index, raw_inventory, eligibility,
        selected_policy=str(config["selected_class_policy"]),
    )
    episode_records = episode_index.to_dict("records")
    episode_manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "episodes": episode_records,
    }
    episode_manifest_hash = stable_hash(episode_manifest_core)
    protocol_core = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "relation_to_blocked_experiment": {
            "experiment": "fomaml_label_q5_diagnostic_task_8X",
            "status": config["old_status"],
            "protocol_hash": config["old_protocol_hash"],
            "preregistration_hash": config["old_preregistration_hash"],
            "repair_or_remap": False,
        },
        "dataset_cache_signature": {
            key: universe[key]
            for key in (
                "dataset_id", "cache_id", "cache_path", "raw_universe_hash",
                "sample_count", "subject_count", "record_count",
                "logical_record_count", "sample_id_hash", "metadata_hash",
            )
        },
        "outer_fold_manifest": outer,
        "eligible_participants": {
            "outer_train": eligible_train,
            "outer_test": eligible_test,
        },
        "ineligible_participants": eligibility.loc[
            ~eligibility[selected_column], ["subject_id", "outer_partition", "base_reasons",
                                            "support_missing_classes", "query_missing_classes"]
        ].to_dict("records"),
        "meta_split": meta_split,
        "episode_spec": episode_spec,
        "class_policy": policy_selection,
        "support_budget": support_summary,
        "episode_manifest_hash": episode_manifest_hash,
        "episodes": episode_records,
        "leakage_audit": leakage,
        "seed": int(config["seed"]),
        "execution_enabled": False,
    }
    protocol_hash = compute_protocol_hash(protocol_core)
    protocol_manifest = {**protocol_core, "protocol_hash": protocol_hash}
    old_hashes_after_audit = _directory_hashes(
        repository_root, config["immutable_runtime_directories"]
    )
    old_unchanged = old_hashes_before == old_hashes_after_audit
    core_ready = all([
        leakage["valid"],
        outer["outer_subject_overlap"] == 0,
        outer["source_hash_matches_blocked_protocol"],
        old_unchanged,
        comparison_summary["old_episode_ids_reused"] == 0,
        len(meta_split["meta_validation_subjects"]) >= int(config["meta_validation"]["minimum_subjects"]),
    ])
    readiness_status = "raw_protocol_ready" if core_ready else "blocked"
    output = output_dir or repository_root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    raw_inventory.to_parquet(output / "raw_sample_inventory.parquet", index=False)
    _write_json(output / "raw_universe_manifest.json", universe)
    _write_json(output / "outer_split_audit.json", outer)
    eligibility.to_csv(output / "participant_eligibility.csv", index=False)
    policy_audit.to_csv(output / "class_policy_audit.csv", index=False)
    support_rows.to_csv(output / "support_budget_audit.csv", index=False)
    _write_json(output / "meta_split_manifest.json", meta_split)
    _write_json(output / "episode_spec.json", episode_spec)
    episode_index.to_parquet(output / "episode_index.parquet", index=False)
    _write_json(output / "episode_manifest.json", {
        **episode_manifest_core, "episode_manifest_hash": episode_manifest_hash
    })
    _episode_balance(episode_index).to_csv(output / "episode_balance.csv", index=False)
    _write_json(output / "episode_leakage_audit.json", leakage)
    comparison.to_csv(output / "old_new_protocol_comparison.csv", index=False)
    _write_json(output / "protocol_manifest.json", protocol_manifest)
    _write_json(output / "protocol_hash.json", {"protocol_hash": protocol_hash})
    errors = eligibility.loc[~eligibility[selected_column]].copy()
    errors = errors.assign(
        error_type="IneligibleRawEpisodeParticipant",
        reason=errors.apply(
            lambda row: ";".join(filter(None, [
                str(row["base_reasons"]),
                (
                    f"support_missing_classes={row['support_missing_classes']}"
                    if row["support_missing_classes"] != "[]" else ""
                ),
                (
                    f"query_missing_classes={row['query_missing_classes']}"
                    if row["query_missing_classes"] != "[]" else ""
                ),
            ])),
            axis=1,
        ),
    )[["subject_id", "outer_partition", "error_type", "reason"]]
    errors.to_csv(output / "errors.csv", index=False)
    if not core_ready:
        preregistration_hash = None
    else:
        preregistration = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "fomaml_label_q5_raw_deduplicated_v2_diagnostic",
            "protocol_id": PROTOCOL_ID,
            "repository_commit": _git_head(repository_root),
            "relation_to_blocked_8X": {
                "status": config["old_status"],
                "protocol_hash": config["old_protocol_hash"],
                "preregistration_hash": config["old_preregistration_hash"],
                "feature_level_ids_reused_or_remapped": False,
            },
            "raw_universe_hash": universe["raw_universe_hash"],
            "protocol_hash": protocol_hash,
            "outer_fold_hash": outer["outer_split_hash"],
            "outer_fold_artifact_sha256": outer["source_fold_assignments_sha256"],
            "architecture_signature": config["future_experiment"]["architecture_signature"],
            "seed": int(config["seed"]),
            "device": config["future_experiment"]["device"],
            "participant_counts": {
                "meta_train": len(meta_split["meta_train_subjects"]),
                "meta_validation": len(meta_split["meta_validation_subjects"]),
                "outer_test": len(eligible_test),
            },
            "episode_counts": {
                str(key): int(value)
                for key, value in episode_index["scope"].value_counts().items()
            },
            "support_query_contract": episode_spec,
            "support_budget": support_summary,
            "class_policy": policy_selection,
            "inner_steps": config["future_experiment"]["inner_steps"],
            "inner_learning_rate": config["future_experiment"]["inner_learning_rate"],
            "meta_learning_rate": config["future_experiment"]["meta_learning_rate"],
            "maximum_epochs": config["future_experiment"]["maximum_epochs"],
            "meta_batch_size": config["future_experiment"]["meta_batch_size"],
            "batchnorm_policies": config["future_experiment"]["batchnorm_policies"],
            "supervised_baselines": config["future_experiment"]["supervised_baselines"],
            "checkpoint_criteria": config["future_experiment"]["checkpoint_criteria"],
            "metrics": config["future_experiment"]["metrics"],
            "decision_rule": config["future_experiment"]["decision_rule"],
            "scientific_hypothesis": config["future_experiment"]["scientific_hypothesis"],
            "protocol_audit_completed_before_preregistration": True,
            "protocol_audit_hash": stable_hash(leakage),
            "execution_enabled": False,
        }
        if _contains_absolute_path(preregistration):
            raise ValueError("Preregistration contains an absolute path")
        preregistration_path = output / "preregistration/experiment_preregistration.json"
        preregistration_hash = _write_immutable_preregistration(
            preregistration_path, preregistration
        )
        _write_json(output / "preregistration/preregistration_hash.json", {
            "sha256": preregistration_hash,
            "execution_enabled": False,
            "created_after_protocol_audit": True,
        })
    episode_counts = {
        scope: int((episode_index["scope"] == scope).sum())
        for scope in ("meta_train", "meta_validation", "outer_test")
    }
    decision = {
        "status": readiness_status,
        "protocol_audit_valid": leakage["valid"],
        "protocol_deterministic": True,
        "preregistration_created": preregistration_hash is not None,
        "preregistration_hash": preregistration_hash,
        "execution_enabled": False,
        "training_performed": False,
        "optimizer_created": False,
        "cuda_tensors_created": False,
        "checkpoint_created": False,
        "old_runtime_artifacts_unchanged": old_unchanged,
        "missing_raw_ids": leakage["missing_raw_ids"],
    }
    _write_json(output / "readiness_decision.json", decision)
    summary = {
        "readiness_status": readiness_status,
        "protocol_id": PROTOCOL_ID,
        "raw_samples": int(len(raw_inventory)),
        "raw_universe_hash": universe["raw_universe_hash"],
        "outer_split_hash": outer["outer_split_hash"],
        "outer_fold_artifact_sha256": outer["source_fold_assignments_sha256"],
        "eligible_outer_train": len(eligible_train),
        "eligible_outer_test": len(eligible_test),
        "meta_train_subjects": meta_split["meta_train_subjects"],
        "meta_validation_subjects": meta_split["meta_validation_subjects"],
        "selected_class_policy": policy_selection["selected_policy"],
        "support_budget": support_summary,
        "episode_counts": episode_counts,
        "skipped_participants": int((~eligibility[selected_column]).sum()),
        "missing_raw_ids": leakage["missing_raw_ids"],
        "leakage_audit": leakage,
        "comparison": comparison_summary,
        "episode_spec_hash": episode_spec["episode_spec_hash"],
        "episode_manifest_hash": episode_manifest_hash,
        "meta_split_hash": meta_split["meta_split_hash"],
        "protocol_hash": protocol_hash,
        "preregistration_hash": preregistration_hash,
        "execution_enabled": False,
        "training_performed": False,
        "old_runtime_artifacts_unchanged": old_unchanged,
    }
    if _contains_absolute_path(summary):
        raise ValueError("Protocol summary contains an absolute path")
    (output / "protocol_report.md").write_text(
        _runtime_report(summary), encoding="utf-8"
    )
    final_old_hashes = _directory_hashes(
        repository_root, config["immutable_runtime_directories"]
    )
    if final_old_hashes != old_hashes_before:
        raise RuntimeError("A blocked task-8X artifact changed during protocol creation")
    return RawProtocolBuildResult(
        summary=summary,
        raw_inventory=raw_inventory,
        eligibility=eligibility,
        episode_index=episode_index,
    )
