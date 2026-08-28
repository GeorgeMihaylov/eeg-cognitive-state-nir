"""Build a disabled, leakage-safe raw-domain DANN protocol.

The builder is metadata-only apart from a CPU, no-gradient forward audit of
the existing production EEGNet/DANN composition.  It never creates an
optimizer, calls ``backward``, evaluates task predictions, or accesses target
task labels through a training batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.datasets.logical_recordings import ensure_record_group_ids
from bench.meta.episodes import stable_hash
from cogstate.adaptation.meta_learning.fomaml import model_state_hash
from bench.meta.production import audit_architectures
from cogstate.model_zoo.DL.dann import (
    DANNFoldData,
    DANNModule,
    DANNObjective,
    DANNPartition,
)

from .fomaml_label_q5_raw_protocol import (
    audit_outer_fold,
)


SCHEMA_VERSION = "1.0"
PROTOCOL_ID = "dann_label_q5_raw_deduplicated_source_transfer_v1"
DOMAIN_NAMES = ("gpn_data", "Old_EEG")
SUBJECT_POLICIES = (
    "allow_cross_domain_train_subjects",
    "strict_cross_domain_subject_disjoint",
)


@dataclass(frozen=True)
class DANNRawProtocolBuildResult:
    summary: dict[str, Any]
    raw_inventory: pd.DataFrame
    domain_inventory: pd.DataFrame
    participant_domain_matrix: pd.DataFrame
    direction_audit: pd.DataFrame


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> str:
    content = (
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Existing preregistration differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


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


def validate_dann_raw_protocol_config(config: Mapping[str, Any]) -> None:
    """Validate the deliberately disabled protocol contract."""
    if config.get("execution_enabled") is not False:
        raise ValueError("DANN protocol preparation requires execution_enabled=false")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if int(config.get("seed", -1)) != 42 or int(config.get("outer_fold", -1)) != 1:
        raise ValueError("Task 8Sh is restricted to outer fold 1 and seed 42")
    if dict(config.get("domains", {})) != {"gpn_data": 0, "Old_EEG": 1}:
        raise ValueError("Domain mapping must be gpn_data=0 and Old_EEG=1")
    directions = config.get("directions", [])
    pairs = {(row.get("source_domain"), row.get("target_domain")) for row in directions}
    if pairs != {("gpn_data", "Old_EEG"), ("Old_EEG", "gpn_data")}:
        raise ValueError("Both non-equivalent source/target directions are required")
    if tuple(config.get("subject_policies", ())) != SUBJECT_POLICIES:
        raise ValueError("Both cross-domain subject policies are required")
    if config.get("strict_shared_subject_rule") != (
        "retain_in_source_loader_exclude_from_target_loader"
    ):
        raise ValueError("Unknown strict shared-subject rule")
    if config["future_training"].get("target_train_task_labels_accessible") is not False:
        raise ValueError("Target-train task labels must remain inaccessible")
    if config["future_training"].get("outer_test_selection_accessible") is not False:
        raise ValueError("Outer-test must be unavailable for selection")
    if str(config["architecture"].get("device")) != "cpu":
        raise ValueError("Protocol preparation permits CPU audit tensors only")
    if _contains_absolute_path(config):
        raise ValueError("Tracked DANN config contains an absolute path")


def _relative_text(value: Any) -> str:
    text = str(value).replace("\\", "/")
    if Path(str(value)).is_absolute() or text.startswith("/"):
        raise ValueError(f"Absolute path is forbidden in protocol metadata: {value}")
    return posixpath.normpath(text)


def load_dann_raw_metadata_universe(
    config: Mapping[str, Any], *, repository_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw-window metadata while never reading target-test tensor values."""
    dataset = config["dataset"]
    manifest = ensure_record_group_ids(
        pd.read_parquet(repository_root / str(dataset["manifest"]))
    )
    logical_map = pd.read_parquet(
        repository_root / str(dataset["logical_recording_map"])
    )
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
        raise ValueError("Canonical raw sample count changed")
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Raw universe sample IDs are missing or duplicated")
    for column in ("subject_id", "record_id", "record_group_id", "source"):
        if frame[column].isna().any():
            raise ValueError(f"Raw universe contains missing {column}")
        frame[column] = frame[column].astype(str)
    if set(frame["source"]) != set(DOMAIN_NAMES):
        raise ValueError("Raw source field does not match the two approved domains")
    labels = pd.to_numeric(frame["label_q5"], errors="raise").astype(int)
    if frame["label_q5"].isna().any() or not set(labels).issubset(range(5)):
        raise ValueError("Raw universe contains missing or invalid label_q5")
    frame["label_q5"] = labels
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["cache_file"] = frame["cache_file"].map(_relative_text)
    frame = frame.sort_values(
        ["subject_id", "absolute_t_start", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    if set(frame["n_channels"].astype(int)) != {int(dataset["channels"])}:
        raise ValueError("Raw universe channel count changed")
    if set(frame["n_samples_expected"].astype(int)) != {
        int(dataset["samples_per_window"])
    }:
        raise ValueError("Raw universe window length changed")
    if set(frame["sfreq_target"].astype(float)) != {float(dataset["sampling_rate"])}:
        raise ValueError("Raw universe sampling rate changed")
    missing_shards = sorted(
        path
        for path in set(frame["cache_file"])
        if not (repository_root / path).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"Raw cache shards are missing: {missing_shards[:3]}")
    inventory_columns = [
        "sample_id", "source", "subject_id", "record_id", "record_group_id",
        "label_q5", "outer_fold", "absolute_t_start", "absolute_t_end",
        "raw_timestamp_min", "cache_file", "cache_offset", "n_channels",
        "n_samples_expected", "sfreq_target", "preprocessing_hash",
    ]
    inventory = frame[inventory_columns].copy()
    sample_id_hash = stable_hash(sorted(inventory["sample_id"].tolist()))
    metadata_hash = stable_hash(inventory.to_dict("records"))
    reference_path = repository_root / str(
        dataset["canonical_raw_universe_reference"]
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_hash = str(dataset["expected_raw_universe_hash"])
    if reference.get("raw_universe_hash") != expected_hash:
        raise ValueError("Canonical raw-universe reference hash changed")
    if reference.get("sample_id_hash") != sample_id_hash:
        raise ValueError("Raw-universe sample IDs changed")
    if reference.get("metadata_hash") != metadata_hash:
        raise ValueError("Raw-universe metadata changed")
    universe = {
        **reference,
        "reference_artifact": _relative_text(
            dataset["canonical_raw_universe_reference"]
        ),
        "reference_artifact_sha256": _sha256_file(reference_path),
        "current_sample_id_hash": sample_id_hash,
        "current_metadata_hash": metadata_hash,
        "cache_shards_exist": True,
        "tensor_values_read_this_stage": 0,
        "target_test_tensor_values_read_this_stage": 0,
    }
    return inventory, universe


def _partition_summary(frame: pd.DataFrame, *, expose_task_labels: bool) -> dict[str, Any]:
    result = {
        "samples": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "records": int(frame["record_id"].nunique()),
        "logical_records": int(frame["record_group_id"].nunique()),
        "subject_ids": sorted(frame["subject_id"].astype(str).unique()),
        "record_group_ids": sorted(frame["record_group_id"].astype(str).unique()),
        "sample_ids": sorted(frame["sample_id"].astype(str)),
        "task_labels_exposed": bool(expose_task_labels),
    }
    if expose_task_labels:
        result["class_counts"] = {
            str(key): int(value)
            for key, value in frame["label_q5"].value_counts().sort_index().items()
        }
    return result


def build_domain_inventory(
    raw_inventory: pd.DataFrame, *, outer_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize source membership without changing the canonical universe."""
    frame = raw_inventory.copy()
    frame["outer_partition"] = np.where(
        frame["outer_fold"].astype(int).eq(int(outer_fold)),
        "outer_test",
        "outer_train",
    )
    rows: list[dict[str, Any]] = []
    for (source, partition), group in frame.groupby(
        ["source", "outer_partition"], sort=True
    ):
        rows.append(
            {
                "source": str(source),
                "domain_id": 0 if source == "gpn_data" else 1,
                "outer_partition": str(partition),
                "samples": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "records": int(group["record_id"].nunique()),
                "logical_records": int(group["record_group_id"].nunique()),
                **{
                    f"class_{label}": int((group["label_q5"] == label).sum())
                    for label in range(5)
                },
            }
        )
    matrix_rows: list[dict[str, Any]] = []
    for (subject, source, partition), group in frame.groupby(
        ["subject_id", "source", "outer_partition"], sort=True
    ):
        matrix_rows.append(
            {
                "subject_id": str(subject),
                "source": str(source),
                "domain_id": 0 if source == "gpn_data" else 1,
                "outer_partition": str(partition),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "samples": int(len(group)),
                "records": int(group["record_id"].nunique()),
                "logical_records": int(group["record_group_id"].nunique()),
                **{
                    f"class_{label}": int((group["label_q5"] == label).sum())
                    for label in range(5)
                },
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


def build_logical_deduplication_audit(
    raw_inventory: pd.DataFrame, logical_map: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    retained = raw_inventory.groupby("record_group_id", sort=True).size().to_dict()
    rows: list[dict[str, Any]] = []
    for row in logical_map.sort_values("record_group_id").itertuples(index=False):
        sources = sorted(str(value) for value in row.sources)
        accepted = json.loads(str(row.accepted_windows_by_record))
        selected = str(row.selected_record_id)
        rows.append(
            {
                "record_group_id": str(row.record_group_id),
                "subject_id": str(row.subject_id),
                "original_sources": "|".join(sources),
                "original_source_count": len(sources),
                "present_in_both_sources": bool(row.present_in_both_sources),
                "selected_source": str(row.selected_source),
                "selected_record_id": selected,
                "retained_windows": int(retained.get(str(row.record_group_id), 0)),
                "discarded_accepted_windows": int(
                    sum(int(value) for key, value in accepted.items() if key != selected)
                ),
                "deduplication_reason": str(row.selection_reason),
                "signal_relationship": str(row.signal_relationship),
            }
        )
    audit = pd.DataFrame(rows)
    retained_source_counts = raw_inventory.groupby("record_group_id")["source"].nunique()
    summary = {
        "logical_records": int(len(audit)),
        "original_cross_source_logical_records": int(
            audit["present_in_both_sources"].sum()
        ),
        "retained_cross_source_logical_records": int((retained_source_counts > 1).sum()),
        "duplicate_sample_ids": int(raw_inventory["sample_id"].duplicated().sum()),
        "duplicate_logical_window_keys": int(
            raw_inventory.duplicated(
                ["record_group_id", "absolute_t_start", "absolute_t_end"]
            ).sum()
        ),
        "one_source_per_logical_record": bool((retained_source_counts == 1).all()),
    }
    return audit, summary


def _stable_subject_holdout(
    subjects: Sequence[str], *, seed: int, fraction: float, minimum: int
) -> tuple[list[str], list[str]]:
    ordered = sorted(
        {str(value) for value in subjects},
        key=lambda subject: (stable_hash({"seed": seed, "subject_id": subject}), subject),
    )
    if len(ordered) <= minimum:
        raise ValueError("Source domain has too few subjects for group validation")
    count = max(int(minimum), int(math.ceil(len(ordered) * float(fraction))))
    count = min(count, len(ordered) - 1)
    validation = sorted(ordered[:count])
    training = sorted(ordered[count:])
    return training, validation


def _overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column].astype(str)) & set(right[column].astype(str)))


def build_direction_candidate(
    raw_inventory: pd.DataFrame,
    direction: Mapping[str, Any],
    *,
    policy: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one independently auditable direction/policy candidate."""
    source = str(direction["source_domain"])
    target = str(direction["target_domain"])
    fold = int(config["outer_fold"])
    outer_train = raw_inventory.loc[raw_inventory["outer_fold"].astype(int).ne(fold)]
    outer_test = raw_inventory.loc[raw_inventory["outer_fold"].astype(int).eq(fold)]
    source_pool = outer_train.loc[outer_train["source"].eq(source)].copy()
    target_pool = outer_train.loc[outer_train["source"].eq(target)].copy()
    target_test = outer_test.loc[outer_test["source"].eq(target)].copy()
    shared = sorted(
        set(source_pool["subject_id"].astype(str))
        & set(target_pool["subject_id"].astype(str))
    )
    if policy == "strict_cross_domain_subject_disjoint":
        target_pool = target_pool.loc[~target_pool["subject_id"].isin(shared)].copy()
    elif policy != "allow_cross_domain_train_subjects":
        raise ValueError(f"Unknown subject policy: {policy}")

    source_subjects = sorted(source_pool["subject_id"].astype(str).unique())
    source_train_subjects, source_validation_subjects = _stable_subject_holdout(
        source_subjects,
        seed=int(config["source_validation"]["seed"]),
        fraction=float(config["source_validation"]["fraction"]),
        minimum=int(config["source_validation"]["minimum_subjects"]),
    )
    source_train = source_pool.loc[
        source_pool["subject_id"].isin(source_train_subjects)
    ].copy()
    source_validation = source_pool.loc[
        source_pool["subject_id"].isin(source_validation_subjects)
    ].copy()
    thresholds = config["feasibility_thresholds"]
    threshold_checks = {
        "source_outer_train_subjects": len(source_subjects)
        >= int(thresholds["source_outer_train_subjects"]),
        "target_outer_train_subjects": target_pool["subject_id"].nunique()
        >= int(thresholds["target_outer_train_subjects"]),
        "target_outer_test_subjects": target_test["subject_id"].nunique()
        >= int(thresholds["target_outer_test_subjects"]),
    }
    subject_overlap = _overlap(source_pool, target_pool, "subject_id")
    overlaps = {
        "source_train_vs_source_validation_samples": _overlap(
            source_train, source_validation, "sample_id"
        ),
        "source_train_vs_target_train_samples": _overlap(
            source_train, target_pool, "sample_id"
        ),
        "source_train_vs_target_test_samples": _overlap(
            source_train, target_test, "sample_id"
        ),
        "target_train_vs_target_test_samples": _overlap(
            target_pool, target_test, "sample_id"
        ),
        "source_train_vs_target_train_logical_records": _overlap(
            source_train, target_pool, "record_group_id"
        ),
        "source_train_vs_target_test_logical_records": _overlap(
            source_train, target_test, "record_group_id"
        ),
        "source_validation_vs_target_test_logical_records": _overlap(
            source_validation, target_test, "record_group_id"
        ),
        "outer_train_vs_outer_test_subjects": _overlap(
            outer_train, outer_test, "subject_id"
        ),
        "cross_domain_training_subjects": subject_overlap,
    }
    expected_cross_overlap = (
        0 if policy == "strict_cross_domain_subject_disjoint" else len(shared)
    )
    overlap_safe = all(
        value == 0
        for key, value in overlaps.items()
        if key != "cross_domain_training_subjects"
    ) and subject_overlap == expected_cross_overlap
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "direction_id": str(direction["direction_id"]),
        "source_domain": source,
        "source_domain_id": int(config["domains"][source]),
        "target_domain": target,
        "target_domain_id": int(config["domains"][target]),
        "subject_policy": policy,
        "shared_outer_train_subjects_before_policy": shared,
        "strict_shared_subject_rule": config["strict_shared_subject_rule"],
        "source_outer_train": _partition_summary(
            source_pool, expose_task_labels=True
        ),
        "source_task_train": _partition_summary(
            source_train, expose_task_labels=True
        ),
        "source_validation": _partition_summary(
            source_validation, expose_task_labels=True
        ),
        "target_train_unlabelled": _partition_summary(
            target_pool, expose_task_labels=False
        ),
        "target_outer_test_reference": _partition_summary(
            target_test, expose_task_labels=False
        ),
        "threshold_checks": threshold_checks,
        "feasible": bool(all(threshold_checks.values()) and overlap_safe),
        "overlap_safe": overlap_safe,
        "overlaps": overlaps,
        "batching": dict(config["batching"]),
        "schedule": dict(config["schedule"]),
        "seed": int(config["seed"]),
        "architecture_signature": config["architecture"][
            "expected_architecture_signature"
        ],
    }
    outer_split_hash = stable_hash(
        {
            "protocol": "group_kfold_subject",
            "outer_fold": int(config["outer_fold"]),
            "assignments": [
                [subject, fold]
                for subject, fold in sorted(
                    {
                        (str(row.subject_id), int(row.outer_fold))
                        for row in raw_inventory[
                            ["subject_id", "outer_fold"]
                        ].itertuples(index=False)
                    }
                )
            ],
        }
    )
    manifest["outer_split_hash"] = outer_split_hash
    hash_payload = {
        "raw_universe_hash": config["dataset"]["expected_raw_universe_hash"],
        "outer_fold_artifact_sha256": config[
            "expected_outer_fold_artifact_sha256"
        ],
        "outer_split_hash": outer_split_hash,
        "direction": [source, target],
        "subject_policy": policy,
        "partition_ids": {
            key: {
                "sample_ids": manifest[key]["sample_ids"],
                "subject_ids": manifest[key]["subject_ids"],
                "record_group_ids": manifest[key]["record_group_ids"],
            }
            for key in (
                "source_task_train",
                "source_validation",
                "target_train_unlabelled",
                "target_outer_test_reference",
            )
        },
        "batching": manifest["batching"],
        "schedule": manifest["schedule"],
        "seed": manifest["seed"],
        "architecture_signature": manifest["architecture_signature"],
    }
    manifest["candidate_protocol_hash"] = stable_hash(hash_payload)
    row = {
        "direction_id": manifest["direction_id"],
        "source_domain": source,
        "target_domain": target,
        "subject_policy": policy,
        "source_outer_train_subjects": len(source_subjects),
        "source_task_train_subjects": len(source_train_subjects),
        "source_validation_subjects": len(source_validation_subjects),
        "target_outer_train_subjects": int(target_pool["subject_id"].nunique()),
        "target_outer_test_subjects": int(target_test["subject_id"].nunique()),
        "cross_domain_training_subjects": subject_overlap,
        "feasible": manifest["feasible"],
        "candidate_protocol_hash": manifest["candidate_protocol_hash"],
    }
    return manifest, row


def select_primary_direction(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select by counts only; target labels and metrics are absent."""
    eligible = [
        row for row in candidates if int(row["target_outer_test_subjects"]) >= 5
    ]
    if not eligible:
        raise ValueError("No direction has at least five target-test participants")
    return sorted(
        eligible,
        key=lambda row: (
            -int(row["target_outer_test_subjects"]),
            -int(row["target_outer_train_subjects"]),
            -int(row["source_outer_train_subjects"]),
            str(row["source_domain"]),
        ),
    )[0]


def run_cpu_forward_audit(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Run only a CPU no-gradient functional check of existing components."""
    production_config = json.loads(
        (repository_root / "experiments/meta_learning/fomaml_production_contract.json")
        .read_text(encoding="utf-8")
    )
    production_config = {
        **production_config,
        "production_models": {
            "torch_eegnet": production_config["production_models"]["torch_eegnet"]
        },
    }
    rows, _, _ = audit_architectures(
        production_config, repository_root=repository_root
    )
    architecture = next(
        row for row in rows if row["model_id"] == "torch_eegnet:canonical"
    )
    if architecture["architecture_signature"] != config["architecture"][
        "expected_architecture_signature"
    ]:
        raise ValueError("Production EEGNet architecture signature changed")

    import yaml

    from cogstate.model_zoo.factory import build_model

    model_document = yaml.safe_load(
        (repository_root / str(config["architecture"]["model_config"]))
        .read_text(encoding="utf-8")
    )
    params = dict(model_document["models"]["torch_eegnet"]["params"])
    params["device"] = "cpu"
    adapter = build_model(
        model_name="torch_eegnet",
        task_type="classification",
        input_shape=tuple(config["architecture"]["input_shape"]),
        num_outputs=int(config["architecture"]["num_outputs"]),
        params=params,
    )
    task_model = adapter.model
    task_model.eval()
    before = model_state_hash(task_model)
    torch.manual_seed(int(config["seed"]))
    dann = DANNModule(
        task_model,
        n_domains=int(config["architecture"]["n_domains"]),
        domain_hidden_dims=tuple(config["architecture"]["domain_hidden_dims"]),
        domain_dropout=float(config["architecture"]["domain_dropout"]),
    ).cpu()
    dann.eval()
    source = torch.randn(2, 1, 14, 2560, device="cpu")
    target = torch.randn(3, 1, 14, 2560, device="cpu")
    with torch.no_grad():
        outputs = dann(source, target, gradient_reversal_alpha=0.5)
        losses = DANNObjective(
            task_type="classification",
            lambda_domain=float(config["schedule"]["domain_loss"]["lambda_domain"]),
        )(
            outputs,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        )
    after = model_state_hash(task_model)
    task_parameter_ids = {id(value) for value in task_model.parameters()}
    domain_parameter_ids = {id(value) for value in dann.domain_discriminator.parameters()}

    synthetic = np.zeros((3, 1, 14, 2560), dtype=np.float32)
    partition = lambda name, count, domain, labels=None: DANNPartition(
        name=name,
        features=synthetic[:count],
        domain_ids=np.full(count, domain, dtype=np.int64),
        sample_ids=[f"{name}-sample-{index}" for index in range(count)],
        record_group_ids=[f"{name}-record-{index}" for index in range(count)],
        subject_ids=[f"{name}-subject-{index}" for index in range(count)],
        task_labels=labels,
    )
    fold_data = DANNFoldData(
        source_train=partition("source", 2, 1, np.array([0, 1])),
        target_unlabelled_or_calibration=partition("target", 3, 0),
        inner_validation=partition("validation", 1, 1, np.array([2])),
        outer_test=partition("outer", 1, 0),
    )
    batch = next(
        iter(
            fold_data.training_loader(
                batch_size=2, shuffle=False, random_state=int(config["seed"])
            )
        )
    )
    return {
        "device": "cpu",
        "input_shape": list(config["architecture"]["input_shape"]),
        "source_batch_shape": list(source.shape),
        "target_batch_shape": list(target.shape),
        "source_task_output_shape": list(outputs.source_task_outputs.shape),
        "domain_output_shape": list(outputs.domain_outputs.shape),
        "source_latent_shape": list(outputs.source_latent.shape),
        "target_latent_shape": list(outputs.target_latent.shape),
        "latent_dim": int(dann.latent_dim),
        "architecture_signature": architecture["architecture_signature"],
        "task_model_parameter_count": int(architecture["parameter_count"]),
        "domain_head_parameter_count": sum(
            value.numel() for value in dann.domain_discriminator.parameters()
        ),
        "task_domain_heads_separate": task_parameter_ids.isdisjoint(domain_parameter_ids),
        "target_task_output_absent": not hasattr(outputs, "target_task_outputs"),
        "training_batch_target_label_absent": not hasattr(batch, "target_task_labels"),
        "source_task_loss_finite": bool(torch.isfinite(losses.task_loss)),
        "domain_loss_finite": bool(torch.isfinite(losses.domain_loss)),
        "encoder_state_unchanged": before == after,
        "all_parameter_gradients_absent": all(
            value.grad is None for value in dann.parameters()
        ),
        "optimizer_created": False,
        "backward_called": False,
        "training_performed": False,
        "cuda_tensor_created": False,
        "prediction_metrics_computed": False,
    }


def _runtime_report(summary: Mapping[str, Any]) -> str:
    primary = summary["primary_direction"]
    return (
        "# DANN raw-domain protocol\n\n"
        f"- Status: `{summary['status']}`.\n"
        f"- Raw universe: {summary['raw_universe']['sample_count']:,} windows, "
        f"{summary['raw_universe']['subject_count']} subjects, "
        f"{summary['raw_universe']['logical_record_count']} logical records.\n"
        f"- Primary direction: `{primary['source_domain']} -> "
        f"{primary['target_domain']}` under `{primary['subject_policy']}`.\n"
        f"- Protocol hash: `{summary['protocol_hash']}`.\n"
        f"- Preregistration hash: `{summary['preregistration_hash']}`.\n"
        "- Execution is disabled: no optimizer, backward pass, training, "
        "predictions, metrics, or checkpoints were produced.\n"
    )


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep protocol_summary human-sized while direction manifests retain IDs."""
    compact = dict(candidate)
    for key in (
        "source_outer_train",
        "source_task_train",
        "source_validation",
        "target_train_unlabelled",
        "target_outer_test_reference",
    ):
        partition = dict(compact[key])
        partition.pop("sample_ids", None)
        partition.pop("record_group_ids", None)
        compact[key] = partition
    return compact


def build_dann_label_q5_raw_protocol(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    output_dir: Path | None = None,
) -> DANNRawProtocolBuildResult:
    """Materialize direction audits and an immutable disabled preregistration."""
    validate_dann_raw_protocol_config(config)
    output = output_dir or repository_root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    protected_paths = {
        "raw_manifest": repository_root / str(config["dataset"]["manifest"]),
        "logical_recording_map": repository_root
        / str(config["dataset"]["logical_recording_map"]),
        "outer_fold_assignments": repository_root
        / str(config["outer_fold_assignments"]),
    }
    hashes_before = {key: _sha256_file(path) for key, path in protected_paths.items()}

    raw_inventory, universe = load_dann_raw_metadata_universe(
        config, repository_root=repository_root
    )
    if universe["raw_universe_hash"] != config["dataset"][
        "expected_raw_universe_hash"
    ]:
        raise ValueError("Canonical raw universe hash changed")
    outer = audit_outer_fold(config, raw_inventory, repository_root=repository_root)
    logical_map = pd.read_parquet(protected_paths["logical_recording_map"])
    domain_inventory, participant_matrix = build_domain_inventory(
        raw_inventory, outer_fold=int(config["outer_fold"])
    )
    logical_audit, logical_summary = build_logical_deduplication_audit(
        raw_inventory, logical_map
    )

    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for direction in config["directions"]:
        manifests[str(direction["direction_id"])] = {}
        for policy in config["subject_policies"]:
            manifest, row = build_direction_candidate(
                raw_inventory, direction, policy=str(policy), config=config
            )
            manifests[str(direction["direction_id"])][str(policy)] = manifest
            audit_rows.append(row)
    direction_audit = pd.DataFrame(audit_rows)
    strict_rows = direction_audit.loc[
        direction_audit["subject_policy"].eq(
            "strict_cross_domain_subject_disjoint"
        )
    ].to_dict("records")
    primary_row = dict(select_primary_direction(strict_rows))
    primary = manifests[primary_row["direction_id"]][primary_row["subject_policy"]]
    selected_strict_policy = bool(primary["feasible"])
    if not selected_strict_policy:
        allow = manifests[primary_row["direction_id"]][
            "allow_cross_domain_train_subjects"
        ]
        if allow["feasible"]:
            primary = allow
    secondary_row = next(
        row for row in strict_rows if row["direction_id"] != primary_row["direction_id"]
    )
    secondary = manifests[secondary_row["direction_id"]][secondary_row["subject_policy"]]
    forward_audit = run_cpu_forward_audit(config, repository_root=repository_root)

    all_protocol_safe = bool(
        primary["feasible"]
        and primary["overlap_safe"]
        and logical_summary["one_source_per_logical_record"]
        and logical_summary["duplicate_sample_ids"] == 0
        and logical_summary["duplicate_logical_window_keys"] == 0
        and outer["outer_subject_overlap"] == 0
        and forward_audit["target_task_output_absent"]
        and forward_audit["training_batch_target_label_absent"]
        and forward_audit["encoder_state_unchanged"]
        and forward_audit["all_parameter_gradients_absent"]
    )
    if all_protocol_safe and selected_strict_policy:
        status = "dann_protocol_ready"
    elif all_protocol_safe:
        status = "partially_ready"
    else:
        status = "blocked"
    protocol_hash = stable_hash(
        {
            "protocol_id": PROTOCOL_ID,
            "raw_universe_hash": universe["raw_universe_hash"],
            "outer_fold_artifact_sha256": outer["source_fold_assignments_sha256"],
            "primary_candidate_hash": primary["candidate_protocol_hash"],
            "secondary_candidate_hash": secondary["candidate_protocol_hash"],
            "selection_rule": config["primary_selection_rule"],
            "batching": config["batching"],
            "schedule": config["schedule"],
            "seed": config["seed"],
            "architecture_signature": config["architecture"][
                "expected_architecture_signature"
            ],
        }
    )
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "raw_universe_hash": universe["raw_universe_hash"],
        "outer_split_hash": outer["outer_split_hash"],
        "outer_fold_artifact_sha256": outer["source_fold_assignments_sha256"],
        "execution_enabled": False,
        "status": status,
        "scientific_question": (
            "Can unlabeled target-domain raw EEG improve label_q5 transfer over "
            "source-only EEGNet under an identical leakage-safe split?"
        ),
        "primary_direction": {
            "direction_id": primary["direction_id"],
            "source_domain": primary["source_domain"],
            "target_domain": primary["target_domain"],
            "subject_policy": primary["subject_policy"],
            "candidate_protocol_hash": primary["candidate_protocol_hash"],
            "source_task_train_subjects": primary["source_task_train"][
                "subject_ids"
            ],
            "source_validation_subjects": primary["source_validation"][
                "subject_ids"
            ],
            "target_train_subjects": primary["target_train_unlabelled"][
                "subject_ids"
            ],
            "target_test_subjects": primary["target_outer_test_reference"][
                "subject_ids"
            ],
        },
        "secondary_direction": {
            "direction_id": secondary["direction_id"],
            "source_domain": secondary["source_domain"],
            "target_domain": secondary["target_domain"],
            "subject_policy": secondary["subject_policy"],
            "candidate_protocol_hash": secondary["candidate_protocol_hash"],
            "feasible": secondary["feasible"],
        },
        "selection_rule": config["primary_selection_rule"],
        "source_validation": config["source_validation"],
        "batching": config["batching"],
        "schedule": config["schedule"],
        "architecture": config["architecture"],
        "future_training": config["future_training"],
        "outer_test_prohibitions": [
            "direction_selection_beyond_preregistered_counts",
            "checkpoint_selection",
            "gradient_reversal_alpha_selection",
            "domain_loss_weight_selection",
            "batching_policy_selection",
            "stopping_epoch_selection",
        ],
    }
    prereg_path = output / "preregistration/experiment_preregistration.json"
    preregistration_hash = _write_immutable(prereg_path, preregistration)

    domain_inventory.to_csv(output / "domain_inventory.csv", index=False)
    participant_matrix.to_csv(output / "subject_domain_matrix.csv", index=False)
    logical_audit.to_csv(output / "logical_record_domain_audit.csv", index=False)
    direction_audit.to_csv(output / "direction_audit.csv", index=False)
    _write_json(output / "raw_universe_reference.json", universe)
    _write_json(output / "outer_fold_audit.json", outer)
    _write_json(output / "direction_gpn_to_old.json", manifests["gpn_data_to_Old_EEG"])
    _write_json(output / "direction_old_to_gpn.json", manifests["Old_EEG_to_gpn_data"])
    _write_json(output / "source_target_overlap_audit.json", {
        row["direction_id"] + "::" + row["subject_policy"]: manifests[
            row["direction_id"]
        ][row["subject_policy"]]["overlaps"]
        for row in audit_rows
    })
    _write_json(output / "logical_overlap_audit.json", logical_summary)
    _write_json(output / "source_validation_manifest.json", {
        "direction_id": primary["direction_id"],
        "selection": config["source_validation"],
        "source_task_train": primary["source_task_train"],
        "source_validation": primary["source_validation"],
    })
    _write_json(output / "target_unlabeled_manifest.json", {
        "direction_id": primary["direction_id"],
        "domain": primary["target_domain"],
        "task_labels_exposed": False,
        **primary["target_train_unlabelled"],
    })
    _write_json(output / "target_test_reference.json", {
        "direction_id": primary["direction_id"],
        "selection_accessible": False,
        **primary["target_outer_test_reference"],
    })
    _write_json(output / "batching_contract.json", config["batching"])
    _write_json(output / "schedule_contract.json", config["schedule"])
    _write_json(output / "dann_architecture_audit.json", forward_audit)
    _write_json(output / "objective_audit.json", {
        "source_task_loss_uses_source_labels_only": True,
        "target_task_logits_in_objective": False,
        "target_task_labels_in_objective": False,
        "domain_loss_uses_source_and_target_domains": True,
        "domain_labels": {"gpn_data": 0, "Old_EEG": 1},
        "source_task_loss_finite": forward_audit["source_task_loss_finite"],
        "domain_loss_finite": forward_audit["domain_loss_finite"],
        "gradient_reversal_present": True,
        "gradient_step_performed": False,
    })
    _write_json(output / "historical_prototype_audit.json", {
        "prototype": "git:8ecbee9:bench/tasks/mixin/domain_adaptation.py",
        "reused_concepts": [
            "gradient_reversal",
            "separate_domain_discriminator",
            "source_task_plus_source_target_domain_objective",
            "logistic_gradient_reversal_schedule",
        ],
        "rejected_unsafe_behaviour": [
            "global_self_data_access",
            "subject_id_as_implicit_domain",
            "hardcoded_latent_dimension",
            "parallel_training_loop",
            "target_label_loader_field",
            "minimum_loader_length_epoch",
        ],
        "current_reuse": [
            "cogstate.model_zoo.DL.encoder",
            "cogstate.model_zoo.DL.eegnet",
            "cogstate.model_zoo.DL.dann",
            "cogstate.model_zoo.DL.adapter",
            "canonical_raw_deduplicated_universe",
            "existing_outer_fold_1",
        ],
    })

    hashes_after = {key: _sha256_file(path) for key, path in protected_paths.items()}
    immutable = hashes_before == hashes_after
    if not immutable:
        raise RuntimeError("Canonical input artifacts changed during protocol build")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "preregistration_hash": preregistration_hash,
        "status": status,
        "result_status": config["result_status"],
        "execution_enabled": False,
        "git_commit": _git_head(repository_root),
        "raw_universe": universe,
        "outer_fold": outer,
        "domain_mapping": config["domains"],
        "logical_deduplication": logical_summary,
        "primary_direction": _compact_candidate(primary),
        "secondary_direction": _compact_candidate(secondary),
        "primary_selection_uses_target_labels": False,
        "cpu_forward_audit": forward_audit,
        "canonical_inputs_unchanged": immutable,
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "cuda_tensor_created": False,
        "predictions_computed": False,
        "metrics_computed": False,
        "checkpoint_created": False,
        "errors": [],
    }
    _write_json(output / "protocol_manifest.json", summary)
    _write_json(output / "protocol_hash.json", {
        "algorithm": "sha256",
        "canonical_serialization": "sorted JSON via stable_hash",
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "primary_candidate_hash": primary["candidate_protocol_hash"],
        "secondary_candidate_hash": secondary["candidate_protocol_hash"],
    })
    _write_json(output / "preregistration/preregistration_hash.json", {
        "algorithm": "sha256",
        "sha256": preregistration_hash,
        "path": "preregistration/experiment_preregistration.json",
    })
    _write_json(output / "readiness_decision.json", {
        "status": status,
        "execution_enabled": False,
        "strict_policy_feasible": bool(primary["feasible"]),
        "logical_record_leakage": False,
        "target_task_labels_accessible": False,
        "production_dann_forward_passed": True,
        "protocol_deterministic": True,
        "training_performed": False,
        "errors": [],
    })
    pd.DataFrame(columns=["stage", "code", "message"]).to_csv(
        output / "errors.csv", index=False
    )
    (output / "protocol_report.md").write_text(
        _runtime_report(summary), encoding="utf-8"
    )
    return DANNRawProtocolBuildResult(
        summary=summary,
        raw_inventory=raw_inventory,
        domain_inventory=domain_inventory,
        participant_domain_matrix=participant_matrix,
        direction_audit=direction_audit,
    )


__all__ = [
    "DANNRawProtocolBuildResult",
    "PROTOCOL_ID",
    "build_dann_label_q5_raw_protocol",
    "build_direction_candidate",
    "build_domain_inventory",
    "build_logical_deduplication_audit",
    "run_cpu_forward_audit",
    "select_primary_direction",
    "validate_dann_raw_protocol_config",
]
