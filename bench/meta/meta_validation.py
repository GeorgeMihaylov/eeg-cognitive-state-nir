"""Deterministic nested subject protocol manifests for future meta-learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .episodes import MetaEpisodeBuilder, MetaEpisodeSpec, stable_hash


@dataclass(frozen=True)
class MetaValidationProtocolResult:
    protocol: dict[str, Any]
    episode_index: pd.DataFrame
    errors: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subject_order(subjects: list[str], seed: int) -> list[str]:
    return sorted(
        subjects,
        key=lambda subject: hashlib.sha256(
            f"{seed}|{subject}".encode("utf-8")
        ).hexdigest(),
    )


def build_meta_validation_protocol(
    config: Mapping[str, Any], *, repository_root: Path
) -> MetaValidationProtocolResult:
    """Reuse one existing outer fold and split only its outer-train subjects."""
    assignment_path = repository_root / str(config["fold_assignments"])
    before_hash = _sha256_file(assignment_path)
    frame = pd.read_parquet(assignment_path).rename(columns={"y_true": "target"})
    required = {"sample_id", "subject_id", "record_id", "fold", "target", "sample_index"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Fold assignment artifact is missing {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Fold assignment artifact contains duplicate sample_id")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["session_id"] = None
    frame["time_order"] = frame["sample_index"]
    outer_fold = int(config["outer_fold"])
    seed = int(config["seed"])
    support_budget = int(config["support_budget"])
    query_budget = int(config["query_budget"])
    outer_test = frame.loc[frame["fold"].eq(outer_fold)].copy()
    outer_train = frame.loc[~frame["fold"].eq(outer_fold)].copy()
    outer_test_subjects = sorted(outer_test["subject_id"].unique())
    outer_train_subjects = sorted(outer_train["subject_id"].unique())
    if set(outer_train_subjects) & set(outer_test_subjects):
        raise ValueError("Outer train/test subjects overlap")

    grouped = outer_train.groupby("subject_id", sort=True)
    eligible = [
        subject for subject, group in grouped
        if group["record_id"].nunique() >= 2
        and len(group) >= support_budget + query_budget
    ]
    requested = max(
        1, int(round(len(outer_train_subjects) * float(config["meta_validation_fraction"])))
    )
    ordered = _subject_order(eligible, seed)
    meta_validation_subjects = sorted(ordered[:requested])
    meta_train_subjects = sorted(
        set(outer_train_subjects) - set(meta_validation_subjects)
    )
    if set(meta_train_subjects) & set(meta_validation_subjects):
        raise ValueError("Meta-train/meta-validation subjects overlap")
    if set(meta_validation_subjects) & set(outer_test_subjects):
        raise ValueError("Outer-test subjects entered meta-validation")

    spec = MetaEpisodeSpec(
        episode_type="subject_personalization",
        task_type="ordinal_classification",
        target_name="label_q5",
        support_unit="sample",
        query_unit="sample",
        support_size=support_budget,
        query_size=query_budget,
        class_balance_policy="none",
        chronological=True,
        group_fields=("subject_id", "record_id", "sample_id"),
        seed=seed,
        minimum_records=2,
        minimum_classes=1,
        insufficient_data_policy="skip_episode",
    )
    builder = MetaEpisodeBuilder()
    scopes = (
        ("meta_train", outer_train, meta_train_subjects, meta_validation_subjects + outer_test_subjects),
        ("meta_validation", outer_train, meta_validation_subjects, meta_train_subjects + outer_test_subjects),
        ("outer_test", outer_test, outer_test_subjects, outer_train_subjects),
    )
    episodes = []
    errors = []
    for scope, partition, subjects, forbidden_subjects in scopes:
        scoped = partition.loc[partition["subject_id"].isin(subjects)]
        result = builder.build(
            dataset_index=frame,
            allowed_sample_ids=scoped["sample_id"],
            forbidden_sample_ids=frame.loc[~frame["sample_id"].isin(scoped["sample_id"]), "sample_id"],
            episode_spec=spec,
            dataset_id="emotiv_cognitive",
            task_id="cognitive_load_5class",
            fold_id=outer_fold,
            allowed_subject_ids=subjects,
            forbidden_subject_ids=forbidden_subjects,
            entity_ids=subjects,
            scope=scope,
        )
        episodes.extend(result.episodes)
        errors.extend(result.errors)
    episode_rows = []
    for episode in episodes:
        support_distribution = pd.Series(episode.support_targets).value_counts().sort_index()
        query_distribution = pd.Series(episode.query_targets).value_counts().sort_index()
        episode_rows.append({
            **episode.to_dict(),
            "support_class_distribution": json.dumps(
                {str(key): int(value) for key, value in support_distribution.items()},
                sort_keys=True,
            ),
            "query_class_distribution": json.dumps(
                {str(key): int(value) for key, value in query_distribution.items()},
                sort_keys=True,
            ),
        })
    episode_index = pd.DataFrame(episode_rows)
    protocol = {
        "schema_version": "1.0",
        "task": "label_q5",
        "outer_protocol": "existing_group_kfold_subject",
        "outer_fold": outer_fold,
        "outer_train_subjects": outer_train_subjects,
        "outer_test_subjects": outer_test_subjects,
        "meta_train_subjects": meta_train_subjects,
        "meta_validation_subjects": meta_validation_subjects,
        "support_budget": support_budget,
        "query_budget": query_budget,
        "episode_type": spec.episode_type,
        "split_preference": ["session", "record", "chronology"],
        "seed": seed,
        "source_split_hash": before_hash,
        "spec_hash": spec.spec_hash,
        "episode_hashes": sorted(episode.episode_id for episode in episodes),
        "episode_counts": {
            str(scope): int(count)
            for scope, count in pd.Series(
                [episode.scope for episode in episodes]
            ).value_counts().items()
        },
        "outer_subject_overlap": 0,
        "meta_subject_overlap": 0,
        "outer_test_in_meta_validation": False,
        "outer_test_selection_forbidden": True,
        "execution_performed": False,
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    if _sha256_file(assignment_path) != before_hash:
        raise RuntimeError("Source outer-fold artifact changed")
    return MetaValidationProtocolResult(
        protocol=protocol,
        episode_index=episode_index,
        errors=pd.DataFrame(
            [error.to_dict() for error in errors],
            columns=["dataset_id", "task_id", "fold_id", "entity_id", "error_type", "message"],
        ),
    )
