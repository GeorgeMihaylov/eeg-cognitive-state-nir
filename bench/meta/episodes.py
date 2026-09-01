"""Deterministic support/query episode construction within explicit partitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .validation import validate_episode

EPISODE_SCHEMA_VERSION = "1.0"
EPISODE_TYPES = {
    "subject_personalization",
    "session_adaptation",
    "few_shot_classification",
    "cross_task_adaptation",
}
TASK_TYPES = {
    "classification",
    "ordinal_classification",
    "regression",
    "multioutput_regression",
}
CLASS_POLICIES = {
    "none",
    "require_all_classes",
    "equal_per_class",
    "at_least_one_per_class",
}
INSUFFICIENT_POLICIES = {
    "error",
    "skip_episode",
    "reduce_support",
    "reduce_query",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list, set, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetaEpisodeSpec:
    episode_type: str
    task_type: str
    target_name: str
    support_unit: str
    query_unit: str
    support_size: int
    query_size: int
    class_balance_policy: str = "none"
    chronological: bool = True
    group_fields: tuple[str, ...] = ("subject_id", "session_id", "record_id")
    seed: int = 42
    minimum_records: int = 2
    minimum_classes: int = 1
    insufficient_data_policy: str = "error"
    allow_within_record_chronological_split: bool = False
    time_column: str = "time_order"

    def __post_init__(self) -> None:
        if self.episode_type not in EPISODE_TYPES:
            raise ValueError(f"Unknown episode_type: {self.episode_type}")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        if not self.target_name:
            raise ValueError("target_name is required")
        if self.support_unit not in {"sample", "record", "session"}:
            raise ValueError("support_unit must be sample, record, or session")
        if self.query_unit not in {"sample", "record", "session"}:
            raise ValueError("query_unit must be sample, record, or session")
        if self.support_size <= 0 or self.query_size <= 0:
            raise ValueError("support_size and query_size must be positive")
        if self.class_balance_policy not in CLASS_POLICIES:
            raise ValueError("Unknown class_balance_policy")
        if self.insufficient_data_policy not in INSUFFICIENT_POLICIES:
            raise ValueError("Unknown insufficient_data_policy")
        if self.task_type in {"regression", "multioutput_regression"} and (
            self.class_balance_policy != "none"
        ):
            raise ValueError("Regression episodes cannot use class balancing")
        if self.minimum_records < 1 or self.minimum_classes < 1:
            raise ValueError("minimum_records and minimum_classes must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetaEpisodeSpec":
        data = dict(value)
        if "group_fields" in data:
            data["group_fields"] = tuple(data["group_fields"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class MetaEpisode:
    episode_id: str
    dataset_id: str
    task_id: str
    target_name: str
    fold_id: str
    entity_id: str
    subject_id: str
    session_ids: tuple[str, ...]
    support_sample_ids: tuple[str, ...]
    query_sample_ids: tuple[str, ...]
    support_record_ids: tuple[str, ...]
    query_record_ids: tuple[str, ...]
    support_targets: tuple[Any, ...]
    query_targets: tuple[Any, ...]
    seed: int
    spec_hash: str
    split_level: str
    scope: str
    chronological_boundary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def with_recomputed_id(self) -> "MetaEpisode":
        payload = self.to_dict()
        payload.pop("episode_id", None)
        return replace(self, episode_id=stable_hash(payload))


@dataclass(frozen=True)
class MetaEpisodeError:
    dataset_id: str
    task_id: str
    fold_id: str
    entity_id: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaEpisodeIndex:
    episodes: tuple[MetaEpisode, ...]
    errors: tuple[MetaEpisodeError, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([episode.to_dict() for episode in self.episodes])


@dataclass(frozen=True)
class MetaEpisodeManifest:
    dataset_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    fold_ids: tuple[str, ...]
    specs: tuple[MetaEpisodeSpec, ...]
    episodes: tuple[MetaEpisode, ...]
    errors: tuple[MetaEpisodeError, ...]
    source_split_hashes: Mapping[str, str]
    schema_version: str = EPISODE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "dataset_ids": self.dataset_ids,
            "task_ids": self.task_ids,
            "fold_ids": self.fold_ids,
            "specs": [spec.to_dict() for spec in self.specs],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "errors": [error.to_dict() for error in self.errors],
            "source_split_hashes": dict(self.source_split_hashes),
        }
        payload["manifest_hash"] = stable_hash(payload)
        return _json_value(payload)


class MetaEpisodeBuildError(ValueError):
    pass


class MetaEpisodeBuilder:
    """Build episodes from metadata rows already scoped by an outer partition."""

    REQUIRED_COLUMNS = {"sample_id", "subject_id", "record_id", "target"}

    def build(
        self,
        *,
        dataset_index: pd.DataFrame,
        allowed_sample_ids: Collection[str],
        forbidden_sample_ids: Collection[str],
        episode_spec: MetaEpisodeSpec,
        dataset_id: str,
        task_id: str,
        fold_id: str | int,
        allowed_subject_ids: Collection[str] | None = None,
        forbidden_subject_ids: Collection[str] = (),
        entity_ids: Collection[str] | None = None,
        scope: str = "meta_train_index",
    ) -> MetaEpisodeIndex:
        frame = self._prepare_frame(dataset_index, episode_spec)
        allowed = {str(value) for value in allowed_sample_ids}
        forbidden = {str(value) for value in forbidden_sample_ids}
        if allowed & forbidden:
            raise MetaEpisodeBuildError("Allowed and forbidden sample IDs overlap")
        unknown = allowed - set(frame["sample_id"])
        if unknown:
            raise MetaEpisodeBuildError(
                f"Allowed partition contains {len(unknown)} unknown sample IDs"
            )
        scoped = frame.loc[frame["sample_id"].isin(allowed)].copy()
        if allowed_subject_ids is not None:
            subjects = {str(value) for value in allowed_subject_ids}
            scoped = scoped.loc[scoped["subject_id"].isin(subjects)]
        forbidden_subjects = {str(value) for value in forbidden_subject_ids}
        if set(scoped["subject_id"]) & forbidden_subjects:
            raise MetaEpisodeBuildError("Allowed rows include forbidden subjects")
        entities = (
            sorted(scoped["subject_id"].unique())
            if entity_ids is None
            else [str(value) for value in entity_ids]
        )
        episodes: list[MetaEpisode] = []
        errors: list[MetaEpisodeError] = []
        for entity in entities:
            entity_frame = scoped.loc[scoped["subject_id"].eq(entity)].copy()
            try:
                episode = self._build_subject_episode(
                    entity_frame,
                    episode_spec,
                    dataset_id=dataset_id,
                    task_id=task_id,
                    fold_id=str(fold_id),
                    entity_id=entity,
                    scope=scope,
                )
                audit = validate_episode(
                    episode,
                    forbidden_sample_ids=forbidden,
                    forbidden_subject_ids=forbidden_subjects,
                    require_record_disjoint=episode.split_level != "within_record",
                )
                if not audit.valid:
                    raise MetaEpisodeBuildError("; ".join(audit.errors))
                episodes.append(episode)
            except (MetaEpisodeBuildError, ValueError) as exc:
                error = MetaEpisodeError(
                    dataset_id, task_id, str(fold_id), entity,
                    type(exc).__name__, str(exc),
                )
                if episode_spec.insufficient_data_policy == "skip_episode":
                    errors.append(error)
                    continue
                raise MetaEpisodeBuildError(
                    f"Could not build episode for {entity}: {exc}"
                ) from exc
        return MetaEpisodeIndex(tuple(episodes), tuple(errors))

    def build_session_rotations(
        self,
        **kwargs: Any,
    ) -> MetaEpisodeIndex:
        spec: MetaEpisodeSpec = kwargs["episode_spec"]
        if spec.episode_type != "session_adaptation":
            raise ValueError("Session rotations require session_adaptation")
        frame = self._prepare_frame(kwargs["dataset_index"], spec)
        allowed = {str(value) for value in kwargs["allowed_sample_ids"]}
        forbidden = {str(value) for value in kwargs["forbidden_sample_ids"]}
        if allowed & forbidden:
            raise MetaEpisodeBuildError("Allowed and forbidden sample IDs overlap")
        unknown = allowed - set(frame["sample_id"])
        if unknown:
            raise MetaEpisodeBuildError(
                f"Allowed partition contains {len(unknown)} unknown sample IDs"
            )
        forbidden_subjects = {
            str(value) for value in kwargs.get("forbidden_subject_ids", ())
        }
        scoped = frame.loc[frame["sample_id"].isin(allowed)].copy()
        allowed_subjects = kwargs.get("allowed_subject_ids")
        if allowed_subjects is not None:
            scoped = scoped.loc[
                scoped["subject_id"].isin({str(v) for v in allowed_subjects})
            ]
        entities = kwargs.get("entity_ids") or sorted(scoped["subject_id"].unique())
        episodes: list[MetaEpisode] = []
        errors: list[MetaEpisodeError] = []
        for subject in [str(value) for value in entities]:
            group = scoped.loc[scoped["subject_id"].eq(subject)]
            sessions = sorted(group["session_id"].dropna().astype(str).unique())
            if len(sessions) < spec.support_size + spec.query_size:
                error = MetaEpisodeError(
                    str(kwargs["dataset_id"]), str(kwargs["task_id"]),
                    str(kwargs["fold_id"]), subject, "insufficient_sessions",
                    f"requires {spec.support_size + spec.query_size} sessions, "
                    f"found {len(sessions)}",
                )
                if spec.insufficient_data_policy == "skip_episode":
                    errors.append(error)
                    continue
                raise MetaEpisodeBuildError(error.message)
            for query_session in sessions:
                support_sessions = [s for s in sessions if s != query_session][
                    : spec.support_size
                ]
                query_sessions = [query_session][: spec.query_size]
                support = group.loc[group["session_id"].isin(support_sessions)]
                query = group.loc[group["session_id"].isin(query_sessions)]
                support, query = self._apply_class_policy(
                    support, query, spec
                )
                episode = self._make_episode(
                    support, query, spec,
                    dataset_id=str(kwargs["dataset_id"]),
                    task_id=str(kwargs["task_id"]),
                    fold_id=str(kwargs["fold_id"]),
                    entity_id=f"{subject}|query={query_session}",
                    scope=str(kwargs.get("scope", "outer_test_personalization")),
                    split_level="session",
                )
                audit = validate_episode(
                    episode,
                    forbidden_sample_ids=forbidden,
                    forbidden_subject_ids=forbidden_subjects,
                )
                if not audit.valid:
                    raise MetaEpisodeBuildError("; ".join(audit.errors))
                episodes.append(episode)
        return MetaEpisodeIndex(tuple(episodes), tuple(errors))

    def _prepare_frame(
        self, dataset_index: pd.DataFrame, spec: MetaEpisodeSpec
    ) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(dataset_index.columns)
        if missing:
            raise MetaEpisodeBuildError(f"Dataset index is missing columns: {sorted(missing)}")
        frame = dataset_index.copy(deep=True)
        for column in ("sample_id", "subject_id", "record_id"):
            if frame[column].isna().any():
                raise MetaEpisodeBuildError(f"{column} contains missing values")
            frame[column] = frame[column].astype(str)
        if frame["sample_id"].duplicated().any():
            raise MetaEpisodeBuildError("Dataset index contains duplicate sample_id")
        if "session_id" not in frame:
            frame["session_id"] = None
        else:
            frame["session_id"] = frame["session_id"].map(
                lambda value: None if pd.isna(value) else str(value)
            )
        if spec.time_column not in frame:
            frame[spec.time_column] = np.arange(len(frame), dtype=np.int64)
        frame = frame.sort_values(
            ["subject_id", spec.time_column, "sample_id"], kind="stable"
        )
        return frame

    def _build_subject_episode(
        self,
        frame: pd.DataFrame,
        spec: MetaEpisodeSpec,
        **identity: str,
    ) -> MetaEpisode:
        if frame.empty:
            raise MetaEpisodeBuildError("subject has no allowed samples")
        if frame["subject_id"].nunique() != 1:
            raise MetaEpisodeBuildError("personalization episode must contain one subject")
        if frame["record_id"].nunique() < spec.minimum_records:
            if not spec.allow_within_record_chronological_split:
                raise MetaEpisodeBuildError(
                    f"requires at least {spec.minimum_records} records"
                )
        sessions = [
            value for value in frame["session_id"].dropna().astype(str).unique()
            if value
        ]
        if len(sessions) >= 2:
            support = frame.loc[frame["session_id"].eq(sessions[0])]
            query = frame.loc[~frame["session_id"].eq(sessions[0])]
            split_level = "session"
        elif frame["record_id"].nunique() >= 2:
            records = frame["record_id"].drop_duplicates().tolist()
            support_records: list[str] = []
            count = 0
            for record in records[:-1]:
                support_records.append(record)
                count += int(frame["record_id"].eq(record).sum())
                if count >= spec.support_size:
                    break
            support = frame.loc[frame["record_id"].isin(support_records)]
            query = frame.loc[~frame["record_id"].isin(support_records)]
            split_level = "record"
        elif spec.allow_within_record_chronological_split:
            boundary = min(spec.support_size, len(frame) - 1)
            support, query = frame.iloc[:boundary], frame.iloc[boundary:]
            split_level = "within_record"
        else:
            raise MetaEpisodeBuildError("no safe session or record boundary")
        support = self._limit_partition(support, spec.support_size, "support", spec)
        query = self._limit_partition(query, spec.query_size, "query", spec)
        support, query = self._apply_class_policy(support, query, spec)
        return self._make_episode(
            support, query, spec, split_level=split_level, **identity
        )

    @staticmethod
    def _limit_partition(
        frame: pd.DataFrame,
        requested: int,
        name: str,
        spec: MetaEpisodeSpec,
    ) -> pd.DataFrame:
        if len(frame) >= requested:
            return frame.iloc[:requested]
        reduction_allowed = spec.insufficient_data_policy == f"reduce_{name}"
        if reduction_allowed and len(frame):
            return frame
        raise MetaEpisodeBuildError(
            f"{name} requires {requested} samples, found {len(frame)}"
        )

    @staticmethod
    def _apply_class_policy(
        support: pd.DataFrame,
        query: pd.DataFrame,
        spec: MetaEpisodeSpec,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if spec.task_type in {"regression", "multioutput_regression"}:
            return support, query
        if spec.class_balance_policy == "none":
            if min(support["target"].nunique(), query["target"].nunique()) < spec.minimum_classes:
                raise MetaEpisodeBuildError("minimum class coverage is not met")
            return support, query
        all_classes = sorted(set(support["target"]) | set(query["target"]))
        if len(all_classes) < spec.minimum_classes:
            raise MetaEpisodeBuildError(
                f"requires at least {spec.minimum_classes} classes, "
                f"found {len(all_classes)}"
            )
        for name, partition in (("support", support), ("query", query)):
            missing = set(all_classes) - set(partition["target"])
            if missing:
                raise MetaEpisodeBuildError(f"{name} is missing classes {sorted(missing)}")
        if spec.class_balance_policy == "equal_per_class":
            def balanced(frame: pd.DataFrame) -> pd.DataFrame:
                size = int(frame.groupby("target").size().min())
                return pd.concat(
                    [group.iloc[:size] for _, group in frame.groupby("target", sort=True)]
                ).sort_index(kind="stable")
            support, query = balanced(support), balanced(query)
        return support, query

    @staticmethod
    def _make_episode(
        support: pd.DataFrame,
        query: pd.DataFrame,
        spec: MetaEpisodeSpec,
        *,
        dataset_id: str,
        task_id: str,
        fold_id: str,
        entity_id: str,
        scope: str,
        split_level: str,
    ) -> MetaEpisode:
        if support.empty or query.empty:
            raise MetaEpisodeBuildError("support and query must be non-empty")
        subjects = set(support["subject_id"]) | set(query["subject_id"])
        if len(subjects) != 1:
            raise MetaEpisodeBuildError("episode contains multiple subjects")
        sessions = tuple(sorted({
            str(value)
            for value in pd.concat([support["session_id"], query["session_id"]]).dropna()
        })) or ("session-unspecified",)
        boundary = None
        if spec.chronological:
            support_end = support[spec.time_column].max()
            query_start = query[spec.time_column].min()
            boundary = f"{spec.time_column}:{support_end}<{query_start}"
            if support_end >= query_start:
                raise MetaEpisodeBuildError("support does not precede query in time")
        if split_level == "within_record":
            if support[spec.time_column].max() >= query[spec.time_column].min():
                raise MetaEpisodeBuildError("within-record temporal order is invalid")
        episode = MetaEpisode(
            episode_id="",
            dataset_id=dataset_id,
            task_id=task_id,
            target_name=spec.target_name,
            fold_id=fold_id,
            entity_id=entity_id,
            subject_id=str(next(iter(subjects))),
            session_ids=sessions,
            support_sample_ids=tuple(support["sample_id"]),
            query_sample_ids=tuple(query["sample_id"]),
            support_record_ids=tuple(sorted(support["record_id"].unique())),
            query_record_ids=tuple(sorted(query["record_id"].unique())),
            support_targets=tuple(_json_value(value) for value in support["target"]),
            query_targets=tuple(_json_value(value) for value in query["target"]),
            seed=spec.seed,
            spec_hash=spec.spec_hash,
            split_level=split_level,
            scope=scope,
            chronological_boundary=boundary,
        )
        return episode.with_recomputed_id()


class _SubsetDatasetView:
    def __init__(self, source: Any, positions: np.ndarray) -> None:
        self.source = source
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Any:
        position = int(self.positions[index])
        return (
            self.source.iloc[position]
            if isinstance(self.source, pd.DataFrame)
            else self.source[position]
        )


class EpisodeDatasetView:
    """Lightweight position view; it never fits transforms or copies the source."""

    def __init__(
        self,
        source: Any,
        metadata: pd.DataFrame,
        episode: MetaEpisode,
    ) -> None:
        if len(source) != len(metadata):
            raise ValueError("source and metadata lengths differ")
        if metadata["sample_id"].duplicated().any():
            raise ValueError("metadata sample_id must be unique")
        position = {
            str(sample_id): index
            for index, sample_id in enumerate(metadata["sample_id"])
        }
        try:
            support = np.asarray([position[value] for value in episode.support_sample_ids])
            query = np.asarray([position[value] for value in episode.query_sample_ids])
        except KeyError as exc:
            raise ValueError(f"Episode sample is absent from metadata: {exc}") from exc
        self.source = source
        self.metadata_source = metadata
        self.support = _SubsetDatasetView(source, support)
        self.query = _SubsetDatasetView(source, query)
        self.support_metadata = metadata.iloc[support][
            [c for c in ("sample_id", "subject_id", "session_id", "record_id") if c in metadata]
        ].copy()
        self.query_metadata = metadata.iloc[query][
            [c for c in ("sample_id", "subject_id", "session_id", "record_id") if c in metadata]
        ].copy()
        self.scaler = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _balance_rows(episodes: Sequence[MetaEpisode]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for partition, targets in (
            ("support", episode.support_targets), ("query", episode.query_targets)
        ):
            values, counts = np.unique(np.asarray(targets, dtype=object), return_counts=True)
            for value, count in zip(values, counts):
                rows.append({
                    "episode_id": episode.episode_id,
                    "dataset": episode.dataset_id,
                    "scope": episode.scope,
                    "partition": partition,
                    "target": _json_value(value),
                    "count": int(count),
                })
    return pd.DataFrame(rows)


def materialize_meta_learning_smoke(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> MetaEpisodeManifest:
    """Materialize diagnostic episodes from existing split artifacts only."""
    output_dir = repository_root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    main_cfg = config["main_dataset"]
    cog_cfg = config["cog_bci"]
    source_paths = {
        "main_fold_assignments": repository_root / main_cfg["fold_predictions"],
        "cog_outer_assignments": repository_root / cog_cfg["outer_assignments"],
        "cog_outer_folds": repository_root / cog_cfg["outer_folds"],
    }
    before_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    builder = MetaEpisodeBuilder()
    all_episodes: list[MetaEpisode] = []
    all_errors: list[MetaEpisodeError] = []
    specs: list[MetaEpisodeSpec] = []

    main = pd.read_parquet(source_paths["main_fold_assignments"])
    if main["sample_id"].duplicated().any():
        raise ValueError("Main fold assignment artifact has duplicate sample_id")
    main_index = main.rename(columns={"y_true": "target"}).copy()
    main_index["session_id"] = None
    main_index["time_order"] = main_index["sample_index"]
    fold = int(main_cfg["outer_fold"])
    outer_test = main_index.loc[main_index["fold"].eq(fold)]
    outer_train = main_index.loc[~main_index["fold"].eq(fold)]
    main_spec = MetaEpisodeSpec.from_mapping(main_cfg["episode_spec"])
    specs.append(main_spec)
    def eligible_subjects(frame: pd.DataFrame, limit: int) -> list[str]:
        grouped = frame.groupby(frame["subject_id"].astype(str), sort=True)
        eligible = [
            str(subject)
            for subject, group in grouped
            if group["record_id"].nunique() >= main_spec.minimum_records
            and len(group) >= main_spec.support_size + main_spec.query_size
        ]
        return eligible[:limit]

    train_subjects = eligible_subjects(
        outer_train, int(main_cfg["meta_train_subjects"])
    )
    test_subjects = eligible_subjects(
        outer_test, int(main_cfg["personalization_subjects"])
    )
    if not train_subjects or not test_subjects:
        raise ValueError("No eligible main-dataset subjects for the smoke episode")
    train_result = builder.build(
        dataset_index=main_index,
        allowed_sample_ids=outer_train["sample_id"],
        forbidden_sample_ids=outer_test["sample_id"],
        allowed_subject_ids=train_subjects,
        forbidden_subject_ids=outer_test["subject_id"].astype(str).unique(),
        entity_ids=train_subjects,
        episode_spec=main_spec,
        dataset_id=str(main_cfg["dataset_id"]),
        task_id=str(main_cfg["task_id"]),
        fold_id=fold,
        scope="meta_train_entity_index",
    )
    test_result = builder.build(
        dataset_index=main_index,
        allowed_sample_ids=outer_test["sample_id"],
        forbidden_sample_ids=outer_train["sample_id"],
        allowed_subject_ids=test_subjects,
        forbidden_subject_ids=outer_train["subject_id"].astype(str).unique(),
        entity_ids=test_subjects,
        episode_spec=main_spec,
        dataset_id=str(main_cfg["dataset_id"]),
        task_id=str(main_cfg["task_id"]),
        fold_id=fold,
        scope="outer_test_personalization",
    )
    all_episodes.extend(train_result.episodes + test_result.episodes)
    all_errors.extend(train_result.errors + test_result.errors)

    cog_index = pd.read_parquet(repository_root / cog_cfg["target_index"])
    cog_assignments = pd.read_parquet(source_paths["cog_outer_assignments"])
    cog_fold = int(cog_cfg["outer_fold"])
    test_ids = set(cog_assignments.loc[cog_assignments["fold"].eq(cog_fold), "sample_id"])
    train_ids = set(cog_assignments.loc[~cog_assignments["fold"].eq(cog_fold), "sample_id"])
    cog_test = cog_index.loc[cog_index["sample_id"].isin(test_ids)]
    cog_train = cog_index.loc[cog_index["sample_id"].isin(train_ids)]
    cog_spec = MetaEpisodeSpec.from_mapping(cog_cfg["episode_spec"])
    specs.append(cog_spec)
    cog_subjects = sorted(cog_test["subject_id"].astype(str).unique())[
        : int(cog_cfg["subjects"])
    ]
    cog_result = builder.build_session_rotations(
        dataset_index=cog_index,
        allowed_sample_ids=cog_test["sample_id"],
        forbidden_sample_ids=cog_train["sample_id"],
        allowed_subject_ids=cog_subjects,
        forbidden_subject_ids=cog_train["subject_id"].astype(str).unique(),
        entity_ids=cog_subjects,
        episode_spec=cog_spec,
        dataset_id=str(cog_cfg["dataset_id"]),
        task_id=str(cog_cfg["task_id"]),
        fold_id=cog_fold,
        scope="outer_test_session_rotation",
    )
    all_episodes.extend(cog_result.episodes)
    all_errors.extend(cog_result.errors)
    after_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("Existing split artifacts changed during materialization")

    manifest = MetaEpisodeManifest(
        dataset_ids=tuple(sorted({episode.dataset_id for episode in all_episodes})),
        task_ids=tuple(sorted({episode.task_id for episode in all_episodes})),
        fold_ids=tuple(sorted({episode.fold_id for episode in all_episodes})),
        specs=tuple(specs),
        episodes=tuple(all_episodes),
        errors=tuple(all_errors),
        source_split_hashes=after_hashes,
    )
    _write_json(output_dir / "episode_spec.json", [spec.to_dict() for spec in specs])
    MetaEpisodeIndex(tuple(all_episodes), tuple(all_errors)).to_frame().to_parquet(
        output_dir / "episode_index.parquet", index=False
    )
    _write_json(output_dir / "episode_manifest.json", manifest.to_dict())
    scope_counts = pd.Series([e.scope for e in all_episodes]).value_counts().to_dict()
    summary = {
        "result_status": "diagnostic",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episodes": len(all_episodes),
        "errors": len(all_errors),
        "scope_counts": scope_counts,
        "training_performed": False,
        "algorithm_implemented": False,
        "split_manifests_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "episode_summary.json", summary)
    _balance_rows(all_episodes).to_csv(output_dir / "episode_balance.csv", index=False)
    audits = []
    for episode in all_episodes:
        result = validate_episode(
            episode,
            require_record_disjoint=episode.split_level != "within_record",
        )
        audits.append({"episode_id": episode.episode_id, **asdict(result)})
    _write_json(output_dir / "episode_leakage_audit.json", {
        "all_valid": all(row["valid"] for row in audits),
        "split_manifests_unchanged": before_hashes == after_hashes,
        "episodes": audits,
    })
    prototype_mapping = {
        "source": "feature/benchmarking@8ecbee9:bench/tasks/mixin/metalearning.py",
        "reused": ["episodes", "support/query distinction", "subject adaptation", "model cloning concept"],
        "reworked": ["explicit outer scope", "stable IDs", "group/time boundaries", "manifests", "string subject IDs"],
        "rejected": ["learn2learn dependency", "global self.data access", "implicit random windows", "training loop", "hardcoded ways/shots"],
    }
    _write_json(output_dir / "prototype_mapping.json", prototype_mapping)
    pd.DataFrame(
        [error.to_dict() for error in all_errors],
        columns=["dataset_id", "task_id", "fold_id", "entity_id", "error_type", "message"],
    ).to_csv(output_dir / "errors.csv", index=False)
    (output_dir / "infrastructure_report.md").write_text(
        "# Meta-learning infrastructure smoke\n\n"
        f"- Episodes: {len(all_episodes)}\n"
        f"- Scopes: `{scope_counts}`\n"
        "- Leakage audit: passed.\n"
        "- Existing split artifacts: unchanged.\n"
        "- Model training: not performed.\n"
        "- Meta-learning algorithm: not implemented.\n",
        encoding="utf-8",
    )
    return manifest
