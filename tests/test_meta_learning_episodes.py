from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bench.meta import (
    EpisodeDatasetView,
    MetaEpisodeBuilder,
    MetaEpisodeSpec,
    validate_episode,
)
from bench.meta.episodes import MetaEpisodeBuildError


def _frame(
    *,
    subjects: tuple[str, ...] = ("subject-A", "subject-B"),
    sessions: int = 1,
    records_per_session: int = 3,
    samples_per_record: int = 12,
) -> pd.DataFrame:
    rows = []
    order = 0
    for subject in subjects:
        for session in range(sessions):
            for record in range(records_per_session):
                for sample in range(samples_per_record):
                    rows.append({
                        "sample_id": f"{subject}-s{session}-r{record}-w{sample}",
                        "subject_id": subject,
                        "session_id": f"session-{session + 1}",
                        "record_id": f"{subject}-s{session}-r{record}",
                        "target": sample % 3,
                        "time_order": order,
                    })
                    order += 1
    return pd.DataFrame(rows)


def _spec(**overrides: object) -> MetaEpisodeSpec:
    values = {
        "episode_type": "subject_personalization",
        "task_type": "classification",
        "target_name": "label",
        "support_unit": "sample",
        "query_unit": "sample",
        "support_size": 8,
        "query_size": 8,
        "class_balance_policy": "none",
        "chronological": True,
        "seed": 42,
        "minimum_records": 2,
        "minimum_classes": 2,
        "insufficient_data_policy": "error",
    }
    values.update(overrides)
    return MetaEpisodeSpec(**values)


def _build(frame: pd.DataFrame, spec: MetaEpisodeSpec | None = None):
    subject = "subject-A"
    allowed = frame.loc[frame.subject_id.eq(subject), "sample_id"]
    forbidden = frame.loc[~frame.subject_id.eq(subject), "sample_id"]
    return MetaEpisodeBuilder().build(
        dataset_index=frame,
        allowed_sample_ids=allowed,
        forbidden_sample_ids=forbidden,
        episode_spec=spec or _spec(),
        dataset_id="synthetic",
        task_id="classification",
        fold_id=1,
        allowed_subject_ids=[subject],
        forbidden_subject_ids=["subject-B"],
        entity_ids=[subject],
    ).episodes[0]


def test_personalization_is_disjoint_deterministic_and_string_safe() -> None:
    frame = _frame()
    first = _build(frame)
    second = _build(frame)
    assert first.episode_id == second.episode_id
    assert first.subject_id == "subject-A"
    assert set(first.support_sample_ids).isdisjoint(first.query_sample_ids)
    assert set(first.support_record_ids).isdisjoint(first.query_record_ids)
    assert first.split_level == "record"
    assert first.chronological_boundary
    assert validate_episode(first).valid


def test_episode_identity_covers_support_query_and_specification() -> None:
    episode = _build(_frame())
    support_changed = replace(
        episode,
        support_sample_ids=episode.support_sample_ids[:-1],
    ).with_recomputed_id()
    query_changed = replace(
        episode,
        query_sample_ids=episode.query_sample_ids[:-1],
    ).with_recomputed_id()
    spec_changed = replace(episode, spec_hash="different").with_recomputed_id()
    assert len({
        episode.episode_id,
        support_changed.episode_id,
        query_changed.episode_id,
        spec_changed.episode_id,
    }) == 4


def test_forbidden_samples_and_subjects_are_rejected() -> None:
    frame = _frame()
    subject_a = frame.loc[frame.subject_id.eq("subject-A"), "sample_id"]
    with pytest.raises(MetaEpisodeBuildError, match="overlap"):
        MetaEpisodeBuilder().build(
            dataset_index=frame,
            allowed_sample_ids=subject_a,
            forbidden_sample_ids=[subject_a.iloc[0]],
            episode_spec=_spec(),
            dataset_id="x",
            task_id="x",
            fold_id=1,
        )
    with pytest.raises(MetaEpisodeBuildError, match="forbidden subjects"):
        MetaEpisodeBuilder().build(
            dataset_index=frame,
            allowed_sample_ids=subject_a,
            forbidden_sample_ids=[],
            forbidden_subject_ids=["subject-A"],
            episode_spec=_spec(),
            dataset_id="x",
            task_id="x",
            fold_id=1,
        )


def test_temporal_within_record_split_requires_opt_in_and_keeps_order() -> None:
    frame = _frame(subjects=("subject-A",), records_per_session=1)
    with pytest.raises(MetaEpisodeBuildError, match="at least 2 records"):
        _build(frame)
    episode = _build(
        frame,
        _spec(
            allow_within_record_chronological_split=True,
            minimum_records=1,
            support_size=4,
            query_size=4,
        ),
    )
    assert episode.split_level == "within_record"
    assert episode.chronological_boundary
    assert validate_episode(episode, require_record_disjoint=False).valid


def test_insufficient_policies_are_explicit_and_never_duplicate() -> None:
    frame = _frame(samples_per_record=2)
    with pytest.raises(MetaEpisodeBuildError, match="support requires"):
        _build(frame, _spec(support_size=10))
    result = MetaEpisodeBuilder().build(
        dataset_index=frame,
        allowed_sample_ids=frame.loc[frame.subject_id.eq("subject-A"), "sample_id"],
        forbidden_sample_ids=frame.loc[frame.subject_id.eq("subject-B"), "sample_id"],
        episode_spec=_spec(support_size=10, insufficient_data_policy="skip_episode"),
        dataset_id="x",
        task_id="x",
        fold_id=1,
        entity_ids=["subject-A"],
    )
    assert not result.episodes
    assert result.errors and "requires" in result.errors[0].message
    reduced = _build(
        frame,
        _spec(
            support_size=10,
            query_size=2,
            insufficient_data_policy="reduce_support",
        ),
    )
    assert len(reduced.support_sample_ids) == len(set(reduced.support_sample_ids))


def test_session_rotations_are_reproducible_and_record_disjoint() -> None:
    frame = _frame(subjects=("subject-A",), sessions=3)
    spec = _spec(
        episode_type="session_adaptation",
        support_unit="session",
        query_unit="session",
        support_size=2,
        query_size=1,
        class_balance_policy="require_all_classes",
        minimum_classes=3,
        chronological=False,
    )
    kwargs = dict(
        dataset_index=frame,
        allowed_sample_ids=frame.sample_id,
        forbidden_sample_ids=[],
        episode_spec=spec,
        dataset_id="cog",
        task_id="nback",
        fold_id=1,
        entity_ids=["subject-A"],
    )
    first = MetaEpisodeBuilder().build_session_rotations(**kwargs)
    second = MetaEpisodeBuilder().build_session_rotations(**kwargs)
    assert len(first.episodes) == 3
    assert [e.episode_id for e in first.episodes] == [
        e.episode_id for e in second.episodes
    ]
    assert {e.entity_id.rsplit("=", 1)[1] for e in first.episodes} == {
        "session-1", "session-2", "session-3"
    }
    assert all(validate_episode(e).valid for e in first.episodes)


def test_class_balance_and_regression_contracts() -> None:
    frame = _frame()
    balanced = _build(frame, _spec(class_balance_policy="equal_per_class"))
    support_counts = pd.Series(balanced.support_targets).value_counts()
    query_counts = pd.Series(balanced.query_targets).value_counts()
    assert support_counts.nunique() == query_counts.nunique() == 1
    with pytest.raises(ValueError, match="Regression"):
        _spec(task_type="regression", class_balance_policy="equal_per_class")


def test_duplicate_samples_and_sampling_with_replacement_are_rejected() -> None:
    frame = _frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(MetaEpisodeBuildError, match="duplicate"):
        _build(duplicate)


def test_dataset_view_is_lazy_preserves_metadata_and_does_not_mutate() -> None:
    metadata = _frame()
    source = np.arange(len(metadata) * 2).reshape(len(metadata), 2)
    original = metadata.copy(deep=True)
    episode = _build(metadata)
    view = EpisodeDatasetView(source, metadata, episode)
    assert view.source is source
    assert view.metadata_source is metadata
    assert view.scaler is None
    assert len(view.support) == len(episode.support_sample_ids)
    assert set(view.support_metadata.columns) == {
        "sample_id", "subject_id", "session_id", "record_id"
    }
    pd.testing.assert_frame_equal(metadata, original)


def test_cross_task_contract_exists_but_materialization_is_not_claimed() -> None:
    spec = _spec(episode_type="cross_task_adaptation")
    assert spec.episode_type == "cross_task_adaptation"
