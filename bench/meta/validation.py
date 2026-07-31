"""Independent leakage checks for materialized meta-learning episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .episodes import MetaEpisode


@dataclass(frozen=True)
class MetaEpisodeValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    support_query_sample_overlap: int
    support_query_record_overlap: int
    forbidden_sample_overlap: int
    forbidden_subject_overlap: int


def validate_episode(
    episode: "MetaEpisode",
    *,
    forbidden_sample_ids: Collection[str] = (),
    forbidden_subject_ids: Collection[str] = (),
    require_record_disjoint: bool = True,
) -> MetaEpisodeValidationResult:
    """Validate sample, record, and subject boundaries without using arrays."""
    support = set(episode.support_sample_ids)
    query = set(episode.query_sample_ids)
    support_records = set(episode.support_record_ids)
    query_records = set(episode.query_record_ids)
    forbidden_samples = {str(value) for value in forbidden_sample_ids}
    forbidden_subjects = {str(value) for value in forbidden_subject_ids}
    errors: list[str] = []
    warnings: list[str] = []

    sample_overlap = len(support & query)
    record_overlap = len(support_records & query_records)
    forbidden_overlap = len((support | query) & forbidden_samples)
    forbidden_subject_overlap = int(
        str(episode.subject_id) in forbidden_subjects
    )
    if sample_overlap:
        errors.append("support and query sample IDs overlap")
    if len(support) != len(episode.support_sample_ids):
        errors.append("support contains duplicate sample IDs")
    if len(query) != len(episode.query_sample_ids):
        errors.append("query contains duplicate sample IDs")
    if forbidden_overlap:
        errors.append("episode uses forbidden sample IDs")
    if forbidden_subject_overlap:
        errors.append("episode uses a forbidden subject")
    if record_overlap and require_record_disjoint:
        errors.append("support and query record IDs overlap")
    elif record_overlap:
        warnings.append("episode uses an explicit within-record chronological split")
    if not support or not query:
        errors.append("support and query must both be non-empty")
    if len(set(episode.session_ids)) < 1:
        errors.append("episode has no session identity")

    return MetaEpisodeValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        support_query_sample_overlap=sample_overlap,
        support_query_record_overlap=record_overlap,
        forbidden_sample_overlap=forbidden_overlap,
        forbidden_subject_overlap=forbidden_subject_overlap,
    )
