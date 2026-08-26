"""Compatibility facade for canonical record-safe sequence construction."""

from model_zoo.DL.sequence_utils import (
    SequenceBuildResult,
    build_sequences,
    sequence_index_sha256,
)

__all__ = ["SequenceBuildResult", "build_sequences", "sequence_index_sha256"]
