"""Leakage-safe episodic infrastructure for future meta-learning."""

from .episodes import (
    EpisodeDatasetView,
    MetaEpisode,
    MetaEpisodeBuilder,
    MetaEpisodeError,
    MetaEpisodeIndex,
    MetaEpisodeManifest,
    MetaEpisodeSpec,
    materialize_meta_learning_smoke,
)
from .protocol import (
    MetaLearnerProtocol,
    clone_model_for_episode,
    validate_model_clone,
)
from .validation import MetaEpisodeValidationResult, validate_episode

__all__ = [
    "EpisodeDatasetView",
    "MetaEpisode",
    "MetaEpisodeBuilder",
    "MetaEpisodeError",
    "MetaEpisodeIndex",
    "MetaEpisodeManifest",
    "MetaEpisodeSpec",
    "MetaEpisodeValidationResult",
    "MetaLearnerProtocol",
    "clone_model_for_episode",
    "materialize_meta_learning_smoke",
    "validate_episode",
    "validate_model_clone",
]
