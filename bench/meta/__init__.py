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
from .fomaml import (
    FOMAMLBatchResult,
    FOMAMLConfig,
    FOMAMLEpisodeResult,
    FOMAMLError,
    FOMAMLStepResult,
    FirstOrderMAML,
    audit_production_model_compatibility,
    model_state_hash,
    validate_parameter_mapping,
)
from .synthetic import (
    SyntheticClassifier,
    SyntheticEpisodeData,
    generate_synthetic_episodes,
    run_fomaml_synthetic_smoke,
)
from .validation import MetaEpisodeValidationResult, validate_episode

__all__ = [
    "EpisodeDatasetView",
    "FOMAMLBatchResult",
    "FOMAMLConfig",
    "FOMAMLEpisodeResult",
    "FOMAMLError",
    "FOMAMLStepResult",
    "FirstOrderMAML",
    "MetaEpisode",
    "MetaEpisodeBuilder",
    "MetaEpisodeError",
    "MetaEpisodeIndex",
    "MetaEpisodeManifest",
    "MetaEpisodeSpec",
    "MetaEpisodeValidationResult",
    "MetaLearnerProtocol",
    "SyntheticClassifier",
    "SyntheticEpisodeData",
    "audit_production_model_compatibility",
    "clone_model_for_episode",
    "materialize_meta_learning_smoke",
    "generate_synthetic_episodes",
    "model_state_hash",
    "run_fomaml_synthetic_smoke",
    "validate_episode",
    "validate_model_clone",
    "validate_parameter_mapping",
]
