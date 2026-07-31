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
from .buffers import (
    BufferAuditResult,
    BufferPolicy,
    FunctionalModelState,
    FunctionalStateError,
    architecture_schema_signature,
    batchnorm_inventory,
    create_functional_state,
    functional_forward,
    validate_functional_state,
)
from .meta_validation import (
    MetaValidationProtocolResult,
    build_meta_validation_protocol,
)
from .production import (
    audit_architectures,
    run_fomaml_production_contract_audit,
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
    "BufferAuditResult",
    "BufferPolicy",
    "FOMAMLBatchResult",
    "FOMAMLConfig",
    "FOMAMLEpisodeResult",
    "FOMAMLError",
    "FOMAMLStepResult",
    "FirstOrderMAML",
    "FunctionalModelState",
    "FunctionalStateError",
    "MetaEpisode",
    "MetaEpisodeBuilder",
    "MetaEpisodeError",
    "MetaEpisodeIndex",
    "MetaEpisodeManifest",
    "MetaEpisodeSpec",
    "MetaEpisodeValidationResult",
    "MetaLearnerProtocol",
    "MetaValidationProtocolResult",
    "SyntheticClassifier",
    "SyntheticEpisodeData",
    "audit_production_model_compatibility",
    "architecture_schema_signature",
    "audit_architectures",
    "batchnorm_inventory",
    "build_meta_validation_protocol",
    "clone_model_for_episode",
    "create_functional_state",
    "materialize_meta_learning_smoke",
    "generate_synthetic_episodes",
    "model_state_hash",
    "functional_forward",
    "run_fomaml_production_contract_audit",
    "run_fomaml_synthetic_smoke",
    "validate_episode",
    "validate_model_clone",
    "validate_parameter_mapping",
    "validate_functional_state",
]
