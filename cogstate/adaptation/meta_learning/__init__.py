"""Dataset-independent functional-state and first-order MAML primitives."""

from .buffers import (
    BufferAuditResult,
    BufferPolicy,
    FunctionalModelState,
    FunctionalStateError,
    architecture_schema_signature,
    batchnorm_inventory,
    create_functional_state,
    functional_forward,
    stable_model_class_path,
    validate_functional_state,
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
from .protocol import MetaLearnerProtocol, clone_model_for_episode, validate_model_clone

__all__ = [
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
    "MetaLearnerProtocol",
    "architecture_schema_signature",
    "audit_production_model_compatibility",
    "batchnorm_inventory",
    "clone_model_for_episode",
    "create_functional_state",
    "functional_forward",
    "model_state_hash",
    "stable_model_class_path",
    "validate_functional_state",
    "validate_model_clone",
    "validate_parameter_mapping",
]
