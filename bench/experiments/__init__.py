"""Programmatic experiment orchestration built on the benchmark API."""

from .preprocessing_ablation import (
    ExperimentTrial,
    PreprocessingAblation,
    TrialPlan,
    expand_factorial_trials,
    load_experiment_spec,
)
from .user_calibration import (
    CalibrationSpec,
    UserCalibrationExperiment,
    calibration_normalization_statistics,
    chronological_window_partition,
    resolve_calibration_parameters,
)
from .ordinal_transformer import (
    OrdinalTransformerSmokeExperiment,
    OrdinalTransformerTrialPlan,
    audit_prediction_probabilities,
    build_ordinal_transformer_experiment,
)
from .auxiliary_corn_transformer import (
    AUXILIARY_WEIGHTS,
    AuxiliaryCornTransformerSmokeExperiment,
    AuxiliaryCornTrialPlan,
    audit_auxiliary_corn_probabilities,
    load_auxiliary_corn_smoke_spec,
)
from .auxiliary_corn_lambda_selection import (
    AuxiliaryCornLambdaSelectionSetupExperiment,
    AuxiliaryCornLambdaSetupPlan,
    LambdaSelectionDecision,
    LambdaValidationResult,
    NoEligibleAuxiliaryWeightError,
    load_auxiliary_corn_lambda_setup_spec,
    select_auxiliary_weight,
)
from .ordinal_transformer_full import (
    OrdinalTransformerFullExperiment,
    OrdinalTransformerFullTrialPlan,
    full_prediction_alignment,
    load_ordinal_transformer_full_spec,
)

__all__ = [
    "ExperimentTrial",
    "PreprocessingAblation",
    "TrialPlan",
    "expand_factorial_trials",
    "load_experiment_spec",
    "CalibrationSpec",
    "UserCalibrationExperiment",
    "calibration_normalization_statistics",
    "chronological_window_partition",
    "resolve_calibration_parameters",
    "AUXILIARY_WEIGHTS",
    "AuxiliaryCornTransformerSmokeExperiment",
    "AuxiliaryCornTrialPlan",
    "audit_auxiliary_corn_probabilities",
    "load_auxiliary_corn_smoke_spec",
    "AuxiliaryCornLambdaSelectionSetupExperiment",
    "AuxiliaryCornLambdaSetupPlan",
    "LambdaSelectionDecision",
    "LambdaValidationResult",
    "NoEligibleAuxiliaryWeightError",
    "load_auxiliary_corn_lambda_setup_spec",
    "select_auxiliary_weight",
    "OrdinalTransformerSmokeExperiment",
    "OrdinalTransformerTrialPlan",
    "OrdinalTransformerFullExperiment",
    "OrdinalTransformerFullTrialPlan",
    "audit_prediction_probabilities",
    "build_ordinal_transformer_experiment",
    "full_prediction_alignment",
    "load_ordinal_transformer_full_spec",
]
