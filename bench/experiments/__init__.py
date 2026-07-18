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
    "OrdinalTransformerSmokeExperiment",
    "OrdinalTransformerTrialPlan",
    "audit_prediction_probabilities",
]
