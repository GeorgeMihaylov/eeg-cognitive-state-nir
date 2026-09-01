"""Inter-subject and participant adaptation methods.

This package contains leakage-safe transformations used to adapt models or
representations to new participants. Feature extraction itself belongs to
``cogstate.features``.
"""

from .feature_alignment import (
    FeatureAligner,
    FeatureAlignmentConfig,
    FeatureAlignmentStats,
    SUPPORTED_ALIGNMENT_METHODS,
    apply_alignment_shrinkage,
)
from .regression_calibration import (
    AffineCalibration,
    apply_affine_calibration,
    apply_bias_correction,
    fit_affine_calibration,
    fit_bias_correction,
)

__all__ = [
    "FeatureAligner",
    "FeatureAlignmentConfig",
    "FeatureAlignmentStats",
    "SUPPORTED_ALIGNMENT_METHODS",
    "apply_alignment_shrinkage",
    "AffineCalibration",
    "apply_affine_calibration",
    "apply_bias_correction",
    "fit_affine_calibration",
    "fit_bias_correction",
]
