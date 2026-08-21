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

__all__ = [
    "FeatureAligner",
    "FeatureAlignmentConfig",
    "FeatureAlignmentStats",
    "SUPPORTED_ALIGNMENT_METHODS",
    "apply_alignment_shrinkage",
]
