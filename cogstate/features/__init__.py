"""Canonical reusable EEG feature extraction package.

All core functions accept finite NumPy arrays in ``[samples, channels]``
layout. The package is target-free and does not import ``bench`` or
``model_zoo``. Feature selection must be fitted only on an authorized train
partition supplied by the caller.
"""

from .connectivity import (
    ConnectivityConfig,
    channel_pairs,
    compute_coherence_matrix,
    compute_plv_matrix,
)
from .entropy import EntropyConfig
from .montage import (
    CANONICAL_REGIONS,
    EMOTIV_14_CHANNELS,
    MONTAGE_SCHEMA_VERSION,
    build_montage_manifest,
)
from .pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeaturePipeline,
    FeaturePipelineConfig,
    build_default_pipeline,
)
from .regional import (
    REGIONAL_FEATURE_SCHEMA_VERSION,
    REGIONAL_FEATURE_SCHEMA_VERSION_V2,
    RegionalFeatureConfig,
    RegionalFeaturePipeline,
)
from .selection import FeatureSelector, SelectionConfig, SelectionResult
from .spectral import DEFAULT_BANDS, PowerSpectrum, SpectralConfig, compute_power_spectrum
from .statistical import StatisticalConfig

__all__ = [
    "ConnectivityConfig",
    "CANONICAL_REGIONS",
    "DEFAULT_BANDS",
    "EMOTIV_14_CHANNELS",
    "EntropyConfig",
    "FEATURE_SCHEMA_VERSION",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeatureSelector",
    "MONTAGE_SCHEMA_VERSION",
    "PowerSpectrum",
    "REGIONAL_FEATURE_SCHEMA_VERSION",
    "REGIONAL_FEATURE_SCHEMA_VERSION_V2",
    "RegionalFeatureConfig",
    "RegionalFeaturePipeline",
    "SelectionConfig",
    "SelectionResult",
    "SpectralConfig",
    "StatisticalConfig",
    "build_default_pipeline",
    "build_montage_manifest",
    "channel_pairs",
    "compute_coherence_matrix",
    "compute_plv_matrix",
    "compute_power_spectrum",
]
