"""Canonical reusable EEG feature extraction package.

All core functions accept finite NumPy arrays in ``[samples, channels]``
layout. The package is target-free and does not import ``bench`` or
``model_zoo``. Feature selection must be fitted only on an authorized train
partition supplied by the caller.
"""

from .connectivity import ConnectivityConfig, channel_pairs
from .entropy import EntropyConfig
from .pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeaturePipeline,
    FeaturePipelineConfig,
    build_default_pipeline,
)
from .selection import FeatureSelector, SelectionConfig, SelectionResult
from .spectral import DEFAULT_BANDS, SpectralConfig
from .statistical import StatisticalConfig

__all__ = [
    "ConnectivityConfig",
    "DEFAULT_BANDS",
    "EntropyConfig",
    "FEATURE_SCHEMA_VERSION",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeatureSelector",
    "SelectionConfig",
    "SelectionResult",
    "SpectralConfig",
    "StatisticalConfig",
    "build_default_pipeline",
    "channel_pairs",
]
