"""EEG preprocessing for offline training and streaming inference."""

from .artifact_removal import (
    ArtifactICA,
    FasterConfig,
    FasterReport,
    IcaConfig,
    apply_faster,
    detect_bad_channel_epoch_pairs,
    detect_bad_channels,
    detect_bad_components,
    detect_bad_epochs,
    interpolate_channels,
    run_faster,
)
from .denoising import (
    WaveletDenoisingConfig,
    WaveletDenoisingReport,
    baseline_correct_epochs,
    detrend_signal,
    wavelet_denoise,
)
from .eog import EOGRegression, EOGRegressionReport, regress_eog
from .offline import (
    OfflinePreprocessingConfig,
    OfflinePreprocessingPipeline,
    OfflinePreprocessingReport,
    OfflinePreprocessingResult,
)
from .filtering import FilterConfig, StreamingFilter, apply_causal, apply_offline
from .referencing import (
    ReferenceMethod,
    ReferenceReport,
    common_average_reference,
    median_reference,
    rereference,
    robust_average_reference,
)

__all__ = [
    "ArtifactICA",
    "EOGRegression",
    "EOGRegressionReport",
    "FasterConfig",
    "FasterReport",
    "FilterConfig",
    "IcaConfig",
    "OfflinePreprocessingConfig",
    "OfflinePreprocessingPipeline",
    "OfflinePreprocessingReport",
    "OfflinePreprocessingResult",
    "ReferenceMethod",
    "ReferenceReport",
    "StreamingFilter",
    "WaveletDenoisingConfig",
    "WaveletDenoisingReport",
    "baseline_correct_epochs",
    "apply_faster",
    "apply_causal",
    "apply_offline",
    "common_average_reference",
    "detrend_signal",
    "detect_bad_channel_epoch_pairs",
    "detect_bad_channels",
    "detect_bad_components",
    "detect_bad_epochs",
    "interpolate_channels",
    "median_reference",
    "regress_eog",
    "rereference",
    "robust_average_reference",
    "run_faster",
    "wavelet_denoise",
]
