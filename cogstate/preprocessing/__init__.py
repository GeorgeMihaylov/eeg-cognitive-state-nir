"""EEG preprocessing for offline training and streaming inference."""

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
from .mne_faster import (
    MNEFasterBundle,
    MNEFasterCalibrator,
    MNEFasterConfig,
    MNEFasterReport,
)
from .referencing import (
    ReferenceMethod,
    ReferenceReport,
    common_average_reference,
    median_reference,
    rereference,
    robust_average_reference,
)

__all__ = [
    "EOGRegression",
    "EOGRegressionReport",
    "FilterConfig",
    "MNEFasterBundle",
    "MNEFasterCalibrator",
    "MNEFasterConfig",
    "MNEFasterReport",
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
    "apply_causal",
    "apply_offline",
    "common_average_reference",
    "detrend_signal",
    "median_reference",
    "regress_eog",
    "rereference",
    "robust_average_reference",
    "wavelet_denoise",
]
