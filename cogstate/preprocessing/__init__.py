"""Public EEG preprocessing API with explicit online/offline semantics.

``apply_faster_online`` and :class:`StreamingPreprocessingPipeline` never use
future samples or fit ICA.  ``run_faster`` is the complete offline, epoched
four-stage FASTER implementation.  The fixed :class:`ArtifactICA` adapter is
fitted only on caller-authorized calibration/training data.
"""

from .artifact_removal import (
    ArtifactICA,
    FasterConfig as OnlineFasterConfig,
    FasterReport as OnlineFasterReport,
    IcaConfig,
    apply_faster as apply_faster_online,
    run_faster as run_faster_lightweight,
)
from .denoising import (
    WaveletDenoisingConfig,
    WaveletDenoisingReport,
    baseline_correct_epochs,
    detrend_signal,
    wavelet_denoise,
)
from .eog import EOGRegression, EOGRegressionReport, regress_eog
from .filtering import FilterConfig, StreamingFilter, apply_causal, apply_offline
from .full_faster import (
    FasterConfig,
    FasterReport,
    detect_bad_channel_epoch_pairs,
    detect_bad_channels,
    detect_bad_components,
    detect_bad_epochs,
    interpolate_channels,
    run_faster,
)
from .offline import (
    OfflinePreprocessingConfig,
    OfflinePreprocessingPipeline,
    OfflinePreprocessingReport,
    OfflinePreprocessingResult,
)
from .pipeline import (
    PreprocessingPipeline as StreamingPreprocessingPipeline,
    build_default_pipeline as build_default_streaming_pipeline,
)
from .referencing import (
    ReferenceMethod,
    ReferenceReport,
    common_average_reference,
    median_reference,
    rereference,
    robust_average_reference,
)

FullFasterConfig = FasterConfig
FullFasterReport = FasterReport
run_faster_full = run_faster

__all__ = [
    "ArtifactICA",
    "EOGRegression",
    "EOGRegressionReport",
    "FasterConfig",
    "FasterReport",
    "FilterConfig",
    "FullFasterConfig",
    "FullFasterReport",
    "IcaConfig",
    "OfflinePreprocessingConfig",
    "OfflinePreprocessingPipeline",
    "OfflinePreprocessingReport",
    "OfflinePreprocessingResult",
    "OnlineFasterConfig",
    "OnlineFasterReport",
    "ReferenceMethod",
    "ReferenceReport",
    "StreamingFilter",
    "StreamingPreprocessingPipeline",
    "WaveletDenoisingConfig",
    "WaveletDenoisingReport",
    "apply_causal",
    "apply_faster_online",
    "apply_offline",
    "baseline_correct_epochs",
    "build_default_streaming_pipeline",
    "common_average_reference",
    "detect_bad_channel_epoch_pairs",
    "detect_bad_channels",
    "detect_bad_components",
    "detect_bad_epochs",
    "detrend_signal",
    "interpolate_channels",
    "median_reference",
    "regress_eog",
    "rereference",
    "robust_average_reference",
    "run_faster",
    "run_faster_full",
    "run_faster_lightweight",
    "wavelet_denoise",
]
