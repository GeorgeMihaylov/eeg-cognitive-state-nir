"""Public processor facade; old imports remain supported."""

from .stream_processor import (
    FeatureExtractor,
    Preprocessor,
    ProcessedResult,
    StreamProcessor,
)

__all__ = ["FeatureExtractor", "Preprocessor", "ProcessedResult", "StreamProcessor"]
