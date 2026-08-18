"""
pipeline.py — сборка filtering + artifact_removal в единую функцию
для потокового StreamProcessor (см. streaming/stream_processor.py,
интерфейс Preprocessor).

Ожидается, что ICA уже откалибрована на записи пользователя заранее
(например, во время сессии калибровки — 10.2.5) и передана в
PreprocessingPipeline готовой; здесь она только применяется.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..streaming.buffer import Window

from .artifact_removal import (
    ArtifactICA,
    FasterConfig as OnlineFasterConfig,
    apply_faster as apply_faster_online,
)
from .filtering import FilterConfig, StreamingFilter


@dataclass
class PreprocessingPipeline:
    """
    Объект, реализующий интерфейс Preprocessor из stream_processor.py
    (вызывается как pipeline(window) -> np.ndarray).
    """

    streaming_filter: StreamingFilter
    faster_config: OnlineFasterConfig
    ica: Optional[ArtifactICA] = None    # None -> без ICA-очистки (например, до калибровки)

    def __call__(self, window: Window) -> np.ndarray:
        raw = window.data["eeg"]                       # [n_samples, n_channels]
        filtered = self.streaming_filter.process(raw)
        cleaned = apply_faster_online(filtered, self.faster_config)

        if self.ica is not None:
            cleaned = self.ica.transform(cleaned)

        return cleaned


def build_default_pipeline(
    sample_rate: float,
    n_channels: int,
    ica: Optional[ArtifactICA] = None,
) -> PreprocessingPipeline:
    """Пайплайн с настройками по умолчанию для быстрого старта прототипа."""
    streaming_filter = StreamingFilter(FilterConfig(sample_rate=sample_rate), n_channels=n_channels)
    return PreprocessingPipeline(
        streaming_filter=streaming_filter,
        faster_config=OnlineFasterConfig(),
        ica=ica,
    )
