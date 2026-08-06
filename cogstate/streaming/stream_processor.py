"""
StreamProcessor связывает SignalBuffer, модуль предобработки
(cogstate.preprocessing), извлечение признаков (cogstate.features)
и InferenceService. Работает в отдельном потоке: периодически
опрашивает буфер, и если накопилось новое окно — прогоняет его
через весь пайплайн и публикует результат.

Модули cogstate.preprocessing / cogstate.features подключаются как
интерфейсы (Protocol), чтобы stream_processor не был жёстко привязан
к конкретной реализации фильтрации или набора признаков — это
упрощает эксперименты с разными вариантами (10.2.2, 10.2.3).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from .buffer import SignalBuffer, Window
from .inference_service import InferenceService, PredictionResult
from .latency_monitor import LatencyMonitor

logger = logging.getLogger(__name__)


class Preprocessor(Protocol):
    """Интерфейс модуля предобработки (фильтрация, FASTER/ICA)."""

    def __call__(self, window: Window) -> np.ndarray:
        """Вернуть очищенный сигнал в виде [n_samples, n_channels]."""
        ...


class FeatureExtractor(Protocol):
    """Интерфейс извлечения признаков (спектральные, энтропийные и т.д.)."""

    def __call__(self, clean_signal: np.ndarray, window: Window) -> np.ndarray:
        """Вернуть вектор признаков фиксированной размерности."""
        ...


@dataclass
class ProcessedResult:
    window: Window
    prediction: PredictionResult
    stage_latencies_ms: dict


class StreamProcessor:
    """
    Потоковый обработчик: буфер -> предобработка -> признаки -> инференс.

    Использование:
        processor = StreamProcessor(
            buffer=signal_buffer,
            preprocessor=my_preprocessor,
            feature_extractor=my_feature_extractor,
            inference_service=inference_service,
            on_result=lambda r: print(r.prediction),
        )
        processor.start()
        ...
        processor.stop()
    """

    def __init__(
        self,
        buffer: SignalBuffer,
        preprocessor: Preprocessor,
        feature_extractor: FeatureExtractor,
        inference_service: InferenceService,
        on_result: Optional[Callable[[ProcessedResult], None]] = None,
        poll_interval_s: float = 0.05,
        latency_monitor: Optional[LatencyMonitor] = None,
    ):
        self.buffer = buffer
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.inference_service = inference_service
        self.on_result = on_result
        self.poll_interval_s = poll_interval_s
        self.latency_monitor = latency_monitor or LatencyMonitor()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("StreamProcessor уже запущен")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("StreamProcessor запущен")

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        logger.info("StreamProcessor остановлен")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            window = self.buffer.poll_window()
            if window is None:
                time.sleep(self.poll_interval_s)
                continue
            try:
                self._process_window(window)
            except Exception:
                logger.exception("Ошибка при обработке окна [%s, %s]",
                                  window.start_time, window.end_time)

    def _process_window(self, window: Window) -> None:
        trace = self.latency_monitor.start_trace(window)

        clean_signal = self.preprocessor(window)
        trace.mark("preprocessing")

        features = self.feature_extractor(clean_signal, window)
        trace.mark("feature_extraction")

        prediction = self.inference_service.predict(features)
        trace.mark("inference")

        trace.finish()
        self.latency_monitor.record(trace)

        result = ProcessedResult(
            window=window,
            prediction=prediction,
            stage_latencies_ms=trace.stage_latencies_ms(),
        )

        if self.on_result is not None:
            self.on_result(result)
