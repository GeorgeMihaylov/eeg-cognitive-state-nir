from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    label: str
    probabilities: dict
    model_version: str
    is_calibrated: bool
    inference_time_ms: float
    target_labels: Optional[dict] = None
    target_probabilities: Optional[dict] = None


class CognitiveStateModel(Protocol):
    """Единый интерфейс модели — реализуется адаптерами в cogstate.models.*."""

    version: str

    def predict_proba(self, features: np.ndarray) -> dict:
        """Вернуть {class_name: probability} для одного вектора признаков."""
        ...


class Calibrator(Protocol):
    """Интерфейс few-shot калибровки (10.2.5)."""

    def calibrate(self, model: CognitiveStateModel, user_id: str,
                  calibration_features: np.ndarray, calibration_labels: np.ndarray) -> CognitiveStateModel:
        """Вернуть модель, адаптированную под конкретного пользователя."""
        ...


class InferenceService:
    """
    Потокобезопасный сервис инференса с горячей заменой модели
    (например, после калибровки под пользователя) без остановки потока.
    """

    def __init__(
        self,
        base_model: CognitiveStateModel,
        calibrator: Optional[Calibrator] = None,
    ):
        self._base_model = base_model
        self._active_model: CognitiveStateModel = base_model
        self._calibrator = calibrator
        self._is_calibrated = False
        self._current_user_id: Optional[str] = None
        self._lock = threading.RLock()

    def predict(self, features: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        with self._lock:
            model = self._active_model
            is_calibrated = self._is_calibrated

        if hasattr(model, "predict_pm_proba"):
            target_probabilities = model.predict_pm_proba(features)
            target_labels = {
                metric: max(probabilities, key=probabilities.get)
                for metric, probabilities in target_probabilities.items()
            }
            label = target_labels.get(
                "attention", next(iter(target_labels.values()), "unknown")
            )
            probabilities = target_probabilities.get("attention", {})
        else:
            target_labels = target_probabilities = None
            probabilities = model.predict_proba(features)
            label = max(probabilities, key=probabilities.get)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            label=label,
            probabilities=probabilities,
            model_version=model.version,
            is_calibrated=is_calibrated,
            inference_time_ms=elapsed_ms,
            target_labels=target_labels,
            target_probabilities=target_probabilities,
        )

    def calibrate_for_user(
        self,
        user_id: str,
        calibration_features: np.ndarray,
        calibration_labels: np.ndarray,
    ) -> None:
        """
        Запускает few-shot калибровку на небольшом объёме данных нового
        пользователя (10.2.5) и атомарно подменяет активную модель —
        текущий поток предсказаний продолжает работать на старой модели
        вплоть до завершения калибровки.
        """
        if self._calibrator is None:
            logger.warning("Калибратор не задан — используется базовая модель")
            return

        logger.info("Запуск калибровки модели для пользователя %s", user_id)
        calibrated_model = self._calibrator.calibrate(
            self._base_model, user_id, calibration_features, calibration_labels
        )

        with self._lock:
            self._active_model = calibrated_model
            self._is_calibrated = True
            self._current_user_id = user_id
        logger.info("Модель для пользователя %s откалибрована (версия %s)",
                    user_id, calibrated_model.version)

    def reset_to_base_model(self) -> None:
        """Сброс к базовой (неперсонализированной) модели."""
        with self._lock:
            self._active_model = self._base_model
            self._is_calibrated = False
            self._current_user_id = None
