from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .abstract_dataset import EEGData


@dataclass
class TaskSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    subject_train: Optional[np.ndarray] = None
    subject_test: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    task_type: str = 'classification'
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    def __init__(self, data: EEGData, config: Dict[str, Any]):
        self.data = data
        self.config = config
        self.task_type = config.get('task_type', 'classification')
        self._validate()

    def _validate(self):
        if self.data.labels is None:
            raise ValueError("Labels are required for task")
        if len(self.data.data) != len(self.data.labels):
            raise ValueError("Data and labels must have same length")
        if self.task_type == 'classification':
            if not np.issubdtype(self.data.labels.dtype, np.integer):
                raise ValueError("Classification requires integer labels")

    @abstractmethod
    def get_split(self, subject_id: Optional[str] = None, split_strategy: str = 'random') -> TaskSplit:
        """
        Получить разбиение для задачи.

        Args:
            subject_id: ID субъекта для тестирования
                - Если None: Within-Subject (смешанные данные)
                - Если указан: LOSO (этот субъект на тест, остальные на обучение)
        """
        pass

    @abstractmethod
    def get_all_splits(self, split_strategy: str = 'loso') -> Dict[str, TaskSplit]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.data.labels))
