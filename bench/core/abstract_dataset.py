from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class EEGData:
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    feature_names: Optional[List[str]] = None
    sampling_rate: float = 128.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_ids: Optional[np.ndarray] = None
    record_ids: Optional[np.ndarray] = None
    row_metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.data.shape[1])]
        if self.sample_ids is None:
            self.sample_ids = np.arange(len(self.data), dtype=np.int64)
        if self.record_ids is None:
            self.record_ids = np.full(len(self.data), 'unknown', dtype=object)
        arrays = {
            'labels': self.labels,
            'subject_ids': self.subject_ids,
            'sample_ids': self.sample_ids,
            'record_ids': self.record_ids,
            **self.row_metadata,
        }
        invalid = {
            name: len(values)
            for name, values in arrays.items()
            if len(values) != len(self.data)
        }
        if invalid:
            raise ValueError(
                f"Row metadata lengths must match data ({len(self.data)}), got {invalid}"
            )

    def get_row_values(self, column: str) -> np.ndarray:
        """Return a configured row-level grouping or identifier column."""
        standard_columns = {
            'subject_id': self.subject_ids,
            'sample_id': self.sample_ids,
            'record_id': self.record_ids,
        }
        if column in standard_columns:
            return np.asarray(standard_columns[column])
        if column in self.row_metadata:
            return np.asarray(self.row_metadata[column])
        available = sorted(set(standard_columns) | set(self.row_metadata))
        raise ValueError(
            f"Row metadata column {column!r} is unavailable. Available: {available}"
        )

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        return self.data.shape[1] if len(self.data.shape) == 2 else self.data.shape[2]

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject_ids))

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.labels))

    @property
    def n_outputs(self) -> int:
        labels = np.asarray(self.labels)
        return 1 if labels.ndim == 1 else int(labels.shape[1])


class BaseDataset(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._data: Optional[EEGData] = None

    @abstractmethod
    def load(self) -> EEGData:
        pass

    @abstractmethod
    def get_description(self) -> Dict[str, Any]:
        pass

    @property
    def data(self) -> EEGData:
        if self._data is None:
            self._data = self.load()
        return self._data
