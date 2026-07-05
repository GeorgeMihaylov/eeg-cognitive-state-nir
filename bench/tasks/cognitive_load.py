import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Any, Optional
from .base import BaseEEGTask
from ..core.task import TaskSplit


class CognitiveLoadTask(BaseEEGTask):
    def __init__(self, data: EEGData, config: Dict[str, Any]):
        super().__init__(data, config)
        self._validate_classes()

    def _validate_classes(self):
        unique = np.unique(self.data.labels)
        if len(unique) != 3:
            print(f"Warning: Expected 3 classes, got {len(unique)}. Proceeding anyway.")

    def get_split(self, subject_id: Optional[str] = None) -> TaskSplit:
        """
        Получить разбиение

        Args:
            subject_id: Если None → Within-Subject (StratifiedKFold)
                       Если указан → LOSO
        """
        X = self.data.data
        y = self.data.labels
        subjects = self.data.subject_ids

        if subject_id is None:
            return self._within_subject_split(X, y)
        else:
            return self._loso_split(X, y, subjects, subject_id)

    def _within_subject_split(self, X: np.ndarray, y: np.ndarray) -> TaskSplit:
        skf = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in skf.split(X, y):
            return TaskSplit(
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                feature_names=self.data.feature_names,
                metadata={'split_type': 'within_subject_stratified'}
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return TaskSplit(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_names=self.data.feature_names,
            metadata={'split_type': 'within_subject_random'}
        )

    def _loso_split(self, X: np.ndarray, y: np.ndarray,
                    subjects: np.ndarray, subject_id: str) -> TaskSplit:
        test_mask = (subjects == subject_id)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            raise ValueError(f"No data found for subject {subject_id}")

        return TaskSplit(
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask],
            y_test=y[test_mask],
            subject_train=subjects[train_mask],
            subject_test=subjects[test_mask],
            feature_names=self.data.feature_names,
            metadata={
                'split_type': 'loso',
                'test_subject': subject_id,
                'n_train_subjects': len(np.unique(subjects[train_mask])),
                'n_test_samples': np.sum(test_mask)
            }
        )

    @property
    def name(self) -> str:
        return 'cognitive_load_3class'
