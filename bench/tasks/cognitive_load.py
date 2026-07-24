import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Any, Optional
from ..core.abstract_task import BaseTask, TaskSplit
from ..core.abstract_dataset import EEGData


class CognitiveLoadTask(BaseTask):
    expected_n_classes = 3
    task_type = 'classification'

    def __init__(self, data: EEGData, config: Dict[str, Any]):
        super().__init__(data, config)
        self.test_size = config.get('test_size', 0.15)
        self.random_state = config.get('random_state', 42)
        self.n_splits = config.get('n_splits', 5)
        self._validate_classes()

    def _validate_classes(self):
        unique = np.unique(self.data.labels)
        if len(unique) != self.expected_n_classes:
            raise ValueError(
                f"Expected {self.expected_n_classes} classes, got {len(unique)}: "
                f"{unique.tolist()}"
            )

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

    def get_all_splits(self) -> Dict[str, TaskSplit]:
        """
        Получить все LOSO разбиения для всех субъектов
        """
        subjects = np.unique(self.data.subject_ids)
        splits = {}

        for subject_id in subjects:
            try:
                splits[str(subject_id)] = self.get_split(subject_id)
            except ValueError as e:
                print(f"Warning: Could not create split for subject {subject_id}: {e}")

        return splits

    def _within_subject_split(self, X: np.ndarray, y: np.ndarray) -> TaskSplit:
        unique, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)

        if min_count >= 2:
            n_splits = min(self.n_splits, min_count)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            for train_idx, test_idx in skf.split(X, y):
                return self._build_indexed_split(
                    train_idx,
                    test_idx,
                    metadata={
                        'split_type': 'random_window_stratified_kfold_first_fold',
                        'n_splits': n_splits,
                        'random_state': self.random_state,
                    }
                )

        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return self._build_indexed_split(
            train_idx,
            test_idx,
            metadata={
                'split_type': 'random_window_train_test',
                'test_size': self.test_size,
                'random_state': self.random_state,
            }
        )

    def _loso_split(self, X: np.ndarray, y: np.ndarray,
                    subjects: np.ndarray, subject_id: str) -> TaskSplit:
        test_mask = (subjects == subject_id)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            raise ValueError(f"No data found for subject {subject_id}")

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        return self._build_indexed_split(
            train_idx,
            test_idx,
            metadata={
                'split_type': 'loso',
                'test_subject': subject_id,
                'n_train_subjects': len(np.unique(subjects[train_mask])),
                'n_test_samples': np.sum(test_mask)
            }
        )

    def _build_indexed_split(
            self,
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            metadata: Dict[str, Any]
    ) -> TaskSplit:
        split_metadata = {
            **metadata,
            'dataset_metadata': self.data.metadata,
            'target_names': self.data.metadata.get('target_cols'),
        }
        return TaskSplit(
            X_train=self.data.data[train_idx],
            y_train=self.data.labels[train_idx],
            X_test=self.data.data[test_idx],
            y_test=self.data.labels[test_idx],
            subject_train=self.data.subject_ids[train_idx],
            subject_test=self.data.subject_ids[test_idx],
            feature_names=self.data.feature_names,
            metadata=split_metadata,
            sample_id_train=self.data.sample_ids[train_idx],
            sample_id_test=self.data.sample_ids[test_idx],
            record_id_train=self.data.record_ids[train_idx],
            record_id_test=self.data.record_ids[test_idx],
            row_metadata_train={
                key: np.asarray(values)[train_idx]
                for key, values in self.data.row_metadata.items()
            },
            row_metadata_test={
                key: np.asarray(values)[test_idx]
                for key, values in self.data.row_metadata.items()
            },
        )

    @property
    def name(self) -> str:
        return 'cognitive_load_3class'


class CognitiveLoad5ClassTask(CognitiveLoadTask):
    expected_n_classes = 5

    @property
    def name(self) -> str:
        return 'cognitive_load_5class'


class FocusRegressionTask(CognitiveLoadTask):
    """Continuous ``target_focus`` task using the shared split machinery."""

    task_type = 'regression'

    def _validate_classes(self):
        labels = np.asarray(self.data.labels, dtype=float)
        if labels.ndim != 1:
            raise ValueError(
                f'Focus regression requires one-dimensional labels, got {labels.shape}'
            )
        if not np.isfinite(labels).all():
            raise ValueError('Regression labels must be finite')
        if len(np.unique(labels)) < 2:
            raise ValueError('Regression labels must contain variation')

    def _within_subject_split(self, X: np.ndarray, y: np.ndarray) -> TaskSplit:
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return self._build_indexed_split(
            train_idx,
            test_idx,
            metadata={
                'split_type': 'random_window_train_test',
                'test_size': self.test_size,
                'random_state': self.random_state,
                'task_type': self.task_type,
            },
        )

    @property
    def name(self) -> str:
        return 'focus_regression'


class PerformanceMetricsRegressionTask(FocusRegressionTask):
    """Joint regression of the configured Performance Metrics targets."""

    def __init__(self, data: EEGData, config: Dict[str, Any]):
        self.target_names = list(data.metadata.get('target_cols', []))
        configured_outputs = config.get('n_outputs')
        self.expected_n_outputs = (
            len(self.target_names)
            if configured_outputs is None
            else int(configured_outputs)
        )
        super().__init__(data, config)

    def _validate_classes(self):
        labels = np.asarray(self.data.labels, dtype=float)
        if labels.ndim != 2:
            raise ValueError(
                'Performance Metrics regression requires two-dimensional labels, '
                f'got {labels.shape}'
            )
        if labels.shape[1] != self.expected_n_outputs:
            raise ValueError(
                f'Expected {self.expected_n_outputs} regression outputs, '
                f'got {labels.shape[1]}'
            )
        if len(self.target_names) != labels.shape[1]:
            raise ValueError(
                'Dataset target_cols metadata must match the regression outputs'
            )
        if not np.isfinite(labels).all():
            raise ValueError('Regression labels must be finite')

    @property
    def name(self) -> str:
        return 'performance_metrics_regression'
