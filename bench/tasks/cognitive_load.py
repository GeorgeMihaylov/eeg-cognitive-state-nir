import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Any, Optional
from ..core.abstract_task import BaseTask, TaskSplit
from ..core.abstract_dataset import EEGData


class CognitiveLoadTask(BaseTask):
    def __init__(self, data: EEGData, config: Dict[str, Any]):
        super().__init__(data, config)
        self.test_size = config.get('test_size', 0.15)
        self.random_state = config.get('random_state', 42)
        self.n_splits = config.get('n_splits', 5)
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
        subjects = self.data.subject_ids
        unique_subjects = np.unique(subjects)
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for subj in unique_subjects:
            mask = (subjects == subj)
            X_subj = X[mask]
            y_subj = y[mask]

            if len(y_subj) < 2:
                continue

            if self.test_size < 1.0:
                test_size = self.test_size
            else:
                test_size = min(self.test_size, len(y_subj) - 1) / len(y_subj)

            if self.task_type == 'classification':
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_subj, y_subj, test_size=test_size,
                        random_state=self.random_state, stratify=y_subj
                    )
                except ValueError:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_subj, y_subj, test_size=test_size, random_state=self.random_state
                    )
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_subj, y_subj, test_size=test_size, random_state=self.random_state
                )

            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)

        if not X_train_list:
            raise ValueError("No subjects with enough data for within-subject split")

        X_train = np.vstack(X_train_list)
        X_test = np.vstack(X_test_list)
        y_train = np.concatenate(y_train_list)
        y_test = np.concatenate(y_test_list)

        return TaskSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=self.data.feature_names,
            task_type=self.task_type,
            metadata={'split_type': 'within_subject'}
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
