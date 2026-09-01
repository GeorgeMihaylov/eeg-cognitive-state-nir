import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any


class SplitUtils:

    @staticmethod
    def subject_wise_split(
            subject_ids: np.ndarray,
            train_size: float = 0.70,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:

        unique_subjects = np.unique(subject_ids)
        unique_subjects = np.sort(unique_subjects)
        train_subjects, temp_subjects = train_test_split(
            unique_subjects,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )
        relative_val_size = val_size / (val_size + test_size)
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            train_size=relative_val_size,
            random_state=random_state + 1,
            shuffle=True
        )
        train_idx = np.isin(subject_ids, train_subjects)
        val_idx = np.isin(subject_ids, val_subjects)
        test_idx = np.isin(subject_ids, test_subjects)

        split_meta = {
            'split_type': 'subject_wise',
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'n_test_subjects': len(test_subjects),
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist(),
            'test_subjects': test_subjects.tolist()
        }

        return train_idx, val_idx, test_idx, split_meta

    @staticmethod
    def record_wise_split(
            record_ids: np.ndarray,
            train_size: float = 0.70,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        return SplitUtils.subject_wise_split(record_ids, train_size, val_size, test_size, random_state)

    @staticmethod
    def sequence_wise_split(
            n_samples: int,
            train_size: float = 0.70,
            val_size: float = 0.15,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_train = int(n_samples * train_size)
        n_val = int(n_samples * val_size)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        split_meta = {
            'split_type': 'sequence_wise',
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx)
        }

        return train_idx, val_idx, test_idx, split_meta
