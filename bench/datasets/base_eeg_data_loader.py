import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..core.dataset import BaseDataset, EEGData


class BaseEEGDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get('data_path', ''))
        self.feature_set = config.get('feature_set', 'pow_plus_eeg')
        self.n_classes = config.get('n_classes', 3)
        self._validate_path()

    def _validate_path(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def _load_dataframe(self) -> pd.DataFrame:
        if self.data_path.suffix == '.parquet':
            return pd.read_parquet(self.data_path)
        elif self.data_path.suffix in ['.csv', '.tsv']:
            return pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        all_cols = df.columns.tolist()

        exclude = ['record_id', 'source', 'subject_id', 'day', 'part',
                   'datetime_from_name', 't_center', 't_start', 't_end',
                   'target_main', 'label_q5']

        exclude.extend([c for c in all_cols if c.startswith('target_')])

        if self.feature_set == 'pow_plus_eeg':
            include = [c for c in all_cols if c not in exclude and
                       ('POW.' in c or 'EEG.' in c or 'eeg' in c.lower())]
        elif self.feature_set == 'pow':
            include = [c for c in all_cols if 'POW.' in c]
        elif self.feature_set == 'eeg':
            include = [c for c in all_cols if 'EEG.' in c]
        elif self.feature_set == 'all':
            include = [c for c in all_cols if c not in exclude]
        else:
            include = [c for c in all_cols if c not in exclude]

        max_features = self.config.get('max_features', 500)
        return include[:max_features]

    def _discretize_target(self, y: np.ndarray) -> np.ndarray:
        if not self.config.get('discretize', True):
            return y

        valid_mask = ~np.isnan(y)
        y_valid = y[valid_mask]

        if len(y_valid) < self.n_classes * 2:
            return pd.qcut(y_valid, q=self.n_classes, labels=False, duplicates='drop').values

        try:
            labels = pd.qcut(y_valid, q=self.n_classes, labels=False, duplicates='drop')
            result = np.full(len(y), -1, dtype=int)
            result[valid_mask] = labels.values
            result[result == -1] = 0
            unique = np.unique(result)
            if len(unique) < self.n_classes:
                for i in range(self.n_classes):
                    if i not in unique:
                        result[result >= i] += 1

            return result
        except Exception as e:
            y_valid = y[valid_mask]
            bins = np.linspace(y_valid.min(), y_valid.max(), self.n_classes + 1)
            labels = np.digitize(y_valid, bins[1:-1])

            result = np.full(len(y), -1, dtype=int)
            result[valid_mask] = labels
            result[result == -1] = 0

            return result

    def get_description(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'data_path': str(self.data_path),
            'feature_set': self.feature_set,
            'n_classes': self.n_classes
        }