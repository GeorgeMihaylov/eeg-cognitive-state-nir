import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..core.abstract_dataset import BaseDataset


FEATURE_SET_ALIASES = {
    'eeg': 'eeg_only',
    'eeg_only': 'eeg_only',
    'pow': 'pow_only',
    'pow_only': 'pow_only',
    'pow_plus_eeg': 'eeg_pow',
    'eeg_pow': 'eeg_pow',
    'all': 'all',
}


def feature_list_sha256(feature_names: List[str]) -> str:
    """Hash an ordered feature list using a documented line serialization."""

    payload = ''.join(f'{name}\n' for name in feature_names).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def resolve_feature_columns(
        columns: List[str],
        feature_set: str,
) -> List[str]:
    """Resolve deterministic EEG/POW groups while preserving dataset order."""

    try:
        normalized = FEATURE_SET_ALIASES[str(feature_set).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f'Unknown feature_set {feature_set!r}. '
            f'Available: {sorted(FEATURE_SET_ALIASES)}'
        ) from exc

    eeg_columns = {column for column in columns if column.startswith('EEG.')}
    pow_columns = {column for column in columns if column.startswith('POW.')}
    if normalized == 'eeg_only':
        return [column for column in columns if column in eeg_columns]
    if normalized == 'pow_only':
        return [column for column in columns if column in pow_columns]
    if normalized == 'eeg_pow':
        selected = eeg_columns | pow_columns
        return [column for column in columns if column in selected]

    exclude = {
        'record_id', 'source', 'subject_id', 'sample_id', 'window_id',
        'day', 'part', 'datetime_from_name', 't_center', 't_start', 't_end',
        'target_main', 'label_q5',
    }
    exclude.update(column for column in columns if column.startswith('target_'))
    exclude.update(column for column in columns if column.startswith('PM.'))
    return [column for column in columns if column not in exclude]


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
        include = resolve_feature_columns(all_cols, self.feature_set)
        max_features = self.config.get('max_features', 500)
        if max_features is not None:
            include = include[:int(max_features)]

        expected_count = self.config.get('expected_feature_count')
        if expected_count is not None and len(include) != int(expected_count):
            raise ValueError(
                f'Feature set {self.feature_set!r} resolved to {len(include)} '
                f'columns, expected {int(expected_count)}'
            )
        expected_hash = self.config.get('feature_list_sha256')
        actual_hash = feature_list_sha256(include)
        if expected_hash is not None and actual_hash != str(expected_hash):
            raise ValueError(
                f'Feature list hash mismatch for {self.feature_set!r}: '
                f'{actual_hash} != {expected_hash}'
            )
        return include

    def _discretize_target(self, y: np.ndarray) -> np.ndarray:
        # Preserve missing values and avoid rediscretizing stored class labels.
        values = np.asarray(y, dtype=float)

        if not self.config.get('discretize', True):
            return values

        valid_mask = np.isfinite(values)
        result = np.full(values.shape, np.nan, dtype=float)
        y_valid = values[valid_mask]

        if y_valid.size == 0:
            return result

        unique_values = np.unique(y_valid)
        integer_like = np.allclose(
            unique_values,
            np.round(unique_values),
        )
        configured_classes = set(range(int(self.n_classes)))
        observed_classes = {
            int(value)
            for value in unique_values.tolist()
        }

        if integer_like and observed_classes == configured_classes:
            result[valid_mask] = y_valid
            return result

        try:
            labels = pd.qcut(
                y_valid,
                q=self.n_classes,
                labels=False,
                duplicates='drop',
            )
            result[valid_mask] = np.asarray(labels, dtype=float)
        except (ValueError, TypeError):
            bins = np.linspace(
                float(y_valid.min()),
                float(y_valid.max()),
                self.n_classes + 1,
            )
            result[valid_mask] = np.digitize(
                y_valid,
                bins[1:-1],
            ).astype(float)

        return result

    def get_description(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'data_path': str(self.data_path),
            'feature_set': self.feature_set,
            'n_classes': self.n_classes
        }
