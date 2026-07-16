import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_eeg_data_loader import BaseEEGDataset
from ..core.abstract_dataset import EEGData


class EmotivDataset(BaseEEGDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_col = config.get('target_col', 'target_main')
        self.subject_col = config.get('subject_col', 'subject_id')

    def load(self) -> EEGData:
        df = self._load_dataframe()
        include_subject_ids = self.config.get('include_subject_ids')
        normalized_subject_ids = None
        if include_subject_ids is not None:
            if isinstance(include_subject_ids, (str, bytes)):
                raise ValueError('include_subject_ids must be a sequence of IDs')
            normalized_subject_ids = sorted({
                str(value) for value in include_subject_ids
            })
            if not normalized_subject_ids:
                raise ValueError('include_subject_ids must not be empty')
            if self.subject_col not in df.columns:
                raise ValueError(
                    f"Subject column {self.subject_col!r} is required for "
                    'include_subject_ids'
                )
            available_subjects = set(df[self.subject_col].astype(str))
            missing_subjects = sorted(
                set(normalized_subject_ids) - available_subjects
            )
            if missing_subjects:
                raise ValueError(
                    'include_subject_ids contains IDs absent from the dataset: '
                    f'{missing_subjects[:20]}'
                )
            df = df.loc[
                df[self.subject_col].astype(str).isin(normalized_subject_ids)
            ].copy()
            if df.empty:
                raise ValueError('include_subject_ids selected no dataset rows')
        sample_ids = (
            df['sample_id'].values
            if 'sample_id' in df.columns
            else np.arange(len(df), dtype=np.int64)
        )
        record_ids = (
            df['record_id'].astype(str).values
            if 'record_id' in df.columns
            else np.full(len(df), 'unknown', dtype=object)
        )
        row_metadata_columns = [
            column
            for column in (
                'source', 't_start', 't_center', 'window_id',
                'record_group_id',
            )
            if column in df.columns
        ]
        row_metadata = {
            column: df[column].values
            for column in row_metadata_columns
        }
        row_metadata.setdefault('record_group_id', record_ids.copy())
        feature_cols = self._select_features(df)
        if self.target_col not in df.columns:
            target_candidates = ['target_main', 'label_q5'] + [c for c in df.columns if c.startswith('target_')]
            for candidate in target_candidates:
                if candidate in df.columns:
                    self.target_col = candidate
                    break
            else:
                raise ValueError(f"No target column found. Available: {df.columns.tolist()[:20]}")
        X = df[feature_cols].values.astype(np.float32)
        y = df[self.target_col].values
        y = self._discretize_target(y)
        if self.subject_col in df.columns:
            subject_ids = df[self.subject_col].astype(str).values
        else:
            subject_ids = np.array(['unknown'] * len(df))
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        subject_ids = subject_ids[valid_mask]
        sample_ids = sample_ids[valid_mask]
        record_ids = record_ids[valid_mask]
        row_metadata = {
            column: values[valid_mask]
            for column, values in row_metadata.items()
        }
        max_windows = self.config.get('max_windows')
        if max_windows is not None:
            max_windows = int(max_windows)
            if max_windows <= 0:
                raise ValueError('max_windows must be positive')
            if len(X) > max_windows:
                selection = pd.DataFrame({
                    'position': np.arange(len(X), dtype=np.int64),
                    'subject_id': subject_ids.astype(str),
                    'record_id': record_ids.astype(str),
                    'source': np.asarray(
                        row_metadata.get(
                            'source', np.full(len(X), 'unknown', dtype=object)
                        )
                    ).astype(str),
                    'time': np.asarray(
                        row_metadata.get(
                            't_start', row_metadata.get(
                                't_center', np.arange(len(X), dtype=float)
                            )
                        )
                    ).astype(float),
                    'sample_id': sample_ids,
                })
                selection = selection.sort_values(
                    ['subject_id', 'source', 'record_id', 'time', 'sample_id'],
                    kind='mergesort',
                )
                selection['subject_rank'] = selection.groupby(
                    'subject_id', sort=True
                ).cumcount()
                selected = selection.sort_values(
                    ['subject_rank', 'subject_id', 'source', 'record_id', 'time'],
                    kind='mergesort',
                ).head(max_windows)['position'].to_numpy(
                    dtype=np.int64, copy=True
                )
                selected.sort()
                X = X[selected]
                y = y[selected]
                subject_ids = subject_ids[selected]
                sample_ids = sample_ids[selected]
                record_ids = record_ids[selected]
                row_metadata = {
                    column: values[selected]
                    for column, values in row_metadata.items()
                }
        unique_classes = np.unique(y)

        return EEGData(
            data=X,
            labels=y,
            subject_ids=subject_ids,
            feature_names=feature_cols,
            sample_ids=sample_ids,
            record_ids=record_ids,
            row_metadata=row_metadata,
            metadata={
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'n_subjects': len(np.unique(subject_ids)),
                'n_records': len(np.unique(record_ids)),
                'max_windows': max_windows,
                'include_subject_ids': normalized_subject_ids,
                'n_classes': len(unique_classes),
                'classes': unique_classes.tolist(),
                'source': 'Emotiv EPOC X',
                'dataset_type': 'windowed_eeg_pm'
            }
        )
