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
            for column in ('source', 't_start', 't_center', 'window_id')
            if column in df.columns
        ]
        row_metadata = {
            column: df[column].values
            for column in row_metadata_columns
        }
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
                'n_classes': len(unique_classes),
                'classes': unique_classes.tolist(),
                'source': 'Emotiv EPOC X',
                'dataset_type': 'windowed_eeg_pm'
            }
        )
