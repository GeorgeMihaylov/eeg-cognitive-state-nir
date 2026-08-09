import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_eeg_data_loader import BaseEEGDataset
from .base_eeg_data_loader import feature_list_sha256
from ..core.abstract_dataset import EEGData
from ..tasks.target_registry import resolve_target_spec


class EmotivDataset(BaseEEGDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        target_config = config
        if not any(
            name in config for name in ('target_id', 'target_col', 'target_cols')
        ):
            target_config = {
                **config,
                'target_col': 'target_main',
                '_legacy_implicit_target_main_classification': True,
            }
        self.target_spec = resolve_target_spec(target_config)
        configured_targets = config.get('target_cols')
        if configured_targets is not None:
            if (
                not isinstance(configured_targets, list)
                or not configured_targets
                or not all(
                    isinstance(column, str) and column.strip()
                    for column in configured_targets
                )
            ):
                raise ValueError('target_cols must be a non-empty list of strings')
            if len(set(configured_targets)) != len(configured_targets):
                raise ValueError('target_cols must contain unique column names')
        self.target_cols: Optional[List[str]] = (
            list(self.target_spec.processed_columns)
            if self.target_spec.output_dim > 1
            else None
        )
        self.target_col = self.target_spec.processed_columns[0]
        self.target_col_explicit = True
        self.subject_col = config.get('subject_col', 'subject_id')

    def load(self) -> EEGData:
        df = self._load_dataframe()
        include_sources = self.config.get('include_sources')
        normalized_sources = None
        if include_sources is not None:
            if isinstance(include_sources, (str, bytes)):
                include_sources = [include_sources]
            normalized_sources = sorted({
                str(value) for value in include_sources
            })
            if not normalized_sources:
                raise ValueError('include_sources must not be empty')
            if 'source' not in df.columns:
                raise ValueError(
                    "Source column is required when include_sources is configured"
                )
            available_sources = set(df['source'].astype(str))
            missing_sources = sorted(
                set(normalized_sources) - available_sources
            )
            if missing_sources:
                raise ValueError(
                    'include_sources contains unavailable sources: '
                    f'{missing_sources}'
                )
            df = df.loc[
                df['source'].astype(str).isin(normalized_sources)
            ].copy()
            if df.empty:
                raise ValueError('include_sources selected no dataset rows')
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
        logical_map_path = self.config.get('logical_recording_map_path')
        duplicate_logical_record_ids: List[str] = []
        logical_record_ids = record_ids.copy()
        if logical_map_path is not None:
            logical_path = Path(logical_map_path)
            if not logical_path.is_file():
                raise FileNotFoundError(
                    f"Logical recording map not found: {logical_path}"
                )
            logical_map = pd.read_parquet(logical_path)
            required_columns = {'record_group_id', 'source_record_ids'}
            missing_columns = sorted(required_columns - set(logical_map.columns))
            if missing_columns:
                raise ValueError(
                    "Logical recording map is missing required columns: "
                    f"{missing_columns}"
                )
            record_to_logical: Dict[str, str] = {}
            for row in logical_map.itertuples(index=False):
                logical_id = str(row.record_group_id)
                for source_record_id in row.source_record_ids:
                    source_record_id = str(source_record_id)
                    previous = record_to_logical.setdefault(
                        source_record_id, logical_id
                    )
                    if previous != logical_id:
                        raise ValueError(
                            f"Source record {source_record_id!r} maps to multiple "
                            "logical recordings"
                        )
            logical_record_ids = np.asarray([
                record_to_logical.get(str(record_id), None)
                for record_id in record_ids
            ], dtype=object)
            if 'present_in_both_sources' in logical_map.columns:
                duplicate_logical_record_ids = sorted(
                    logical_map.loc[
                        logical_map['present_in_both_sources'].astype(bool),
                        'record_group_id',
                    ].astype(str).unique().tolist()
                )
        row_metadata_columns = [
            column
            for column in (
                'source', 't_start', 't_center', 'window_id',
                'record_group_id', 'datetime_from_name', 'day', 'part',
            )
            if column in df.columns
        ]
        row_metadata = {
            column: df[column].values
            for column in row_metadata_columns
        }
        row_metadata['record_group_id'] = logical_record_ids
        feature_cols = self._select_features(df)
        forbidden_features = [
            column
            for column in feature_cols
            if column.startswith('target_') or column.startswith('PM.')
        ]
        if forbidden_features:
            raise ValueError(
                'Target/Performance Metric columns cannot be input features: '
                f'{forbidden_features[:20]}'
            )
        if self.target_cols is not None:
            missing_targets = [
                column for column in self.target_cols if column not in df.columns
            ]
            if missing_targets:
                raise ValueError(
                    f'Configured target columns are absent from the dataset: '
                    f'{missing_targets}'
                )
        elif self.target_col not in df.columns:
            raise ValueError(
                f"Configured target column {self.target_col!r} "
                "is absent from the dataset"
            )
        n_samples_before_target_filter = len(df)
        X = (
            df[feature_cols]
            .apply(pd.to_numeric, errors='coerce')
            .to_numpy(dtype=np.float32)
        )
        if self.target_cols is None:
            y = pd.to_numeric(df[self.target_col], errors='coerce').to_numpy()
            if (
                self.target_spec.is_classification
                and not self.target_spec.requires_fold_local_transform
            ):
                if self.target_spec.registry_status == 'deprecated_ad_hoc_legacy':
                    y = self._discretize_target(y)
                finite = np.isfinite(y)
                if not np.allclose(y[finite], np.round(y[finite])):
                    raise ValueError(
                        f"Classification target {self.target_col!r} contains "
                        "non-integer values"
                    )
            else:
                y = np.asarray(y, dtype=np.float32)
            target_valid_mask = np.isfinite(y)
        else:
            y = (
                df[self.target_cols]
                .apply(pd.to_numeric, errors='coerce')
                .to_numpy(dtype=np.float32)
            )
            target_valid_mask = np.isfinite(y).all(axis=1)
        if self.subject_col in df.columns:
            subject_ids = df[self.subject_col].astype(str).values
        else:
            subject_ids = np.array(['unknown'] * len(df))
        feature_valid_mask = np.isfinite(X).all(axis=1)
        valid_mask = target_valid_mask & feature_valid_mask
        n_samples_after_target_filter = int(target_valid_mask.sum())
        dropped_target_rows = int((~target_valid_mask).sum())
        dropped_feature_rows = int((target_valid_mask & ~feature_valid_mask).sum())
        X = X[valid_mask]
        y = y[valid_mask]
        if (
            self.target_spec.is_classification
            and not self.target_spec.requires_fold_local_transform
        ):
            y = np.asarray(np.round(y), dtype=np.int64)
        else:
            y = np.asarray(y, dtype=np.float32)
        subject_ids = subject_ids[valid_mask]
        sample_ids = sample_ids[valid_mask]
        record_ids = record_ids[valid_mask]
        row_metadata = {
            column: values[valid_mask]
            for column, values in row_metadata.items()
        }
        if logical_map_path is not None:
            unmapped = pd.isna(row_metadata['record_group_id'])
            if np.any(unmapped):
                missing_records = sorted(
                    np.unique(record_ids[valid_mask][unmapped]).astype(str).tolist()
                )
                raise ValueError(
                    "Supervised source records are absent from the logical "
                    f"recording map: {missing_records[:20]}"
                )
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
        unique_classes = (
            np.unique(y)
            if (
                self.target_spec.is_classification
                and not self.target_spec.requires_fold_local_transform
            )
            else np.asarray([], dtype=float)
        )

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
                'feature_set': self.feature_set,
                'feature_list_sha256': feature_list_sha256(feature_cols),
                'target_col': (
                    self.target_col if self.target_cols is None else None
                ),
                'target_id': self.target_spec.target_id,
                'target_type': self.target_spec.target_type,
                'target_registry_status': self.target_spec.registry_status,
                'target_output_names': list(self.target_spec.output_names),
                'target_cols': (
                    None if self.target_cols is None else list(self.target_cols)
                ),
                'n_outputs': (
                    1 if self.target_cols is None else len(self.target_cols)
                ),
                'task_type': (
                    self.target_spec.task_type
                ),
                'n_samples_before_target_filter': n_samples_before_target_filter,
                'n_samples_after_target_filter': n_samples_after_target_filter,
                'n_samples_after_complete_case_filter': len(X),
                'dropped_target_rows': dropped_target_rows,
                'dropped_feature_rows': dropped_feature_rows,
                'discretize': bool(self.config.get('discretize', True)),
                'n_subjects': len(np.unique(subject_ids)),
                'n_records': len(np.unique(record_ids)),
                'max_windows': max_windows,
                'include_subject_ids': normalized_subject_ids,
                'include_sources': normalized_sources,
                'logical_recording_map_path': (
                    None if logical_map_path is None else str(logical_map_path)
                ),
                'duplicated_logical_record_ids': duplicate_logical_record_ids,
                'n_logical_recordings': int(
                    len(np.unique(row_metadata['record_group_id']))
                ),
                'n_classes': len(unique_classes),
                'classes': unique_classes.tolist(),
                'source': 'Emotiv EPOC X',
                'dataset_type': 'windowed_eeg_pm'
            }
        )
