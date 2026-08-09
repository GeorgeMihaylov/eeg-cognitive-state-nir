import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple, Any, Optional
from ..core.abstract_task import BaseTask, TaskSplit
from ..tasks.target_registry import TARGET_REGISTRY
from ..tasks.target_transforms import (
    build_fold_local_target_transform,
    build_target_transform_manifest,
)
from .metrics import MetricsCalculator


def deterministic_group_kfold_indices(
    groups: np.ndarray,
    *,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return deterministic indices for a one-dimensional group array."""
    group_values = np.asarray(groups)
    if group_values.ndim != 1:
        raise ValueError(
            f"GroupKFold groups must be one-dimensional, got {group_values.shape}"
        )
    if len(group_values) == 0:
        raise ValueError("GroupKFold groups cannot be empty")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"GroupKFold needs at least {n_splits} unique groups, "
            f"got {len(unique_groups)}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(group_values), dtype=np.uint8)
    return [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in splitter.split(
            placeholder, groups=group_values
        )
    ]


class CrossValidator:
    def __init__(self, task: BaseTask):
        self.task = task
        self.subjects = task._get_subject_ids() if hasattr(task, '_get_subject_ids') else []

    def run_within_subject(self) -> TaskSplit:
        return self.task.get_split(subject_id=None)

    def run_loso(self) -> Dict[str, TaskSplit]:
        return self.task.get_all_splits()

    def run_group_kfold(
            self,
            group_column: str,
            n_splits: int = 5,
            random_state: int = 42,
            precomputed_fold_column: Optional[str] = None,
    ) -> Dict[str, TaskSplit]:
        """Build all deterministic GroupKFold splits from row-level metadata."""
        data = self.task.data
        groups = data.get_row_values(group_column)
        unique_groups = np.unique(groups)
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"GroupKFold needs at least {n_splits} unique groups in "
                f"{group_column!r}, got {len(unique_groups)}"
            )

        if precomputed_fold_column is not None:
            fold_values = data.get_row_values(precomputed_fold_column).astype(int)
            expected_folds = set(range(1, n_splits + 1))
            actual_folds = set(np.unique(fold_values).tolist())
            if actual_folds != expected_folds:
                raise ValueError(
                    f"Precomputed folds in {precomputed_fold_column!r} must be "
                    f"{sorted(expected_folds)}, got {sorted(actual_folds)}"
                )
            group_fold_counts = {
                str(group): int(len(np.unique(fold_values[groups == group])))
                for group in unique_groups
            }
            split_groups = [
                group for group, count in group_fold_counts.items() if count != 1
            ]
            if split_groups:
                raise ValueError(
                    "Precomputed folds split grouping values across folds: "
                    f"{split_groups[:20]}"
                )
            split_iterator = [
                (
                    np.flatnonzero(fold_values != fold_index),
                    np.flatnonzero(fold_values == fold_index),
                )
                for fold_index in range(1, n_splits + 1)
            ]
        else:
            split_iterator = deterministic_group_kfold_indices(
                groups,
                n_splits=n_splits,
            )
        splits: Dict[str, TaskSplit] = {}
        test_counts = np.zeros(data.n_samples, dtype=np.int64)

        for fold_index, (train_idx, test_idx) in enumerate(
            split_iterator,
            start=1,
        ):
            train_groups = np.unique(groups[train_idx])
            test_groups = np.unique(groups[test_idx])
            group_overlap = np.intersect1d(train_groups, test_groups)
            if len(group_overlap):
                raise RuntimeError(
                    f"Group leakage detected in fold {fold_index}: "
                    f"{group_overlap.astype(str).tolist()}"
                )

            train_subjects = np.unique(data.subject_ids[train_idx])
            test_subjects = np.unique(data.subject_ids[test_idx])
            subject_overlap = np.intersect1d(train_subjects, test_subjects)
            if group_column == 'subject_id' and len(subject_overlap):
                raise RuntimeError(
                    f"Subject leakage detected in fold {fold_index}: "
                    f"{subject_overlap.astype(str).tolist()}"
                )

            train_records = np.unique(data.record_ids[train_idx])
            test_records = np.unique(data.record_ids[test_idx])
            y_train, y_test, target_transform_manifest = (
                self._outer_fold_targets(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fold_index=fold_index,
                )
            )
            test_counts[test_idx] += 1
            fold_name = f"fold_{fold_index:02d}"
            split_metadata = {
                'split_type': 'group_kfold_subject',
                'protocol': 'group_kfold_subject',
                'task_type': getattr(
                    self.task, 'task_type', 'classification'
                ),
                'fold': fold_index,
                'fold_name': fold_name,
                'n_splits': n_splits,
                'group_column': group_column,
                'shuffle': False,
                'random_state': random_state,
                'precomputed_fold_column': precomputed_fold_column,
                'observation_unit': data.metadata.get(
                    'observation_unit', 'window'
                ),
                'dataset_metadata': data.metadata,
                'target_names': data.metadata.get('target_cols'),
                'n_train_rows': len(train_idx),
                'n_test_rows': len(test_idx),
                'n_train_subjects': len(train_subjects),
                'n_test_subjects': len(test_subjects),
                'n_train_records': len(train_records),
                'n_test_records': len(test_records),
                'train_subject_ids': train_subjects.astype(str).tolist(),
                'test_subject_ids': test_subjects.astype(str).tolist(),
                'train_group_ids': train_groups.astype(str).tolist(),
                'test_group_ids': test_groups.astype(str).tolist(),
                'group_overlap': [],
                'subject_overlap': subject_overlap.astype(str).tolist(),
            }
            if target_transform_manifest is not None:
                split_metadata['target_transform'] = target_transform_manifest
                split_metadata['target_transform_hash'] = (
                    target_transform_manifest['transform_hash']
                )
            splits[fold_name] = TaskSplit(
                X_train=data.data[train_idx],
                y_train=y_train,
                X_test=data.data[test_idx],
                y_test=y_test,
                subject_train=data.subject_ids[train_idx],
                subject_test=data.subject_ids[test_idx],
                feature_names=data.feature_names,
                sample_id_train=data.sample_ids[train_idx],
                sample_id_test=data.sample_ids[test_idx],
                record_id_train=data.record_ids[train_idx],
                record_id_test=data.record_ids[test_idx],
                row_metadata_train={
                    key: np.asarray(values)[train_idx]
                    for key, values in data.row_metadata.items()
                },
                row_metadata_test={
                    key: np.asarray(values)[test_idx]
                    for key, values in data.row_metadata.items()
                },
                metadata=split_metadata,
            )

        if len(splits) != n_splits:
            raise RuntimeError(
                f"Expected {n_splits} GroupKFold splits, created {len(splits)}"
            )
        invalid_samples = np.flatnonzero(test_counts != 1)
        if len(invalid_samples):
            raise RuntimeError(
                "Every supervised sample must appear in test exactly once; "
                f"violations: {invalid_samples[:20].tolist()}"
            )
        return splits

    def _outer_fold_targets(
            self,
            *,
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            fold_index: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
        """Fit any derived target exactly once on the current outer train."""
        data = self.task.data
        target_id = data.metadata.get('target_id')
        if target_id is None:
            return data.labels[train_idx], data.labels[test_idx], None
        spec = TARGET_REGISTRY.get(str(target_id))
        if spec is None:
            return data.labels[train_idx], data.labels[test_idx], None
        if not spec.requires_fold_local_transform:
            return data.labels[train_idx], data.labels[test_idx], None

        transform = build_fold_local_target_transform(spec)
        transform.fit(data.labels[train_idx])
        if transform.actual_class_count != 3:
            raise ValueError(
                f"Fold {fold_index} target {spec.target_id!r} produced "
                f"{transform.actual_class_count} classes instead of 3; "
                "outer-train quantile boundaries collapsed"
            )
        y_train = transform.transform(data.labels[train_idx])
        y_test = transform.transform(data.labels[test_idx])
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError(
                f"Fold {fold_index} target transform produced missing classes"
            )
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        train_classes = np.unique(y_train)
        if not np.array_equal(train_classes, np.arange(3, dtype=np.int64)):
            raise ValueError(
                f"Fold {fold_index} target {spec.target_id!r} has outer-train "
                f"classes {train_classes.tolist()}, expected [0, 1, 2]"
            )
        manifest = build_target_transform_manifest(
            spec,
            transform,
            outer_fold=fold_index,
            outer_train_sample_ids=data.sample_ids[train_idx],
            outer_train_targets=data.labels[train_idx],
        )
        return y_train, y_test, manifest

    @staticmethod
    def _limited_indices(
            data,
            indices: np.ndarray,
            limit: Optional[int],
    ) -> np.ndarray:
        """Select a deterministic subject-balanced prefix for technical runs."""
        indices = np.asarray(indices, dtype=np.int64)
        if limit is None or len(indices) <= int(limit):
            return indices
        limit = int(limit)
        if limit <= 0:
            raise ValueError("Cross-source window limits must be positive")
        source = data.get_row_values('source').astype(str)
        time_values = None
        for column in ('t_start', 't_center'):
            try:
                time_values = data.get_row_values(column).astype(float)
                break
            except ValueError:
                continue
        if time_values is None:
            time_values = np.arange(data.n_samples, dtype=float)
        frame = pd.DataFrame({
            'index': indices,
            'subject_id': data.subject_ids[indices].astype(str),
            'source': source[indices],
            'record_id': data.record_ids[indices].astype(str),
            'time': time_values[indices],
            'sample_id': data.sample_ids[indices],
        }).sort_values(
            ['subject_id', 'source', 'record_id', 'time', 'sample_id'],
            kind='mergesort',
        )
        frame['subject_rank'] = frame.groupby(
            'subject_id', sort=True
        ).cumcount()
        selected = frame.sort_values(
            ['subject_rank', 'subject_id', 'source', 'record_id', 'time'],
            kind='mergesort',
        ).head(limit)['index'].to_numpy(dtype=np.int64, copy=True)
        selected.sort()
        return selected

    def run_cross_source_holdout(
            self,
            *,
            train_source: str,
            test_source: str,
            subject_mode: str = 'source_exclusive',
            remove_logical_duplicates: bool = True,
            minimum_train_subjects: int = 5,
            minimum_test_subjects: int = 3,
            minimum_train_classes: int = 5,
            minimum_test_classes: int = 2,
            minimum_predictions_per_test_subject: int = 20,
            max_train_windows: Optional[int] = None,
            max_test_windows: Optional[int] = None,
            random_state: int = 42,
    ) -> TaskSplit:
        """Build one directional source holdout without fitting any model."""
        data = self.task.data
        train_source = str(train_source)
        test_source = str(test_source)
        if not train_source or not test_source or train_source == test_source:
            raise ValueError("train_source and test_source must be distinct")
        normalized_mode = str(subject_mode).strip().lower().replace('-', '_')
        aliases = {
            'source_exclusive': 'source_exclusive',
            'exclusive': 'source_exclusive',
            'shared_subject': 'shared_subject',
            'shared_subjects': 'shared_subject',
            'shared': 'shared_subject',
        }
        if normalized_mode not in aliases:
            raise ValueError(
                "subject_mode must be 'source_exclusive' or 'shared_subject'"
            )
        normalized_mode = aliases[normalized_mode]

        sources = data.get_row_values('source').astype(str)
        available_sources = set(np.unique(sources).tolist())
        missing_sources = sorted(
            {train_source, test_source} - available_sources
        )
        if missing_sources:
            raise ValueError(
                f"Cross-source split sources are unavailable: {missing_sources}"
            )
        logical_ids = data.get_row_values('record_group_id').astype(str)
        source_subjects = {
            source: set(data.subject_ids[sources == source].astype(str))
            for source in (train_source, test_source)
        }
        shared_subjects = source_subjects[train_source] & source_subjects[test_source]
        exclusive_subjects = {
            source: source_subjects[source] - shared_subjects
            for source in (train_source, test_source)
        }
        logical_sources: Dict[str, set[str]] = {}
        for logical_id, source in zip(logical_ids, sources):
            logical_sources.setdefault(str(logical_id), set()).add(str(source))
        duplicated_logical_ids = {
            logical_id
            for logical_id, group_sources in logical_sources.items()
            if train_source in group_sources and test_source in group_sources
        }

        if normalized_mode == 'source_exclusive':
            train_subjects = exclusive_subjects[train_source]
            test_subjects = exclusive_subjects[test_source]
        else:
            train_subjects = shared_subjects
            test_subjects = shared_subjects
        train_mask = (
            (sources == train_source)
            & np.isin(data.subject_ids.astype(str), sorted(train_subjects))
        )
        test_mask = (
            (sources == test_source)
            & np.isin(data.subject_ids.astype(str), sorted(test_subjects))
        )
        excluded_logical_ids: list[str] = []
        duplicate_mask = np.zeros(len(logical_ids), dtype=bool)
        if remove_logical_duplicates:
            excluded_logical_ids = sorted(duplicated_logical_ids)
            duplicate_mask = np.isin(logical_ids, excluded_logical_ids)
            train_mask &= ~duplicate_mask
            test_mask &= ~duplicate_mask
        residual_source_subjects = {
            source: set(
                data.subject_ids[
                    (sources == source) & ~duplicate_mask
                ].astype(str)
            )
            for source in (train_source, test_source)
        }
        eligible_shared_subjects = (
            shared_subjects
            & residual_source_subjects[train_source]
            & residual_source_subjects[test_source]
        )
        if normalized_mode == 'shared_subject':
            eligible_mask = np.isin(
                data.subject_ids.astype(str), sorted(eligible_shared_subjects)
            )
            train_mask &= eligible_mask
            test_mask &= eligible_mask

        train_idx = self._limited_indices(
            data, np.flatnonzero(train_mask), max_train_windows
        )
        test_idx = self._limited_indices(
            data, np.flatnonzero(test_mask), max_test_windows
        )
        train_subject_values = np.unique(data.subject_ids[train_idx]).astype(str)
        test_subject_values = np.unique(data.subject_ids[test_idx]).astype(str)
        train_logical = np.unique(logical_ids[train_idx]).astype(str)
        test_logical = np.unique(logical_ids[test_idx]).astype(str)
        train_records = np.unique(data.record_ids[train_idx]).astype(str)
        test_records = np.unique(data.record_ids[test_idx]).astype(str)
        train_samples = np.unique(data.sample_ids[train_idx])
        test_samples = np.unique(data.sample_ids[test_idx])
        subject_overlap = np.intersect1d(
            train_subject_values, test_subject_values
        ).astype(str).tolist()
        logical_overlap = np.intersect1d(
            train_logical, test_logical
        ).astype(str).tolist()
        record_overlap = np.intersect1d(
            train_records, test_records
        ).astype(str).tolist()
        sample_overlap = np.intersect1d(
            train_samples, test_samples
        ).tolist()

        invalid_reasons: list[str] = []
        checks = (
            ('train subjects', len(train_subject_values), int(minimum_train_subjects)),
            ('test subjects', len(test_subject_values), int(minimum_test_subjects)),
            ('train classes', len(np.unique(data.labels[train_idx])), int(minimum_train_classes)),
            ('test classes', len(np.unique(data.labels[test_idx])), int(minimum_test_classes)),
        )
        for label, actual, minimum in checks:
            if actual < minimum:
                invalid_reasons.append(
                    f"{label}={actual} is below configured minimum {minimum}"
                )
        if len(test_idx):
            _, per_subject_counts = np.unique(
                data.subject_ids[test_idx].astype(str), return_counts=True
            )
            minimum_test_predictions = int(per_subject_counts.min())
        else:
            minimum_test_predictions = 0
        if minimum_test_predictions < int(minimum_predictions_per_test_subject):
            invalid_reasons.append(
                "minimum test predictions per subject="
                f"{minimum_test_predictions} is below configured minimum "
                f"{int(minimum_predictions_per_test_subject)}"
            )
        if normalized_mode == 'source_exclusive' and subject_overlap:
            invalid_reasons.append(
                f"source-exclusive subject overlap detected: {subject_overlap}"
            )
        if logical_overlap:
            invalid_reasons.append(
                f"logical recording overlap detected: {logical_overlap[:20]}"
            )
        if record_overlap:
            invalid_reasons.append(
                f"source record overlap detected: {record_overlap[:20]}"
            )
        if sample_overlap:
            invalid_reasons.append(
                f"sample overlap detected: {sample_overlap[:20]}"
            )

        split_name = (
            f"{train_source}_to_{test_source}_{normalized_mode}"
        )
        metadata = {
            'split_type': 'cross_source_holdout',
            'protocol': 'cross_source_holdout',
            'fold': split_name,
            'fold_name': split_name,
            'status': 'valid' if not invalid_reasons else 'invalid',
            'invalid_reasons': invalid_reasons,
            'train_source': train_source,
            'test_source': test_source,
            'subject_mode': normalized_mode,
            'remove_logical_duplicates': bool(remove_logical_duplicates),
            'random_state': int(random_state),
            'max_train_windows': max_train_windows,
            'max_test_windows': max_test_windows,
            'observation_unit': data.metadata.get('observation_unit', 'window'),
            'dataset_metadata': data.metadata,
            'n_train_rows': int(len(train_idx)),
            'n_test_rows': int(len(test_idx)),
            'n_train_subjects': int(len(train_subject_values)),
            'n_test_subjects': int(len(test_subject_values)),
            'n_train_records': int(len(train_records)),
            'n_test_records': int(len(test_records)),
            'n_train_logical_recordings': int(len(train_logical)),
            'n_test_logical_recordings': int(len(test_logical)),
            'train_subject_ids': train_subject_values.tolist(),
            'test_subject_ids': test_subject_values.tolist(),
            'train_record_ids': train_records.tolist(),
            'test_record_ids': test_records.tolist(),
            'train_logical_record_ids': train_logical.tolist(),
            'test_logical_record_ids': test_logical.tolist(),
            'shared_subject_ids': sorted(shared_subjects),
            'eligible_shared_subject_ids': sorted(eligible_shared_subjects),
            'excluded_subjects': {
                'shared': sorted(shared_subjects) if normalized_mode == 'source_exclusive' else [],
                'train_source_exclusive': sorted(exclusive_subjects[train_source]),
                'test_source_exclusive': sorted(exclusive_subjects[test_source]),
                'shared_without_residual_data_in_both_sources': sorted(
                    shared_subjects - eligible_shared_subjects
                ) if normalized_mode == 'shared_subject' else [],
            },
            'excluded_logical_record_ids': excluded_logical_ids,
            'group_overlap': logical_overlap,
            'logical_record_overlap': logical_overlap,
            'raw_interval_overlap': logical_overlap,
            'record_overlap': record_overlap,
            'subject_overlap': subject_overlap,
            'sample_overlap': sample_overlap,
            'allow_subject_overlap': normalized_mode == 'shared_subject',
            'minimum_test_predictions_per_subject_actual': minimum_test_predictions,
            'thresholds': {
                'minimum_train_subjects': int(minimum_train_subjects),
                'minimum_test_subjects': int(minimum_test_subjects),
                'minimum_train_classes': int(minimum_train_classes),
                'minimum_test_classes': int(minimum_test_classes),
                'minimum_predictions_per_test_subject': int(
                    minimum_predictions_per_test_subject
                ),
            },
            'source_distribution': {
                'train': {train_source: int(len(train_idx))},
                'test': {test_source: int(len(test_idx))},
            },
        }
        return TaskSplit(
            X_train=data.data[train_idx],
            y_train=data.labels[train_idx],
            X_test=data.data[test_idx],
            y_test=data.labels[test_idx],
            subject_train=data.subject_ids[train_idx],
            subject_test=data.subject_ids[test_idx],
            feature_names=data.feature_names,
            sample_id_train=data.sample_ids[train_idx],
            sample_id_test=data.sample_ids[test_idx],
            record_id_train=data.record_ids[train_idx],
            record_id_test=data.record_ids[test_idx],
            row_metadata_train={
                key: np.asarray(values)[train_idx]
                for key, values in data.row_metadata.items()
            },
            row_metadata_test={
                key: np.asarray(values)[test_idx]
                for key, values in data.row_metadata.items()
            },
            metadata=metadata,
        )

    def evaluate_model(self, model, split: TaskSplit) -> Dict[str, Any]:
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(split.X_test)
            except Exception:
                pass

        metrics = MetricsCalculator.calculate_all_metrics(
            split.y_test, y_pred, y_proba
        )

        return {
            'metrics': metrics,
            'feature_importance': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
        }

    def evaluate_loso(self, model_class, model_config: Dict[str, Any]) -> Dict[str, Any]:
        loso_splits = self.run_loso()
        results = {}

        for subject_id, split in loso_splits.items():
            model = model_class(model_config)
            results[str(subject_id)] = self.evaluate_model(model, split)
        aggregated = self._aggregate_loso_results(results)

        return {
            'per_subject': results,
            'aggregated': aggregated
        }

    def _aggregate_loso_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        accuracies = []
        f1_scores = []
        kappas = []

        for subject_id, result in results.items():
            metrics = result['metrics']
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_weighted'])
            kappas.append(metrics['kappa'])

        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'kappa_mean': np.mean(kappas),
            'kappa_std': np.std(kappas),
            'n_subjects': len(results)
        }
