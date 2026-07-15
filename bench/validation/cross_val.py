import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple, Any, Optional
from ..core.abstract_task import BaseTask, TaskSplit
from .metrics import MetricsCalculator


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
            splitter = GroupKFold(n_splits=n_splits)
            split_iterator = list(
                splitter.split(data.data, data.labels, groups)
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
            test_counts[test_idx] += 1
            fold_name = f"fold_{fold_index:02d}"
            splits[fold_name] = TaskSplit(
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
                metadata={
                    'split_type': 'group_kfold_subject',
                    'protocol': 'group_kfold_subject',
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
                },
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
