import numpy as np
import pandas as pd
import re
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
    precision_recall_fscore_support,
)
from typing import Dict, Any, List, Optional


class MetricsCalculator:
    @staticmethod
    def calculate_class_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Return deterministic one-vs-rest precision/recall/F1 per class."""

        truth = np.asarray(y_true).reshape(-1)
        prediction = np.asarray(y_pred).reshape(-1)
        if truth.shape != prediction.shape:
            raise ValueError(
                f'Classification arrays must have equal shape: '
                f'{truth.shape} != {prediction.shape}'
            )
        resolved_labels = (
            np.unique(np.concatenate([truth, prediction]))
            if labels is None
            else np.asarray(labels).reshape(-1)
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            prediction,
            labels=resolved_labels,
            zero_division=0,
        )
        return [
            {
                'class_id': int(class_id),
                'precision': float(class_precision),
                'recall': float(class_recall),
                'f1': float(class_f1),
                'support': int(class_support),
            }
            for class_id, class_precision, class_recall, class_f1, class_support
            in zip(resolved_labels, precision, recall, f1, support)
        ]

    @staticmethod
    def calculate_all_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_proba: Optional[np.ndarray] = None,
            average: str = 'weighted',
            task_type: str = 'classification',
            expected_rank: Optional[np.ndarray] = None,
            target_names: Optional[List[str]] = None,
            labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        normalized_task = str(task_type).strip().lower()
        if normalized_task in {'regression', 'regressor'}:
            return MetricsCalculator.calculate_regression_metrics(
                y_true,
                y_pred,
                target_names=target_names,
            )
        if normalized_task not in {'classification', 'classifier'}:
            raise ValueError(f'Unknown task_type {task_type!r}')
        resolved_labels = (
            None if labels is None else np.asarray(labels).reshape(-1)
        )
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(
            y_true, y_pred, labels=resolved_labels,
            average=average, zero_division=0,
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, labels=resolved_labels,
            average=average, zero_division=0,
        )
        metrics['macro_f1'] = f1_score(
            y_true, y_pred, labels=resolved_labels,
            average='macro', zero_division=0,
        )
        metrics['weighted_f1'] = f1_score(
            y_true, y_pred, labels=resolved_labels,
            average='weighted', zero_division=0,
        )
        # Backward-compatible keys used by the existing runner and summaries.
        metrics['f1_macro'] = metrics['macro_f1']
        metrics['f1_weighted'] = metrics['weighted_f1']
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        probability_array = (
            None if y_proba is None else np.asarray(y_proba)
        )
        qwk_labels = (
            resolved_labels.tolist()
            if resolved_labels is not None
            else
            list(range(probability_array.shape[1]))
            if probability_array is not None and probability_array.ndim == 2
            else None
        )
        metrics['quadratic_weighted_kappa'] = cohen_kappa_score(
            y_true,
            y_pred,
            labels=qwk_labels,
            weights='quadratic',
        )
        ordinal_distance = np.abs(
            np.asarray(y_pred, dtype=np.float64)
            - np.asarray(y_true, dtype=np.float64)
        )
        metrics['ordinal_mae'] = float(np.mean(ordinal_distance))
        metrics['adjacent_accuracy'] = float(np.mean(ordinal_distance <= 1.0))
        metrics['severe_error_rate'] = float(np.mean(ordinal_distance >= 2.0))
        metrics['confusion_matrix'] = confusion_matrix(
            y_true, y_pred, labels=resolved_labels
        ).tolist()
        if y_proba is not None:
            try:
                n_classes = probability_array.shape[1]
                if n_classes == 2:
                    metrics['auc'] = roc_auc_score(
                        y_true, probability_array[:, 1]
                    )
                else:
                    metrics['auc'] = roc_auc_score(
                        y_true,
                        probability_array,
                        multi_class='ovr',
                        average='weighted',
                    )
            except Exception:
                metrics['auc'] = np.nan
        if expected_rank is not None:
            truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
            ranks = np.asarray(expected_rank, dtype=np.float64).reshape(-1)
            if ranks.shape != truth.shape:
                raise ValueError(
                    'expected_rank must have the same shape as y_true: '
                    f'{ranks.shape} != {truth.shape}'
                )
            if not np.isfinite(ranks).all():
                raise ValueError('expected_rank must contain only finite values')
            metrics['expected_rank_mae'] = float(
                mean_absolute_error(truth, ranks)
            )
            metrics['expected_rank_spearman'] = (
                float(spearmanr(truth, ranks).statistic)
                if len(truth) >= 2
                and np.ptp(truth) > 0
                and np.ptp(ranks) > 0
                else np.nan
            )
        metrics['n_samples'] = len(y_true)
        metrics['n_classes'] = (
            len(np.unique(y_true))
            if resolved_labels is None
            else len(resolved_labels)
        )

        return metrics

    @staticmethod
    def calculate_regression_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        if truth.shape != prediction.shape:
            raise ValueError(
                f'Regression arrays must have equal shape: '
                f'{truth.shape} != {prediction.shape}'
            )
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError('Regression metrics require finite values')

        if truth.ndim == 1:
            return MetricsCalculator._single_regression_metrics(
                truth,
                prediction,
            )
        if truth.ndim != 2:
            raise ValueError(
                'Regression arrays must be one- or two-dimensional, '
                f'got {truth.shape}'
            )
        n_outputs = truth.shape[1]
        resolved_names = (
            [f'target_{index}' for index in range(n_outputs)]
            if target_names is None
            else list(target_names)
        )
        if len(resolved_names) != n_outputs:
            raise ValueError(
                f'target_names must contain {n_outputs} values, '
                f'got {len(resolved_names)}'
            )
        normalized_names = [
            MetricsCalculator.normalize_target_name(name)
            for name in resolved_names
        ]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(
                f'Normalized target names must be unique: {normalized_names}'
            )

        result: Dict[str, Any] = {
            'n_samples': int(len(truth)),
            'n_outputs': int(n_outputs),
            'target_names': resolved_names,
            'task_type': 'regression',
        }
        per_target = []
        values_by_metric = {
            name: []
            for name in (
                'mae', 'rmse', 'r2', 'pearson', 'spearman',
                'mean_error', 'abs_bias',
            )
        }
        for target_index, (target_name, normalized_name) in enumerate(
            zip(resolved_names, normalized_names)
        ):
            target_metrics = MetricsCalculator._single_regression_metrics(
                truth[:, target_index],
                prediction[:, target_index],
            )
            per_target.append({
                'target_name': target_name,
                'target_key': normalized_name,
                **{
                    name: target_metrics[name]
                    for name in values_by_metric
                },
                'n_samples': int(len(truth)),
            })
            for metric_name in values_by_metric:
                value = target_metrics[metric_name]
                values_by_metric[metric_name].append(value)
                result[f'{metric_name}_{normalized_name}'] = value
                result[f'window_{metric_name}_{normalized_name}'] = value

        result['per_target'] = per_target
        for metric_name, values in values_by_metric.items():
            metric_values = np.asarray(values, dtype=float)
            finite = metric_values[np.isfinite(metric_values)]
            macro = (
                float(np.mean(finite))
                if len(finite)
                else np.nan
            )
            result[f'{metric_name}_macro'] = macro
            result[f'window_{metric_name}_macro'] = macro
            result[f'{metric_name}_valid_targets'] = int(len(finite))
        return result

    @staticmethod
    def _single_regression_metrics(
            truth: np.ndarray,
            prediction: np.ndarray,
    ) -> Dict[str, Any]:
        truth = np.asarray(truth, dtype=float).reshape(-1)
        prediction = np.asarray(prediction, dtype=float).reshape(-1)

        has_truth_variation = len(truth) >= 2 and np.ptp(truth) > 0
        has_prediction_variation = len(prediction) >= 2 and np.ptp(prediction) > 0
        r2 = (
            float(r2_score(truth, prediction))
            if has_truth_variation
            else np.nan
        )
        pearson = (
            float(pearsonr(truth, prediction).statistic)
            if has_truth_variation and has_prediction_variation
            else np.nan
        )
        spearman = (
            float(spearmanr(truth, prediction).statistic)
            if has_truth_variation and has_prediction_variation
            else np.nan
        )
        mean_error = float(np.mean(prediction - truth))
        return {
            'mae': float(mean_absolute_error(truth, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(truth, prediction))),
            'r2': r2,
            'pearson': pearson,
            'spearman': spearman,
            'mean_error': mean_error,
            'abs_bias': abs(mean_error),
            'n_samples': int(len(truth)),
            'task_type': 'regression',
        }

    @staticmethod
    def normalize_target_name(target_name: str) -> str:
        normalized = str(target_name).strip()
        if normalized.lower().startswith('target_'):
            normalized = normalized[7:]
        normalized = re.sub(r'[^0-9A-Za-z]+', '_', normalized)
        return normalized.strip('_').lower()

    @staticmethod
    def calculate_subject_regression_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            subject_ids: np.ndarray,
            target_names: List[str],
            fold: Any,
    ) -> tuple[Dict[str, Any], pd.DataFrame]:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        subjects = np.asarray(subject_ids).astype(str)
        if truth.ndim != 2 or truth.shape != prediction.shape:
            raise ValueError('Subject-level regression requires equal 2D arrays')
        if len(subjects) != len(truth):
            raise ValueError('subject_ids must match regression rows')
        if len(target_names) != truth.shape[1]:
            raise ValueError('target_names must match regression outputs')

        rows = []
        subject_truth = []
        subject_prediction = []
        for subject_id in sorted(np.unique(subjects).tolist()):
            mask = subjects == subject_id
            truth_mean = np.mean(truth[mask], axis=0)
            prediction_mean = np.mean(prediction[mask], axis=0)
            subject_truth.append(truth_mean)
            subject_prediction.append(prediction_mean)
            for target_index, target_name in enumerate(target_names):
                rows.append({
                    'fold': fold,
                    'subject_id': subject_id,
                    'target_name': target_name,
                    'y_true_mean': float(truth_mean[target_index]),
                    'y_pred_mean': float(prediction_mean[target_index]),
                    'n_windows': int(mask.sum()),
                })
        subject_metrics = MetricsCalculator.calculate_regression_metrics(
            np.asarray(subject_truth),
            np.asarray(subject_prediction),
            target_names=target_names,
        )
        prefixed = {
            f'subject_{key}': value
            for key, value in subject_metrics.items()
            if key not in {'task_type', 'target_names', 'per_target'}
        }
        prefixed['subject_per_target'] = subject_metrics['per_target']
        prefixed['subject_n_subjects'] = int(len(subject_truth))
        return prefixed, pd.DataFrame(rows)

    @staticmethod
    def get_baseline_accuracy(n_classes: int) -> float:
        return 1.0 / n_classes

    @staticmethod
    def is_above_baseline(accuracy: float, n_classes: int) -> bool:
        return accuracy > 1.0 / n_classes
