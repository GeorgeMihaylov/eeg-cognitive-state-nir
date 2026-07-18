import numpy as np
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
    ) -> Dict[str, Any]:
        normalized_task = str(task_type).strip().lower()
        if normalized_task in {'regression', 'regressor'}:
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        if normalized_task not in {'classification', 'classifier'}:
            raise ValueError(f'Unknown task_type {task_type!r}')
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # Backward-compatible keys used by the existing runner and summaries.
        metrics['f1_macro'] = metrics['macro_f1']
        metrics['f1_weighted'] = metrics['weighted_f1']
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        probability_array = (
            None if y_proba is None else np.asarray(y_proba)
        )
        qwk_labels = (
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
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
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
        metrics['n_classes'] = len(np.unique(y_true))

        return metrics

    @staticmethod
    def calculate_regression_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        truth = np.asarray(y_true, dtype=float).reshape(-1)
        prediction = np.asarray(y_pred, dtype=float).reshape(-1)
        if truth.shape != prediction.shape:
            raise ValueError(
                f'Regression arrays must have equal shape: '
                f'{truth.shape} != {prediction.shape}'
            )
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError('Regression metrics require finite values')

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
        return {
            'mae': float(mean_absolute_error(truth, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(truth, prediction))),
            'r2': r2,
            'pearson': pearson,
            'spearman': spearman,
            'n_samples': int(len(truth)),
            'task_type': 'regression',
        }

    @staticmethod
    def get_baseline_accuracy(n_classes: int) -> float:
        return 1.0 / n_classes

    @staticmethod
    def is_above_baseline(accuracy: float, n_classes: int) -> bool:
        return accuracy > 1.0 / n_classes
