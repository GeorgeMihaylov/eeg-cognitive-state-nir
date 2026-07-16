import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, roc_auc_score
)
from typing import Dict, Any, List, Optional


class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_proba: Optional[np.ndarray] = None,
            average: str = 'weighted'
    ) -> Dict[str, Any]:
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
                n_classes = y_proba.shape[1]
                if n_classes == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                metrics['auc'] = np.nan
        metrics['n_samples'] = len(y_true)
        metrics['n_classes'] = len(np.unique(y_true))

        return metrics

    @staticmethod
    def get_baseline_accuracy(n_classes: int) -> float:
        return 1.0 / n_classes

    @staticmethod
    def is_above_baseline(accuracy: float, n_classes: int) -> bool:
        return accuracy > 1.0 / n_classes
