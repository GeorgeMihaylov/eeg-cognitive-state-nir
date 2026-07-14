import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Any, Optional, Union

class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_proba=None, average='weighted'):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
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
    def calculate_regression_metrics(y_true, y_pred):
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        if len(y_true) >= 2:
            try:
                metrics['pearson'], _ = pearsonr(y_true, y_pred)
            except:
                metrics['pearson'] = np.nan
            try:
                metrics['spearman'], _ = spearmanr(y_true, y_pred)
            except:
                metrics['spearman'] = np.nan
        else:
            metrics['pearson'] = metrics['spearman'] = np.nan
        metrics['n_samples'] = len(y_true)
        return metrics

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_proba=None, task_type='classification', average='weighted'):
        if task_type == 'classification':
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_proba, average)
        elif task_type == 'regression':
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
