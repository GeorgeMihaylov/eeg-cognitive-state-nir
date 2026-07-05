import numpy as np
from typing import Dict, List, Tuple, Any
from ..core.task import BaseTask, TaskSplit
from .metrics import MetricsCalculator


class CrossValidator:
    def __init__(self, task: BaseTask):
        self.task = task
        self.subjects = task._get_subject_ids() if hasattr(task, '_get_subject_ids') else []

    def run_within_subject(self) -> TaskSplit:
        return self.task.get_split(subject_id=None)

    def run_loso(self) -> Dict[str, TaskSplit]:
        return self.task.get_all_splits()

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
