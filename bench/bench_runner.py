import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .core.abstract_dataset import BaseDataset, EEGData
from .core.abstract_task import BaseTask, TaskSplit
from .datasets.datasets_registry import get_dataset
from .tasks.tasks_registry import get_task
from .validation.cross_val import CrossValidator
from .validation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './benchmark_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.models = {}

        self._setup_models()

    def _setup_models(self):
        model_configs = self.config.get('models', {})

        for model_name, model_config in model_configs.items():
            try:
                model = self._create_model(model_config)
                self.models[model_name] = {
                    'model': model,
                    'config': model_config
                }
                logger.info(f"Model '{model_name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize model '{model_name}': {e}")

    def _create_model(self, model_config: Dict[str, Any]) -> BaseEstimator:
        model_type = model_config.get('type')
        params = model_config.get('params', {})

        if model_type == 'svm':
            from sklearn.svm import SVC
            return SVC(**params)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif model_type == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_dataset(self, dataset_name: str) -> EEGData:
        try:
            dataset_config = self.config['datasets'][dataset_name]
            dataset_config['data_path'] = Path(dataset_config['data_path'])

            dataset = get_dataset(dataset_name, dataset_config)
            data = dataset.load()

            logger.info(f"Loaded dataset '{dataset_name}': "
                        f"{data.n_samples} samples, {data.n_features} features, "
                        f"{data.n_subjects} subjects, {data.n_classes} classes")

            return data
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise

    def run_for_dataset(self, dataset_name: str) -> Dict[str, Any]:
        logger.info(f"Running benchmark for dataset: {dataset_name}")
        data = self.load_dataset(dataset_name)
        results = {
            'dataset': dataset_name,
            'timestamp': self.timestamp,
            'models': {}
        }
        task_names = self.config.get('tasks', ['cognitive_load_3class'])
        for task_name in task_names:
            logger.info(f"  Running task: {task_name}")
            task_config = self.config.get('task_config', {})
            task = get_task(task_name, data, task_config)
            cv = CrossValidator(task)
            subject_ids = np.unique(data.subject_ids)
            if self.config.get('run_within_subject', True):
                logger.info("    Running within-subject evaluation")
                split = cv.run_within_subject()
                for model_name, model_info in self.models.items():
                    model = model_info['model']
                    try:
                        eval_result = self._evaluate_split(model, split, model_name)
                        if task_name not in results['models']:
                            results['models'][task_name] = {}
                        if model_name not in results['models'][task_name]:
                            results['models'][task_name][model_name] = {}
                        results['models'][task_name][model_name]['within_subject'] = eval_result
                    except Exception as e:
                        logger.error(f"      Model '{model_name}' failed: {e}")
            if self.config.get('run_loso', True) and len(subject_ids) > 1:
                logger.info("    Running LOSO evaluation")
                loso_splits = cv.run_loso()

                for model_name, model_info in self.models.items():
                    model = model_info['model']
                    try:
                        loso_results = self._evaluate_loso(model, loso_splits, model_name)
                        if task_name not in results['models']:
                            results['models'][task_name] = {}
                        if model_name not in results['models'][task_name]:
                            results['models'][task_name][model_name] = {}
                        results['models'][task_name][model_name]['loso'] = loso_results
                    except Exception as e:
                        logger.error(f"      Model '{model_name}' LOSO failed: {e}")

        return results

    def _evaluate_split(self, model: BaseEstimator, split: TaskSplit, model_name: str) -> Dict[str, Any]:
        start_time = time.time()
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(split.X_test)
            except Exception:
                pass

        training_time = time.time() - start_time
        metrics = MetricsCalculator.calculate_all_metrics(
            split.y_test, y_pred, y_proba
        )

        return {
            'metrics': metrics,
            'training_time': training_time,
            'n_train': len(split.y_train),
            'n_test': len(split.y_test)
        }

    def _evaluate_loso(self, model: BaseEstimator, loso_splits: Dict[str, TaskSplit],
                       model_name: str) -> Dict[str, Any]:
        per_subject_results = {}

        for subject_id, split in loso_splits.items():
            result = self._evaluate_split(model, split, model_name)
            per_subject_results[str(subject_id)] = result
        aggregated = self._aggregate_loso_metrics(per_subject_results)

        return {
            'per_subject': per_subject_results,
            'aggregated': aggregated
        }

    def _aggregate_loso_metrics(self, per_subject_results: Dict[str, Any]) -> Dict[str, Any]:
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_weighted', 'kappa']

        aggregated = {
            'n_subjects': len(per_subject_results)
        }

        for metric_name in metrics_names:
            values = []
            for result in per_subject_results.values():
                if metric_name in result['metrics']:
                    values.append(result['metrics'][metric_name])

            if values:
                aggregated[metric_name + '_mean'] = np.mean(values)
                aggregated[metric_name + '_std'] = np.std(values)
                aggregated[metric_name + '_min'] = np.min(values)
                aggregated[metric_name + '_max'] = np.max(values)

        return aggregated

    def run(self) -> pd.DataFrame:
        all_results = {}

        for dataset_name in self.config.get('datasets', {}).keys():
            try:
                results = self.run_for_dataset(dataset_name)
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Failed to run benchmark for dataset '{dataset_name}': {e}")

        self.results = all_results
        self._save_results()

        return self.get_summary()

    def _save_results(self):
        output_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        with open(output_file, 'w') as f:
            json.dump(self.results, f, default=convert_to_serializable, indent=2)

        logger.info(f"Results saved to {output_file}")
        self._save_csv_summary()

    def _save_csv_summary(self):
        summary = self.get_summary()
        csv_file = self.output_dir / f"summary_{self.timestamp}.csv"
        summary.to_csv(csv_file, index=False)
        logger.info(f"Summary saved to {csv_file}")

    def get_summary(self) -> pd.DataFrame:
        rows = []

        for dataset_name, dataset_results in self.results.items():
            for task_name, task_results in dataset_results.get('models', {}).items():
                for model_name, model_results in task_results.items():
                    if 'within_subject' in model_results:
                        ws = model_results['within_subject']
                        rows.append({
                            'dataset': dataset_name,
                            'task': task_name,
                            'model': model_name,
                            'evaluation': 'within_subject',
                            'accuracy': ws['metrics']['accuracy'],
                            'f1_weighted': ws['metrics']['f1_weighted'],
                            'kappa': ws['metrics']['kappa'],
                            'training_time': ws['training_time'],
                            'n_train': ws['n_train'],
                            'n_test': ws['n_test']
                        })
                    if 'loso' in model_results:
                        aggregated = model_results['loso'].get('aggregated', {})
                        rows.append({
                            'dataset': dataset_name,
                            'task': task_name,
                            'model': model_name,
                            'evaluation': 'loso',
                            'accuracy': aggregated.get('accuracy_mean', np.nan),
                            'accuracy_std': aggregated.get('accuracy_std', np.nan),
                            'f1_weighted': aggregated.get('f1_weighted_mean', np.nan),
                            'kappa': aggregated.get('kappa_mean', np.nan),
                            'n_subjects': aggregated.get('n_subjects', 0),
                            'training_time': np.nan,
                            'n_train': np.nan,
                            'n_test': np.nan
                        })

        return pd.DataFrame(rows)
