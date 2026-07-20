import json
import hashlib
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Mapping, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from .core.abstract_dataset import BaseDataset, EEGData
from .core.abstract_task import BaseTask, TaskSplit
from .datasets.datasets_registry import get_dataset
from .datasets.base_eeg_data_loader import feature_list_sha256
from .tasks.tasks_registry import get_task
from .validation.cross_val import CrossValidator
from .validation.metrics import MetricsCalculator
from model_zoo import (
    BaseModelAdapter,
    ModelLike,
    build_model,
    model_requires_data_shape,
    model_requires_sequences,
)
from model_zoo.DL.sequence_utils import (
    SEQUENCE_INDEX_COLUMNS,
    build_sequences,
    sequence_index_sha256,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RUN_MANIFEST_SCHEMA_VERSION = "benchmark-run-v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serializable_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _serializable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_serializable_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def canonical_benchmark_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the deterministic scientific config used for run identity.

    ``output_dir`` is execution placement rather than an experimental
    parameter. Excluding it also permits content-addressed output directories.
    """
    scientific = {
        key: value for key, value in config.items() if key != 'output_dir'
    }
    return _serializable_config_value(scientific)


def benchmark_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a benchmark configuration."""
    payload = json.dumps(
        canonical_benchmark_config(config),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class CompletedBenchmarkRun:
    """Validated pointer to one authoritative standard benchmark run."""

    config_hash: str
    run_directory: Path
    result_file: Path
    summary_file: Path | None
    manifest_file: Path | None
    legacy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_hash': self.config_hash,
            'benchmark_run_directory': str(self.run_directory),
            'benchmark_result_file': str(self.result_file),
            'benchmark_summary_file': (
                None if self.summary_file is None else str(self.summary_file)
            ),
            'benchmark_manifest_file': (
                None if self.manifest_file is None else str(self.manifest_file)
            ),
            'legacy': bool(self.legacy),
            'status': 'completed',
        }


class BenchmarkRunner:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './benchmark_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.timestamp
        self.result_file = (
            self.output_dir / f"benchmark_results_{self.timestamp}.json"
        )
        self.summary_file = self.output_dir / f"summary_{self.timestamp}.csv"
        self.config_hash = benchmark_config_hash(config)
        self.results = {}
        self.models = {}

        self._setup_models()

    @staticmethod
    def config_hash_for(config: Mapping[str, Any]) -> str:
        """Public config identity shared by CLI, experiments and AutoML."""
        return benchmark_config_hash(config)

    @classmethod
    def _validate_result_artifacts(cls, results: Mapping[str, Any]) -> None:
        supported_results = 0
        for dataset_result in results.values():
            if not isinstance(dataset_result, Mapping):
                continue
            for task_result in dataset_result.get('models', {}).values():
                for model_result in task_result.values():
                    group = model_result.get('group_kfold_subject')
                    cross_source = model_result.get('cross_source_holdout')
                    if isinstance(group, Mapping):
                        protocol_result = group
                        partitions = group.get('folds', {})
                        expected_partitions = int(group.get('n_folds', -1))
                        incomplete_message = (
                            "Standard benchmark result has incomplete "
                            "GroupKFold folds"
                        )
                    elif isinstance(cross_source, Mapping):
                        protocol_result = cross_source
                        partitions = cross_source.get('splits', {})
                        expected_partitions = int(
                            cross_source.get('n_splits', -1)
                        )
                        incomplete_message = (
                            "Standard benchmark result has incomplete "
                            "cross-source splits"
                        )
                    else:
                        continue
                    supported_results += 1
                    if (
                        not partitions
                        or len(partitions) != expected_partitions
                    ):
                        raise ValueError(incomplete_message)
                    expected_predictions = 0
                    for partition_name, partition_result in partitions.items():
                        expected_predictions += int(
                            partition_result.get('n_test', 0)
                        )
                        artifacts = partition_result.get('artifacts', {})
                        if not artifacts:
                            raise ValueError(
                                "Standard benchmark partition "
                                f"{partition_name} has no artifacts"
                            )
                        missing = [
                            key for key, path in artifacts.items()
                            if not Path(path).exists()
                        ]
                        if missing:
                            raise ValueError(
                                "Standard benchmark partition "
                                f"{partition_name} is missing "
                                f"artifacts: {missing}"
                            )
                    predictions_path = Path(
                        protocol_result.get('artifacts', {}).get(
                            'predictions', ''
                        )
                    )
                    if not predictions_path.is_file():
                        raise ValueError(
                            "Standard benchmark unified predictions are missing"
                        )
                    predictions = pd.read_parquet(predictions_path)
                    identity = (
                        'sequence_id'
                        if 'sequence_id' in predictions.columns
                        else 'sample_id'
                    )
                    if identity not in predictions.columns:
                        raise ValueError(
                            "Standard benchmark predictions have no observation ID"
                        )
                    if predictions[identity].duplicated().any():
                        raise ValueError(
                            "Standard benchmark predictions contain duplicate IDs"
                        )
                    if len(predictions) != expected_predictions:
                        raise ValueError(
                            "Standard benchmark predictions are incomplete: "
                            f"rows={len(predictions)}, expected={expected_predictions}"
                        )
        if supported_results == 0:
            raise ValueError(
                "Standard benchmark result has no supported evaluation result"
            )

    @classmethod
    def validate_completed_run(
            cls,
            run_directory: str | Path,
            *,
            expected_config_hash: str,
            result_file: str | Path | None = None,
            manifest_file: str | Path | None = None,
            legacy: bool = False,
    ) -> CompletedBenchmarkRun:
        """Validate config identity and required standard benchmark artifacts."""
        run_dir = Path(run_directory)
        config_path = run_dir / 'config.yaml'
        metrics_path = run_dir / 'metrics.json'
        if not config_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"Incomplete standard benchmark run: {run_dir}")
        with open(config_path, encoding='utf-8') as input_file:
            saved_config = yaml.safe_load(input_file) or {}
        actual_hash = benchmark_config_hash(saved_config)
        if actual_hash != expected_config_hash:
            raise ValueError(
                "Benchmark config hash mismatch: "
                f"expected={expected_config_hash}, actual={actual_hash}"
            )
        with open(metrics_path, encoding='utf-8') as input_file:
            results = json.load(input_file)
        cls._validate_result_artifacts(results)

        resolved_result = (
            Path(result_file)
            if result_file is not None
            else run_dir.parent / f"benchmark_results_{run_dir.name}.json"
        )
        if not resolved_result.is_file():
            raise ValueError(
                f"Standard benchmark result JSON is missing: {resolved_result}"
            )
        summary = run_dir.parent / f"summary_{run_dir.name}.csv"
        return CompletedBenchmarkRun(
            config_hash=actual_hash,
            run_directory=run_dir,
            result_file=resolved_result,
            summary_file=summary if summary.is_file() else None,
            manifest_file=(
                None if manifest_file is None else Path(manifest_file)
            ),
            legacy=legacy,
        )

    @classmethod
    def find_completed_run(
            cls,
            config: Mapping[str, Any],
            *,
            search_directories: Iterable[str | Path] | None = None,
    ) -> CompletedBenchmarkRun | None:
        """Find a valid standard run by config hash, including legacy runs."""
        expected_hash = benchmark_config_hash(config)
        roots = list(search_directories or [config.get('output_dir', '.')])
        candidates: list[tuple[Path, Path | None, Path | None, bool]] = []
        for root_value in roots:
            root = Path(root_value)
            if not root.exists():
                continue
            for manifest_path in root.glob('*/run_manifest.json'):
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding='utf-8')
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    manifest.get('status') != 'completed'
                    or manifest.get('config_hash') != expected_hash
                ):
                    continue
                candidates.append((
                    manifest_path.parent,
                    Path(manifest.get('benchmark_result_file', '')),
                    manifest_path,
                    False,
                ))
            for config_path in root.glob('*/config.yaml'):
                if (config_path.parent / 'run_manifest.json').exists():
                    continue
                candidates.append((config_path.parent, None, None, True))

        candidates.sort(
            key=lambda item: item[0].stat().st_mtime,
            reverse=True,
        )
        for run_dir, result_file, manifest_file, legacy in candidates:
            try:
                return cls.validate_completed_run(
                    run_dir,
                    expected_config_hash=expected_hash,
                    result_file=result_file,
                    manifest_file=manifest_file,
                    legacy=legacy,
                )
            except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
                continue
        return None

    def completed_run(self) -> CompletedBenchmarkRun:
        """Return the validated standard reference after ``run()``."""
        return self.validate_completed_run(
            self.run_dir,
            expected_config_hash=self.config_hash,
            result_file=self.result_file,
            manifest_file=self.run_dir / 'run_manifest.json',
        )

    def _setup_models(self):
        model_configs = self.config.get('models', {})

        for model_name, model_config in model_configs.items():
            try:
                model_type = model_config.get('type')
                if not model_type:
                    raise ValueError("Model config must define a non-empty 'type'")
                lazy_protocol = self.config.get('evaluation', {}).get(
                    'protocol'
                ) in {'group_kfold_subject', 'cross_source_holdout'}
                if model_requires_data_shape(model_type) or lazy_protocol:
                    model = None
                    logger.info(
                        f"Model '{model_name}' will be initialized after data loading"
                    )
                else:
                    model = self._create_model(model_config)
                self.models[model_name] = {
                    'model': model,
                    'config': model_config
                }
                logger.info(f"Model '{model_name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize model '{model_name}': {e}")

    def _create_model(
            self,
            model_config: Dict[str, Any],
            input_shape: Optional[Tuple[int, ...]] = None,
            num_outputs: Optional[int] = None
    ) -> ModelLike:
        model_type = model_config.get('type')
        if not model_type:
            raise ValueError("Model config must define a non-empty 'type'")
        params = model_config.get('params', {})
        return build_model(
            model_name=model_type,
            task_type=model_config.get('task_type', 'classification'),
            input_shape=(
                input_shape
                if input_shape is not None
                else model_config.get('input_shape')
            ),
            num_outputs=(
                num_outputs
                if num_outputs is not None
                else model_config.get('num_outputs')
            ),
            params=params,
        )

    def _get_model_for_split(
            self,
            model_info: Dict[str, Any],
            split: TaskSplit,
            num_outputs: int
    ) -> ModelLike:
        model_config = model_info['config']
        model_type = model_config.get('type', '')
        model = model_info.get('model')
        if model is None or model_requires_data_shape(model_type):
            input_shape = tuple(split.X_train.shape[1:])
            model = self._create_model(
                model_config,
                input_shape=input_shape,
                num_outputs=num_outputs,
            )
            model_info['model'] = model
            logger.info(
                f"      Built model '{model_type}' with input_shape={input_shape}, "
                f"num_outputs={num_outputs}"
            )
        return model

    def load_dataset(self, dataset_name: str) -> EEGData:
        try:
            dataset_config = deepcopy(self.config['datasets'][dataset_name])
            if (
                'raw_preprocessing' in self.config
                and 'raw_preprocessing' not in dataset_config
            ):
                dataset_config['raw_preprocessing'] = deepcopy(
                    self.config['raw_preprocessing']
                )
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
            configured_task_type = getattr(task, 'task_type', 'classification')
            task_type = (
                configured_task_type.strip().lower()
                if isinstance(configured_task_type, str)
                else 'classification'
            )
            task_num_outputs = task.n_classes if task_type == 'classification' else 1
            cv = CrossValidator(task)
            subject_ids = np.unique(data.subject_ids)
            evaluation_config = self.config.get('evaluation')
            if evaluation_config:
                protocol = evaluation_config.get('protocol')
                if protocol == 'cross_source_holdout':
                    thresholds = evaluation_config.get('thresholds', {})
                    cross_source_split = cv.run_cross_source_holdout(
                        train_source=evaluation_config.get('train_source', ''),
                        test_source=evaluation_config.get('test_source', ''),
                        subject_mode=evaluation_config.get(
                            'subject_mode', 'source_exclusive'
                        ),
                        remove_logical_duplicates=bool(
                            evaluation_config.get(
                                'remove_logical_duplicates', True
                            )
                        ),
                        minimum_train_subjects=int(
                            thresholds.get('minimum_train_subjects', 5)
                        ),
                        minimum_test_subjects=int(
                            thresholds.get('minimum_test_subjects', 3)
                        ),
                        minimum_train_classes=int(
                            thresholds.get('minimum_train_classes', 5)
                        ),
                        minimum_test_classes=int(
                            thresholds.get('minimum_test_classes', 2)
                        ),
                        minimum_predictions_per_test_subject=int(
                            thresholds.get(
                                'minimum_predictions_per_test_subject', 20
                            )
                        ),
                        max_train_windows=evaluation_config.get(
                            'max_train_windows'
                        ),
                        max_test_windows=evaluation_config.get(
                            'max_test_windows'
                        ),
                        random_state=int(
                            evaluation_config.get('random_state', 42)
                        ),
                    )
                    if cross_source_split.metadata['status'] != 'valid':
                        raise ValueError(
                            "Cross-source split is invalid: "
                            + '; '.join(
                                cross_source_split.metadata['invalid_reasons']
                            )
                        )
                    for model_name, model_info in self.models.items():
                        cross_source_result = self._evaluate_cross_source(
                            outer_split=cross_source_split,
                            model_name=model_name,
                            model_config=model_info['config'],
                            num_outputs=task_num_outputs,
                            dataset_name=dataset_name,
                            task_name=task_name,
                        )
                        results['models'].setdefault(task_name, {})
                        results['models'][task_name].setdefault(model_name, {})
                        results['models'][task_name][model_name][
                            protocol
                        ] = cross_source_result
                    continue
                if protocol != 'group_kfold_subject':
                    raise ValueError(
                        f"Unknown evaluation protocol {protocol!r}. "
                        "Available: ['group_kfold_subject', "
                        "'cross_source_holdout']"
                    )
                group_column = evaluation_config.get('group_column')
                if not group_column:
                    raise ValueError(
                        "evaluation.group_column is required for group_kfold_subject"
                    )
                group_splits = cv.run_group_kfold(
                    group_column=group_column,
                    n_splits=int(evaluation_config.get('n_splits', 5)),
                    random_state=int(evaluation_config.get('random_state', 42)),
                    precomputed_fold_column=evaluation_config.get(
                        'precomputed_fold_column'
                    ),
                )
                requested_folds = evaluation_config.get('folds')
                if requested_folds is not None:
                    fold_names = {
                        f"fold_{int(fold):02d}" for fold in requested_folds
                    }
                    unknown = sorted(fold_names - set(group_splits))
                    if unknown:
                        raise ValueError(
                            f"Requested evaluation folds do not exist: {unknown}"
                        )
                    group_splits = {
                        name: split
                        for name, split in group_splits.items()
                        if name in fold_names
                    }
                for model_name, model_info in self.models.items():
                    group_results = self._evaluate_group_kfold(
                        group_splits=group_splits,
                        model_name=model_name,
                        model_config=model_info['config'],
                        num_outputs=task_num_outputs,
                        dataset_name=dataset_name,
                        task_name=task_name,
                        task_type=task_type,
                    )
                    results['models'].setdefault(task_name, {})
                    results['models'][task_name].setdefault(model_name, {})
                    results['models'][task_name][model_name][protocol] = group_results
                continue
            if self.config.get('run_within_subject', True):
                logger.info("    Running within-subject evaluation")
                split = cv.run_within_subject()
                for model_name, model_info in self.models.items():
                    try:
                        model = self._get_model_for_split(
                            model_info,
                            split,
                            num_outputs=task_num_outputs,
                        )
                        split_name = split.metadata.get(
                            'split_type', 'within_subject'
                        )
                        eval_result = self._evaluate_split(
                            model,
                            split,
                            model_name,
                            dataset_name=dataset_name,
                            task_name=task_name,
                            artifact_split_name=split_name,
                            task_type=task_type,
                        )
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
                    try:
                        loso_results = self._evaluate_loso(
                            model_info.get('model'),
                            loso_splits,
                            model_name,
                            model_config=model_info['config'],
                            num_outputs=task_num_outputs,
                            dataset_name=dataset_name,
                            task_name=task_name,
                            task_type=task_type,
                        )
                        if task_name not in results['models']:
                            results['models'][task_name] = {}
                        if model_name not in results['models'][task_name]:
                            results['models'][task_name][model_name] = {}
                        results['models'][task_name][model_name]['loso'] = loso_results
                    except Exception as e:
                        logger.error(f"      Model '{model_name}' LOSO failed: {e}")

        return results

    def _evaluate_split(
            self,
            model: ModelLike,
            split: TaskSplit,
            model_name: str,
            dataset_name: Optional[str] = None,
            task_name: Optional[str] = None,
            artifact_split_name: Optional[str] = None,
            task_type: str = 'classification',
    ) -> Dict[str, Any]:
        start_time = time.time()
        self._configure_model_validation(model, split)
        model.fit(split.X_train, split.y_train)
        detailed_predictions = None
        detailed_predictor = getattr(model, 'predict_detailed', None)
        if callable(detailed_predictor):
            detailed_predictions = detailed_predictor(split.X_test)
            y_pred = np.asarray(detailed_predictions['y_pred'])
            y_proba = np.asarray(
                detailed_predictions['class_probabilities']
            )
        else:
            y_pred = model.predict(split.X_test)
            y_proba = None
        if detailed_predictions is None and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(split.X_test)
            except Exception:
                pass

        training_time = time.time() - start_time
        metrics = MetricsCalculator.calculate_all_metrics(
            split.y_test,
            y_pred,
            y_proba,
            task_type=task_type,
            expected_rank=(
                None
                if detailed_predictions is None
                else detailed_predictions.get(
                    'categorical_expected_rank',
                    detailed_predictions.get('expected_rank'),
                )
            ),
        )
        if (
            detailed_predictions is not None
            and str(detailed_predictions.get('head_type', '')).strip().lower()
            == 'categorical_corn'
        ):
            auxiliary_metrics = MetricsCalculator.calculate_all_metrics(
                split.y_test,
                np.asarray(detailed_predictions['aux_ordinal_prediction']),
                np.asarray(detailed_predictions['aux_class_probabilities']),
                task_type=task_type,
                expected_rank=np.asarray(
                    detailed_predictions['aux_expected_rank']
                ),
            )
            metrics.update({
                f'aux_{name}': value
                for name, value in auxiliary_metrics.items()
                if name != 'confusion_matrix'
            })
            metrics['aux_confusion_matrix'] = auxiliary_metrics[
                'confusion_matrix'
            ]
            metrics['categorical_aux_prediction_agreement'] = float(
                np.mean(
                    np.asarray(y_pred)
                    == np.asarray(
                        detailed_predictions['aux_ordinal_prediction']
                    )
                )
            )

        result = {
            'metrics': metrics,
            'training_time': training_time,
            'n_train': len(split.y_train),
            'n_test': len(split.y_test),
            'split_metadata': split.metadata,
        }

        if isinstance(model, BaseModelAdapter):
            result['training'] = model.get_training_summary()

        if dataset_name and task_name and artifact_split_name:
            result['artifacts'] = self._save_split_artifacts(
                model=model,
                split=split,
                y_pred=y_pred,
                y_proba=y_proba,
                dataset_name=dataset_name,
                task_name=task_name,
                model_name=model_name,
                artifact_split_name=artifact_split_name,
                metrics=metrics,
                task_type=task_type,
                detailed_predictions=detailed_predictions,
            )

        return result

    @staticmethod
    def _safe_path_component(value: Any) -> str:
        component = ''.join(
            char if char.isalnum() or char in {'-', '_', '.'} else '_'
            for char in str(value)
        ).strip('._')
        return component or 'unnamed'

    def _model_artifact_dir(
            self,
            dataset_name: str,
            task_name: str,
            model_name: str
    ) -> Path:
        artifact_dir = self.output_dir / self.timestamp
        for component in (dataset_name, task_name, model_name):
            artifact_dir = artifact_dir / self._safe_path_component(component)
        return artifact_dir

    def _save_split_artifacts(
            self,
            model: ModelLike,
            split: TaskSplit,
            y_pred: np.ndarray,
            y_proba: Optional[np.ndarray],
            dataset_name: str,
            task_name: str,
            model_name: str,
            artifact_split_name: str,
            metrics: Dict[str, Any],
            task_type: str = 'classification',
            detailed_predictions: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, str]:
        artifact_dir = self._model_artifact_dir(
            dataset_name, task_name, model_name
        )
        protocol = split.metadata.get('protocol')
        if protocol:
            fold_name = split.metadata.get('fold_name', artifact_split_name)
            artifact_dir = (
                artifact_dir
                / self._safe_path_component(protocol)
                / self._safe_path_component(fold_name)
            )
        else:
            artifact_dir = artifact_dir / self._safe_path_component(
                artifact_split_name
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        n_test = len(split.y_test)
        sample_ids = (
            np.asarray(split.sample_id_test)
            if split.sample_id_test is not None
            else np.arange(n_test, dtype=np.int64)
        )
        subject_ids = (
            np.asarray(split.subject_test)
            if split.subject_test is not None
            else np.full(n_test, 'unknown', dtype=object)
        )
        record_ids = (
            np.asarray(split.record_id_test)
            if split.record_id_test is not None
            else np.full(n_test, 'unknown', dtype=object)
        )
        identifier_lengths = {
            'sample_id': len(sample_ids),
            'subject_id': len(subject_ids),
            'record_id': len(record_ids),
        }
        invalid_lengths = {
            key: value
            for key, value in identifier_lengths.items()
            if value != n_test
        }
        if invalid_lengths:
            raise ValueError(
                f"Prediction identifiers must have {n_test} rows, got "
                f"{invalid_lengths}"
            )

        protocol_name = split.metadata.get(
            'protocol', split.metadata.get('split_type', artifact_split_name)
        )
        prediction_data = {
            'dataset': dataset_name,
            'task': task_name,
            'model': model_name,
            'split': split.metadata.get('split_type', artifact_split_name),
            'protocol': protocol_name,
            'fold': split.metadata.get('fold'),
            'sample_index': sample_ids,
            'sample_id': sample_ids,
            'subject_id': subject_ids,
            'record_id': record_ids,
            'y_true': np.asarray(split.y_test),
            'y_pred': np.asarray(y_pred),
        }
        for column, values in split.row_metadata_test.items():
            row_values = np.asarray(values)
            if len(row_values) != n_test:
                raise ValueError(
                    f"Prediction metadata {column!r} must have {n_test} rows, "
                    f"got {len(row_values)}"
                )
            if column not in prediction_data:
                prediction_data[column] = row_values
        predictions = pd.DataFrame(prediction_data)
        if y_proba is not None:
            probabilities = np.asarray(y_proba)
            if probabilities.ndim != 2 or len(probabilities) != len(predictions):
                raise ValueError(
                    f"Invalid probability shape {probabilities.shape} for "
                    f"{len(predictions)} predictions"
                )
            for class_index in range(probabilities.shape[1]):
                predictions[f'proba_{class_index}'] = probabilities[:, class_index]

        head_type = 'categorical'
        if detailed_predictions is not None:
            head_type = str(
                detailed_predictions.get('head_type', 'categorical')
            ).strip().lower()
        if head_type in {'coral', 'corn'}:
            required = {
                'threshold_logits': int(model.num_classes) - 1,
                'threshold_probabilities': int(model.num_classes) - 1,
                'class_probabilities': int(model.num_classes),
            }
            arrays: Dict[str, np.ndarray] = {}
            for name, width in required.items():
                values = np.asarray(detailed_predictions[name])
                if values.shape != (n_test, width):
                    raise ValueError(
                        f"Ordinal detailed output {name!r} must have shape "
                        f"{(n_test, width)}, got {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Ordinal detailed output {name!r} is not finite"
                    )
                arrays[name] = values
            for threshold_index in range(int(model.num_classes) - 1):
                predictions[f'threshold_logit_{threshold_index}'] = (
                    arrays['threshold_logits'][:, threshold_index]
                )
                predictions[f'threshold_probability_{threshold_index}'] = (
                    arrays['threshold_probabilities'][:, threshold_index]
                )
            for class_index in range(int(model.num_classes)):
                predictions[f'class_probability_{class_index}'] = (
                    arrays['class_probabilities'][:, class_index]
                )
            expected = np.asarray(detailed_predictions['expected_rank'])
            ordinal_argmax = np.asarray(
                detailed_predictions['ordinal_argmax']
            )
            if expected.shape != (n_test,) or ordinal_argmax.shape != (n_test,):
                raise ValueError(
                    "Ordinal expected_rank and ordinal_argmax must be one-dimensional"
                )
            if not np.isfinite(expected).all():
                raise ValueError("Ordinal expected_rank is not finite")
            predictions['expected_rank'] = expected
            predictions['ordinal_argmax'] = ordinal_argmax
            predictions['y_pred_argmax'] = ordinal_argmax
            predictions['head_type'] = head_type
            conditional = detailed_predictions.get('conditional_probabilities')
            if conditional is not None:
                conditional_values = np.asarray(conditional)
                if conditional_values.shape != (
                    n_test,
                    int(model.num_classes) - 1,
                ):
                    raise ValueError(
                        "CORN conditional probabilities have an invalid shape"
                    )
                if not np.isfinite(conditional_values).all():
                    raise ValueError(
                        "CORN conditional probabilities are not finite"
                    )
                for threshold_index in range(int(model.num_classes) - 1):
                    predictions[
                        f'conditional_probability_{threshold_index}'
                    ] = conditional_values[:, threshold_index]

        if head_type == 'categorical_corn':
            required_joint = {
                'class_probabilities': int(model.num_classes),
                'aux_threshold_probabilities': int(model.num_classes) - 1,
                'aux_class_probabilities': int(model.num_classes),
                'auxiliary_raw_outputs': int(model.num_classes) - 1,
            }
            joint_arrays: Dict[str, np.ndarray] = {}
            for name, width in required_joint.items():
                values = np.asarray(detailed_predictions[name])
                if values.shape != (n_test, width):
                    raise ValueError(
                        f"Auxiliary CORN detailed output {name!r} must have "
                        f"shape {(n_test, width)}, got {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Auxiliary CORN detailed output {name!r} is not finite"
                    )
                joint_arrays[name] = values
            categorical_expected = np.asarray(
                detailed_predictions['categorical_expected_rank']
            )
            aux_expected = np.asarray(detailed_predictions['aux_expected_rank'])
            aux_prediction = np.asarray(
                detailed_predictions['aux_ordinal_prediction']
            )
            aux_argmax = np.asarray(detailed_predictions['aux_ordinal_argmax'])
            for name, values in {
                'categorical_expected_rank': categorical_expected,
                'aux_expected_rank': aux_expected,
                'aux_ordinal_prediction': aux_prediction,
                'aux_ordinal_argmax': aux_argmax,
            }.items():
                if values.shape != (n_test,):
                    raise ValueError(
                        f"Auxiliary CORN detailed output {name!r} must be one-dimensional"
                    )
                if name.endswith('expected_rank') and not np.isfinite(values).all():
                    raise ValueError(f"{name} contains non-finite values")
            predictions['head_type'] = head_type
            predictions['categorical_expected_rank'] = categorical_expected
            predictions['aux_expected_rank'] = aux_expected
            predictions['aux_ordinal_prediction'] = aux_prediction
            predictions['aux_ordinal_argmax'] = aux_argmax
            predictions['auxiliary_weight'] = float(
                detailed_predictions['auxiliary_weight']
            )
            for class_index in range(int(model.num_classes)):
                predictions[f'class_probability_{class_index}'] = (
                    joint_arrays['class_probabilities'][:, class_index]
                )
                predictions[f'aux_class_probability_{class_index}'] = (
                    joint_arrays['aux_class_probabilities'][:, class_index]
                )
            for threshold_index in range(int(model.num_classes) - 1):
                predictions[f'aux_threshold_logit_{threshold_index}'] = (
                    joint_arrays['auxiliary_raw_outputs'][:, threshold_index]
                )
                predictions[f'aux_threshold_probability_{threshold_index}'] = (
                    joint_arrays['aux_threshold_probabilities'][:, threshold_index]
                )
            conditional = detailed_predictions.get(
                'aux_conditional_probabilities'
            )
            if conditional is not None:
                conditional_values = np.asarray(conditional)
                if conditional_values.shape != (
                    n_test, int(model.num_classes) - 1
                ) or not np.isfinite(conditional_values).all():
                    raise ValueError(
                        'Auxiliary CORN conditional probabilities are invalid'
                    )
                for threshold_index in range(int(model.num_classes) - 1):
                    predictions[f'aux_conditional_probability_{threshold_index}'] = (
                        conditional_values[:, threshold_index]
                    )

        predictions_path = artifact_dir / 'predictions.parquet'
        predictions.to_parquet(predictions_path, index=False)
        artifacts = {'predictions': str(predictions_path)}
        if head_type in {'coral', 'corn'}:
            ordinal_metadata_path = artifact_dir / 'ordinal_metadata.json'
            cumulative = arrays['threshold_probabilities']
            class_probabilities = arrays['class_probabilities']
            tolerance = float(model.objective_handler.tolerance)
            raw_class_probabilities = np.concatenate(
                [
                    1.0 - cumulative[:, :1],
                    cumulative[:, :-1] - cumulative[:, 1:],
                    cumulative[:, -1:],
                ],
                axis=1,
            )
            monotonicity_violation = cumulative[:, 1:] - cumulative[:, :-1]
            maximum_monotonicity_violation = float(
                max(0.0, np.max(monotonicity_violation, initial=0.0))
            )
            row_sum_error = float(
                np.max(
                    np.abs(class_probabilities.sum(axis=1) - 1.0),
                    initial=0.0,
                )
            )
            ordinal_metadata = {
                **dict(model.objective_handler.to_metadata()),
                'probability_conversion_version': 1,
                'primary_prediction_column': 'y_pred',
                'diagnostic_argmax_column': 'ordinal_argmax',
                'class_probability_columns': [
                    f'class_probability_{index}'
                    for index in range(int(model.num_classes))
                ],
                'threshold_probability_columns': [
                    f'threshold_probability_{index}'
                    for index in range(int(model.num_classes) - 1)
                ],
                'round_off_correction_count': int(
                    np.count_nonzero(raw_class_probabilities < 0.0)
                ),
                'maximum_monotonicity_violation': (
                    maximum_monotonicity_violation
                ),
                'monotonicity_within_tolerance': bool(
                    maximum_monotonicity_violation <= tolerance
                ),
                'maximum_class_probability_row_sum_error': row_sum_error,
            }
            with open(
                ordinal_metadata_path, 'w', encoding='utf-8'
            ) as output:
                json.dump(ordinal_metadata, output, indent=2)
            artifacts['ordinal_metadata'] = str(ordinal_metadata_path)
        if head_type == 'categorical_corn':
            auxiliary_metadata_path = artifact_dir / 'auxiliary_corn_metadata.json'
            cumulative = joint_arrays['aux_threshold_probabilities']
            primary_probabilities = joint_arrays['class_probabilities']
            auxiliary_probabilities = joint_arrays['aux_class_probabilities']
            monotonicity_violation = cumulative[:, 1:] - cumulative[:, :-1]
            metadata = {
                **dict(model.objective_handler.to_metadata()),
                'primary_prediction_column': 'y_pred',
                'auxiliary_prediction_column': 'aux_ordinal_prediction',
                'primary_probability_columns': [
                    f'class_probability_{index}'
                    for index in range(int(model.num_classes))
                ],
                'auxiliary_threshold_probability_columns': [
                    f'aux_threshold_probability_{index}'
                    for index in range(int(model.num_classes) - 1)
                ],
                'maximum_primary_probability_row_sum_error': float(
                    np.max(np.abs(primary_probabilities.sum(axis=1) - 1.0), initial=0.0)
                ),
                'maximum_auxiliary_probability_row_sum_error': float(
                    np.max(np.abs(auxiliary_probabilities.sum(axis=1) - 1.0), initial=0.0)
                ),
                'maximum_auxiliary_monotonicity_violation': float(
                    max(0.0, np.max(monotonicity_violation, initial=0.0))
                ),
                'categorical_aux_prediction_agreement': float(
                    np.mean(np.asarray(y_pred) == aux_prediction)
                ),
            }
            with open(auxiliary_metadata_path, 'w', encoding='utf-8') as output:
                json.dump(metadata, output, indent=2)
            artifacts['auxiliary_corn_metadata'] = str(auxiliary_metadata_path)

        metrics_path = artifact_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as output:
            json.dump(
                metrics,
                output,
                indent=2,
                default=lambda value: (
                    value.tolist() if isinstance(value, np.ndarray) else float(value)
                ),
            )
        artifacts['metrics'] = str(metrics_path)

        if str(task_type).strip().lower() in {'classification', 'classifier'}:
            labels = (
                np.arange(np.asarray(y_proba).shape[1], dtype=np.int64)
                if y_proba is not None
                else None
            )
            class_metrics = MetricsCalculator.calculate_class_metrics(
                split.y_test, y_pred, labels=labels
            )
            class_metrics_path = artifact_dir / 'class_metrics.json'
            with open(class_metrics_path, 'w', encoding='utf-8') as output:
                json.dump(class_metrics, output, indent=2)
            artifacts['class_metrics'] = str(class_metrics_path)

        feature_names = list(split.feature_names or [])
        feature_manifest = {
            'feature_group': split.metadata.get('dataset_metadata', {}).get(
                'feature_set'
            ),
            'feature_count': len(feature_names),
            'ordered_feature_names': feature_names,
            'feature_list_sha256': feature_list_sha256(feature_names),
            'serialization': 'UTF-8 feature name plus newline, in model order',
        }
        feature_manifest_path = artifact_dir / 'feature_manifest.json'
        with open(feature_manifest_path, 'w', encoding='utf-8') as output:
            json.dump(feature_manifest, output, indent=2)
        artifacts['feature_manifest'] = str(feature_manifest_path)

        feature_importances = getattr(model, 'feature_importances_', None)
        if feature_importances is not None:
            importance = np.asarray(feature_importances, dtype=float).reshape(-1)
            if len(importance) != len(feature_names):
                raise ValueError(
                    'Model feature importance length does not match feature names: '
                    f'{len(importance)} != {len(feature_names)}'
                )
            order = np.argsort(-importance, kind='mergesort')
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
            importance_path = artifact_dir / 'feature_importance.parquet'
            pd.DataFrame({
                'feature_index': np.arange(len(feature_names), dtype=np.int64),
                'feature_name': feature_names,
                'importance': importance,
                'rank': ranks,
            }).to_parquet(importance_path, index=False)
            artifacts['feature_importance'] = str(importance_path)

        sequence_stats = split.metadata.get('sequence_stats')
        if sequence_stats is not None:
            sequence_stats_path = artifact_dir / 'sequence_stats.json'
            with open(sequence_stats_path, 'w', encoding='utf-8') as output:
                json.dump(sequence_stats, output, indent=2, default=_json_default)
            artifacts['sequence_stats'] = str(sequence_stats_path)
            sequence_index = predictions.loc[:, list(SEQUENCE_INDEX_COLUMNS)]
            sequence_index_manifest = {
                'observation_unit': 'sequence',
                'sequence_count': int(len(sequence_index)),
                'sequence_index_sha256': sequence_index_sha256(sequence_index),
                'columns': list(SEQUENCE_INDEX_COLUMNS),
                'serialization': (
                    'Rows sorted by sequence_id; compact UTF-8 JSON array plus newline'
                ),
            }
            sequence_manifest_path = artifact_dir / 'sequence_index_manifest.json'
            with open(sequence_manifest_path, 'w', encoding='utf-8') as output:
                json.dump(sequence_index_manifest, output, indent=2)
            artifacts['sequence_index_manifest'] = str(sequence_manifest_path)

        if split.metadata.get('observation_unit') == 'raw_eeg_window':
            dataset_metadata = split.metadata.get('dataset_metadata', {})
            raw_stats: Dict[str, Any] = {
                'input_shape': list(split.X_test.shape[1:]),
                'channel_names': list(split.feature_names or []),
                'dataset': split.metadata.get('dataset_metadata', {}),
                'train': self._raw_partition_stats(split, 'train'),
                'test': self._raw_partition_stats(split, 'test'),
            }
            raw_stats_path = artifact_dir / 'raw_eeg_stats.json'
            with open(raw_stats_path, 'w', encoding='utf-8') as output:
                json.dump(raw_stats, output, indent=2, default=_json_default)
            artifacts['raw_eeg_stats'] = str(raw_stats_path)

            preprocessing_path = artifact_dir / 'preprocessing_metadata.json'
            preprocessing_metadata = {
                key: dataset_metadata.get(key)
                for key in (
                    'dataset_mode', 'raw_preprocessing', 'preprocessing_hashes',
                    'cache_roots', 'manifest_path',
                    'accepted_windows_before_deduplication',
                    'accepted_windows_after_deduplication', 'windows_loaded',
                )
            }
            with open(preprocessing_path, 'w', encoding='utf-8') as output:
                json.dump(
                    preprocessing_metadata, output, indent=2,
                    default=_json_default,
                )
            artifacts['preprocessing_metadata'] = str(preprocessing_path)

            selected_path = artifact_dir / 'selected_logical_records.parquet'
            logical_map_path = dataset_metadata.get('logical_recording_map_path')
            if (
                dataset_metadata.get('dataset_mode')
                == 'raw_deduplicated_logical_records'
                and logical_map_path
                and Path(logical_map_path).exists()
            ):
                selected_records = pd.read_parquet(logical_map_path)
            else:
                selected_records = pd.concat([
                    pd.DataFrame({
                        'record_id': np.asarray(split.record_id_train).astype(str),
                        'subject_id': np.asarray(split.subject_train).astype(str),
                        **{
                            key: np.asarray(values)
                            for key, values in split.row_metadata_train.items()
                            if key in {'record_group_id', 'source'}
                        },
                    }),
                    pd.DataFrame({
                        'record_id': np.asarray(split.record_id_test).astype(str),
                        'subject_id': np.asarray(split.subject_test).astype(str),
                        **{
                            key: np.asarray(values)
                            for key, values in split.row_metadata_test.items()
                            if key in {'record_group_id', 'source'}
                        },
                    }),
                ], ignore_index=True).drop_duplicates()
            selected_ids = set(dataset_metadata.get('selected_record_ids', []))
            selected_records['selected_for_dataset'] = (
                selected_records.get('selected_record_id', selected_records.get('record_id'))
                .astype(str).isin(selected_ids)
            )
            selected_records.to_parquet(selected_path, index=False)
            artifacts['selected_logical_records'] = str(selected_path)

            manifest_path = dataset_metadata.get('manifest_path')
            if manifest_path and Path(manifest_path).exists():
                rejected = pd.read_parquet(manifest_path)
                rejected = rejected.loc[rejected['status'].astype(str) != 'ok'].copy()
                if selected_ids and dataset_metadata.get('dataset_mode') == (
                    'raw_deduplicated_logical_records'
                ):
                    rejected = rejected.loc[
                        rejected['record_id'].astype(str).isin(selected_ids)
                    ]
                rejected_path = artifact_dir / 'rejected_windows.parquet'
                rejected.to_parquet(rejected_path, index=False)
                artifacts['rejected_windows'] = str(rejected_path)

        if protocol == 'cross_source_holdout':
            split_payload = {
                key: value
                for key, value in split.metadata.items()
                if key != 'dataset_metadata'
            }
            cross_source_split_path = artifact_dir / 'cross_source_split.json'
            with open(
                cross_source_split_path, 'w', encoding='utf-8'
            ) as output:
                json.dump(
                    split_payload, output, indent=2, default=_json_default
                )
            artifacts['cross_source_split'] = str(cross_source_split_path)

            excluded_subjects_path = artifact_dir / 'excluded_subjects.json'
            with open(
                excluded_subjects_path, 'w', encoding='utf-8'
            ) as output:
                json.dump(
                    split.metadata.get('excluded_subjects', {}),
                    output,
                    indent=2,
                    default=_json_default,
                )
            artifacts['excluded_subjects'] = str(excluded_subjects_path)

            excluded_logical_path = (
                artifact_dir / 'excluded_logical_recordings.json'
            )
            with open(
                excluded_logical_path, 'w', encoding='utf-8'
            ) as output:
                json.dump(
                    split.metadata.get('excluded_logical_record_ids', []),
                    output,
                    indent=2,
                    default=_json_default,
                )
            artifacts['excluded_logical_recordings'] = str(
                excluded_logical_path
            )

            source_distribution_path = (
                artifact_dir / 'source_distribution.json'
            )
            with open(
                source_distribution_path, 'w', encoding='utf-8'
            ) as output:
                json.dump(
                    split.metadata.get('source_distribution', {}),
                    output,
                    indent=2,
                    default=_json_default,
                )
            artifacts['source_distribution'] = str(source_distribution_path)

        validation_split = getattr(model, 'validation_split_', None)
        if validation_split is not None:
            validation_split_path = artifact_dir / 'validation_split.json'
            with open(validation_split_path, 'w', encoding='utf-8') as output:
                json.dump(
                    validation_split, output, indent=2, default=_json_default
                )
            artifacts['validation_split'] = str(validation_split_path)

        if isinstance(model, BaseModelAdapter):
            model_path = artifact_dir / 'model.pt'
            model.save(model_path)
            training_log_path = artifact_dir / 'training_log.csv'
            pd.DataFrame(model.training_log_).to_csv(training_log_path, index=False)
            artifacts['model'] = str(model_path)
            artifacts['training_log'] = str(training_log_path)
            if getattr(model, 'feature_mean_', None) is not None:
                normalization_path = artifact_dir / 'normalization_stats.json'
                normalization = {
                    'scope': 'inner_train_only',
                    'feature_names': list(split.feature_names or []),
                    'mean': np.asarray(model.feature_mean_).tolist(),
                    'scale': np.asarray(model.feature_scale_).tolist(),
                }
                if split.metadata.get('observation_unit') == 'raw_eeg_window':
                    normalization['channel_names'] = list(
                        split.feature_names or []
                    )
                with open(normalization_path, 'w', encoding='utf-8') as output:
                    json.dump(normalization, output, indent=2)
                artifacts['normalization_stats'] = str(normalization_path)

        return artifacts

    @staticmethod
    def _raw_partition_stats(split: TaskSplit, partition: str) -> Dict[str, Any]:
        metadata = getattr(split, f'row_metadata_{partition}')
        labels = np.asarray(getattr(split, f'y_{partition}'))
        stats: Dict[str, Any] = {
            'windows': int(len(labels)),
            'class_distribution': {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            },
        }
        for column in ('source', 'sfreq_original', 'sfreq_target'):
            if column in metadata:
                values, counts = np.unique(np.asarray(metadata[column]), return_counts=True)
                stats[f'{column}_counts'] = {
                    str(value): int(count)
                    for value, count in zip(values, counts)
                }
        if 'missing_fraction' in metadata:
            missing = np.asarray(metadata['missing_fraction'], dtype=float)
            finite = missing[np.isfinite(missing)]
            if len(finite):
                stats['missing_fraction'] = {
                    'min': float(np.min(finite)),
                    'mean': float(np.mean(finite)),
                    'max': float(np.max(finite)),
                }
        return stats

    def _configure_model_validation(
            self,
            model: ModelLike,
            split: TaskSplit
    ) -> None:
        validation_config = self.config.get('validation')
        if not validation_config:
            return
        observation_unit = split.metadata.get('observation_unit')
        if observation_unit not in {'sequence', 'raw_eeg_window'}:
            return
        strategy = str(
            validation_config.get('strategy', 'group_record')
        ).strip().lower()
        if strategy != 'group_record':
            raise ValueError(
                f"Unknown grouped validation strategy {strategy!r}. "
                "Available: ['group_record']"
            )
        group_column = str(
            validation_config.get('group_column', 'record_id')
        )
        if group_column == 'record_id':
            train_groups = np.asarray(split.record_id_train).astype(str)
            test_groups = np.asarray(split.record_id_test).astype(str)
        elif group_column == 'subject_id':
            train_groups = np.asarray(split.subject_train).astype(str)
            test_groups = np.asarray(split.subject_test).astype(str)
        elif (
            group_column in split.row_metadata_train
            and group_column in split.row_metadata_test
        ):
            train_groups = np.asarray(
                split.row_metadata_train[group_column]
            ).astype(str)
            test_groups = np.asarray(
                split.row_metadata_test[group_column]
            ).astype(str)
        else:
            available = sorted(
                set(split.row_metadata_train) & set(split.row_metadata_test)
            )
            raise ValueError(
                f"validation.group_column={group_column!r} is unavailable. "
                f"Available metadata columns: {available}"
            )
        if not hasattr(model, 'set_validation_groups'):
            raise TypeError(
                "PyTorch model does not support group-aware validation metadata"
            )
        train_subjects = np.asarray(split.subject_train).astype(str)
        outer_group_overlap = np.intersect1d(
            np.unique(train_groups), np.unique(test_groups)
        )
        if len(outer_group_overlap):
            raise RuntimeError(
                "Outer train/test validation groups overlap before inner "
                f"validation: {outer_group_overlap.astype(str).tolist()}"
            )
        model.set_validation_groups(
            train_groups,
            subject_ids=train_subjects,
            record_ids=np.asarray(split.record_id_train).astype(str),
            outer_test_record_ids=np.asarray(split.record_id_test).astype(str),
            strategy=strategy,
            group_column=group_column,
            validation_size=float(
                validation_config.get('validation_size', 0.15)
            ),
            random_state=int(validation_config.get('random_state', 42)),
        )

    @staticmethod
    def _partition_sequence_metadata(
            split: TaskSplit,
            partition: str
    ) -> pd.DataFrame:
        if partition not in {'train', 'test'}:
            raise ValueError(f"Unknown split partition {partition!r}")
        row_metadata = getattr(split, f'row_metadata_{partition}')
        subject_ids = getattr(split, f'subject_{partition}')
        sample_ids = getattr(split, f'sample_id_{partition}')
        record_ids = getattr(split, f'record_id_{partition}')
        required = {
            'subject_id': subject_ids,
            'sample_id': sample_ids,
            'record_id': record_ids,
        }
        missing = [key for key, values in required.items() if values is None]
        if missing:
            raise ValueError(
                f"Cannot build {partition} sequences without metadata: {missing}"
            )
        metadata = {
            key: np.asarray(values)
            for key, values in required.items()
        }
        metadata.update({
            key: np.asarray(values)
            for key, values in row_metadata.items()
        })
        return pd.DataFrame(metadata)

    def _build_sequence_split(self, split: TaskSplit) -> TaskSplit:
        sequence_config = self.config.get('sequence')
        if not sequence_config:
            raise ValueError(
                "Sequence models require a top-level 'sequence' configuration"
            )
        sequence_length = int(
            sequence_config.get('length', sequence_config.get('sequence_length', 10))
        )
        stride = int(sequence_config.get('stride', 1))
        target_position = str(sequence_config.get('target_position', 'last'))
        expected_step_seconds = sequence_config.get('expected_step_seconds')
        max_gap_seconds = sequence_config.get('max_gap_seconds')
        train_result = build_sequences(
            X=split.X_train,
            y=split.y_train,
            metadata=self._partition_sequence_metadata(split, 'train'),
            sequence_length=sequence_length,
            stride=stride,
            target_position=target_position,
            expected_step_seconds=expected_step_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        test_result = build_sequences(
            X=split.X_test,
            y=split.y_test,
            metadata=self._partition_sequence_metadata(split, 'test'),
            sequence_length=sequence_length,
            stride=stride,
            target_position=target_position,
            expected_step_seconds=expected_step_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        if len(train_result.X) == 0 or len(test_result.X) == 0:
            raise ValueError(
                "Sequence construction produced an empty partition: "
                f"train={len(train_result.X)}, test={len(test_result.X)}"
            )
        sequence_subject_overlap = np.intersect1d(
            train_result.metadata['subject_id'].unique(),
            test_result.metadata['subject_id'].unique(),
        )
        if (
            len(sequence_subject_overlap)
            and not split.metadata.get('allow_subject_overlap', False)
        ):
            raise RuntimeError(
                "Subject leakage detected after sequence construction: "
                f"{sequence_subject_overlap.astype(str).tolist()}"
            )

        metadata = dict(split.metadata)
        metadata.update({
            'observation_unit': 'sequence',
            'sequence_length': sequence_length,
            'sequence_stride': stride,
            'sequence_target_position': target_position,
            'sequence_expected_step_seconds': expected_step_seconds,
            'sequence_max_gap_seconds': max_gap_seconds,
            'n_train_sequences': len(train_result.X),
            'n_test_sequences': len(test_result.X),
            'subject_overlap': sequence_subject_overlap.astype(str).tolist(),
            'sequence_stats': {
                'train': train_result.stats,
                'test': test_result.stats,
            },
        })
        train_metadata = {
            column: train_result.metadata[column].to_numpy()
            for column in train_result.metadata.columns
        }
        test_metadata = {
            column: test_result.metadata[column].to_numpy()
            for column in test_result.metadata.columns
        }
        return TaskSplit(
            X_train=train_result.X,
            y_train=train_result.y,
            X_test=test_result.X,
            y_test=test_result.y,
            subject_train=train_result.metadata['subject_id'].to_numpy(),
            subject_test=test_result.metadata['subject_id'].to_numpy(),
            feature_names=split.feature_names,
            metadata=metadata,
            sample_id_train=train_result.metadata['target_sample_id'].to_numpy(),
            sample_id_test=test_result.metadata['target_sample_id'].to_numpy(),
            record_id_train=train_result.metadata['record_id'].to_numpy(),
            record_id_test=test_result.metadata['record_id'].to_numpy(),
            row_metadata_train=train_metadata,
            row_metadata_test=test_metadata,
        )

    def _evaluate_cross_source(
            self,
            outer_split: TaskSplit,
            model_name: str,
            model_config: Dict[str, Any],
            num_outputs: int,
            dataset_name: str,
            task_name: str,
    ) -> Dict[str, Any]:
        """Evaluate one directional transfer through the standard split path."""
        sequence_model = model_requires_sequences(model_config.get('type', ''))
        split = (
            self._build_sequence_split(outer_split)
            if sequence_model
            else outer_split
        )
        logical_overlap = split.metadata.get('logical_record_overlap', [])
        record_overlap = split.metadata.get('record_overlap', [])
        sample_overlap = split.metadata.get('sample_overlap', [])
        subject_overlap = split.metadata.get('subject_overlap', [])
        if logical_overlap or record_overlap or sample_overlap:
            raise RuntimeError(
                "Cross-source leakage detected before training: "
                f"logical={logical_overlap}, records={record_overlap}, "
                f"samples={sample_overlap}"
            )
        if subject_overlap and not split.metadata.get(
            'allow_subject_overlap', False
        ):
            raise RuntimeError(
                "Unexpected subject overlap before cross-source training: "
                f"{subject_overlap}"
            )
        split_model_config = deepcopy(model_config)
        model = self._create_model(
            split_model_config,
            input_shape=tuple(split.X_train.shape[1:]),
            num_outputs=num_outputs,
        )
        split_name = str(split.metadata['fold_name'])
        logger.info(
            f"      {model_name} {split_name}: "
            f"train={len(split.y_train)}, test={len(split.y_test)}, "
            f"train_subjects={split.metadata['n_train_subjects']}, "
            f"test_subjects={split.metadata['n_test_subjects']}"
        )
        result = self._evaluate_split(
            model=model,
            split=split,
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
            artifact_split_name=split_name,
        )
        protocol_dir = (
            self._model_artifact_dir(dataset_name, task_name, model_name)
            / 'cross_source_holdout'
        )
        protocol_dir.mkdir(parents=True, exist_ok=True)
        unified_predictions = protocol_dir / 'predictions.parquet'
        predictions = pd.read_parquet(result['artifacts']['predictions'])
        identity_column = (
            'sequence_id' if 'sequence_id' in predictions.columns else 'sample_id'
        )
        if predictions[identity_column].duplicated().any():
            raise RuntimeError(
                "Cross-source predictions contain duplicate observation IDs"
            )
        predictions.sort_values(identity_column).to_parquet(
            unified_predictions, index=False
        )
        return {
            'protocol': 'cross_source_holdout',
            'status': 'completed',
            'n_splits': 1,
            'splits': {split_name: result},
            'metrics': result['metrics'],
            'training_time_total': result['training_time'],
            'split_metadata': split.metadata,
            'artifacts': {'predictions': str(unified_predictions)},
        }

    def _evaluate_group_kfold(
            self,
            group_splits: Dict[str, TaskSplit],
            model_name: str,
            model_config: Dict[str, Any],
            num_outputs: int,
            dataset_name: str,
            task_name: str,
            task_type: str = 'classification',
    ) -> Dict[str, Any]:
        per_fold_results: Dict[str, Any] = {}
        prediction_frames = []
        protocol = 'group_kfold_subject'
        sequence_model = model_requires_sequences(model_config.get('type', ''))
        expected_predictions = 0

        for fold_name, outer_split in group_splits.items():
            split = (
                self._build_sequence_split(outer_split)
                if sequence_model
                else outer_split
            )
            group_overlap = split.metadata.get('group_overlap', [])
            subject_overlap = split.metadata.get('subject_overlap', [])
            if group_overlap or subject_overlap:
                raise RuntimeError(
                    f"Leakage detected before training {fold_name}: "
                    f"groups={group_overlap}, subjects={subject_overlap}"
                )
            split_model_config = deepcopy(model_config)
            if split.metadata.get('observation_unit') == 'raw_eeg_window':
                params = split_model_config.setdefault('params', {})
                target_rates = np.asarray(
                    split.row_metadata_train.get('sfreq_target'), dtype=float
                )
                params.setdefault('sampling_rate', float(np.median(target_rates)))
                params.setdefault(
                    'channel_names', list(split.feature_names or [])
                )
            model = self._create_model(
                split_model_config,
                input_shape=tuple(split.X_train.shape[1:]),
                num_outputs=num_outputs,
            )
            logger.info(
                f"      {model_name} {fold_name}: "
                f"train={len(split.y_train)}, test={len(split.y_test)}, "
                f"train_subjects={split.metadata['n_train_subjects']}, "
                f"test_subjects={split.metadata['n_test_subjects']}"
            )
            result = self._evaluate_split(
                model=model,
                split=split,
                model_name=model_name,
                dataset_name=dataset_name,
                task_name=task_name,
                artifact_split_name=fold_name,
                task_type=task_type,
            )
            per_fold_results[fold_name] = result
            expected_predictions += len(split.y_test)
            prediction_frames.append(
                pd.read_parquet(result['artifacts']['predictions'])
            )

        aggregated = self._aggregate_group_metrics(per_fold_results)
        protocol_dir = (
            self._model_artifact_dir(dataset_name, task_name, model_name)
            / protocol
        )
        protocol_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = protocol_dir / 'predictions.parquet'
        predictions = pd.concat(prediction_frames, ignore_index=True)
        identity_column = (
            'sequence_id' if 'sequence_id' in predictions.columns else 'sample_id'
        )
        if predictions[identity_column].duplicated().any():
            duplicates = predictions.loc[
                predictions[identity_column].duplicated(), identity_column
            ].head(20).tolist()
            raise RuntimeError(
                f"Observations appear in test more than once across folds: {duplicates}"
            )
        if len(predictions) != expected_predictions:
            raise RuntimeError(
                f"Unified predictions contain {len(predictions)} rows, "
                f"expected {expected_predictions}"
            )
        predictions.sort_values(['fold', identity_column]).to_parquet(
            predictions_path, index=False
        )
        return {
            'protocol': protocol,
            'n_folds': len(per_fold_results),
            'folds': per_fold_results,
            'aggregated': aggregated,
            'artifacts': {'predictions': str(predictions_path)},
        }

    @staticmethod
    def _aggregate_group_metrics(
            per_fold_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        metric_names = [
            'accuracy',
            'balanced_accuracy',
            'macro_f1',
            'weighted_f1',
            'kappa',
            'auc',
            'ordinal_mae',
            'adjacent_accuracy',
            'severe_error_rate',
            'mae',
            'rmse',
            'r2',
            'pearson',
            'spearman',
        ]
        aggregated: Dict[str, Any] = {
            'n_folds': len(per_fold_results),
            'training_time_total': float(sum(
                result['training_time']
                for result in per_fold_results.values()
            )),
        }
        for metric_name in metric_names:
            values = np.asarray([
                result['metrics'].get(metric_name, np.nan)
                for result in per_fold_results.values()
            ], dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values):
                aggregated[f'{metric_name}_mean'] = float(np.mean(finite_values))
                aggregated[f'{metric_name}_std'] = float(np.std(finite_values))
        for training_name in (
            'epochs_trained',
            'best_epoch',
            'best_validation_loss',
        ):
            values = np.asarray([
                result.get('training', {}).get(training_name, np.nan)
                for result in per_fold_results.values()
            ], dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values):
                aggregated[f'{training_name}_mean'] = float(np.mean(finite_values))
                aggregated[f'{training_name}_std'] = float(np.std(finite_values))
        return aggregated

    def _evaluate_loso(
            self,
            model: Optional[ModelLike],
            loso_splits: Dict[str, TaskSplit],
            model_name: str,
            model_config: Optional[Dict[str, Any]] = None,
            num_outputs: Optional[int] = None,
            dataset_name: Optional[str] = None,
            task_name: Optional[str] = None,
            task_type: str = 'classification',
    ) -> Dict[str, Any]:
        per_subject_results = {}

        for subject_id, split in loso_splits.items():
            fold_model = model
            if model_config is not None:
                fold_model = self._create_model(
                    model_config,
                    input_shape=tuple(split.X_train.shape[1:]),
                    num_outputs=num_outputs,
                )
            if fold_model is None:
                raise ValueError(f"Model '{model_name}' has not been initialized")
            result = self._evaluate_split(
                fold_model,
                split,
                model_name,
                dataset_name=dataset_name,
                task_name=task_name,
                artifact_split_name=f"loso_{subject_id}",
                task_type=task_type,
            )
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
                if self.config.get('evaluation'):
                    raise

        self.results = all_results
        self._save_results()

        return self.get_summary()

    def _save_results(self):
        output_file = self.result_file

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

        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        serializable_config = json.loads(json.dumps(
            self.config, default=convert_to_serializable
        ))
        config_path = run_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as output:
            yaml.safe_dump(serializable_config, output, sort_keys=False)
        metrics_path = run_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as output:
            json.dump(
                self.results,
                output,
                default=convert_to_serializable,
                indent=2,
            )

        logger.info(f"Results saved to {output_file}")
        self._save_csv_summary()
        manifest = {
            'schema_version': RUN_MANIFEST_SCHEMA_VERSION,
            'status': 'completed',
            'config_hash': self.config_hash,
            'benchmark_run_directory': str(run_dir),
            'benchmark_result_file': str(self.result_file),
            'benchmark_summary_file': str(self.summary_file),
            'config_file': str(config_path),
            'metrics_file': str(metrics_path),
            'timestamp': self.timestamp,
        }
        with open(run_dir / 'run_manifest.json', 'w', encoding='utf-8') as output:
            json.dump(manifest, output, indent=2)

    def _save_csv_summary(self):
        summary = self.get_summary()
        csv_file = self.summary_file
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
                            'evaluation': ws.get('split_metadata', {}).get(
                                'split_type', 'within_subject'
                            ),
                            'accuracy': ws['metrics']['accuracy'],
                            'balanced_accuracy': ws['metrics'].get(
                                'balanced_accuracy', np.nan
                            ),
                            'macro_f1': ws['metrics'].get('macro_f1', np.nan),
                            'weighted_f1': ws['metrics'].get(
                                'weighted_f1', ws['metrics']['f1_weighted']
                            ),
                            'f1_weighted': ws['metrics']['f1_weighted'],
                            'kappa': ws['metrics']['kappa'],
                            'training_time': ws['training_time'],
                            'n_train': ws['n_train'],
                            'n_test': ws['n_test']
                        })
                    if 'group_kfold_subject' in model_results:
                        group_result = model_results['group_kfold_subject']
                        aggregated = group_result.get('aggregated', {})
                        rows.append({
                            'dataset': dataset_name,
                            'task': task_name,
                            'model': model_name,
                            'evaluation': 'group_kfold_subject',
                            'n_folds': group_result.get('n_folds', 0),
                            'accuracy': aggregated.get('accuracy_mean', np.nan),
                            'accuracy_mean': aggregated.get('accuracy_mean', np.nan),
                            'accuracy_std': aggregated.get('accuracy_std', np.nan),
                            'balanced_accuracy': aggregated.get(
                                'balanced_accuracy_mean', np.nan
                            ),
                            'balanced_accuracy_mean': aggregated.get(
                                'balanced_accuracy_mean', np.nan
                            ),
                            'balanced_accuracy_std': aggregated.get(
                                'balanced_accuracy_std', np.nan
                            ),
                            'macro_f1': aggregated.get('macro_f1_mean', np.nan),
                            'macro_f1_mean': aggregated.get('macro_f1_mean', np.nan),
                            'macro_f1_std': aggregated.get('macro_f1_std', np.nan),
                            'weighted_f1': aggregated.get(
                                'weighted_f1_mean', np.nan
                            ),
                            'weighted_f1_mean': aggregated.get(
                                'weighted_f1_mean', np.nan
                            ),
                            'weighted_f1_std': aggregated.get(
                                'weighted_f1_std', np.nan
                            ),
                            'kappa': aggregated.get('kappa_mean', np.nan),
                            'kappa_mean': aggregated.get('kappa_mean', np.nan),
                            'kappa_std': aggregated.get('kappa_std', np.nan),
                            'auc': aggregated.get('auc_mean', np.nan),
                            'auc_mean': aggregated.get('auc_mean', np.nan),
                            'auc_std': aggregated.get('auc_std', np.nan),
                            'ordinal_mae': aggregated.get(
                                'ordinal_mae_mean', np.nan
                            ),
                            'ordinal_mae_mean': aggregated.get(
                                'ordinal_mae_mean', np.nan
                            ),
                            'ordinal_mae_std': aggregated.get(
                                'ordinal_mae_std', np.nan
                            ),
                            'adjacent_accuracy': aggregated.get(
                                'adjacent_accuracy_mean', np.nan
                            ),
                            'severe_error_rate': aggregated.get(
                                'severe_error_rate_mean', np.nan
                            ),
                            'mae': aggregated.get('mae_mean', np.nan),
                            'mae_mean': aggregated.get('mae_mean', np.nan),
                            'mae_std': aggregated.get('mae_std', np.nan),
                            'rmse': aggregated.get('rmse_mean', np.nan),
                            'rmse_mean': aggregated.get('rmse_mean', np.nan),
                            'rmse_std': aggregated.get('rmse_std', np.nan),
                            'r2': aggregated.get('r2_mean', np.nan),
                            'r2_mean': aggregated.get('r2_mean', np.nan),
                            'r2_std': aggregated.get('r2_std', np.nan),
                            'pearson': aggregated.get('pearson_mean', np.nan),
                            'pearson_mean': aggregated.get(
                                'pearson_mean', np.nan
                            ),
                            'pearson_std': aggregated.get(
                                'pearson_std', np.nan
                            ),
                            'spearman': aggregated.get('spearman_mean', np.nan),
                            'spearman_mean': aggregated.get(
                                'spearman_mean', np.nan
                            ),
                            'spearman_std': aggregated.get(
                                'spearman_std', np.nan
                            ),
                            'training_time': aggregated.get(
                                'training_time_total', np.nan
                            ),
                        })
                    if 'cross_source_holdout' in model_results:
                        cross_result = model_results['cross_source_holdout']
                        metrics = cross_result.get('metrics', {})
                        split_metadata = cross_result.get(
                            'split_metadata', {}
                        )
                        rows.append({
                            'dataset': dataset_name,
                            'task': task_name,
                            'model': model_name,
                            'evaluation': 'cross_source_holdout',
                            'train_source': split_metadata.get('train_source'),
                            'test_source': split_metadata.get('test_source'),
                            'subject_mode': split_metadata.get('subject_mode'),
                            'accuracy': metrics.get('accuracy', np.nan),
                            'balanced_accuracy': metrics.get(
                                'balanced_accuracy', np.nan
                            ),
                            'macro_f1': metrics.get('macro_f1', np.nan),
                            'weighted_f1': metrics.get(
                                'weighted_f1', np.nan
                            ),
                            'kappa': metrics.get('kappa', np.nan),
                            'auc': metrics.get('auc', np.nan),
                            'ordinal_mae': metrics.get(
                                'ordinal_mae', np.nan
                            ),
                            'severe_error_rate': metrics.get(
                                'severe_error_rate', np.nan
                            ),
                            'training_time': cross_result.get(
                                'training_time_total', np.nan
                            ),
                            'n_train': split_metadata.get('n_train_rows'),
                            'n_test': split_metadata.get('n_test_rows'),
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
