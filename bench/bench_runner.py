import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from .core.abstract_dataset import BaseDataset, EEGData
from .core.abstract_task import BaseTask, TaskSplit
from .datasets.datasets_registry import get_dataset
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
from model_zoo.DL.sequence_utils import build_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
                model_type = model_config.get('type')
                if not model_type:
                    raise ValueError("Model config must define a non-empty 'type'")
                group_protocol = self.config.get('evaluation', {}).get(
                    'protocol'
                ) == 'group_kfold_subject'
                if model_requires_data_shape(model_type) or group_protocol:
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
            cv = CrossValidator(task)
            subject_ids = np.unique(data.subject_ids)
            evaluation_config = self.config.get('evaluation')
            if evaluation_config:
                protocol = evaluation_config.get('protocol')
                if protocol != 'group_kfold_subject':
                    raise ValueError(
                        f"Unknown evaluation protocol {protocol!r}. "
                        "Available: ['group_kfold_subject']"
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
                        num_outputs=task.n_classes,
                        dataset_name=dataset_name,
                        task_name=task_name,
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
                            num_outputs=task.n_classes,
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
                            num_outputs=task.n_classes,
                            dataset_name=dataset_name,
                            task_name=task_name,
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
            artifact_split_name: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        self._configure_model_validation(model, split)
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

        predictions_path = artifact_dir / 'predictions.parquet'
        predictions.to_parquet(predictions_path, index=False)
        artifacts = {'predictions': str(predictions_path)}
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

        sequence_stats = split.metadata.get('sequence_stats')
        if sequence_stats is not None:
            sequence_stats_path = artifact_dir / 'sequence_stats.json'
            with open(sequence_stats_path, 'w', encoding='utf-8') as output:
                json.dump(sequence_stats, output, indent=2, default=_json_default)
            artifacts['sequence_stats'] = str(sequence_stats_path)

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
            if (
                split.metadata.get('observation_unit') == 'raw_eeg_window'
                and getattr(model, 'feature_mean_', None) is not None
            ):
                normalization_path = artifact_dir / 'normalization_stats.json'
                normalization = {
                    'scope': 'inner_train_only',
                    'channel_names': list(split.feature_names or []),
                    'mean': np.asarray(model.feature_mean_).tolist(),
                    'scale': np.asarray(model.feature_scale_).tolist(),
                }
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
        outer_record_overlap = np.intersect1d(
            np.unique(train_groups), np.unique(test_groups)
        )
        if len(outer_record_overlap):
            raise RuntimeError(
                "Outer train/test records overlap before inner validation: "
                f"{outer_record_overlap.astype(str).tolist()}"
            )
        model.set_validation_groups(
            train_groups,
            subject_ids=train_subjects,
            record_ids=train_groups,
            outer_test_record_ids=test_groups,
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
        if len(sequence_subject_overlap):
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

    def _evaluate_group_kfold(
            self,
            group_splits: Dict[str, TaskSplit],
            model_name: str,
            model_config: Dict[str, Any],
            num_outputs: int,
            dataset_name: str,
            task_name: str
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
            task_name: Optional[str] = None
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

        run_dir = self.output_dir / self.timestamp
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
                            'training_time': aggregated.get(
                                'training_time_total', np.nan
                            ),
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
