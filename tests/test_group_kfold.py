import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_dataset import EEGData
from bench.tasks.cognitive_load import CognitiveLoadTask
from bench.validation.cross_val import CrossValidator
from model_zoo import build_model
from model_zoo.DL import TorchClassificationAdapter


@pytest.fixture
def grouped_data() -> EEGData:
    n_subjects = 10
    samples_per_subject = 6
    n_samples = n_subjects * samples_per_subject
    sample_ids = np.arange(1000, 1000 + n_samples, dtype=np.int64)
    subject_ids = np.repeat(
        [f"S{index:02d}" for index in range(n_subjects)],
        samples_per_subject,
    )
    record_ids = np.asarray([
        f"{subject}_R{within_subject // 3}"
        for subject in subject_ids[::samples_per_subject]
        for within_subject in range(samples_per_subject)
    ])
    labels = np.tile(np.arange(3), n_samples // 3)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(n_samples, 6)).astype(np.float32)
    features[:, 0] = sample_ids
    features[:, 1] += labels
    return EEGData(
        data=features,
        labels=labels,
        subject_ids=subject_ids,
        sample_ids=sample_ids,
        record_ids=record_ids,
        row_metadata={
            'source': np.full(n_samples, 'synthetic', dtype=object),
            't_start': np.tile(np.arange(3, dtype=float), n_samples // 3),
        },
    )


@pytest.fixture
def group_splits(grouped_data: EEGData):
    task = CognitiveLoadTask(grouped_data, {'random_state': 42, 'n_splits': 5})
    return CrossValidator(task).run_group_kfold(
        group_column='subject_id',
        n_splits=5,
        random_state=42,
    )


def test_group_kfold_subject_isolation_and_coverage(
    group_splits,
    grouped_data: EEGData,
) -> None:
    assert len(group_splits) == 5
    all_test_samples = []
    for split in group_splits.values():
        train_subjects = set(split.subject_train)
        test_subjects = set(split.subject_test)
        assert train_subjects.isdisjoint(test_subjects)
        assert split.metadata['group_overlap'] == []
        assert split.metadata['subject_overlap'] == []
        assert split.metadata['n_train_subjects'] == len(train_subjects)
        assert split.metadata['n_test_subjects'] == len(test_subjects)
        assert split.metadata['n_train_records'] == len(set(split.record_id_train))
        assert split.metadata['n_test_records'] == len(set(split.record_id_test))
        all_test_samples.extend(split.sample_id_test.tolist())

    unique, counts = np.unique(all_test_samples, return_counts=True)
    np.testing.assert_array_equal(np.sort(unique), np.sort(grouped_data.sample_ids))
    np.testing.assert_array_equal(counts, np.ones_like(counts))


def test_group_kfold_rejects_missing_group_column(grouped_data: EEGData) -> None:
    task = CognitiveLoadTask(grouped_data, {'random_state': 42})

    with pytest.raises(ValueError, match="Row metadata column"):
        CrossValidator(task).run_group_kfold('missing_group', n_splits=5)


def test_torch_standardizer_only_sees_outer_train(
    group_splits,
    monkeypatch,
) -> None:
    split = group_splits['fold_01']
    captured_sample_ids = []
    original = TorchClassificationAdapter._fit_standardizer

    def capture_train_rows(self, X_train):
        captured_sample_ids.extend(X_train[:, 0].astype(np.int64).tolist())
        return original(self, X_train)

    monkeypatch.setattr(
        TorchClassificationAdapter,
        '_fit_standardizer',
        capture_train_rows,
    )
    model = build_model(
        'torch_mlp',
        'classification',
        input_shape=(split.X_train.shape[1],),
        num_outputs=3,
        params={
            'hidden_dims': [8],
            'batch_size': 16,
            'max_epochs': 1,
            'validation_size': 0.2,
            'early_stopping_patience': 1,
            'device': 'cpu',
            'random_state': 42,
        },
    )

    model.fit(split.X_train, split.y_train)

    assert set(captured_sample_ids).issubset(set(split.sample_id_train))
    assert set(captured_sample_ids).isdisjoint(set(split.sample_id_test))


@patch('bench.bench_runner.get_dataset')
def test_runner_group_kfold_shared_indices_and_artifacts(
    mock_get_dataset,
    grouped_data: EEGData,
    tmp_path,
) -> None:
    dataset = Mock()
    dataset.load.return_value = grouped_data
    mock_get_dataset.return_value = dataset
    config = {
        'output_dir': str(tmp_path),
        'datasets': {'synthetic': {'data_path': 'unused.parquet'}},
        'tasks': ['cognitive_load_3class'],
        'models': {
            'random_forest': {
                'type': 'random_forest',
                'task_type': 'classification',
                'params': {'n_estimators': 3, 'random_state': 42},
            },
            'torch_mlp': {
                'type': 'torch_mlp',
                'task_type': 'classification',
                'params': {
                    'hidden_dims': [8],
                    'batch_size': 16,
                    'max_epochs': 1,
                    'validation_size': 0.2,
                    'early_stopping_patience': 1,
                    'device': 'cpu',
                    'random_state': 42,
                },
            },
            'torch_lstm': {
                'type': 'torch_lstm',
                'task_type': 'classification',
                'params': {
                    'hidden_size': 8,
                    'classifier_hidden': 6,
                    'batch_size': 16,
                    'max_epochs': 1,
                    'validation_size': 0.2,
                    'early_stopping_patience': 1,
                    'device': 'cpu',
                    'random_state': 42,
                },
            },
        },
        'sequence': {
            'length': 2,
            'stride': 1,
            'target_position': 'last',
        },
        'validation': {
            'strategy': 'group_record',
            'group_column': 'record_id',
            'validation_size': 0.2,
            'random_state': 42,
        },
        'task_config': {'random_state': 42},
        'evaluation': {
            'protocol': 'group_kfold_subject',
            'n_splits': 5,
            'group_column': 'subject_id',
            'random_state': 42,
        },
    }
    runner = BenchmarkRunner(config)

    with patch.object(
        runner,
        '_create_model',
        wraps=runner._create_model,
    ) as create_model:
        summary = runner.run()

    assert create_model.call_count == 15
    assert len(summary) == 3
    required_summary = {
        'accuracy_mean', 'accuracy_std',
        'balanced_accuracy_mean', 'balanced_accuracy_std',
        'macro_f1_mean', 'macro_f1_std',
        'weighted_f1_mean', 'weighted_f1_std',
        'kappa_mean', 'kappa_std',
    }
    assert required_summary.issubset(summary.columns)

    model_predictions = {}
    for model_name in ('random_forest', 'torch_mlp'):
        result = runner.results['synthetic']['models'][
            'cognitive_load_3class'
        ][model_name]['group_kfold_subject']
        assert result['n_folds'] == 5
        predictions = pd.read_parquet(result['artifacts']['predictions'])
        required_columns = {
            'protocol', 'fold', 'sample_index', 'sample_id',
            'subject_id', 'record_id', 'y_true', 'y_pred',
        }
        assert required_columns.issubset(predictions.columns)
        assert predictions['sample_id'].is_unique
        model_predictions[model_name] = predictions[
            ['sample_id', 'fold']
        ].sort_values('sample_id').reset_index(drop=True)

    pd.testing.assert_frame_equal(
        model_predictions['random_forest'],
        model_predictions['torch_mlp'],
    )
    lstm_result = runner.results['synthetic']['models'][
        'cognitive_load_3class'
    ]['torch_lstm']['group_kfold_subject']
    lstm_predictions = pd.read_parquet(lstm_result['artifacts']['predictions'])
    assert {
        'sequence_id', 'source', 'subject_id', 'record_id', 'fold',
        'sequence_length', 'sequence_start_sample_id',
        'sequence_end_sample_id', 'target_sample_id', 'target_time',
    }.issubset(lstm_predictions.columns)
    assert lstm_predictions['sequence_id'].is_unique
    for fold_name in lstm_result['folds']:
        reference_subjects = runner.results['synthetic']['models'][
            'cognitive_load_3class'
        ]['random_forest']['group_kfold_subject']['folds'][fold_name][
            'split_metadata'
        ]['test_subject_ids']
        lstm_subjects = lstm_result['folds'][fold_name][
            'split_metadata'
        ]['test_subject_ids']
        assert lstm_subjects == reference_subjects
        validation_path = lstm_result['folds'][fold_name]['artifacts'][
            'validation_split'
        ]
        with open(validation_path, encoding='utf-8') as input_file:
            validation_split = json.load(input_file)
        assert validation_split['record_overlap'] == []
        assert validation_split['outer_test_record_overlap'] == []
        assert set(validation_split['inner_train_record_ids']).isdisjoint(
            validation_split['inner_validation_record_ids']
        )
    assert len(list(Path(tmp_path).rglob('model.pt'))) == 10
    assert len(list(Path(tmp_path).rglob('training_log.csv'))) == 10
    assert len(list(Path(tmp_path).rglob('sequence_stats.json'))) == 5
    assert len(list(Path(tmp_path).rglob('validation_split.json'))) == 10


def test_random_window_protocol_remains_available(grouped_data: EEGData) -> None:
    task = CognitiveLoadTask(grouped_data, {'random_state': 42, 'n_splits': 5})

    split = task.get_split()

    assert split.metadata['split_type'] == (
        'random_window_stratified_kfold_first_fold'
    )
    assert split.sample_id_test is not None
    assert split.subject_test is not None


def test_inner_validation_keeps_logical_recordings_disjoint() -> None:
    logical_groups = np.repeat([f"logical_{index:02d}" for index in range(20)], 4)
    source_records = np.asarray([
        f"{'gpn_data' if row % 2 == 0 else 'Old_EEG'}__{logical_group}"
        for row, logical_group in enumerate(logical_groups)
    ])
    labels = np.tile(np.arange(4), 20)
    subjects = np.repeat([f"S{index:02d}" for index in range(10)], 8)
    model = build_model(
        'torch_mlp',
        'classification',
        input_shape=(3,),
        num_outputs=4,
        params={
            'hidden_dims': [4],
            'batch_size': 8,
            'max_epochs': 1,
            'validation_size': 0.2,
            'early_stopping_patience': 1,
            'device': 'cpu',
            'random_state': 42,
        },
    )
    model.set_validation_groups(
        logical_groups,
        subject_ids=subjects,
        record_ids=source_records,
        strategy='group_record',
        group_column='record_group_id',
    )

    train_idx, validation_idx = model._validation_indices(labels)

    assert set(logical_groups[train_idx]).isdisjoint(logical_groups[validation_idx])
