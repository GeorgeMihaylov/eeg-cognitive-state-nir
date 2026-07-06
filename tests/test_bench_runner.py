import pytest
import tempfile
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from bench.bench_runner import BenchmarkRunner
from bench.core.abstract_dataset import EEGData
from bench.core.abstract_task import TaskSplit

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def test_config(temp_dir):
    return {
        'output_dir': temp_dir,
        'datasets': {
            'test_dataset': {
                'data_path': '/path/to/data.parquet',
                'feature_set': 'pow_plus_eeg',
                'n_classes': 3,
                'discretize': True,
                'max_features': 100
            }
        },
        'tasks': ['cognitive_load_3class'],
        'models': {
            'test_model': {
                'type': 'random_forest',
                'params': {'n_estimators': 10, 'max_depth': 3}
            }
        },
        'task_config': {
            'test_size': 0.15,
            'random_state': 42
        },
        'run_within_subject': True,
        'run_loso': True
    }


@pytest.fixture
def test_data():
    return EEGData(
        data=np.random.randn(100, 20),
        labels=np.random.randint(0, 3, 100),
        subject_ids=np.array(['S1'] * 30 + ['S2'] * 30 + ['S3'] * 40),
        sampling_rate=128.0
    )


@pytest.fixture
def test_split():
    return TaskSplit(
        X_train=np.random.randn(70, 20),
        y_train=np.random.randint(0, 3, 70),
        X_test=np.random.randn(30, 20),
        y_test=np.random.randint(0, 3, 30),
        feature_names=[f'feature_{i}' for i in range(20)]
    )


@pytest.fixture
def benchmark_runner(test_config):
    return BenchmarkRunner(test_config)


def test_init(test_config, temp_dir):
    runner = BenchmarkRunner(test_config)
    assert runner.output_dir == Path(temp_dir)
    assert isinstance(runner.models, dict)
    assert 'test_model' in runner.models


def test_create_model_random_forest(benchmark_runner):
    model_config = {
        'type': 'random_forest',
        'params': {'n_estimators': 10}
    }
    model = benchmark_runner._create_model(model_config)
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


def test_create_model_svm(benchmark_runner):
    model_config = {
        'type': 'svm',
        'params': {'C': 1.0}
    }
    model = benchmark_runner._create_model(model_config)
    from sklearn.svm import SVC
    assert isinstance(model, SVC)


def test_create_model_unknown(benchmark_runner):
    model_config = {'type': 'unknown_model'}
    with pytest.raises(ValueError):
        benchmark_runner._create_model(model_config)


@patch('bench.bench_runner.get_dataset')
def test_load_dataset(mock_get_dataset, benchmark_runner, test_data, test_config):
    mock_dataset = Mock()
    mock_dataset.load.return_value = test_data
    mock_get_dataset.return_value = mock_dataset

    data = benchmark_runner.load_dataset('test_dataset')

    assert data.n_samples == 100
    assert data.n_features == 20
    assert data.n_subjects == 3
    assert data.n_classes == 3

    mock_get_dataset.assert_called_once_with('test_dataset', {
        'data_path': Path('/path/to/data.parquet'),
        'feature_set': 'pow_plus_eeg',
        'n_classes': 3,
        'discretize': True,
        'max_features': 100
    })


def test_evaluate_split(benchmark_runner, test_split):
    model = benchmark_runner.models['test_model']['model']

    result = benchmark_runner._evaluate_split(model, test_split, 'test_model')

    assert 'metrics' in result
    assert 'accuracy' in result['metrics']
    assert 'training_time' in result
    assert result['n_train'] == 70
    assert result['n_test'] == 30


def test_evaluate_loso(benchmark_runner, test_split):
    model = benchmark_runner.models['test_model']['model']

    loso_splits = {
        'S1': test_split,
        'S2': test_split,
        'S3': test_split
    }

    result = benchmark_runner._evaluate_loso(model, loso_splits, 'test_model')

    assert 'per_subject' in result
    assert 'aggregated' in result
    assert len(result['per_subject']) == 3
    assert 'accuracy_mean' in result['aggregated']


def test_aggregate_loso_metrics(benchmark_runner):
    per_subject_results = {
        'S1': {
            'metrics': {
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.8,
                'f1_weighted': 0.78,
                'kappa': 0.7
            }
        },
        'S2': {
            'metrics': {
                'accuracy': 0.9,
                'precision': 0.85,
                'recall': 0.9,
                'f1_weighted': 0.88,
                'kappa': 0.85
            }
        }
    }

    aggregated = benchmark_runner._aggregate_loso_metrics(per_subject_results)

    assert aggregated['n_subjects'] == 2
    assert aggregated['accuracy_mean'] == pytest.approx(0.85)
    assert aggregated['f1_weighted_mean'] == pytest.approx(0.83)


def test_get_summary(benchmark_runner):
    benchmark_runner.results = {
        'test_dataset': {
            'models': {
                'cognitive_load_3class': {
                    'test_model': {
                        'within_subject': {
                            'metrics': {
                                'accuracy': 0.85,
                                'f1_weighted': 0.84,
                                'kappa': 0.75
                            },
                            'training_time': 1.5,
                            'n_train': 70,
                            'n_test': 30
                        },
                        'loso': {
                            'aggregated': {
                                'accuracy_mean': 0.82,
                                'accuracy_std': 0.05,
                                'f1_weighted_mean': 0.81,
                                'kappa_mean': 0.72,
                                'n_subjects': 3
                            }
                        }
                    }
                }
            }
        }
    }

    summary = benchmark_runner.get_summary()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2
    assert 'accuracy' in summary.columns
    assert 'model' in summary.columns


@patch('bench.bench_runner.get_task')
@patch('bench.bench_runner.get_dataset')
def test_run(mock_get_dataset, mock_get_task, benchmark_runner, test_data, test_split):
    mock_dataset = Mock()
    mock_dataset.load.return_value = test_data
    mock_get_dataset.return_value = mock_dataset

    mock_task = Mock()
    mock_task.get_split.return_value = test_split
    mock_task.get_all_splits.return_value = {
        'S1': test_split,
        'S2': test_split,
        'S3': test_split
    }
    mock_get_task.return_value = mock_task

    summary = benchmark_runner.run()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0

    output_dir = Path(benchmark_runner.output_dir)
    json_files = list(output_dir.glob("benchmark_results_*.json"))
    csv_files = list(output_dir.glob("summary_*.csv"))

    assert len(json_files) > 0
    assert len(csv_files) > 0


@patch('bench.bench_runner.get_task')
@patch('bench.bench_runner.get_dataset')
def test_run_without_loso(mock_get_dataset, mock_get_task, test_config, test_data, test_split):
    test_config['run_loso'] = False
    runner = BenchmarkRunner(test_config)

    mock_dataset = Mock()
    mock_dataset.load.return_value = test_data
    mock_get_dataset.return_value = mock_dataset

    mock_task = Mock()
    mock_task.get_split.return_value = test_split
    mock_get_task.return_value = mock_task

    summary = runner.run()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0


@patch('bench.bench_runner.get_task')
@patch('bench.bench_runner.get_dataset')
def test_run_without_within_subject(mock_get_dataset, mock_get_task, test_config, test_data, test_split):
    test_config['run_within_subject'] = False
    runner = BenchmarkRunner(test_config)

    mock_dataset = Mock()
    mock_dataset.load.return_value = test_data
    mock_get_dataset.return_value = mock_dataset

    mock_task = Mock()
    mock_task.get_split.return_value = test_split
    mock_task.get_all_splits.return_value = {
        'S1': test_split,
        'S2': test_split,
        'S3': test_split
    }
    mock_get_task.return_value = mock_task

    summary = runner.run()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0


@patch('bench.bench_runner.get_dataset')
def test_load_dataset_error(mock_get_dataset, benchmark_runner):
    mock_get_dataset.side_effect = Exception("Failed to load dataset")

    with pytest.raises(Exception, match="Failed to load dataset"):
        benchmark_runner.load_dataset('test_dataset')


def test_save_results(benchmark_runner, temp_dir):
    benchmark_runner.results = {
        'test_dataset': {
            'models': {
                'cognitive_load_3class': {
                    'test_model': {
                        'within_subject': {
                            'metrics': {
                                'accuracy': 0.85,
                                'f1_weighted': 0.84,
                                'kappa': 0.75
                            },
                            'training_time': 1.5,
                            'n_train': 70,
                            'n_test': 30
                        }
                    }
                }
            }
        }
    }

    benchmark_runner._save_results()

    json_files = list(Path(temp_dir).glob("benchmark_results_*.json"))
    assert len(json_files) > 0

    csv_files = list(Path(temp_dir).glob("summary_*.csv"))
    assert len(csv_files) > 0


@pytest.mark.parametrize("model_type", [
    'random_forest',
    'svm',
])
def test_create_model_types(benchmark_runner, model_type):
    model_config = {
        'type': model_type,
        'params': {}
    }
    model = benchmark_runner._create_model(model_config)
    assert model is not None


def test_run_with_multiple_tasks(test_config, test_data, test_split):
    test_config['tasks'] = ['cognitive_load_3class', 'another_task']
    runner = BenchmarkRunner(test_config)

    with patch('bench.bench_runner.get_dataset') as mock_get_dataset:
        mock_dataset = Mock()
        mock_dataset.load.return_value = test_data
        mock_get_dataset.return_value = mock_dataset

        with patch('bench.bench_runner.get_task') as mock_get_task:
            mock_task = Mock()
            mock_task.get_split.return_value = test_split
            mock_task.get_all_splits.return_value = {
                'S1': test_split,
                'S2': test_split,
                'S3': test_split
            }
            mock_get_task.return_value = mock_task

            summary = runner.run()

            assert isinstance(summary, pd.DataFrame)
            assert mock_get_task.call_count == len(test_config['tasks'])


# Интеграционные тесты

@pytest.fixture
def integration_test_data(temp_dir):
    test_data_path = Path(temp_dir) / "test_data.parquet"

    n_samples = 200
    n_features = 20

    data = {
        'subject_id': ['S1'] * 50 + ['S2'] * 50 + ['S3'] * 50 + ['S4'] * 50,
        'target_main': np.random.randint(0, 3, n_samples),
    }

    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples) + np.random.randint(0, 3, n_samples) * 0.5

    df = pd.DataFrame(data)
    df.to_parquet(test_data_path)

    return test_data_path


@pytest.fixture
def integration_config(temp_dir, integration_test_data):
    return {
        'output_dir': temp_dir,
        'datasets': {
            'test_dataset': {
                'data_path': str(integration_test_data),
                'feature_set': 'all',
                'n_classes': 3,
                'discretize': True,
                'max_features': 100
            }
        },
        'tasks': ['cognitive_load_3class'],
        'models': {
            'random_forest': {
                'type': 'random_forest',
                'params': {'n_estimators': 10, 'max_depth': 3, 'random_state': 42}
            }
        },
        'task_config': {
            'test_size': 0.15,
            'random_state': 42
        },
        'run_within_subject': True,
        'run_loso': True
    }


@pytest.mark.integration
def test_end_to_end_integration(integration_config, temp_dir):
    try:
        import pyarrow
    except ImportError:
        pytest.skip("pyarrow not installed")

    runner = BenchmarkRunner(integration_config)
    summary = runner.run()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0

    output_dir = Path(temp_dir)
    json_files = list(output_dir.glob("benchmark_results_*.json"))
    csv_files = list(output_dir.glob("summary_*.csv"))

    assert len(json_files) > 0
    assert len(csv_files) > 0

    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            assert 'test_dataset' in data


@pytest.mark.integration
def test_integration_with_multiple_models(integration_config, temp_dir):
    try:
        import pyarrow
    except ImportError:
        pytest.skip("pyarrow not installed")

    integration_config['models'] = {
        'random_forest': {
            'type': 'random_forest',
            'params': {'n_estimators': 5, 'max_depth': 2, 'random_state': 42}
        },
        'svm': {
            'type': 'svm',
            'params': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
        }
    }

    runner = BenchmarkRunner(integration_config)
    summary = runner.run()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0

    models_in_results = summary['model'].unique()
    assert 'random_forest' in models_in_results
    assert 'svm' in models_in_results


@pytest.mark.integration
def test_integration_with_custom_output_dir(integration_config, temp_dir):
    try:
        import pyarrow
    except ImportError:
        pytest.skip("pyarrow not installed")

    custom_dir = Path(temp_dir) / "custom_output"
    integration_config['output_dir'] = str(custom_dir)

    runner = BenchmarkRunner(integration_config)
    summary = runner.run()

    assert custom_dir.exists()

    json_files = list(custom_dir.glob("benchmark_results_*.json"))
    csv_files = list(custom_dir.glob("summary_*.csv"))

    assert len(json_files) > 0
    assert len(csv_files) > 0


def test_get_summary_empty(benchmark_runner):
    benchmark_runner.results = {}
    summary = benchmark_runner.get_summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0


def test_get_summary_with_missing_keys(benchmark_runner):
    benchmark_runner.results = {
        'test_dataset': {
            'models': {}
        }
    }
    summary = benchmark_runner.get_summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0


def test_evaluate_split_with_probabilities(benchmark_runner, test_split):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(test_split.X_train, test_split.y_train)

    result = benchmark_runner._evaluate_split(model, test_split, 'logistic_regression')

    assert 'metrics' in result
    assert 'auc' in result['metrics']


@patch('bench.bench_runner.get_task')
@patch('bench.bench_runner.get_dataset')
def test_run_with_error_in_model(mock_get_dataset, mock_get_task, test_config, test_data, test_split):
    test_config['models'] = {
        'failing_model': {
            'type': 'random_forest',
            'params': {'n_estimators': -1}
        }
    }

    runner = BenchmarkRunner(test_config)

    mock_dataset = Mock()
    mock_dataset.load.return_value = test_data
    mock_get_dataset.return_value = mock_dataset

    mock_task = Mock()
    mock_task.get_split.return_value = test_split
    mock_task.get_all_splits.return_value = {
        'S1': test_split,
        'S2': test_split,
        'S3': test_split
    }
    mock_get_task.return_value = mock_task

    summary = runner.run()
    assert isinstance(summary, pd.DataFrame)
