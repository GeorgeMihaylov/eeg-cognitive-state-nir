import argparse
import json
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from bench.bench_runner import BenchmarkRunner
from bench.tasks.tasks_registry import TASK_REGISTRY
from bench.tasks.target_registry import get_target_spec
from model_zoo.DL.feature_preprocessing import (
    SUPPORTED_FEATURE_SCALING_STRATEGIES,
)
from model_zoo.factory import SKLEARN_MODEL_NAMES, TORCH_MODEL_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)


def create_default_config() -> Dict[str, Any]:

    return {
        'output_dir': './benchmark_results',
        'datasets': {
            'emotiv_cognitive': {
                'data_path': './data/processed/windowed_eeg_pm_dataset_w10.parquet',
                'feature_set': 'pow_plus_eeg',
                'target_col': 'label_q5',
                'n_classes': 5,
                'discretize': False,
                'max_features': 500
            }
        },
        'tasks': ['cognitive_load_5class'],
        'models': {
            'random_forest': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'svm': {
                'type': 'svm',
                'params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'random_state': 42
                }
            }
        },
        'task_config': {
            'test_size': 0.15,
            'random_state': 42,
            'n_splits': 5
        },
        'run_within_subject': True,
        'run_loso': True
    }


def validate_config(config: Dict[str, Any]) -> bool:
    # Validate the canonical benchmark configuration before loading data.
    errors = []

    if not isinstance(config, dict):
        logger.error("Configuration validation failed:")
        logger.error(" - Config root must be a mapping")
        return False

    datasets = config.get('datasets')
    models = config.get('models')
    tasks = config.get('tasks')

    if not isinstance(datasets, dict) or not datasets:
        errors.append("No datasets specified in config")
        datasets = {}

    if not isinstance(models, dict) or not models:
        errors.append("No models specified in config")
        models = {}

    if not isinstance(tasks, list) or not tasks:
        errors.append("No tasks specified in config")
        tasks = []
    elif not all(isinstance(task, str) and task.strip() for task in tasks):
        errors.append("Config 'tasks' must be a list of non-empty strings")

    known_task_names = set(TASK_REGISTRY)
    valid_task_names = []
    for task_name in tasks:
        if not isinstance(task_name, str) or not task_name.strip():
            continue
        if task_name not in known_task_names:
            errors.append(
                f"Unknown task '{task_name}'. "
                f"Available: {sorted(known_task_names)}"
            )
        else:
            valid_task_names.append(task_name)

    task_types = {
        str(
            getattr(TASK_REGISTRY[task_name], 'task_type', 'classification')
        ).strip().lower()
        for task_name in valid_task_names
    }
    if len(task_types) > 1:
        errors.append(
            "A canonical benchmark config cannot mix classification and "
            "regression tasks. Use separate experiment configs."
        )
    expected_task_type = next(iter(task_types), None)

    known_model_types = set(SKLEARN_MODEL_NAMES) | set(TORCH_MODEL_NAMES)

    def validate_feature_scaling(
        scaling_config: Any,
        *,
        location: str,
    ) -> None:
        if not isinstance(scaling_config, dict):
            errors.append(f"{location} must be a mapping")
            return
        strategy = str(
            scaling_config.get('strategy', 'standard')
        ).strip().lower()
        if strategy not in SUPPORTED_FEATURE_SCALING_STRATEGIES:
            errors.append(
                f"{location}.strategy '{strategy}' is unknown. Available: "
                f"{sorted(SUPPORTED_FEATURE_SCALING_STRATEGIES)}"
            )
        for key in ('quantile_range', 'clip_percentiles'):
            if key not in scaling_config:
                continue
            values = scaling_config[key]
            try:
                low, high = (float(value) for value in values)
            except (TypeError, ValueError):
                errors.append(f"{location}.{key} must contain two numbers")
                continue
            if not 0 <= low < high <= 100:
                errors.append(
                    f"{location}.{key} must satisfy "
                    "0 <= low < high <= 100"
                )
        if 'scale_floor' in scaling_config:
            try:
                if float(scaling_config['scale_floor']) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(
                    f"{location}.scale_floor must be positive"
                )

    if 'feature_scaling' in config:
        validate_feature_scaling(
            config['feature_scaling'],
            location='feature_scaling',
        )

    for model_alias, model_config in models.items():
        if not isinstance(model_alias, str) or not model_alias.strip():
            errors.append("Model aliases must be non-empty strings")
            continue
        if not isinstance(model_config, dict):
            errors.append(f"Model '{model_alias}' config must be a mapping")
            continue

        model_type = str(model_config.get('type', '')).strip().lower()
        if not model_type:
            errors.append(f"Model '{model_alias}' missing non-empty 'type'")
            continue
        if model_type not in known_model_types:
            errors.append(
                f"Unknown model type '{model_type}' for '{model_alias}'. "
                f"Available: {sorted(known_model_types)}"
            )
            continue

        declared_task_type = str(
            model_config.get('task_type', expected_task_type or 'classification')
        ).strip().lower()
        normalized_task_type = {
            'classifier': 'classification',
            'regressor': 'regression',
        }.get(declared_task_type, declared_task_type)

        if normalized_task_type not in {'classification', 'regression'}:
            errors.append(
                f"Model '{model_alias}' has unsupported task_type "
                f"'{declared_task_type}'"
            )
            continue

        if (
            expected_task_type is not None
            and normalized_task_type != expected_task_type
        ):
            errors.append(
                f"Model '{model_alias}' task_type '{normalized_task_type}' "
                f"does not match configured task type '{expected_task_type}'"
            )

        if (
            model_type in TORCH_MODEL_NAMES
            and normalized_task_type == 'regression'
            and model_type != 'torch_mlp'
        ):
            errors.append(
                f"Model '{model_alias}' uses '{model_type}', which does not "
                "support regression yet"
            )
        if model_type == 'mean_regressor' and normalized_task_type != 'regression':
            errors.append(
                f"Model '{model_alias}' uses mean_regressor, which is "
                'regression-only'
            )
        if 'feature_scaling' in model_config:
            if model_type not in TORCH_MODEL_NAMES:
                errors.append(
                    f"Model '{model_alias}' cannot configure feature_scaling "
                    "because it is not a Torch model"
                )
            validate_feature_scaling(
                model_config['feature_scaling'],
                location=f"models.{model_alias}.feature_scaling",
            )

    for dataset_name, dataset_config in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            errors.append("Dataset names must be non-empty strings")
            continue
        if not isinstance(dataset_config, dict):
            errors.append(f"Dataset '{dataset_name}' config must be a mapping")
            continue

        data_path = dataset_config.get('data_path')
        if data_path in (None, ''):
            errors.append(f"Dataset '{dataset_name}' missing non-empty 'data_path'")
            continue
        if not Path(data_path).exists():
            errors.append(
                f"Dataset '{dataset_name}' data path not found: {data_path}"
            )
        has_target_col = 'target_col' in dataset_config
        has_target_cols = 'target_cols' in dataset_config
        has_target_id = 'target_id' in dataset_config
        if sum((has_target_id, has_target_col, has_target_cols)) > 1:
            errors.append(
                f"Dataset '{dataset_name}' must define only one of target_id, "
                'target_col, or target_cols'
            )
        target_spec = None
        if has_target_id:
            try:
                target_spec = get_target_spec(str(dataset_config['target_id']))
            except ValueError as exc:
                errors.append(f"Dataset '{dataset_name}': {exc}")
        target_cols = dataset_config.get('target_cols')
        if has_target_cols and (
            not isinstance(target_cols, list)
            or not target_cols
            or not all(
                isinstance(column, str) and column.strip()
                for column in target_cols
            )
        ):
            errors.append(
                f"Dataset '{dataset_name}' target_cols must be a non-empty "
                'list of strings'
            )
        elif has_target_cols and len(set(target_cols)) != len(target_cols):
            errors.append(
                f"Dataset '{dataset_name}' target_cols must be unique"
            )
        if expected_task_type == 'classification' and (
            has_target_cols
            or (target_spec is not None and not target_spec.is_classification)
        ):
            errors.append(
                f"Classification dataset '{dataset_name}' has a non-classification target"
            )
        if 'performance_metrics_regression' in valid_task_names:
            canonical_multioutput = (
                target_spec is not None
                and target_spec.target_id == 'pm_multioutput_regression_7'
            )
            if not has_target_cols and not canonical_multioutput:
                errors.append(
                    f"Dataset '{dataset_name}' must define target_cols or "
                    "target_id: pm_multioutput_regression_7 for "
                    'performance_metrics_regression'
                )
            if not canonical_multioutput and dataset_config.get('discretize', True) is not False:
                errors.append(
                    f"Dataset '{dataset_name}' must set discretize: false for "
                    'performance_metrics_regression'
                )
            configured_outputs = dataset_config.get(
                'n_outputs',
                config.get('task_config', {}).get('n_outputs'),
            )
            if (
                isinstance(target_cols, list)
                and configured_outputs is not None
                and int(configured_outputs) != len(target_cols)
            ):
                errors.append(
                    f"Dataset '{dataset_name}' n_outputs={configured_outputs} "
                    f"does not match {len(target_cols)} target_cols"
                )

    validation_config = config.get('validation')
    if validation_config is not None:
        if not isinstance(validation_config, dict):
            errors.append("Config 'validation' must be a mapping")
        else:
            validation_strategy = str(
                validation_config.get('strategy', 'group_holdout')
            ).strip().lower()
            supported_validation_strategies = {
                'group_holdout',
                'random_holdout',
                'group_record',
            }
            if validation_strategy not in supported_validation_strategies:
                errors.append(
                    f"Unknown validation strategy '{validation_strategy}'. "
                    f"Available: {sorted(supported_validation_strategies)}"
                )
            if (
                validation_strategy in {'group_holdout', 'group_record'}
                and not str(
                    validation_config.get('group_column', '')
                ).strip()
            ):
                errors.append(
                    f"validation.group_column is required for "
                    f"{validation_strategy}"
                )
            validation_fraction = validation_config.get(
                'fraction',
                validation_config.get('validation_size', 0.15),
            )
            try:
                validation_fraction = float(validation_fraction)
                if not 0 < validation_fraction < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(
                    "validation.fraction/validation_size must be between 0 and 1"
                )
            try:
                int(validation_config.get('random_state', 42))
            except (TypeError, ValueError):
                errors.append("validation.random_state must be an integer")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f" - {error}")
        return False

    return True


def override_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:

    if args.output_dir:
        config['output_dir'] = args.output_dir
        logger.info(f"Output directory overridden: {args.output_dir}")

    if args.dataset:
        if args.dataset in config['datasets']:
            config['datasets'] = {args.dataset: config['datasets'][args.dataset]}
            logger.info(f"Running only dataset: {args.dataset}")
        else:
            logger.error(f"Dataset '{args.dataset}' not found in config")
            sys.exit(1)

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        config['models'] = {
            name: config['models'][name]
            for name in model_names
            if name in config['models']
        }
        logger.info(f"Running only models: {', '.join(config['models'].keys())}")

    if args.task:
        if args.task in config['tasks']:
            config['tasks'] = [args.task]
            logger.info(f"Running only task: {args.task}")
        else:
            logger.error(f"Task '{args.task}' not found in config")
            sys.exit(1)

    if args.no_loso:
        config['run_loso'] = False
        logger.info("LOSO evaluation disabled")

    if args.no_within:
        config['run_within_subject'] = False
        logger.info("Within-subject evaluation disabled")

    if args.feature_set:
        for dataset in config['datasets'].values():
            dataset['feature_set'] = args.feature_set
        logger.info(f"Feature set overridden: {args.feature_set}")

    seed = getattr(args, 'seed', None)
    if seed is not None:
        seed = int(seed)
        for model in config['models'].values():
            params = model.setdefault('params', {})
            if (
                str(model.get('type', '')).startswith('torch_')
                or 'random_state' in params
            ):
                params['random_state'] = seed
        config.setdefault('evaluation', {})['random_state'] = seed
        config.setdefault('task_config', {})['random_state'] = seed
        if 'validation' in config:
            config['validation']['random_state'] = seed
        logger.info(f"Random seed overridden: {seed}")

    fold_limit = getattr(args, 'fold_limit', None)
    if fold_limit is not None:
        fold_limit = int(fold_limit)
        if fold_limit <= 0:
            raise ValueError('--fold-limit must be positive')
        evaluation = config.get('evaluation')
        if not evaluation:
            raise ValueError('--fold-limit requires an evaluation section')
        n_splits = int(evaluation.get('n_splits', 5))
        if fold_limit > n_splits:
            raise ValueError(
                f'--fold-limit={fold_limit} exceeds n_splits={n_splits}'
            )
        evaluation['folds'] = list(range(1, fold_limit + 1))
        logger.info(f"Evaluation limited to {fold_limit} fold(s)")

    max_windows = getattr(args, 'max_windows', None)
    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows <= 0:
            raise ValueError('--max-windows must be positive')
        for dataset in config['datasets'].values():
            dataset['max_windows'] = max_windows
        logger.info(f"Datasets limited to {max_windows} windows")

    max_epochs = getattr(args, 'max_epochs', None)
    if max_epochs is not None:
        max_epochs = int(max_epochs)
        if max_epochs <= 0:
            raise ValueError('--max-epochs must be positive')
        updated = 0
        for model in config['models'].values():
            if str(model.get('type', '')).startswith('torch_'):
                model.setdefault('params', {})['max_epochs'] = max_epochs
                updated += 1
        if not updated:
            raise ValueError('--max-epochs requires at least one torch model')
        logger.info(f"PyTorch models limited to {max_epochs} epochs")

    return config


def _parse_trial_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip().upper() for item in value.split(',') if item.strip()]


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run EEG benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m bench.cli --config configs.yaml

  # Run with default config (for testing)
  python -m bench.cli --test

  # Run only specific dataset and models
  python -m bench.cli --config configs.yaml --dataset emotiv_cognitive --models random_forest,svm

  # Run with custom output directory
  python -m bench.cli --config configs.yaml --output-dir ./my_results

  # Disable LOSO evaluation
  python -m bench.cli --config configs.yaml --no-loso
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (YAML)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run with default test configuration'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (run only this dataset)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated model names (run only these models)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        help='Task name (default: all tasks from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['pow_plus_eeg', 'pow', 'eeg', 'all'],
        help='Feature set to use (default: from config)'
    )
    
    parser.add_argument(
        '--no-loso',
        action='store_true',
        help='Disable LOSO evaluation'
    )
    
    parser.add_argument(
        '--no-within',
        action='store_true',
        help='Disable within-subject evaluation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    parser.add_argument(
        '--experiment-matrix',
        type=str,
        help='Resolve and run a preprocessing experiment matrix'
    )
    parser.add_argument('--trial-ids', help='Comma-separated matrix trial labels')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--build-missing-caches', action='store_true')
    parser.add_argument('--cache-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument(
        '--dry-execution',
        action='store_true',
        help='Resolve execution costs and checkpoint reuse without training',
    )
    parser.add_argument('--fold-limit', type=int)
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument(
        '--calibration-experiment',
        type=str,
        help='Run subject calibration from an existing benchmark run',
    )
    parser.add_argument(
        '--personalization-calibration',
        type=str,
        help='Plan the unified seven-PM leakage-safe personalization protocol',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        help='Runtime-only data root for protocol planning',
    )
    parser.add_argument('--pm', type=str)
    parser.add_argument(
        '--task-type',
        choices=['classification', 'regression'],
    )
    parser.add_argument(
        '--calibration-mode',
        choices=['zero_shot', 'head_only', 'full_model'],
    )
    parser.add_argument('--subject-limit', type=int)
    parser.add_argument(
        '--calibration-budgets',
        help='Comma-separated calibration budgets in seconds',
    )
    parser.add_argument(
        '--calibration-methods',
        help='Comma-separated calibration methods',
    )
    parser.add_argument('--max-calibration-epochs', type=int)
    parser.add_argument(
        '--calibration-budget-fraction',
        type=float,
        help='Select one personalization calibration fraction',
    )
    parser.add_argument(
        '--device', choices=['auto', 'cpu', 'cuda'],
        help='Runtime device override for personalization execution',
    )
    parser.add_argument(
        '--execution-model',
        type=str,
        help=(
            'Execution-only personalization model filter; preserves the full '
            'scientific plan and all condition identities'
        ),
    )
    parser.add_argument(
        '--automl-study',
        type=str,
        help='Run a nested AutoML study through the canonical benchmark',
    )
    parser.add_argument('--study-name', type=str)
    parser.add_argument('--storage', type=str)
    parser.add_argument('--outer-fold', type=int)
    parser.add_argument('--n-trials', type=int)
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--inner-splits', type=int)
    parser.add_argument('--no-evaluate-best', action='store_true')
    parser.add_argument(
        '--statistical-analysis',
        type=str,
        help='Analyze canonical completed runs without training models',
    )
    parser.add_argument(
        '--label-target-audit',
        type=str,
        help='Audit target provenance and structure without training models',
    )
    parser.add_argument(
        '--temporal-target-audit',
        type=str,
        help='Audit temporal target structure with non-EEG diagnostics',
    )
    parser.add_argument(
        '--label-definition-sensitivity',
        type=str,
        help='Compare global and outer-train-fitted label thresholds',
    )
    parser.add_argument(
        '--feature-group-experiment',
        type=str,
        help='Plan or run RF EEG/POW feature-group classification and regression',
    )
    parser.add_argument(
        '--ordinal-transformer-experiment',
        type=str,
        help='Plan or run the one-fold ordinal Transformer technical smoke',
    )
    parser.add_argument(
        '--ordinal-transformer-analysis',
        type=str,
        help='Plan or run paired subject-level ordinal Transformer statistics',
    )
    parser.add_argument(
        '--feature-groups',
        help='Comma-separated feature groups for the RF experiment',
    )
    parser.add_argument(
        '--tasks',
        help='Comma-separated task families for the RF experiment',
    )
    parser.add_argument(
        '--tracks',
        help='Comma-separated statistical analysis tracks',
    )
    parser.add_argument('--bootstrap-samples', type=int)
    parser.add_argument(
        '--cross-source-experiment',
        type=str,
        help='Plan or run strict directional cross-source transfer',
    )
    parser.add_argument(
        '--directions',
        help='Comma-separated train->test source directions',
    )
    parser.add_argument(
        '--subject-modes',
        help='Comma-separated cross-source subject modes',
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Execute valid cross-source trials after planning',
    )
    parser.add_argument('--max-train-windows', type=int)
    parser.add_argument('--max-test-windows', type=int)
    
    args = parser.parse_args(argv)

    if args.execution_model and not args.personalization_calibration:
        parser.error('--execution-model requires --personalization-calibration')

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if args.personalization_calibration:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--feature-group-experiment': args.feature_group_experiment,
            '--ordinal-transformer-experiment': args.ordinal_transformer_experiment,
            '--ordinal-transformer-analysis': args.ordinal_transformer_analysis,
            '--label-target-audit': args.label_target_audit,
            '--temporal-target-audit': args.temporal_target_audit,
            '--label-definition-sensitivity': args.label_definition_sensitivity,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--personalization-calibration cannot be combined with '
                + ', '.join(active_conflicts)
            )
        selected_modes = sum(bool(value) for value in (
            args.plan_only, args.dry_execution, args.run
        ))
        if selected_modes != 1:
            parser.error(
                '--personalization-calibration requires exactly one of '
                '--plan-only, --dry-execution, or --run'
            )
        if args.execution_model and args.plan_only:
            parser.error(
                '--execution-model is execution-only and cannot be used with '
                '--plan-only'
            )
        from bench.experiments.personalization_calibration import (
            PersonalizationCalibrationPlanner,
            PlanFilters,
        )
        from bench.experiments.personalization_calibration_execution import (
            PersonalizationCalibrationExecutor,
        )

        planner = PersonalizationCalibrationPlanner(
            args.personalization_calibration,
            data_root=args.data_root,
            output_dir=args.output_dir,
        )
        personalization_models = None
        if args.models:
            personalization_models = [
                value.strip() for value in args.models.split(',') if value.strip()
            ]
            if len(personalization_models) != 1:
                parser.error(
                    '--personalization-calibration accepts exactly one --models value'
                )
        filters = PlanFilters(
            outer_fold=args.outer_fold,
            pm=args.pm,
            task_type=args.task_type,
            calibration_mode=args.calibration_mode,
            model=(None if personalization_models is None else personalization_models[0]),
            budget_fraction=args.calibration_budget_fraction,
        )
        if args.plan_only:
            result = planner.plan(
                filters=filters, resume=args.resume, write_artifacts=True,
            )
        elif args.dry_execution:
            result = PersonalizationCalibrationExecutor(planner).dry_execution(
                filters=filters,
                execution_model=args.execution_model,
            )
        else:
            result = PersonalizationCalibrationExecutor(planner).run(
                filters=filters,
                execution_model=args.execution_model,
                resume=args.resume,
                subject_limit=args.subject_limit,
                max_epochs=(
                    args.max_calibration_epochs
                    if args.max_calibration_epochs is not None
                    else args.max_epochs
                ),
                device=args.device,
            )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.ordinal_transformer_analysis:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--feature-group-experiment': args.feature_group_experiment,
            '--ordinal-transformer-experiment': args.ordinal_transformer_experiment,
            '--label-target-audit': args.label_target_audit,
            '--temporal-target-audit': args.temporal_target_audit,
            '--label-definition-sensitivity': args.label_definition_sensitivity,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--ordinal-transformer-analysis cannot be combined with '
                + ', '.join(active_conflicts)
            )
        if args.plan_only and args.run:
            parser.error('--plan-only and --run are mutually exclusive')
        if not args.plan_only and not args.run:
            parser.error(
                '--ordinal-transformer-analysis requires --plan-only or --run'
            )
        analysis_config_path = Path(args.ordinal_transformer_analysis)
        analysis_document = (
            yaml.safe_load(analysis_config_path.read_text(encoding='utf-8')) or {}
            if analysis_config_path.is_file()
            else {}
        )
        analysis_type = str(analysis_document.get('analysis', {}).get('type', ''))
        if analysis_type == 'ordinal_transformer_multiseed_statistics':
            from bench.analysis.ordinal_transformer_multiseed_statistics import (
                OrdinalTransformerMultiseedStatistics,
            )

            analysis = OrdinalTransformerMultiseedStatistics(
                args.ordinal_transformer_analysis,
                output_dir=args.output_dir,
            )
        elif analysis_type == 'auxiliary_corn_policy_statistics':
            from bench.analysis.auxiliary_corn_policy_statistics import (
                AuxiliaryCornPolicyStatistics,
            )

            analysis = AuxiliaryCornPolicyStatistics(
                args.ordinal_transformer_analysis,
                output_dir=args.output_dir,
            )
        else:
            from bench.analysis.ordinal_transformer_statistics import (
                OrdinalTransformerStatistics,
            )

            analysis = OrdinalTransformerStatistics(
                args.ordinal_transformer_analysis,
                output_dir=args.output_dir,
            )
        if args.plan_only:
            print(analysis.render_plan(analysis.plan()))
            return
        result = analysis.execute()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.ordinal_transformer_experiment:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--feature-group-experiment': args.feature_group_experiment,
            '--label-target-audit': args.label_target_audit,
            '--temporal-target-audit': args.temporal_target_audit,
            '--label-definition-sensitivity': args.label_definition_sensitivity,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--ordinal-transformer-experiment cannot be combined with '
                + ', '.join(active_conflicts)
            )
        if args.plan_only and args.run:
            parser.error('--plan-only and --run are mutually exclusive')
        if not args.plan_only and not args.run:
            parser.error(
                '--ordinal-transformer-experiment requires --plan-only or --run'
            )
        from bench.experiments.ordinal_transformer import (
            build_ordinal_transformer_experiment,
        )

        experiment = build_ordinal_transformer_experiment(
            args.ordinal_transformer_experiment,
            output_dir=args.output_dir,
        )
        plans = experiment.plan()
        if args.plan_only:
            print(experiment.render_plan(plans))
            return
        result = experiment.execute(plans, resume=args.resume)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.feature_group_experiment:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--label-target-audit': args.label_target_audit,
            '--temporal-target-audit': args.temporal_target_audit,
            '--label-definition-sensitivity': args.label_definition_sensitivity,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--feature-group-experiment cannot be combined with '
                + ', '.join(active_conflicts)
            )
        if args.plan_only and args.run:
            parser.error('--plan-only and --run are mutually exclusive')
        if not args.plan_only and not args.run:
            parser.error(
                '--feature-group-experiment requires --plan-only or --run'
            )
        from bench.experiments.feature_group_ablation import (
            build_feature_group_experiment,
        )

        parse_values = lambda value: (
            None
            if value is None
            else [item.strip() for item in value.split(',') if item.strip()]
        )
        experiment = build_feature_group_experiment(
            args.feature_group_experiment,
            output_dir=args.output_dir,
        )
        plans = experiment.plan(
            feature_groups=parse_values(args.feature_groups),
            tasks=parse_values(args.tasks),
            models=parse_values(args.models),
            seed=42 if args.seed is None else args.seed,
        )
        if args.plan_only:
            print(experiment.render_plan(plans))
            return
        result = experiment.execute(plans, resume=args.resume)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.label_definition_sensitivity:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--label-target-audit': args.label_target_audit,
            '--temporal-target-audit': args.temporal_target_audit,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--label-definition-sensitivity cannot be combined with '
                + ', '.join(active_conflicts)
            )
        from bench.analysis.label_definition_sensitivity import (
            LabelDefinitionSensitivity,
        )

        analysis = LabelDefinitionSensitivity(
            args.label_definition_sensitivity,
            output_dir=args.output_dir,
        )
        if args.plan_only:
            print(analysis.render_plan(analysis.plan()))
            return
        result = analysis.execute()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.temporal_target_audit:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
            '--label-target-audit': args.label_target_audit,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--temporal-target-audit cannot be combined with '
                + ', '.join(active_conflicts)
            )
        from bench.analysis.temporal_target_structure import TemporalTargetAudit

        analysis = TemporalTargetAudit(
            args.temporal_target_audit,
            output_dir=args.output_dir,
        )
        if args.plan_only:
            print(analysis.render_plan(analysis.plan()))
            return
        result = analysis.execute()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.label_target_audit:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
            '--cross-source-experiment': args.cross_source_experiment,
        }
        active_conflicts = [name for name, value in conflicts.items() if value]
        if active_conflicts:
            parser.error(
                '--label-target-audit cannot be combined with '
                + ', '.join(active_conflicts)
            )
        from bench.analysis.label_target_audit import LabelTargetAudit

        analysis = LabelTargetAudit(
            args.label_target_audit,
            output_dir=args.output_dir,
        )
        if args.plan_only:
            print(analysis.render_plan(analysis.plan()))
            return
        result = analysis.execute()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.cross_source_experiment:
        conflicts = {
            '--config': args.config,
            '--test': args.test,
            '--experiment-matrix': args.experiment_matrix,
            '--calibration-experiment': args.calibration_experiment,
            '--automl-study': args.automl_study,
            '--statistical-analysis': args.statistical_analysis,
        }
        active_conflicts = [
            name for name, value in conflicts.items() if value
        ]
        if active_conflicts:
            parser.error(
                '--cross-source-experiment cannot be combined with '
                + ', '.join(active_conflicts)
            )
        if args.plan_only and args.run:
            parser.error('--plan-only and --run are mutually exclusive')
        if not args.plan_only and not args.run:
            parser.error(
                '--cross-source-experiment requires --plan-only or --run'
            )
        from bench.experiments.cross_source_generalization import (
            CrossSourceExperiment,
        )

        directions = (
            None
            if args.directions is None
            else [
                value.strip()
                for value in args.directions.split(',')
                if value.strip()
            ]
        )
        subject_modes = (
            None
            if args.subject_modes is None
            else [
                value.strip()
                for value in args.subject_modes.split(',')
                if value.strip()
            ]
        )
        models = (
            None
            if args.models is None
            else [
                value.strip()
                for value in args.models.split(',')
                if value.strip()
            ]
        )
        experiment = CrossSourceExperiment(args.cross_source_experiment)
        plans = experiment.plan(
            directions=directions,
            subject_modes=subject_modes,
            models=models,
            seed=42 if args.seed is None else args.seed,
            max_train_windows=args.max_train_windows,
            max_test_windows=args.max_test_windows,
            max_epochs=args.max_epochs,
        )
        if args.plan_only:
            plan_reports = experiment.write_plan_reports(plans)
            print(experiment.render_plan(plans))
            print(json.dumps(plan_reports, indent=2))
            return
        result = experiment.execute(plans, resume=args.resume)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.statistical_analysis:
        if (
            args.config
            or args.test
            or args.experiment_matrix
            or args.calibration_experiment
            or args.automl_study
        ):
            parser.error(
                '--statistical-analysis cannot be combined with --config, --test, '
                '--experiment-matrix, --calibration-experiment, or --automl-study'
            )
        from bench.analysis.report_builder import StatisticalAnalysis

        tracks = (
            None
            if args.tracks is None
            else [value.strip() for value in args.tracks.split(',') if value.strip()]
        )
        analysis = StatisticalAnalysis(
            args.statistical_analysis,
            tracks=tracks,
            bootstrap_samples=args.bootstrap_samples,
            random_state=args.seed,
            output_dir=args.output_dir,
        )
        if args.plan_only:
            print(analysis.render_plan(analysis.plan()))
            return
        result = analysis.execute()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.automl_study:
        if args.config or args.test or args.experiment_matrix or args.calibration_experiment:
            parser.error(
                '--automl-study cannot be combined with --config, --test, '
                '--experiment-matrix, or --calibration-experiment'
            )
        from bench.automl.study_runner import AutoMLStudyRunner

        study = AutoMLStudyRunner(
            args.automl_study,
            outer_fold=1 if args.outer_fold is None else args.outer_fold,
            study_name=args.study_name,
            storage=args.storage,
            n_trials=args.n_trials,
            timeout_seconds=args.timeout,
            seed=args.seed,
            inner_splits=args.inner_splits,
            max_epochs=args.max_epochs,
            max_windows=args.max_windows,
            evaluate_best=False if args.no_evaluate_best else None,
        )
        if args.plan_only:
            print(study.render_plan(study.plan()))
            return
        result = study.execute(resume=args.resume)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.calibration_experiment:
        if args.config or args.test or args.experiment_matrix:
            parser.error(
                '--calibration-experiment cannot be combined with '
                '--config, --test, or --experiment-matrix'
            )
        budgets = (
            None
            if args.calibration_budgets is None
            else [
                float(value.strip())
                for value in args.calibration_budgets.split(',')
                if value.strip()
            ]
        )
        methods = (
            None
            if args.calibration_methods is None
            else [
                value.strip()
                for value in args.calibration_methods.split(',')
                if value.strip()
            ]
        )
        calibration_path = Path(args.calibration_experiment)
        calibration_document = (
            yaml.safe_load(calibration_path.read_text(encoding='utf-8')) or {}
            if calibration_path.is_file()
            else {}
        )
        calibration_type = str(
            calibration_document.get('experiment', {}).get('type', '')
        )
        if calibration_type == 'user_calibration_multiseed':
            if budgets is not None or methods is not None or args.seed is not None:
                parser.error(
                    'Multiseed calibration takes model seeds, the 20% budget, '
                    'and methods from its config'
                )
            from bench.experiments.user_calibration_multiseed import (
                UserCalibrationMultiseedExperiment,
            )

            experiment = UserCalibrationMultiseedExperiment(
                args.calibration_experiment
            )
            result = experiment.execute(
                fold_limit=args.fold_limit,
                subject_limit=args.subject_limit,
                max_epochs=args.max_calibration_epochs,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        elif calibration_type == 'pm_regression_personalization_multiseed':
            if budgets is not None or methods is not None or args.seed is not None:
                parser.error(
                    'PM regression multiseed personalization takes seeds, '
                    'the 20% budget, and methods from its config'
                )
            from bench.experiments.pm_regression_personalization_multiseed import (
                PMRegressionPersonalizationMultiseedExperiment,
            )

            experiment = PMRegressionPersonalizationMultiseedExperiment(
                args.calibration_experiment
            )
            result = experiment.execute(
                fold_limit=args.fold_limit,
                subject_limit=args.subject_limit,
                max_epochs=args.max_calibration_epochs,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        elif calibration_type == 'pm_regression_personalization':
            if budgets is not None or args.seed is not None:
                parser.error(
                    'PM regression personalization fixes the 20% budget and '
                    'model/split seed 42 in its config'
                )
            from bench.experiments.pm_regression_personalization import (
                PMRegressionPersonalizationExperiment,
            )

            experiment = PMRegressionPersonalizationExperiment(
                args.calibration_experiment
            )
            result = experiment.execute(
                fold_limit=args.fold_limit,
                subject_limit=args.subject_limit,
                methods=methods,
                max_epochs=args.max_calibration_epochs,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        else:
            from bench.experiments.user_calibration import (
                UserCalibrationExperiment,
            )

            experiment = UserCalibrationExperiment(args.calibration_experiment)
            result = experiment.execute(
                fold_limit=args.fold_limit,
                subject_limit=args.subject_limit,
                budgets_seconds=budgets,
                methods=methods,
                max_epochs=args.max_calibration_epochs,
                random_state=args.seed,
                output_dir=args.output_dir,
                write_reports=True,
                resume=args.resume,
            )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.experiment_matrix:
        if args.config or args.test:
            parser.error(
                '--experiment-matrix cannot be combined with --config or --test'
            )
        from bench.experiments.preprocessing_ablation import (
            PreprocessingAblation,
            render_plan_csv,
            render_plan_markdown,
        )

        experiment = PreprocessingAblation(args.experiment_matrix)
        plans = experiment.plan(
            trial_ids=_parse_trial_ids(args.trial_ids),
            seed=42 if args.seed is None else args.seed,
            fold_limit=args.fold_limit,
            max_windows=args.max_windows,
            max_epochs=args.max_epochs,
        )
        if args.plan_only:
            print(render_plan_markdown(plans))
            print("\n--- CSV ---\n")
            print(render_plan_csv(plans), end='')
            return
        if args.cache_only and not args.build_missing_caches:
            parser.error('--cache-only requires --build-missing-caches')
        results = experiment.execute(
            plans,
            build_missing_caches=args.build_missing_caches,
            run=not args.cache_only,
            resume=args.resume,
        )
        print(json.dumps(results, indent=2, default=str))
        return

    if args.test:
        config = create_default_config()
    elif args.config:
        config = load_config(args.config)
    
    else:
        parser.print_help()
        logger.error("Either --config or --test is required")
        sys.exit(1)

    config = override_config_with_args(config, args)

    if not validate_config(config):
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Starting EEG Benchmark")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Datasets: {list(config['datasets'].keys())}")
    logger.info(f"Tasks: {config['tasks']}")
    logger.info(f"Models: {list(config['models'].keys())}")
    logger.info("=" * 60)

    try:
        runner = BenchmarkRunner(config)
        summary = runner.run()

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        if len(summary) > 0:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            print(summary.to_string(index=False))
        else:
            print("No results to display")
        
        print("=" * 80)
        print(f"\nResults saved to: {runner.output_dir}")
        print(f"JSON: benchmark_results_{runner.timestamp}.json")
        print(f"CSV:  summary_{runner.timestamp}.csv")
        
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
