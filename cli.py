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
                'data_path': './data/emotiv_data.parquet',
                'feature_set': 'pow_plus_eeg',
                'n_classes': 3,
                'discretize': True,
                'max_features': 500
            }
        },
        'tasks': ['cognitive_load_3class'],
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

    errors = []

    if 'datasets' not in config or not config['datasets']:
        errors.append("No datasets specified in config")

    if 'models' not in config or not config['models']:
        errors.append("No models specified in config")

    if 'tasks' not in config or not config['tasks']:
        errors.append("No tasks specified in config")

    for dataset_name, dataset_config in config.get('datasets', {}).items():
        if 'data_path' not in dataset_config:
            errors.append(f"Dataset '{dataset_name}' missing 'data_path'")
        elif not Path(dataset_config['data_path']).exists():
            errors.append(f"Dataset '{dataset_name}' data path not found: {dataset_config['data_path']}")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--build-missing-caches', action='store_true')
    parser.add_argument('--cache-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--fold-limit', type=int)
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--max-epochs', type=int)
    
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

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
            seed=args.seed,
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
