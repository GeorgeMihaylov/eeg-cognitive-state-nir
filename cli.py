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
    parser.add_argument('--fold-limit', type=int)
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument(
        '--calibration-experiment',
        type=str,
        help='Run subject calibration from an existing benchmark run',
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
        '--automl-study',
        type=str,
        help='Run a nested AutoML study through the canonical benchmark',
    )
    parser.add_argument('--study-name', type=str)
    parser.add_argument('--storage', type=str)
    parser.add_argument('--outer-fold', type=int, default=1)
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

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
            OrdinalTransformerSmokeExperiment,
        )

        experiment = OrdinalTransformerSmokeExperiment(
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
            outer_fold=args.outer_fold,
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
        from bench.experiments.user_calibration import UserCalibrationExperiment

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
