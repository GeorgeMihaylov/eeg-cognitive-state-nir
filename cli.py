"""
Основная точка входа для запуска бенчмарка

TODO: написать раннер
"""

import argparse
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run EEG benchmark")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Dataset name or path')
    parser.add_argument('--models', type=str, help='Comma-separated model names')
    parser.add_argument('--task', type=str, default='cognitive_load_3class')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    parser.add_argument('--feature-set', type=str, default='pow_plus_eeg')

    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    runner = BenchmarkRunner(config)
    results = runner.run()
    print("Final Summary:")
    summary = runner.get_summary()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
