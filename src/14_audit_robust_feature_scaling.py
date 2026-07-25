"""Reproduce fold-1 feature audit and combine robust-scaling smoke results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.bench_runner import BenchmarkRunner  # noqa: E402
from bench.data_quality import (  # noqa: E402
    run_feature_outlier_audit,
    summarize_scaling_results,
)
from bench.tasks.tasks_registry import get_task  # noqa: E402
from bench.validation.cross_val import CrossValidator  # noqa: E402
from cli import validate_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit feature outliers using canonical benchmark splits"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--summarize-results",
        type=Path,
        help="Completed benchmark_results_*.json to aggregate after the audit",
    )
    parser.add_argument(
        "--near-constant-threshold", type=float, default=1e-8
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    if not validate_config(config):
        raise SystemExit("Invalid benchmark configuration")
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(config)
    if len(config["datasets"]) != 1:
        raise ValueError("Feature audit requires exactly one dataset")
    dataset_name = next(iter(config["datasets"]))
    data = runner.load_dataset(dataset_name)
    task = get_task(
        "performance_metrics_regression",
        data,
        config.get("task_config", {}),
    )
    evaluation = config["evaluation"]
    splits = CrossValidator(task).run_group_kfold(
        group_column=evaluation["group_column"],
        n_splits=int(evaluation.get("n_splits", 5)),
        random_state=int(evaluation.get("random_state", 42)),
        precomputed_fold_column=evaluation.get("precomputed_fold_column"),
    )
    split = splits["fold_01"]
    model_name = next(iter(config["models"]))
    model = runner._get_model_for_split(
        runner.models[model_name],
        split,
        num_outputs=int(data.n_outputs),
    )
    runner._configure_model_validation(model, split)
    inner_train, inner_validation = model.resolve_validation_indices(
        split.y_train
    )
    artifacts = run_feature_outlier_audit(
        split,
        inner_train,
        inner_validation,
        output_dir,
        near_constant_threshold=args.near_constant_threshold,
    )
    print("Feature audit complete:")
    for name, path in artifacts.items():
        if name != "summary_payload":
            print(f"  {name}: {path}")

    if args.summarize_results is not None:
        summaries = summarize_scaling_results(
            args.summarize_results,
            output_dir,
        )
        print("Scaling summary complete:")
        for name, path in summaries.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
