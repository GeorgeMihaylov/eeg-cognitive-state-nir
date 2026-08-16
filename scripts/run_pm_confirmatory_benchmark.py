"""Plan or execute the selected-model seven-PM confirmatory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.pm_confirmatory_benchmark import (
    ConfirmatoryExecutor,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/pm_confirmatory/selected_models_5fold_v1.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--preliminary-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--model", choices=(
        "random_forest", "xgboost", "torch_shallow_convnet", "torch_lstm"
    ), action="append", dest="models")
    parser.add_argument("--target-id", action="append", dest="targets")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["output_dir"])
    manifest = write_plan(
        config,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        preliminary_root=args.preliminary_root,
        output_dir=output,
    )
    if args.plan_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    executor = ConfirmatoryExecutor(
        config,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        preliminary_root=args.preliminary_root,
        output_dir=output,
        resume=args.resume,
    )
    counts = executor.run(
        folds=args.folds,
        models=args.models,
        targets=args.targets,
    )
    print(json.dumps({"status_counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
