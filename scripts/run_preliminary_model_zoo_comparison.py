"""Plan or execute the preliminary one-fold model-zoo comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.preliminary_model_zoo_comparison import (
    PreliminaryComparisonExecutor,
    comparison_protocol_hash,
    import_reusable_shallow_results,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/model_zoo/preliminary_model_zoo_comparison_fold1.json"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--task-type",
        choices=("classification", "regression", "all"),
        default="all",
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--target-id", action="append", dest="target_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.plan_only == args.execute:
        parser.error("select exactly one of --plan-only or --execute")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    manifest_path = output / "manifest.json"
    status_path = output / "run_status.csv"
    if args.plan_only or not (manifest_path.is_file() and status_path.is_file()):
        if args.plan_only and manifest_path.is_file() and status_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_hash = manifest.get("protocol_hash")
            expected_hash = comparison_protocol_hash(config)
            if stored_hash is not None and stored_hash != expected_hash:
                raise ValueError("Comparison protocol hash mismatch in plan resume")
        else:
            manifest = write_plan(output_dir=output, config=config)
            reuse = config.get("reuse_shallowconvnet")
            if reuse:
                imported = import_reusable_shallow_results(
                    output_dir=output,
                    source_dir=reuse["source_dir"],
                    raw_preprocessing_hash=reuse["raw_preprocessing_hash"],
                )
                manifest["reused_shallowconvnet_rows"] = int(len(imported))
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.plan_only:
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    if args.data_root is None:
        parser.error("--data-root is required with --execute")
    executor = PreliminaryComparisonExecutor(
        config,
        data_root=args.data_root,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    status_counts = executor.run(
        task_type=args.task_type,
        models=args.models,
        targets=args.target_ids,
    )
    print(json.dumps({"status_counts": status_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
