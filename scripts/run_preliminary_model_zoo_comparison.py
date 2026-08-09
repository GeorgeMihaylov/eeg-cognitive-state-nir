"""Plan the preliminary one-fold model-zoo comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.preliminary_model_zoo_comparison import (
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
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = write_plan(output_dir=config["output_dir"], config=config)
    reuse = config.get("reuse_shallowconvnet")
    if reuse:
        imported = import_reusable_shallow_results(
            output_dir=config["output_dir"],
            source_dir=reuse["source_dir"],
            raw_preprocessing_hash=reuse["raw_preprocessing_hash"],
        )
        manifest["reused_shallowconvnet_rows"] = int(len(imported))
        Path(config["output_dir"], "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, default=str))
    if not args.plan_only:
        raise SystemExit(
            "Execution gate is closed until the full feature cache is authorized and verified."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
