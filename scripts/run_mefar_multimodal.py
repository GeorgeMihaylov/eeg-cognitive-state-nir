"""Inventory, plan, or execute the leakage-safe MEFAR multimodal protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.mefar_multimodal import (
    plan_experiment,
    run_experiment,
    safe_extract_nested,
    write_inventory_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--extract", action="store_true")
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.extract:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        result = safe_extract_nested(
            Path(config["dataset"]["archive"]),
            Path(config["dataset"]["extracted_root"]),
        )
    elif args.inventory:
        result = write_inventory_artifacts(args.config, args.output_dir)
    elif args.plan_only:
        result = plan_experiment(args.config)
    else:
        result = run_experiment(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
