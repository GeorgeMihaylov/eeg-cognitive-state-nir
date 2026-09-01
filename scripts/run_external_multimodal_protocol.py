"""Inventory, plan, or execute the CLARE/CL-Drive/MEFAR multimodal protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.external_multimodal_protocol import (
    extract_archives,
    load_config,
    plan_experiment,
    run_model_family,
    write_inventory_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--extract", action="store_true")
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run-xgboost", action="store_true")
    action.add_argument("--run-shallow", action="store_true")
    args = parser.parse_args()
    if args.extract:
        result = extract_archives(load_config(args.config))
    elif args.inventory:
        result = write_inventory_artifacts(args.config)
    elif args.plan_only:
        result = plan_experiment(args.config)
    elif args.run_xgboost:
        result = run_model_family(args.config, "xgboost")
    else:
        result = run_model_family(args.config, "shallow")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
