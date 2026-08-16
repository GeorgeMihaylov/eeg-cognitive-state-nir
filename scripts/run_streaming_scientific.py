from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.streaming_worker.scientific import (
    check_api,
    materialize_replay,
    plan_experiment,
    run_replay,
    train_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific held-out EEG streaming experiment")
    parser.add_argument("--config", default="configs/streaming_scientific_v1.yaml")
    parser.add_argument(
        "--action",
        choices=("plan", "train", "materialize-replay", "replay", "api-check", "all"),
        default="plan",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        result = plan_experiment(args.config)
    elif args.action == "train":
        result = train_bundle(args.config)
    elif args.action == "materialize-replay":
        result = materialize_replay(args.config)
    elif args.action == "replay":
        result = run_replay(args.config, overwrite=args.overwrite)
    elif args.action == "api-check":
        result = check_api(args.config)
    else:
        train_bundle(args.config)
        materialize_replay(args.config)
        result = run_replay(args.config, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
