from __future__ import annotations

import argparse
import logging

from .config import WorkerConfig
from .runtime import StreamingRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EEG streaming worker")
    parser.add_argument("--config", default="configs/streaming.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime = StreamingRuntime(WorkerConfig.from_yaml(args.config))
    try:
        runtime.run()
    except KeyboardInterrupt:
        runtime.stop()


if __name__ == "__main__":
    main()
