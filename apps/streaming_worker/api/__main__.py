from __future__ import annotations

import argparse
import os

import uvicorn

from ..config import WorkerConfig
from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cogstate streaming API")
    parser.add_argument("--config", default="configs/streaming.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    config = WorkerConfig.from_yaml(args.config)
    host = args.host or config.api.host
    port = args.port or config.api.port
    if args.reload:
        os.environ["COGSTATE_STREAMING_CONFIG"] = args.config
        uvicorn.run(
            "apps.streaming_worker.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(config=config), host=host, port=port)


if __name__ == "__main__":
    main()
