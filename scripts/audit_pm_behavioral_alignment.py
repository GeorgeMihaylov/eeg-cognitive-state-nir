"""Standalone lightweight inventory/alignment smoke for PM behavioral metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_temporal_quality import (
    _repo_path,
    _write_csv,
    audit_behavioral_sources,
    load_config,
    prepare_pm_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--raw-index-path", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = _repo_path(args.output_dir or config["output_dir"])
    if args.plan_only:
        print(json.dumps({"behavioral_audit": True, "writes_performed": False}, indent=2))
        return 0
    columns = ["source", "subject_id", "record_id", "t_start", *config["pm_targets"].values()]
    frame = prepare_pm_frame(pd.read_parquet(args.data_path, columns=columns), config)
    inventory, feasibility, aligned, summary = audit_behavioral_sources(
        Path(args.raw_root),
        Path(args.raw_index_path),
        frame,
        smoke_limit=int(config["behavioral_audit"]["alignment_smoke_limit"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "behavioral_inventory.csv", inventory)
    _write_csv(output / "behavioral_alignment_feasibility.csv", feasibility)
    aligned.to_parquet(output / "behavioral_alignment_smoke.parquet", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
