"""CLI orchestration for the PM target validity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bench.analysis.pm_target_validity_audit import (
    _jsonable,
    _load_table,
    _write_json,
    build_catalog_inventory,
    q3_boundary_audit,
    render_markdown,
    summarize_catalog_inventory,
    summarize_raw_audit,
)
from bench.analysis.pm_target_validity_streaming import audit_raw_records_streaming


AUDIT_SCHEMA_VERSION = "pm-target-validity-audit-v1"


def run_audit(
    *,
    root: Path,
    catalog_path: Path,
    processed_path: Path | None,
    output_dir: Path,
    chunk_size: int,
    max_records: int | None,
    skip_raw: bool,
) -> dict[str, object]:
    catalog = pd.read_csv(catalog_path)
    inventory = build_catalog_inventory(catalog)
    inventory_summary = summarize_catalog_inventory(inventory)
    raw = (
        pd.DataFrame()
        if skip_raw
        else audit_raw_records_streaming(
            catalog,
            root=root,
            chunk_size=chunk_size,
            max_records=max_records,
        )
    )
    raw_summary = summarize_raw_audit(raw)
    boundary = (
        pd.DataFrame()
        if processed_path is None
        else q3_boundary_audit(_load_table(processed_path))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "pm_field_inventory_records.csv", index=False)
    inventory_summary.to_csv(output_dir / "pm_field_inventory_summary.csv", index=False)
    if not raw.empty:
        raw.to_csv(output_dir / "pm_raw_record_audit.csv", index=False)
        raw_summary.to_csv(output_dir / "pm_raw_record_summary.csv", index=False)
    if not boundary.empty:
        boundary.to_csv(output_dir / "pm_q3_boundary_audit.csv", index=False)
    (output_dir / "pm_target_validity_audit.md").write_text(
        render_markdown(inventory_summary, raw_summary, boundary),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "catalog_path": str(catalog_path),
        "processed_path": None if processed_path is None else str(processed_path),
        "output_dir": str(output_dir),
        "catalog_records": int(len(catalog)),
        "inventory_rows": int(len(inventory)),
        "raw_audit_rows": int(len(raw)),
        "q3_boundary_rows": int(len(boundary)),
        "raw_skipped": bool(skip_raw),
        "max_records": max_records,
        "models_trained": 0,
        "raw_io_policy": "one_chunked_pass_per_record",
    }
    _write_json(output_dir / "pm_target_validity_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--catalog", default="data/interim/emotiv_record_catalog.csv"
    )
    parser.add_argument(
        "--processed",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
        help="Processed target table for Q3 boundary diagnostics; use 'none' to skip.",
    )
    parser.add_argument(
        "--output-dir", default="reports/diagnostics/pm_target_validity_audit"
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Run only catalog inventory and processed-target diagnostics.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    catalog = Path(args.catalog)
    if not catalog.is_absolute():
        catalog = root / catalog
    processed: Path | None
    if str(args.processed).strip().lower() == "none":
        processed = None
    else:
        processed = Path(args.processed)
        if not processed.is_absolute():
            processed = root / processed
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output

    summary = run_audit(
        root=root,
        catalog_path=catalog,
        processed_path=processed,
        output_dir=output,
        chunk_size=int(args.chunk_size),
        max_records=args.max_records,
        skip_raw=bool(args.skip_raw),
    )
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
