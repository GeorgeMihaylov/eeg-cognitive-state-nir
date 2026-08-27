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
DEFAULT_EXPECTED_CANONICAL_WINDOWS = 30_958


def canonical_target_frame(
    processed_path: Path,
    logical_recording_map_path: Path | None,
    raw_window_index_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, int | str | None]]:
    """Reproduce the canonical raw-deduplicated/QC cohort used by confirmatory runs."""
    frame = _load_table(processed_path)
    sample_id_source = "existing_column"
    if "sample_id" not in frame.columns:
        # Historical raw-eeg-window-v3 built sample_id from the original processed
        # row index before any supervised/QC filtering.  Reproduce that contract
        # exactly instead of inventing a new positional id after filtering.
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy(dtype="int64"))
        sample_id_source = "reconstructed_processed_row_index"

    diagnostics: dict[str, int | str | None] = {
        "processed_rows": int(len(frame)),
        "selected_logical_rows": None,
        "raw_qc_index_rows": None,
        "canonical_rows": None,
        "sample_id_source": sample_id_source,
        "policy": "processed_table_unfiltered",
    }

    selected_records: set[str] | None = None
    if logical_recording_map_path is not None:
        logical = _load_table(logical_recording_map_path)
        if "selected_record_id" not in logical.columns:
            raise ValueError("Logical recording map must contain selected_record_id")
        if (
            "record_group_id" in logical.columns
            and logical["record_group_id"].astype(str).duplicated().any()
        ):
            raise ValueError("Logical recording map has duplicate record_group_id values")
        selected_records = set(logical["selected_record_id"].dropna().astype(str))
        if not selected_records:
            raise ValueError("Logical recording map selected no records")
        if "record_id" not in frame.columns:
            raise ValueError("Processed target table must contain record_id")
        logical_frame = frame.loc[
            frame["record_id"].astype(str).isin(selected_records)
        ].copy()
        diagnostics["selected_logical_rows"] = int(len(logical_frame))
        if logical_frame.empty:
            raise ValueError("Logical-recording filter selected no processed rows")
    else:
        logical_frame = frame.copy()

    if raw_window_index_path is None:
        selected = logical_frame
        diagnostics["policy"] = (
            "selected_logical_recordings"
            if logical_recording_map_path is not None
            else "processed_table_unfiltered"
        )
    else:
        raw_index = _load_table(raw_window_index_path)
        required = {"sample_id", "record_id"}
        missing = sorted(required - set(raw_index.columns))
        if missing:
            raise ValueError(f"Raw EEG window index is missing columns: {missing}")
        if "status" in raw_index.columns:
            raw_index = raw_index.loc[raw_index["status"].astype(str).eq("ok")].copy()
        if selected_records is not None:
            raw_index = raw_index.loc[
                raw_index["record_id"].astype(str).isin(selected_records)
            ].copy()
        if raw_index.empty:
            raise ValueError("Raw-QC/logical-recording filter selected no index rows")
        diagnostics["raw_qc_index_rows"] = int(len(raw_index))
        canonical_ids = set(raw_index["sample_id"].dropna().astype(str))
        selected = logical_frame.loc[
            logical_frame["sample_id"].astype(str).isin(canonical_ids)
        ].copy()
        missing_from_targets = canonical_ids - set(selected["sample_id"].astype(str))
        if missing_from_targets:
            raise ValueError(
                "Canonical raw-window index contains sample IDs absent from target table: "
                f"{sorted(missing_from_targets)[:20]}"
            )
        diagnostics["policy"] = "raw_qc_index_and_selected_logical_recordings"

    if selected.empty:
        raise ValueError("Canonical cohort is empty")
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("Canonical target cohort contains duplicate sample_id values")
    diagnostics["canonical_rows"] = int(len(selected))
    return selected.reset_index(drop=True), diagnostics


def run_audit(
    *,
    root: Path,
    catalog_path: Path,
    processed_path: Path | None,
    logical_recording_map_path: Path | None,
    raw_window_index_path: Path | None,
    output_dir: Path,
    chunk_size: int,
    max_records: int | None,
    skip_raw: bool,
    expected_canonical_windows: int | None,
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

    boundary = pd.DataFrame()
    cohort_diagnostics: dict[str, object] = {}
    if processed_path is not None:
        target_frame, cohort_diagnostics = canonical_target_frame(
            processed_path,
            logical_recording_map_path,
            raw_window_index_path,
        )
        canonical_rows = int(len(target_frame))
        if (
            expected_canonical_windows is not None
            and canonical_rows != int(expected_canonical_windows)
        ):
            raise RuntimeError(
                "Canonical cohort mismatch: "
                f"expected {expected_canonical_windows}, got {canonical_rows}. "
                "Q3 diagnostics aborted to avoid comparing different cohorts."
            )
        boundary = q3_boundary_audit(target_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "pm_field_inventory_records.csv", index=False)
    inventory_summary.to_csv(output_dir / "pm_field_inventory_summary.csv", index=False)
    if not raw.empty:
        raw.to_csv(output_dir / "pm_raw_record_audit.csv", index=False)
        raw_summary.to_csv(output_dir / "pm_raw_record_summary.csv", index=False)
    if not boundary.empty:
        boundary.to_csv(output_dir / "pm_q3_boundary_audit.csv", index=False)
    if cohort_diagnostics:
        _write_json(output_dir / "pm_q3_cohort_diagnostics.json", cohort_diagnostics)
    (output_dir / "pm_target_validity_audit.md").write_text(
        render_markdown(inventory_summary, raw_summary, boundary),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "catalog_path": str(catalog_path),
        "processed_path": None if processed_path is None else str(processed_path),
        "logical_recording_map_path": (
            None if logical_recording_map_path is None else str(logical_recording_map_path)
        ),
        "raw_window_index_path": (
            None if raw_window_index_path is None else str(raw_window_index_path)
        ),
        "output_dir": str(output_dir),
        "catalog_records": int(len(catalog)),
        "inventory_rows": int(len(inventory)),
        "raw_audit_rows": int(len(raw)),
        "canonical_target_rows": cohort_diagnostics.get("canonical_rows"),
        "processed_target_rows": cohort_diagnostics.get("processed_rows"),
        "selected_logical_target_rows": cohort_diagnostics.get("selected_logical_rows"),
        "raw_qc_index_rows": cohort_diagnostics.get("raw_qc_index_rows"),
        "sample_id_source": cohort_diagnostics.get("sample_id_source"),
        "q3_boundary_rows": int(len(boundary)),
        "raw_skipped": bool(skip_raw),
        "max_records": max_records,
        "models_trained": 0,
        "raw_io_policy": "one_chunked_pass_per_record",
        "q3_cohort_policy": cohort_diagnostics.get("policy"),
        "expected_canonical_windows": expected_canonical_windows,
    }
    _write_json(output_dir / "pm_target_validity_audit_summary.json", summary)
    return summary


def _resolve_optional(root: Path, value: str) -> Path | None:
    if str(value).strip().lower() == "none":
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--catalog", default="data/interim/emotiv_record_catalog.csv")
    parser.add_argument(
        "--processed",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
        help="Processed target table for Q3 boundary diagnostics; use 'none' to skip.",
    )
    parser.add_argument(
        "--logical-recording-map",
        default="data/interim/logical_recording_map.parquet",
        help="Canonical selected logical recordings; use 'none' to disable filtering.",
    )
    parser.add_argument(
        "--raw-window-index",
        default="data/interim/raw_eeg_window_index_w10_raw_v3.parquet",
        help="Accepted raw-EEG window index used by the canonical A/raw contour.",
    )
    parser.add_argument(
        "--expected-canonical-windows",
        type=int,
        default=DEFAULT_EXPECTED_CANONICAL_WINDOWS,
        help="Abort Q3 audit if the reconstructed canonical cohort has another size; use 0 to disable.",
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
    processed = _resolve_optional(root, args.processed)
    logical_map = _resolve_optional(root, args.logical_recording_map)
    raw_index = _resolve_optional(root, args.raw_window_index)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    expected = (
        None if int(args.expected_canonical_windows) <= 0
        else int(args.expected_canonical_windows)
    )

    summary = run_audit(
        root=root,
        catalog_path=catalog,
        processed_path=processed,
        logical_recording_map_path=logical_map,
        raw_window_index_path=raw_index,
        output_dir=output,
        chunk_size=int(args.chunk_size),
        max_records=args.max_records,
        skip_raw=bool(args.skip_raw),
        expected_canonical_windows=expected,
    )
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
