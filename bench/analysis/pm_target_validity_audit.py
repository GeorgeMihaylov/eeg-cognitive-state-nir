"""Training-free validity audit for Emotiv Performance Metric targets.

The audit is intentionally read-only with respect to canonical datasets. It
answers four questions before any new model training is attempted:

1. Which PM representations (Raw/Scaled/Min/Max/IsActive) are actually present?
2. What is the empirical PM update cadence and phase relative to the nominal
   ten-second update period?
3. How strongly do Raw and Scaled values agree, and how often is the PM detector
   active when values are exported?
4. How much of each continuous target lies close to fold-local Q3 boundaries?

Large raw CSV/CSV.BZ2 recordings are processed in chunks. Output artifacts are
small CSV/JSON/Markdown summaries under a user-selected report directory.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bench.datasets.logical_recordings import ensure_record_group_ids
from bench.tasks.target_registry import PM_METRICS
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform


AUDIT_SCHEMA_VERSION = "pm-target-validity-audit-v1"
PM_FIELDS = ("Raw", "Scaled", "Min", "Max", "IsActive")
NOMINAL_PM_PERIOD_SECONDS = 10.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def pm_column(metric: str, field: str) -> str:
    canonical = str(metric).strip().lower()
    if canonical not in PM_METRICS:
        raise ValueError(f"Unknown PM metric {metric!r}")
    if field not in PM_FIELDS:
        raise ValueError(f"Unknown PM field {field!r}")
    return f"PM.{canonical.capitalize()}.{field}"


def discover_pm_columns(columns: Iterable[str]) -> dict[str, dict[str, bool]]:
    observed = {str(column) for column in columns}
    return {
        metric: {
            field: pm_column(metric, field) in observed
            for field in PM_FIELDS
        }
        for metric in PM_METRICS
    }


def build_catalog_inventory(catalog: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "subject_id", "main_rel_path", "pm_columns"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for _, record in catalog.iterrows():
        columns = _parse_list(record["pm_columns"])
        discovered = discover_pm_columns(columns)
        for metric in PM_METRICS:
            rows.append({
                "source": str(record["source"]),
                "subject_id": str(record["subject_id"]),
                "main_rel_path": str(record["main_rel_path"]),
                "metric": metric,
                **{
                    f"has_{field.lower()}": bool(discovered[metric][field])
                    for field in PM_FIELDS
                },
            })
    return pd.DataFrame(rows)


def summarize_catalog_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    boolean_columns = [f"has_{field.lower()}" for field in PM_FIELDS]
    grouped = inventory.groupby(["source", "metric"], sort=True, observed=True)
    rows: list[dict[str, Any]] = []
    for (source, metric), group in grouped:
        row: dict[str, Any] = {
            "source": str(source),
            "metric": str(metric),
            "records": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
        }
        for column in boolean_columns:
            present = group[column].astype(bool)
            row[f"{column}_records"] = int(present.sum())
            row[f"{column}_fraction"] = float(present.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "metric"]).reset_index(drop=True)


def _open_text(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(mode="rt", encoding="utf-8", errors="replace")


def _find_header(path: Path, max_lines: int = 30) -> tuple[int, str, list[str]]:
    with _open_text(path) as stream:
        for index, line in enumerate(stream):
            if index >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            if "Timestamp" in text and "PM." in text:
                separator = "," if text.count(",") >= text.count(";") else ";"
                return index, separator, [part.strip() for part in text.split(separator)]
    raise ValueError(f"Unable to find Emotiv CSV header in {path}")


def _resolve_record_path(record: Mapping[str, Any], root: Path) -> Path:
    absolute = record.get("main_path")
    if absolute is not None and not pd.isna(absolute):
        candidate = Path(str(absolute))
        if candidate.is_file():
            return candidate
    relative = record.get("main_rel_path")
    if relative is None or pd.isna(relative):
        raise FileNotFoundError("Catalog row has no usable main path")
    candidate = root / Path(str(relative))
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def normalize_is_active(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
        "active": 1.0,
        "inactive": 0.0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def changed_value_events(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    work = frame[["Timestamp", value_column]].copy()
    work["Timestamp"] = pd.to_numeric(work["Timestamp"], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna().sort_values("Timestamp", kind="stable")
    if work.empty:
        return work
    previous = work[value_column].shift(1)
    changed = previous.isna() | (work[value_column] != previous)
    return work.loc[changed].drop_duplicates(["Timestamp", value_column]).reset_index(drop=True)


def circular_phase_summary(
    timestamps: Sequence[float],
    *,
    period_seconds: float = NOMINAL_PM_PERIOD_SECONDS,
) -> dict[str, Any]:
    values = np.asarray(timestamps, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "event_count": 0,
            "phase_mean_seconds": None,
            "phase_concentration": None,
            "phase_std_seconds": None,
        }
    phase = np.mod(values, period_seconds)
    angles = 2.0 * np.pi * phase / period_seconds
    vector = np.mean(np.exp(1j * angles))
    concentration = float(np.abs(vector))
    mean_angle = float(np.mod(np.angle(vector), 2.0 * np.pi))
    mean_phase = mean_angle * period_seconds / (2.0 * np.pi)
    circular_std = (
        float(np.sqrt(-2.0 * np.log(max(concentration, 1e-12))) * period_seconds / (2.0 * np.pi))
        if concentration < 1.0
        else 0.0
    )
    return {
        "event_count": int(len(values)),
        "phase_mean_seconds": mean_phase,
        "phase_concentration": concentration,
        "phase_std_seconds": circular_std,
    }


def interval_summary(timestamps: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(timestamps, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    intervals = np.diff(values)
    intervals = intervals[intervals > 0]
    if not len(intervals):
        return {
            "interval_count": 0,
            "interval_median_seconds": None,
            "interval_p90_seconds": None,
            "near_10s_fraction": None,
        }
    return {
        "interval_count": int(len(intervals)),
        "interval_median_seconds": float(np.median(intervals)),
        "interval_p90_seconds": float(np.quantile(intervals, 0.90)),
        "near_10s_fraction": float(np.mean(np.abs(intervals - 10.0) <= 0.25)),
    }


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return None
    first = pair.iloc[:, 0].to_numpy(dtype=float)
    second = pair.iloc[:, 1].to_numpy(dtype=float)
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


@dataclass
class RecordMetricAudit:
    source: str
    subject_id: str
    path: str
    metric: str
    rows_read: int
    raw_scaled_corr: float | None
    isactive_mean: float | None
    scaled_when_inactive_fraction: float | None
    event_count: int
    interval_median_seconds: float | None
    interval_p90_seconds: float | None
    near_10s_fraction: float | None
    phase_mean_seconds: float | None
    phase_concentration: float | None
    phase_std_seconds: float | None


def audit_record_metric(
    record: Mapping[str, Any],
    *,
    metric: str,
    root: Path,
    chunk_size: int,
) -> RecordMetricAudit | None:
    path = _resolve_record_path(record, root)
    header_row, separator, actual_columns = _find_header(path)
    candidates = ["Timestamp", *[pm_column(metric, field) for field in PM_FIELDS]]
    usecols = [column for column in candidates if column in actual_columns]
    scaled_column = pm_column(metric, "Scaled")
    if "Timestamp" not in usecols or scaled_column not in usecols:
        return None

    compression = "bz2" if path.suffix.lower() == ".bz2" else None
    frames: list[pd.DataFrame] = []
    rows_read = 0
    for chunk in pd.read_csv(
        path,
        compression=compression,
        sep=separator,
        header=header_row,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    ):
        rows_read += len(chunk)
        frames.append(chunk)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame["Timestamp"] = pd.to_numeric(frame["Timestamp"], errors="coerce")
    for field in ("Raw", "Scaled", "Min", "Max"):
        column = pm_column(metric, field)
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    active_column = pm_column(metric, "IsActive")
    if active_column in frame:
        frame[active_column] = normalize_is_active(frame[active_column])

    events = changed_value_events(frame, scaled_column)
    timing = {**interval_summary(events["Timestamp"]), **circular_phase_summary(events["Timestamp"])}
    raw_column = pm_column(metric, "Raw")
    raw_scaled_corr = (
        _safe_corr(frame[raw_column], frame[scaled_column])
        if raw_column in frame
        else None
    )
    isactive_mean = None
    scaled_when_inactive_fraction = None
    if active_column in frame:
        active = pd.to_numeric(frame[active_column], errors="coerce")
        finite_active = active.dropna()
        if len(finite_active):
            isactive_mean = float(finite_active.mean())
        scaled_valid = frame[scaled_column].notna() & active.notna()
        if scaled_valid.any():
            scaled_when_inactive_fraction = float(
                (active.loc[scaled_valid] < 0.5).mean()
            )

    return RecordMetricAudit(
        source=str(record.get("source", "unknown")),
        subject_id=str(record.get("subject_id", "unknown")),
        path=str(path),
        metric=metric,
        rows_read=int(rows_read),
        raw_scaled_corr=raw_scaled_corr,
        isactive_mean=isactive_mean,
        scaled_when_inactive_fraction=scaled_when_inactive_fraction,
        event_count=int(timing["event_count"]),
        interval_median_seconds=timing["interval_median_seconds"],
        interval_p90_seconds=timing["interval_p90_seconds"],
        near_10s_fraction=timing["near_10s_fraction"],
        phase_mean_seconds=timing["phase_mean_seconds"],
        phase_concentration=timing["phase_concentration"],
        phase_std_seconds=timing["phase_std_seconds"],
    )


def audit_raw_records(
    catalog: pd.DataFrame,
    *,
    root: Path,
    chunk_size: int,
    max_records: int | None,
) -> pd.DataFrame:
    work = catalog.copy()
    if "status" in work:
        work = work.loc[work["status"].astype(str) == "ok"]
    if max_records is not None:
        work = work.head(int(max_records))
    rows: list[dict[str, Any]] = []
    for _, record in work.iterrows():
        present = discover_pm_columns(_parse_list(record.get("pm_columns")))
        for metric in PM_METRICS:
            if not present[metric]["Scaled"]:
                continue
            audited = audit_record_metric(
                record,
                metric=metric,
                root=root,
                chunk_size=chunk_size,
            )
            if audited is not None:
                rows.append(audited.__dict__)
    return pd.DataFrame(rows)


def summarize_raw_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows: list[dict[str, Any]] = []
    for (source, metric), group in frame.groupby(["source", "metric"], sort=True):
        row: dict[str, Any] = {
            "source": str(source),
            "metric": str(metric),
            "records": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
        }
        for column in (
            "raw_scaled_corr",
            "isactive_mean",
            "scaled_when_inactive_fraction",
            "interval_median_seconds",
            "near_10s_fraction",
            "phase_concentration",
            "phase_std_seconds",
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "metric"]).reset_index(drop=True)


def boundary_distance_fraction(
    values: np.ndarray,
    boundaries: Sequence[float],
    *,
    relative_margin: float,
) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    edges = np.asarray(boundaries, dtype=float)
    if not len(y) or len(edges) < 3:
        return np.nan
    scale = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.ptp(y))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    threshold = float(relative_margin) * scale
    internal = edges[1:-1]
    distance = np.min(np.abs(y[:, None] - internal[None, :]), axis=1)
    return float(np.mean(distance <= threshold))


def q3_boundary_audit(
    frame: pd.DataFrame,
    *,
    margins: Sequence[float] = (0.01, 0.025, 0.05, 0.10),
    n_splits: int = 5,
) -> pd.DataFrame:
    required = {"subject_id", *[f"target_{metric}" for metric in PM_METRICS]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Processed target table is missing columns: {missing}")
    work = ensure_record_group_ids(frame)
    subjects = work["subject_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(work), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(placeholder, groups=subjects), start=1
    ):
        for metric in PM_METRICS:
            column = f"target_{metric}"
            train = pd.to_numeric(work.iloc[train_idx][column], errors="coerce").to_numpy(dtype=float)
            test = pd.to_numeric(work.iloc[test_idx][column], errors="coerce").to_numpy(dtype=float)
            train_finite = train[np.isfinite(train)]
            test_finite = test[np.isfinite(test)]
            if not len(train_finite) or not len(test_finite):
                continue
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(train_finite)
            manifest = transform.manifest()
            if transform.actual_class_count != 3:
                continue
            boundaries = manifest["boundaries"]
            row: dict[str, Any] = {
                "fold": int(fold),
                "metric": metric,
                "train_samples": int(len(train_finite)),
                "test_samples": int(len(test_finite)),
                "boundary_1": float(boundaries[1]),
                "boundary_2": float(boundaries[2]),
            }
            for margin in margins:
                key = f"within_{margin:g}_iqr"
                row[f"train_{key}"] = boundary_distance_fraction(
                    train_finite, boundaries, relative_margin=float(margin)
                )
                row[f"test_{key}"] = boundary_distance_fraction(
                    test_finite, boundaries, relative_margin=float(margin)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(
    inventory_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    boundary: pd.DataFrame,
) -> str:
    lines = [
        "# PM target validity audit",
        "",
        f"Schema: `{AUDIT_SCHEMA_VERSION}`",
        "",
        "This is a training-free diagnostic. It does not modify canonical datasets.",
        "",
        "## PM field inventory",
        "",
    ]
    lines.append(inventory_summary.to_markdown(index=False) if not inventory_summary.empty else "_No inventory rows._")
    lines.extend(["", "## Raw-record PM timing and activity summary", ""])
    lines.append(raw_summary.to_markdown(index=False) if not raw_summary.empty else "_Raw-record audit was not run._")
    lines.extend(["", "## Fold-local Q3 boundary proximity", ""])
    if boundary.empty:
        lines.append("_Boundary audit was not run._")
    else:
        value_columns = [column for column in boundary.columns if column.startswith("test_within_")]
        aggregate = boundary.groupby("metric", sort=True)[value_columns].mean().reset_index()
        lines.append(aggregate.to_markdown(index=False))
    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- Low `raw_scaled_corr` suggests that individual scaling materially changes the target representation.",
        "- High `scaled_when_inactive_fraction` suggests that exported targets may include detector-inactive periods.",
        "- High `phase_concentration` with a stable non-zero phase suggests that PM updates are periodic but not aligned to absolute 10-second window boundaries.",
        "- High Q3 boundary proximity means hard three-class labels are sensitive to small continuous-target perturbations.",
        "",
    ])
    return "\n".join(lines)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def run_audit(
    *,
    root: Path,
    catalog_path: Path,
    processed_path: Path | None,
    output_dir: Path,
    chunk_size: int,
    max_records: int | None,
    skip_raw: bool,
) -> dict[str, Any]:
    catalog = pd.read_csv(catalog_path)
    inventory = build_catalog_inventory(catalog)
    inventory_summary = summarize_catalog_inventory(inventory)
    raw = (
        pd.DataFrame()
        if skip_raw
        else audit_raw_records(
            catalog, root=root, chunk_size=chunk_size, max_records=max_records
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
    report = render_markdown(inventory_summary, raw_summary, boundary)
    (output_dir / "pm_target_validity_audit.md").write_text(report, encoding="utf-8")

    summary = {
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
