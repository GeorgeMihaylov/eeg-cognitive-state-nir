"""Single-pass raw-record statistics for the PM target validity audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.tasks.target_registry import PM_METRICS
from bench.analysis.pm_target_validity_audit import (
    PM_FIELDS,
    _find_header,
    _parse_list,
    _resolve_record_path,
    circular_phase_summary,
    discover_pm_columns,
    interval_summary,
    normalize_is_active,
    pm_column,
)


@dataclass
class _MetricAccumulator:
    metric: str
    previous_scaled: float | None = None
    event_timestamps: list[float] = field(default_factory=list)
    corr_n: int = 0
    corr_sum_x: float = 0.0
    corr_sum_y: float = 0.0
    corr_sum_xx: float = 0.0
    corr_sum_yy: float = 0.0
    corr_sum_xy: float = 0.0
    active_n: int = 0
    active_sum: float = 0.0
    scaled_active_pair_n: int = 0
    scaled_inactive_n: int = 0

    def update(self, chunk: pd.DataFrame) -> None:
        timestamp = pd.to_numeric(chunk["Timestamp"], errors="coerce")
        scaled_col = pm_column(self.metric, "Scaled")
        scaled = pd.to_numeric(chunk[scaled_col], errors="coerce")

        # Preserve event continuity across pandas chunks. PM values are commonly
        # repeated on many EEG rows between real updates.
        finite = timestamp.notna() & scaled.notna()
        for ts, value in zip(
            timestamp.loc[finite].to_numpy(dtype=float),
            scaled.loc[finite].to_numpy(dtype=float),
        ):
            if self.previous_scaled is None or value != self.previous_scaled:
                self.event_timestamps.append(float(ts))
                self.previous_scaled = float(value)

        raw_col = pm_column(self.metric, "Raw")
        if raw_col in chunk:
            raw = pd.to_numeric(chunk[raw_col], errors="coerce")
            pair = raw.notna() & scaled.notna()
            if pair.any():
                x = raw.loc[pair].to_numpy(dtype=np.float64)
                y = scaled.loc[pair].to_numpy(dtype=np.float64)
                self.corr_n += int(len(x))
                self.corr_sum_x += float(x.sum())
                self.corr_sum_y += float(y.sum())
                self.corr_sum_xx += float(np.dot(x, x))
                self.corr_sum_yy += float(np.dot(y, y))
                self.corr_sum_xy += float(np.dot(x, y))

        active_col = pm_column(self.metric, "IsActive")
        if active_col in chunk:
            active = normalize_is_active(chunk[active_col])
            valid_active = active.notna()
            if valid_active.any():
                values = active.loc[valid_active].to_numpy(dtype=float)
                self.active_n += int(len(values))
                self.active_sum += float(values.sum())
            pair = scaled.notna() & active.notna()
            if pair.any():
                pair_active = active.loc[pair].to_numpy(dtype=float)
                self.scaled_active_pair_n += int(len(pair_active))
                self.scaled_inactive_n += int(np.sum(pair_active < 0.5))

    def correlation(self) -> float | None:
        n = self.corr_n
        if n < 3:
            return None
        numerator = n * self.corr_sum_xy - self.corr_sum_x * self.corr_sum_y
        denominator_x = n * self.corr_sum_xx - self.corr_sum_x**2
        denominator_y = n * self.corr_sum_yy - self.corr_sum_y**2
        denominator = float(np.sqrt(max(0.0, denominator_x) * max(0.0, denominator_y)))
        if denominator <= 0:
            return None
        value = float(numerator / denominator)
        return value if np.isfinite(value) else None

    def result(self, *, source: str, subject_id: str, path: Path, rows_read: int) -> dict[str, Any]:
        timing = {
            **interval_summary(self.event_timestamps),
            **circular_phase_summary(self.event_timestamps),
        }
        return {
            "source": source,
            "subject_id": subject_id,
            "path": str(path),
            "metric": self.metric,
            "rows_read": int(rows_read),
            "raw_scaled_corr": self.correlation(),
            "isactive_mean": (
                self.active_sum / self.active_n if self.active_n else None
            ),
            "scaled_when_inactive_fraction": (
                self.scaled_inactive_n / self.scaled_active_pair_n
                if self.scaled_active_pair_n
                else None
            ),
            "event_count": int(timing["event_count"]),
            "interval_median_seconds": timing["interval_median_seconds"],
            "interval_p90_seconds": timing["interval_p90_seconds"],
            "near_10s_fraction": timing["near_10s_fraction"],
            "phase_mean_seconds": timing["phase_mean_seconds"],
            "phase_concentration": timing["phase_concentration"],
            "phase_std_seconds": timing["phase_std_seconds"],
        }


def audit_one_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    chunk_size: int,
) -> list[dict[str, Any]]:
    path = _resolve_record_path(record, root)
    header_row, separator, actual_columns = _find_header(path)
    actual = set(actual_columns)
    catalog_presence = discover_pm_columns(_parse_list(record.get("pm_columns")))
    metrics = [
        metric
        for metric in PM_METRICS
        if catalog_presence[metric]["Scaled"]
        and pm_column(metric, "Scaled") in actual
    ]
    if not metrics:
        return []

    usecols = ["Timestamp"]
    for metric in metrics:
        for field_name in PM_FIELDS:
            column = pm_column(metric, field_name)
            if column in actual:
                usecols.append(column)
    usecols = list(dict.fromkeys(usecols))
    accumulators = {metric: _MetricAccumulator(metric) for metric in metrics}
    rows_read = 0
    compression = "bz2" if path.suffix.lower() == ".bz2" else None
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
        rows_read += int(len(chunk))
        for accumulator in accumulators.values():
            accumulator.update(chunk)

    source = str(record.get("source", "unknown"))
    subject_id = str(record.get("subject_id", "unknown"))
    return [
        accumulator.result(
            source=source,
            subject_id=subject_id,
            path=path,
            rows_read=rows_read,
        )
        for accumulator in accumulators.values()
    ]


def audit_raw_records_streaming(
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
        rows.extend(
            audit_one_record(record, root=root, chunk_size=chunk_size)
        )
    return pd.DataFrame(rows)
