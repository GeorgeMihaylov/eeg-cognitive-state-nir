# -*- coding: utf-8 -*-
"""
05_analyze_pm_sampling.py

Описательная статистика PM-метрик Emotiv.

Задачи:
1. Прочитать все Emotiv-записи из каталога.
2. Для PM.*.Scaled и PM.*.IsActive посчитать:
   - количество валидных значений;
   - первый/последний timestamp;
   - интервалы между валидными PM-точками;
   - mean/median/std/min/max/p75/p90/p95/p99 интервала;
   - долю валидных значений;
   - активность PM.IsActive.
3. Сравнить gpn_data и Old_EEG.
4. Дать рекомендацию по размеру окна.

Вход:
    data/interim/emotiv_record_catalog.csv
    data/interim/validated_columns.json

Выход:
    data/interim/pm_sampling_record_stats.csv
    data/interim/pm_sampling_metric_stats.csv
    reports/pm_sampling_report.md

Запуск:
    python src/05_analyze_pm_sampling.py

Быстрый тест:
    python src/05_analyze_pm_sampling.py --max-records 10

Только один источник:
    python src/05_analyze_pm_sampling.py --source gpn_data
    python src/05_analyze_pm_sampling.py --source Old_EEG
"""

from __future__ import annotations

import argparse
import bz2
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_COL = "Timestamp"

DEFAULT_PM_SCALED_COLUMNS = [
    "PM.Attention.Scaled",
    "PM.Engagement.Scaled",
    "PM.Excitement.Scaled",
    "PM.Stress.Scaled",
    "PM.Relaxation.Scaled",
    "PM.Interest.Scaled",
    "PM.Focus.Scaled",
]

DEFAULT_PM_ACTIVE_COLUMNS = [
    "PM.Attention.IsActive",
    "PM.Engagement.IsActive",
    "PM.Excitement.IsActive",
    "PM.Stress.IsActive",
    "PM.Relaxation.IsActive",
    "PM.Interest.IsActive",
    "PM.Focus.IsActive",
]


def open_text(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, mode="rt", encoding="utf-8", errors="replace")


def find_header_row(path: Path, max_lines: int = 30) -> Tuple[int, str, List[str]]:
    with open_text(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            s = line.strip()
            if not s:
                continue

            if "Timestamp" in s and "PM." in s:
                sep = "," if s.count(",") >= s.count(";") else ";"
                columns = [c.strip() for c in s.split(sep)]
                return i, sep, columns

    raise RuntimeError(f"Не найдена строка заголовка в файле: {path}")


def read_validated_pm_columns(path: Path) -> Tuple[List[str], List[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rec = obj.get("recommended", {})

    pm_scaled = rec.get("pm_scaled_columns", DEFAULT_PM_SCALED_COLUMNS)
    pm_active = rec.get("pm_active_columns", DEFAULT_PM_ACTIVE_COLUMNS)

    return pm_scaled, pm_active


def normalize_active(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

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

    return s.astype(str).str.strip().str.lower().map(mapping)


def safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return np.nan
    return float(np.nanquantile(x, q))


def summarize_intervals(intervals: np.ndarray) -> Dict[str, float]:
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[intervals >= 0]

    if intervals.size == 0:
        return {
            "interval_count": 0,
            "interval_mean_s": np.nan,
            "interval_std_s": np.nan,
            "interval_min_s": np.nan,
            "interval_p25_s": np.nan,
            "interval_median_s": np.nan,
            "interval_p75_s": np.nan,
            "interval_p90_s": np.nan,
            "interval_p95_s": np.nan,
            "interval_p99_s": np.nan,
            "interval_max_s": np.nan,
        }

    return {
        "interval_count": int(intervals.size),
        "interval_mean_s": float(np.mean(intervals)),
        "interval_std_s": float(np.std(intervals)),
        "interval_min_s": float(np.min(intervals)),
        "interval_p25_s": safe_quantile(intervals, 0.25),
        "interval_median_s": safe_quantile(intervals, 0.50),
        "interval_p75_s": safe_quantile(intervals, 0.75),
        "interval_p90_s": safe_quantile(intervals, 0.90),
        "interval_p95_s": safe_quantile(intervals, 0.95),
        "interval_p99_s": safe_quantile(intervals, 0.99),
        "interval_max_s": float(np.max(intervals)),
    }


def infer_record_id(row: pd.Series) -> str:
    source = str(row.get("source", "unknown"))
    subject_id = str(row.get("subject_id", "unknown"))
    day = str(row.get("day", "unknown"))
    part = str(row.get("part", ""))
    dt = str(row.get("datetime_from_name", ""))

    if part.lower() in {"nan", "none"}:
        part = ""

    raw = "__".join([source, subject_id, day, part, dt])
    raw = raw.replace(" ", "_").replace(":", "-").replace("\\", "_").replace("/", "_").replace("+", "p")
    return raw.strip("_")


def analyze_one_record(
    row: pd.Series,
    root: Path,
    pm_scaled_cols: List[str],
    pm_active_cols: List[str],
    chunk_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    main_path = Path(str(row["main_path"]))
    if not main_path.is_absolute():
        main_path = root / main_path

    record_id = infer_record_id(row)

    record_meta = {
        "record_id": record_id,
        "source": row.get("source"),
        "subject_id": row.get("subject_id"),
        "day": row.get("day"),
        "part": row.get("part"),
        "main_path": str(main_path),
        "status": "unknown",
        "error": None,
        "n_rows_total": 0,
        "timestamp_min": np.nan,
        "timestamp_max": np.nan,
        "duration_s": np.nan,
    }

    metric_buffers: Dict[str, List[pd.DataFrame]] = {
        c: [] for c in pm_scaled_cols + pm_active_cols
    }

    try:
        header_row, sep, actual_columns = find_header_row(main_path)

        usecols = [TIME_COL]
        for c in pm_scaled_cols + pm_active_cols:
            if c in actual_columns:
                usecols.append(c)

        usecols = list(dict.fromkeys(usecols))

        if TIME_COL not in usecols:
            raise RuntimeError(f"Нет {TIME_COL}: {main_path}")

        compression = "bz2" if main_path.suffix.lower() == ".bz2" else None

        reader = pd.read_csv(
            main_path,
            compression=compression,
            sep=sep,
            header=header_row,
            usecols=usecols,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines="skip",
        )

        for chunk in reader:
            if chunk.empty:
                continue

            record_meta["n_rows_total"] += int(len(chunk))

            chunk[TIME_COL] = pd.to_numeric(chunk[TIME_COL], errors="coerce")
            chunk = chunk.dropna(subset=[TIME_COL])

            if chunk.empty:
                continue

            ts = chunk[TIME_COL].astype(float)

            tmin = float(ts.min())
            tmax = float(ts.max())

            if math.isnan(record_meta["timestamp_min"]):
                record_meta["timestamp_min"] = tmin
                record_meta["timestamp_max"] = tmax
            else:
                record_meta["timestamp_min"] = min(record_meta["timestamp_min"], tmin)
                record_meta["timestamp_max"] = max(record_meta["timestamp_max"], tmax)

            for c in pm_scaled_cols:
                if c not in chunk.columns:
                    continue

                y = pd.to_numeric(chunk[c], errors="coerce")
                m = y.notna()

                if m.any():
                    metric_buffers[c].append(
                        pd.DataFrame({
                            "timestamp": ts[m].to_numpy(dtype=float),
                            "value": y[m].to_numpy(dtype=float),
                        })
                    )

            for c in pm_active_cols:
                if c not in chunk.columns:
                    continue

                y = normalize_active(chunk[c])
                m = y.notna()

                if m.any():
                    metric_buffers[c].append(
                        pd.DataFrame({
                            "timestamp": ts[m].to_numpy(dtype=float),
                            "value": y[m].to_numpy(dtype=float),
                        })
                    )

        if not math.isnan(record_meta["timestamp_min"]) and not math.isnan(record_meta["timestamp_max"]):
            record_meta["duration_s"] = float(record_meta["timestamp_max"] - record_meta["timestamp_min"])

        rows = []

        for metric_col, parts in metric_buffers.items():
            base_row = {
                "record_id": record_id,
                "source": row.get("source"),
                "subject_id": row.get("subject_id"),
                "day": row.get("day"),
                "part": row.get("part"),
                "metric": metric_col,
                "metric_type": "active" if metric_col.endswith(".IsActive") else "scaled",
                "record_duration_s": record_meta["duration_s"],
                "record_rows_total": record_meta["n_rows_total"],
            }

            if not parts:
                r = dict(base_row)
                r.update({
                    "valid_count": 0,
                    "valid_ratio_rows": 0.0,
                    "first_valid_timestamp": np.nan,
                    "last_valid_timestamp": np.nan,
                    "valid_span_s": np.nan,
                    "value_mean": np.nan,
                    "value_std": np.nan,
                    "value_min": np.nan,
                    "value_p25": np.nan,
                    "value_median": np.nan,
                    "value_p75": np.nan,
                    "value_max": np.nan,
                    "active_ratio": np.nan,
                })
                r.update(summarize_intervals(np.array([], dtype=float)))
                rows.append(r)
                continue

            mdf = pd.concat(parts, ignore_index=True)
            mdf = mdf.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

            # Часто PM-значение повторяется во многих строках между реальными обновлениями.
            # Чтобы оценить реальный шаг PM, убираем последовательные дубликаты значения.
            mdf["value_prev"] = mdf["value"].shift(1)
            mdf["timestamp_prev"] = mdf["timestamp"].shift(1)
            changed = (mdf["value"] != mdf["value_prev"]) | mdf["value_prev"].isna()
            event_df = mdf[changed].copy()

            # Дополнительно убираем одинаковые timestamp.
            event_df = event_df.drop_duplicates(subset=["timestamp", "value"])

            ts_event = event_df["timestamp"].to_numpy(dtype=float)
            values_event = event_df["value"].to_numpy(dtype=float)

            intervals = np.diff(ts_event)

            r = dict(base_row)
            r.update({
                "valid_count": int(len(event_df)),
                "valid_ratio_rows": float(len(mdf) / max(1, record_meta["n_rows_total"])),
                "first_valid_timestamp": float(ts_event[0]) if len(ts_event) else np.nan,
                "last_valid_timestamp": float(ts_event[-1]) if len(ts_event) else np.nan,
                "valid_span_s": float(ts_event[-1] - ts_event[0]) if len(ts_event) >= 2 else np.nan,
                "value_mean": float(np.nanmean(values_event)) if len(values_event) else np.nan,
                "value_std": float(np.nanstd(values_event)) if len(values_event) else np.nan,
                "value_min": float(np.nanmin(values_event)) if len(values_event) else np.nan,
                "value_p25": safe_quantile(values_event, 0.25),
                "value_median": safe_quantile(values_event, 0.50),
                "value_p75": safe_quantile(values_event, 0.75),
                "value_max": float(np.nanmax(values_event)) if len(values_event) else np.nan,
                "active_ratio": float(np.nanmean(values_event)) if metric_col.endswith(".IsActive") and len(values_event) else np.nan,
            })
            r.update(summarize_intervals(intervals))
            rows.append(r)

        record_meta["status"] = "ok"
        return rows, record_meta

    except Exception as e:
        record_meta["status"] = "failed"
        record_meta["error"] = str(e)
        record_meta["traceback"] = traceback.format_exc(limit=2)
        return [], record_meta


def aggregate_metric_stats(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["source", "metric", "metric_type"]

    for keys, g in metric_df.groupby(group_cols, dropna=False):
        source, metric, metric_type = keys

        valid = g[g["valid_count"] > 1].copy()

        row = {
            "source": source,
            "metric": metric,
            "metric_type": metric_type,
            "records": int(g["record_id"].nunique()),
            "records_with_valid": int(valid["record_id"].nunique()),
            "valid_count_sum": int(g["valid_count"].sum()),
            "valid_count_median_per_record": float(g["valid_count"].median()),
            "record_duration_median_s": float(g["record_duration_s"].median()),
            "value_mean_across_records": float(g["value_mean"].mean()),
            "value_median_across_records": float(g["value_median"].median()),
            "active_ratio_mean": float(g["active_ratio"].mean()) if metric_type == "active" else np.nan,
        }

        for stat_col in [
            "interval_mean_s",
            "interval_median_s",
            "interval_p75_s",
            "interval_p90_s",
            "interval_p95_s",
            "interval_p99_s",
            "interval_max_s",
        ]:
            row[f"{stat_col}_median_across_records"] = float(valid[stat_col].median()) if not valid.empty else np.nan
            row[f"{stat_col}_mean_across_records"] = float(valid[stat_col].mean()) if not valid.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["source", "metric_type", "metric"]).reset_index(drop=True)


def recommend_windows(metric_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Простая рекомендация:
    - берем scaled PM-метрики;
    - смотрим median across records от interval_p90/p95;
    - окно должно быть не меньше p90 или p95 интервала, если хотим чаще получать PM внутри окна.
    """
    if metric_stats.empty:
        return pd.DataFrame()

    scaled = metric_stats[metric_stats["metric_type"] == "scaled"].copy()

    rows = []

    for source, g in scaled.groupby("source", dropna=False):
        p50 = float(g["interval_median_s_median_across_records"].median())
        p75 = float(g["interval_p75_s_median_across_records"].median())
        p90 = float(g["interval_p90_s_median_across_records"].median())
        p95 = float(g["interval_p95_s_median_across_records"].median())

        rows.append({
            "source": source,
            "pm_interval_median_s": p50,
            "pm_interval_p75_s": p75,
            "pm_interval_p90_s": p90,
            "pm_interval_p95_s": p95,
            "recommended_window_conservative_s": round(p95, 2) if np.isfinite(p95) else np.nan,
            "recommended_window_balanced_s": round(p90, 2) if np.isfinite(p90) else np.nan,
            "recommended_window_fast_s": round(p75, 2) if np.isfinite(p75) else np.nan,
        })

    all_scaled = scaled.copy()
    if not all_scaled.empty:
        p50 = float(all_scaled["interval_median_s_median_across_records"].median())
        p75 = float(all_scaled["interval_p75_s_median_across_records"].median())
        p90 = float(all_scaled["interval_p90_s_median_across_records"].median())
        p95 = float(all_scaled["interval_p95_s_median_across_records"].median())

        rows.append({
            "source": "all",
            "pm_interval_median_s": p50,
            "pm_interval_p75_s": p75,
            "pm_interval_p90_s": p90,
            "pm_interval_p95_s": p95,
            "recommended_window_conservative_s": round(p95, 2) if np.isfinite(p95) else np.nan,
            "recommended_window_balanced_s": round(p90, 2) if np.isfinite(p90) else np.nan,
            "recommended_window_fast_s": round(p75, 2) if np.isfinite(p75) else np.nan,
        })

    return pd.DataFrame(rows)


def make_markdown_report(
    record_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric_stats: pd.DataFrame,
    recommendations: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []

    lines.append("# PM sampling statistics report")
    lines.append("")

    lines.append("## Record processing status")
    lines.append("")
    if record_df.empty:
        lines.append("_No records._")
    else:
        lines.append(record_df["status"].value_counts(dropna=False).to_frame("count").to_markdown())
    lines.append("")

    lines.append("## Records by source")
    lines.append("")
    if not record_df.empty:
        lines.append(record_df["source"].value_counts(dropna=False).to_frame("records").to_markdown())
    lines.append("")

    lines.append("## Metric-level recommendation")
    lines.append("")
    if recommendations.empty:
        lines.append("_No recommendations._")
    else:
        lines.append(recommendations.to_markdown(index=False))
    lines.append("")

    lines.append("## Aggregated PM interval statistics")
    lines.append("")
    if metric_stats.empty:
        lines.append("_No metric statistics._")
    else:
        show_cols = [
            "source",
            "metric",
            "metric_type",
            "records",
            "records_with_valid",
            "valid_count_median_per_record",
            "interval_median_s_median_across_records",
            "interval_p75_s_median_across_records",
            "interval_p90_s_median_across_records",
            "interval_p95_s_median_across_records",
            "value_mean_across_records",
            "active_ratio_mean",
        ]
        show_cols = [c for c in show_cols if c in metric_stats.columns]
        lines.append(metric_stats[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Per-record PM statistics preview")
    lines.append("")
    if metric_df.empty:
        lines.append("_No per-record metric statistics._")
    else:
        show_cols = [
            "source",
            "subject_id",
            "day",
            "metric",
            "metric_type",
            "valid_count",
            "interval_median_s",
            "interval_p90_s",
            "interval_p95_s",
            "value_mean",
            "active_ratio",
        ]
        show_cols = [c for c in show_cols if c in metric_df.columns]
        lines.append(metric_df[show_cols].head(80).to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. `interval_median_s` shows the typical interval between real PM updates.")
    lines.append("2. `interval_p90_s` and `interval_p95_s` are more useful for selecting a robust window size.")
    lines.append("3. If the chosen window is much shorter than the PM interval, many windows will have no PM target.")
    lines.append("4. For a first baseline, use the balanced or conservative recommendation.")
    lines.append("5. For a later real-time prototype, use shorter windows plus forward-fill/nearest PM assignment.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\PycharmProjects\eeg-cognitive-state-nir",
        help="Корень проекта.",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=r"data\interim\emotiv_record_catalog.csv",
    )
    parser.add_argument(
        "--validated-columns",
        type=str,
        default=r"data\interim\validated_columns.json",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "gpn_data", "Old_EEG"],
        default="all",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300_000,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path

    validated_path = Path(args.validated_columns)
    if not validated_path.is_absolute():
        validated_path = root / validated_path

    interim_dir = root / "data" / "interim"
    reports_dir = root / "reports"
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(catalog_path)
    catalog = catalog[catalog["status"] == "ok"].copy()

    if args.source != "all":
        catalog = catalog[catalog["source"] == args.source].copy()

    if args.max_records is not None:
        catalog = catalog.head(args.max_records).copy()

    pm_scaled_cols, pm_active_cols = read_validated_pm_columns(validated_path)

    print("=" * 80)
    print("Analyze PM sampling")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Records: {len(catalog)}")
    print(f"Source: {args.source}")
    print(f"PM scaled columns: {pm_scaled_cols}")
    print(f"PM active columns: {pm_active_cols}")

    all_metric_rows = []
    record_rows = []

    for i, (_, row) in enumerate(catalog.iterrows(), start=1):
        print("-" * 80)
        print(f"[{i}/{len(catalog)}] {row['source']} | {row['subject_id']} | {row['day']} | {row['main_rel_path']}")

        metric_rows, record_meta = analyze_one_record(
            row=row,
            root=root,
            pm_scaled_cols=pm_scaled_cols,
            pm_active_cols=pm_active_cols,
            chunk_size=args.chunk_size,
        )

        all_metric_rows.extend(metric_rows)
        record_rows.append(record_meta)

        print(
            f"status={record_meta['status']} | rows={record_meta['n_rows_total']} | "
            f"duration_s={record_meta['duration_s']}"
        )

    record_df = pd.DataFrame(record_rows)
    metric_df = pd.DataFrame(all_metric_rows)
    metric_stats = aggregate_metric_stats(metric_df)
    recommendations = recommend_windows(metric_stats)

    record_path = interim_dir / "pm_sampling_record_stats.csv"
    metric_path = interim_dir / "pm_sampling_metric_record_stats.csv"
    stats_path = interim_dir / "pm_sampling_metric_stats.csv"
    rec_path = interim_dir / "pm_sampling_window_recommendations.csv"
    md_path = reports_dir / "pm_sampling_report.md"

    record_df.to_csv(record_path, index=False, encoding="utf-8-sig")
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    metric_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    recommendations.to_csv(rec_path, index=False, encoding="utf-8-sig")

    make_markdown_report(
        record_df=record_df,
        metric_df=metric_df,
        metric_stats=metric_stats,
        recommendations=recommendations,
        out_path=md_path,
    )

    print("\nSaved:")
    print(f"  {record_path}")
    print(f"  {metric_path}")
    print(f"  {stats_path}")
    print(f"  {rec_path}")
    print(f"  {md_path}")

    print("\nRecommendations:")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()