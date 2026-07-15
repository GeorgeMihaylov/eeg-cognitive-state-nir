# -*- coding: utf-8 -*-
"""
08_build_eeg_features.py

Строит признаки из сырых EEG-каналов Emotiv и объединяет их
с уже построенным windowed PM/POW датасетом.

Вход:
    data/interim/emotiv_record_catalog.csv
    data/processed/windowed_pm_dataset_w10.parquet

Выход:
    data/interim/windowed_eeg_features_w10.parquet
    data/interim/windowed_eeg_features_w10_record_report.csv
    data/processed/windowed_eeg_pm_dataset_w10.parquet
    data/processed/windowed_eeg_pm_dataset_w10.csv
    reports/windowed_eeg_features_w10_report.md

Запуск:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\08_build_eeg_features.py

Быстрый тест:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\08_build_eeg_features.py --max-records 5 --output-name windowed_eeg_pm_dataset_w10_test

Только gpn_data:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\08_build_eeg_features.py --source gpn_data

Только Old_EEG:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\08_build_eeg_features.py --source Old_EEG
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

EEG_SIGNAL_CHANNELS = [
    "EEG.AF3",
    "EEG.F7",
    "EEG.F3",
    "EEG.FC5",
    "EEG.T7",
    "EEG.P7",
    "EEG.O1",
    "EEG.O2",
    "EEG.P8",
    "EEG.T8",
    "EEG.FC6",
    "EEG.F4",
    "EEG.F8",
    "EEG.AF4",
]

DEFAULT_WINDOW_S = 10.0
RANDOM_STATE = 42


def open_text(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, mode="rt", encoding="utf-8", errors="replace")


def find_header_row(path: Path, max_lines: int = 40) -> Tuple[int, str, List[str]]:
    """
    Ищет строку заголовка в Emotiv CSV / CSV.BZ2.

    Для новых gpn_data обычно header_row=1.
    Для Old_EEG до таблицы могут идти строки метаданных.
    """
    with open_text(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            s = line.strip()
            if not s:
                continue

            if "Timestamp" in s and "EEG." in s:
                sep = "," if s.count(",") >= s.count(";") else ";"
                columns = [c.strip() for c in s.split(sep)]
                return i, sep, columns

    raise RuntimeError(f"Не найдена строка заголовка в файле: {path}")


def infer_record_id(row: pd.Series) -> str:
    source = str(row.get("source", "unknown"))
    subject_id = str(row.get("subject_id", "unknown"))
    day = str(row.get("day", "unknown"))
    part = str(row.get("part", ""))
    dt = str(row.get("datetime_from_name", ""))

    if part.lower() in {"nan", "none"}:
        part = ""

    raw = "__".join([source, subject_id, day, part, dt])
    raw = raw.replace(" ", "_").replace(":", "-").replace("\\", "_").replace("/", "_")
    raw = raw.replace("+", "p")
    return raw.strip("_")


def robust_iqr(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(x.quantile(0.75) - x.quantile(0.25))


def zero_crossing_rate(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan

    arr = arr - np.nanmean(arr)
    signs = np.sign(arr)
    signs[signs == 0] = 1
    zc = np.sum(signs[1:] != signs[:-1])
    return float(zc / max(1, arr.size - 1))


def line_length(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(arr))))


def signal_energy(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr ** 2))


def mean_abs_diff(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(arr))))


def compute_chunk_eeg_features(
    chunk: pd.DataFrame,
    eeg_cols: List[str],
    win_s: float,
) -> pd.DataFrame:
    """
    Считает признаки EEG по абсолютным window_id.

    Важно:
    - Используем absolute window id = floor(Timestamp / win_s),
      чтобы окна совпадали с ранее построенным PM/POW датасетом.
    - Вторичная агрегация одинаковых окон между чанками будет ниже.
    """
    chunk = chunk.copy()

    chunk[TIME_COL] = pd.to_numeric(chunk[TIME_COL], errors="coerce")
    chunk = chunk.dropna(subset=[TIME_COL])

    if chunk.empty:
        return pd.DataFrame()

    for c in eeg_cols:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    ts = chunk[TIME_COL].astype(float)
    chunk["_window_id_abs"] = np.floor(ts / win_s).astype("int64")
    chunk["_t_center_abs"] = (chunk["_window_id_abs"].astype(float) + 0.5) * win_s

    agg_spec = {}

    for c in eeg_cols:
        if c not in chunk.columns:
            continue

        agg_spec[c] = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            robust_iqr,
            "skew",
            pd.Series.kurt,
            signal_energy,
            zero_crossing_rate,
            line_length,
            mean_abs_diff,
        ]

    if not agg_spec:
        return pd.DataFrame()

    g = chunk.groupby("_window_id_abs").agg(agg_spec)

    new_cols = []
    for base, stat in g.columns:
        if callable(stat):
            stat_name = getattr(stat, "__name__", str(stat))
        else:
            stat_name = str(stat)

        stat_name = stat_name.replace("<lambda>", "lambda")
        new_cols.append(f"{base}__{stat_name}")

    g.columns = new_cols
    g = g.reset_index()
    g["t_center_abs"] = (g["_window_id_abs"].astype(float) + 0.5) * win_s

    return g


def read_and_build_eeg_features_for_record(
    row: pd.Series,
    root: Path,
    eeg_cols: List[str],
    win_s: float,
    chunk_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    main_path = Path(str(row["main_path"]))
    if not main_path.is_absolute():
        main_path = root / main_path

    record_id = infer_record_id(row)

    meta = {
        "record_id": record_id,
        "source": row.get("source"),
        "subject_id": row.get("subject_id"),
        "day": row.get("day"),
        "part": row.get("part"),
        "main_path": str(main_path),
        "status": "unknown",
        "error": None,
        "n_input_rows": 0,
        "n_output_windows": 0,
        "timestamp_min": np.nan,
        "timestamp_max": np.nan,
        "duration_s": np.nan,
        "available_eeg_channels": None,
        "missing_eeg_channels": None,
    }

    try:
        header_row, sep, actual_columns = find_header_row(main_path)

        available_eeg = [c for c in eeg_cols if c in actual_columns]
        missing_eeg = [c for c in eeg_cols if c not in actual_columns]

        meta["available_eeg_channels"] = json.dumps(available_eeg, ensure_ascii=False)
        meta["missing_eeg_channels"] = json.dumps(missing_eeg, ensure_ascii=False)

        if not available_eeg:
            meta["status"] = "no_eeg_channels"
            return pd.DataFrame(), meta

        usecols = [TIME_COL] + available_eeg
        compression = "bz2" if main_path.suffix.lower() == ".bz2" else None

        parts = []

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

            meta["n_input_rows"] += int(len(chunk))

            ts = pd.to_numeric(chunk[TIME_COL], errors="coerce").dropna()
            if not ts.empty:
                tmin = float(ts.min())
                tmax = float(ts.max())

                if math.isnan(meta["timestamp_min"]):
                    meta["timestamp_min"] = tmin
                    meta["timestamp_max"] = tmax
                else:
                    meta["timestamp_min"] = min(meta["timestamp_min"], tmin)
                    meta["timestamp_max"] = max(meta["timestamp_max"], tmax)

            feat = compute_chunk_eeg_features(
                chunk=chunk,
                eeg_cols=available_eeg,
                win_s=win_s,
            )

            if not feat.empty:
                parts.append(feat)

        if not math.isnan(meta["timestamp_min"]) and not math.isnan(meta["timestamp_max"]):
            meta["duration_s"] = float(meta["timestamp_max"] - meta["timestamp_min"])

        if not parts:
            meta["status"] = "empty_after_processing"
            return pd.DataFrame(), meta

        tmp = pd.concat(parts, ignore_index=True)

        feature_cols = [c for c in tmp.columns if c not in {"_window_id_abs", "t_center_abs"}]

        # Чанки могут пересекаться по window_id на границах.
        # Для mean/std/min/... вторичная агрегация mean является приближением.
        # Для первого baseline это допустимо; позже можно сделать online-агрегатор с накоплением достаточных статистик.
        second_agg = {c: "mean" for c in feature_cols}
        second_agg["t_center_abs"] = "mean"

        out = tmp.groupby("_window_id_abs", as_index=False).agg(second_agg)

        t0_abs = float(out["t_center_abs"].min())
        out["t_center"] = out["t_center_abs"] - t0_abs
        out["t_start"] = out["t_center"] - win_s / 2.0
        out["t_end"] = out["t_center"] + win_s / 2.0

        meta_cols = pd.DataFrame(
            {
                "record_id": record_id,
                "source": row.get("source"),
                "subject_id": row.get("subject_id"),
                "day": row.get("day"),
                "part": row.get("part"),
                "datetime_from_name": row.get("datetime_from_name"),
            },
            index=out.index,
        )

        out = pd.concat([meta_cols, out], axis=1)
        out = out.drop(columns=["_window_id_abs", "t_center_abs"], errors="ignore")
        out = out.copy()

        meta["status"] = "ok"
        meta["n_output_windows"] = int(len(out))

        return out, meta

    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
        meta["traceback"] = traceback.format_exc(limit=2)
        return pd.DataFrame(), meta


def filter_catalog(
    catalog: pd.DataFrame,
    source: str,
    max_records: Optional[int],
) -> pd.DataFrame:
    out = catalog.copy()
    out = out[out["status"] == "ok"].copy()

    if source != "all":
        out = out[out["source"] == source].copy()

    out = out.sort_values(["source", "subject_id", "day", "main_rel_path"]).reset_index(drop=True)

    if max_records is not None:
        out = out.head(max_records).copy()

    return out


def merge_eeg_with_pm_dataset(
    pm_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    win_s: float,
) -> pd.DataFrame:
    """
    Объединяет PM/POW датасет и EEG-признаки.

    Используем ключ:
        record_id + t_center rounded

    Это устойчивее, чем float-merge напрямую.
    """
    pm = pm_df.copy()
    eeg = eeg_df.copy()

    pm["_t_key"] = np.round(pd.to_numeric(pm["t_center"], errors="coerce") / win_s).astype("Int64")
    eeg["_t_key"] = np.round(pd.to_numeric(eeg["t_center"], errors="coerce") / win_s).astype("Int64")

    base_keys = ["record_id", "_t_key"]

    # Удаляем дублирующие служебные поля из EEG-части, кроме ключей.
    duplicate_meta = [
        "source",
        "subject_id",
        "day",
        "part",
        "datetime_from_name",
        "t_center",
        "t_start",
        "t_end",
    ]
    eeg = eeg.drop(columns=[c for c in duplicate_meta if c in eeg.columns], errors="ignore")

    merged = pm.merge(
        eeg,
        on=base_keys,
        how="left",
        suffixes=("", "_eeg"),
    )

    merged = merged.drop(columns=["_t_key"], errors="ignore")
    merged = merged.copy()

    return merged


def missingness_summary(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(prefix)]
    rows = []

    for c in cols:
        rows.append(
            {
                "column": c,
                "missing_count": int(df[c].isna().sum()),
                "missing_ratio": float(df[c].isna().mean()),
                "non_null_count": int(df[c].notna().sum()),
                "dtype": str(df[c].dtype),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("missing_ratio", ascending=False).reset_index(drop=True)


def df_to_markdown_safe(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


def make_markdown_report(
    out_path: Path,
    root: Path,
    eeg_features_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    record_report: pd.DataFrame,
    eeg_missing_df: pd.DataFrame,
    output_paths: Dict[str, Path],
    win_s: float,
) -> None:
    lines = []

    lines.append("# EEG feature dataset report")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- window_s: **{win_s}**")
    lines.append(f"- EEG channels: `{', '.join(EEG_SIGNAL_CHANNELS)}`")
    lines.append("")

    lines.append("## Record processing status")
    lines.append("")
    if record_report.empty:
        lines.append("_No record report._")
    else:
        lines.append(df_to_markdown_safe(record_report["status"].value_counts(dropna=False).to_frame("count")))
    lines.append("")

    lines.append("## EEG feature table summary")
    lines.append("")
    lines.append(f"- Rows/windows: **{len(eeg_features_df)}**")
    lines.append(f"- Columns: **{eeg_features_df.shape[1]}**")
    if not eeg_features_df.empty:
        lines.append(f"- Records: **{eeg_features_df['record_id'].nunique(dropna=True)}**")
        lines.append(f"- Subjects: **{eeg_features_df['subject_id'].nunique(dropna=True)}**")
        lines.append(f"- Sources: `{eeg_features_df['source'].value_counts(dropna=False).to_dict()}`")
    lines.append("")

    lines.append("## Merged PM/POW + EEG dataset summary")
    lines.append("")
    lines.append(f"- Rows/windows: **{len(merged_df)}**")
    lines.append(f"- Columns: **{merged_df.shape[1]}**")
    if "record_id" in merged_df.columns:
        lines.append(f"- Records: **{merged_df['record_id'].nunique(dropna=True)}**")
    if "subject_id" in merged_df.columns:
        lines.append(f"- Subjects: **{merged_df['subject_id'].nunique(dropna=True)}**")
    if "source" in merged_df.columns:
        lines.append(f"- Sources: `{merged_df['source'].value_counts(dropna=False).to_dict()}`")
    lines.append("")

    eeg_feature_cols = [c for c in merged_df.columns if c.startswith("EEG.")]
    lines.append("## EEG feature columns")
    lines.append("")
    lines.append(f"- EEG feature columns in merged dataset: **{len(eeg_feature_cols)}**")
    lines.append("")

    lines.append("## EEG missingness preview")
    lines.append("")
    if eeg_missing_df.empty:
        lines.append("_No EEG missingness table._")
    else:
        lines.append(df_to_markdown_safe(eeg_missing_df.head(40), index=False))
    lines.append("")

    lines.append("## Record report preview")
    lines.append("")
    if record_report.empty:
        lines.append("_No record report._")
    else:
        show_cols = [
            "record_id",
            "source",
            "subject_id",
            "day",
            "status",
            "n_input_rows",
            "n_output_windows",
            "duration_s",
            "missing_eeg_channels",
        ]
        show_cols = [c for c in show_cols if c in record_report.columns]
        lines.append(df_to_markdown_safe(record_report[show_cols].head(40), index=False))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. This dataset extends the previous PM/POW windowed dataset with raw EEG-derived statistical features.")
    lines.append("2. The EEG features are computed on the same 10-second windows as PM/POW features.")
    lines.append("3. The current EEG features are time-domain statistics. Spectral features from raw EEG should be added in a later stage.")
    lines.append("4. The first comparison should check whether adding EEG features improves GroupKFold and cross-source no-overlap metrics.")
    lines.append("5. PM-derived columns must still be excluded from model features to avoid target leakage.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    root: Path,
    output_name: str,
    eeg_features_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    record_report: pd.DataFrame,
    win_s: float,
) -> None:
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eeg_features_parquet = interim_dir / "windowed_eeg_features_w10.parquet"
    eeg_features_csv = interim_dir / "windowed_eeg_features_w10.csv"
    record_report_csv = interim_dir / "windowed_eeg_features_w10_record_report.csv"

    merged_parquet = processed_dir / f"{output_name}.parquet"
    merged_csv = processed_dir / f"{output_name}.csv"

    report_md = reports_dir / "windowed_eeg_features_w10_report.md"
    eeg_missing_csv = interim_dir / "windowed_eeg_features_w10_missingness.csv"

    eeg_features_df.to_parquet(eeg_features_parquet, index=False)
    eeg_features_df.to_csv(eeg_features_csv, index=False, encoding="utf-8-sig")

    merged_df.to_parquet(merged_parquet, index=False)
    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    record_report.to_csv(record_report_csv, index=False, encoding="utf-8-sig")

    eeg_missing_df = missingness_summary(merged_df, prefix="EEG.")
    eeg_missing_df.to_csv(eeg_missing_csv, index=False, encoding="utf-8-sig")

    output_paths = {
        "eeg_features_parquet": eeg_features_parquet,
        "eeg_features_csv": eeg_features_csv,
        "record_report_csv": record_report_csv,
        "merged_parquet": merged_parquet,
        "merged_csv": merged_csv,
        "eeg_missing_csv": eeg_missing_csv,
        "report_md": report_md,
    }

    make_markdown_report(
        out_path=report_md,
        root=root,
        eeg_features_df=eeg_features_df,
        merged_df=merged_df,
        record_report=record_report,
        eeg_missing_df=eeg_missing_df,
        output_paths=output_paths,
        win_s=win_s,
    )

    print("[OK] Saved:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


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
        help="Каталог Emotiv-записей.",
    )
    parser.add_argument(
        "--pm-dataset",
        type=str,
        default=r"data\processed\windowed_pm_dataset_w10.parquet",
        help="Предыдущий PM/POW оконный датасет.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="windowed_eeg_pm_dataset_w10",
        help="Имя итогового датасета без расширения.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "gpn_data", "Old_EEG"],
        default="all",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=DEFAULT_WINDOW_S,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Ограничение числа записей для теста.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path

    pm_dataset_path = Path(args.pm_dataset)
    if not pm_dataset_path.is_absolute():
        pm_dataset_path = root / pm_dataset_path

    print("=" * 80)
    print("Build raw EEG time-domain features")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Catalog: {catalog_path}")
    print(f"PM/POW dataset: {pm_dataset_path}")
    print(f"Source: {args.source}")
    print(f"Window size: {args.window_s}")
    print(f"Chunk size: {args.chunk_size}")

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    if not pm_dataset_path.exists():
        raise FileNotFoundError(f"PM/POW dataset not found: {pm_dataset_path}")

    catalog = pd.read_csv(catalog_path)
    catalog = filter_catalog(
        catalog=catalog,
        source=args.source,
        max_records=args.max_records,
    )

    print(f"Records to process: {len(catalog)}")

    all_parts = []
    reports = []

    for i, (_, row) in enumerate(catalog.iterrows(), start=1):
        print("-" * 80)
        print(f"[{i}/{len(catalog)}] {row['source']} | {row['subject_id']} | {row['day']} | {row['main_rel_path']}")

        part_df, meta = read_and_build_eeg_features_for_record(
            row=row,
            root=root,
            eeg_cols=EEG_SIGNAL_CHANNELS,
            win_s=args.window_s,
            chunk_size=args.chunk_size,
        )

        reports.append(meta)

        print(
            f"status={meta['status']} | input_rows={meta['n_input_rows']} | "
            f"windows={meta['n_output_windows']} | duration_s={meta['duration_s']}"
        )

        if not part_df.empty:
            all_parts.append(part_df)

    record_report = pd.DataFrame(reports)

    if not all_parts:
        raise RuntimeError("No EEG feature windows were built. Check record report.")

    eeg_features_df = pd.concat(all_parts, ignore_index=True)
    eeg_features_df = eeg_features_df.replace([np.inf, -np.inf], np.nan)
    eeg_features_df = eeg_features_df.copy()

    print("\nLoading PM/POW dataset...")
    pm_df = pd.read_parquet(pm_dataset_path)

    if args.source != "all":
        pm_df = pm_df[pm_df["source"] == args.source].copy()

    print(f"PM/POW rows: {len(pm_df)}")
    print(f"EEG feature rows: {len(eeg_features_df)}")

    print("Merging PM/POW and EEG features...")
    merged_df = merge_eeg_with_pm_dataset(
        pm_df=pm_df,
        eeg_df=eeg_features_df,
        win_s=args.window_s,
    )

    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns: {merged_df.shape[1]}")

    save_outputs(
        root=root,
        output_name=args.output_name,
        eeg_features_df=eeg_features_df,
        merged_df=merged_df,
        record_report=record_report,
        win_s=args.window_s,
    )

    print("\nFinal summary:")
    print(f"EEG feature rows: {len(eeg_features_df)}")
    print(f"EEG feature columns: {eeg_features_df.shape[1]}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns: {merged_df.shape[1]}")

    eeg_cols = [c for c in merged_df.columns if c.startswith("EEG.")]
    print(f"EEG columns in merged dataset: {len(eeg_cols)}")

    if "source" in merged_df.columns:
        print("\nSources:")
        print(merged_df["source"].value_counts(dropna=False).to_string())

    if "label_q5" in merged_df.columns:
        print("\nLabels:")
        print(merged_df["label_q5"].value_counts(dropna=False).sort_index().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()