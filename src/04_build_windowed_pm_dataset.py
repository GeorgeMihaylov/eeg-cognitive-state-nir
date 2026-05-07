# -*- coding: utf-8 -*-
"""
04_build_windowed_pm_dataset.py

Строит первый рабочий оконный датасет по Emotiv-файлам.

Использует:
- Timestamp
- POW.* готовые спектральные признаки
- PM.*Scaled целевые/когнитивные метрики
- PM.*IsActive служебные индикаторы активности PM
- source, subject_id, day, record_id

Не использует пока сырые EEG-каналы, чтобы быстро получить устойчивый первый датасет.

Вход:
    data/interim/emotiv_record_catalog.csv
    data/interim/validated_columns.json

Выход:
    data/processed/windowed_pm_dataset.parquet
    data/processed/windowed_pm_dataset.csv
    reports/windowed_pm_dataset_report.md

Пример запуска:
    python src/04_build_windowed_pm_dataset.py

Быстрый тест:
    python src/04_build_windowed_pm_dataset.py --max-records 5

Только gpn_data:
    python src/04_build_windowed_pm_dataset.py --source gpn_data

Только Old_EEG:
    python src/04_build_windowed_pm_dataset.py --source Old_EEG
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


DEFAULT_TIME_COL = "Timestamp"

DEFAULT_PM_TARGET_COLUMNS = [
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


def find_header_row(path: Path, max_lines: int = 20) -> Tuple[int, str, List[str]]:
    """
    Ищет строку заголовка в Emotiv CSV / CSV.BZ2.

    Для новых gpn_data обычно header_row=1.
    Для старых Old_EEG до таблицы могут идти строки метаданных.
    """
    with open_text(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            s = line.strip()
            if not s:
                continue

            if "Timestamp" in s and "EEG." in s and "PM." in s:
                sep = "," if s.count(",") >= s.count(";") else ";"
                columns = [c.strip() for c in s.split(sep)]
                return i, sep, columns

    raise RuntimeError(f"Не найдена строка заголовка в файле: {path}")


def parse_jsonish_cell(x: Any):
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, dict):
        return x

    s = str(x).strip()
    if not s:
        return []

    try:
        return json.loads(s)
    except Exception:
        return []


def read_validated_columns(path: Path) -> Dict[str, List[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rec = obj["recommended"]

    return {
        "time_columns": rec.get("time_columns", [DEFAULT_TIME_COL]),
        "pm_scaled_columns": rec.get("pm_scaled_columns", DEFAULT_PM_TARGET_COLUMNS),
        "pm_active_columns": rec.get("pm_active_columns", DEFAULT_PM_ACTIVE_COLUMNS),
        "pow_columns": rec.get("pow_columns", []),
        "motion_columns": rec.get("motion_columns", []),
        "facial_columns": rec.get("facial_columns", []),
    }


def normalize_boolish_series(s: pd.Series) -> pd.Series:
    """
    Приводит IsActive-поля к 0/1.
    """
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


def coerce_numeric_df(df: pd.DataFrame, active_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if c in active_cols:
            out[c] = normalize_boolish_series(out[c])
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


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


def read_and_aggregate_record(
    row: pd.Series,
    root: Path,
    selected_columns: List[str],
    pm_scaled_cols: List[str],
    pm_active_cols: List[str],
    pow_cols: List[str],
    win_s: float,
    chunk_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Читает одну запись чанками и агрегирует по окнам.

    Для первого датасета:
    - POW агрегируем mean/std/min/max;
    - PM.Scaled агрегируем mean/std/min/max + last;
    - PM.IsActive агрегируем mean;
    - целевые continuous labels берутся как mean PM.Scaled по окну.
    """
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
    }

    try:
        header_row, sep, actual_columns = find_header_row(main_path)

        missing = [c for c in selected_columns if c not in actual_columns]
        usecols = [c for c in selected_columns if c in actual_columns]

        if DEFAULT_TIME_COL not in usecols:
            raise RuntimeError(f"В файле нет обязательной колонки {DEFAULT_TIME_COL}: {main_path}")

        compression = "bz2" if main_path.suffix.lower() == ".bz2" else None

        grouped_parts = []

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

            chunk = coerce_numeric_df(chunk, active_cols=pm_active_cols)

            chunk = chunk.dropna(subset=[DEFAULT_TIME_COL])
            if chunk.empty:
                continue

            ts = chunk[DEFAULT_TIME_COL].astype(float)

            current_min = float(ts.min())
            current_max = float(ts.max())

            if math.isnan(meta["timestamp_min"]):
                meta["timestamp_min"] = current_min
                meta["timestamp_max"] = current_max
            else:
                meta["timestamp_min"] = min(meta["timestamp_min"], current_min)
                meta["timestamp_max"] = max(meta["timestamp_max"], current_max)

            # На случай абсолютного Unix timestamp: окна строим относительно начала записи.
            t0 = float(ts.min())
            chunk["_t_rel"] = ts - t0

            # Но если файл читается чанками, локальный t0 в каждом чанке ломает глобальные окна.
            # Поэтому ниже пересчитаем относительно глобального начала после чтения невозможно.
            # Для устойчивости используем absolute window id от Timestamp.
            chunk["_window_id_abs"] = np.floor(ts / win_s).astype("int64")
            chunk["_t_center_abs"] = (chunk["_window_id_abs"].astype(float) + 0.5) * win_s

            agg_spec = {}

            for c in pow_cols:
                if c in chunk.columns:
                    agg_spec[c] = ["mean", "std", "min", "max"]

            for c in pm_scaled_cols:
                if c in chunk.columns:
                    agg_spec[c] = ["mean", "std", "min", "max", "last"]

            for c in pm_active_cols:
                if c in chunk.columns:
                    agg_spec[c] = ["mean"]

            if not agg_spec:
                continue

            g = chunk.groupby("_window_id_abs").agg(agg_spec)
            g.columns = [f"{a}__{b}" for a, b in g.columns]
            g = g.reset_index()

            # t_center пока абсолютный. Потом нормализуем на уровне записи.
            g["t_center_abs"] = (g["_window_id_abs"].astype(float) + 0.5) * win_s

            grouped_parts.append(g)

        if not grouped_parts:
            meta["status"] = "empty_after_processing"
            return pd.DataFrame(), meta

        # Чанки могли давать одинаковые window_id на границах.
        tmp = pd.concat(grouped_parts, ignore_index=True)

        feature_cols = [c for c in tmp.columns if c not in {"_window_id_abs", "t_center_abs"}]

        # Вторичная агрегация одинаковых окон между чанками.
        second_agg = {c: "mean" for c in feature_cols}
        second_agg["t_center_abs"] = "mean"

        out = tmp.groupby("_window_id_abs", as_index=False).agg(second_agg)

        t0_abs = float(out["t_center_abs"].min())
        out["t_center"] = out["t_center_abs"] - t0_abs
        out["t_start"] = out["t_center"] - win_s / 2.0
        out["t_end"] = out["t_center"] + win_s / 2.0

        out.insert(0, "record_id", record_id)
        out.insert(1, "source", row.get("source"))
        out.insert(2, "subject_id", row.get("subject_id"))
        out.insert(3, "day", row.get("day"))
        out.insert(4, "part", row.get("part"))
        out.insert(5, "datetime_from_name", row.get("datetime_from_name"))

        # Целевые непрерывные метки: средние PM.Scaled по окну.
        for c in pm_scaled_cols:
            src_col = f"{c}__mean"
            if src_col in out.columns:
                label_col = "target_" + c.replace("PM.", "").replace(".Scaled", "").lower()
                out[label_col] = out[src_col]

        # Основная целевая метка по умолчанию: Focus.
        if "target_focus" in out.columns:
            out["target_main"] = out["target_focus"]
        elif "target_attention" in out.columns:
            out["target_main"] = out["target_attention"]
        elif "target_engagement" in out.columns:
            out["target_main"] = out["target_engagement"]
        else:
            out["target_main"] = np.nan

        # Удаляем технические абсолютные поля, чтобы датасет был независим от Unix-time.
        out = out.drop(columns=["_window_id_abs", "t_center_abs"], errors="ignore")

        meta["status"] = "ok"
        meta["n_output_windows"] = int(len(out))
        meta["missing_selected_columns"] = missing

        return out, meta

    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
        meta["traceback"] = traceback.format_exc(limit=2)
        return pd.DataFrame(), meta


def make_quality_labels(df: pd.DataFrame, target_col: str, n_classes: int) -> pd.Series:
    """
    Делает квантильные классы по target_main.

    Если уникальных значений мало, возвращает NaN.
    """
    s = pd.to_numeric(df[target_col], errors="coerce")

    if s.notna().sum() < n_classes * 5:
        return pd.Series(np.nan, index=df.index)

    if s.nunique(dropna=True) < n_classes:
        return pd.Series(np.nan, index=df.index)

    try:
        return pd.qcut(s, q=n_classes, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=df.index)


def save_outputs(
    dataset: pd.DataFrame,
    record_report: pd.DataFrame,
    root: Path,
    output_name: str,
    n_classes: int,
) -> None:
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    interim_dir = root / "data" / "interim"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = processed_dir / f"{output_name}.parquet"
    csv_path = processed_dir / f"{output_name}.csv"
    record_report_path = interim_dir / f"{output_name}_record_report.csv"
    md_path = reports_dir / f"{output_name}_report.md"

    dataset.to_parquet(parquet_path, index=False)
    dataset.to_csv(csv_path, index=False, encoding="utf-8-sig")
    record_report.to_csv(record_report_path, index=False, encoding="utf-8-sig")

    make_markdown_report(
        dataset=dataset,
        record_report=record_report,
        out_path=md_path,
        parquet_path=parquet_path,
        csv_path=csv_path,
        n_classes=n_classes,
    )

    print(f"[OK] Saved dataset parquet: {parquet_path}")
    print(f"[OK] Saved dataset csv: {csv_path}")
    print(f"[OK] Saved record report: {record_report_path}")
    print(f"[OK] Saved markdown report: {md_path}")


def make_markdown_report(
    dataset: pd.DataFrame,
    record_report: pd.DataFrame,
    out_path: Path,
    parquet_path: Path,
    csv_path: Path,
    n_classes: int,
) -> None:
    lines = []

    lines.append("# Windowed PM/POW dataset report")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- Parquet: `{parquet_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append("")

    lines.append("## Dataset summary")
    lines.append("")
    lines.append(f"- Rows/windows: **{len(dataset)}**")
    lines.append(f"- Columns: **{dataset.shape[1]}**")
    lines.append(f"- Records: **{dataset['record_id'].nunique(dropna=True)}**")
    lines.append(f"- Subjects: **{dataset['subject_id'].nunique(dropna=True)}**")
    lines.append(f"- Sources: `{dataset['source'].value_counts(dropna=False).to_dict()}`")
    lines.append(f"- Days: `{dataset['day'].value_counts(dropna=False).to_dict()}`")
    lines.append("")

    lines.append("## Record processing status")
    lines.append("")
    lines.append(record_report["status"].value_counts(dropna=False).to_frame("count").to_markdown())
    lines.append("")

    lines.append("## Windows by source")
    lines.append("")
    lines.append(dataset["source"].value_counts(dropna=False).to_frame("windows").to_markdown())
    lines.append("")

    lines.append("## Windows by day")
    lines.append("")
    lines.append(dataset["day"].value_counts(dropna=False).to_frame("windows").to_markdown())
    lines.append("")

    lines.append("## Target columns")
    lines.append("")
    target_cols = [c for c in dataset.columns if c.startswith("target_")]
    if target_cols:
        desc = dataset[target_cols].describe().T
        lines.append(desc.to_markdown())
    else:
        lines.append("_Target columns not found._")
    lines.append("")

    label_col = f"label_q{n_classes}"
    if label_col in dataset.columns:
        lines.append(f"## Quantile label `{label_col}`")
        lines.append("")
        lines.append(dataset[label_col].value_counts(dropna=False).sort_index().to_frame("count").to_markdown())
        lines.append("")

    lines.append("## Example columns")
    lines.append("")
    lines.append("```text")
    for c in dataset.columns[:80]:
        lines.append(c)
    lines.append("```")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. This is the first working dataset based on Emotiv POW and PM streams.")
    lines.append("2. Raw EEG channels are not used yet; they should be added in the next stage.")
    lines.append("3. The field `source` must be preserved for domain-specific validation.")
    lines.append("4. The column `target_main` is based on PM.Focus when available.")
    lines.append("5. The quantile label is preliminary and should be treated as weak labeling.")

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
        help="Каталог записей.",
    )
    parser.add_argument(
        "--validated-columns",
        type=str,
        default=r"data\interim\validated_columns.json",
        help="JSON с валидированными колонками.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "gpn_data", "Old_EEG"],
        default="all",
        help="Какие источники обрабатывать.",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=2.0,
        help="Размер окна в секундах.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Размер чанка при чтении CSV.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Ограничение числа записей для быстрого теста.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=5,
        help="Число квантильных классов для weak-label.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="windowed_pm_dataset",
        help="Имя выходного датасета без расширения.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path

    validated_path = Path(args.validated_columns)
    if not validated_path.is_absolute():
        validated_path = root / validated_path

    catalog = pd.read_csv(catalog_path)
    cols = read_validated_columns(validated_path)

    if args.source != "all":
        catalog = catalog[catalog["source"] == args.source].copy()

    catalog = catalog[catalog["status"] == "ok"].copy()

    if args.max_records is not None:
        catalog = catalog.head(args.max_records).copy()

    time_cols = cols["time_columns"]
    pm_scaled_cols = cols["pm_scaled_columns"]
    pm_active_cols = cols["pm_active_columns"]
    pow_cols = cols["pow_columns"]

    selected_columns = []
    selected_columns.extend(time_cols)
    selected_columns.extend(pm_scaled_cols)
    selected_columns.extend(pm_active_cols)
    selected_columns.extend(pow_cols)

    # Убираем дубли, сохраняя порядок.
    selected_columns = list(dict.fromkeys(selected_columns))

    print("=" * 80)
    print("Build windowed PM/POW dataset")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Catalog: {catalog_path}")
    print(f"Validated columns: {validated_path}")
    print(f"Records to process: {len(catalog)}")
    print(f"Source mode: {args.source}")
    print(f"Window size: {args.window_s} s")
    print(f"Selected columns: {len(selected_columns)}")

    all_parts = []
    reports = []

    for i, (_, row) in enumerate(catalog.iterrows(), start=1):
        print("-" * 80)
        print(f"[{i}/{len(catalog)}] {row['source']} | {row['subject_id']} | {row['day']} | {row['main_rel_path']}")

        part_df, meta = read_and_aggregate_record(
            row=row,
            root=root,
            selected_columns=selected_columns,
            pm_scaled_cols=pm_scaled_cols,
            pm_active_cols=pm_active_cols,
            pow_cols=pow_cols,
            win_s=args.window_s,
            chunk_size=args.chunk_size,
        )

        reports.append(meta)

        print(f"status={meta['status']} | input_rows={meta['n_input_rows']} | windows={meta['n_output_windows']}")

        if not part_df.empty:
            all_parts.append(part_df)

    record_report = pd.DataFrame(reports)

    if not all_parts:
        raise RuntimeError("Не удалось построить ни одного окна. Проверь record_report.")

    dataset = pd.concat(all_parts, ignore_index=True)

    dataset = dataset.copy()

    # Финальная слабая дискретная разметка.
    dataset[f"label_q{args.n_classes}"] = make_quality_labels(
        dataset,
        target_col="target_main",
        n_classes=args.n_classes,
    )

    # Чистим бесконечности.
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    save_outputs(
        dataset=dataset,
        record_report=record_report,
        root=root,
        output_name=args.output_name,
        n_classes=args.n_classes,
    )

    print("\nFinal summary:")
    print(f"Rows/windows: {len(dataset)}")
    print(f"Columns: {dataset.shape[1]}")
    print(f"Records: {dataset['record_id'].nunique(dropna=True)}")
    print(f"Subjects: {dataset['subject_id'].nunique(dropna=True)}")
    print("Sources:")
    print(dataset["source"].value_counts(dropna=False).to_string())
    print("Labels:")
    print(dataset[f"label_q{args.n_classes}"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()