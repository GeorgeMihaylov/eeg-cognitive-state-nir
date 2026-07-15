# -*- coding: utf-8 -*-
"""
01_inspect_emotiv_files.py

Детальная диагностика EEG/PM/marker/annotation файлов после первичной инвентаризации.

Задачи:
1. Найти Emotiv-файлы .md.mc.pm.fe.bp.csv и .csv.bz2.
2. Прочитать их без полной распаковки на диск.
3. Найти метаданные в начале файла.
4. Найти реальную строку заголовка таблицы.
5. Показать реальные колонки, числовые поля, временные поля.
6. Отдельно проверить intervalMarker-файлы.
7. Отдельно проверить annotation CSV из Old_EEG.
8. Сохранить отчеты в reports/ и data/interim/.

Запуск:
    python src/01_inspect_emotiv_files.py

Быстрый тест:
    python src/01_inspect_emotiv_files.py --max-files-per-kind 20
"""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SUBJECT_RE = re.compile(r"([0-9a-fA-F]{8})")

TIME_HINTS = {
    "time",
    "timestamp",
    "timestamps",
    "date",
    "datetime",
    "sample_time",
    "utc",
    "sec",
    "seconds",
    "ms",
}

EEG_CHANNEL_HINTS = {
    "af3", "f7", "f3", "fc5", "t7", "p7", "o1",
    "o2", "p8", "t8", "fc6", "f4", "f8", "af4",
}

PM_HINTS = {
    "pm",
    "focus",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
}

BANDPOWER_HINTS = {
    "theta",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "pow",
    "bp",
}

MOTION_HINTS = {
    "gyro",
    "acc",
    "motion",
    "mc",
    "quaternion",
    "mag",
}

FACIAL_HINTS = {
    "fe",
    "facial",
    "expression",
    "blink",
    "wink",
    "smile",
    "clench",
}


def open_text(path: Path):
    """
    Открывает обычный текстовый файл или .bz2 как текст.
    """
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, mode="rt", encoding="utf-8", errors="replace")


def infer_subject_id(path: Path) -> Optional[str]:
    m = SUBJECT_RE.search(str(path))
    return m.group(1).lower() if m else None


def infer_day(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if p in {"day1", "day2", "day3"}:
            return p
    name = path.name.lower()
    if "1day" in name or "_1day_" in name:
        return "day1"
    if "2day" in name or "_2day_" in name:
        return "day2"
    return None


def infer_part(path: Path) -> Optional[str]:
    name = path.name.lower()
    for token in ["part1", "part2", "part3", "part4", "1part", "2part", "3part", "4part"]:
        if token in name:
            return token
    return None


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except Exception:
        return np.nan


def read_head_lines(path: Path, n_lines: int = 80) -> List[str]:
    lines = []
    with open_text(path) as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n\r"))
    return lines


def parse_metadata(lines: List[str]) -> Dict[str, str]:
    """
    Извлекает метаданные из верхней части Emotiv CSV.
    Обычно там строки вида:
        title:...
        start timestamp:...
        sampling rate:eeg_256;mot_64;mc_8;pm_0.1;fe_32;pow_8
    """
    meta = {}
    for line in lines[:30]:
        s = line.strip().strip(",").strip()
        if not s:
            continue

        if ":" in s and s.count(",") <= 2:
            k, v = s.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k:
                meta[k] = v

    return meta


def parse_sampling_rates(meta: Dict[str, str]) -> Dict[str, float]:
    raw = meta.get("sampling rate") or meta.get("sampling_rate") or ""
    out = {}

    for item in raw.split(";"):
        item = item.strip()
        if "_" not in item:
            continue
        k, v = item.rsplit("_", 1)
        try:
            out[k.strip().lower()] = float(v)
        except Exception:
            pass

    return out


def sniff_delimiter(line: str) -> str:
    candidates = [",", ";", "\t", "|"]
    counts = {sep: line.count(sep) for sep in candidates}
    best = max(counts.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else ","


def looks_like_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    low = s.lower()

    if low.startswith(("title:", "start timestamp:", "stop timestamp:", "headset type:", "headset serial:", "sampling rate:")):
        return False

    sep = sniff_delimiter(s)
    parts = [p.strip().lower() for p in s.split(sep)]

    if len(parts) < 3:
        return False

    text = " ".join(parts)

    header_score = 0

    if any(h in text for h in TIME_HINTS):
        header_score += 2
    if any(h in text for h in EEG_CHANNEL_HINTS):
        header_score += 3
    if any(h in text for h in PM_HINTS):
        header_score += 2
    if any(h in text for h in BANDPOWER_HINTS):
        header_score += 2
    if any(h in text for h in MOTION_HINTS):
        header_score += 1
    if any(h in text for h in FACIAL_HINTS):
        header_score += 1

    # Если это строка данных, там обычно много чисел.
    numeric_like = 0
    for p in parts[:20]:
        try:
            float(p)
            numeric_like += 1
        except Exception:
            pass

    if numeric_like >= max(3, len(parts[:20]) // 2):
        return False

    return header_score >= 2


def find_header_row(path: Path, max_scan_lines: int = 200) -> Tuple[Optional[int], Optional[str], List[str], Dict[str, str]]:
    lines = read_head_lines(path, n_lines=max_scan_lines)
    meta = parse_metadata(lines)

    for i, line in enumerate(lines):
        if looks_like_header(line):
            sep = sniff_delimiter(line)
            return i, sep, lines, meta

    return None, None, lines, meta


def read_csv_preview_with_header(
    path: Path,
    header_row: int,
    sep: str,
    nrows: int = 5000,
) -> pd.DataFrame:
    """
    Читает CSV или CSV.BZ2 с найденной строкой заголовка.
    """
    compression = "bz2" if path.suffix.lower() == ".bz2" else None

    return pd.read_csv(
        path,
        compression=compression,
        sep=sep,
        header=header_row,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
    )


def classify_columns(columns: List[str]) -> Dict[str, List[str]]:
    out = {
        "time": [],
        "eeg": [],
        "pm": [],
        "bandpower": [],
        "motion": [],
        "facial": [],
        "other": [],
    }

    for col in columns:
        c = str(col)
        low = c.lower()

        if any(h in low for h in TIME_HINTS):
            out["time"].append(c)
        elif any(h in low for h in EEG_CHANNEL_HINTS):
            out["eeg"].append(c)
        elif any(h in low for h in PM_HINTS):
            out["pm"].append(c)
        elif any(h in low for h in BANDPOWER_HINTS):
            out["bandpower"].append(c)
        elif any(h in low for h in MOTION_HINTS):
            out["motion"].append(c)
        elif any(h in low for h in FACIAL_HINTS):
            out["facial"].append(c)
        else:
            out["other"].append(c)

    return out


def numeric_convertibility(df: pd.DataFrame) -> Dict[str, float]:
    """
    Доля значений, которые можно привести к числу, по каждой колонке.
    """
    ratios = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        ratios[str(col)] = float(s.notna().mean())
    return ratios


def inspect_csv_like(path: Path, nrows: int = 5000) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_mb": round(file_size_mb(path), 6),
        "subject_id": infer_subject_id(path),
        "day": infer_day(path),
        "part": infer_part(path),
        "status": "unknown",
        "error": None,
    }

    try:
        header_row, sep, head_lines, meta = find_header_row(path)

        row["metadata"] = meta
        row["sampling_rates"] = parse_sampling_rates(meta)
        row["header_row"] = header_row
        row["separator"] = sep
        row["head_lines_preview"] = head_lines[:20]

        if header_row is None or sep is None:
            row["status"] = "header_not_found"
            return row

        df = read_csv_preview_with_header(path, header_row=header_row, sep=sep, nrows=nrows)

        columns = [str(c) for c in df.columns]
        groups = classify_columns(columns)
        ratios = numeric_convertibility(df)

        numeric_cols = [c for c, r in ratios.items() if r >= 0.80]

        row["status"] = "ok"
        row["n_rows_preview"] = int(len(df))
        row["n_cols"] = int(len(df.columns))
        row["columns"] = columns
        row["columns_preview"] = columns[:80]
        row["column_groups"] = groups
        row["numeric_columns_count"] = len(numeric_cols)
        row["numeric_columns_preview"] = numeric_cols[:80]
        row["time_columns"] = groups["time"]
        row["eeg_columns_count"] = len(groups["eeg"])
        row["pm_columns_count"] = len(groups["pm"])
        row["bandpower_columns_count"] = len(groups["bandpower"])
        row["motion_columns_count"] = len(groups["motion"])
        row["facial_columns_count"] = len(groups["facial"])

        # Несколько строк примера сохраняем отдельно компактно.
        row["sample_rows"] = df.head(3).astype(str).to_dict(orient="records")

        return row

    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
        row["traceback"] = traceback.format_exc(limit=2)
        return row


def inspect_annotation_csv(path: Path, nrows: int = 5000) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_mb": round(file_size_mb(path), 6),
        "subject_id": infer_subject_id(path),
        "status": "unknown",
        "error": None,
    }

    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines="skip")

        columns = [str(c) for c in df.columns]
        ratios = numeric_convertibility(df)
        numeric_cols = [c for c, r in ratios.items() if r >= 0.80]

        row["status"] = "ok"
        row["n_rows_preview"] = int(len(df))
        row["n_cols"] = int(len(df.columns))
        row["columns"] = columns
        row["columns_preview"] = columns[:80]
        row["numeric_columns_count"] = len(numeric_cols)
        row["numeric_columns_preview"] = numeric_cols[:80]
        row["time_like_columns"] = [c for c in columns if any(h in c.lower() for h in TIME_HINTS)]
        row["sample_rows"] = df.head(5).astype(str).to_dict(orient="records")

        return row

    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
        row["traceback"] = traceback.format_exc(limit=2)
        return row


def json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)


def prepare_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: json.dumps(json_safe(x), ensure_ascii=False)
                if isinstance(x, (list, tuple, dict))
                else x
            )
    return df


def compact_list(x: Any, max_items: int = 12) -> str:
    if isinstance(x, str):
        try:
            obj = json.loads(x)
        except Exception:
            return x
    else:
        obj = x

    if isinstance(obj, list):
        return ", ".join(map(str, obj[:max_items])) + (" ..." if len(obj) > max_items else "")
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)[:500]
    return str(obj)


def make_markdown_report(
    emotiv_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    out_path: Path,
    project_root: Path,
) -> None:
    lines = []
    lines.append("# Детальная диагностика Emotiv/annotation файлов")
    lines.append("")
    lines.append(f"Корень проекта: `{project_root}`")
    lines.append("")

    def add_status_block(title: str, df: pd.DataFrame):
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("_Файлы не найдены._")
            lines.append("")
            return

        lines.append("### Статусы")
        lines.append("")
        lines.append(df["status"].value_counts(dropna=False).to_frame("count").to_markdown())
        lines.append("")

        if "sampling_rates" in df.columns:
            lines.append("### Sampling rates, найденные в метаданных")
            lines.append("")
            vals = df["sampling_rates"].dropna().head(20).tolist()
            if vals:
                for v in vals[:10]:
                    lines.append(f"- `{v}`")
            else:
                lines.append("_Не найдены._")
            lines.append("")

        show_cols = [
            "filename",
            "size_mb",
            "subject_id",
            "day",
            "part",
            "status",
            "header_row",
            "n_rows_preview",
            "n_cols",
            "numeric_columns_count",
            "eeg_columns_count",
            "pm_columns_count",
            "bandpower_columns_count",
            "motion_columns_count",
            "facial_columns_count",
            "time_columns",
            "columns_preview",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        show = df[show_cols].head(30).copy()

        for c in ["time_columns", "columns_preview"]:
            if c in show.columns:
                show[c] = show[c].apply(compact_list)

        lines.append("### Первые файлы")
        lines.append("")
        lines.append(show.to_markdown(index=False))
        lines.append("")

    add_status_block("Emotiv main files: *.md.mc.pm.fe.bp.csv / *.csv.bz2", emotiv_df)
    add_status_block("Interval marker files", marker_df)
    add_status_block("Annotation CSV files", ann_df)

    lines.append("## Интерпретация")
    lines.append("")
    lines.append("1. Если `header_row` найден и `n_cols` большое, файл читается корректно.")
    lines.append("2. Если `eeg_columns_count > 0`, можно строить EEG-сегменты.")
    lines.append("3. Если `pm_columns_count > 0`, можно строить разметку по PM-метрикам.")
    lines.append("4. Если `bandpower_columns_count > 0`, можно отдельно использовать готовые спектральные признаки Emotiv.")
    lines.append("5. Если marker-файлы читаются, их можно использовать для синхронизации событий.")
    lines.append("6. Annotation CSV из `Old_EEG` можно использовать как поведенческую разметку, если удастся надежно сопоставить времена.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        "--max-files-per-kind",
        type=int,
        default=50,
        help="Сколько файлов каждого типа инспектировать.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5000,
        help="Сколько строк читать из каждого файла.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_dir = root / "data" / "raw"
    interim_dir = root / "data" / "interim"
    reports_dir = root / "reports"

    print("=" * 80)
    print("Detailed Emotiv data inspection")
    print("=" * 80)
    print(f"Root: {root}")

    emotiv_files = []
    emotiv_files.extend(raw_dir.glob("**/*.md.mc.pm.fe.bp.csv"))
    emotiv_files.extend(raw_dir.glob("**/*.md.mc.pm.fe.bp.csv.bz2"))
    emotiv_files = sorted(set(emotiv_files), key=lambda p: str(p).lower())

    marker_files = []
    marker_files.extend(raw_dir.glob("**/*intervalMarker*.csv"))
    marker_files.extend(raw_dir.glob("**/*intervalMarker*.csv.bz2"))
    marker_files = sorted(set(marker_files), key=lambda p: str(p).lower())

    annotation_files = sorted(
        (raw_dir / "Old_EEG").glob("**/annotations/**/*.csv"),
        key=lambda p: str(p).lower(),
    )

    emotiv_files = emotiv_files[: args.max_files_per_kind]
    marker_files = marker_files[: args.max_files_per_kind]
    annotation_files = annotation_files[: args.max_files_per_kind]

    print(f"Emotiv files selected: {len(emotiv_files)}")
    print(f"Marker files selected: {len(marker_files)}")
    print(f"Annotation files selected: {len(annotation_files)}")

    emotiv_rows = []
    for i, p in enumerate(emotiv_files, 1):
        print(f"[EMOTIV {i}/{len(emotiv_files)}] {p.name}")
        emotiv_rows.append(inspect_csv_like(p, nrows=args.preview_rows))

    marker_rows = []
    for i, p in enumerate(marker_files, 1):
        print(f"[MARKER {i}/{len(marker_files)}] {p.name}")
        marker_rows.append(inspect_csv_like(p, nrows=args.preview_rows))

    ann_rows = []
    for i, p in enumerate(annotation_files, 1):
        print(f"[ANN {i}/{len(annotation_files)}] {p.name}")
        ann_rows.append(inspect_annotation_csv(p, nrows=args.preview_rows))

    emotiv_df = prepare_df(emotiv_rows)
    marker_df = prepare_df(marker_rows)
    ann_df = prepare_df(ann_rows)

    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    emotiv_csv = interim_dir / "emotiv_files_inspection.csv"
    marker_csv = interim_dir / "marker_files_inspection.csv"
    ann_csv = interim_dir / "annotation_files_inspection.csv"

    emotiv_df.to_csv(emotiv_csv, index=False, encoding="utf-8-sig")
    marker_df.to_csv(marker_csv, index=False, encoding="utf-8-sig")
    ann_df.to_csv(ann_csv, index=False, encoding="utf-8-sig")

    for df, name in [
        (emotiv_df, "emotiv_files_inspection.parquet"),
        (marker_df, "marker_files_inspection.parquet"),
        (ann_df, "annotation_files_inspection.parquet"),
    ]:
        try:
            df.to_parquet(interim_dir / name, index=False)
        except Exception as e:
            print(f"[WARN] Could not save {name}: {e}")

    md_path = reports_dir / "emotiv_files_inspection.md"
    make_markdown_report(
        emotiv_df=emotiv_df,
        marker_df=marker_df,
        ann_df=ann_df,
        out_path=md_path,
        project_root=root,
    )

    print("\nSaved:")
    print(f"  {emotiv_csv}")
    print(f"  {marker_csv}")
    print(f"  {ann_csv}")
    print(f"  {md_path}")

    print("\nEmotiv status:")
    if not emotiv_df.empty:
        print(emotiv_df["status"].value_counts(dropna=False).to_string())

    print("\nMarker status:")
    if not marker_df.empty:
        print(marker_df["status"].value_counts(dropna=False).to_string())

    print("\nAnnotation status:")
    if not ann_df.empty:
        print(ann_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()