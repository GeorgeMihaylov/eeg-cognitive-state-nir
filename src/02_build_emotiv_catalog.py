# -*- coding: utf-8 -*-
"""
02_build_emotiv_catalog.py

Строит каталог Emotiv-записей для нового EEG NIR проекта.

Вход:
    data/raw/gpn_data
    data/raw/Old_EEG

Выход:
    data/interim/emotiv_record_catalog.csv
    data/interim/emotiv_record_catalog.parquet
    reports/emotiv_record_catalog.md

Каталог нужен, чтобы перед построением сегментов понять:
- сколько субъектов;
- сколько дней;
- какие main-файлы есть;
- есть ли рядом json;
- есть ли рядом intervalMarker;
- какие колонки реально присутствуют;
- какие потоки есть в файле: EEG, PM, bandpower, motion, facial.

Запуск:
    python src/02_build_emotiv_catalog.py

Только новые данные:
    python src/02_build_emotiv_catalog.py --source gpn_data

Старые и новые:
    python src/02_build_emotiv_catalog.py --source all
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SUBJECT_RE = re.compile(r"([0-9a-fA-F]{8})")
DATETIME_RE = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}T\d{2}\.\d{2}\.\d{2}(?:[+-]\d{2}\.\d{2})?)"
)

EEG_CHANNELS = {
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1",
    "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4",
}

TIME_HINTS = {
    "timestamp",
    "originaltimestamp",
    "time",
    "date",
    "datetime",
}

PM_HINTS = {
    "PM.",
    "Focus",
    "Engagement",
    "Excitement",
    "Stress",
    "Relaxation",
    "Interest",
}

BANDPOWER_HINTS = {
    "POW.",
    "BP.",
    "alpha",
    "theta",
    "beta",
    "gamma",
    "delta",
}

MOTION_HINTS = {
    "MOT.",
    "MC.",
    "Gyro",
    "Accel",
    "Mag",
    "Quaternion",
}

FACIAL_HINTS = {
    "FE.",
    "Facial",
    "Expression",
    "Blink",
    "Wink",
    "Smile",
    "Clench",
}

DEFAULT_SAMPLING_RATES = {
    "eeg": 256.0,
    "mot": 64.0,
    "mc": 8.0,
    "pm": 0.1,
    "fe": 32.0,
    "pow": 8.0,
}


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except Exception:
        return np.nan


def infer_subject_id(path: Path) -> Optional[str]:
    m = SUBJECT_RE.search(str(path))
    return m.group(1).lower() if m else None


def infer_day(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if p in {"day1", "day2", "day3"}:
            return p

    name = path.name.lower()
    if "1day" in name or "_day1_" in name or "day1" in name:
        return "day1"
    if "2day" in name or "_day2_" in name or "day2" in name:
        return "day2"
    if "3day" in name or "_day3_" in name or "day3" in name:
        return "day3"

    return None


def infer_part(path: Path) -> Optional[str]:
    name = path.name.lower()

    patterns = [
        "part1", "part2", "part3", "part4",
        "1part", "2part", "3part", "4part",
    ]

    for p in patterns:
        if p in name:
            return p

    return None


def infer_datetime_from_name(path: Path) -> Optional[str]:
    m = DATETIME_RE.search(path.name)
    if not m:
        return None
    return m.group(1)


def normalize_record_key(path: Path) -> str:
    """
    Убирает служебный суффикс, чтобы сопоставить:
    - main .md.mc.pm.fe.bp.csv.bz2
    - intervalMarker.csv.bz2
    - json
    """
    name = path.name

    suffixes = [
        ".md.mc.pm.fe.bp.csv.bz2",
        ".md.mc.pm.fe.bp.csv",
        "_intervalMarker.csv.bz2",
        "_intervalMarker.csv",
        ".json",
    ]

    for s in suffixes:
        if name.endswith(s):
            return name[: -len(s)]

    return path.stem


def open_text(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, mode="rt", encoding="utf-8", errors="replace")


def read_header_line(path: Path, max_lines: int = 10) -> Tuple[Optional[int], Optional[str]]:
    """
    Для новых .csv.bz2 из отчета header_row обычно равен 1.
    Но делаем поиск по строкам, чтобы не завязываться жестко.
    """
    with open_text(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            s = line.strip()
            if not s:
                continue

            if "Timestamp" in s and "EEG." in s:
                return i, s

            if "OriginalTimestamp" in s and "EEG." in s:
                return i, s

    return None, None


def sniff_separator(header_line: str) -> str:
    candidates = [",", ";", "\t", "|"]
    counts = {sep: header_line.count(sep) for sep in candidates}
    best_sep, best_count = max(counts.items(), key=lambda x: x[1])
    return best_sep if best_count > 0 else ","


def read_columns(path: Path) -> Tuple[Optional[int], Optional[str], List[str]]:
    header_row, header_line = read_header_line(path)

    if header_row is None or header_line is None:
        return None, None, []

    sep = sniff_separator(header_line)
    columns = [c.strip() for c in header_line.split(sep)]
    return header_row, sep, columns


def classify_columns(columns: List[str]) -> Dict[str, List[str]]:
    groups = {
        "time": [],
        "eeg": [],
        "pm": [],
        "bandpower": [],
        "motion": [],
        "facial": [],
        "other": [],
    }

    for c in columns:
        cl = c.lower()

        if c in EEG_CHANNELS:
            groups["eeg"].append(c)
        elif c.startswith("EEG."):
            groups["eeg"].append(c)
        elif any(h.lower() in cl for h in TIME_HINTS):
            groups["time"].append(c)
        elif any(h.lower() in cl for h in PM_HINTS):
            groups["pm"].append(c)
        elif any(h.lower() in cl for h in BANDPOWER_HINTS):
            groups["bandpower"].append(c)
        elif any(h.lower() in cl for h in MOTION_HINTS):
            groups["motion"].append(c)
        elif any(h.lower() in cl for h in FACIAL_HINTS):
            groups["facial"].append(c)
        else:
            groups["other"].append(c)

    return groups


def try_read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

        return {"json_type": type(obj).__name__}

    except Exception as e:
        return {"json_read_error": str(e)}


def extract_sampling_rates_from_json(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Поддерживает несколько возможных вариантов.
    Если в json нет частот, вернет пустой dict.
    """
    out = {}

    def scan_dict(d: Dict[str, Any]):
        for k, v in d.items():
            kl = str(k).lower()

            if isinstance(v, dict):
                scan_dict(v)
                continue

            if "sampling" in kl or "sample" in kl or "rate" in kl or "sfreq" in kl:
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass

    if isinstance(obj, dict):
        scan_dict(obj)

    return out


def find_companion_file(main_path: Path, candidates: Dict[str, Path], suffix_kind: str) -> Optional[Path]:
    key = normalize_record_key(main_path)

    if suffix_kind == "json":
        return candidates.get(key + ".json") or candidates.get(key)

    if suffix_kind == "marker":
        return candidates.get(key + "_intervalMarker") or candidates.get(key)

    return None


def rel(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def build_catalog_for_root(data_root: Path, project_root: Path, source_name: str) -> List[Dict[str, Any]]:
    main_files = []
    main_files.extend(data_root.glob("**/*.md.mc.pm.fe.bp.csv"))
    main_files.extend(data_root.glob("**/*.md.mc.pm.fe.bp.csv.bz2"))
    main_files = sorted(set(main_files), key=lambda p: str(p).lower())

    json_files = sorted(data_root.glob("**/*.json"), key=lambda p: str(p).lower())
    marker_files = []
    marker_files.extend(data_root.glob("**/*intervalMarker*.csv"))
    marker_files.extend(data_root.glob("**/*intervalMarker*.csv.bz2"))
    marker_files = sorted(set(marker_files), key=lambda p: str(p).lower())

    json_by_key = {}
    for p in json_files:
        json_by_key[normalize_record_key(p)] = p
        json_by_key[normalize_record_key(p) + ".json"] = p

    marker_by_key = {}
    for p in marker_files:
        k = normalize_record_key(p)
        marker_by_key[k] = p
        marker_by_key[k + "_intervalMarker"] = p

    rows = []

    for i, main_path in enumerate(main_files, start=1):
        print(f"[{source_name}] {i}/{len(main_files)} {main_path.name}")

        row: Dict[str, Any] = {
            "source": source_name,
            "status": "unknown",
            "error": None,
            "main_path": str(main_path),
            "main_rel_path": rel(main_path, project_root),
            "filename": main_path.name,
            "record_key": normalize_record_key(main_path),
            "subject_id": infer_subject_id(main_path),
            "day": infer_day(main_path),
            "part": infer_part(main_path),
            "datetime_from_name": infer_datetime_from_name(main_path),
            "size_mb": round(file_size_mb(main_path), 6),
        }

        try:
            json_path = find_companion_file(main_path, json_by_key, "json")
            marker_path = find_companion_file(main_path, marker_by_key, "marker")

            row["json_path"] = str(json_path) if json_path else None
            row["json_rel_path"] = rel(json_path, project_root)
            row["marker_path"] = str(marker_path) if marker_path else None
            row["marker_rel_path"] = rel(marker_path, project_root)
            row["has_json"] = json_path is not None
            row["has_marker"] = marker_path is not None

            header_row, sep, columns = read_columns(main_path)
            groups = classify_columns(columns)

            row["header_row"] = header_row
            row["separator"] = sep
            row["n_cols"] = len(columns)
            row["columns"] = columns
            row["columns_preview"] = columns[:60]

            row["time_columns"] = groups["time"]
            row["eeg_columns"] = groups["eeg"]
            row["pm_columns"] = groups["pm"]
            row["bandpower_columns"] = groups["bandpower"]
            row["motion_columns"] = groups["motion"]
            row["facial_columns"] = groups["facial"]

            row["time_columns_count"] = len(groups["time"])
            row["eeg_columns_count"] = len(groups["eeg"])
            row["pm_columns_count"] = len(groups["pm"])
            row["bandpower_columns_count"] = len(groups["bandpower"])
            row["motion_columns_count"] = len(groups["motion"])
            row["facial_columns_count"] = len(groups["facial"])

            json_obj = try_read_json(json_path)
            row["json_content"] = json_obj
            row["json_keys"] = list(json_obj.keys()) if isinstance(json_obj, dict) else []

            json_sampling = extract_sampling_rates_from_json(json_obj)
            row["sampling_rates_from_json"] = json_sampling
            row["sampling_rates_assumed"] = DEFAULT_SAMPLING_RATES

            if columns:
                row["status"] = "ok"
            else:
                row["status"] = "no_columns_found"

        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            row["traceback"] = traceback.format_exc(limit=2)

        rows.append(row)

    return rows


def json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    if isinstance(x, Path):
        return str(x)

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


def prepare_for_save(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: json.dumps(json_safe(x), ensure_ascii=False)
                if isinstance(x, (list, tuple, dict))
                else x
            )

    return df


def compact_json_cell(x: Any, max_len: int = 140) -> str:
    if pd.isna(x):
        return ""

    s = str(x)

    if len(s) > max_len:
        return s[:max_len] + " ..."

    return s


def make_markdown_report(df: pd.DataFrame, out_path: Path, project_root: Path) -> None:
    lines = []

    lines.append("# Emotiv record catalog")
    lines.append("")
    lines.append(f"Корень проекта: `{project_root}`")
    lines.append("")

    lines.append("## Общая сводка")
    lines.append("")
    lines.append(f"- Всего записей: **{len(df)}**")
    lines.append(f"- Уникальных субъектов: **{df['subject_id'].nunique(dropna=True)}**")
    lines.append(f"- Суммарный размер main-файлов: **{df['size_mb'].sum():.2f} MB**")
    lines.append("")

    lines.append("## Статусы")
    lines.append("")
    lines.append(df["status"].value_counts(dropna=False).to_frame("count").to_markdown())
    lines.append("")

    lines.append("## Источники")
    lines.append("")
    lines.append(df["source"].value_counts(dropna=False).to_frame("count").to_markdown())
    lines.append("")

    lines.append("## Дни")
    lines.append("")
    lines.append(df["day"].value_counts(dropna=False).to_frame("count").to_markdown())
    lines.append("")

    lines.append("## Наличие companion-файлов")
    lines.append("")
    companion = pd.DataFrame({
        "has_json": df["has_json"].value_counts(dropna=False),
        "has_marker": df["has_marker"].value_counts(dropna=False),
    }).fillna(0).astype(int)
    lines.append(companion.to_markdown())
    lines.append("")

    lines.append("## Количество колонок по потокам")
    lines.append("")
    stream_cols = [
        "n_cols",
        "time_columns_count",
        "eeg_columns_count",
        "pm_columns_count",
        "bandpower_columns_count",
        "motion_columns_count",
        "facial_columns_count",
    ]
    desc = df[stream_cols].describe().T
    lines.append(desc.to_markdown())
    lines.append("")

    lines.append("## Первые 40 записей каталога")
    lines.append("")
    show_cols = [
        "source",
        "subject_id",
        "day",
        "part",
        "datetime_from_name",
        "size_mb",
        "has_json",
        "has_marker",
        "n_cols",
        "eeg_columns_count",
        "pm_columns_count",
        "bandpower_columns_count",
        "motion_columns_count",
        "facial_columns_count",
        "main_rel_path",
    ]
    show = df[show_cols].head(40).copy()
    lines.append(show.to_markdown(index=False))
    lines.append("")

    lines.append("## Примеры колонок")
    lines.append("")
    if len(df) > 0:
        first = df.iloc[0]
        for key in [
            "time_columns",
            "eeg_columns",
            "pm_columns",
            "bandpower_columns",
            "motion_columns",
            "facial_columns",
        ]:
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"`{compact_json_cell(first.get(key), 1000)}`")
            lines.append("")

    lines.append("## Интерпретация")
    lines.append("")
    lines.append("1. Если `n_cols = 183`, структура main-файлов стабильна.")
    lines.append("2. Если `has_json = True`, JSON можно использовать как источник метаданных.")
    lines.append("3. Если `has_marker = True`, marker-файлы можно сохранить в каталоге, даже если они пустые.")
    lines.append("4. Для первого рабочего датасета лучше использовать только `gpn_data`, не смешивая со старым `Old_EEG`.")
    lines.append("5. Следующий этап — построение оконных сегментов по EEG и PM-колонкам.")

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
        "--source",
        type=str,
        choices=["gpn_data", "Old_EEG", "all"],
        default="gpn_data",
        help="Какой источник каталогизировать.",
    )

    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    raw_root = project_root / "data" / "raw"

    sources = []
    if args.source in {"gpn_data", "all"}:
        sources.append(("gpn_data", raw_root / "gpn_data"))
    if args.source in {"Old_EEG", "all"}:
        sources.append(("Old_EEG", raw_root / "Old_EEG"))

    all_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("Build Emotiv record catalog")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Source mode: {args.source}")

    for source_name, source_path in sources:
        print("-" * 80)
        print(f"Source: {source_name}")
        print(f"Path: {source_path}")
        print(f"Exists: {source_path.exists()}")

        if not source_path.exists():
            continue

        rows = build_catalog_for_root(
            data_root=source_path,
            project_root=project_root,
            source_name=source_name,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No records found.")
        return

    df = prepare_for_save(all_rows)

    interim_dir = project_root / "data" / "interim"
    reports_dir = project_root / "reports"
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = interim_dir / "emotiv_record_catalog.csv"
    parquet_path = interim_dir / "emotiv_record_catalog.parquet"
    md_path = reports_dir / "emotiv_record_catalog.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Saved parquet: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Could not save parquet: {e}")

    make_markdown_report(df, md_path, project_root)

    print(f"[OK] Saved csv: {csv_path}")
    print(f"[OK] Saved report: {md_path}")

    print("\nSummary:")
    print(f"  Records: {len(df)}")
    print(f"  Subjects: {df['subject_id'].nunique(dropna=True)}")
    print(f"  Total size MB: {df['size_mb'].sum():.2f}")

    print("\nStatus:")
    print(df["status"].value_counts(dropna=False).to_string())

    print("\nDays:")
    print(df["day"].value_counts(dropna=False).to_string())

    print("\nCompanion files:")
    print("has_json:")
    print(df["has_json"].value_counts(dropna=False).to_string())
    print("has_marker:")
    print(df["has_marker"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()