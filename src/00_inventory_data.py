# -*- coding: utf-8 -*-
"""
00_inventory_data.py

Первичный осмотр структуры данных для проекта EEG NIR.

Скрипт ничего не изменяет в исходных данных.
Он рекурсивно сканирует data/raw/gpn_data и data/raw/Old_EEG,
собирает информацию о файлах, пробует прочитать табличные форматы
и сохраняет сводный отчет.

Запуск:
    python src/00_inventory_data.py

Или с явными путями:
    python src/00_inventory_data.py ^
      --root "D:\\PycharmProjects\\eeg-cognitive-state-nir" ^
      --raw-dirs "data\\raw\\gpn_data" "data\\raw\\Old_EEG"
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_CANDIDATES = {
    "t",
    "time",
    "timestamp",
    "timestamps",
    "datetime",
    "date",
    "utc_time",
    "sample_time",
    "seconds",
    "sec",
    "ms",
    "millis",
    "milliseconds",
}

EEG_HINTS = {
    "eeg",
    "electrode",
    "electroencephalography",
    "af3",
    "af4",
    "f3",
    "f4",
    "f7",
    "f8",
    "fc5",
    "fc6",
    "t7",
    "t8",
    "p7",
    "p8",
    "o1",
    "o2",
}

PM_HINTS = {
    "pm",
    "performance",
    "metric",
    "metrics",
    "focus",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "attention",
    "cognitive",
}

LABEL_HINTS = {
    "label",
    "labels",
    "target",
    "class",
    "state",
    "y",
    "annotation",
    "annotations",
    "event",
    "events",
    "marker",
    "markers",
}

WEARABLE_HINTS = {
    "hr",
    "heart",
    "pulse",
    "ppg",
    "ecg",
    "eda",
    "gsr",
    "acc",
    "gyro",
    "imu",
    "wearable",
    "watch",
    "sensor",
    "sensors",
    "fitbit",
    "band",
}

SUPPORTED_TABLE_EXTS = {
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
}

SUPPORTED_ARRAY_EXTS = {
    ".npy",
    ".npz",
    ".mat",
}

SUPPORTED_EEG_EXTS = {
    ".edf",
    ".bdf",
    ".set",
    ".fif",
    ".vhdr",
}


def normalize_token(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except Exception:
        return np.nan


def guess_category(path: Path, columns: Optional[List[str]] = None) -> str:
    text_parts = [
        path.name.lower(),
        path.parent.name.lower(),
        path.suffix.lower(),
    ]

    if columns:
        text_parts.extend([str(c).lower() for c in columns])

    text = " ".join(text_parts)
    tokens = {normalize_token(x) for x in text.replace(".", "_").replace("/", "_").split("_")}

    def has_any(hints: set[str]) -> bool:
        if any(h in text for h in hints):
            return True
        if any(h in tokens for h in hints):
            return True
        return False

    if has_any(EEG_HINTS):
        return "possible_eeg"
    if has_any(PM_HINTS):
        return "possible_pm_metrics"
    if has_any(LABEL_HINTS):
        return "possible_labels_or_events"
    if has_any(WEARABLE_HINTS):
        return "possible_wearable_or_sensor"
    if path.suffix.lower() in SUPPORTED_EEG_EXTS:
        return "possible_eeg_raw_format"

    return "unknown"


def find_time_columns(columns: List[str]) -> List[str]:
    out = []
    for c in columns:
        cn = normalize_token(c)
        if cn in TIME_CANDIDATES:
            out.append(c)
        elif any(t in cn for t in TIME_CANDIDATES):
            out.append(c)
    return out


def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(str(c))
        except Exception:
            pass
    return numeric_cols


def detect_delimiter(path: Path) -> Optional[str]:
    candidates = [",", ";", "\t", "|"]
    try:
        with open(path, "rb") as f:
            raw = f.read(8192)
        text = raw.decode("utf-8", errors="ignore")
        first_lines = "\n".join(text.splitlines()[:10])
        counts = {sep: first_lines.count(sep) for sep in candidates}
        best_sep, best_count = max(counts.items(), key=lambda x: x[1])
        if best_count > 0:
            return best_sep
    except Exception:
        pass
    return None


def read_table_preview(path: Path, max_rows: int = 5000) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    ext = path.suffix.lower()
    meta: Dict[str, Any] = {}

    try:
        if ext in {".csv", ".tsv", ".txt"}:
            sep = "\t" if ext == ".tsv" else detect_delimiter(path)
            if sep is None:
                sep = ","

            encodings = ["utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1", "utf-16"]
            last_error = None

            for enc in encodings:
                try:
                    df = pd.read_csv(
                        path,
                        sep=sep,
                        encoding=enc,
                        nrows=max_rows,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                    meta["reader"] = "pandas.read_csv"
                    meta["encoding"] = enc
                    meta["separator"] = sep
                    meta["is_preview"] = True
                    return df, meta
                except Exception as e:
                    last_error = str(e)

            meta["read_error"] = last_error
            return None, meta

        if ext == ".parquet":
            df = pd.read_parquet(path)
            if len(df) > max_rows:
                df = df.head(max_rows)
                meta["is_preview"] = True
            else:
                meta["is_preview"] = False
            meta["reader"] = "pandas.read_parquet"
            return df, meta

        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=max_rows)
            meta["reader"] = "pandas.read_excel"
            meta["is_preview"] = True
            return df, meta

        if ext == ".json":
            try:
                df = pd.read_json(path)
                if len(df) > max_rows:
                    df = df.head(max_rows)
                    meta["is_preview"] = True
                else:
                    meta["is_preview"] = False
                meta["reader"] = "pandas.read_json"
                return df, meta
            except Exception:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
                meta["reader"] = "json.load"
                meta["json_type"] = type(obj).__name__
                if isinstance(obj, dict):
                    meta["json_keys"] = list(obj.keys())[:50]
                elif isinstance(obj, list):
                    meta["json_len"] = len(obj)
                return None, meta

        if ext == ".jsonl":
            df = pd.read_json(path, lines=True, nrows=max_rows)
            meta["reader"] = "pandas.read_json_lines"
            meta["is_preview"] = True
            return df, meta

    except Exception as e:
        meta["read_error"] = str(e)
        meta["traceback_short"] = traceback.format_exc(limit=1)

    return None, meta


def inspect_array_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    meta: Dict[str, Any] = {}

    try:
        if ext == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            meta["array_shape"] = tuple(arr.shape)
            meta["array_dtype"] = str(arr.dtype)
            return meta

        if ext == ".npz":
            z = np.load(path, allow_pickle=False)
            meta["npz_keys"] = list(z.keys())
            meta["npz_shapes"] = {k: tuple(z[k].shape) for k in z.keys()}
            meta["npz_dtypes"] = {k: str(z[k].dtype) for k in z.keys()}
            return meta

        if ext == ".mat":
            try:
                from scipy.io import whosmat

                info = whosmat(path)
                meta["mat_variables"] = [
                    {"name": name, "shape": shape, "class": cls}
                    for name, shape, cls in info
                ]
                return meta
            except Exception as e:
                meta["read_error"] = f"scipy.io.whosmat failed: {e}"
                return meta

    except Exception as e:
        meta["read_error"] = str(e)

    return meta


def inspect_eeg_raw_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    meta: Dict[str, Any] = {}

    try:
        import mne

        if ext == ".edf":
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        elif ext == ".bdf":
            raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
        elif ext == ".fif":
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        elif ext == ".set":
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
        elif ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
        else:
            return meta

        meta["mne_n_channels"] = len(raw.ch_names)
        meta["mne_channels_preview"] = raw.ch_names[:30]
        meta["mne_sfreq"] = float(raw.info.get("sfreq", np.nan))
        meta["mne_duration_s"] = float(raw.n_times / raw.info["sfreq"])
        meta["reader"] = "mne"
        return meta

    except Exception as e:
        meta["read_error"] = str(e)
        return meta


def scan_one_file(path: Path, project_root: Path, source_root: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    row: Dict[str, Any] = {
        "path": str(path),
        "rel_path_project": safe_relpath(path, project_root),
        "rel_path_source": safe_relpath(path, source_root),
        "source_root": source_root.name,
        "filename": path.name,
        "stem": path.stem,
        "extension": ext if ext else "[no_ext]",
        "size_mb": round(file_size_mb(path), 6),
        "category_guess": "unknown",
        "read_status": "not_attempted",
        "reader": None,
        "n_rows_preview": None,
        "n_cols": None,
        "columns_preview": None,
        "time_columns": None,
        "numeric_columns_count": None,
        "numeric_columns_preview": None,
        "error": None,
    }

    try:
        if ext in SUPPORTED_TABLE_EXTS:
            df, meta = read_table_preview(path)
            row.update({k: v for k, v in meta.items() if k not in {"read_error"}})

            if df is not None:
                columns = [str(c) for c in df.columns]
                numeric_cols = find_numeric_columns(df)
                time_cols = find_time_columns(columns)

                row["read_status"] = "ok"
                row["reader"] = meta.get("reader")
                row["n_rows_preview"] = int(len(df))
                row["n_cols"] = int(len(df.columns))
                row["columns_preview"] = columns[:80]
                row["time_columns"] = time_cols
                row["numeric_columns_count"] = int(len(numeric_cols))
                row["numeric_columns_preview"] = numeric_cols[:80]
                row["category_guess"] = guess_category(path, columns)
            else:
                row["read_status"] = "partial" if meta else "failed"
                row["reader"] = meta.get("reader")
                row["category_guess"] = guess_category(path)
                if "read_error" in meta:
                    row["error"] = meta["read_error"]

        elif ext in SUPPORTED_ARRAY_EXTS:
            meta = inspect_array_file(path)
            row.update(meta)
            row["read_status"] = "ok" if "read_error" not in meta else "failed"
            row["category_guess"] = guess_category(path)
            row["error"] = meta.get("read_error")

        elif ext in SUPPORTED_EEG_EXTS:
            meta = inspect_eeg_raw_file(path)
            row.update(meta)
            row["read_status"] = "ok" if "read_error" not in meta else "failed"
            row["category_guess"] = "possible_eeg_raw_format"
            row["error"] = meta.get("read_error")

        else:
            row["category_guess"] = guess_category(path)
            row["read_status"] = "not_supported"

    except Exception as e:
        row["read_status"] = "failed"
        row["error"] = str(e)

    return row


def make_markdown_summary(df: pd.DataFrame, project_root: Path, out_path: Path) -> None:
    total_files = len(df)
    total_size_mb = df["size_mb"].sum()

    ext_counts = df["extension"].value_counts(dropna=False).head(50)
    category_counts = df["category_guess"].value_counts(dropna=False)
    source_counts = df["source_root"].value_counts(dropna=False)
    read_counts = df["read_status"].value_counts(dropna=False)

    largest = df.sort_values("size_mb", ascending=False).head(20)
    readable = df[df["read_status"].isin(["ok", "partial"])].copy()

    possible_eeg = df[df["category_guess"].astype(str).str.contains("eeg", case=False, na=False)].head(50)
    possible_pm = df[df["category_guess"].astype(str).str.contains("pm|metric", case=False, na=False)].head(50)
    possible_labels = df[df["category_guess"].astype(str).str.contains("label|event", case=False, na=False)].head(50)
    possible_wearables = df[df["category_guess"].astype(str).str.contains("wearable|sensor", case=False, na=False)].head(50)

    def table_from_series(s: pd.Series) -> str:
        if s.empty:
            return "_Нет данных._"
        tmp = s.reset_index()
        tmp.columns = ["value", "count"]
        return tmp.to_markdown(index=False)

    def compact_file_table(x: pd.DataFrame) -> str:
        if x.empty:
            return "_Не найдено._"
        cols = [
            "rel_path_project",
            "extension",
            "size_mb",
            "read_status",
            "n_rows_preview",
            "n_cols",
            "time_columns",
            "numeric_columns_count",
        ]
        cols = [c for c in cols if c in x.columns]
        return x[cols].to_markdown(index=False)

    lines = []
    lines.append("# Первичный отчет по данным EEG NIR")
    lines.append("")
    lines.append(f"Корень проекта: `{project_root}`")
    lines.append("")
    lines.append("## Общая сводка")
    lines.append("")
    lines.append(f"- Всего файлов: **{total_files}**")
    lines.append(f"- Суммарный размер: **{total_size_mb:.2f} MB**")
    lines.append("")
    lines.append("## Файлы по источникам")
    lines.append("")
    lines.append(table_from_series(source_counts))
    lines.append("")
    lines.append("## Расширения файлов")
    lines.append("")
    lines.append(table_from_series(ext_counts))
    lines.append("")
    lines.append("## Статус чтения")
    lines.append("")
    lines.append(table_from_series(read_counts))
    lines.append("")
    lines.append("## Предполагаемые категории")
    lines.append("")
    lines.append(table_from_series(category_counts))
    lines.append("")
    lines.append("## 20 самых крупных файлов")
    lines.append("")
    lines.append(compact_file_table(largest))
    lines.append("")
    lines.append("## Возможные EEG-файлы")
    lines.append("")
    lines.append(compact_file_table(possible_eeg))
    lines.append("")
    lines.append("## Возможные PM / cognitive metrics файлы")
    lines.append("")
    lines.append(compact_file_table(possible_pm))
    lines.append("")
    lines.append("## Возможные labels / events / annotations файлы")
    lines.append("")
    lines.append(compact_file_table(possible_labels))
    lines.append("")
    lines.append("## Возможные wearable / sensor файлы")
    lines.append("")
    lines.append(compact_file_table(possible_wearables))
    lines.append("")
    lines.append("## Табличные файлы, которые удалось прочитать")
    lines.append("")
    if readable.empty:
        lines.append("_Нет успешно прочитанных табличных или EEG-файлов._")
    else:
        show = readable.sort_values(["source_root", "rel_path_project"]).head(100)
        lines.append(compact_file_table(show))
    lines.append("")
    lines.append("## Что смотреть дальше")
    lines.append("")
    lines.append("1. Проверить, какие файлы попали в `possible_eeg`.")
    lines.append("2. Проверить наличие временных колонок.")
    lines.append("3. Проверить, есть ли PM-метрики, события, разметка или поведенческие данные.")
    lines.append("4. После этого писать отдельный загрузчик под фактический формат новых данных.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def json_safe_value(x: Any) -> Any:
    if isinstance(x, (list, tuple, dict, str, int, float, bool)) or x is None:
        return x
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)


def prepare_for_saving(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: json.dumps(json_safe_value(x), ensure_ascii=False)
                if isinstance(x, (list, tuple, dict))
                else x
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Первичная инвентаризация EEG-данных.")
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\PycharmProjects\eeg-cognitive-state-nir",
        help="Корень проекта.",
    )
    parser.add_argument(
        "--raw-dirs",
        type=str,
        nargs="+",
        default=[
            r"data\raw\gpn_data",
            r"data\raw\Old_EEG",
        ],
        help="Папки с распакованными исходными данными относительно root или абсолютные пути.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ограничение на число файлов для быстрой проверки. По умолчанию без ограничения.",
    )
    parser.add_argument(
        "--max-preview-rows",
        type=int,
        default=5000,
        help="Сколько строк читать из табличных файлов для диагностики.",
    )

    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    interim_dir = project_root / "data" / "interim"
    reports_dir = project_root / "reports"

    raw_dirs = []
    for raw_dir in args.raw_dirs:
        p = Path(raw_dir)
        if not p.is_absolute():
            p = project_root / p
        raw_dirs.append(p.resolve())

    print("=" * 80)
    print("EEG NIR data inventory")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("Raw dirs:")
    for d in raw_dirs:
        print(f"  - {d} | exists={d.exists()}")
    print("=" * 80)

    all_files: List[Tuple[Path, Path]] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            print(f"[WARN] Папка не найдена: {raw_dir}")
            continue

        for path in raw_dir.rglob("*"):
            if path.is_file():
                all_files.append((path, raw_dir))

    all_files = sorted(all_files, key=lambda x: str(x[0]).lower())

    if args.max_files is not None:
        all_files = all_files[: args.max_files]

    print(f"Found files: {len(all_files)}")

    rows = []
    for i, (path, source_root) in enumerate(all_files, start=1):
        if i == 1 or i % 50 == 0 or i == len(all_files):
            print(f"[{i}/{len(all_files)}] {path}")

        row = scan_one_file(
            path=path,
            project_root=project_root,
            source_root=source_root,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("[WARN] Файлы не найдены. Отчет не создан.")
        return

    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_save = prepare_for_saving(df)

    csv_path = interim_dir / "data_inventory.csv"
    parquet_path = interim_dir / "data_inventory.parquet"
    md_path = reports_dir / "data_inventory_summary.md"

    df_save.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        df_save.to_parquet(parquet_path, index=False)
        print(f"[OK] Saved parquet: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Could not save parquet: {e}")

    make_markdown_summary(df, project_root=project_root, out_path=md_path)

    print(f"[OK] Saved csv: {csv_path}")
    print(f"[OK] Saved markdown report: {md_path}")

    print("\nSummary:")
    print(f"  Files: {len(df)}")
    print(f"  Total size MB: {df['size_mb'].sum():.2f}")
    print("\nExtensions:")
    print(df["extension"].value_counts(dropna=False).head(20).to_string())
    print("\nCategory guesses:")
    print(df["category_guess"].value_counts(dropna=False).to_string())
    print("\nRead status:")
    print(df["read_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()