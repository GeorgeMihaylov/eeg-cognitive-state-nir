# -*- coding: utf-8 -*-
"""
03_validate_catalog_and_columns.py

Проверяет каталог Emotiv-записей:
1. Все ли записи читаются.
2. Есть ли различия колонок между gpn_data и Old_EEG.
3. Какая запись имеет n_cols=182.
4. Какие EEG/PM/POW/MOT/FE колонки являются общими.
5. Какие колонки можно использовать для первого рабочего датасета.

Вход:
    data/interim/emotiv_record_catalog.csv

Выход:
    data/interim/validated_columns.json
    data/interim/validated_columns_report.csv
    reports/validated_columns.md

Запуск:
    python src/03_validate_catalog_and_columns.py
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


EEG_SIGNAL_CHANNELS = [
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1",
    "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4",
]

PM_SCALED_COLUMNS = [
    "PM.Attention.Scaled",
    "PM.Engagement.Scaled",
    "PM.Excitement.Scaled",
    "PM.Stress.Scaled",
    "PM.Relaxation.Scaled",
    "PM.Interest.Scaled",
    "PM.Focus.Scaled",
]

PM_ACTIVE_COLUMNS = [
    "PM.Attention.IsActive",
    "PM.Engagement.IsActive",
    "PM.Excitement.IsActive",
    "PM.Stress.IsActive",
    "PM.Relaxation.IsActive",
    "PM.Interest.IsActive",
    "PM.Focus.IsActive",
]

TIME_COLUMNS_CANDIDATES = [
    "Timestamp",
    "OriginalTimestamp",
]


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
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    return []


def as_set(x: Any) -> Set[str]:
    obj = parse_jsonish_cell(x)

    if isinstance(obj, list):
        return set(map(str, obj))

    if isinstance(obj, dict):
        return set(map(str, obj.keys()))

    return set()


def list_from_cell(x: Any) -> List[str]:
    obj = parse_jsonish_cell(x)

    if isinstance(obj, list):
        return list(map(str, obj))

    return []


def safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def intersection_all(list_of_sets: List[Set[str]]) -> Set[str]:
    if not list_of_sets:
        return set()
    out = set(list_of_sets[0])
    for s in list_of_sets[1:]:
        out &= s
    return out


def union_all(list_of_sets: List[Set[str]]) -> Set[str]:
    out = set()
    for s in list_of_sets:
        out |= s
    return out


def make_presence_table(df: pd.DataFrame, column_field: str) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        source = row["source"]
        subject_id = row["subject_id"]
        main_rel_path = row["main_rel_path"]
        cols = as_set(row[column_field])

        for c in cols:
            rows.append({
                "column": c,
                "column_field": column_field,
                "source": source,
                "subject_id": subject_id,
                "main_rel_path": main_rel_path,
            })

    if not rows:
        return pd.DataFrame()

    tmp = pd.DataFrame(rows)

    agg = (
        tmp.groupby(["column_field", "column"])
        .agg(
            records=("main_rel_path", "nunique"),
            sources=("source", lambda x: sorted(set(x))),
            subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )

    return agg.sort_values(["column_field", "column"]).reset_index(drop=True)


def compact_list(items: List[str], max_items: int = 30) -> str:
    if not items:
        return ""

    items = list(items)
    text = ", ".join(items[:max_items])

    if len(items) > max_items:
        text += f" ... (+{len(items) - max_items})"

    return text


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
        help="Путь к каталогу относительно root.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path

    interim_dir = root / "data" / "interim"
    reports_dir = root / "reports"
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_path)

    print("=" * 80)
    print("Validate Emotiv catalog and columns")
    print("=" * 80)
    print(f"Catalog: {catalog_path}")
    print(f"Records: {len(df)}")

    required_fields = [
        "source",
        "subject_id",
        "main_rel_path",
        "n_cols",
        "time_columns",
        "eeg_columns",
        "pm_columns",
        "bandpower_columns",
        "motion_columns",
        "facial_columns",
    ]

    missing_fields = [c for c in required_fields if c not in df.columns]
    if missing_fields:
        raise ValueError(f"В каталоге нет обязательных колонок: {missing_fields}")

    sources = sorted(df["source"].dropna().unique().tolist())

    column_fields = [
        "time_columns",
        "eeg_columns",
        "pm_columns",
        "bandpower_columns",
        "motion_columns",
        "facial_columns",
    ]

    validation: Dict[str, Any] = {
        "n_records": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique(dropna=True)),
        "sources": sources,
        "records_by_source": df["source"].value_counts(dropna=False).to_dict(),
        "n_cols_distribution": df["n_cols"].value_counts(dropna=False).sort_index().to_dict(),
        "recommended": {},
        "by_field": {},
        "problem_records": [],
    }

    problem_df = df[df["n_cols"] != df["n_cols"].mode().iloc[0]].copy()
    if not problem_df.empty:
        validation["problem_records"] = problem_df[
            ["source", "subject_id", "day", "part", "n_cols", "main_rel_path"]
        ].to_dict(orient="records")

    for field in column_fields:
        sets_all = [as_set(x) for x in df[field]]
        common_all = sorted(intersection_all(sets_all))
        union = sorted(union_all(sets_all))

        by_source = {}
        for source in sources:
            sdf = df[df["source"] == source]
            sets_source = [as_set(x) for x in sdf[field]]
            by_source[source] = {
                "common": sorted(intersection_all(sets_source)),
                "union": sorted(union_all(sets_source)),
                "n_common": len(intersection_all(sets_source)),
                "n_union": len(union_all(sets_source)),
            }

        validation["by_field"][field] = {
            "common_all": common_all,
            "union_all": union,
            "n_common_all": len(common_all),
            "n_union_all": len(union),
            "by_source": by_source,
        }

    all_columns_sets = [as_set(x) for x in df["columns"]]
    common_columns_all = sorted(intersection_all(all_columns_sets))
    union_columns_all = sorted(union_all(all_columns_sets))

    validation["all_columns"] = {
        "n_common_all": len(common_columns_all),
        "n_union_all": len(union_columns_all),
        "common_all": common_columns_all,
        "union_all": union_columns_all,
    }

    common_all_set = set(common_columns_all)

    recommended_eeg = [c for c in EEG_SIGNAL_CHANNELS if c in common_all_set]
    missing_recommended_eeg = [c for c in EEG_SIGNAL_CHANNELS if c not in common_all_set]

    recommended_pm_scaled = [c for c in PM_SCALED_COLUMNS if c in common_all_set]
    missing_pm_scaled = [c for c in PM_SCALED_COLUMNS if c not in common_all_set]

    recommended_pm_active = [c for c in PM_ACTIVE_COLUMNS if c in common_all_set]

    recommended_time = [c for c in TIME_COLUMNS_CANDIDATES if c in common_all_set]

    pow_cols = sorted([
        c for c in common_all_set
        if c.startswith("POW.")
    ])

    motion_cols = sorted([
        c for c in common_all_set
        if c.startswith("MOT.") or c.startswith("MC.")
    ])

    facial_cols = sorted([
        c for c in common_all_set
        if c.startswith("FE.")
    ])

    validation["recommended"] = {
        "time_columns": recommended_time,
        "eeg_signal_columns": recommended_eeg,
        "missing_eeg_signal_columns": missing_recommended_eeg,
        "pm_scaled_columns": recommended_pm_scaled,
        "missing_pm_scaled_columns": missing_pm_scaled,
        "pm_active_columns": recommended_pm_active,
        "pow_columns": pow_cols,
        "motion_columns": motion_cols,
        "facial_columns": facial_cols,
    }

    presence_parts = []
    for field in column_fields:
        p = make_presence_table(df, field)
        if not p.empty:
            presence_parts.append(p)

    if presence_parts:
        presence_df = pd.concat(presence_parts, ignore_index=True)
    else:
        presence_df = pd.DataFrame()

    presence_csv = interim_dir / "validated_columns_report.csv"
    if not presence_df.empty:
        presence_df.to_csv(presence_csv, index=False, encoding="utf-8-sig")

    json_path = interim_dir / "validated_columns.json"
    json_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = reports_dir / "validated_columns.md"
    make_markdown_report(
        validation=validation,
        df=df,
        presence_df=presence_df,
        out_path=md_path,
        root=root,
    )

    print(f"[OK] Saved json: {json_path}")
    print(f"[OK] Saved csv: {presence_csv}")
    print(f"[OK] Saved report: {md_path}")

    print("\nSummary:")
    print(f"Records: {validation['n_records']}")
    print(f"Subjects: {validation['n_subjects']}")
    print(f"Sources: {validation['records_by_source']}")
    print(f"n_cols distribution: {validation['n_cols_distribution']}")
    print(f"Common columns: {validation['all_columns']['n_common_all']}")
    print(f"Union columns: {validation['all_columns']['n_union_all']}")
    print(f"Recommended EEG columns: {len(recommended_eeg)}")
    print(f"Recommended PM scaled columns: {len(recommended_pm_scaled)}")


def make_markdown_report(
    validation: Dict[str, Any],
    df: pd.DataFrame,
    presence_df: pd.DataFrame,
    out_path: Path,
    root: Path,
) -> None:
    lines = []

    lines.append("# Валидация Emotiv-каталога и колонок")
    lines.append("")
    lines.append(f"Корень проекта: `{root}`")
    lines.append("")

    lines.append("## Общая сводка")
    lines.append("")
    lines.append(f"- Записей: **{validation['n_records']}**")
    lines.append(f"- Субъектов: **{validation['n_subjects']}**")
    lines.append(f"- Источники: `{validation['records_by_source']}`")
    lines.append(f"- Распределение `n_cols`: `{validation['n_cols_distribution']}`")
    lines.append(f"- Общих колонок во всех записях: **{validation['all_columns']['n_common_all']}**")
    lines.append(f"- Уникальных колонок суммарно: **{validation['all_columns']['n_union_all']}**")
    lines.append("")

    lines.append("## Проблемные записи")
    lines.append("")
    problems = validation.get("problem_records", [])
    if problems:
        pdf = pd.DataFrame(problems)
        lines.append(pdf.to_markdown(index=False))
    else:
        lines.append("_Проблемных записей по числу колонок не найдено._")
    lines.append("")

    lines.append("## Общие колонки по группам")
    lines.append("")
    rows = []
    for field, info in validation["by_field"].items():
        rows.append({
            "field": field,
            "n_common_all": info["n_common_all"],
            "n_union_all": info["n_union_all"],
        })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

    lines.append("## Рекомендуемый минимальный набор колонок")
    lines.append("")

    rec = validation["recommended"]

    lines.append("### Time")
    lines.append("")
    lines.append(f"`{compact_list(rec['time_columns'])}`")
    lines.append("")

    lines.append("### EEG signal channels")
    lines.append("")
    lines.append(f"`{compact_list(rec['eeg_signal_columns'], 50)}`")
    lines.append("")
    if rec["missing_eeg_signal_columns"]:
        lines.append("Недостающие EEG-каналы:")
        lines.append("")
        lines.append(f"`{compact_list(rec['missing_eeg_signal_columns'], 50)}`")
        lines.append("")

    lines.append("### PM scaled columns")
    lines.append("")
    lines.append(f"`{compact_list(rec['pm_scaled_columns'], 50)}`")
    lines.append("")
    if rec["missing_pm_scaled_columns"]:
        lines.append("Недостающие PM scaled:")
        lines.append("")
        lines.append(f"`{compact_list(rec['missing_pm_scaled_columns'], 50)}`")
        lines.append("")

    lines.append("### PM active columns")
    lines.append("")
    lines.append(f"`{compact_list(rec['pm_active_columns'], 50)}`")
    lines.append("")

    lines.append("### POW columns")
    lines.append("")
    lines.append(f"Количество: **{len(rec['pow_columns'])}**")
    lines.append("")
    lines.append(f"`{compact_list(rec['pow_columns'], 40)}`")
    lines.append("")

    lines.append("### Motion columns")
    lines.append("")
    lines.append(f"Количество: **{len(rec['motion_columns'])}**")
    lines.append("")
    lines.append(f"`{compact_list(rec['motion_columns'], 40)}`")
    lines.append("")

    lines.append("### Facial columns")
    lines.append("")
    lines.append(f"Количество: **{len(rec['facial_columns'])}**")
    lines.append("")
    lines.append(f"`{compact_list(rec['facial_columns'], 40)}`")
    lines.append("")

    lines.append("## Сравнение источников")
    lines.append("")
    src_rows = []
    for field, info in validation["by_field"].items():
        for source, source_info in info["by_source"].items():
            src_rows.append({
                "field": field,
                "source": source,
                "n_common": source_info["n_common"],
                "n_union": source_info["n_union"],
            })
    lines.append(pd.DataFrame(src_rows).to_markdown(index=False))
    lines.append("")

    lines.append("## Вывод")
    lines.append("")
    lines.append("1. Если число общих колонок близко к 183, можно строить единый загрузчик для `gpn_data` и `Old_EEG`.")
    lines.append("2. Если полный набор PM/EEG-каналов общий, первый датасет можно строить сразу по двум источникам с обязательным полем `source`.")
    lines.append("3. Для моделей надо делать отдельные эксперименты: `gpn_data`, `Old_EEG`, `all`, а также cross-source проверку.")
    lines.append("4. Следующий этап — построение оконных сегментов и агрегация PM-метрик.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()