# -*- coding: utf-8 -*-
"""
06_eda_windowed_dataset.py

EDA для оконного PM/POW датасета.

Вход:
    data/processed/windowed_pm_dataset_w10.parquet

Выход:
    reports/eda_windowed_pm_dataset_w10.md
    reports/figures/eda_windowed_pm_dataset_w10/*.png
    data/interim/eda_windowed_pm_dataset_w10_summary.csv
    data/interim/eda_windowed_pm_dataset_w10_missingness.csv
    data/interim/eda_windowed_pm_dataset_w10_subject_summary.csv
    data/interim/eda_windowed_pm_dataset_w10_source_summary.csv

Запуск:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\06_eda_windowed_dataset.py

Для другого датасета:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\06_eda_windowed_dataset.py ^
      --dataset data\\processed\\windowed_pm_dataset_w10.parquet ^
      --output-name eda_windowed_pm_dataset_w10
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLS_DEFAULT = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
    "target_main",
]

META_COLS = [
    "record_id",
    "source",
    "subject_id",
    "day",
    "part",
    "datetime_from_name",
    "t_center",
    "t_start",
    "t_end",
    "label_q5",
]

REQUIRED_COLS = [
    "source",
    "subject_id",
    "day",
    "record_id",
    "target_main",
    "label_q5",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def df_to_markdown_safe(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def infer_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = list(df.columns)

    groups = {
        "meta": [c for c in META_COLS if c in cols],
        "target": [c for c in TARGET_COLS_DEFAULT if c in cols],
        "label": [c for c in cols if c.startswith("label_")],
        "pow": [c for c in cols if c.startswith("POW.")],
        "pm": [c for c in cols if c.startswith("PM.")],
        "motion": [c for c in cols if c.startswith("MOT.") or c.startswith("MC.")],
        "facial": [c for c in cols if c.startswith("FE.")],
    }

    known = set()
    for v in groups.values():
        known.update(v)

    groups["other"] = [c for c in cols if c not in known]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    groups["numeric"] = numeric_cols

    feature_cols = [
        c for c in numeric_cols
        if c not in groups["target"]
        and c not in groups["label"]
        and c not in {"t_center", "t_start", "t_end"}
    ]
    groups["features_numeric"] = feature_cols

    return groups


def basic_dataset_summary(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []

    rows.append({"metric": "rows", "value": len(df)})
    rows.append({"metric": "columns", "value": df.shape[1]})

    for col in ["source", "subject_id", "record_id", "day"]:
        if col in df.columns:
            rows.append({"metric": f"unique_{col}", "value": df[col].nunique(dropna=True)})

    for group_name, group_cols in groups.items():
        rows.append({"metric": f"columns_{group_name}", "value": len(group_cols)})

    if "target_main" in df.columns:
        rows.append({"metric": "target_main_non_null", "value": int(df["target_main"].notna().sum())})
        rows.append({"metric": "target_main_null", "value": int(df["target_main"].isna().sum())})
        rows.append({"metric": "target_main_non_null_ratio", "value": float(df["target_main"].notna().mean())})

    if "label_q5" in df.columns:
        rows.append({"metric": "label_q5_non_null", "value": int(df["label_q5"].notna().sum())})
        rows.append({"metric": "label_q5_null", "value": int(df["label_q5"].isna().sum())})
        rows.append({"metric": "label_q5_non_null_ratio", "value": float(df["label_q5"].notna().mean())})

    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []

    for c in df.columns:
        miss = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "missing_count": miss,
            "missing_ratio": miss / max(1, n),
            "non_null_count": int(df[c].notna().sum()),
            "dtype": str(df[c].dtype),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["missing_ratio", "missing_count"], ascending=False).reset_index(drop=True)
    return out


def group_missingness_summary(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []

    for group, cols in groups.items():
        if not cols:
            continue

        sub = df[cols]
        rows.append({
            "group": group,
            "n_columns": len(cols),
            "mean_missing_ratio": float(sub.isna().mean().mean()),
            "max_missing_ratio": float(sub.isna().mean().max()),
            "columns_with_any_missing": int((sub.isna().sum() > 0).sum()),
            "columns_all_missing": int((sub.isna().sum() == len(sub)).sum()),
        })

    return pd.DataFrame(rows).sort_values("mean_missing_ratio", ascending=False)


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source, g in df.groupby("source", dropna=False):
        row = {
            "source": source,
            "windows": len(g),
            "records": g["record_id"].nunique(dropna=True) if "record_id" in g.columns else np.nan,
            "subjects": g["subject_id"].nunique(dropna=True) if "subject_id" in g.columns else np.nan,
            "target_main_non_null": int(g["target_main"].notna().sum()) if "target_main" in g.columns else np.nan,
            "target_main_non_null_ratio": float(g["target_main"].notna().mean()) if "target_main" in g.columns else np.nan,
            "label_q5_non_null": int(g["label_q5"].notna().sum()) if "label_q5" in g.columns else np.nan,
            "label_q5_non_null_ratio": float(g["label_q5"].notna().mean()) if "label_q5" in g.columns else np.nan,
        }

        if "target_main" in g.columns:
            s = pd.to_numeric(g["target_main"], errors="coerce")
            row.update({
                "target_main_mean": float(s.mean()),
                "target_main_std": float(s.std()),
                "target_main_min": float(s.min()),
                "target_main_median": float(s.median()),
                "target_main_max": float(s.max()),
            })

        rows.append(row)

    return pd.DataFrame(rows).sort_values("windows", ascending=False).reset_index(drop=True)


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for subject_id, g in df.groupby("subject_id", dropna=False):
        row = {
            "subject_id": subject_id,
            "windows": len(g),
            "records": g["record_id"].nunique(dropna=True) if "record_id" in g.columns else np.nan,
            "sources": ",".join(sorted(map(str, g["source"].dropna().unique()))) if "source" in g.columns else "",
            "days": ",".join(sorted(map(str, g["day"].dropna().unique()))) if "day" in g.columns else "",
            "target_main_non_null": int(g["target_main"].notna().sum()) if "target_main" in g.columns else np.nan,
            "target_main_non_null_ratio": float(g["target_main"].notna().mean()) if "target_main" in g.columns else np.nan,
        }

        if "target_main" in g.columns:
            s = pd.to_numeric(g["target_main"], errors="coerce")
            row.update({
                "target_main_mean": float(s.mean()),
                "target_main_std": float(s.std()),
                "target_main_median": float(s.median()),
            })

        rows.append(row)

    return pd.DataFrame(rows).sort_values("windows", ascending=False).reset_index(drop=True)


def numeric_describe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    desc = df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc = desc.reset_index().rename(columns={"index": "column"})
    desc["missing_ratio"] = df[cols].isna().mean().values
    return desc


def outlier_summary(df: pd.DataFrame, cols: List[str], max_cols: int = 500) -> pd.DataFrame:
    """
    Простая оценка выбросов по IQR.
    Не удаляет выбросы, только считает долю.
    """
    rows = []
    cols = [c for c in cols if c in df.columns][:max_cols]

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 10:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr == 0:
            lower = np.nan
            upper = np.nan
            out_ratio = 0.0
            out_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out = (s < lower) | (s > upper)
            out_count = int(out.sum())
            out_ratio = float(out.mean())

        rows.append({
            "column": c,
            "count": int(len(s)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_iqr_bound": float(lower) if np.isfinite(lower) else np.nan,
            "upper_iqr_bound": float(upper) if np.isfinite(upper) else np.nan,
            "outlier_count_iqr": out_count,
            "outlier_ratio_iqr": out_ratio,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("outlier_ratio_iqr", ascending=False).reset_index(drop=True)


def correlation_summary(df: pd.DataFrame, target_col: str, feature_cols: List[str], max_features: int = 400) -> pd.DataFrame:
    """
    Корреляция признаков с target_main.
    Для EDA используем Pearson и Spearman.
    """
    if target_col not in df.columns:
        return pd.DataFrame()

    y = pd.to_numeric(df[target_col], errors="coerce")
    rows = []

    for c in feature_cols[:max_features]:
        x = pd.to_numeric(df[c], errors="coerce")
        valid = x.notna() & y.notna()

        if valid.sum() < 20:
            continue

        xv = x[valid]
        yv = y[valid]

        pearson = xv.corr(yv, method="pearson")
        spearman = xv.corr(yv, method="spearman")

        rows.append({
            "feature": c,
            "n_valid": int(valid.sum()),
            "pearson_target_main": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman_target_main": float(spearman) if pd.notna(spearman) else np.nan,
            "abs_pearson": abs(float(pearson)) if pd.notna(pearson) else np.nan,
            "abs_spearman": abs(float(spearman)) if pd.notna(spearman) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def plot_label_distribution(df: pd.DataFrame, fig_dir: Path, output_name: str) -> Path:
    path = fig_dir / f"{output_name}_label_q5_distribution.png"

    counts = df["label_q5"].value_counts(dropna=False).sort_index()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("label_q5 distribution")
    plt.xlabel("label_q5")
    plt.ylabel("windows")
    save_plot(path)

    return path


def plot_source_distribution(df: pd.DataFrame, fig_dir: Path, output_name: str) -> Path:
    path = fig_dir / f"{output_name}_source_distribution.png"

    counts = df["source"].value_counts(dropna=False)

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Windows by source")
    plt.xlabel("source")
    plt.ylabel("windows")
    save_plot(path)

    return path


def plot_day_distribution(df: pd.DataFrame, fig_dir: Path, output_name: str) -> Path:
    path = fig_dir / f"{output_name}_day_distribution.png"

    counts = df["day"].value_counts(dropna=False)

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Windows by day")
    plt.xlabel("day")
    plt.ylabel("windows")
    save_plot(path)

    return path


def plot_target_histograms(df: pd.DataFrame, target_cols: List[str], fig_dir: Path, output_name: str) -> List[Path]:
    paths = []

    for c in target_cols:
        if c not in df.columns:
            continue

        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue

        path = fig_dir / f"{output_name}_{c}_hist.png"

        plt.figure(figsize=(8, 5))
        plt.hist(s, bins=50)
        plt.title(f"{c} distribution")
        plt.xlabel(c)
        plt.ylabel("count")
        save_plot(path)

        paths.append(path)

    return paths


def plot_target_by_source_boxplot(df: pd.DataFrame, fig_dir: Path, output_name: str) -> Path:
    path = fig_dir / f"{output_name}_target_main_by_source_boxplot.png"

    sub = df[["source", "target_main"]].copy()
    sub["target_main"] = pd.to_numeric(sub["target_main"], errors="coerce")
    sub = sub.dropna()

    sources = sorted(sub["source"].dropna().unique())
    data = [sub.loc[sub["source"] == source, "target_main"].values for source in sources]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=sources)
    plt.title("target_main by source")
    plt.xlabel("source")
    plt.ylabel("target_main")
    save_plot(path)

    return path


def plot_windows_by_subject(subject_df: pd.DataFrame, fig_dir: Path, output_name: str, top_n: int = 40) -> Path:
    path = fig_dir / f"{output_name}_windows_by_subject_top{top_n}.png"

    show = subject_df.sort_values("windows", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(show["subject_id"].astype(str), show["windows"])
    plt.title(f"Windows by subject, top {top_n}")
    plt.xlabel("subject_id")
    plt.ylabel("windows")
    plt.xticks(rotation=90)
    save_plot(path)

    return path


def plot_missingness_top(missing_df: pd.DataFrame, fig_dir: Path, output_name: str, top_n: int = 40) -> Path:
    path = fig_dir / f"{output_name}_missingness_top{top_n}.png"

    show = missing_df.sort_values("missing_ratio", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(show["column"].astype(str), show["missing_ratio"])
    plt.title(f"Top {top_n} columns by missing ratio")
    plt.xlabel("column")
    plt.ylabel("missing ratio")
    plt.xticks(rotation=90)
    save_plot(path)

    return path


def plot_feature_correlation_top(corr_df: pd.DataFrame, fig_dir: Path, output_name: str, top_n: int = 30) -> Path:
    path = fig_dir / f"{output_name}_top_feature_spearman_target_main.png"

    if corr_df.empty:
        return path

    show = corr_df.sort_values("abs_spearman", ascending=False).head(top_n).copy()
    show = show.sort_values("spearman_target_main", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(show["feature"], show["spearman_target_main"])
    plt.title(f"Top {top_n} Spearman correlations with target_main")
    plt.xlabel("Spearman correlation")
    plt.ylabel("feature")
    save_plot(path)

    return path


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def make_markdown_report(
    root: Path,
    dataset_path: Path,
    output_name: str,
    summary_df: pd.DataFrame,
    source_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    group_missing_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    target_desc_df: pd.DataFrame,
    feature_desc_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    figures: List[Path],
    report_path: Path,
) -> None:
    lines = []

    lines.append(f"# EDA report: {output_name}")
    lines.append("")
    lines.append(f"Dataset: `{dataset_path}`")
    lines.append("")

    lines.append("## Basic summary")
    lines.append("")
    lines.append(df_to_markdown_safe(summary_df, index=False))
    lines.append("")

    lines.append("## Source summary")
    lines.append("")
    lines.append(df_to_markdown_safe(source_df, index=False))
    lines.append("")

    lines.append("## Subject summary preview")
    lines.append("")
    lines.append(df_to_markdown_safe(subject_df.head(30), index=False))
    lines.append("")

    lines.append("## Missingness by feature group")
    lines.append("")
    lines.append(df_to_markdown_safe(group_missing_df, index=False))
    lines.append("")

    lines.append("## Top columns by missingness")
    lines.append("")
    lines.append(df_to_markdown_safe(missing_df.head(30), index=False))
    lines.append("")

    lines.append("## Target descriptive statistics")
    lines.append("")
    if target_desc_df.empty:
        lines.append("_No target columns found._")
    else:
        lines.append(df_to_markdown_safe(target_desc_df, index=False))
    lines.append("")

    lines.append("## Feature descriptive statistics preview")
    lines.append("")
    if feature_desc_df.empty:
        lines.append("_No numeric feature columns found._")
    else:
        lines.append(df_to_markdown_safe(feature_desc_df.head(40), index=False))
    lines.append("")

    lines.append("## Top IQR outlier ratios")
    lines.append("")
    if outlier_df.empty:
        lines.append("_No outlier statistics._")
    else:
        lines.append(df_to_markdown_safe(outlier_df.head(40), index=False))
    lines.append("")

    lines.append("## Top feature correlations with target_main")
    lines.append("")
    if corr_df.empty:
        lines.append("_No correlation statistics._")
    else:
        show_cols = [
            "feature",
            "n_valid",
            "pearson_target_main",
            "spearman_target_main",
            "abs_spearman",
        ]
        show_cols = [c for c in show_cols if c in corr_df.columns]
        lines.append(df_to_markdown_safe(corr_df[show_cols].head(40), index=False))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        rel = relative_path(fig, root)
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. This EDA checks whether the 10-second windowed PM/POW dataset is suitable for baseline modeling.")
    lines.append("2. `source` must be preserved because `gpn_data` and `Old_EEG` are different data domains.")
    lines.append("3. Rows with missing `label_q5` or `target_main` should be excluded from the first supervised baseline.")
    lines.append("4. Quantile labels are balanced by construction and should be treated as weak labels.")
    lines.append("5. Strong source-level or subject-level distribution shifts should be considered during validation.")
    lines.append("6. The first baseline should use GroupKFold/LOSO by `subject_id`, not only random split.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\PycharmProjects\eeg-cognitive-state-nir",
        help="Корень проекта.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"data\processed\windowed_pm_dataset_w10.parquet",
        help="Путь к оконному датасету относительно root.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="eda_windowed_pm_dataset_w10",
        help="Префикс выходных файлов.",
    )
    parser.add_argument(
        "--max-corr-features",
        type=int,
        default=400,
        help="Максимум признаков для корреляций.",
    )
    parser.add_argument(
        "--max-outlier-features",
        type=int,
        default=500,
        help="Максимум признаков для IQR-анализа.",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    root = Path(args.root).resolve()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    output_name = args.output_name

    reports_dir = root / "reports"
    interim_dir = root / "data" / "interim"
    fig_dir = reports_dir / "figures" / output_name

    ensure_dir(reports_dir)
    ensure_dir(interim_dir)
    ensure_dir(fig_dir)

    print("=" * 80)
    print("EDA windowed dataset")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")
    print(f"Output name: {output_name}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    print(f"Loaded dataset: rows={len(df)}, cols={df.shape[1]}")

    groups = infer_feature_groups(df)

    summary_df = basic_dataset_summary(df, groups)
    missing_df = missingness_table(df)
    group_missing_df = group_missingness_summary(df, groups)
    source_df = source_summary(df)
    subject_df = subject_summary(df)

    target_cols = [c for c in TARGET_COLS_DEFAULT if c in df.columns]
    target_desc_df = numeric_describe(df, target_cols)

    feature_desc_df = numeric_describe(df, groups["features_numeric"])

    outlier_df = outlier_summary(
        df,
        cols=groups["features_numeric"],
        max_cols=args.max_outlier_features,
    )

    corr_df = correlation_summary(
        df,
        target_col="target_main",
        feature_cols=groups["features_numeric"],
        max_features=args.max_corr_features,
    )

    summary_path = interim_dir / f"{output_name}_summary.csv"
    missing_path = interim_dir / f"{output_name}_missingness.csv"
    group_missing_path = interim_dir / f"{output_name}_group_missingness.csv"
    source_path = interim_dir / f"{output_name}_source_summary.csv"
    subject_path = interim_dir / f"{output_name}_subject_summary.csv"
    target_desc_path = interim_dir / f"{output_name}_target_describe.csv"
    feature_desc_path = interim_dir / f"{output_name}_feature_describe.csv"
    outlier_path = interim_dir / f"{output_name}_outlier_summary.csv"
    corr_path = interim_dir / f"{output_name}_feature_target_correlations.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")
    group_missing_df.to_csv(group_missing_path, index=False, encoding="utf-8-sig")
    source_df.to_csv(source_path, index=False, encoding="utf-8-sig")
    subject_df.to_csv(subject_path, index=False, encoding="utf-8-sig")
    target_desc_df.to_csv(target_desc_path, index=False, encoding="utf-8-sig")
    feature_desc_df.to_csv(feature_desc_path, index=False, encoding="utf-8-sig")
    outlier_df.to_csv(outlier_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")

    figures = []
    figures.append(plot_label_distribution(df, fig_dir, output_name))
    figures.append(plot_source_distribution(df, fig_dir, output_name))
    figures.append(plot_day_distribution(df, fig_dir, output_name))
    figures.extend(plot_target_histograms(df, target_cols, fig_dir, output_name))
    figures.append(plot_target_by_source_boxplot(df, fig_dir, output_name))
    figures.append(plot_windows_by_subject(subject_df, fig_dir, output_name))
    figures.append(plot_missingness_top(missing_df, fig_dir, output_name))
    figures.append(plot_feature_correlation_top(corr_df, fig_dir, output_name))

    report_path = reports_dir / f"{output_name}.md"

    make_markdown_report(
        root=root,
        dataset_path=dataset_path,
        output_name=output_name,
        summary_df=summary_df,
        source_df=source_df,
        subject_df=subject_df,
        group_missing_df=group_missing_df,
        missing_df=missing_df,
        target_desc_df=target_desc_df,
        feature_desc_df=feature_desc_df,
        outlier_df=outlier_df,
        corr_df=corr_df,
        figures=figures,
        report_path=report_path,
    )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {missing_path}")
    print(f"  {group_missing_path}")
    print(f"  {source_path}")
    print(f"  {subject_path}")
    print(f"  {target_desc_path}")
    print(f"  {feature_desc_path}")
    print(f"  {outlier_path}")
    print(f"  {corr_path}")
    print(f"  {report_path}")
    print(f"  figures: {fig_dir}")

    print("\nKey summary:")
    print(summary_df.to_string(index=False))

    print("\nSource summary:")
    print(source_df.to_string(index=False))

    print("\nLabel distribution:")
    print(df["label_q5"].value_counts(dropna=False).sort_index().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()