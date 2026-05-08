# -*- coding: utf-8 -*-
"""
10_describe_multi_pm_baseline.py

Описательная статистика результатов multi-PM baseline.

Скрипт анализирует:
- какие PM-таргеты использовались;
- сколько строк было доступно по каждому таргету;
- какие модели обучались;
- как считались fold-метрики;
- какие таргеты лучше предсказываются;
- насколько стабильны метрики между fold.

Пример запуска:

D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\10_describe_multi_pm_baseline.py ^
  --run-dir reports\\runs\\20260508_133220_multi_pm_test_pow_plus_eeg_log_pow

Если есть отдельный target_summary.csv:

D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\10_describe_multi_pm_baseline.py ^
  --run-dir reports\\runs\\20260508_133220_multi_pm_test_pow_plus_eeg_log_pow ^
  --target-summary data\\processed\\target_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def find_first_existing(run_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    return None


def parse_train_log(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Парсит train.log, если fold_metrics.csv отсутствует.

    Возвращает:
        target_info_df
        fold_metrics_df
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    target_info_rows = []
    target_info_re = re.compile(
        r"Target=(?P<target>\w+) \| column=(?P<column>[^|]+) "
        r"\| rows=(?P<rows>\d+) \| subjects=(?P<subjects>\d+) "
        r"\| records=(?P<records>\d+) \| y_mean=(?P<y_mean>[-.\d]+) "
        r"\| y_std=(?P<y_std>[-.\d]+)"
    )

    for m in target_info_re.finditer(text):
        target_info_rows.append(
            {
                "target_name": m.group("target"),
                "target_column": m.group("column").strip(),
                "rows": int(m.group("rows")),
                "subjects": int(m.group("subjects")),
                "records": int(m.group("records")),
                "y_mean": float(m.group("y_mean")),
                "y_std": float(m.group("y_std")),
            }
        )

    metric_rows = []
    metric_re = re.compile(
        r"Target=(?P<target>\w+) \| split=(?P<split>[^|]+) "
        r"\| model=(?P<model>\w+) \| RMSE=(?P<rmse>[-.\d]+) "
        r"\| R2=(?P<r2>[-.\d]+) \| Spearman=(?P<spearman>[-.\d]+) "
        r"\| elapsed=(?P<elapsed>[-.\d]+)s"
    )

    for m in metric_re.finditer(text):
        metric_rows.append(
            {
                "target_name": m.group("target"),
                "validation": m.group("split").strip(),
                "model": m.group("model"),
                "rmse": float(m.group("rmse")),
                "r2": float(m.group("r2")),
                "spearman": float(m.group("spearman")),
                "elapsed_s": float(m.group("elapsed")),
            }
        )

    return pd.DataFrame(target_info_rows), pd.DataFrame(metric_rows)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит возможные варианты имен колонок к единому виду:
    target_name, model, mae, rmse, r2, pearson, spearman, elapsed_s.
    """
    out = df.copy()

    rename = {}

    candidates = {
        "target_name": ["target", "pm_target", "target_name"],
        "model": ["model", "model_name"],
        "mae": ["mae", "mae_mean"],
        "rmse": ["rmse", "rmse_mean"],
        "r2": ["r2", "r2_mean"],
        "pearson": ["pearson", "pearson_mean"],
        "spearman": ["spearman", "spearman_mean"],
        "elapsed_s": ["elapsed_s", "elapsed"],
    }

    for canonical, names in candidates.items():
        for name in names:
            if name in out.columns and canonical not in out.columns:
                rename[name] = canonical
                break

    out = out.rename(columns=rename)
    return out


def aggregate_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    fold_df = normalize_metric_columns(fold_df)

    metric_cols = [c for c in ["mae", "rmse", "r2", "pearson", "spearman", "elapsed_s"] if c in fold_df.columns]

    rows = []
    for (target, model), g in fold_df.groupby(["target_name", "model"], dropna=False):
        row = {
            "target_name": target,
            "model": model,
            "folds": len(g),
        }

        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std()) if len(g) > 1 else 0.0
            row[f"{c}_min"] = float(g[c].min())
            row[f"{c}_max"] = float(g[c].max())

        rows.append(row)

    return pd.DataFrame(rows)


def make_target_ranking(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Выбирает лучшую модель для каждого target.
    Приоритет:
        1. max r2_mean
        2. max spearman_mean
        3. min rmse_mean
    """
    df = agg_df.copy()

    sort_cols = []
    ascending = []

    if "r2_mean" in df.columns:
        sort_cols.append("r2_mean")
        ascending.append(False)

    if "spearman_mean" in df.columns:
        sort_cols.append("spearman_mean")
        ascending.append(False)

    if "rmse_mean" in df.columns:
        sort_cols.append("rmse_mean")
        ascending.append(True)

    if not sort_cols:
        return pd.DataFrame()

    best = (
        df.sort_values(["target_name"] + sort_cols, ascending=[True] + ascending)
        .groupby("target_name", as_index=False)
        .head(1)
        .copy()
    )

    rank_sort = []
    rank_ascending = []

    if "r2_mean" in best.columns:
        rank_sort.append("r2_mean")
        rank_ascending.append(False)

    if "spearman_mean" in best.columns:
        rank_sort.append("spearman_mean")
        rank_ascending.append(False)

    if "rmse_mean" in best.columns:
        rank_sort.append("rmse_mean")
        rank_ascending.append(True)

    best = best.sort_values(rank_sort, ascending=rank_ascending).reset_index(drop=True)
    best.insert(0, "rank", np.arange(1, len(best) + 1))

    return best


def make_global_descriptive_stats(agg_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in [
            "mae_mean",
            "rmse_mean",
            "r2_mean",
            "pearson_mean",
            "spearman_mean",
            "elapsed_s_mean",
            "elapsed_s_std",
        ]
        if c in agg_df.columns
    ]

    rows = []
    for c in metric_cols:
        s = pd.to_numeric(agg_df[c], errors="coerce").dropna()
        if s.empty:
            continue

        rows.append(
            {
                "metric": c,
                "count": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )

    return pd.DataFrame(rows)


def df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def make_report(
    report_path: Path,
    run_dir: Path,
    target_info: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    agg_metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    global_stats: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Descriptive statistics for multi-PM baseline")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## 1. How metrics were obtained")
    lines.append("")
    lines.append("For each PM target, rows with non-missing target values were selected.")
    lines.append("The model was evaluated with subject-aware GroupKFold, so test subjects were not present in train for the same fold.")
    lines.append("For each target and each fold, HGB/LGBM regressors were trained and evaluated.")
    lines.append("Final metrics are mean values across folds.")
    lines.append("")

    lines.append("## 2. Target availability and distribution")
    lines.append("")
    if target_info.empty:
        lines.append("_No target info was found._")
    else:
        lines.append(df_to_markdown_safe(target_info))
    lines.append("")

    lines.append("## 3. Fold-level metrics preview")
    lines.append("")
    if fold_metrics.empty:
        lines.append("_No fold-level metrics were found._")
    else:
        lines.append(df_to_markdown_safe(fold_metrics.head(40)))
    lines.append("")

    lines.append("## 4. Aggregated metrics by target and model")
    lines.append("")
    if agg_metrics.empty:
        lines.append("_No aggregated metrics were computed._")
    else:
        lines.append(df_to_markdown_safe(agg_metrics))
    lines.append("")

    lines.append("## 5. Best model per target")
    lines.append("")
    if ranking.empty:
        lines.append("_No ranking was computed._")
    else:
        lines.append(df_to_markdown_safe(ranking))
    lines.append("")

    lines.append("## 6. Global descriptive statistics")
    lines.append("")
    if global_stats.empty:
        lines.append("_No global descriptive statistics were computed._")
    else:
        lines.append(df_to_markdown_safe(global_stats))
    lines.append("")

    lines.append("## 7. Interpretation")
    lines.append("")
    lines.append("The most reliable targets should be selected by both R2 and Spearman.")
    lines.append("R2 shows explained variance, while Spearman shows rank-order agreement between true and predicted PM level.")
    lines.append("Large fold-level standard deviation indicates instability across subjects.")
    lines.append("The result should be treated as exploratory if it was computed with --max-rows.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Папка reports/runs/<run_id> последнего multi-PM запуска.",
    )
    parser.add_argument(
        "--target-summary",
        type=str,
        default=None,
        help="Опционально: готовый target_summary.csv.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "descriptive_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"

    # 1. Пытаемся найти fold-level метрики.
    fold_metrics_path = find_first_existing(
        run_dir,
        [
            "fold_metrics.csv",
            "target_metrics.csv",
            "metrics.csv",
            "all_metrics.csv",
        ],
    )

    target_info = pd.DataFrame()
    fold_metrics = pd.DataFrame()

    if fold_metrics_path is not None:
        fold_metrics = pd.read_csv(fold_metrics_path)
        fold_metrics = normalize_metric_columns(fold_metrics)

    # 2. Если есть train.log, парсим из него target info и fold metrics.
    if log_path.exists():
        parsed_target_info, parsed_fold_metrics = parse_train_log(log_path)

        if target_info.empty:
            target_info = parsed_target_info

        if fold_metrics.empty:
            fold_metrics = parsed_fold_metrics

    # 3. Если есть готовый target_summary.csv, читаем его.
    target_summary = pd.DataFrame()
    if args.target_summary is not None:
        target_summary_path = Path(args.target_summary).resolve()
        if target_summary_path.exists():
            target_summary = pd.read_csv(target_summary_path)
            target_summary = normalize_metric_columns(target_summary)

    local_target_summary_path = run_dir / "target_summary.csv"
    if target_summary.empty and local_target_summary_path.exists():
        target_summary = pd.read_csv(local_target_summary_path)
        target_summary = normalize_metric_columns(target_summary)

    # 4. Агрегируем fold-level метрики.
    if not fold_metrics.empty:
        agg_metrics = aggregate_fold_metrics(fold_metrics)
    elif not target_summary.empty:
        # Если есть только summary, используем ее как aggregated.
        agg_metrics = target_summary.copy()
        rename = {}
        for c in ["mae", "rmse", "r2", "pearson", "spearman"]:
            if c in agg_metrics.columns:
                rename[c] = f"{c}_mean"
        agg_metrics = agg_metrics.rename(columns=rename)
    else:
        raise RuntimeError("No metrics found. Expected train.log, fold_metrics.csv, target_metrics.csv, or target_summary.csv.")

    ranking = make_target_ranking(agg_metrics)
    global_stats = make_global_descriptive_stats(agg_metrics)

    # 5. Сохраняем.
    target_info_path = out_dir / "target_availability_stats.csv"
    fold_metrics_out_path = out_dir / "fold_metrics_normalized.csv"
    agg_metrics_path = out_dir / "metrics_by_target_model.csv"
    ranking_path = out_dir / "target_ranking.csv"
    global_stats_path = out_dir / "global_metric_descriptive_stats.csv"
    report_path = out_dir / "report_descriptive.md"

    target_info.to_csv(target_info_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(fold_metrics_out_path, index=False, encoding="utf-8-sig")
    agg_metrics.to_csv(agg_metrics_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    global_stats.to_csv(global_stats_path, index=False, encoding="utf-8-sig")

    make_report(
        report_path=report_path,
        run_dir=run_dir,
        target_info=target_info,
        fold_metrics=fold_metrics,
        agg_metrics=agg_metrics,
        ranking=ranking,
        global_stats=global_stats,
    )

    print("=" * 80)
    print("Descriptive statistics for multi-PM baseline")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Saved target info: {target_info_path}")
    print(f"Saved fold metrics: {fold_metrics_out_path}")
    print(f"Saved aggregated metrics: {agg_metrics_path}")
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved global stats: {global_stats_path}")
    print(f"Saved report: {report_path}")

    print("\nBest targets:")
    print(ranking.to_string(index=False) if not ranking.empty else "empty")


if __name__ == "__main__":
    main()