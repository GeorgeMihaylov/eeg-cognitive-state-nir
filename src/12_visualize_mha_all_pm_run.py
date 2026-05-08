# -*- coding: utf-8 -*-
"""
12_visualize_mha_all_pm_run.py

Комплексная визуализация последнего MHA all-PM прогона.

Скрипт строит:
1. Рейтинг PM-таргетов по R2, Spearman, RMSE, MAE.
2. Errorbar-графики по folds.
3. Heatmap метрик target x metric.
4. Fold-level heatmaps.
5. Scatter y_true vs y_pred для каждого PM.
6. Residual plots.
7. Residual histograms.
8. Calibration / binned prediction plots.
9. Распределения y_true по target.
10. Loss curves по target/fold.
11. Сравнительный dashboard.
12. Markdown-отчет со списком всех графиков.

Пример запуска:

D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\12_visualize_mha_all_pm_run.py ^
  --run-dir reports\\runs\\20260508_154633_mha_all_pm_full_all_pow_plus_eeg_len3

Опционально можно передать tabular baseline summary для сравнения:

D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\12_visualize_mha_all_pm_run.py ^
  --run-dir reports\\runs\\20260508_154633_mha_all_pm_full_all_pow_plus_eeg_len3 ^
  --baseline-summary reports\\runs\\<tabular_run>\\target_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_ORDER_DEFAULT = [
    "excitement",
    "relaxation",
    "stress",
    "engagement",
    "interest",
    "focus",
    "attention",
]

METRICS_MAIN = ["r2_mean", "spearman_mean", "pearson_mean", "rmse_mean", "mae_mean"]
METRICS_STD = ["r2_std", "spearman_std", "pearson_std", "rmse_std", "mae_std"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_optional_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Иногда target может называться target_name.
    if "target_name" in out.columns and "target" not in out.columns:
        out = out.rename(columns={"target_name": "target"})

    # Если metrics без _mean, приводим к _mean.
    for col in ["mae", "rmse", "r2", "pearson", "spearman"]:
        if col in out.columns and f"{col}_mean" not in out.columns:
            out = out.rename(columns={col: f"{col}_mean"})

    return out


def target_order_from_summary(summary: pd.DataFrame) -> List[str]:
    if "target" not in summary.columns:
        return TARGET_ORDER_DEFAULT

    targets = summary["target"].dropna().astype(str).tolist()

    if "rank" in summary.columns:
        ranked = (
            summary.sort_values("rank")["target"]
            .dropna()
            .astype(str)
            .tolist()
        )
        return ranked

    known = [t for t in TARGET_ORDER_DEFAULT if t in targets]
    rest = [t for t in targets if t not in known]
    return known + rest


def apply_target_order(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    if df.empty or "target" not in df.columns:
        return df

    out = df.copy()
    out["target"] = pd.Categorical(out["target"], categories=order, ordered=True)
    out = out.sort_values("target").reset_index(drop=True)
    out["target"] = out["target"].astype(str)
    return out


def load_run_data(run_dir: Path) -> Dict[str, pd.DataFrame]:
    summary = read_csv(run_dir / "all_targets_summary.csv")
    summary = normalize_summary_columns(summary)

    agg = read_optional_csv(run_dir / "all_targets_aggregated_metrics.csv")
    agg = normalize_summary_columns(agg) if not agg.empty else agg

    fold = read_optional_csv(run_dir / "all_targets_fold_metrics.csv")
    if not fold.empty and "target_name" in fold.columns and "target" not in fold.columns:
        fold = fold.rename(columns={"target_name": "target"})

    return {
        "summary": summary,
        "agg": agg,
        "fold": fold,
    }


def load_target_tables(run_dir: Path, target: str) -> Dict[str, pd.DataFrame]:
    target_dir = run_dir / "targets" / target

    return {
        "sequence": read_optional_csv(target_dir / "sequence_metadata.csv"),
        "fold_metrics": read_optional_csv(target_dir / "fold_metrics.csv"),
        "history": read_optional_csv(target_dir / "epoch_history.csv"),
        "predictions": read_optional_parquet(target_dir / "predictions.parquet"),
    }


def plot_metric_bars(summary: pd.DataFrame, out_dir: Path, order: List[str]) -> List[Path]:
    paths = []
    df = apply_target_order(summary, order)

    metric_specs = [
        ("r2_mean", "r2_std", "R2 mean", "r2_bar.png"),
        ("spearman_mean", "spearman_std", "Spearman mean", "spearman_bar.png"),
        ("pearson_mean", "pearson_std", "Pearson mean", "pearson_bar.png"),
        ("rmse_mean", "rmse_std", "RMSE mean", "rmse_bar.png"),
        ("mae_mean", "mae_std", "MAE mean", "mae_bar.png"),
    ]

    for metric, std_col, title, filename in metric_specs:
        if metric not in df.columns:
            continue

        path = out_dir / filename

        x = np.arange(len(df))
        y = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        yerr = (
            pd.to_numeric(df[std_col], errors="coerce").to_numpy()
            if std_col in df.columns
            else None
        )

        plt.figure(figsize=(10, 5))
        plt.bar(x, y, yerr=yerr, capsize=4)
        plt.xticks(x, df["target"].astype(str), rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)

        save_fig(path)
        paths.append(path)

    return paths


def plot_metric_heatmap(summary: pd.DataFrame, out_dir: Path, order: List[str]) -> Path:
    df = apply_target_order(summary, order)

    metrics = [m for m in METRICS_MAIN if m in df.columns]
    values = df[metrics].apply(pd.to_numeric, errors="coerce").to_numpy()

    path = out_dir / "target_metric_heatmap.png"

    plt.figure(figsize=(10, 6))
    im = plt.imshow(values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(np.arange(len(metrics)), metrics, rotation=30, ha="right")
    plt.yticks(np.arange(len(df)), df["target"].astype(str))
    plt.title("Target x metric heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)

    save_fig(path)
    return path


def plot_fold_heatmaps(fold_df: pd.DataFrame, out_dir: Path, order: List[str]) -> List[Path]:
    paths = []

    if fold_df.empty:
        return paths

    if "target" not in fold_df.columns or "fold" not in fold_df.columns:
        return paths

    metrics = ["r2", "spearman", "pearson", "rmse", "mae"]

    for metric in metrics:
        if metric not in fold_df.columns:
            continue

        pivot = fold_df.pivot_table(
            index="target",
            columns="fold",
            values=metric,
            aggfunc="mean",
        )

        pivot = pivot.reindex([t for t in order if t in pivot.index])

        path = out_dir / f"fold_heatmap_{metric}.png"

        values = pivot.to_numpy(dtype=float)

        plt.figure(figsize=(8, 5))
        im = plt.imshow(values, aspect="auto")
        plt.colorbar(im)
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("fold")
        plt.ylabel("target")
        plt.title(f"Fold-level {metric}")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)

        save_fig(path)
        paths.append(path)

    return paths


def plot_metric_boxplots(fold_df: pd.DataFrame, out_dir: Path, order: List[str]) -> List[Path]:
    paths = []

    if fold_df.empty:
        return paths

    metrics = ["r2", "spearman", "pearson", "rmse", "mae"]

    for metric in metrics:
        if metric not in fold_df.columns:
            continue

        data = []
        labels = []

        for target in order:
            values = fold_df.loc[fold_df["target"] == target, metric].dropna().to_numpy()
            if len(values) > 0:
                data.append(values)
                labels.append(target)

        if not data:
            continue

        path = out_dir / f"fold_boxplot_{metric}.png"

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"Fold distribution: {metric}")
        plt.grid(axis="y", alpha=0.3)

        save_fig(path)
        paths.append(path)

    return paths


def plot_scatter_predictions(pred: pd.DataFrame, out_dir: Path, target: str, max_points: int) -> Optional[Path]:
    if pred.empty or "y_true" not in pred.columns or "y_pred" not in pred.columns:
        return None

    df = pred.copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    path = out_dir / f"{target}_scatter_y_true_y_pred.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(df["y_true"], df["y_pred"], s=6, alpha=0.35)

    vmin = min(df["y_true"].min(), df["y_pred"].min())
    vmax = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"{target}: y_true vs y_pred")
    plt.grid(alpha=0.3)

    save_fig(path)
    return path


def plot_residuals(pred: pd.DataFrame, out_dir: Path, target: str, max_points: int) -> List[Path]:
    paths = []

    if pred.empty or "y_true" not in pred.columns or "y_pred" not in pred.columns:
        return paths

    df = pred.copy()
    df["residual"] = df["y_pred"] - df["y_true"]

    if len(df) > max_points:
        df_sample = df.sample(max_points, random_state=42)
    else:
        df_sample = df

    # Residual vs true
    path1 = out_dir / f"{target}_residual_vs_true.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(df_sample["y_true"], df_sample["residual"], s=6, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("y_true")
    plt.ylabel("y_pred - y_true")
    plt.title(f"{target}: residual vs true")
    plt.grid(alpha=0.3)
    save_fig(path1)
    paths.append(path1)

    # Residual histogram
    path2 = out_dir / f"{target}_residual_hist.png"
    plt.figure(figsize=(7, 5))
    plt.hist(df["residual"].dropna(), bins=50)
    plt.xlabel("residual")
    plt.ylabel("count")
    plt.title(f"{target}: residual distribution")
    plt.grid(axis="y", alpha=0.3)
    save_fig(path2)
    paths.append(path2)

    # Residual by fold
    if "fold" in df.columns:
        data = []
        labels = []

        for fold in sorted(df["fold"].dropna().unique()):
            values = df.loc[df["fold"] == fold, "residual"].dropna().to_numpy()
            if len(values) > 0:
                data.append(values)
                labels.append(str(fold))

        if data:
            path3 = out_dir / f"{target}_residual_by_fold.png"
            plt.figure(figsize=(8, 5))
            plt.boxplot(data, labels=labels, showmeans=True)
            plt.axhline(0, linestyle="--")
            plt.xlabel("fold")
            plt.ylabel("residual")
            plt.title(f"{target}: residual by fold")
            plt.grid(axis="y", alpha=0.3)
            save_fig(path3)
            paths.append(path3)

    return paths


def plot_calibration(pred: pd.DataFrame, out_dir: Path, target: str, bins: int) -> Optional[Path]:
    if pred.empty or "y_true" not in pred.columns or "y_pred" not in pred.columns:
        return None

    df = pred[["y_true", "y_pred"]].dropna().copy()

    if df.empty:
        return None

    try:
        df["bin"] = pd.qcut(df["y_pred"], q=bins, duplicates="drop")
    except ValueError:
        return None

    grouped = (
        df.groupby("bin", observed=False)
        .agg(
            pred_mean=("y_pred", "mean"),
            true_mean=("y_true", "mean"),
            true_std=("y_true", "std"),
            count=("y_true", "size"),
        )
        .reset_index(drop=True)
    )

    path = out_dir / f"{target}_calibration_binned.png"

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        grouped["pred_mean"],
        grouped["true_mean"],
        yerr=grouped["true_std"],
        fmt="o",
        capsize=3,
    )

    vmin = min(grouped["pred_mean"].min(), grouped["true_mean"].min())
    vmax = max(grouped["pred_mean"].max(), grouped["true_mean"].max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    plt.xlabel("mean predicted value per bin")
    plt.ylabel("mean true value per bin")
    plt.title(f"{target}: binned calibration")
    plt.grid(alpha=0.3)

    save_fig(path)
    return path


def plot_target_distribution(seq: pd.DataFrame, pred: pd.DataFrame, out_dir: Path, target: str) -> Optional[Path]:
    values = None

    if not seq.empty and "target" in seq.columns:
        values = seq["target"].dropna()
    elif not pred.empty and "y_true" in pred.columns:
        values = pred["y_true"].dropna()

    if values is None or len(values) == 0:
        return None

    path = out_dir / f"{target}_target_distribution.png"

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=50)
    plt.xlabel("target value")
    plt.ylabel("count")
    plt.title(f"{target}: target distribution")
    plt.grid(axis="y", alpha=0.3)

    save_fig(path)
    return path


def plot_loss_curves(history: pd.DataFrame, out_dir: Path, target: str) -> List[Path]:
    paths = []

    if history.empty or "epoch" not in history.columns:
        return paths

    # One figure with all folds
    if "fold" in history.columns:
        path = out_dir / f"{target}_loss_all_folds.png"

        plt.figure(figsize=(9, 5))
        for fold, group in history.groupby("fold"):
            group = group.sort_values("epoch")
            if "val_loss" in group.columns:
                plt.plot(group["epoch"], group["val_loss"], label=f"fold {fold} val")
        plt.xlabel("epoch")
        plt.ylabel("val_loss")
        plt.title(f"{target}: validation loss by fold")
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(path)
        paths.append(path)

    # Train vs val per fold
    if "fold" in history.columns:
        for fold, group in history.groupby("fold"):
            group = group.sort_values("epoch")
            path = out_dir / f"{target}_loss_fold_{fold}.png"

            plt.figure(figsize=(8, 5))
            if "train_loss" in group.columns:
                plt.plot(group["epoch"], group["train_loss"], label="train_loss")
            if "val_loss" in group.columns:
                plt.plot(group["epoch"], group["val_loss"], label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{target}: loss fold {fold}")
            plt.legend()
            plt.grid(alpha=0.3)
            save_fig(path)
            paths.append(path)

    return paths


def plot_dashboard(summary: pd.DataFrame, out_dir: Path, order: List[str]) -> Path:
    df = apply_target_order(summary, order)

    path = out_dir / "dashboard_main_metrics.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    specs = [
        ("r2_mean", "r2_std", "R2"),
        ("spearman_mean", "spearman_std", "Spearman"),
        ("rmse_mean", "rmse_std", "RMSE"),
        ("mae_mean", "mae_std", "MAE"),
    ]

    x = np.arange(len(df))

    for ax, (metric, std_col, title) in zip(axes, specs):
        if metric not in df.columns:
            ax.axis("off")
            continue

        y = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        yerr = (
            pd.to_numeric(df[std_col], errors="coerce").to_numpy()
            if std_col in df.columns
            else None
        )

        ax.bar(x, y, yerr=yerr, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df["target"].astype(str), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("MHA all-PM main metrics", fontsize=14)
    save_fig(path)
    return path


def plot_baseline_comparison(
    mha_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    out_dir: Path,
    order: List[str],
) -> List[Path]:
    paths = []

    if baseline_summary.empty:
        return paths

    baseline = normalize_summary_columns(baseline_summary)
    mha = normalize_summary_columns(mha_summary)

    if "target" not in baseline.columns or "target" not in mha.columns:
        return paths

    merged = pd.merge(
        mha,
        baseline,
        on="target",
        how="inner",
        suffixes=("_mha", "_baseline"),
    )

    if merged.empty:
        return paths

    merged = apply_target_order(merged, order)

    for metric in ["r2_mean", "spearman_mean", "rmse_mean", "mae_mean"]:
        mha_col = f"{metric}_mha"
        base_col = f"{metric}_baseline"

        if mha_col not in merged.columns or base_col not in merged.columns:
            continue

        path = out_dir / f"baseline_comparison_{metric}.png"

        x = np.arange(len(merged))
        width = 0.38

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, merged[base_col], width, label="tabular baseline")
        plt.bar(x + width / 2, merged[mha_col], width, label="MHA")

        plt.xticks(x, merged["target"].astype(str), rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"MHA vs tabular baseline: {metric}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        save_fig(path)
        paths.append(path)

        # Delta plot
        path_delta = out_dir / f"baseline_delta_{metric}.png"
        delta = merged[mha_col] - merged[base_col]

        plt.figure(figsize=(10, 5))
        plt.bar(x, delta)
        plt.axhline(0, linestyle="--")
        plt.xticks(x, merged["target"].astype(str), rotation=30, ha="right")
        plt.ylabel(f"delta {metric}")
        plt.title(f"MHA - tabular baseline: {metric}")
        plt.grid(axis="y", alpha=0.3)

        save_fig(path_delta)
        paths.append(path_delta)

    return paths


def make_markdown_report(
    report_path: Path,
    run_dir: Path,
    figures: List[Path],
    summary: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Visualization report for MHA all-PM run")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    try:
        lines.append(summary.to_markdown(index=False))
    except Exception:
        lines.append(summary.to_string(index=False))
    lines.append("")

    if not fold_df.empty:
        lines.append("## Fold metrics preview")
        lines.append("")
        try:
            lines.append(fold_df.head(30).to_markdown(index=False))
        except Exception:
            lines.append(fold_df.head(30).to_string(index=False))
        lines.append("")

    lines.append("## Generated figures")
    lines.append("")
    for fig in figures:
        try:
            rel = fig.relative_to(report_path.parent)
        except Exception:
            rel = fig
        lines.append(f"- `{rel}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to reports/runs/<mha_all_pm_run_id>",
    )
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default=None,
        help="Optional tabular baseline summary CSV for comparison.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "visualizations"
    global_dir = out_dir / "global"
    per_target_dir = out_dir / "per_target"

    ensure_dir(out_dir)
    ensure_dir(global_dir)
    ensure_dir(per_target_dir)

    data = load_run_data(run_dir)
    summary = data["summary"]
    fold_df = data["fold"]

    order = target_order_from_summary(summary)

    summary = apply_target_order(summary, order)
    if not fold_df.empty:
        fold_df = apply_target_order(fold_df, order)

    figures: List[Path] = []

    # Global plots
    figures.extend(plot_metric_bars(summary, global_dir, order))
    figures.append(plot_metric_heatmap(summary, global_dir, order))
    figures.extend(plot_fold_heatmaps(fold_df, global_dir, order))
    figures.extend(plot_metric_boxplots(fold_df, global_dir, order))
    figures.append(plot_dashboard(summary, global_dir, order))

    # Optional baseline comparison
    if args.baseline_summary:
        baseline_path = Path(args.baseline_summary).resolve()
        baseline_summary = read_csv(baseline_path)
        figures.extend(plot_baseline_comparison(summary, baseline_summary, global_dir, order))

    # Per-target plots
    for target in order:
        target_out_dir = per_target_dir / target
        ensure_dir(target_out_dir)

        tables = load_target_tables(run_dir, target)

        seq = tables["sequence"]
        pred = tables["predictions"]
        history = tables["history"]

        fig = plot_target_distribution(seq, pred, target_out_dir, target)
        if fig is not None:
            figures.append(fig)

        fig = plot_scatter_predictions(pred, target_out_dir, target, args.max_scatter_points)
        if fig is not None:
            figures.append(fig)

        figures.extend(plot_residuals(pred, target_out_dir, target, args.max_scatter_points))

        fig = plot_calibration(pred, target_out_dir, target, args.calibration_bins)
        if fig is not None:
            figures.append(fig)

        figures.extend(plot_loss_curves(history, target_out_dir, target))

    # Save normalized summary copies.
    summary.to_csv(out_dir / "visualization_summary.csv", index=False, encoding="utf-8-sig")
    if not fold_df.empty:
        fold_df.to_csv(out_dir / "visualization_fold_metrics.csv", index=False, encoding="utf-8-sig")

    make_markdown_report(
        report_path=out_dir / "visualization_report.md",
        run_dir=run_dir,
        figures=figures,
        summary=summary,
        fold_df=fold_df,
    )

    print("=" * 80)
    print("MHA all-PM visualization complete")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Figures generated: {len(figures)}")
    print(f"Report: {out_dir / 'visualization_report.md'}")


if __name__ == "__main__":
    main()