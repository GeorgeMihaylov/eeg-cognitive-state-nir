#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze WESAD wearable stress baseline results.

Input run directory:
    reports/wearable_pm_alignment/runs/<timestamp>_wesad_stress_full/

Expected files:
    fold_metrics.csv
    metrics_summary.csv
    predictions.parquet
    feature_importance.csv

Outputs:
    <run_dir>/analysis/
        per_subject_metrics.csv
        per_subject_metrics_summary.csv
        threshold_analysis.csv
        best_thresholds.csv
        feature_importance_top.csv
        fold_instability.csv
        error_cases.csv
        analysis_report.md
        figures/
            per_subject_balanced_accuracy.png
            per_subject_f1_stress.png
            threshold_balanced_accuracy.png
            threshold_f1_stress.png
            top_features_<model>.png
            score_distribution_<model>.png
            subject_error_rate_<model>.png

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\19_analyze_wesad_stress_results.py `
      --run-dir reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full

If --run-dir is not provided, the script uses the latest run under:
    reports/wearable_pm_alignment/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def find_latest_run(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    candidates = [
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "fold_metrics.csv").exists() and (p / "predictions.parquet").exists()
    ]

    if not candidates:
        raise FileNotFoundError(f"No valid WESAD run directories found under: {runs_root}")

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_run_files(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_metrics_path = run_dir / "fold_metrics.csv"
    summary_path = run_dir / "metrics_summary.csv"
    predictions_path = run_dir / "predictions.parquet"
    feature_importance_path = run_dir / "feature_importance.csv"

    missing = [
        str(p) for p in [fold_metrics_path, summary_path, predictions_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required run files: {missing}")

    fold_metrics = pd.read_csv(fold_metrics_path)
    summary = pd.read_csv(summary_path)
    predictions = pd.read_parquet(predictions_path)

    if feature_importance_path.exists():
        feature_importance = pd.read_csv(feature_importance_path)
    else:
        feature_importance = pd.DataFrame(columns=["model", "feature", "importance", "importance_type"])

    return fold_metrics, summary, predictions, feature_importance


def safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "n_stress": int(np.sum(y_true == 1)),
        "n_non_stress": int(np.sum(y_true == 0)),
        "stress_rate": float(np.mean(y_true == 1)) if len(y_true) else np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "precision_stress": np.nan,
        "recall_stress": np.nan,
        "f1_stress": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
    }

    if len(y_true) == 0:
        return out

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["precision_stress"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["recall_stress"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["f1_stress"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = np.nan
        try:
            out["average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["average_precision"] = np.nan

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out["tn"] = int(cm[0, 0])
        out["fp"] = int(cm[0, 1])
        out["fn"] = int(cm[1, 0])
        out["tp"] = int(cm[1, 1])

    return out


def compute_per_subject_metrics(pred_df: pd.DataFrame, threshold_by_model: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    rows = []

    threshold_by_model = threshold_by_model or {}

    for (model, subject_id), g in pred_df.groupby(["model", "subject_id"]):
        y_true = g["y_true"].astype(int).to_numpy()
        y_score = g["y_score"].astype(float).to_numpy()
        threshold = float(threshold_by_model.get(model, 0.5))
        y_pred = (y_score >= threshold).astype(int)

        row = {
            "model": model,
            "subject_id": subject_id,
            "threshold": threshold,
        }
        row.update(safe_binary_metrics(y_true, y_pred, y_score))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "balanced_accuracy", "subject_id"], ascending=[True, True, True])


def summarize_per_subject(per_subject_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_stress",
        "recall_stress",
        "f1_stress",
        "roc_auc",
        "average_precision",
    ]

    rows = []
    for model, g in per_subject_df.groupby("model"):
        row = {
            "model": model,
            "subjects": int(g["subject_id"].nunique()),
            "n_total": int(g["n"].sum()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean"],
        ascending=False,
    )


def threshold_sweep(pred_df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []

    for model, g in pred_df.groupby("model"):
        y_true = g["y_true"].astype(int).to_numpy()
        y_score = g["y_score"].astype(float).to_numpy()

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            metrics = safe_binary_metrics(y_true, y_pred, y_score)
            row = {
                "model": model,
                "threshold": float(threshold),
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_thresholds(threshold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    objectives = ["balanced_accuracy", "macro_f1", "f1_stress"]

    for model, g in threshold_df.groupby("model"):
        row = {"model": model}
        for objective in objectives:
            idx = g[objective].idxmax()
            best = g.loc[idx]
            row[f"best_threshold_by_{objective}"] = float(best["threshold"])
            row[f"best_{objective}"] = float(best[objective])
            row[f"at_{objective}_precision_stress"] = float(best["precision_stress"])
            row[f"at_{objective}_recall_stress"] = float(best["recall_stress"])
            row[f"at_{objective}_balanced_accuracy"] = float(best["balanced_accuracy"])
            row[f"at_{objective}_macro_f1"] = float(best["macro_f1"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("best_balanced_accuracy", ascending=False)


def compute_fold_instability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "f1_stress",
        "recall_stress",
        "precision_stress",
    ]

    rows = []
    for model, g in fold_metrics.groupby("model"):
        row = {"model": model, "folds": int(g["fold"].nunique())}
        for col in metric_cols:
            if col not in g.columns:
                continue
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_range"] = float(g[col].max() - g[col].min())
            row[f"{col}_min_fold"] = int(g.loc[g[col].idxmin(), "fold"])
            row[f"{col}_max_fold"] = int(g.loc[g[col].idxmax(), "fold"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("balanced_accuracy_range", ascending=False)


def build_error_cases(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["error"] = (df["y_true"].astype(int) != df["y_pred"].astype(int)).astype(int)
    df["error_type"] = "correct"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "false_positive"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "false_negative"

    keep_cols = [
        "model", "fold", "subject_id", "window_id", "start_sec", "end_sec",
        "y_true", "y_pred", "y_score", "error", "error_type"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[df["error"] == 1][keep_cols].sort_values(
        ["model", "subject_id", "start_sec" if "start_sec" in df.columns else "window_id"]
    )


def top_feature_importance(feature_importance: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if feature_importance.empty or "importance" not in feature_importance.columns:
        return pd.DataFrame(columns=["model", "feature", "importance", "importance_type", "rank"])

    out_frames = []
    for model, g in feature_importance.groupby("model"):
        gg = g.sort_values("importance", ascending=False).head(top_n).copy()
        gg["rank"] = np.arange(1, len(gg) + 1)
        out_frames.append(gg)

    if not out_frames:
        return pd.DataFrame(columns=["model", "feature", "importance", "importance_type", "rank"])

    return pd.concat(out_frames, ignore_index=True)


def plot_per_subject_metric(per_subject_df: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    if per_subject_df.empty or metric not in per_subject_df.columns:
        return None

    models = per_subject_df["model"].unique().tolist()
    subjects = sorted(per_subject_df["subject_id"].unique().tolist(), key=lambda s: int(str(s)[1:]) if str(s).startswith("S") and str(s)[1:].isdigit() else 10**9)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.55), 5))

    x = np.arange(len(subjects))
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        g = per_subject_df[per_subject_df["model"] == model].set_index("subject_id")
        values = [g.loc[s, metric] if s in g.index else np.nan for s in subjects]
        ax.bar(x + i * width, values, width=width, label=model)

    ax.set_title(f"Per-subject {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("subject_id")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / f"per_subject_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_threshold_metric(threshold_df: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    if threshold_df.empty or metric not in threshold_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for model, g in threshold_df.groupby("model"):
        gg = g.sort_values("threshold")
        ax.plot(gg["threshold"], gg[metric], label=model)

    ax.set_title(f"Threshold sweep: {metric}")
    ax.set_xlabel("threshold")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / f"threshold_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top_features(top_features: pd.DataFrame, figures_dir: Path) -> List[Path]:
    paths: List[Path] = []

    if top_features.empty:
        return paths

    for model, g in top_features.groupby("model"):
        gg = g.sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(5, len(gg) * 0.28)))
        ax.barh(gg["feature"], gg["importance"])
        ax.set_title(f"Top features: {model}")
        ax.set_xlabel("importance")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path = figures_dir / f"top_features_{model}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_score_distribution(pred_df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    paths: List[Path] = []

    for model, g in pred_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, name in [(0, "non-stress"), (1, "stress")]:
            values = g.loc[g["y_true"] == label, "y_score"].astype(float).to_numpy()
            ax.hist(values, bins=40, alpha=0.5, density=True, label=name)

        ax.set_title(f"Score distribution: {model}")
        ax.set_xlabel("predicted stress score")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()

        path = figures_dir / f"score_distribution_{model}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_subject_error_rate(per_subject_df: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if per_subject_df.empty:
        return None

    df = per_subject_df.copy()
    df["error_rate"] = 1.0 - df["accuracy"]

    models = df["model"].unique().tolist()
    subjects = sorted(df["subject_id"].unique().tolist(), key=lambda s: int(str(s)[1:]) if str(s).startswith("S") and str(s)[1:].isdigit() else 10**9)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.55), 5))
    x = np.arange(len(subjects))
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        g = df[df["model"] == model].set_index("subject_id")
        values = [g.loc[s, "error_rate"] if s in g.index else np.nan for s in subjects]
        ax.bar(x + i * width, values, width=width, label=model)

    ax.set_title("Per-subject error rate")
    ax.set_ylabel("error rate")
    ax.set_xlabel("subject_id")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / "subject_error_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_markdown_report(
    run_dir: Path,
    analysis_dir: Path,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    per_subject_summary: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    fold_instability: pd.DataFrame,
    top_features: pd.DataFrame,
    figure_paths: List[Path],
) -> str:
    lines: List[str] = []

    lines.append("# WESAD stress baseline analysis")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Analysis dir: `{analysis_dir}`")
    lines.append("")
    lines.append("## Original model summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-subject metric summary")
    lines.append("")
    lines.append(per_subject_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Best thresholds on out-of-fold predictions")
    lines.append("")
    lines.append(best_thresholds.to_markdown(index=False))
    lines.append("")
    lines.append("## Fold instability")
    lines.append("")
    lines.append(fold_instability.to_markdown(index=False))
    lines.append("")

    if not top_features.empty:
        lines.append("## Top feature importance")
        lines.append("")
        show_cols = ["model", "rank", "feature", "importance", "importance_type"]
        lines.append(top_features[show_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        try:
            rel = path.relative_to(analysis_dir)
        except Exception:
            rel = path
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append("- Compare threshold-optimized metrics with default threshold 0.5.")
    lines.append("- Inspect fold instability: high range indicates subject/domain shift.")
    lines.append("- Inspect per-subject metrics: weak subjects are candidates for calibration analysis.")
    lines.append("- Inspect feature groups: EDA/BVP/TEMP/ACC contribution should be validated with ablation.")
    lines.append("- Next script should implement feature-group ablation or calibration, not only another generic model.")
    lines.append("")

    lines.append("## Recommended next experiments")
    lines.append("")
    lines.append("1. Feature-group ablation: `EDA-only`, `BVP-only`, `TEMP-only`, `ACC-only`, `all`.")
    lines.append("2. Window-size ablation: `30s`, `60s`, `120s`.")
    lines.append("3. Subject calibration: train subjects -> calibration subset from test subject -> evaluation subset.")
    lines.append("4. Threshold tuning inside each train fold, then apply to validation fold.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze WESAD stress baseline results.")

    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory. If omitted, latest valid run is used.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="reports/wearable_pm_alignment/runs",
        help="Runs root for latest-run detection.",
    )
    parser.add_argument("--output-name", type=str, default="analysis", help="Analysis subdirectory name.")
    parser.add_argument("--top-n-features", type=int, default=25)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        help="Minimum threshold for sweep.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
        help="Maximum threshold for sweep.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()

    if args.run_dir:
        run_dir = resolve_path(root, args.run_dir)
    else:
        run_dir = find_latest_run(resolve_path(root, args.runs_root))

    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

    analysis_dir = run_dir / args.output_name
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics, summary, predictions, feature_importance = load_run_files(run_dir)

    required_pred_cols = {"model", "subject_id", "y_true", "y_pred", "y_score"}
    missing = required_pred_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file misses required columns: {missing}")

    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step),
        6,
    )

    print("=" * 80)
    print("Analyze WESAD stress baseline results")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Predictions rows: {len(predictions)}")
    print(f"Models: {sorted(predictions['model'].unique().tolist())}")
    print(f"Subjects: {predictions['subject_id'].nunique()}")
    print(f"Analysis dir: {analysis_dir}")
    print("")

    threshold_df = threshold_sweep(predictions, thresholds)
    best_thresholds = select_best_thresholds(threshold_df)

    threshold_by_model = {
        row["model"]: row["best_threshold_by_balanced_accuracy"]
        for _, row in best_thresholds.iterrows()
    }

    per_subject_default = compute_per_subject_metrics(predictions)
    per_subject_default["threshold_mode"] = "default_0.5"

    per_subject_best = compute_per_subject_metrics(predictions, threshold_by_model=threshold_by_model)
    per_subject_best["threshold_mode"] = "best_oof_balanced_accuracy"

    per_subject_all = pd.concat([per_subject_default, per_subject_best], ignore_index=True)
    per_subject_summary = summarize_per_subject(per_subject_all[per_subject_all["threshold_mode"] == "default_0.5"])

    fold_instability = compute_fold_instability(fold_metrics)
    errors = build_error_cases(predictions)
    top_features = top_feature_importance(feature_importance, args.top_n_features)

    # Save tables.
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_subject_path = analysis_dir / "per_subject_metrics.csv"
    per_subject_summary_path = analysis_dir / "per_subject_metrics_summary.csv"
    threshold_path = analysis_dir / "threshold_analysis.csv"
    best_thresholds_path = analysis_dir / "best_thresholds.csv"
    fold_instability_path = analysis_dir / "fold_instability.csv"
    errors_path = analysis_dir / "error_cases.csv"
    top_features_path = analysis_dir / "feature_importance_top.csv"

    per_subject_all.to_csv(per_subject_path, index=False, encoding="utf-8")
    per_subject_summary.to_csv(per_subject_summary_path, index=False, encoding="utf-8")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8")
    best_thresholds.to_csv(best_thresholds_path, index=False, encoding="utf-8")
    fold_instability.to_csv(fold_instability_path, index=False, encoding="utf-8")
    errors.to_csv(errors_path, index=False, encoding="utf-8")
    top_features.to_csv(top_features_path, index=False, encoding="utf-8")

    # Plots.
    figure_paths: List[Path] = []

    for metric in ["balanced_accuracy", "f1_stress", "roc_auc"]:
        p = plot_per_subject_metric(
            per_subject_default[per_subject_default["threshold_mode"] == "default_0.5"],
            metric,
            figures_dir,
        )
        if p:
            figure_paths.append(p)

    for metric in ["balanced_accuracy", "macro_f1", "f1_stress", "recall_stress", "precision_stress"]:
        p = plot_threshold_metric(threshold_df, metric, figures_dir)
        if p:
            figure_paths.append(p)

    figure_paths.extend(plot_top_features(top_features, figures_dir))
    figure_paths.extend(plot_score_distribution(predictions, figures_dir))

    p = plot_subject_error_rate(
        per_subject_default[per_subject_default["threshold_mode"] == "default_0.5"],
        figures_dir,
    )
    if p:
        figure_paths.append(p)

    report_path = analysis_dir / "analysis_report.md"
    report_path.write_text(
        build_markdown_report(
            run_dir=run_dir,
            analysis_dir=analysis_dir,
            fold_metrics=fold_metrics,
            summary=summary,
            per_subject_summary=per_subject_summary,
            best_thresholds=best_thresholds,
            fold_instability=fold_instability,
            top_features=top_features,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )

    source_files = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "inputs": {
            "fold_metrics": str(run_dir / "fold_metrics.csv"),
            "metrics_summary": str(run_dir / "metrics_summary.csv"),
            "predictions": str(run_dir / "predictions.parquet"),
            "feature_importance": str(run_dir / "feature_importance.csv"),
        },
        "outputs": {
            "per_subject_metrics": str(per_subject_path),
            "per_subject_metrics_summary": str(per_subject_summary_path),
            "threshold_analysis": str(threshold_path),
            "best_thresholds": str(best_thresholds_path),
            "fold_instability": str(fold_instability_path),
            "error_cases": str(errors_path),
            "feature_importance_top": str(top_features_path),
            "analysis_report": str(report_path),
            "figures": [str(p) for p in figure_paths],
        },
    }
    (analysis_dir / "source_files.json").write_text(
        json.dumps(source_files, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Saved WESAD analysis outputs")
    print("=" * 80)
    print(f"Analysis dir: {analysis_dir}")
    print(f"Per-subject metrics: {per_subject_path}")
    print(f"Threshold analysis: {threshold_path}")
    print(f"Best thresholds: {best_thresholds_path}")
    print(f"Fold instability: {fold_instability_path}")
    print(f"Top features: {top_features_path}")
    print(f"Error cases: {errors_path}")
    print(f"Report: {report_path}")
    print("")
    print("Best thresholds:")
    show_cols = [
        "model",
        "best_threshold_by_balanced_accuracy",
        "best_balanced_accuracy",
        "at_balanced_accuracy_precision_stress",
        "at_balanced_accuracy_recall_stress",
        "best_threshold_by_f1_stress",
        "best_f1_stress",
    ]
    show_cols = [c for c in show_cols if c in best_thresholds.columns]
    print(best_thresholds[show_cols].to_string(index=False))
    print("")
    print("Per-subject default-threshold summary:")
    show_cols = [
        "model",
        "subjects",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_min",
        "balanced_accuracy_max",
        "f1_stress_mean",
        "f1_stress_std",
    ]
    show_cols = [c for c in show_cols if c in per_subject_summary.columns]
    print(per_subject_summary[show_cols].to_string(index=False))
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
