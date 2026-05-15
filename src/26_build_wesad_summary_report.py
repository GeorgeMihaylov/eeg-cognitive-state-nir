#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build final WESAD summary report.

Purpose:
    Aggregate the completed WESAD wearable stress benchmark stages into one
    compact scientific/engineering report.

Expected previous stages:
    16_inspect_wesad_dataset.py
    17_prepare_wesad_windowed_dataset.py
    18_train_wesad_stress_baseline.py
    19_analyze_wesad_stress_results.py
    20_train_wesad_feature_group_ablation.py
    21_train_wesad_protocol_control.py

Default inputs are detected automatically:
    reports/wearable_pm_alignment/wesad_windowed_stress_dataset_report.md
    latest reports/wearable_pm_alignment/runs/*wesad_stress_full*
    latest reports/wearable_pm_alignment/runs/*wesad_feature_group_ablation*
    latest reports/wearable_pm_alignment/runs/*wesad_protocol_control*

Outputs:
    reports/wearable_pm_alignment/wesad_final_summary/
        wesad_final_summary.md
        wesad_key_metrics.csv
        wesad_baseline_summary.csv
        wesad_threshold_summary.csv
        wesad_feature_group_summary.csv
        wesad_protocol_control_summary.csv
        source_files.json
        figures/
            wesad_baseline_balanced_accuracy.png
            wesad_feature_group_balanced_accuracy.png
            wesad_protocol_control_balanced_accuracy.png
            wesad_protocol_control_best_threshold.png
            wesad_key_takeaways.png

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\26_build_wesad_summary_report.py

Explicit command:
    D:\miniconda3\envs\eeg_nir\python.exe src\26_build_wesad_summary_report.py `
      --baseline-run reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full `
      --ablation-run reports\wearable_pm_alignment\runs\20260514_144608_wesad_feature_group_ablation `
      --protocol-run reports\wearable_pm_alignment\runs\20260514_160829_wesad_protocol_control
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_REPORT_ROOT = "reports/wearable_pm_alignment"
DEFAULT_OUTPUT_DIR = "reports/wearable_pm_alignment/wesad_final_summary"


def resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def read_csv_if_exists(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_run(runs_root: Path, pattern: str, required_files: List[str]) -> Optional[Path]:
    if not runs_root.exists():
        return None

    candidates = []
    for p in runs_root.glob(pattern):
        if not p.is_dir():
            continue
        ok = all((p / f).exists() for f in required_files)
        if ok:
            candidates.append(p)

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def safe_rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def copy_existing_figures(src_run: Optional[Path], dst_dir: Path, prefix: str) -> List[Path]:
    copied: List[Path] = []

    if src_run is None:
        return copied

    src_fig_dir = src_run / "figures"
    if not src_fig_dir.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_fig_dir.glob("*.png")):
        dst = dst_dir / f"{prefix}_{src.name}"
        try:
            shutil.copy2(src, dst)
            copied.append(dst)
        except Exception:
            pass

    analysis_fig_dir = src_run / "analysis" / "figures"
    if analysis_fig_dir.exists():
        for src in sorted(analysis_fig_dir.glob("*.png")):
            dst = dst_dir / f"{prefix}_analysis_{src.name}"
            try:
                shutil.copy2(src, dst)
                copied.append(dst)
            except Exception:
                pass

    return copied


def normalize_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "stage", "baseline")
    out.insert(1, "experiment_family", "wesad_stress_baseline")
    return out


def normalize_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "stage", "threshold_analysis")
    return out


def normalize_feature_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "stage", "feature_group_ablation")
    return out


def normalize_protocol_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "stage", "protocol_control")
    return out


def build_key_metrics(
    baseline_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    protocol_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_row(name: str, value: object, comment: str = "") -> None:
        rows.append({"metric": name, "value": value, "comment": comment})

    if not baseline_summary.empty:
        best = baseline_summary.sort_values(
            ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"],
            ascending=False,
        ).iloc[0]
        add_row("best_baseline_model", best.get("model", ""), "Best default-threshold model in WESAD stress baseline.")
        add_row("best_baseline_balanced_accuracy", round(float(best.get("balanced_accuracy_mean", np.nan)), 6), "")
        add_row("best_baseline_macro_f1", round(float(best.get("macro_f1_mean", np.nan)), 6), "")
        add_row("best_baseline_roc_auc", round(float(best.get("roc_auc_mean", np.nan)), 6), "")
        add_row("best_baseline_average_precision", round(float(best.get("average_precision_mean", np.nan)), 6), "")

    if not threshold_summary.empty:
        best = threshold_summary.sort_values("best_balanced_accuracy", ascending=False).iloc[0]
        add_row(
            "best_threshold_optimized_model",
            str(best.get("model", "")),
            "Best OOF threshold-optimized model from stress baseline analysis.",
        )
        add_row("best_threshold_optimized_threshold", round(float(best.get("best_threshold_by_balanced_accuracy", np.nan)), 6), "")
        add_row("best_threshold_optimized_balanced_accuracy", round(float(best.get("best_balanced_accuracy", np.nan)), 6), "")

    if not ablation_summary.empty:
        best = ablation_summary.sort_values(
            ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"],
            ascending=False,
        ).iloc[0]
        add_row(
            "best_feature_group_default",
            f"{best.get('feature_group', '')} / {best.get('model', '')}",
            "Best default-threshold feature group ablation result.",
        )
        add_row("best_feature_group_default_balanced_accuracy", round(float(best.get("balanced_accuracy_mean", np.nan)), 6), "")

    if not protocol_summary.empty:
        best = protocol_summary.sort_values(
            ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"],
            ascending=False,
        ).iloc[0]
        add_row(
            "best_protocol_default",
            f"{best.get('feature_group', '')} / {best.get('model', '')}",
            "Best default-threshold protocol-control result.",
        )
        add_row("best_protocol_default_balanced_accuracy", round(float(best.get("balanced_accuracy_mean", np.nan)), 6), "")

    if not protocol_comparison.empty:
        # Prefer logistic_robust row if present because it was the most stable baseline.
        if "model" in protocol_comparison.columns and "logistic_robust" in set(protocol_comparison["model"]):
            row = protocol_comparison[protocol_comparison["model"] == "logistic_robust"].iloc[0]
        else:
            row = protocol_comparison.iloc[0]

        for col in [
            "delta_all_minus_no_acc_balanced_accuracy",
            "delta_no_acc_minus_acc_only_balanced_accuracy",
            "delta_bvp_only_minus_acc_only_balanced_accuracy",
            "delta_all_minus_no_acc_best_balanced_accuracy",
            "delta_no_acc_minus_acc_only_best_balanced_accuracy",
        ]:
            if col in row.index:
                add_row(col, round(float(row[col]), 6), f"Protocol-control delta for model={row.get('model', '')}.")

    return pd.DataFrame(rows)


def save_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    label_cols: List[str],
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> Optional[Path]:
    if df.empty or value_col not in df.columns:
        return None

    plot_df = df.copy()
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_n)
    plot_df["label"] = plot_df[label_cols].astype(str).agg(" / ".join, axis=1)
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)))
    ax.barh(plot_df["label"], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_protocol_threshold_plot(best_th: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if best_th.empty or "best_balanced_accuracy" not in best_th.columns:
        return None

    df = best_th.copy()
    cols = ["feature_group", "model"]
    if not set(cols).issubset(df.columns):
        return None

    df["label"] = df[cols].astype(str).agg(" / ".join, axis=1)
    df = df.sort_values("best_balanced_accuracy", ascending=False).head(20)
    df = df.sort_values("best_balanced_accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    ax.barh(df["label"], df["best_balanced_accuracy"])
    ax.set_title("WESAD protocol control: threshold-optimized balanced accuracy")
    ax.set_xlabel("best_balanced_accuracy")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_key_takeaways_plot(key_metrics: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if key_metrics.empty:
        return None

    selected = key_metrics[
        key_metrics["metric"].isin(
            [
                "best_baseline_balanced_accuracy",
                "best_threshold_optimized_balanced_accuracy",
                "best_feature_group_default_balanced_accuracy",
                "best_protocol_default_balanced_accuracy",
            ]
        )
    ].copy()

    if selected.empty:
        return None

    selected["value_num"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.dropna(subset=["value_num"])

    if selected.empty:
        return None

    selected = selected.sort_values("value_num", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(selected["metric"], selected["value_num"])
    ax.set_title("WESAD key balanced-accuracy milestones")
    ax.set_xlabel("balanced accuracy")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "No data."
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def build_markdown_report(
    root: Path,
    output_dir: Path,
    source_files: Dict[str, str],
    dataset_report_text: str,
    key_metrics: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    protocol_comparison: pd.DataFrame,
    figure_paths: List[Path],
) -> str:
    lines: List[str] = []

    lines.append("# WESAD wearable stress benchmark: final summary")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report summarizes the external wearable benchmark based on WESAD. "
        "The goal was to test whether wrist physiological signals can recover a stress-like state "
        "under subject-aware validation and to identify whether the result is physiological or partly driven by movement/protocol confounding."
    )
    lines.append("")
    lines.append("The benchmark is not a direct Emotiv PM prediction experiment. It is an external validation line for the broader hypothesis:")
    lines.append("")
    lines.append("```text")
    lines.append("wearable physiology can provide useful proxy signals for stress/arousal-related cognitive-state estimation")
    lines.append("```")
    lines.append("")
    lines.append("## 2. Source files")
    lines.append("")
    src_rows = [{"name": k, "path": v} for k, v in source_files.items()]
    lines.append(pd.DataFrame(src_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Dataset preparation status")
    lines.append("")
    if dataset_report_text:
        # Keep only the first useful part; avoid duplicating a huge report.
        keep = dataset_report_text[:5000]
        lines.append("Excerpt from the windowed-dataset report:")
        lines.append("")
        lines.append("```markdown")
        lines.append(keep)
        lines.append("```")
    else:
        lines.append(
            "Dataset preparation report was not found. Expected file: "
            "`reports/wearable_pm_alignment/wesad_windowed_stress_dataset_report.md`."
        )
    lines.append("")
    lines.append("## 4. Key metrics")
    lines.append("")
    lines.append(df_to_md(key_metrics, max_rows=50))
    lines.append("")
    lines.append("## 5. Baseline stress classification")
    lines.append("")
    lines.append(
        "The baseline stage trained standard classifiers on the 60s/10s WESAD windowed dataset using subject-aware validation. "
        "Primary metrics are balanced accuracy, macro-F1, ROC-AUC, average precision and stress-class F1."
    )
    lines.append("")
    show_cols = [
        c for c in [
            "model",
            "folds",
            "n_val_total",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "roc_auc_mean",
            "average_precision_mean",
            "f1_stress_mean",
            "recall_stress_mean",
            "precision_stress_mean",
        ]
        if c in baseline_summary.columns
    ]
    lines.append(df_to_md(baseline_summary[show_cols] if show_cols else baseline_summary))
    lines.append("")
    lines.append("## 6. Threshold analysis")
    lines.append("")
    lines.append(
        "Threshold analysis was used because tree-based models often had strong ranking quality "
        "but weaker default-threshold metrics. The threshold was selected on out-of-fold predictions for diagnostic comparison."
    )
    lines.append("")
    show_cols = [
        c for c in [
            "model",
            "best_threshold_by_balanced_accuracy",
            "best_balanced_accuracy",
            "at_balanced_accuracy_precision_stress",
            "at_balanced_accuracy_recall_stress",
            "best_threshold_by_f1_stress",
            "best_f1_stress",
        ]
        if c in threshold_summary.columns
    ]
    lines.append(df_to_md(threshold_summary[show_cols] if show_cols else threshold_summary))
    lines.append("")
    lines.append("## 7. Feature-group ablation")
    lines.append("")
    lines.append(
        "Feature-group ablation tested whether stress prediction depends on EDA, BVP, TEMP, ACC, or their combinations. "
        "This stage is important because high ACC performance may indicate movement or protocol confounding."
    )
    lines.append("")
    show_cols = [
        c for c in [
            "feature_group",
            "model",
            "n_features",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "roc_auc_mean",
            "average_precision_mean",
            "f1_stress_mean",
            "recall_stress_mean",
            "precision_stress_mean",
        ]
        if c in ablation_summary.columns
    ]
    lines.append(df_to_md(ablation_summary[show_cols] if show_cols else ablation_summary, max_rows=25))
    lines.append("")
    lines.append("## 8. Protocol-control experiment")
    lines.append("")
    lines.append(
        "The protocol-control stage compared `all`, `no_acc`, `acc_only`, `bvp_only`, `eda_only`, `temp_only`, "
        "`eda_bvp`, `eda_bvp_temp`, and `bvp_temp`. "
        "The key question was whether `no_acc` remains strong and whether `acc_only` is suspiciously competitive."
    )
    lines.append("")
    show_cols = [
        c for c in [
            "feature_group",
            "model",
            "n_features",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "roc_auc_mean",
            "average_precision_mean",
            "f1_stress_mean",
            "recall_stress_mean",
            "precision_stress_mean",
        ]
        if c in protocol_summary.columns
    ]
    lines.append(df_to_md(protocol_summary[show_cols] if show_cols else protocol_summary, max_rows=25))
    lines.append("")
    lines.append("### Protocol-control deltas")
    lines.append("")
    lines.append(df_to_md(protocol_comparison, max_rows=20))
    lines.append("")
    lines.append("## 9. Interpretation")
    lines.append("")
    lines.append("Main conclusions:")
    lines.append("")
    lines.append("1. **Wearable stress detection is feasible on WESAD under subject-aware validation.**")
    lines.append("2. **BVP/PPG and BVP+TEMP are the strongest compact physiological proxies in the current feature set.**")
    lines.append("3. **The `no_acc` group remains strong**, so the result is not entirely explained by accelerometer signals.")
    lines.append("4. **`ACC-only` is also strong**, which means WESAD contains movement/protocol information. ACC should be treated as a control channel, not as a pure physiological stress marker.")
    lines.append("5. **The full feature set is not necessarily best**, likely because additional channels add subject-specific noise and protocol dependencies.")
    lines.append("")
    lines.append("Recommended wording for the project:")
    lines.append("")
    lines.append("```text")
    lines.append(
        "On WESAD, wrist physiological signals predict stress-like state with good subject-aware performance. "
        "The strongest compact physiological proxy is based on BVP/PPG and temperature, while EDA alone is weaker with the current simple statistical features. "
        "However, high ACC-only performance indicates protocol/movement confounding. Therefore, ACC should be used as a control/artifact channel, "
        "and PM.Stress wearable alignment should primarily rely on BVP/PPG, EDA and TEMP."
    )
    lines.append("```")
    lines.append("")
    lines.append("## 10. Figures")
    lines.append("")
    if figure_paths:
        for path in figure_paths:
            try:
                rel = path.relative_to(output_dir)
            except Exception:
                rel = path
            lines.append(f"- `{rel}`")
    else:
        lines.append("No figures generated.")
    lines.append("")
    lines.append("## 11. Next steps")
    lines.append("")
    lines.append("Recommended next work items:")
    lines.append("")
    lines.append("1. Move back to the main EEG/PM task and run subject-calibration experiments.")
    lines.append("2. Add WESAD result summary to README/roadmap.")
    lines.append("3. Treat COLET as blocked until MATLAB conversion is available.")
    lines.append("4. Optionally improve WESAD EDA features with SCL/SCR decomposition and lag-aware features.")
    lines.append("5. Optionally run WESAD window-size ablation: 30s, 60s, 120s.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final WESAD benchmark summary report.")

    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--report-root", type=str, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--runs-root", type=str, default="reports/wearable_pm_alignment/runs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--dataset-report", type=str, default=None)
    parser.add_argument("--baseline-run", type=str, default=None)
    parser.add_argument("--ablation-run", type=str, default=None)
    parser.add_argument("--protocol-run", type=str, default=None)

    parser.add_argument("--copy-run-figures", type=str, default="true", choices=["true", "false"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    report_root = resolve_path(root, args.report_root)
    runs_root = resolve_path(root, args.runs_root)
    output_dir = resolve_path(root, args.output_dir)
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_report_path = resolve_path(root, args.dataset_report) if args.dataset_report else report_root / "wesad_windowed_stress_dataset_report.md"

    baseline_run = resolve_path(root, args.baseline_run) if args.baseline_run else latest_run(
        runs_root,
        "*wesad_stress_full*",
        ["metrics_summary.csv"],
    )
    ablation_run = resolve_path(root, args.ablation_run) if args.ablation_run else latest_run(
        runs_root,
        "*wesad_feature_group_ablation*",
        ["ablation_summary.csv"],
    )
    protocol_run = resolve_path(root, args.protocol_run) if args.protocol_run else latest_run(
        runs_root,
        "*wesad_protocol_control*",
        ["protocol_control_summary.csv", "protocol_control_comparison.csv"],
    )

    baseline_summary_path = baseline_run / "metrics_summary.csv" if baseline_run else None
    threshold_summary_path = baseline_run / "analysis" / "best_thresholds.csv" if baseline_run else None
    baseline_analysis_summary_path = baseline_run / "analysis" / "per_subject_metrics_summary.csv" if baseline_run else None

    ablation_summary_path = ablation_run / "ablation_summary.csv" if ablation_run else None
    ablation_best_path = ablation_run / "best_by_feature_group.csv" if ablation_run else None

    protocol_summary_path = protocol_run / "protocol_control_summary.csv" if protocol_run else None
    protocol_threshold_path = protocol_run / "best_thresholds.csv" if protocol_run else None
    protocol_comparison_path = protocol_run / "protocol_control_comparison.csv" if protocol_run else None

    baseline_summary = normalize_baseline_summary(read_csv_if_exists(baseline_summary_path))
    threshold_summary = normalize_threshold_summary(read_csv_if_exists(threshold_summary_path))
    baseline_analysis_summary = read_csv_if_exists(baseline_analysis_summary_path)

    ablation_summary = normalize_feature_group_summary(read_csv_if_exists(ablation_summary_path))
    ablation_best = read_csv_if_exists(ablation_best_path)

    protocol_summary = normalize_protocol_summary(read_csv_if_exists(protocol_summary_path))
    protocol_threshold = read_csv_if_exists(protocol_threshold_path)
    protocol_comparison = read_csv_if_exists(protocol_comparison_path)

    dataset_report_text = ""
    if dataset_report_path and dataset_report_path.exists():
        dataset_report_text = dataset_report_path.read_text(encoding="utf-8", errors="replace")

    key_metrics = build_key_metrics(
        baseline_summary=baseline_summary,
        threshold_summary=threshold_summary,
        ablation_summary=ablation_summary,
        protocol_summary=protocol_summary,
        protocol_comparison=protocol_comparison,
    )

    # Save normalized summary tables.
    baseline_out = output_dir / "wesad_baseline_summary.csv"
    threshold_out = output_dir / "wesad_threshold_summary.csv"
    ablation_out = output_dir / "wesad_feature_group_summary.csv"
    protocol_out = output_dir / "wesad_protocol_control_summary.csv"
    protocol_cmp_out = output_dir / "wesad_protocol_conclusions.csv"
    key_metrics_out = output_dir / "wesad_key_metrics.csv"
    baseline_analysis_out = output_dir / "wesad_per_subject_summary.csv"
    ablation_best_out = output_dir / "wesad_best_by_feature_group.csv"

    baseline_summary.to_csv(baseline_out, index=False, encoding="utf-8")
    threshold_summary.to_csv(threshold_out, index=False, encoding="utf-8")
    ablation_summary.to_csv(ablation_out, index=False, encoding="utf-8")
    protocol_summary.to_csv(protocol_out, index=False, encoding="utf-8")
    protocol_comparison.to_csv(protocol_cmp_out, index=False, encoding="utf-8")
    key_metrics.to_csv(key_metrics_out, index=False, encoding="utf-8")

    if not baseline_analysis_summary.empty:
        baseline_analysis_summary.to_csv(baseline_analysis_out, index=False, encoding="utf-8")
    if not ablation_best.empty:
        ablation_best.to_csv(ablation_best_out, index=False, encoding="utf-8")

    # Generate summary figures.
    figure_paths: List[Path] = []

    p = save_bar_plot(
        baseline_summary,
        value_col="balanced_accuracy_mean",
        label_cols=["model"],
        title="WESAD baseline: balanced accuracy by model",
        output_path=figures_dir / "wesad_baseline_balanced_accuracy.png",
    )
    if p:
        figure_paths.append(p)

    p = save_bar_plot(
        ablation_summary,
        value_col="balanced_accuracy_mean",
        label_cols=["feature_group", "model"],
        title="WESAD feature-group ablation: balanced accuracy",
        output_path=figures_dir / "wesad_feature_group_balanced_accuracy.png",
    )
    if p:
        figure_paths.append(p)

    p = save_bar_plot(
        protocol_summary,
        value_col="balanced_accuracy_mean",
        label_cols=["feature_group", "model"],
        title="WESAD protocol control: balanced accuracy",
        output_path=figures_dir / "wesad_protocol_control_balanced_accuracy.png",
    )
    if p:
        figure_paths.append(p)

    p = save_protocol_threshold_plot(
        protocol_threshold,
        output_path=figures_dir / "wesad_protocol_control_best_threshold.png",
    )
    if p:
        figure_paths.append(p)

    p = save_key_takeaways_plot(
        key_metrics,
        output_path=figures_dir / "wesad_key_takeaways.png",
    )
    if p:
        figure_paths.append(p)

    if args.copy_run_figures == "true":
        figure_paths.extend(copy_existing_figures(baseline_run, figures_dir, "baseline"))
        figure_paths.extend(copy_existing_figures(ablation_run, figures_dir, "ablation"))
        figure_paths.extend(copy_existing_figures(protocol_run, figures_dir, "protocol"))

    source_files = {
        "dataset_report": str(dataset_report_path) if dataset_report_path else "",
        "baseline_run": str(baseline_run) if baseline_run else "",
        "baseline_summary": str(baseline_summary_path) if baseline_summary_path else "",
        "threshold_summary": str(threshold_summary_path) if threshold_summary_path else "",
        "baseline_per_subject_summary": str(baseline_analysis_summary_path) if baseline_analysis_summary_path else "",
        "ablation_run": str(ablation_run) if ablation_run else "",
        "ablation_summary": str(ablation_summary_path) if ablation_summary_path else "",
        "ablation_best_by_group": str(ablation_best_path) if ablation_best_path else "",
        "protocol_run": str(protocol_run) if protocol_run else "",
        "protocol_summary": str(protocol_summary_path) if protocol_summary_path else "",
        "protocol_threshold": str(protocol_threshold_path) if protocol_threshold_path else "",
        "protocol_comparison": str(protocol_comparison_path) if protocol_comparison_path else "",
        "output_dir": str(output_dir),
    }

    source_json_path = output_dir / "source_files.json"
    source_json_path.write_text(json.dumps(source_files, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = build_markdown_report(
        root=root,
        output_dir=output_dir,
        source_files=source_files,
        dataset_report_text=dataset_report_text,
        key_metrics=key_metrics,
        baseline_summary=baseline_summary,
        threshold_summary=threshold_summary,
        ablation_summary=ablation_summary,
        protocol_summary=protocol_summary,
        protocol_comparison=protocol_comparison,
        figure_paths=figure_paths,
    )

    report_path = output_dir / "wesad_final_summary.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("=" * 80)
    print("WESAD final summary report built")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Key metrics: {key_metrics_out}")
    print(f"Baseline summary: {baseline_out}")
    print(f"Threshold summary: {threshold_out}")
    print(f"Feature-group summary: {ablation_out}")
    print(f"Protocol summary: {protocol_out}")
    print(f"Protocol conclusions: {protocol_cmp_out}")
    print(f"Source files: {source_json_path}")
    print("")
    print("Detected source runs:")
    print(f"  baseline_run: {baseline_run}")
    print(f"  ablation_run: {ablation_run}")
    print(f"  protocol_run: {protocol_run}")
    print("")
    print("Key metrics:")
    if not key_metrics.empty:
        print(key_metrics.to_string(index=False))
    else:
        print("No key metrics were produced. Check input run paths.")
    print("")
    print("Figures:")
    for p in figure_paths[:50]:
        print(f"  {p}")
    if len(figure_paths) > 50:
        print(f"  ... and {len(figure_paths) - 50} more")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
