#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-subject calibration diagnostics for the final EEG latent-state setup.

This script analyzes whether personal head-only calibration improves quality
for most held-out subjects, not only in the aggregate mean.

Expected input from script 46:
  reports/feature_ablation_v2/calibration_pow_plus_eeg/val_test_per_subject_metrics.csv

Recommended final setup after feature ablation:
  feature_set = pow_plus_eeg
  seq_len = 8
  targets = slow_pca_1,slow_pca_2,slow_pca_3
  calibration_lr = 0.0001
  calibration_frac = 0.20

Example:
  D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\47_analyze_per_subject_calibration_diagnostics.py `
    --root . `
    --input reports\\feature_ablation_v2\\calibration_pow_plus_eeg\\val_test_per_subject_metrics.csv `
    --output-dir reports\\feature_ablation_v2\\subject_diagnostics_pow_plus_eeg `
    --calibration-lr 0.0001 `
    --calibration-frac 0.20 `
    --targets slow_pca_1,slow_pca_2,slow_pca_3 `
    --eval-splits val,test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required for diagnostic plots.") from exc


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"


@dataclass
class DiagnosticsConfig:
    root: str
    input: str
    output_dir: str
    calibration_lr: float
    calibration_frac: float
    targets: list[str]
    eval_splits: list[str]
    seq_len: int | None
    min_subjects_for_summary: int


def parse_csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated strings")
    return out


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def filter_float(df: pd.DataFrame, col: str, value: float, atol: float = 1e-12) -> pd.DataFrame:
    values = pd.to_numeric(df[col], errors="coerce")
    return df[np.isclose(values.to_numpy(dtype=float), float(value), atol=atol, rtol=0.0)].copy()


def prepare_paired_gains(df: pd.DataFrame, cfg: DiagnosticsConfig) -> pd.DataFrame:
    require_columns(
        df,
        [
            "eval_split",
            "seq_len",
            "subject_id",
            "target",
            "calibration_lr",
            "calibration_frac",
            "r2",
            "spearman",
            "mae",
            "rmse",
        ],
    )

    work = df.copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work["eval_split"] = work["eval_split"].astype(str)
    work["target"] = work["target"].astype(str)

    work = work[work["target"].isin(cfg.targets)].copy()
    work = work[work["eval_split"].isin(cfg.eval_splits)].copy()

    if cfg.seq_len is not None:
        work = work[pd.to_numeric(work["seq_len"], errors="coerce") == cfg.seq_len].copy()

    if work.empty:
        raise ValueError("No rows left after filtering targets/eval_splits/seq_len.")

    zero = filter_float(work, "calibration_frac", 0.0)
    zero = filter_float(zero, "calibration_lr", cfg.calibration_lr)

    cal = filter_float(work, "calibration_frac", cfg.calibration_frac)
    cal = filter_float(cal, "calibration_lr", cfg.calibration_lr)

    if zero.empty:
        raise ValueError(
            f"No zero-shot rows found for calibration_lr={cfg.calibration_lr}. "
            "Check calibration_lr or input file."
        )
    if cal.empty:
        raise ValueError(
            f"No calibrated rows found for calibration_lr={cfg.calibration_lr}, "
            f"calibration_frac={cfg.calibration_frac}."
        )

    keys = ["eval_split", "seq_len", "subject_id", "target"]

    zero = zero.sort_values(keys).drop_duplicates(keys, keep="first")
    cal = cal.sort_values(keys).drop_duplicates(keys, keep="first")

    keep_metrics = keys + [
        "n_eval",
        "n_cal_train",
        "n_cal_val",
        "epochs_ran",
        "best_val_loss",
        "r2",
        "spearman",
        "mae",
        "rmse",
    ]

    for col in keep_metrics:
        if col not in zero.columns:
            zero[col] = np.nan
        if col not in cal.columns:
            cal[col] = np.nan

    paired = zero[keep_metrics].merge(
        cal[keep_metrics],
        on=keys,
        suffixes=("_zero", "_calibrated"),
        how="inner",
    )

    if paired.empty:
        raise ValueError("Could not pair zero-shot and calibrated rows.")

    paired["calibration_lr"] = cfg.calibration_lr
    paired["calibration_frac"] = cfg.calibration_frac

    paired["r2_gain"] = paired["r2_calibrated"] - paired["r2_zero"]
    paired["spearman_gain"] = paired["spearman_calibrated"] - paired["spearman_zero"]
    paired["mae_reduction"] = paired["mae_zero"] - paired["mae_calibrated"]
    paired["rmse_reduction"] = paired["rmse_zero"] - paired["rmse_calibrated"]

    paired["r2_improved"] = paired["r2_gain"] > 0
    paired["spearman_improved"] = paired["spearman_gain"] > 0
    paired["mae_improved"] = paired["mae_reduction"] > 0
    paired["rmse_improved"] = paired["rmse_reduction"] > 0

    return paired


def safe_mean(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").mean())


def safe_median(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").median())


def positive_rate(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float((values > 0).mean())


def summarize_by_target(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, target), g in paired.groupby(["eval_split", "target"], sort=True):
        rows.append(
            {
                "eval_split": split,
                "target": target,
                "n_subjects": int(g["subject_id"].nunique()),
                "mean_r2_zero": safe_mean(g["r2_zero"]),
                "mean_r2_calibrated": safe_mean(g["r2_calibrated"]),
                "mean_r2_gain": safe_mean(g["r2_gain"]),
                "median_r2_gain": safe_median(g["r2_gain"]),
                "r2_positive_rate": positive_rate(g["r2_gain"]),
                "mean_spearman_zero": safe_mean(g["spearman_zero"]),
                "mean_spearman_calibrated": safe_mean(g["spearman_calibrated"]),
                "mean_spearman_gain": safe_mean(g["spearman_gain"]),
                "spearman_positive_rate": positive_rate(g["spearman_gain"]),
                "mean_mae_reduction": safe_mean(g["mae_reduction"]),
                "mae_positive_rate": positive_rate(g["mae_reduction"]),
                "mean_rmse_reduction": safe_mean(g["rmse_reduction"]),
                "rmse_positive_rate": positive_rate(g["rmse_reduction"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_subject(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, subject), g in paired.groupby(["eval_split", "subject_id"], sort=True):
        rows.append(
            {
                "eval_split": split,
                "subject_id": subject,
                "n_targets": int(g["target"].nunique()),
                "mean_r2_zero": safe_mean(g["r2_zero"]),
                "mean_r2_calibrated": safe_mean(g["r2_calibrated"]),
                "mean_r2_gain": safe_mean(g["r2_gain"]),
                "targets_r2_improved": int(g["r2_improved"].sum()),
                "r2_target_positive_rate": positive_rate(g["r2_gain"]),
                "mean_spearman_zero": safe_mean(g["spearman_zero"]),
                "mean_spearman_calibrated": safe_mean(g["spearman_calibrated"]),
                "mean_spearman_gain": safe_mean(g["spearman_gain"]),
                "targets_spearman_improved": int(g["spearman_improved"].sum()),
                "spearman_target_positive_rate": positive_rate(g["spearman_gain"]),
                "mean_mae_reduction": safe_mean(g["mae_reduction"]),
                "mean_rmse_reduction": safe_mean(g["rmse_reduction"]),
                "mean_n_eval": safe_mean(g["n_eval_calibrated"])
                if "n_eval_calibrated" in g.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall(paired: pd.DataFrame, by_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in paired.groupby("eval_split", sort=True):
        subj = by_subject[by_subject["eval_split"] == split]
        rows.append(
            {
                "eval_split": split,
                "n_subjects": int(g["subject_id"].nunique()),
                "n_target_subject_pairs": int(len(g)),
                "mean_r2_zero": safe_mean(g["r2_zero"]),
                "mean_r2_calibrated": safe_mean(g["r2_calibrated"]),
                "mean_r2_gain": safe_mean(g["r2_gain"]),
                "median_r2_gain": safe_median(g["r2_gain"]),
                "target_subject_r2_positive_rate": positive_rate(g["r2_gain"]),
                "subject_mean_r2_positive_rate": positive_rate(subj["mean_r2_gain"]),
                "mean_spearman_zero": safe_mean(g["spearman_zero"]),
                "mean_spearman_calibrated": safe_mean(g["spearman_calibrated"]),
                "mean_spearman_gain": safe_mean(g["spearman_gain"]),
                "target_subject_spearman_positive_rate": positive_rate(g["spearman_gain"]),
                "subject_mean_spearman_positive_rate": positive_rate(subj["mean_spearman_gain"]),
                "mean_mae_reduction": safe_mean(g["mae_reduction"]),
                "mean_rmse_reduction": safe_mean(g["rmse_reduction"]),
            }
        )
    return pd.DataFrame(rows)


def plot_subject_r2_gain(by_subject: pd.DataFrame, output_dir: Path) -> None:
    for split, g in by_subject.groupby("eval_split", sort=True):
        if g.empty:
            continue
        data = g.sort_values("mean_r2_gain")
        plt.figure(figsize=(10, max(4, 0.35 * len(data))))
        plt.barh(data["subject_id"].astype(str), data["mean_r2_gain"])
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("Mean R² gain after calibration")
        plt.ylabel("Subject")
        plt.title(f"Per-subject mean R² gain ({split})")
        plt.tight_layout()
        plt.savefig(output_dir / f"subject_mean_r2_gain_{split}.png", dpi=180)
        plt.close()


def plot_target_r2_gain(summary_by_target: pd.DataFrame, output_dir: Path) -> None:
    for split, g in summary_by_target.groupby("eval_split", sort=True):
        if g.empty:
            continue
        data = g.sort_values("target")
        plt.figure(figsize=(8, 4.5))
        plt.bar(data["target"], data["mean_r2_gain"])
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("Mean R² gain")
        plt.xlabel("Target")
        plt.title(f"Mean R² gain by target ({split})")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"target_mean_r2_gain_{split}.png", dpi=180)
        plt.close()


def plot_zero_vs_calibrated(paired: pd.DataFrame, output_dir: Path) -> None:
    for split, g in paired.groupby("eval_split", sort=True):
        if g.empty:
            continue
        plt.figure(figsize=(6, 6))
        plt.scatter(g["r2_zero"], g["r2_calibrated"])
        min_v = float(np.nanmin([g["r2_zero"].min(), g["r2_calibrated"].min()]))
        max_v = float(np.nanmax([g["r2_zero"].max(), g["r2_calibrated"].max()]))
        pad = max(0.05, 0.05 * (max_v - min_v))
        plt.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], linewidth=1)
        plt.xlabel("Zero-shot R²")
        plt.ylabel("Calibrated R²")
        plt.title(f"Zero-shot vs calibrated R² ({split})")
        plt.tight_layout()
        plt.savefig(output_dir / f"zero_vs_calibrated_r2_{split}.png", dpi=180)
        plt.close()


def plot_positive_rate(summary_by_target: pd.DataFrame, output_dir: Path) -> None:
    for split, g in summary_by_target.groupby("eval_split", sort=True):
        if g.empty:
            continue
        data = g.sort_values("target")
        plt.figure(figsize=(8, 4.5))
        plt.bar(data["target"], data["r2_positive_rate"])
        plt.ylim(0, 1)
        plt.ylabel("Share of subjects with R² gain > 0")
        plt.xlabel("Target")
        plt.title(f"R² positive-rate by target ({split})")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"target_r2_positive_rate_{split}.png", dpi=180)
        plt.close()


def build_report(
    output_dir: Path,
    cfg: DiagnosticsConfig,
    paired: pd.DataFrame,
    summary_by_target: pd.DataFrame,
    summary_by_subject: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Per-subject calibration diagnostics")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Check whether the selected personal calibration protocol improves performance "
        "for most held-out subjects, not only in the aggregate mean."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")
    lines.append(overall.to_markdown(index=False))
    lines.append("")

    lines.append("## Summary by target")
    lines.append("")
    lines.append(summary_by_target.to_markdown(index=False))
    lines.append("")

    for split in cfg.eval_splits:
        subj = summary_by_subject[summary_by_subject["eval_split"] == split].copy()
        if subj.empty:
            continue

        lines.append(f"## Subject-level summary: {split}")
        lines.append("")
        display_cols = [
            "subject_id",
            "n_targets",
            "mean_r2_zero",
            "mean_r2_calibrated",
            "mean_r2_gain",
            "targets_r2_improved",
            "r2_target_positive_rate",
            "mean_spearman_gain",
            "targets_spearman_improved",
            "mean_n_eval",
        ]
        display_cols = [c for c in display_cols if c in subj.columns]
        lines.append(subj[display_cols].sort_values("mean_r2_gain", ascending=False).to_markdown(index=False))
        lines.append("")

        lines.append(f"### Best subjects by mean R² gain: {split}")
        lines.append("")
        lines.append(subj.sort_values("mean_r2_gain", ascending=False).head(5)[display_cols].to_markdown(index=False))
        lines.append("")

        lines.append(f"### Worst subjects by mean R² gain: {split}")
        lines.append("")
        lines.append(subj.sort_values("mean_r2_gain", ascending=True).head(5)[display_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `mean_r2_gain = calibrated R² - zero-shot R²`.")
    lines.append("- `mae_reduction = zero-shot MAE - calibrated MAE`; positive values mean improvement.")
    lines.append("- `rmse_reduction = zero-shot RMSE - calibrated RMSE`; positive values mean improvement.")
    lines.append("- Subject-level rows average the gains over the selected latent targets.")
    lines.append("- A strong calibration protocol should have positive mean gain and high positive-rate across subjects.")
    lines.append("")

    report_path = output_dir / "per_subject_calibration_diagnostics_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze per-subject gains for personal calibration."
    )

    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--input",
        default="reports/feature_ablation_v2/calibration_pow_plus_eeg/val_test_per_subject_metrics.csv",
        help="Input val_test_per_subject_metrics.csv from script 46.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg",
        help="Output diagnostics directory.",
    )
    parser.add_argument("--calibration-lr", type=float, default=0.0001)
    parser.add_argument("--calibration-frac", type=float, default=0.20)
    parser.add_argument("--targets", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_TARGETS))
    parser.add_argument("--eval-splits", type=parse_csv_strings, default=parse_csv_strings("val,test"))
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--min-subjects-for-summary", type=int, default=1)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = DiagnosticsConfig(
        root=args.root,
        input=args.input,
        output_dir=args.output_dir,
        calibration_lr=args.calibration_lr,
        calibration_frac=args.calibration_frac,
        targets=args.targets,
        eval_splits=args.eval_splits,
        seq_len=args.seq_len,
        min_subjects_for_summary=args.min_subjects_for_summary,
    )

    root = Path(cfg.root).resolve()
    input_path = repo_path(root, cfg.input)
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    paired = prepare_paired_gains(df, cfg)

    summary_by_target = summarize_by_target(paired)
    summary_by_subject = summarize_by_subject(paired)
    overall = summarize_overall(paired, summary_by_subject)

    paired_path = output_dir / "paired_subject_target_gains.csv"
    target_path = output_dir / "summary_by_target.csv"
    subject_path = output_dir / "summary_by_subject.csv"
    overall_path = output_dir / "overall_summary.csv"
    config_path = output_dir / "per_subject_diagnostics_config.json"

    paired.to_csv(paired_path, index=False)
    summary_by_target.to_csv(target_path, index=False)
    summary_by_subject.to_csv(subject_path, index=False)
    overall.to_csv(overall_path, index=False)
    save_json(config_path, asdict(cfg))

    plot_subject_r2_gain(summary_by_subject, output_dir)
    plot_target_r2_gain(summary_by_target, output_dir)
    plot_zero_vs_calibrated(paired, output_dir)
    plot_positive_rate(summary_by_target, output_dir)

    build_report(
        output_dir=output_dir,
        cfg=cfg,
        paired=paired,
        summary_by_target=summary_by_target,
        summary_by_subject=summary_by_subject,
        overall=overall,
    )

    print(f"Saved: {paired_path}")
    print(f"Saved: {target_path}")
    print(f"Saved: {subject_path}")
    print(f"Saved: {overall_path}")
    print(f"Saved: {output_dir / 'per_subject_calibration_diagnostics_report.md'}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
