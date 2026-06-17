#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final experiment summary for EEG latent-state NIR project.

This script aggregates already produced experiment outputs into a single
technical summary report.

It does not train models.

Expected default inputs:
  1) Feature ablation v2:
     reports/feature_ablation_v2/calibration_pow/val_test_mean_protocol_summary.csv
     reports/feature_ablation_v2/calibration_eeg/val_test_mean_protocol_summary.csv
     reports/feature_ablation_v2/calibration_pow_plus_eeg/val_test_mean_protocol_summary.csv

  2) Per-subject calibration diagnostics:
     reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg/overall_summary.csv
     reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg/summary_by_target.csv
     reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg/summary_by_subject.csv

  3) Temporal baseline comparison:
     reports/temporal_baselines/pow_plus_eeg_seq8_pca123/model_summary.csv
     reports/temporal_baselines/pow_plus_eeg_seq8_pca123/calibration_summary.csv
     reports/temporal_baselines/pow_plus_eeg_seq8_pca123/train_info.csv

Outputs:
  reports/final_experiment_summary/final_experiment_summary_report.md
  reports/final_experiment_summary/feature_ablation_selected_protocols.csv
  reports/final_experiment_summary/temporal_baseline_zero_full_test.csv
  reports/final_experiment_summary/temporal_baseline_calibrated_test.csv
  reports/final_experiment_summary/final_key_results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_SETS_DEFAULT = ["pow", "eeg", "pow_plus_eeg"]


@dataclass
class FinalSummaryConfig:
    root: str
    output_dir: str
    feature_ablation_dir: str
    subject_diagnostics_dir: str
    temporal_baselines_dir: str
    feature_sets: list[str]
    final_feature_set: str
    final_targets: list[str]
    final_seq_len: int
    final_calibration_lr: float
    final_calibration_frac: float


def parse_csv_strings(value: str) -> list[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected comma-separated non-empty values.")
    return items


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def read_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return None


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_value(x: Any, digits: int = 4) -> str:
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        return f"{float(x):.{digits}f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if pd.isna(x):
        return ""
    return str(x)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None, digits: int = 4) -> str:
    """Small markdown-table formatter without requiring tabulate."""
    if df is None or df.empty:
        return "_No data._"

    view = df.copy()
    if max_rows is not None and len(view) > max_rows:
        view = view.head(max_rows).copy()

    cols = list(view.columns)

    def esc(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    rows: list[list[str]] = []
    rows.append([esc(c) for c in cols])
    rows.append(["---" for _ in cols])

    for _, row in view.iterrows():
        rows.append([esc(fmt_value(row[c], digits=digits)) for c in cols])

    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    lines = []
    for idx, r in enumerate(rows):
        line = "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        lines.append(line)
    return "\n".join(lines)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_feature_ablation(cfg: FinalSummaryConfig, root: Path) -> pd.DataFrame:
    base = repo_path(root, cfg.feature_ablation_dir)
    frames = []

    for fs in cfg.feature_sets:
        path = base / f"calibration_{fs}" / "val_test_mean_protocol_summary.csv"
        df = read_csv_if_exists(path, required=False)
        if df is None:
            print(f"[WARN] Missing feature ablation file for {fs}: {path}")
            continue

        df = df.copy()
        df["feature_set"] = fs
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = normalize_numeric(
        out,
        [
            "seq_len",
            "calibration_lr",
            "calibration_frac",
            "mean_r2",
            "mean_spearman",
            "mean_mae",
            "mean_rmse",
        ],
    )
    return out


def select_feature_ablation_protocols(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every feature_set:
      - choose best validation protocol by max validation mean_r2;
      - report matched test row with the same seq_len/lr/frac;
      - also report test-best as diagnostic, not for selection.
    """
    if feature_df.empty:
        return pd.DataFrame()

    rows = []

    for fs, g in feature_df.groupby("feature_set", sort=False):
        val = g[g["eval_split"].astype(str) == "val"].copy()
        test = g[g["eval_split"].astype(str) == "test"].copy()

        if val.empty or test.empty:
            continue

        val_best = val.loc[val["mean_r2"].idxmax()].copy()

        cond = (
            np.isclose(test["seq_len"], val_best["seq_len"], rtol=0.0, atol=1e-12)
            & np.isclose(test["calibration_lr"], val_best["calibration_lr"], rtol=0.0, atol=1e-12)
            & np.isclose(test["calibration_frac"], val_best["calibration_frac"], rtol=0.0, atol=1e-12)
        )
        matched_test = test[cond]
        if matched_test.empty:
            test_selected = pd.Series(dtype=object)
        else:
            test_selected = matched_test.iloc[0]

        test_best = test.loc[test["mean_r2"].idxmax()].copy()

        rows.append(
            {
                "feature_set": fs,
                "selected_seq_len": int(val_best["seq_len"]),
                "selected_lr": float(val_best["calibration_lr"]),
                "selected_frac": float(val_best["calibration_frac"]),
                "val_selected_mean_r2": safe_float(val_best["mean_r2"]),
                "val_selected_spearman": safe_float(val_best["mean_spearman"]),
                "test_at_selected_mean_r2": safe_float(test_selected.get("mean_r2", np.nan)),
                "test_at_selected_spearman": safe_float(test_selected.get("mean_spearman", np.nan)),
                "test_best_lr": float(test_best["calibration_lr"]),
                "test_best_frac": float(test_best["calibration_frac"]),
                "test_best_mean_r2": safe_float(test_best["mean_r2"]),
                "test_best_spearman": safe_float(test_best["mean_spearman"]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("test_at_selected_mean_r2", ascending=False).reset_index(drop=True)
    return out


def read_subject_diagnostics(cfg: FinalSummaryConfig, root: Path) -> dict[str, pd.DataFrame]:
    base = repo_path(root, cfg.subject_diagnostics_dir)

    files = {
        "overall": base / "overall_summary.csv",
        "by_target": base / "summary_by_target.csv",
        "by_subject": base / "summary_by_subject.csv",
    }

    out: dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        df = read_csv_if_exists(path, required=False)
        if df is None:
            print(f"[WARN] Missing subject diagnostics file: {path}")
            out[key] = pd.DataFrame()
        else:
            out[key] = df

    return out


def read_temporal_baselines(cfg: FinalSummaryConfig, root: Path) -> dict[str, pd.DataFrame]:
    base = repo_path(root, cfg.temporal_baselines_dir)

    files = {
        "model_summary": base / "model_summary.csv",
        "calibration_summary": base / "calibration_summary.csv",
        "train_info": base / "train_info.csv",
    }

    out: dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        df = read_csv_if_exists(path, required=False)
        if df is None:
            print(f"[WARN] Missing temporal baseline file: {path}")
            out[key] = pd.DataFrame()
        else:
            out[key] = df

    return out


def summarize_temporal_zero_full(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame()

    df = model_summary.copy()
    df = normalize_numeric(df, ["mean_r2", "mean_spearman", "mean_mae", "mean_rmse"])

    required = {"model", "eval_split", "phase", "mean_r2", "mean_spearman"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df[
        (df["eval_split"].astype(str) == "test")
        & (df["phase"].astype(str) == "zero_full")
    ].copy()

    if out.empty:
        return out

    keep = ["model", "eval_split", "phase", "mean_r2", "mean_spearman", "mean_mae", "mean_rmse"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].sort_values("mean_r2", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_test_r2", np.arange(1, len(out) + 1))
    return out


def summarize_temporal_calibrated(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame()

    df = model_summary.copy()
    df = normalize_numeric(df, ["mean_r2", "mean_spearman", "mean_mae", "mean_rmse"])

    required = {"model", "eval_split", "phase", "mean_r2", "mean_spearman"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df[
        (df["eval_split"].astype(str) == "test")
        & (df["phase"].astype(str) == "calibrated")
    ].copy()

    if out.empty:
        return out

    keep = ["model", "eval_split", "phase", "mean_r2", "mean_spearman", "mean_mae", "mean_rmse"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].sort_values("mean_r2", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_test_r2", np.arange(1, len(out) + 1))
    return out


def extract_test_overall(subject_overall: pd.DataFrame) -> dict[str, Any]:
    if subject_overall.empty or "eval_split" not in subject_overall.columns:
        return {}

    df = subject_overall.copy()
    test = df[df["eval_split"].astype(str) == "test"].copy()
    if test.empty:
        return {}

    row = test.iloc[0].to_dict()
    return row


def extract_test_target_summary(subject_by_target: pd.DataFrame) -> pd.DataFrame:
    if subject_by_target.empty or "eval_split" not in subject_by_target.columns:
        return pd.DataFrame()

    df = subject_by_target.copy()
    df = df[df["eval_split"].astype(str) == "test"].copy()

    keep = [
        "target",
        "n_subjects",
        "mean_r2_zero",
        "mean_r2_calibrated",
        "mean_r2_gain",
        "r2_positive_rate",
        "mean_spearman_zero",
        "mean_spearman_calibrated",
        "mean_spearman_gain",
        "spearman_positive_rate",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def build_key_results(
    cfg: FinalSummaryConfig,
    feature_selected: pd.DataFrame,
    subject_diag: dict[str, pd.DataFrame],
    temporal_zero: pd.DataFrame,
    temporal_calibrated: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_configuration": {
            "feature_set": cfg.final_feature_set,
            "seq_len": cfg.final_seq_len,
            "targets": cfg.final_targets,
            "calibration_lr": cfg.final_calibration_lr,
            "calibration_frac": cfg.final_calibration_frac,
        },
    }

    if not feature_selected.empty:
        final_row = feature_selected[feature_selected["feature_set"] == cfg.final_feature_set]
        if not final_row.empty:
            r = final_row.iloc[0].to_dict()
            out["feature_ablation_final_feature_set"] = {
                k: safe_float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
                for k, v in r.items()
            }

    overall = extract_test_overall(subject_diag.get("overall", pd.DataFrame()))
    if overall:
        out["per_subject_calibration_test_overall"] = {
            k: safe_float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
            for k, v in overall.items()
        }

    if not temporal_zero.empty:
        best_zero = temporal_zero.iloc[0].to_dict()
        out["temporal_baseline_best_zero_full_test"] = {
            k: safe_float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
            for k, v in best_zero.items()
        }

    if not temporal_calibrated.empty:
        best_cal = temporal_calibrated.iloc[0].to_dict()
        out["temporal_baseline_best_calibrated_test"] = {
            k: safe_float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
            for k, v in best_cal.items()
        }

    return out


def build_report(
    cfg: FinalSummaryConfig,
    output_dir: Path,
    feature_selected: pd.DataFrame,
    subject_diag: dict[str, pd.DataFrame],
    temporal_zero: pd.DataFrame,
    temporal_calibrated: pd.DataFrame,
    train_info: pd.DataFrame,
    key_results: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("# Final experiment summary")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Final selected setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| feature_set | `{cfg.final_feature_set}` |")
    lines.append(f"| seq_len | `{cfg.final_seq_len}` |")
    lines.append(f"| targets | `{', '.join(cfg.final_targets)}` |")
    lines.append(f"| calibration_lr | `{cfg.final_calibration_lr}` |")
    lines.append(f"| calibration_frac | `{cfg.final_calibration_frac}` |")
    lines.append("")

    lines.append("## 1. Feature ablation")
    lines.append("")
    if feature_selected.empty:
        lines.append("_Feature ablation data not found._")
    else:
        display = feature_selected.copy()
        lines.append(df_to_markdown(display, digits=4))
        lines.append("")
        final_row = display[display["feature_set"] == cfg.final_feature_set]
        if not final_row.empty:
            r = final_row.iloc[0]
            lines.append(
                f"Selected final feature set: `{cfg.final_feature_set}`. "
                f"Validation-selected test mean R² = `{fmt_value(r['test_at_selected_mean_r2'])}`, "
                f"test Spearman = `{fmt_value(r['test_at_selected_spearman'])}`."
            )
    lines.append("")

    lines.append("## 2. Per-subject calibration diagnostics")
    lines.append("")
    overall = subject_diag.get("overall", pd.DataFrame())
    by_target_test = extract_test_target_summary(subject_diag.get("by_target", pd.DataFrame()))

    if overall.empty:
        lines.append("_Per-subject diagnostics not found._")
    else:
        lines.append("### Overall")
        lines.append("")
        lines.append(df_to_markdown(overall, digits=4))
        lines.append("")

        test_overall = extract_test_overall(overall)
        if test_overall:
            lines.append("### Main test interpretation")
            lines.append("")
            lines.append(
                "For the final configuration, personal head-only calibration "
                f"changed test mean R² from `{fmt_value(test_overall.get('mean_r2_zero'))}` "
                f"to `{fmt_value(test_overall.get('mean_r2_calibrated'))}`, "
                f"with mean gain `{fmt_value(test_overall.get('mean_r2_gain'))}`."
            )
            lines.append("")
            lines.append(
                "Subject-level positive-rate by mean R² gain: "
                f"`{fmt_value(test_overall.get('subject_mean_r2_positive_rate'))}`. "
                "This is the key check that the improvement is not driven only by one or two subjects."
            )
            lines.append("")

        if not by_target_test.empty:
            lines.append("### Test summary by latent target")
            lines.append("")
            lines.append(df_to_markdown(by_target_test, digits=4))
            lines.append("")

    lines.append("## 3. Temporal baseline comparison")
    lines.append("")
    lines.append(
        "The temporal baseline experiment is used mainly to compare architectures "
        "in zero-shot full evaluation. The calibrated part of that experiment is diagnostic, "
        "because the main calibration result is taken from the dedicated feature_ablation_v2 "
        "and per-subject diagnostics pipeline."
    )
    lines.append("")

    lines.append("### Zero-shot full test ranking")
    lines.append("")
    if temporal_zero.empty:
        lines.append("_Temporal zero-shot summary not found._")
    else:
        lines.append(df_to_markdown(temporal_zero, digits=4))
        lines.append("")
        best = temporal_zero.iloc[0]
        lines.append(
            f"Best zero-shot full test model: `{best['model']}` "
            f"with mean R² = `{fmt_value(best['mean_r2'])}` "
            f"and Spearman = `{fmt_value(best['mean_spearman'])}`."
        )
    lines.append("")

    lines.append("### Calibrated test ranking from temporal-baseline script")
    lines.append("")
    if temporal_calibrated.empty:
        lines.append("_Temporal calibrated summary not found._")
    else:
        lines.append(df_to_markdown(temporal_calibrated, digits=4))
        lines.append("")
        lines.append(
            "This table is kept as a diagnostic comparison only. "
            "It should not override the main calibrated result from scripts 46/47."
        )
    lines.append("")

    lines.append("### Training summary for temporal baselines")
    lines.append("")
    if train_info.empty:
        lines.append("_Training info not found._")
    else:
        cols = [c for c in ["model", "best_epoch", "best_val_loss"] if c in train_info.columns]
        lines.append(df_to_markdown(train_info[cols] if cols else train_info, digits=4))
    lines.append("")

    lines.append("## Final conclusions")
    lines.append("")
    lines.append("1. The final selected feature representation is `pow_plus_eeg`.")
    lines.append("2. The final latent targets are `slow_pca_1`, `slow_pca_2`, and `slow_pca_3`; `slow_pca_4` remains excluded as unstable.")
    lines.append("3. Personal head-only calibration is supported by per-subject diagnostics, especially on the final test split.")
    lines.append("4. Transformer is supported as the best temporal architecture in zero-shot full test comparison.")
    lines.append("5. The safest final claim is not that calibration universally helps every possible user, but that it improved mean R² for all held-out test subjects in the final selected protocol.")
    lines.append("")

    lines.append("## Files generated by this script")
    lines.append("")
    lines.append("- `feature_ablation_selected_protocols.csv`")
    lines.append("- `temporal_baseline_zero_full_test.csv`")
    lines.append("- `temporal_baseline_calibrated_test.csv`")
    lines.append("- `final_key_results.json`")
    lines.append("- `final_experiment_summary_report.md`")
    lines.append("")

    report_text = "\n".join(lines)
    return report_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final EEG experiment summary report.")

    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument(
        "--output-dir",
        default="reports/final_experiment_summary",
        help="Output directory for final summary artifacts.",
    )
    parser.add_argument(
        "--feature-ablation-dir",
        default="reports/feature_ablation_v2",
        help="Directory with feature ablation v2 outputs.",
    )
    parser.add_argument(
        "--subject-diagnostics-dir",
        default="reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg",
        help="Directory with per-subject diagnostics outputs.",
    )
    parser.add_argument(
        "--temporal-baselines-dir",
        default="reports/temporal_baselines/pow_plus_eeg_seq8_pca123",
        help="Directory with temporal baseline outputs.",
    )
    parser.add_argument(
        "--feature-sets",
        type=parse_csv_strings,
        default=FEATURE_SETS_DEFAULT,
        help="Comma-separated feature sets used in feature ablation.",
    )
    parser.add_argument("--final-feature-set", default="pow_plus_eeg")
    parser.add_argument(
        "--final-targets",
        type=parse_csv_strings,
        default=parse_csv_strings("slow_pca_1,slow_pca_2,slow_pca_3"),
    )
    parser.add_argument("--final-seq-len", type=int, default=8)
    parser.add_argument("--final-calibration-lr", type=float, default=0.0001)
    parser.add_argument("--final-calibration-frac", type=float, default=0.20)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = FinalSummaryConfig(
        root=args.root,
        output_dir=args.output_dir,
        feature_ablation_dir=args.feature_ablation_dir,
        subject_diagnostics_dir=args.subject_diagnostics_dir,
        temporal_baselines_dir=args.temporal_baselines_dir,
        feature_sets=args.feature_sets,
        final_feature_set=args.final_feature_set,
        final_targets=args.final_targets,
        final_seq_len=args.final_seq_len,
        final_calibration_lr=args.final_calibration_lr,
        final_calibration_frac=args.final_calibration_frac,
    )

    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "final_summary_config.json"
    save_json(config_path, asdict(cfg))

    print("=" * 80)
    print("Reading feature ablation outputs")
    print("=" * 80)
    feature_df = read_feature_ablation(cfg, root)
    feature_selected = select_feature_ablation_protocols(feature_df)

    feature_selected_path = output_dir / "feature_ablation_selected_protocols.csv"
    feature_selected.to_csv(feature_selected_path, index=False)
    print(f"Saved: {feature_selected_path}")

    print("=" * 80)
    print("Reading subject diagnostics")
    print("=" * 80)
    subject_diag = read_subject_diagnostics(cfg, root)

    subject_overall_copy = subject_diag.get("overall", pd.DataFrame())
    if not subject_overall_copy.empty:
        path = output_dir / "subject_calibration_overall.csv"
        subject_overall_copy.to_csv(path, index=False)
        print(f"Saved: {path}")

    subject_target_copy = subject_diag.get("by_target", pd.DataFrame())
    if not subject_target_copy.empty:
        path = output_dir / "subject_calibration_by_target.csv"
        subject_target_copy.to_csv(path, index=False)
        print(f"Saved: {path}")

    subject_copy = subject_diag.get("by_subject", pd.DataFrame())
    if not subject_copy.empty:
        path = output_dir / "subject_calibration_by_subject.csv"
        subject_copy.to_csv(path, index=False)
        print(f"Saved: {path}")

    print("=" * 80)
    print("Reading temporal baseline outputs")
    print("=" * 80)
    temporal = read_temporal_baselines(cfg, root)

    temporal_zero = summarize_temporal_zero_full(temporal.get("model_summary", pd.DataFrame()))
    temporal_calibrated = summarize_temporal_calibrated(temporal.get("model_summary", pd.DataFrame()))
    train_info = temporal.get("train_info", pd.DataFrame())

    temporal_zero_path = output_dir / "temporal_baseline_zero_full_test.csv"
    temporal_calibrated_path = output_dir / "temporal_baseline_calibrated_test.csv"

    temporal_zero.to_csv(temporal_zero_path, index=False)
    temporal_calibrated.to_csv(temporal_calibrated_path, index=False)

    print(f"Saved: {temporal_zero_path}")
    print(f"Saved: {temporal_calibrated_path}")

    key_results = build_key_results(
        cfg=cfg,
        feature_selected=feature_selected,
        subject_diag=subject_diag,
        temporal_zero=temporal_zero,
        temporal_calibrated=temporal_calibrated,
    )

    key_results_path = output_dir / "final_key_results.json"
    save_json(key_results_path, key_results)
    print(f"Saved: {key_results_path}")

    report_text = build_report(
        cfg=cfg,
        output_dir=output_dir,
        feature_selected=feature_selected,
        subject_diag=subject_diag,
        temporal_zero=temporal_zero,
        temporal_calibrated=temporal_calibrated,
        train_info=train_info,
        key_results=key_results,
    )

    report_path = output_dir / "final_experiment_summary_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved: {report_path}")

    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()