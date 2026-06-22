#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split-seed robustness experiment for the final EEG latent-state protocol.

Goal
----
Check whether the final result is stable across several subject-wise train/val/test
splits, not only on one random_state.

This script orchestrates existing scripts:
  - src/44_run_seq_len_sensitivity.py
  - src/46_run_reliable_axes_calibration_val_test.py

It then aggregates:
  - validation/test mean metrics for the fixed final calibration protocol
  - per-subject and per-target positive-rate diagnostics
  - mean/std/min/max across random seeds

Final default protocol
----------------------
feature_set = pow_plus_eeg
seq_len = 8
targets = slow_pca_1,slow_pca_2,slow_pca_3
calibration_lr = 0.0001
calibration_frac = 0.20

Example
-------
D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\50_run_split_seed_robustness.py `
  --root . `
  --dataset reports\\slow_latent_states\\pm_w10\\slow_pm_latent_states_w10.parquet `
  --output-dir reports\\split_seed_robustness\\pow_plus_eeg_seq8_pca123 `
  --seeds 42,123,2024,3407,777 `
  --device cuda

Fast smoke test
---------------
D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\50_run_split_seed_robustness.py `
  --root . `
  --dataset reports\\slow_latent_states\\pm_w10\\slow_pm_latent_states_w10.parquet `
  --output-dir reports\\split_seed_robustness\\smoke `
  --seeds 42 `
  --device cuda

Outputs
-------
reports/split_seed_robustness/.../
  split_seed_robustness_report.md
  per_seed_protocol_summary.csv
  aggregate_protocol_summary.csv
  per_seed_target_summary.csv
  aggregate_target_summary.csv
  per_seed_subject_summary.csv
  split_seed_config.json
  logs/
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"
DEFAULT_SEEDS = "42,123,2024,3407,777"


@dataclass
class SplitSeedConfig:
    root: str
    dataset: str
    output_dir: str
    seeds: list[int]
    feature_set: str
    max_features: int
    targets: list[str]
    seq_len: int
    split_level: str
    train_size: float
    val_size: float
    test_size: float
    calibration_lr: float
    calibration_frac: float
    calibration_epochs: int
    calibration_patience: int
    calibration_val_frac: float
    max_subjects: int
    min_subject_sequences: int
    min_eval_sequences: int
    subject_selection: str
    device: str
    python_exe: str
    run_training: bool
    run_calibration: bool
    reuse_existing: bool
    dry_run: bool


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated integer list.")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated non-empty string list.")
    return out


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return ""
            return f"{float(x):.{digits}f}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)
    except Exception:
        return str(x)


def df_to_markdown(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
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
        rows.append([esc(fmt(row[c], digits=digits)) for c in cols])

    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    lines = []
    for r in rows:
        lines.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    return "\n".join(lines)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def run_command(
    cmd: list[str],
    log_path: Path,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_text = " ".join(str(x) for x in cmd)
    print("=" * 100)
    print(cmd_text)
    print("=" * 100)

    if dry_run:
        log_path.write_text(cmd_text + "\n", encoding="utf-8")
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(cmd_text + "\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)

        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}. See log: {log_path}")


def build_train_cmd(cfg: SplitSeedConfig, root: Path, seed: int, base_dir: Path) -> list[str]:
    script = root / "src" / "44_run_seq_len_sensitivity.py"

    cmd = [
        cfg.python_exe,
        str(script),
        "--root",
        str(root),
        "--dataset",
        cfg.dataset,
        "--output-dir",
        str(base_dir),
        "--seq-lens",
        str(cfg.seq_len),
        "--targets",
        ",".join(cfg.targets),
        "--feature-set",
        cfg.feature_set,
        "--max-features",
        str(cfg.max_features),
        "--split-level",
        cfg.split_level,
        "--train-size",
        str(cfg.train_size),
        "--val-size",
        str(cfg.val_size),
        "--test-size",
        str(cfg.test_size),
        "--random-state",
        str(seed),
        "--mode",
        "transformer",
        "--device",
        cfg.device,
    ]

    return cmd


def build_calibration_cmd(cfg: SplitSeedConfig, root: Path, base_dir: Path, calib_dir: Path) -> list[str]:
    script = root / "src" / "46_run_reliable_axes_calibration_val_test.py"

    cmd = [
        cfg.python_exe,
        str(script),
        "--root",
        str(root),
        "--dataset",
        cfg.dataset,
        "--base-run-dir",
        str(base_dir),
        "--output-dir",
        str(calib_dir),
        "--seq-lens",
        str(cfg.seq_len),
        "--targets",
        ",".join(cfg.targets),
        "--feature-set",
        cfg.feature_set,
        "--eval-splits",
        "val,test",
        "--calibration-lrs",
        str(cfg.calibration_lr),
        "--calibration-fracs",
        f"0,{cfg.calibration_frac}",
        "--calibration-val-frac",
        str(cfg.calibration_val_frac),
        "--calibration-epochs",
        str(cfg.calibration_epochs),
        "--calibration-patience",
        str(cfg.calibration_patience),
        "--max-subjects",
        str(cfg.max_subjects),
        "--min-subject-sequences",
        str(cfg.min_subject_sequences),
        "--min-eval-sequences",
        str(cfg.min_eval_sequences),
        "--subject-selection",
        cfg.subject_selection,
        "--device",
        cfg.device,
    ]

    return cmd


def check_required_files(root: Path) -> None:
    required = [
        root / "src" / "44_run_seq_len_sensitivity.py",
        root / "src" / "46_run_reliable_axes_calibration_val_test.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required scripts:\n" + "\n".join(missing))


def read_mean_protocol_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing mean protocol summary: {path}")

    df = pd.read_csv(path)
    df = numeric(
        df,
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
    return df


def isclose_series(series: pd.Series, value: float, atol: float = 1e-12) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce"), value, rtol=0.0, atol=atol)


def get_protocol_row(
    df: pd.DataFrame,
    eval_split: str,
    seq_len: int,
    lr: float,
    frac: float,
) -> pd.Series | None:
    mask = (
        (df["eval_split"].astype(str) == eval_split)
        & isclose_series(df["seq_len"], seq_len)
        & isclose_series(df["calibration_lr"], lr)
        & isclose_series(df["calibration_frac"], frac)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def safe_value(row: pd.Series | None, col: str) -> float:
    if row is None:
        return float("nan")
    try:
        return float(row[col])
    except Exception:
        return float("nan")


def summarize_seed_protocol(
    cfg: SplitSeedConfig,
    seed: int,
    calib_dir: Path,
) -> list[dict[str, Any]]:
    path = calib_dir / "val_test_mean_protocol_summary.csv"
    df = read_mean_protocol_summary(path)

    rows: list[dict[str, Any]] = []

    for split in ["val", "test"]:
        zero = get_protocol_row(
            df=df,
            eval_split=split,
            seq_len=cfg.seq_len,
            lr=cfg.calibration_lr,
            frac=0.0,
        )
        calibrated = get_protocol_row(
            df=df,
            eval_split=split,
            seq_len=cfg.seq_len,
            lr=cfg.calibration_lr,
            frac=cfg.calibration_frac,
        )

        rows.append(
            {
                "seed": seed,
                "eval_split": split,
                "feature_set": cfg.feature_set,
                "seq_len": cfg.seq_len,
                "targets": ",".join(cfg.targets),
                "calibration_lr": cfg.calibration_lr,
                "calibration_frac": cfg.calibration_frac,
                "mean_r2_zero": safe_value(zero, "mean_r2"),
                "mean_r2_calibrated": safe_value(calibrated, "mean_r2"),
                "mean_r2_gain": safe_value(calibrated, "mean_r2") - safe_value(zero, "mean_r2"),
                "mean_spearman_zero": safe_value(zero, "mean_spearman"),
                "mean_spearman_calibrated": safe_value(calibrated, "mean_spearman"),
                "mean_spearman_gain": safe_value(calibrated, "mean_spearman") - safe_value(zero, "mean_spearman"),
                "mean_mae_zero": safe_value(zero, "mean_mae"),
                "mean_mae_calibrated": safe_value(calibrated, "mean_mae"),
                "mean_mae_reduction": safe_value(zero, "mean_mae") - safe_value(calibrated, "mean_mae"),
                "mean_rmse_zero": safe_value(zero, "mean_rmse"),
                "mean_rmse_calibrated": safe_value(calibrated, "mean_rmse"),
                "mean_rmse_reduction": safe_value(zero, "mean_rmse") - safe_value(calibrated, "mean_rmse"),
                "source_file": str(path),
            }
        )

    return rows


def read_per_subject_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing per-subject metrics: {path}")
        return None

    df = pd.read_csv(path)
    df = numeric(
        df,
        [
            "seq_len",
            "calibration_lr",
            "calibration_frac",
            "r2",
            "spearman",
            "mae",
            "rmse",
            "n_eval",
            "n_cal_train",
            "n_cal_val",
            "epochs_ran",
        ],
    )

    required = [
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
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Per-subject file has missing columns {missing}: {path}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["target"] = df["target"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)

    return df


def pair_subject_metrics(
    cfg: SplitSeedConfig,
    seed: int,
    per_subject_df: pd.DataFrame,
) -> pd.DataFrame:
    df = per_subject_df.copy()

    df = df[df["target"].isin(cfg.targets)].copy()
    df = df[isclose_series(df["seq_len"], cfg.seq_len)].copy()
    df = df[isclose_series(df["calibration_lr"], cfg.calibration_lr)].copy()

    zero = df[isclose_series(df["calibration_frac"], 0.0)].copy()
    cal = df[isclose_series(df["calibration_frac"], cfg.calibration_frac)].copy()

    keys = ["eval_split", "seq_len", "subject_id", "target"]

    zero = zero.sort_values(keys).drop_duplicates(keys, keep="first")
    cal = cal.sort_values(keys).drop_duplicates(keys, keep="first")

    keep = keys + [
        "r2",
        "spearman",
        "mae",
        "rmse",
        "n_eval",
        "n_cal_train",
        "n_cal_val",
        "epochs_ran",
    ]

    for col in keep:
        if col not in zero.columns:
            zero[col] = np.nan
        if col not in cal.columns:
            cal[col] = np.nan

    paired = zero[keep].merge(
        cal[keep],
        on=keys,
        suffixes=("_zero", "_calibrated"),
        how="inner",
    )

    if paired.empty:
        return paired

    paired["seed"] = seed
    paired["feature_set"] = cfg.feature_set
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


def positive_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float((values > 0).mean())


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return float("nan")
    return float(values.mean())


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return float("nan")
    return float(values.median())


def summarize_subject_positive_rates(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()

    target_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for (seed, split, target), g in paired.groupby(["seed", "eval_split", "target"], sort=True):
        target_rows.append(
            {
                "seed": seed,
                "eval_split": split,
                "target": target,
                "n_subjects": int(g["subject_id"].nunique()),
                "mean_r2_zero": mean_or_nan(g["r2_zero"]),
                "mean_r2_calibrated": mean_or_nan(g["r2_calibrated"]),
                "mean_r2_gain": mean_or_nan(g["r2_gain"]),
                "median_r2_gain": median_or_nan(g["r2_gain"]),
                "r2_positive_rate": positive_rate(g["r2_gain"]),
                "mean_spearman_zero": mean_or_nan(g["spearman_zero"]),
                "mean_spearman_calibrated": mean_or_nan(g["spearman_calibrated"]),
                "mean_spearman_gain": mean_or_nan(g["spearman_gain"]),
                "spearman_positive_rate": positive_rate(g["spearman_gain"]),
                "mean_mae_reduction": mean_or_nan(g["mae_reduction"]),
                "mean_rmse_reduction": mean_or_nan(g["rmse_reduction"]),
            }
        )

    for (seed, split, subject_id), g in paired.groupby(["seed", "eval_split", "subject_id"], sort=True):
        subject_rows.append(
            {
                "seed": seed,
                "eval_split": split,
                "subject_id": subject_id,
                "n_targets": int(g["target"].nunique()),
                "mean_r2_zero": mean_or_nan(g["r2_zero"]),
                "mean_r2_calibrated": mean_or_nan(g["r2_calibrated"]),
                "mean_r2_gain": mean_or_nan(g["r2_gain"]),
                "targets_r2_improved": int(g["r2_improved"].sum()),
                "r2_target_positive_rate": positive_rate(g["r2_gain"]),
                "mean_spearman_zero": mean_or_nan(g["spearman_zero"]),
                "mean_spearman_calibrated": mean_or_nan(g["spearman_calibrated"]),
                "mean_spearman_gain": mean_or_nan(g["spearman_gain"]),
                "targets_spearman_improved": int(g["spearman_improved"].sum()),
                "spearman_target_positive_rate": positive_rate(g["spearman_gain"]),
                "mean_mae_reduction": mean_or_nan(g["mae_reduction"]),
                "mean_rmse_reduction": mean_or_nan(g["rmse_reduction"]),
            }
        )

    return pd.DataFrame(target_rows), pd.DataFrame(subject_rows)


def add_subject_rates_to_seed_summary(
    per_seed_protocol: pd.DataFrame,
    paired: pd.DataFrame,
    subject_summary: pd.DataFrame,
) -> pd.DataFrame:
    if per_seed_protocol.empty or paired.empty:
        return per_seed_protocol

    extra_rows: list[dict[str, Any]] = []

    for (seed, split), g in paired.groupby(["seed", "eval_split"], sort=True):
        subj = subject_summary[
            (subject_summary["seed"] == seed)
            & (subject_summary["eval_split"].astype(str) == str(split))
        ]

        extra_rows.append(
            {
                "seed": seed,
                "eval_split": split,
                "n_subjects": int(g["subject_id"].nunique()),
                "n_subject_target_pairs": int(len(g)),
                "target_subject_r2_positive_rate": positive_rate(g["r2_gain"]),
                "subject_mean_r2_positive_rate": positive_rate(subj["mean_r2_gain"])
                if not subj.empty
                else float("nan"),
                "target_subject_spearman_positive_rate": positive_rate(g["spearman_gain"]),
                "subject_mean_spearman_positive_rate": positive_rate(subj["mean_spearman_gain"])
                if not subj.empty
                else float("nan"),
            }
        )

    extra = pd.DataFrame(extra_rows)

    out = per_seed_protocol.merge(
        extra,
        on=["seed", "eval_split"],
        how="left",
    )

    return out


def aggregate_protocol_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()

    metric_cols = [
        "mean_r2_zero",
        "mean_r2_calibrated",
        "mean_r2_gain",
        "mean_spearman_zero",
        "mean_spearman_calibrated",
        "mean_spearman_gain",
        "mean_mae_reduction",
        "mean_rmse_reduction",
        "target_subject_r2_positive_rate",
        "subject_mean_r2_positive_rate",
        "target_subject_spearman_positive_rate",
        "subject_mean_spearman_positive_rate",
    ]

    existing = [c for c in metric_cols if c in per_seed.columns]
    rows: list[dict[str, Any]] = []

    for split, g in per_seed.groupby("eval_split", sort=True):
        row: dict[str, Any] = {
            "eval_split": split,
            "n_seeds": int(g["seed"].nunique()),
        }

        for col in existing:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_target_summary(per_seed_target: pd.DataFrame) -> pd.DataFrame:
    if per_seed_target.empty:
        return pd.DataFrame()

    metric_cols = [
        "mean_r2_zero",
        "mean_r2_calibrated",
        "mean_r2_gain",
        "r2_positive_rate",
        "mean_spearman_zero",
        "mean_spearman_calibrated",
        "mean_spearman_gain",
        "spearman_positive_rate",
        "mean_mae_reduction",
        "mean_rmse_reduction",
    ]

    existing = [c for c in metric_cols if c in per_seed_target.columns]
    rows: list[dict[str, Any]] = []

    for (split, target), g in per_seed_target.groupby(["eval_split", "target"], sort=True):
        row: dict[str, Any] = {
            "eval_split": split,
            "target": target,
            "n_seeds": int(g["seed"].nunique()),
            "mean_n_subjects": float(pd.to_numeric(g["n_subjects"], errors="coerce").mean())
            if "n_subjects" in g.columns
            else float("nan"),
        }

        for col in existing:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())

        rows.append(row)

    return pd.DataFrame(rows)


def build_report(
    cfg: SplitSeedConfig,
    per_seed_protocol: pd.DataFrame,
    aggregate_protocol: pd.DataFrame,
    per_seed_target: pd.DataFrame,
    aggregate_target: pd.DataFrame,
    per_seed_subject: pd.DataFrame,
    failed_seeds: list[dict[str, Any]],
) -> str:
    lines: list[str] = []

    lines.append("# Split-seed robustness report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Check whether the final subject-wise result is stable across several random train/validation/test splits."
    )
    lines.append("")

    lines.append("## Fixed protocol")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| feature_set | `{cfg.feature_set}` |")
    lines.append(f"| seq_len | `{cfg.seq_len}` |")
    lines.append(f"| targets | `{', '.join(cfg.targets)}` |")
    lines.append(f"| calibration_lr | `{cfg.calibration_lr}` |")
    lines.append(f"| calibration_frac | `{cfg.calibration_frac}` |")
    lines.append(f"| seeds | `{', '.join(str(s) for s in cfg.seeds)}` |")
    lines.append("")

    if failed_seeds:
        lines.append("## Failed seeds")
        lines.append("")
        lines.append(df_to_markdown(pd.DataFrame(failed_seeds), digits=4))
        lines.append("")

    lines.append("## Aggregate protocol summary")
    lines.append("")
    if aggregate_protocol.empty:
        lines.append("_No aggregate protocol summary available._")
    else:
        compact_cols = [
            "eval_split",
            "n_seeds",
            "mean_r2_zero_mean",
            "mean_r2_zero_std",
            "mean_r2_calibrated_mean",
            "mean_r2_calibrated_std",
            "mean_r2_gain_mean",
            "mean_r2_gain_std",
            "mean_spearman_zero_mean",
            "mean_spearman_calibrated_mean",
            "mean_spearman_gain_mean",
            "subject_mean_r2_positive_rate_mean",
            "target_subject_r2_positive_rate_mean",
        ]
        compact_cols = [c for c in compact_cols if c in aggregate_protocol.columns]
        lines.append(df_to_markdown(aggregate_protocol[compact_cols], digits=4))
    lines.append("")

    lines.append("## Per-seed protocol summary")
    lines.append("")
    if per_seed_protocol.empty:
        lines.append("_No per-seed protocol summary available._")
    else:
        compact_cols = [
            "seed",
            "eval_split",
            "mean_r2_zero",
            "mean_r2_calibrated",
            "mean_r2_gain",
            "mean_spearman_zero",
            "mean_spearman_calibrated",
            "mean_spearman_gain",
            "subject_mean_r2_positive_rate",
            "target_subject_r2_positive_rate",
        ]
        compact_cols = [c for c in compact_cols if c in per_seed_protocol.columns]
        lines.append(df_to_markdown(per_seed_protocol[compact_cols], digits=4))
    lines.append("")

    lines.append("## Aggregate target summary")
    lines.append("")
    if aggregate_target.empty:
        lines.append("_No aggregate target summary available._")
    else:
        compact_cols = [
            "eval_split",
            "target",
            "n_seeds",
            "mean_r2_zero_mean",
            "mean_r2_calibrated_mean",
            "mean_r2_gain_mean",
            "mean_r2_gain_std",
            "r2_positive_rate_mean",
            "mean_spearman_zero_mean",
            "mean_spearman_calibrated_mean",
            "mean_spearman_gain_mean",
            "spearman_positive_rate_mean",
        ]
        compact_cols = [c for c in compact_cols if c in aggregate_target.columns]
        lines.append(df_to_markdown(aggregate_target[compact_cols], digits=4))
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `mean_r2_gain = calibrated R² - zero-shot R²`.")
    lines.append("- `subject_mean_r2_positive_rate` is the share of subjects whose mean gain over targets is positive.")
    lines.append("- `target_subject_r2_positive_rate` is the share of all subject × target pairs with positive R² gain.")
    lines.append("- A robust result should preserve positive mean R² gain across most seeds.")
    lines.append("- If the standard deviation across seeds is high, final claims should explicitly mention split sensitivity.")
    lines.append("")

    lines.append("## Suggested final wording")
    lines.append("")
    lines.append(
        "If the aggregate test gain remains positive across seeds, write: "
        "`The final protocol was evaluated across multiple subject-wise random splits. "
        "The calibration effect remained positive on average, which indicates that the result is not an artifact of a single split.`"
    )
    lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split-seed robustness for the final EEG protocol.")

    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument(
        "--dataset",
        default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet",
        help="Dataset path relative to root.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/split_seed_robustness/pow_plus_eeg_seq8_pca123",
        help="Output directory.",
    )
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list(DEFAULT_SEEDS))
    parser.add_argument("--feature-set", default="pow_plus_eeg")
    parser.add_argument("--max-features", type=int, default=448)
    parser.add_argument("--targets", type=parse_str_list, default=parse_str_list(DEFAULT_TARGETS))
    parser.add_argument("--seq-len", type=int, default=8)

    parser.add_argument("--split-level", default="subject")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)

    parser.add_argument("--calibration-lr", type=float, default=0.0001)
    parser.add_argument("--calibration-frac", type=float, default=0.20)
    parser.add_argument("--calibration-epochs", type=int, default=40)
    parser.add_argument("--calibration-patience", type=int, default=6)
    parser.add_argument("--calibration-val-frac", type=float, default=0.25)

    parser.add_argument("--max-subjects", type=int, default=30)
    parser.add_argument("--min-subject-sequences", type=int, default=80)
    parser.add_argument("--min-eval-sequences", type=int, default=20)
    parser.add_argument("--subject-selection", default="largest")

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable. Defaults to the interpreter that runs this script.",
    )

    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Do not run script 44; only aggregate existing outputs and optionally run calibration.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Do not run script 46; only aggregate existing outputs.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip train/calibration for a seed if expected output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write logs without executing them.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = SplitSeedConfig(
        root=args.root,
        dataset=args.dataset,
        output_dir=args.output_dir,
        seeds=args.seeds,
        feature_set=args.feature_set,
        max_features=args.max_features,
        targets=args.targets,
        seq_len=args.seq_len,
        split_level=args.split_level,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        calibration_lr=args.calibration_lr,
        calibration_frac=args.calibration_frac,
        calibration_epochs=args.calibration_epochs,
        calibration_patience=args.calibration_patience,
        calibration_val_frac=args.calibration_val_frac,
        max_subjects=args.max_subjects,
        min_subject_sequences=args.min_subject_sequences,
        min_eval_sequences=args.min_eval_sequences,
        subject_selection=args.subject_selection,
        device=args.device,
        python_exe=args.python_exe,
        run_training=not args.no_training,
        run_calibration=not args.no_calibration,
        reuse_existing=args.reuse_existing,
        dry_run=args.dry_run,
    )

    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "split_seed_config.json", asdict(cfg))

    check_required_files(root)

    dataset_path = repo_path(root, cfg.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    all_protocol_rows: list[dict[str, Any]] = []
    all_paired: list[pd.DataFrame] = []
    failed_seeds: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        print("\n" + "#" * 100)
        print(f"SEED {seed}")
        print("#" * 100)

        seed_dir = output_dir / f"seed_{seed}"
        base_dir = seed_dir / f"base_{cfg.feature_set}"
        calib_dir = seed_dir / f"calibration_{cfg.feature_set}"
        log_dir = output_dir / "logs"
        seed_dir.mkdir(parents=True, exist_ok=True)

        base_checkpoint = base_dir / f"seq{cfg.seq_len}" / "best_model.pt"
        calib_mean_file = calib_dir / "val_test_mean_protocol_summary.csv"
        calib_subject_file = calib_dir / "val_test_per_subject_metrics.csv"

        try:
            if cfg.run_training:
                if cfg.reuse_existing and base_checkpoint.exists():
                    print(f"[SKIP] Existing checkpoint found: {base_checkpoint}")
                else:
                    cmd = build_train_cmd(cfg, root, seed, base_dir)
                    run_command(
                        cmd=cmd,
                        log_path=log_dir / f"seed_{seed}_train.log",
                        dry_run=cfg.dry_run,
                        cwd=root,
                    )

            if cfg.run_calibration:
                if cfg.reuse_existing and calib_mean_file.exists() and calib_subject_file.exists():
                    print(f"[SKIP] Existing calibration files found: {calib_dir}")
                else:
                    if not cfg.dry_run and not base_checkpoint.exists():
                        raise FileNotFoundError(
                            f"Base checkpoint not found for seed {seed}: {base_checkpoint}"
                        )

                    cmd = build_calibration_cmd(cfg, root, base_dir, calib_dir)
                    run_command(
                        cmd=cmd,
                        log_path=log_dir / f"seed_{seed}_calibration.log",
                        dry_run=cfg.dry_run,
                        cwd=root,
                    )

            if cfg.dry_run:
                continue

            if not calib_mean_file.exists():
                raise FileNotFoundError(f"Missing calibration mean summary: {calib_mean_file}")

            all_protocol_rows.extend(summarize_seed_protocol(cfg, seed, calib_dir))

            per_subject = read_per_subject_metrics(calib_subject_file)
            if per_subject is not None:
                paired = pair_subject_metrics(cfg, seed, per_subject)
                if not paired.empty:
                    all_paired.append(paired)

        except Exception as exc:
            print(f"[ERROR] Seed {seed} failed: {exc}")
            failed_seeds.append(
                {
                    "seed": seed,
                    "error": repr(exc),
                    "seed_dir": str(seed_dir),
                }
            )

    if cfg.dry_run:
        print("Dry run finished. No aggregation was performed.")
        return

    per_seed_protocol = pd.DataFrame(all_protocol_rows)

    if all_paired:
        paired_all = pd.concat(all_paired, ignore_index=True)
    else:
        paired_all = pd.DataFrame()

    per_seed_target, per_seed_subject = summarize_subject_positive_rates(paired_all)

    per_seed_protocol = add_subject_rates_to_seed_summary(
        per_seed_protocol=per_seed_protocol,
        paired=paired_all,
        subject_summary=per_seed_subject,
    )

    aggregate_protocol = aggregate_protocol_summary(per_seed_protocol)
    aggregate_target = aggregate_target_summary(per_seed_target)

    per_seed_protocol_path = output_dir / "per_seed_protocol_summary.csv"
    aggregate_protocol_path = output_dir / "aggregate_protocol_summary.csv"
    paired_path = output_dir / "paired_subject_target_gains.csv"
    per_seed_target_path = output_dir / "per_seed_target_summary.csv"
    aggregate_target_path = output_dir / "aggregate_target_summary.csv"
    per_seed_subject_path = output_dir / "per_seed_subject_summary.csv"
    failed_path = output_dir / "failed_seeds.csv"

    per_seed_protocol.to_csv(per_seed_protocol_path, index=False)
    aggregate_protocol.to_csv(aggregate_protocol_path, index=False)
    paired_all.to_csv(paired_path, index=False)
    per_seed_target.to_csv(per_seed_target_path, index=False)
    aggregate_target.to_csv(aggregate_target_path, index=False)
    per_seed_subject.to_csv(per_seed_subject_path, index=False)

    if failed_seeds:
        pd.DataFrame(failed_seeds).to_csv(failed_path, index=False)

    report = build_report(
        cfg=cfg,
        per_seed_protocol=per_seed_protocol,
        aggregate_protocol=aggregate_protocol,
        per_seed_target=per_seed_target,
        aggregate_target=aggregate_target,
        per_seed_subject=per_seed_subject,
        failed_seeds=failed_seeds,
    )

    report_path = output_dir / "split_seed_robustness_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 100)
    print("Saved outputs")
    print("=" * 100)
    print(f"Saved: {per_seed_protocol_path}")
    print(f"Saved: {aggregate_protocol_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {per_seed_target_path}")
    print(f"Saved: {aggregate_target_path}")
    print(f"Saved: {per_seed_subject_path}")
    print(f"Saved: {report_path}")
    if failed_seeds:
        print(f"Saved: {failed_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()