#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reliable-axes calibration validation/test check.

Purpose
-------
Run calibration protocol sensitivity for a reduced target set, usually:
  slow_pca_1,slow_pca_2,slow_pca_3

The script evaluates calibration separately on:
  - validation subjects: for protocol selection / diagnostics;
  - test subjects: for final held-out check.

It reuses helper functions from:
  src/45_run_calibration_protocol_sensitivity.py
and the trained base models produced by:
  src/44_run_seq_len_sensitivity.py

Expected base run directory layout:
  reports/seq_len_sensitivity/pm_w10_classic_split_pca123/seq8/best_model.pt
  reports/seq_len_sensitivity/pm_w10_classic_split_pca123/seq8/preprocessing.pkl
  reports/seq_len_sensitivity/pm_w10_classic_split_pca123/seq8/split_meta.json

Recommended usage:
  1) First train a base model with targets slow_pca_1,slow_pca_2,slow_pca_3 using script 44.
  2) Then run this script to compare validation and test calibration behavior.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"


@dataclass
class EvalConfig:
    root: str
    dataset: str
    base_run_dir: str
    output_dir: str
    seq_lens: list[int]
    targets: list[str]
    feature_set: str
    max_features: int
    stride: int
    eval_splits: list[str]
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    pooling: str
    batch_size: int
    calibration_fracs: list[float]
    calibration_lrs: list[float]
    calibration_epochs: int
    calibration_patience: int
    calibration_val_frac: float
    calibration_mode: str
    calibration_seed: int
    max_subjects: int | None
    min_subject_sequences: int
    subject_selection: str
    min_eval_sequences: int
    device: str
    dry_run: bool


def parse_csv_ints(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers")
    return out


def parse_csv_floats(value: str) -> list[float]:
    out = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated floats")
    return out


def parse_csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated strings")
    return out


def repo_path(root: Path, relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else root / p


def load_numbered_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("reliable_axes_calibration_val_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "reliable_axes_calibration_val_test.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def recover_split_indices(meta: pd.DataFrame, split_meta: dict, split_name: str) -> np.ndarray:
    split_level = split_meta.get("split_level", "subject")
    if split_level == "subject":
        key = "subject_id"
    elif split_level == "record":
        key = "group_key"
    else:
        raise ValueError(f"Unsupported split_level in split_meta: {split_level}")

    group_key = f"{split_name}_groups"
    groups = split_meta.get(group_key)
    if not groups:
        raise ValueError(f"No {group_key} found in split_meta.json")

    values = meta[key].astype(str).to_numpy(dtype=object)
    groups_arr = np.asarray([str(x) for x in groups], dtype=object)
    idx = np.flatnonzero(np.isin(values, groups_arr)).astype(np.int64)
    if len(idx) == 0:
        raise ValueError(f"Recovered empty {split_name} indices from split_meta.json")
    return idx


def select_subjects(meta: pd.DataFrame, split_idx: np.ndarray, cfg: EvalConfig) -> list[str]:
    split_meta = meta.iloc[split_idx].copy()
    counts = split_meta["subject_id"].astype(str).value_counts()
    counts = counts[counts >= cfg.min_subject_sequences]
    if counts.empty:
        return []

    if cfg.subject_selection == "largest":
        subjects = counts.sort_values(ascending=False).index.tolist()
    elif cfg.subject_selection == "random":
        rng = np.random.default_rng(cfg.calibration_seed)
        arr = counts.index.to_numpy(dtype=object)
        rng.shuffle(arr)
        subjects = [str(x) for x in arr]
    else:
        subjects = counts.index.sort_values().tolist()

    if cfg.max_subjects is not None:
        subjects = subjects[: cfg.max_subjects]
    return [str(x) for x in subjects]


def build_subject_splits(subject_indices: np.ndarray, frac: float, calibration_val_frac: float, min_eval_sequences: int):
    n = len(subject_indices)
    if frac <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), subject_indices.astype(np.int64)

    cal_n = max(1, int(np.floor(n * frac)))
    cal_n = min(cal_n, n - min_eval_sequences)
    if cal_n <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cal_pool = subject_indices[:cal_n]
    eval_idx = subject_indices[cal_n:]
    if len(eval_idx) < min_eval_sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if len(cal_pool) <= 2:
        return cal_pool.astype(np.int64), np.array([], dtype=np.int64), eval_idx.astype(np.int64)

    val_n = int(np.floor(len(cal_pool) * calibration_val_frac))
    val_n = max(1, val_n)
    val_n = min(val_n, len(cal_pool) - 1)
    return cal_pool[:-val_n].astype(np.int64), cal_pool[-val_n:].astype(np.int64), eval_idx.astype(np.int64)


def run_one_seq_split(seq_module: Any, cal_module: Any, cfg: EvalConfig, root: Path, seq_len: int, eval_split: str, logger: logging.Logger):
    base_root = repo_path(root, cfg.base_run_dir)
    seq_dir = base_root / f"seq{seq_len}"
    checkpoint_path = seq_dir / "best_model.pt"
    preprocessing_path = seq_dir / "preprocessing.pkl"
    split_meta_path = seq_dir / "split_meta.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Missing preprocessing: {preprocessing_path}")
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Missing split meta: {split_meta_path}")

    logger.info("=" * 90)
    logger.info("seq_len=%d | eval_split=%s", seq_len, eval_split)
    logger.info("=" * 90)

    df = seq_module.read_table(repo_path(root, cfg.dataset))
    id_cols = seq_module.detect_id_columns(df)
    feature_cols = seq_module.select_feature_columns(df, cfg.targets, cfg.max_features, logger, feature_set=cfg.feature_set)

    x_raw, y, meta = seq_module.build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=cfg.targets,
        id_cols=id_cols,
        seq_len=seq_len,
        stride=cfg.stride,
        logger=logger,
    )

    x = cal_module.transform_features_with_saved_preprocessing(x_raw, preprocessing_path)
    split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
    split_idx = recover_split_indices(meta, split_meta, eval_split)

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = seq_module.TransformerRegressor(
        n_features=x.shape[-1],
        n_targets=y.shape[-1],
        seq_len=seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    subjects = select_subjects(meta, split_idx, cfg)
    logger.info("Selected %d subjects from %s split", len(subjects), eval_split)

    split_meta_df = meta.iloc[split_idx].copy()
    split_meta_df["_seq_idx"] = split_idx

    per_subject_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []

    # Build a minimal object compatible with functions from script 45.
    class CompatCfg:
        pass

    compat = CompatCfg()
    compat.calibration_mode = cfg.calibration_mode
    compat.batch_size = cfg.batch_size
    compat.calibration_epochs = cfg.calibration_epochs
    compat.calibration_patience = cfg.calibration_patience
    compat.calibration_val_frac = cfg.calibration_val_frac
    compat.min_eval_sequences = cfg.min_eval_sequences
    compat.targets = cfg.targets

    for lr in cfg.calibration_lrs:
        logger.info("Calibration lr=%.6g", lr)
        for subject in subjects:
            sm = split_meta_df[split_meta_df["subject_id"].astype(str) == str(subject)].copy()
            sort_cols = [c for c in ["source", "record_id", "sequence_start", "sequence_end"] if c in sm.columns]
            if sort_cols:
                sm = sm.sort_values(sort_cols)
            subject_indices = sm["_seq_idx"].to_numpy(dtype=np.int64)
            if len(subject_indices) < cfg.min_subject_sequences:
                continue

            for frac in cfg.calibration_fracs:
                if frac == 0:
                    eval_idx = subject_indices
                    if len(eval_idx) < cfg.min_eval_sequences:
                        continue
                    y_pred = cal_module.predict_model(seq_module, model, x, eval_idx, cfg.batch_size, device)
                    info = {
                        "best_val_loss": np.nan,
                        "epochs_ran": 0,
                        "n_cal_train": 0,
                        "n_cal_val": 0,
                        "n_eval": int(len(eval_idx)),
                    }
                else:
                    cal_train_idx, cal_val_idx, eval_idx = build_subject_splits(
                        subject_indices=subject_indices,
                        frac=float(frac),
                        calibration_val_frac=cfg.calibration_val_frac,
                        min_eval_sequences=cfg.min_eval_sequences,
                    )
                    if len(cal_train_idx) == 0 or len(eval_idx) == 0:
                        continue

                    y_pred, info = cal_module.calibrate_one_subject_with_val(
                        seq_module=seq_module,
                        base_model=model,
                        x=x,
                        y=y,
                        cal_train_idx=cal_train_idx,
                        cal_val_idx=cal_val_idx,
                        eval_idx=eval_idx,
                        cfg=compat,
                        lr=lr,
                        device=device,
                    )

                y_true = y[eval_idx]
                metrics = seq_module.compute_metrics(
                    y_true,
                    y_pred,
                    cfg.targets,
                    prefix={
                        "eval_split": eval_split,
                        "seq_len": seq_len,
                        "subject_id": str(subject),
                        "calibration_lr": float(lr),
                        "calibration_frac": float(frac),
                        "n_subject_sequences": int(len(subject_indices)),
                        "n_cal_train": int(info["n_cal_train"]),
                        "n_cal_val": int(info["n_cal_val"]),
                        "n_eval": int(info["n_eval"]),
                        "epochs_ran": int(info["epochs_ran"]),
                        "best_val_loss": float(info["best_val_loss"]) if np.isfinite(info["best_val_loss"]) else np.nan,
                    },
                )
                per_subject_parts.append(metrics)

                pred_df = pd.DataFrame(
                    {
                        "eval_split": eval_split,
                        "seq_len": seq_len,
                        "subject_id": str(subject),
                        "calibration_lr": float(lr),
                        "calibration_frac": float(frac),
                        "global_sequence_idx": eval_idx,
                    }
                )
                for j, target in enumerate(cfg.targets):
                    pred_df[f"y_true_{target}"] = y_true[:, j]
                    pred_df[f"y_pred_{target}"] = y_pred[:, j]
                pred_parts.append(pred_df)

    per_subject = pd.concat(per_subject_parts, ignore_index=True) if per_subject_parts else pd.DataFrame()
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return per_subject, preds


def summarize(per_subject: pd.DataFrame):
    if per_subject.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    keys = ["eval_split", "seq_len", "target", "calibration_lr", "calibration_frac"]
    summary = (
        per_subject.groupby(keys, as_index=False)[
            ["mae", "rmse", "r2", "spearman", "epochs_ran", "n_cal_train", "n_cal_val", "n_eval"]
        ]
        .mean(numeric_only=True)
    )
    counts = per_subject.groupby(keys, as_index=False).agg(
        n_subjects=("subject_id", "nunique"),
        n_eval_total=("n_eval", "sum"),
    )
    summary = summary.merge(counts, on=keys, how="left")

    protocol = (
        summary.groupby(["eval_split", "seq_len", "calibration_lr", "calibration_frac"], as_index=False)[
            ["r2", "spearman", "mae", "rmse"]
        ]
        .mean(numeric_only=True)
        .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman", "mae": "mean_mae", "rmse": "mean_rmse"})
    )

    gains = []
    for (eval_split, seq_len, target, lr), g in summary.groupby(["eval_split", "seq_len", "target", "calibration_lr"]):
        z = g[g["calibration_frac"] == 0]
        if z.empty:
            continue
        z_r2 = float(z["r2"].iloc[0])
        z_sp = float(z["spearman"].iloc[0])
        for _, row in g.iterrows():
            gains.append(
                {
                    "eval_split": eval_split,
                    "seq_len": int(seq_len),
                    "target": target,
                    "calibration_lr": float(lr),
                    "calibration_frac": float(row["calibration_frac"]),
                    "zero_shot_r2": z_r2,
                    "calibrated_r2": float(row["r2"]),
                    "r2_gain": float(row["r2"] - z_r2),
                    "zero_shot_spearman": z_sp,
                    "calibrated_spearman": float(row["spearman"]),
                    "spearman_gain": float(row["spearman"] - z_sp),
                    "n_subjects": int(row["n_subjects"]),
                }
            )
    return summary, protocol, pd.DataFrame(gains)


def build_report(output_dir: Path, cfg: EvalConfig, summary: pd.DataFrame, protocol: pd.DataFrame, gains: pd.DataFrame) -> None:
    lines = []
    lines.append("# Reliable axes calibration validation/test report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Evaluate head-only calibration on reduced latent target set and compare protocol behavior "
        "on validation subjects and final test subjects."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    if not protocol.empty:
        lines.append("## Mean protocol metrics")
        lines.append("")
        lines.append(protocol.sort_values(["eval_split", "seq_len", "calibration_lr", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

        best_val = protocol[protocol["eval_split"] == "val"].sort_values("mean_r2", ascending=False)
        best_test = protocol[protocol["eval_split"] == "test"].sort_values("mean_r2", ascending=False)
        lines.append("## Best protocols")
        lines.append("")
        if not best_val.empty:
            r = best_val.iloc[0]
            lines.append(f"- Validation best: seq_len={int(r.seq_len)}, lr={r.calibration_lr}, frac={r.calibration_frac}, mean R²={r.mean_r2:.4f}, mean Spearman={r.mean_spearman:.4f}.")
        if not best_test.empty:
            r = best_test.iloc[0]
            lines.append(f"- Test best: seq_len={int(r.seq_len)}, lr={r.calibration_lr}, frac={r.calibration_frac}, mean R²={r.mean_r2:.4f}, mean Spearman={r.mean_spearman:.4f}.")
        lines.append("")

    if not summary.empty:
        lines.append("## Per-target summary")
        lines.append("")
        cols = ["eval_split", "seq_len", "target", "calibration_lr", "calibration_frac", "n_subjects", "r2", "spearman", "mae", "rmse", "epochs_ran"]
        lines.append(summary[cols].sort_values(["eval_split", "seq_len", "target", "calibration_lr", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

    if not gains.empty:
        lines.append("## Calibration gains")
        lines.append("")
        lines.append(gains.sort_values(["eval_split", "seq_len", "target", "calibration_lr", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- Validation split is used for protocol selection and diagnostics.")
    lines.append("- Test split is used for final held-out check.")
    lines.append("- Calibration is always subject-specific: a part of the held-out subject sequence is used for calibration, and the remaining part is used for evaluation.")
    lines.append("- Non-zero calibration fractions use an internal calibration-train / calibration-validation split for early stopping.")
    lines.append("- Reduced target set removes the weakest slow_pca_4 axis from this experiment.")

    (output_dir / "reliable_axes_calibration_val_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate reduced-target calibration on validation and test splits.")
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet")
    p.add_argument("--base-run-dir", default="reports/seq_len_sensitivity/pm_w10_classic_split_pca123")
    p.add_argument("--output-dir", default="reports/reliable_axes_calibration_val_test/pm_w10_seq8_pca123")
    p.add_argument("--seq-lens", type=parse_csv_ints, default=parse_csv_ints("8"))
    p.add_argument("--targets", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_TARGETS))
    p.add_argument("--feature-set", default="pow_plus_eeg")
    p.add_argument("--max-features", type=int, default=448)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--eval-splits", type=parse_csv_strings, default=parse_csv_strings("val,test"))
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--calibration-fracs", type=parse_csv_floats, default=parse_csv_floats("0,0.05,0.10,0.15,0.20"))
    p.add_argument("--calibration-lrs", type=parse_csv_floats, default=parse_csv_floats("0.0001,0.0003,0.001"))
    p.add_argument("--calibration-epochs", type=int, default=40)
    p.add_argument("--calibration-patience", type=int, default=6)
    p.add_argument("--calibration-val-frac", type=float, default=0.25)
    p.add_argument("--calibration-mode", choices=["head_only"], default="head_only")
    p.add_argument("--calibration-seed", type=int, default=123)
    p.add_argument("--max-subjects", type=int, default=30)
    p.add_argument("--min-subject-sequences", type=int, default=80)
    p.add_argument("--subject-selection", choices=["largest", "random", "sorted"], default="largest")
    p.add_argument("--min-eval-sequences", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = EvalConfig(**vars(args))
    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    save_json(output_dir / "reliable_axes_calibration_val_test_config.json", asdict(cfg))

    logger.info("Saved config: %s", output_dir / "reliable_axes_calibration_val_test_config.json")
    logger.info("Targets: %s", cfg.targets)
    logger.info("Eval splits: %s", cfg.eval_splits)

    if cfg.dry_run:
        logger.info("[dry-run] Would evaluate base_run_dir=%s", cfg.base_run_dir)
        return

    np.random.seed(cfg.calibration_seed)
    torch.manual_seed(cfg.calibration_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.calibration_seed)

    seq_module = load_numbered_module(root / "src" / "44_run_seq_len_sensitivity.py", "seq_len_module_for_46")
    cal_module = load_numbered_module(root / "src" / "45_run_calibration_protocol_sensitivity.py", "cal_module_for_46")

    all_metrics: list[pd.DataFrame] = []
    all_preds: list[pd.DataFrame] = []
    t0 = time.time()

    for seq_len in cfg.seq_lens:
        for eval_split in cfg.eval_splits:
            metrics, preds = run_one_seq_split(seq_module, cal_module, cfg, root, seq_len, eval_split, logger)
            if not metrics.empty:
                all_metrics.append(metrics)
            if not preds.empty:
                all_preds.append(preds)

    per_subject = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    summary, protocol, gains = summarize(per_subject)

    if not per_subject.empty:
        path = output_dir / "val_test_per_subject_metrics.csv"
        per_subject.to_csv(path, index=False)
        logger.info("Saved: %s", path)
    if not summary.empty:
        path = output_dir / "val_test_protocol_summary.csv"
        summary.to_csv(path, index=False)
        logger.info("Saved: %s", path)
    if not protocol.empty:
        path = output_dir / "val_test_mean_protocol_summary.csv"
        protocol.to_csv(path, index=False)
        logger.info("Saved: %s", path)
    if not gains.empty:
        path = output_dir / "val_test_gain_summary.csv"
        gains.to_csv(path, index=False)
        logger.info("Saved: %s", path)
    if not predictions.empty:
        path = output_dir / "val_test_predictions.csv"
        predictions.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    build_report(output_dir, cfg, summary, protocol, gains)
    logger.info("Saved report: %s", output_dir / "reliable_axes_calibration_val_test_report.md")
    logger.info("Done. Elapsed: %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
