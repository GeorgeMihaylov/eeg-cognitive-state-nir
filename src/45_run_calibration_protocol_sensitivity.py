#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibration protocol sensitivity for EEG latent trajectory Transformer.

This script uses the already trained models from:
  reports/seq_len_sensitivity/pm_w10_classic_split/seq{seq_len}/best_model.pt

It does NOT retrain the base Transformer. It only tests different personal
calibration protocols:

  - different calibration learning rates;
  - different calibration fractions;
  - calibration-train / calibration-validation split inside the user's
    calibration subset;
  - head-only calibration with early stopping.

Expected previous step:
  src/44_run_seq_len_sensitivity.py was already run with explicit
  train / validation / test split and saved:
    seq{seq_len}/best_model.pt
    seq{seq_len}/preprocessing.pkl
    seq{seq_len}/split_meta.json

Example:
  D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\45_run_calibration_protocol_sensitivity.py `
    --root . `
    --dataset reports\\slow_latent_states\\pm_w10\\slow_pm_latent_states_w10.parquet `
    --base-run-dir reports\\seq_len_sensitivity\\pm_w10_classic_split `
    --output-dir reports\\calibration_protocol_sensitivity\\pm_w10_seq8 `
    --seq-lens 8 `
    --calibration-lrs 0.0001,0.0003,0.001 `
    --calibration-fracs 0,0.05,0.10,0.15,0.20 `
    --calibration-val-frac 0.25 `
    --calibration-epochs 40 `
    --calibration-patience 6 `
    --device cuda
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except Exception as exc:
    raise RuntimeError("PyTorch is required for this script.") from exc


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3,slow_pca_4"


@dataclass
class CalibrationProtocolConfig:
    root: str
    dataset: str
    base_run_dir: str
    output_dir: str

    seq_lens: list[int]
    targets: list[str]
    feature_set: str
    max_features: int
    stride: int

    split_level: str
    train_size: float
    val_size: float
    test_size: float
    random_state: int

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
    skip_existing: bool
    dry_run: bool


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("All integer values must be positive")
    return out


def parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("calibration_protocol_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "calibration_protocol_sensitivity.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def load_seq_module(root: Path) -> Any:
    """
    Dynamically load src/44_run_seq_len_sensitivity.py.

    The filename starts with a number, so it cannot be imported via a normal
    Python import. This loader gives us access to the same model class and
    preprocessing functions used in the previous experiment.
    """
    module_path = root / "src" / "44_run_seq_len_sensitivity.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find previous script: {module_path}")

    spec = importlib.util.spec_from_file_location("seq_len_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["seq_len_module"] = module
    spec.loader.exec_module(module)
    return module


def transform_features_with_saved_preprocessing(
    x_raw: np.ndarray,
    preprocessing_path: Path,
) -> np.ndarray:
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Missing preprocessing file: {preprocessing_path}")

    with open(preprocessing_path, "rb") as f:
        obj = pickle.load(f)

    imputer = obj["imputer"]
    scaler = obj["scaler"]

    n, seq_len, n_features = x_raw.shape
    flat = x_raw.reshape(-1, n_features)
    flat = imputer.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n, seq_len, n_features).astype(np.float32)


def load_split_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing split metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def recover_test_indices_from_split_meta(meta: pd.DataFrame, split_meta: dict) -> np.ndarray:
    split_level = split_meta.get("split_level", "subject")

    if split_level == "subject":
        key = "subject_id"
        test_groups = split_meta.get("test_groups")
    elif split_level == "record":
        key = "group_key"
        test_groups = split_meta.get("test_groups")
    else:
        raise ValueError(
            f"Only subject/record split recovery is supported here. Got split_level={split_level}"
        )

    if not test_groups:
        raise ValueError(f"No test_groups found in split_meta.json for split_level={split_level}")

    values = meta[key].astype(str).to_numpy(dtype=object)
    mask = np.isin(values, np.asarray([str(x) for x in test_groups], dtype=object))
    idx = np.flatnonzero(mask).astype(np.int64)

    if len(idx) == 0:
        raise ValueError("Recovered empty test index from split_meta.json")
    return idx


def select_subjects_for_calibration(
    meta: pd.DataFrame,
    test_idx: np.ndarray,
    cfg: CalibrationProtocolConfig,
) -> list[str]:
    test_meta = meta.iloc[test_idx].copy()
    counts = test_meta["subject_id"].astype(str).value_counts()
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


def freeze_for_calibration(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head_only":
        for p in model.regression_head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported calibration mode: {mode}")


def make_loader(seq_module: Any, x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool):
    return seq_module.make_loader(x, y, idx, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(seq_module: Any, model: nn.Module, x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, device: torch.device) -> float:
    if len(idx) == 0:
        return float("inf")
    criterion = nn.MSELoss()
    loader = make_loader(seq_module, x, y, idx, batch_size=batch_size, shuffle=False)

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("inf")


def predict_model(seq_module: Any, model: nn.Module, x: np.ndarray, idx: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    return seq_module.predict_model(model, x, idx, batch_size, device)


def calibrate_one_subject_with_val(
    seq_module: Any,
    base_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    cal_train_idx: np.ndarray,
    cal_val_idx: np.ndarray,
    eval_idx: np.ndarray,
    cfg: CalibrationProtocolConfig,
    lr: float,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    model = copy.deepcopy(base_model)
    model.to(device)
    freeze_for_calibration(model, cfg.calibration_mode)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for calibration.")

    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0

    history = []
    train_loader = make_loader(seq_module, x, y, cal_train_idx, cfg.batch_size, shuffle=True)

    for epoch in range(1, cfg.calibration_epochs + 1):
        model.train()
        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        if len(cal_val_idx) > 0:
            val_loss = evaluate_loss(seq_module, model, x, y, cal_val_idx, cfg.batch_size, device)
        else:
            val_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.calibration_patience:
            break

    model.load_state_dict(best_state)

    y_pred = predict_model(seq_module, model, x, eval_idx, cfg.batch_size, device)
    info = {
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "n_cal_train": int(len(cal_train_idx)),
        "n_cal_val": int(len(cal_val_idx)),
        "n_eval": int(len(eval_idx)),
    }
    return y_pred, info


def build_subject_splits_for_calibration(
    subject_indices: np.ndarray,
    frac: float,
    calibration_val_frac: float,
    min_eval_sequences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_subject = len(subject_indices)

    if frac <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), subject_indices

    cal_n = max(1, int(math.floor(n_subject * frac)))
    cal_n = min(cal_n, n_subject - min_eval_sequences)

    if cal_n <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cal_pool = subject_indices[:cal_n]
    eval_idx = subject_indices[cal_n:]

    if len(eval_idx) < min_eval_sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if len(cal_pool) <= 2:
        cal_train_idx = cal_pool
        cal_val_idx = np.array([], dtype=np.int64)
    else:
        val_n = int(math.floor(len(cal_pool) * calibration_val_frac))
        val_n = max(1, val_n)
        val_n = min(val_n, len(cal_pool) - 1)

        cal_train_idx = cal_pool[:-val_n]
        cal_val_idx = cal_pool[-val_n:]

    return cal_train_idx.astype(np.int64), cal_val_idx.astype(np.int64), eval_idx.astype(np.int64)


def run_protocol_for_seq_len(
    seq_module: Any,
    cfg: CalibrationProtocolConfig,
    root: Path,
    seq_len: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_root = repo_path(root, cfg.base_run_dir)
    seq_dir = base_root / f"seq{seq_len}"

    if not seq_dir.exists():
        raise FileNotFoundError(f"Missing base seq_len directory: {seq_dir}")

    checkpoint_path = seq_dir / "best_model.pt"
    preprocessing_path = seq_dir / "preprocessing.pkl"
    split_meta_path = seq_dir / "split_meta.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    logger.info("=" * 90)
    logger.info("Processing seq_len=%d", seq_len)
    logger.info("=" * 90)

    dataset_path = repo_path(root, cfg.dataset)
    df = seq_module.read_table(dataset_path)
    id_cols = seq_module.detect_id_columns(df)
    feature_cols = seq_module.select_feature_columns(df, cfg.targets, cfg.max_features, logger)

    x_raw, y, meta = seq_module.build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=cfg.targets,
        id_cols=id_cols,
        seq_len=seq_len,
        stride=cfg.stride,
        logger=logger,
    )

    x = transform_features_with_saved_preprocessing(x_raw, preprocessing_path)
    split_meta = load_split_meta(split_meta_path)
    test_idx = recover_test_indices_from_split_meta(meta, split_meta)

    logger.info("Recovered test sequences: %d", len(test_idx))

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    if str(device) != cfg.device:
        logger.warning("Requested device=%s, but CUDA is unavailable. Using CPU.", cfg.device)

    base_model = seq_module.TransformerRegressor(
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

    base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    base_model.to(device)
    base_model.eval()

    subjects = select_subjects_for_calibration(meta, test_idx, cfg)
    logger.info("Calibration subjects selected: %d", len(subjects))

    test_meta = meta.iloc[test_idx].copy()
    test_meta["_seq_idx"] = test_idx

    per_subject_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []

    for lr in cfg.calibration_lrs:
        logger.info("Calibration learning rate: %.6g", lr)

        for subject in subjects:
            sm = test_meta[test_meta["subject_id"].astype(str) == str(subject)].copy()
            sm = sm.sort_values(["source", "record_id", "sequence_start", "sequence_end"])
            subject_indices = sm["_seq_idx"].to_numpy(dtype=np.int64)

            if len(subject_indices) < cfg.min_subject_sequences:
                continue

            for frac in cfg.calibration_fracs:
                if frac == 0:
                    eval_idx = subject_indices
                    if len(eval_idx) < cfg.min_eval_sequences:
                        continue

                    y_pred = predict_model(seq_module, base_model, x, eval_idx, cfg.batch_size, device)
                    info = {
                        "best_val_loss": np.nan,
                        "epochs_ran": 0,
                        "n_cal_train": 0,
                        "n_cal_val": 0,
                        "n_eval": int(len(eval_idx)),
                    }
                else:
                    cal_train_idx, cal_val_idx, eval_idx = build_subject_splits_for_calibration(
                        subject_indices=subject_indices,
                        frac=float(frac),
                        calibration_val_frac=cfg.calibration_val_frac,
                        min_eval_sequences=cfg.min_eval_sequences,
                    )

                    if len(cal_train_idx) == 0 or len(eval_idx) == 0:
                        continue

                    y_pred, info = calibrate_one_subject_with_val(
                        seq_module=seq_module,
                        base_model=base_model,
                        x=x,
                        y=y,
                        cal_train_idx=cal_train_idx,
                        cal_val_idx=cal_val_idx,
                        eval_idx=eval_idx,
                        cfg=cfg,
                        lr=lr,
                        device=device,
                    )

                y_true = y[eval_idx]

                metrics = seq_module.compute_metrics(
                    y_true,
                    y_pred,
                    cfg.targets,
                    prefix={
                        "seq_len": seq_len,
                        "subject_id": str(subject),
                        "calibration_lr": float(lr),
                        "calibration_frac": float(frac),
                        "n_subject_sequences": int(len(subject_indices)),
                        "n_cal_train": int(info["n_cal_train"]),
                        "n_cal_val": int(info["n_cal_val"]),
                        "n_eval": int(info["n_eval"]),
                        "epochs_ran": int(info["epochs_ran"]),
                        "best_val_loss": float(info["best_val_loss"])
                        if np.isfinite(info["best_val_loss"])
                        else np.nan,
                    },
                )
                per_subject_parts.append(metrics)

                # Compact prediction output for later diagnostics.
                pred_df = pd.DataFrame(
                    {
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
                prediction_parts.append(pred_df)

    per_subject = pd.concat(per_subject_parts, ignore_index=True) if per_subject_parts else pd.DataFrame()
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()

    return per_subject, predictions


def summarize_protocol(per_subject: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_subject.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        per_subject.groupby(
            ["seq_len", "target", "calibration_lr", "calibration_frac"],
            as_index=False,
        )[["mae", "rmse", "r2", "spearman", "epochs_ran", "n_cal_train", "n_cal_val", "n_eval"]]
        .mean(numeric_only=True)
    )

    count_df = (
        per_subject.groupby(
            ["seq_len", "target", "calibration_lr", "calibration_frac"],
            as_index=False,
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_eval_total=("n_eval", "sum"),
        )
    )

    summary = summary.merge(
        count_df,
        on=["seq_len", "target", "calibration_lr", "calibration_frac"],
        how="left",
    )

    gains = []
    for (seq_len, target, lr), g in summary.groupby(["seq_len", "target", "calibration_lr"]):
        zero = g[g["calibration_frac"] == 0]
        if zero.empty:
            continue

        zero_r2 = float(zero["r2"].iloc[0])
        zero_spearman = float(zero["spearman"].iloc[0])

        for _, row in g.iterrows():
            gains.append(
                {
                    "seq_len": int(seq_len),
                    "target": target,
                    "calibration_lr": float(lr),
                    "calibration_frac": float(row["calibration_frac"]),
                    "zero_shot_r2": zero_r2,
                    "calibrated_r2": float(row["r2"]),
                    "r2_gain": float(row["r2"] - zero_r2),
                    "zero_shot_spearman": zero_spearman,
                    "calibrated_spearman": float(row["spearman"]),
                    "spearman_gain": float(row["spearman"] - zero_spearman),
                    "n_subjects": int(row["n_subjects"]),
                }
            )

    gain_df = pd.DataFrame(gains)
    return summary, gain_df


def build_report(
    output_dir: Path,
    cfg: CalibrationProtocolConfig,
    summary: pd.DataFrame,
    gain_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Calibration protocol sensitivity report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Evaluate whether the instability of head-only personal calibration can be reduced "
        "by tuning the calibration learning rate, calibration fraction and internal "
        "calibration-train / calibration-validation split."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    if not summary.empty:
        lines.append("## Protocol summary")
        lines.append("")
        cols = [
            "seq_len",
            "target",
            "calibration_lr",
            "calibration_frac",
            "n_subjects",
            "r2",
            "spearman",
            "mae",
            "rmse",
            "epochs_ran",
        ]
        cols = [c for c in cols if c in summary.columns]
        lines.append(summary[cols].sort_values(["seq_len", "target", "calibration_lr", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

        lines.append("## Mean R² by protocol")
        lines.append("")
        mean_df = (
            summary.groupby(["seq_len", "calibration_lr", "calibration_frac"], as_index=False)[
                ["r2", "spearman"]
            ]
            .mean(numeric_only=True)
            .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman"})
        )
        lines.append(mean_df.to_markdown(index=False))
        lines.append("")

        if not mean_df.empty:
            best_row = mean_df.sort_values("mean_r2", ascending=False).iloc[0]
            lines.append("## Best mean R² protocol")
            lines.append("")
            lines.append(
                f"- `seq_len={int(best_row['seq_len'])}`; "
                f"`calibration_lr={best_row['calibration_lr']}`; "
                f"`calibration_frac={best_row['calibration_frac']}`; "
                f"mean R² = `{best_row['mean_r2']:.4f}`; "
                f"mean Spearman = `{best_row['mean_spearman']:.4f}`."
            )
            lines.append("")

    if not gain_df.empty:
        lines.append("## Calibration gains")
        lines.append("")
        cols = [
            "seq_len",
            "target",
            "calibration_lr",
            "calibration_frac",
            "zero_shot_r2",
            "calibrated_r2",
            "r2_gain",
            "zero_shot_spearman",
            "calibrated_spearman",
            "spearman_gain",
            "n_subjects",
        ]
        cols = [c for c in cols if c in gain_df.columns]
        lines.append(gain_df[cols].sort_values(["seq_len", "target", "calibration_lr", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- The base Transformer is fixed; only the calibration protocol changes.")
    lines.append("- Calibration is performed only on held-out test subjects.")
    lines.append("- For non-zero calibration fractions, the calibration pool is split into calibration-train and calibration-validation.")
    lines.append("- Early stopping uses the calibration-validation subset.")
    lines.append("- A stable protocol should improve mean R² and Spearman without strongly degrading individual axes.")
    lines.append("")

    report_path = output_dir / "calibration_protocol_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibration protocol sensitivity over trained seq_len Transformer models."
    )

    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--dataset",
        default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet",
        help="Path to slow latent states dataset.",
    )
    parser.add_argument(
        "--base-run-dir",
        default="reports/seq_len_sensitivity/pm_w10_classic_split",
        help="Directory produced by 44_run_seq_len_sensitivity.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/calibration_protocol_sensitivity/pm_w10_seq8",
        help="Output directory.",
    )

    parser.add_argument("--seq-lens", type=parse_csv_ints, default=parse_csv_ints("8"))
    parser.add_argument("--targets", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_TARGETS))
    parser.add_argument("--feature-set", default="pow_plus_eeg")
    parser.add_argument("--max-features", type=int, default=448)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--split-level", choices=["subject", "record"], default="subject")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--calibration-fracs", type=parse_csv_floats, default=parse_csv_floats("0,0.05,0.10,0.15,0.20"))
    parser.add_argument("--calibration-lrs", type=parse_csv_floats, default=parse_csv_floats("0.0001,0.0003,0.001"))
    parser.add_argument("--calibration-epochs", type=int, default=40)
    parser.add_argument("--calibration-patience", type=int, default=6)
    parser.add_argument("--calibration-val-frac", type=float, default=0.25)
    parser.add_argument("--calibration-mode", choices=["head_only"], default="head_only")
    parser.add_argument("--calibration-seed", type=int, default=123)

    parser.add_argument("--max-subjects", type=int, default=30)
    parser.add_argument("--min-subject-sequences", type=int, default=80)
    parser.add_argument("--subject-selection", choices=["largest", "random", "sorted"], default="largest")
    parser.add_argument("--min-eval-sequences", type=int, default=20)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = CalibrationProtocolConfig(
        root=args.root,
        dataset=args.dataset,
        base_run_dir=args.base_run_dir,
        output_dir=args.output_dir,
        seq_lens=args.seq_lens,
        targets=args.targets,
        feature_set=args.feature_set,
        max_features=args.max_features,
        stride=args.stride,
        split_level=args.split_level,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
        batch_size=args.batch_size,
        calibration_fracs=args.calibration_fracs,
        calibration_lrs=args.calibration_lrs,
        calibration_epochs=args.calibration_epochs,
        calibration_patience=args.calibration_patience,
        calibration_val_frac=args.calibration_val_frac,
        calibration_mode=args.calibration_mode,
        calibration_seed=args.calibration_seed,
        max_subjects=args.max_subjects,
        min_subject_sequences=args.min_subject_sequences,
        subject_selection=args.subject_selection,
        min_eval_sequences=args.min_eval_sequences,
        device=args.device,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )

    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    save_json(output_dir / "calibration_protocol_config.json", asdict(cfg))

    logger.info("Saved config: %s", output_dir / "calibration_protocol_config.json")

    if cfg.dry_run:
        logger.info("[dry-run] Would run calibration protocol sensitivity.")
        logger.info("seq_lens=%s", cfg.seq_lens)
        logger.info("calibration_lrs=%s", cfg.calibration_lrs)
        logger.info("calibration_fracs=%s", cfg.calibration_fracs)
        return

    set_seed(cfg.calibration_seed)

    seq_module = load_seq_module(root)

    all_per_subject: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    t0 = time.time()

    for seq_len in cfg.seq_lens:
        per_subject, predictions = run_protocol_for_seq_len(
            seq_module=seq_module,
            cfg=cfg,
            root=root,
            seq_len=seq_len,
            logger=logger,
        )
        if not per_subject.empty:
            all_per_subject.append(per_subject)
        if not predictions.empty:
            all_predictions.append(predictions)

    per_subject_df = pd.concat(all_per_subject, ignore_index=True) if all_per_subject else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    summary_df, gain_df = summarize_protocol(per_subject_df)

    if not per_subject_df.empty:
        path = output_dir / "protocol_per_subject_metrics.csv"
        per_subject_df.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    if not summary_df.empty:
        path = output_dir / "protocol_summary.csv"
        summary_df.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    if not gain_df.empty:
        path = output_dir / "protocol_gain_summary.csv"
        gain_df.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    if not predictions_df.empty:
        path = output_dir / "protocol_predictions.csv"
        predictions_df.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    build_report(output_dir, cfg, summary_df, gain_df)
    logger.info("Saved report: %s", output_dir / "calibration_protocol_report.md")

    elapsed = time.time() - t0
    logger.info("Done. Elapsed: %.1f sec", elapsed)


if __name__ == "__main__":
    main()