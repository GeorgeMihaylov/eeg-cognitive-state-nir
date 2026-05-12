#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized context-tabular baseline for EEG/POW window datasets.

Purpose:
    Check whether temporal context alone explains the gain observed with MHA.

Main comparison:
    X_t -> tabular baseline
    concat(X_{t-1}, X_t, X_{t+1}) -> tabular baseline
    [X_{t-1}, X_t, X_{t+1}] -> MHA

Optimizations:
    - --models allows running selected models, e.g. lgbm_reg,hgb_reg
    - vectorized context-window construction per record
    - --no-plots
    - --save-predictions false

Typical quick test:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\13_train_context_tabular_baselines.py `
      --dataset data\\processed\\windowed_eeg_pm_dataset_w10.parquet `
      --pm-target focus `
      --feature-set pow_plus_eeg `
      --feature-mode log_pow `
      --seq-len 3 `
      --max-samples 10000 `
      --fold-limit 2 `
      --fast `
      --models lgbm_reg,hgb_reg `
      --run-name context_tabular_focus_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


PM_TARGETS: Dict[str, str] = {
    "attention": "PM.Attention.Scaled__mean",
    "engagement": "PM.Engagement.Scaled__mean",
    "excitement": "PM.Excitement.Scaled__mean",
    "stress": "PM.Stress.Scaled__mean",
    "relaxation": "PM.Relaxation.Scaled__mean",
    "interest": "PM.Interest.Scaled__mean",
    "focus": "PM.Focus.Scaled__mean",
}

RANDOM_SEED = 42


@dataclass
class RunPaths:
    root: Path
    run_dir: Path
    figures_dir: Path
    logs_dir: Path


def sanitize_name(value: str) -> str:
    value = str(value).strip().replace(" ", "_")
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "run"


def setup_logging(run_dir: Path) -> logging.Logger:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("context_tabular")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def make_run_paths(root: Path, run_name: str, pm_target: str, feature_set: str, seq_len: int) -> RunPaths:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(run_name or "context_tabular")
    run_dir = root / "reports" / "runs" / f"{ts}_{safe_name}_{pm_target}_{feature_set}_len{seq_len}"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, run_dir=run_dir, figures_dir=figures_dir, logs_dir=logs_dir)


def resolve_path(root: Path, path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else root / p


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def is_pm_column(col: str) -> bool:
    return col.startswith("PM.") or col.startswith("pm_")


def is_target_or_label_column(col: str) -> bool:
    low = col.lower()
    return low.startswith("target") or low.startswith("label") or "_target" in low or "_label" in low


def is_metadata_column(col: str) -> bool:
    low = col.lower()
    exact = {
        "source", "subject_id", "record_id", "day", "part", "datetime_from_name",
        "path", "file_path", "relative_path", "t_center", "t_start", "t_end",
        "t_center_abs", "window_id", "_window_id_abs",
    }
    return low in exact or low.endswith("_path") or "datetime" in low


def is_leakage_column(col: str) -> bool:
    return is_pm_column(col) or is_target_or_label_column(col) or is_metadata_column(col)


def detect_pow_columns(df: pd.DataFrame) -> List[str]:
    band_tokens = ("theta", "alpha", "beta", "gamma", "delta", "lowbeta", "highbeta", "low_beta", "high_beta")
    cols = []
    for col in df.columns:
        if is_leakage_column(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        low = col.lower()
        has_pow_marker = (
            low.startswith("pow.") or low.startswith("pow_") or ".pow." in low
            or "__pow" in low or "bandpower" in low or "band_power" in low
        )
        has_band_marker = any(tok in low for tok in band_tokens)
        if has_pow_marker or has_band_marker:
            cols.append(col)
    return sorted(set(cols))


def detect_eeg_columns(df: pd.DataFrame, pow_cols: Optional[Sequence[str]] = None) -> List[str]:
    pow_set = set(pow_cols or [])
    channel_tokens = (
        "af3", "f7", "f3", "fc5", "t7", "p7", "o1", "o2", "p8", "t8", "fc6", "f4", "f8", "af4",
        "cz", "c3", "c4", "p3", "p4", "fz", "pz",
    )
    stat_tokens = ("__mean", "__std", "__min", "__max", "__median", "_mean", "_std", "_rms", "_skew", "_kurt", "_iqr")
    cols = []
    for col in df.columns:
        if col in pow_set or is_leakage_column(col) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        low = col.lower()
        eeg_marker = low.startswith("eeg.") or low.startswith("eeg_") or ".eeg." in low or "__eeg" in low or low.startswith("raw_eeg")
        channel_marker = any(token in low for token in channel_tokens)
        stat_marker = any(token in low for token in stat_tokens)
        if eeg_marker or (channel_marker and stat_marker):
            cols.append(col)
    return sorted(set(cols))


def select_feature_columns(df: pd.DataFrame, feature_set: str, logger: logging.Logger) -> Tuple[List[str], List[str], List[str]]:
    pow_cols = detect_pow_columns(df)
    eeg_cols = detect_eeg_columns(df, pow_cols=pow_cols)

    if feature_set == "pow":
        features = pow_cols
    elif feature_set == "eeg":
        features = eeg_cols
    elif feature_set == "pow_plus_eeg":
        features = list(dict.fromkeys(pow_cols + eeg_cols))
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if not features:
        features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and not is_leakage_column(col)]
        logger.warning("Fallback to all non-leakage numeric columns: %d", len(features))

    logger.info("Available POW feature columns: %d", len(pow_cols))
    logger.info("Available EEG feature columns: %d", len(eeg_cols))
    logger.info("Final feature columns: %d", len(features))
    return features, pow_cols, eeg_cols


def apply_feature_mode(df: pd.DataFrame, feature_cols: Sequence[str], pow_cols: Sequence[str], feature_mode: str) -> pd.DataFrame:
    out = df.loc[:, feature_cols].copy()
    if feature_mode == "raw_pow":
        return out
    if feature_mode == "log_pow":
        pow_set = set(pow_cols)
        for col in [c for c in feature_cols if c in pow_set]:
            arr = pd.to_numeric(out[col], errors="coerce").astype(float).to_numpy()
            finite = np.isfinite(arr)
            if finite.any() and np.nanmin(arr) < 0:
                arr = arr - np.nanmin(arr)
            out[col] = np.log1p(arr)
        return out
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def target_keys_from_arg(pm_target: str) -> List[str]:
    if pm_target == "all":
        return list(PM_TARGETS.keys())
    keys = [x.strip().lower() for x in pm_target.split(",") if x.strip()]
    unknown = [k for k in keys if k not in PM_TARGETS]
    if unknown:
        raise ValueError(f"Unknown PM target(s): {unknown}. Available: {sorted(PM_TARGETS)}")
    return keys


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")


def get_time_sort_column(df: pd.DataFrame) -> str:
    require_columns(df, ["record_id", "subject_id"])
    for col in ("t_center", "t_start", "t_center_abs", "window_id"):
        if col in df.columns:
            return col
    raise KeyError("No time/order column found. Expected one of: t_center, t_start, t_center_abs, window_id")


def build_context_dataset_fast(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    seq_len: int,
    max_time_gap_s: Optional[float],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    if seq_len <= 0 or seq_len % 2 == 0:
        raise ValueError("--seq-len must be a positive odd integer, e.g. 1, 3, 5, 7")

    require_columns(df, ["record_id", "subject_id", target_col])
    sort_col = get_time_sort_column(df)

    half = seq_len // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    context_feature_names = [f"ctx_{offset:+d}__{col}" for offset in offsets for col in feature_cols]

    feature_values = feature_matrix.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)
    target_values = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)

    center_indices_all: List[np.ndarray] = []
    context_indices_all: List[np.ndarray] = []
    skipped_nan_target = 0
    skipped_time_gap = 0
    skipped_short_records = 0

    for _, raw_idx in df.groupby("record_id", sort=False, dropna=False).groups.items():
        idx_arr = np.asarray(list(raw_idx), dtype=np.int64)
        if idx_arr.size < seq_len:
            skipped_short_records += 1
            continue

        order_values = pd.to_numeric(df.iloc[idx_arr][sort_col], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(order_values, kind="mergesort")
        idx_sorted = idx_arr[order]
        times_sorted = order_values[order]

        n = idx_sorted.size
        local_centers = np.arange(half, n - half, dtype=np.int64)
        if local_centers.size == 0:
            skipped_short_records += 1
            continue

        context_local = local_centers[:, None] + offsets[None, :]
        context_global = idx_sorted[context_local]
        center_global = idx_sorted[local_centers]

        valid = np.isfinite(target_values[center_global])
        skipped_nan_target += int((~valid).sum())

        if max_time_gap_s is not None and seq_len > 1:
            context_times = times_sorted[context_local]
            finite_times = np.isfinite(context_times).all(axis=1)
            gap_ok = np.ones(len(local_centers), dtype=bool)
            if finite_times.any():
                diffs = np.diff(context_times[finite_times], axis=1)
                tmp_indices = np.where(finite_times)[0]
                gap_ok[tmp_indices] = np.all(diffs <= max_time_gap_s, axis=1)
            valid = valid & gap_ok
            skipped_time_gap += int((~gap_ok).sum())

        if valid.any():
            center_indices_all.append(center_global[valid])
            context_indices_all.append(context_global[valid])

    if not center_indices_all:
        raise RuntimeError(f"No valid context samples built for target={target_col}")

    center_indices = np.concatenate(center_indices_all)
    context_indices = np.vstack(context_indices_all)

    X = feature_values[context_indices, :].reshape(context_indices.shape[0], -1)
    y = target_values[center_indices]

    meta_cols = [
        c for c in ["source", "subject_id", "record_id", "day", "part", "t_center", "t_start", "t_end", "datetime_from_name"]
        if c in df.columns
    ]
    meta = df.iloc[center_indices][meta_cols].reset_index(drop=True).copy()
    meta.insert(0, "center_original_index", center_indices.astype(int))
    meta.insert(1, "target_col", target_col)
    meta.insert(2, "seq_len", seq_len)

    logger.info(
        "Built context dataset for %s | samples=%d | X_shape=%s | skipped_nan_target=%d | skipped_time_gap=%d | skipped_short_records=%d",
        target_col, len(y), tuple(X.shape), skipped_nan_target, skipped_time_gap, skipped_short_records,
    )
    return X, y, meta, context_feature_names


def maybe_sample(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, max_samples: Optional[int], seed: int, logger: logging.Logger):
    if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
        return X, y, meta
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(y), size=max_samples, replace=False)
    selected.sort()
    logger.info("Sampled max_samples=%d from %d", max_samples, len(y))
    return X[selected], y[selected], meta.iloc[selected].reset_index(drop=True)


def make_all_models(fast: bool) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["ridge_robust"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])
    models["hgb_reg"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=100 if fast else 220,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    if HAS_LGBM:
        models["lgbm_reg"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                objective="regression",
                n_estimators=220 if fast else 500,
                learning_rate=0.045,
                num_leaves=31,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
            )),
        ])
    if not fast:
        models["rf_reg"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)),
        ])
    return models


def select_models(models_arg: str, fast: bool, logger: logging.Logger) -> Dict[str, object]:
    all_models = make_all_models(fast=fast)
    if models_arg.strip().lower() in ("all", ""):
        return all_models
    requested = [x.strip() for x in models_arg.split(",") if x.strip()]
    unknown = [m for m in requested if m not in all_models]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(all_models)}")
    selected = {name: all_models[name] for name in requested}
    logger.info("Selected models via --models: %s", list(selected.keys()))
    return selected


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    try:
        return float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    try:
        return float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def run_groupkfold(X, y, meta, models, target_key, target_col, args, logger):
    require_columns(meta, ["subject_id"])
    groups = meta["subject_id"].astype(str).to_numpy()
    n_splits = min(args.n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError(f"Not enough unique subjects for GroupKFold: {len(np.unique(groups))}")

    cv = GroupKFold(n_splits=n_splits)
    metric_rows = []
    pred_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        if args.fold_limit and fold_idx > args.fold_limit:
            break
        logger.info(
            "[%s] fold=%d/%d | n_train=%d | n_val=%d | train_subjects=%d | val_subjects=%d",
            target_key, fold_idx, n_splits, len(train_idx), len(val_idx),
            len(np.unique(groups[train_idx])), len(np.unique(groups[val_idx])),
        )
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for model_name, model_template in models.items():
            start = time.time()
            logger.info("[%s] fold=%d model=%s", target_key, fold_idx, model_name)
            model = clone(model_template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_val), dtype=float)
            metrics = compute_metrics(y_val, y_pred)
            elapsed = time.time() - start
            metric_rows.append({
                "task": "regression",
                "target": target_key,
                "target_col": target_col,
                "validation": "groupkfold_subject",
                "model": model_name,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "train_subjects": int(len(np.unique(groups[train_idx]))),
                "val_subjects": int(len(np.unique(groups[val_idx]))),
                "seq_len": int(args.seq_len),
                "feature_set": args.feature_set,
                "feature_mode": args.feature_mode,
                "elapsed_s": float(elapsed),
                **metrics,
            })
            if args.save_predictions:
                fold_meta = meta.iloc[val_idx].reset_index(drop=True).copy()
                fold_meta.insert(0, "target", target_key)
                fold_meta.insert(1, "model", model_name)
                fold_meta.insert(2, "fold", fold_idx)
                fold_meta["y_true"] = y_val
                fold_meta["y_pred"] = y_pred
                pred_rows.append(fold_meta)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return metrics_df, predictions_df


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    group_cols = ["task", "target", "target_col", "validation", "model", "seq_len", "feature_set", "feature_mode"]
    metric_cols = ["mae", "rmse", "r2", "pearson", "spearman"]
    rows = []
    for keys, grp in metrics_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(grp["fold"].nunique())
        row["n_val_total"] = int(grp["n_val"].sum())
        for metric in metric_cols:
            values = grp[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "r2_mean" in out.columns:
        out = out.sort_values(["target", "r2_mean", "spearman_mean"], ascending=[True, False, False])
    return out.reset_index(drop=True)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_simple_plots(run_paths: RunPaths, agg_df: pd.DataFrame, logger: logging.Logger) -> None:
    if agg_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Could not import matplotlib, skipping plots: %s", exc)
        return
    best = pd.DataFrame([grp.sort_values(["r2_mean", "spearman_mean"], ascending=False).iloc[0] for _, grp in agg_df.groupby("target", dropna=False)])
    for metric in ("r2_mean", "spearman_mean", "rmse_mean", "mae_mean"):
        if metric not in best.columns:
            continue
        plot_df = best.sort_values(metric, ascending=(metric in ("rmse_mean", "mae_mean")))
        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["target"].astype(str), plot_df[metric].astype(float))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Best model per target: {metric}")
        plt.tight_layout()
        out_path = run_paths.figures_dir / f"best_by_target_{metric}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        logger.info("Saved figure: %s", out_path)


def save_report(run_paths: RunPaths, args, dataset_info, feature_info, all_agg, logger):
    report_path = run_paths.run_dir / "report.md"
    lines = [
        "# Context-tabular baseline report", "",
        "## Run configuration", "", "```json", json.dumps(vars(args), ensure_ascii=False, indent=2), "```", "",
        "## Dataset info", "", "```json", json.dumps(dataset_info, ensure_ascii=False, indent=2), "```", "",
        "## Feature info", "", "```json", json.dumps(feature_info, ensure_ascii=False, indent=2), "```", "",
        "## Aggregated metrics", "",
    ]
    if not all_agg.empty:
        show_cols = ["target", "model", "folds", "n_val_total", "mae_mean", "rmse_mean", "r2_mean", "pearson_mean", "spearman_mean"]
        show_cols = [c for c in show_cols if c in all_agg.columns]
        lines.append(all_agg[show_cols].to_markdown(index=False))
    else:
        lines.append("No aggregated metrics.")
    lines.extend([
        "", "## Notes", "",
        "- Target PM columns are not used as input features.",
        "- `target_*`, `label_*`, metadata and `PM.*` columns are excluded from feature selection.",
        "- Context samples are built only inside the same `record_id`.",
        "- Validation uses `GroupKFold` by `subject_id`.",
        "- This is a control experiment for MHA: it tests temporal context without attention.", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved markdown report: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optimized context-tabular baselines on neighboring EEG windows.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="data/processed/windowed_eeg_pm_dataset_w10.parquet")
    parser.add_argument("--pm-target", type=str, default="all")
    parser.add_argument("--feature-set", type=str, default="pow_plus_eeg", choices=["pow", "eeg", "pow_plus_eeg"])
    parser.add_argument("--feature-mode", type=str, default="log_pow", choices=["log_pow", "raw_pow"])
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-time-gap-s", type=float, default=0.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--models", type=str, default="lgbm_reg,hgb_reg", help="Comma-separated model names or all.")
    parser.add_argument("--save-predictions", type=str_to_bool, default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-name", type=str, default="context_tabular")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_path = resolve_path(root, args.dataset)
    target_keys = target_keys_from_arg(args.pm_target)
    run_target_name = args.pm_target.replace(",", "_")
    run_paths = make_run_paths(root, args.run_name, run_target_name, args.feature_set, args.seq_len)
    logger = setup_logging(run_paths.run_dir)

    logger.info("=" * 80)
    logger.info("Context-tabular baseline optimized")
    logger.info("=" * 80)
    logger.info("Root: %s", root)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Run dir: %s", run_paths.run_dir)
    logger.info("PM targets: %s", target_keys)
    logger.info("Feature set: %s", args.feature_set)
    logger.info("Feature mode: %s", args.feature_mode)
    logger.info("Seq len: %s", args.seq_len)
    logger.info("Models arg: %s", args.models)

    df = read_dataset(dataset_path).reset_index(drop=True)
    dataset_info = {
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "records": int(df["record_id"].nunique()) if "record_id" in df.columns else None,
        "sources": df["source"].value_counts(dropna=False).to_dict() if "source" in df.columns else None,
    }
    logger.info("Dataset info: %s", dataset_info)

    feature_cols, pow_cols, eeg_cols = select_feature_columns(df, args.feature_set, logger)
    feature_df = apply_feature_mode(df, feature_cols, pow_cols, args.feature_mode)
    feature_info = {
        "feature_set": args.feature_set,
        "feature_mode": args.feature_mode,
        "pow_available": len(pow_cols),
        "eeg_available": len(eeg_cols),
        "features_used": len(feature_cols),
        "context_features_used": len(feature_cols) * args.seq_len,
        "seq_len": args.seq_len,
    }
    logger.info("Feature info: %s", feature_info)

    models = select_models(args.models, fast=args.fast, logger=logger)
    logger.info("Regression models: %s", list(models.keys()))

    all_metrics_list = []
    all_predictions_list = []
    max_time_gap_s = args.max_time_gap_s if args.max_time_gap_s and args.max_time_gap_s > 0 else None

    for target_key in target_keys:
        target_col = PM_TARGETS[target_key]
        if target_col not in df.columns:
            logger.warning("Skipping target=%s because column is missing: %s", target_key, target_col)
            continue
        logger.info("-" * 80)
        logger.info("Target: %s -> %s", target_key, target_col)

        start_build = time.time()
        X, y, meta, _ = build_context_dataset_fast(df, feature_df, feature_cols, target_col, args.seq_len, max_time_gap_s, logger)
        logger.info("Context build elapsed: %.2f s", time.time() - start_build)

        X, y, meta = maybe_sample(X, y, meta, args.max_samples, args.seed, logger)
        target_info = {
            "target": target_key,
            "target_col": target_col,
            "samples": int(len(y)),
            "target_mean": float(np.nanmean(y)),
            "target_std": float(np.nanstd(y)),
            "target_min": float(np.nanmin(y)),
            "target_median": float(np.nanmedian(y)),
            "target_max": float(np.nanmax(y)),
            "subjects": int(meta["subject_id"].nunique()) if "subject_id" in meta.columns else None,
            "records": int(meta["record_id"].nunique()) if "record_id" in meta.columns else None,
            "sources": meta["source"].value_counts(dropna=False).to_dict() if "source" in meta.columns else None,
        }
        save_json(run_paths.run_dir / f"{target_key}_context_dataset_info.json", target_info)
        logger.info("Target dataset info: %s", target_info)

        metrics_df, predictions_df = run_groupkfold(X, y, meta, models, target_key, target_col, args, logger)
        target_metrics_path = run_paths.run_dir / f"{target_key}_fold_metrics.csv"
        metrics_df.to_csv(target_metrics_path, index=False, encoding="utf-8")
        logger.info("Saved target metrics: %s", target_metrics_path)

        if args.save_predictions and not predictions_df.empty:
            target_predictions_path = run_paths.run_dir / f"{target_key}_predictions.parquet"
            predictions_df.to_parquet(target_predictions_path, index=False)
            logger.info("Saved target predictions: %s", target_predictions_path)
            all_predictions_list.append(predictions_df)
        all_metrics_list.append(metrics_df)

    if not all_metrics_list:
        raise RuntimeError("No targets were processed.")

    all_metrics = pd.concat(all_metrics_list, ignore_index=True)
    all_agg = aggregate_metrics(all_metrics)
    metrics_path = run_paths.run_dir / "all_targets_fold_metrics.csv"
    agg_path = run_paths.run_dir / "all_targets_summary.csv"
    feature_path = run_paths.run_dir / "feature_columns.json"
    config_path = run_paths.run_dir / "config.json"
    all_metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    all_agg.to_csv(agg_path, index=False, encoding="utf-8")

    if args.save_predictions and all_predictions_list:
        pred_path = run_paths.run_dir / "all_targets_predictions.parquet"
        pd.concat(all_predictions_list, ignore_index=True).to_parquet(pred_path, index=False)
        logger.info("Predictions: %s", pred_path)

    save_json(feature_path, {"feature_columns": list(feature_cols), "pow_columns": list(pow_cols), "eeg_columns": list(eeg_cols), "feature_info": feature_info})
    save_json(config_path, vars(args))

    if not args.no_plots:
        create_simple_plots(run_paths, all_agg, logger)
    save_report(run_paths, args, dataset_info, feature_info, all_agg, logger)

    logger.info("=" * 80)
    logger.info("Saved outputs")
    logger.info("=" * 80)
    logger.info("Run dir: %s", run_paths.run_dir)
    logger.info("All fold metrics: %s", metrics_path)
    logger.info("Summary: %s", agg_path)
    logger.info("")
    logger.info("Aggregated metrics:")
    show_cols = ["target", "model", "folds", "n_val_total", "mae_mean", "rmse_mean", "r2_mean", "pearson_mean", "spearman_mean"]
    show_cols = [c for c in show_cols if c in all_agg.columns]
    if show_cols:
        logger.info("\n%s", all_agg[show_cols].to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
