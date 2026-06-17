#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Naive hypothesis baselines for EEG latent proxy-state modeling.

This script is a lightweight sanity-check block for the integrated baseline v1.
It does not train neural networks. It computes simple statistical and
persistence baselines for PM-derived latent targets:

  - train_mean: constant prediction equal to the train target mean;
  - subject_calibration_mean: constant prediction equal to the mean of the
    first calibration fraction of a held-out subject;
  - subject_calibration_last: constant prediction equal to the last target
    value in the calibration fraction of a held-out subject;
  - previous_state: persistence baseline, predicting the current target by the
    previous true target within the same subject and record.

Default final protocol:
  feature_set = pow_plus_eeg   # kept for traceability; input features are not used
  seq_len = 8
  targets = slow_pca_1,slow_pca_2,slow_pca_3
  split_level = subject
  seeds = 42,123,2024,3407,777
  calibration_frac = 0.20

Recommended command:
D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\51_run_naive_hypothesis_baselines.py `
  --root . `
  --dataset reports\\slow_latent_states\\pm_w10\\slow_pm_latent_states_w10.parquet `
  --output-dir reports\\naive_hypothesis_baselines\\pow_plus_eeg_seq8_pca123 `
  --seeds 42,123,2024,3407,777 `
  --feature-set pow_plus_eeg `
  --targets slow_pca_1,slow_pca_2,slow_pca_3 `
  --seq-len 8 `
  --calibration-frac 0.20

Outputs:
  naive_baseline_config.json
  naive_baseline_summary.csv
  naive_baseline_by_target.csv
  aggregate_naive_baseline_summary.csv
  aggregate_naive_baseline_by_target.csv
  subject_tail_counts.csv
  naive_baseline_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"
DEFAULT_SEEDS = "42,123,2024,3407,777"


@dataclass
class Config:
    root: str
    dataset: str
    output_dir: str
    seeds: list[int]
    feature_set: str
    targets: list[str]
    seq_len: int
    stride: int
    split_level: str
    train_size: float
    val_size: float
    test_size: float
    calibration_frac: float
    max_subjects: int | None
    min_subject_sequences: int
    min_eval_sequences: int
    subject_selection: str


def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated integer list")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated string list")
    return out


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def detect_id_columns(df: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "source": ["source", "dataset", "data_source"],
        "subject_id": ["subject_id", "subject", "participant", "participant_id", "user_id"],
        "record_id": ["record_id", "record", "session", "session_id", "file_id", "trial_id"],
        "window_start": ["t_start", "window_start", "start_time", "start", "time_start"],
        "window_end": ["t_end", "window_end", "end_time", "end", "time_end"],
    }

    lower_to_col = {str(c).lower(): str(c) for c in df.columns}
    found: dict[str, str] = {}

    for key, names in candidates.items():
        for name in names:
            if name.lower() in lower_to_col:
                found[key] = lower_to_col[name.lower()]
                break

    if "source" not in found:
        df["source"] = "unknown_source"
        found["source"] = "source"

    if "subject_id" not in found:
        raise ValueError("Could not detect subject column. Expected subject_id/subject/participant/user_id.")

    if "record_id" not in found:
        df["record_id"] = "record_0"
        found["record_id"] = "record_id"

    if "window_start" not in found:
        df["_row_order"] = np.arange(len(df), dtype=np.int64)
        found["window_start"] = "_row_order"

    if "window_end" not in found:
        found["window_end"] = found["window_start"]

    return found


def build_target_sequences(
    df: pd.DataFrame,
    target_cols: list[str],
    id_cols: dict[str, str],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in dataset: {missing}")

    group_cols = [id_cols["source"], id_cols["subject_id"], id_cols["record_id"]]
    sort_col = id_cols["window_start"]

    df2 = df.sort_values(group_cols + [sort_col]).reset_index(drop=True)
    y_parts: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    global_order = 0

    for group_key, g in df2.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values(sort_col).reset_index(drop=False)
        if len(g) < seq_len:
            continue
        yg = g[target_cols].to_numpy(dtype=np.float32, copy=True)
        group_order = 0
        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            y_last = yg[end - 1]
            if np.any(~np.isfinite(y_last)):
                continue

            first_row = g.iloc[start]
            last_row = g.iloc[end - 1]
            y_parts.append(y_last)
            meta_rows.append(
                {
                    "source": str(last_row[id_cols["source"]]),
                    "subject_id": str(last_row[id_cols["subject_id"]]),
                    "record_id": str(last_row[id_cols["record_id"]]),
                    "group_key": "|".join(str(x) for x in group_key),
                    "sequence_start": first_row[sort_col],
                    "sequence_end": last_row[sort_col],
                    "group_seq_order": int(group_order),
                    "global_seq_order": int(global_order),
                }
            )
            group_order += 1
            global_order += 1

    if not y_parts:
        raise ValueError(f"No target sequences created for seq_len={seq_len}")

    return np.stack(y_parts).astype(np.float32), pd.DataFrame(meta_rows)


def split_indices(
    meta: pd.DataFrame,
    split_level: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"train_size + val_size + test_size must be 1.0, got {total}")

    if split_level == "subject":
        split_key = "subject_id"
    elif split_level == "record":
        split_key = "group_key"
    elif split_level == "sequence":
        rng = np.random.default_rng(seed)
        all_idx = np.arange(len(meta), dtype=np.int64)
        rng.shuffle(all_idx)
        n_train = int(round(len(all_idx) * train_size))
        n_val = int(round(len(all_idx) * val_size))
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train : n_train + n_val]
        test_idx = all_idx[n_train + n_val :]
        return train_idx, val_idx, test_idx, {"split_level": split_level}
    else:
        raise ValueError(f"Unsupported split_level: {split_level}")

    groups = np.asarray(meta[split_key].astype(str).dropna().unique().tolist(), dtype=object)
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 groups for {split_level}-wise split, got {len(groups)}")

    train_groups, temp_groups = train_test_split(groups, train_size=train_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (val_size + test_size)
    val_groups, test_groups = train_test_split(
        np.asarray(temp_groups, dtype=object), train_size=relative_val_size, random_state=seed + 1, shuffle=True
    )

    values = meta[split_key].astype(str).to_numpy(dtype=object)
    train_idx = np.flatnonzero(np.isin(values, np.asarray(train_groups, dtype=object))).astype(np.int64)
    val_idx = np.flatnonzero(np.isin(values, np.asarray(val_groups, dtype=object))).astype(np.int64)
    test_idx = np.flatnonzero(np.isin(values, np.asarray(test_groups, dtype=object))).astype(np.int64)

    split_meta = {
        "split_level": split_level,
        "split_key": split_key,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": int(len(train_groups)),
        "n_groups_val": int(len(val_groups)),
        "n_groups_test": int(len(test_groups)),
    }
    return train_idx, val_idx, test_idx, split_meta


def finite_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.isfinite(y_true) & np.isfinite(y_pred)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 2:
        return float("nan")
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    if denom <= 0:
        return float("nan")
    num = float(np.sum((yt - yp) ** 2))
    return 1.0 - num / denom


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def spearman_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 2 or len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
        return float("nan")
    val = pd.Series(yt).corr(pd.Series(yp), method="spearman")
    return float(val) if pd.notna(val) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    rows: list[dict[str, Any]] = []
    for j, target in enumerate(targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append(
            {
                "target": target,
                "n_eval": int(np.sum(finite_mask(yt, yp))),
                "r2": r2_score_np(yt, yp),
                "spearman": spearman_np(yt, yp),
                "mae": mae_np(yt, yp),
                "rmse": rmse_np(yt, yp),
            }
        )

    by_target = pd.DataFrame(rows)
    overall = {
        "n_eval": int(y_true.shape[0]),
        "mean_r2": float(pd.to_numeric(by_target["r2"], errors="coerce").mean()),
        "mean_spearman": float(pd.to_numeric(by_target["spearman"], errors="coerce").mean()),
        "mean_mae": float(pd.to_numeric(by_target["mae"], errors="coerce").mean()),
        "mean_rmse": float(pd.to_numeric(by_target["rmse"], errors="coerce").mean()),
    }
    return overall, rows


def constant_pred(value: np.ndarray, n: int) -> np.ndarray:
    return np.tile(value.reshape(1, -1), (n, 1)).astype(np.float32)


def append_metrics(
    summary_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    seed: int,
    eval_split: str,
    phase: str,
    baseline: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    calibration_frac: float,
    uses_target_history: bool,
    notes: str,
) -> None:
    if len(y_true) == 0:
        return

    overall, by_target = regression_metrics(y_true, y_pred, targets)
    summary_rows.append(
        {
            "seed": seed,
            "eval_split": eval_split,
            "phase": phase,
            "baseline": baseline,
            "calibration_frac": calibration_frac,
            "uses_target_history": bool(uses_target_history),
            **overall,
            "notes": notes,
        }
    )

    for row in by_target:
        target_rows.append(
            {
                "seed": seed,
                "eval_split": eval_split,
                "phase": phase,
                "baseline": baseline,
                "calibration_frac": calibration_frac,
                "uses_target_history": bool(uses_target_history),
                **row,
                "notes": notes,
            }
        )


def previous_state_pairs(
    meta: pd.DataFrame,
    y: np.ndarray,
    target_idx: np.ndarray,
    context_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_set = set(int(x) for x in target_idx)
    context_set = set(int(x) for x in context_idx)
    allowed = sorted(target_set.union(context_set))
    if not allowed:
        return np.empty((0, y.shape[1]), dtype=np.float32), np.empty((0, y.shape[1]), dtype=np.float32)

    m = meta.iloc[allowed].copy()
    m["_orig_idx"] = allowed
    m = m.sort_values(["subject_id", "record_id", "group_seq_order", "global_seq_order"])

    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    for _, g in m.groupby(["subject_id", "record_id"], sort=False):
        prev_idx: int | None = None
        for cur_idx in [int(x) for x in g["_orig_idx"].tolist()]:
            if cur_idx in target_set and prev_idx is not None:
                true_parts.append(y[cur_idx])
                pred_parts.append(y[prev_idx])
            if cur_idx in context_set or cur_idx in target_set:
                prev_idx = cur_idx

    if not true_parts:
        return np.empty((0, y.shape[1]), dtype=np.float32), np.empty((0, y.shape[1]), dtype=np.float32)

    return np.stack(true_parts).astype(np.float32), np.stack(pred_parts).astype(np.float32)


def select_calibration_subjects(
    meta: pd.DataFrame,
    split_idx: np.ndarray,
    max_subjects: int | None,
    min_subject_sequences: int,
    selection: str,
    seed: int,
) -> list[str]:
    split_meta = meta.iloc[split_idx].copy()
    counts = split_meta["subject_id"].astype(str).value_counts()
    counts = counts[counts >= min_subject_sequences]
    if counts.empty:
        return []

    if selection == "largest":
        subjects = counts.sort_values(ascending=False).index.tolist()
    elif selection == "random":
        rng = np.random.default_rng(seed)
        arr = counts.index.to_numpy(dtype=object)
        rng.shuffle(arr)
        subjects = [str(x) for x in arr]
    else:
        subjects = sorted(str(x) for x in counts.index.tolist())

    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]
    return [str(x) for x in subjects]


def calibration_tail(subject_idx: np.ndarray, frac: float, min_eval_sequences: int) -> tuple[np.ndarray, np.ndarray]:
    subject_idx = np.asarray(subject_idx, dtype=np.int64)
    n = len(subject_idx)
    if n <= min_eval_sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if frac <= 0:
        return np.array([], dtype=np.int64), subject_idx

    cal_n = max(1, int(math.floor(n * frac)))
    cal_n = min(cal_n, n - min_eval_sequences)
    if cal_n <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cal_idx = subject_idx[:cal_n]
    tail_idx = subject_idx[cal_n:]
    if len(tail_idx) < min_eval_sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return cal_idx.astype(np.int64), tail_idx.astype(np.int64)


def compute_one_seed(
    cfg: Config,
    y: np.ndarray,
    meta: pd.DataFrame,
    seed: int,
    summary_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
) -> None:
    train_idx, val_idx, test_idx, _ = split_indices(
        meta=meta,
        split_level=cfg.split_level,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        seed=seed,
    )

    train_mean = np.nanmean(y[train_idx], axis=0).astype(np.float32)

    for split_name, split_idx in {"val": val_idx, "test": test_idx}.items():
        split_idx = np.asarray(split_idx, dtype=np.int64)

        # Full split: global train mean.
        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "zero_full",
            "train_mean",
            y[split_idx],
            constant_pred(train_mean, len(split_idx)),
            cfg.targets,
            0.0,
            False,
            "Constant prediction fitted only on train targets.",
        )

        # Full split: persistence inside each held-out subject/record.
        yt_prev, yp_prev = previous_state_pairs(meta, y, target_idx=split_idx, context_idx=split_idx)
        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "zero_full",
            "previous_state",
            yt_prev,
            yp_prev,
            cfg.targets,
            0.0,
            True,
            "Persistence baseline; uses previous true target within the same subject and record.",
        )

        # Tail after subject-specific calibration prefix.
        subjects = select_calibration_subjects(
            meta,
            split_idx,
            cfg.max_subjects,
            cfg.min_subject_sequences,
            cfg.subject_selection,
            seed,
        )

        tail_train_mean_true: list[np.ndarray] = []
        tail_train_mean_pred: list[np.ndarray] = []
        tail_subject_mean_true: list[np.ndarray] = []
        tail_subject_mean_pred: list[np.ndarray] = []
        tail_subject_last_true: list[np.ndarray] = []
        tail_subject_last_pred: list[np.ndarray] = []
        tail_prev_true: list[np.ndarray] = []
        tail_prev_pred: list[np.ndarray] = []

        split_meta = meta.iloc[split_idx].copy()
        split_meta["_seq_idx"] = split_idx

        for subject in subjects:
            sm = split_meta[split_meta["subject_id"].astype(str) == str(subject)].copy()
            sm = sm.sort_values(["source", "record_id", "sequence_start", "sequence_end", "global_seq_order"])
            subject_indices = sm["_seq_idx"].to_numpy(dtype=np.int64)
            cal_idx, tail_idx = calibration_tail(subject_indices, cfg.calibration_frac, cfg.min_eval_sequences)
            if len(cal_idx) == 0 or len(tail_idx) == 0:
                continue

            cal_mean = np.nanmean(y[cal_idx], axis=0).astype(np.float32)
            cal_last = y[cal_idx[-1]].astype(np.float32)

            tail_train_mean_true.append(y[tail_idx])
            tail_train_mean_pred.append(constant_pred(train_mean, len(tail_idx)))

            tail_subject_mean_true.append(y[tail_idx])
            tail_subject_mean_pred.append(constant_pred(cal_mean, len(tail_idx)))

            tail_subject_last_true.append(y[tail_idx])
            tail_subject_last_pred.append(constant_pred(cal_last, len(tail_idx)))

            yt_tail_prev, yp_tail_prev = previous_state_pairs(
                meta,
                y,
                target_idx=tail_idx,
                context_idx=np.concatenate([cal_idx, tail_idx]),
            )
            if len(yt_tail_prev):
                tail_prev_true.append(yt_tail_prev)
                tail_prev_pred.append(yp_tail_prev)

            subject_rows.append(
                {
                    "seed": seed,
                    "eval_split": split_name,
                    "subject_id": subject,
                    "n_subject_sequences": int(len(subject_indices)),
                    "n_calibration_sequences": int(len(cal_idx)),
                    "n_eval_tail_sequences": int(len(tail_idx)),
                    "calibration_frac": cfg.calibration_frac,
                }
            )

        def cat(parts: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts, axis=0) if parts else np.empty((0, y.shape[1]), dtype=np.float32)

        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "post_calibration_tail",
            "train_mean",
            cat(tail_train_mean_true),
            cat(tail_train_mean_pred),
            cfg.targets,
            cfg.calibration_frac,
            False,
            "Train-mean baseline evaluated on the same tail as calibration baselines.",
        )
        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "post_calibration_tail",
            "subject_calibration_mean",
            cat(tail_subject_mean_true),
            cat(tail_subject_mean_pred),
            cfg.targets,
            cfg.calibration_frac,
            False,
            "Predicts the tail by mean latent state from the first calibration fraction of the subject.",
        )
        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "post_calibration_tail",
            "subject_calibration_last",
            cat(tail_subject_last_true),
            cat(tail_subject_last_pred),
            cfg.targets,
            cfg.calibration_frac,
            False,
            "Predicts the tail by the last latent state from the calibration fraction.",
        )
        append_metrics(
            summary_rows,
            target_rows,
            seed,
            split_name,
            "post_calibration_tail",
            "previous_state",
            cat(tail_prev_true),
            cat(tail_prev_pred),
            cfg.targets,
            cfg.calibration_frac,
            True,
            "Persistence baseline on the post-calibration tail; uses previous true target.",
        )


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    metric_cols = ["mean_r2", "mean_spearman", "mean_mae", "mean_rmse", "n_eval"]
    group_cols = ["eval_split", "phase", "baseline", "calibration_frac", "uses_target_history"]
    rows: list[dict[str, Any]] = []

    for keys, g in summary.groupby(group_cols, sort=True, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["eval_split", "phase", "mean_r2_mean"], ascending=[True, True, False])


def aggregate_by_target(by_target: pd.DataFrame) -> pd.DataFrame:
    if by_target.empty:
        return pd.DataFrame()

    metric_cols = ["r2", "spearman", "mae", "rmse", "n_eval"]
    group_cols = ["eval_split", "phase", "baseline", "target", "calibration_frac", "uses_target_history"]
    rows: list[dict[str, Any]] = []

    for keys, g in by_target.groupby(group_cols, sort=True, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["eval_split", "phase", "target", "r2_mean"], ascending=[True, True, True, False])


def fmt_val(x: Any, digits: int = 4) -> str:
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

    def esc(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows: list[list[str]] = []
    rows.append([esc(c) for c in cols])
    rows.append(["---" for _ in cols])
    for _, row in view.iterrows():
        rows.append([esc(fmt_val(row[c], digits=digits)) for c in cols])
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    return "\n".join("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows)


def build_report(cfg: Config, y_shape: tuple[int, int], aggregate: pd.DataFrame, aggregate_target: pd.DataFrame, subject_counts: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Naive hypothesis baselines report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Compute simple statistical and persistence baselines for the final latent proxy-state setup. "
        "These baselines close the H6 sanity-check block in baseline v1."
    )
    lines.append("")

    lines.append("## Fixed setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| feature_set | `{cfg.feature_set}` |")
    lines.append(f"| seq_len | `{cfg.seq_len}` |")
    lines.append(f"| targets | `{', '.join(cfg.targets)}` |")
    lines.append(f"| split_level | `{cfg.split_level}` |")
    lines.append(f"| seeds | `{', '.join(str(s) for s in cfg.seeds)}` |")
    lines.append(f"| calibration_frac | `{cfg.calibration_frac}` |")
    lines.append(f"| sequences | `{y_shape[0]}` |")
    lines.append("")

    lines.append("## Aggregate summary across seeds")
    lines.append("")
    if aggregate.empty:
        lines.append("_No aggregate data._")
    else:
        cols = [
            "eval_split", "phase", "baseline", "uses_target_history", "n_seeds",
            "mean_r2_mean", "mean_r2_std", "mean_r2_min", "mean_r2_max",
            "mean_spearman_mean", "mean_mae_mean", "mean_rmse_mean", "n_eval_mean",
        ]
        cols = [c for c in cols if c in aggregate.columns]
        lines.append(df_to_markdown(aggregate[cols], digits=4))
    lines.append("")

    lines.append("## Test-only compact summary")
    lines.append("")
    if aggregate.empty:
        lines.append("_No test data._")
    else:
        test = aggregate[aggregate["eval_split"].astype(str) == "test"].copy()
        cols = [
            "phase", "baseline", "uses_target_history", "mean_r2_mean", "mean_r2_std",
            "mean_spearman_mean", "mean_mae_mean", "mean_rmse_mean", "n_eval_mean",
        ]
        cols = [c for c in cols if c in test.columns]
        lines.append(df_to_markdown(test[cols], digits=4))
    lines.append("")

    lines.append("## Test target-level summary")
    lines.append("")
    if aggregate_target.empty:
        lines.append("_No target-level data._")
    else:
        test_t = aggregate_target[aggregate_target["eval_split"].astype(str) == "test"].copy()
        cols = [
            "phase", "baseline", "target", "uses_target_history", "r2_mean", "r2_std",
            "spearman_mean", "mae_mean", "rmse_mean",
        ]
        cols = [c for c in cols if c in test_t.columns]
        lines.append(df_to_markdown(test_t[cols], digits=4, max_rows=120))
    lines.append("")

    lines.append("## Subject tail counts")
    lines.append("")
    if subject_counts.empty:
        lines.append("_No subject tail counts._")
    else:
        count_summary = (
            subject_counts.groupby(["seed", "eval_split"], as_index=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                mean_calibration_sequences=("n_calibration_sequences", "mean"),
                mean_eval_tail_sequences=("n_eval_tail_sequences", "mean"),
                min_eval_tail_sequences=("n_eval_tail_sequences", "min"),
            )
            .sort_values(["eval_split", "seed"])
        )
        lines.append(df_to_markdown(count_summary, digits=2))
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `train_mean` checks whether a global constant baseline is enough.")
    lines.append("- `subject_calibration_mean` checks whether the model only needs a subject-specific offset from the calibration prefix.")
    lines.append("- `subject_calibration_last` checks whether the last calibration state is enough.")
    lines.append("- `previous_state` is a strong persistence sanity-check and uses target history; it is not an EEG-only deployable model.")
    lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run naive hypothesis baselines for latent EEG proxy-states.")
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet")
    p.add_argument("--output-dir", default="reports/naive_hypothesis_baselines/pow_plus_eeg_seq8_pca123")
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list(DEFAULT_SEEDS))
    p.add_argument("--feature-set", default="pow_plus_eeg")
    p.add_argument("--targets", type=parse_str_list, default=parse_str_list(DEFAULT_TARGETS))
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--split-level", choices=["subject", "record", "sequence"], default="subject")
    p.add_argument("--train-size", type=float, default=0.70)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--calibration-frac", type=float, default=0.20)
    p.add_argument("--max-subjects", type=int, default=30)
    p.add_argument("--min-subject-sequences", type=int, default=80)
    p.add_argument("--min-eval-sequences", type=int, default=20)
    p.add_argument("--subject-selection", choices=["largest", "random", "sorted"], default="largest")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = Config(**vars(args))

    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "naive_baseline_config.json", asdict(cfg))

    dataset_path = repo_path(root, cfg.dataset)
    print("=" * 100)
    print(f"Reading dataset: {dataset_path}")
    print("=" * 100)
    df = read_table(dataset_path)
    id_cols = detect_id_columns(df)

    print("=" * 100)
    print("Building target sequences")
    print("=" * 100)
    y, meta = build_target_sequences(df, cfg.targets, id_cols, cfg.seq_len, cfg.stride)
    print(f"Target sequences: y={y.shape}, meta={meta.shape}")

    summary_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        print("\n" + "#" * 100)
        print(f"SEED {seed}")
        print("#" * 100)
        compute_one_seed(cfg, y, meta, seed, summary_rows, target_rows, subject_rows)

    summary = pd.DataFrame(summary_rows)
    by_target = pd.DataFrame(target_rows)
    subject_counts = pd.DataFrame(subject_rows)
    aggregate = aggregate_summary(summary)
    aggregate_target = aggregate_by_target(by_target)

    summary.to_csv(output_dir / "naive_baseline_summary.csv", index=False)
    by_target.to_csv(output_dir / "naive_baseline_by_target.csv", index=False)
    subject_counts.to_csv(output_dir / "subject_tail_counts.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_naive_baseline_summary.csv", index=False)
    aggregate_target.to_csv(output_dir / "aggregate_naive_baseline_by_target.csv", index=False)

    report = build_report(cfg, y.shape, aggregate, aggregate_target, subject_counts)
    (output_dir / "naive_baseline_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 100)
    print("Saved outputs")
    print("=" * 100)
    for name in [
        "naive_baseline_config.json",
        "naive_baseline_summary.csv",
        "naive_baseline_by_target.csv",
        "aggregate_naive_baseline_summary.csv",
        "aggregate_naive_baseline_by_target.csv",
        "subject_tail_counts.csv",
        "naive_baseline_report.md",
    ]:
        print(output_dir / name)
    print("=" * 100)


if __name__ == "__main__":
    main()
