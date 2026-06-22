#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train temporal baselines for EEG latent trajectory modeling.

Goal
----
Compare whether Transformer is actually useful against simpler temporal baselines
under the final project setup:
  - feature_set = pow_plus_eeg
  - seq_len = 8
  - targets = slow_pca_1, slow_pca_2, slow_pca_3
  - subject-wise train / validation / test split
  - optional subject-specific head-only calibration

Models
------
  - last_window_mlp: MLP on only the last window in the sequence
  - mean_pool_mlp: MLP on the mean-pooled sequence features
  - gru: recurrent temporal baseline
  - transformer: TransformerEncoder model, compatible with script 44 architecture

Recommended command
-------------------
D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\48_train_temporal_baselines.py `
  --root . `
  --dataset reports\\slow_latent_states\\pm_w10\\slow_pm_latent_states_w10.parquet `
  --output-dir reports\\temporal_baselines\\pow_plus_eeg_seq8_pca123 `
  --models last_window_mlp,mean_pool_mlp,gru,transformer `
  --feature-set pow_plus_eeg `
  --targets slow_pca_1,slow_pca_2,slow_pca_3 `
  --seq-len 8 `
  --calibration-lr 0.0001 `
  --calibration-frac 0.20 `
  --device cuda
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required") from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"
DEFAULT_MODELS = "last_window_mlp,mean_pool_mlp,gru,transformer"


@dataclass
class BaselineConfig:
    root: str
    dataset: str
    output_dir: str
    models: list[str]
    feature_set: str
    max_features: int
    targets: list[str]
    seq_len: int
    stride: int
    split_level: str
    train_size: float
    val_size: float
    test_size: float
    random_state: int
    hidden_dim: int
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    pooling: str
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    calibration_lr: float
    calibration_frac: float
    calibration_epochs: int
    calibration_patience: int
    calibration_val_frac: float
    max_subjects: int | None
    min_subject_sequences: int
    min_eval_sequences: int
    subject_selection: str
    device: str
    reuse_existing: bool
    no_calibration: bool
    dry_run: bool


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LastWindowMLP(nn.Module):
    def __init__(self, n_features: int, n_targets: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, -1, :])


class MeanPoolMLP(nn.Module):
    def __init__(self, n_features: int, n_targets: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=1))


class GRURegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def parse_csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated non-empty string list")
    return out


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("temporal_baselines")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required module is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def call_select_feature_columns(seq_module, df: pd.DataFrame, cfg: BaselineConfig, logger: logging.Logger) -> list[str]:
    fn = seq_module.select_feature_columns
    sig = inspect.signature(fn)
    if "feature_set" in sig.parameters:
        return fn(df, cfg.targets, cfg.max_features, logger, feature_set=cfg.feature_set)

    logger.warning(
        "select_feature_columns does not accept feature_set. "
        "This means feature ablation may be ignored. Use patched script 44."
    )
    return fn(df, cfg.targets, cfg.max_features, logger)


def make_loader(x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SequenceDataset(x[idx], y[idx]), batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_model(model: nn.Module, x: np.ndarray, idx: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    if len(idx) == 0:
        return np.empty((0, 0), dtype=np.float32)
    dummy_y = np.zeros((len(idx), 1), dtype=np.float32)
    loader = DataLoader(SequenceDataset(x[idx], dummy_y), batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    preds: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.empty((0, 0), dtype=np.float32)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str], prefix: dict | None = None) -> pd.DataFrame:
    prefix = prefix or {}
    rows: list[dict] = []
    for j, target in enumerate(targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append(
            {
                **prefix,
                "target": target,
                "n": int(len(yt)),
                "mae": float(mean_absolute_error(yt, yp)) if len(yt) else float("nan"),
                "rmse": float(math.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan"),
                "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
                "spearman": safe_spearman(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def mean_metric_summary(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in metrics.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_targets": int(g["target"].nunique()) if "target" in g.columns else int(len(g)),
                "mean_r2": float(pd.to_numeric(g["r2"], errors="coerce").mean()),
                "mean_spearman": float(pd.to_numeric(g["spearman"], errors="coerce").mean()),
                "mean_mae": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
                "mean_rmse": float(pd.to_numeric(g["rmse"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def train_base_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: BaselineConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[nn.Module, pd.DataFrame, dict]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = make_loader(x, y, train_idx, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x, y, val_idx, cfg.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    patience_left = cfg.patience
    history_rows: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("epoch=%03d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch=%d; best_epoch=%d best_val=%.6f", epoch, best_epoch, best_val)
                break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    return model, history, {"best_epoch": int(best_epoch), "best_val_loss": float(best_val)}


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def build_model(model_name: str, n_features: int, n_targets: int, seq_len: int, cfg: BaselineConfig, seq_module) -> nn.Module:
    if model_name == "last_window_mlp":
        return LastWindowMLP(n_features=n_features, n_targets=n_targets, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if model_name == "mean_pool_mlp":
        return MeanPoolMLP(n_features=n_features, n_targets=n_targets, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if model_name == "gru":
        return GRURegressor(
            n_features=n_features,
            n_targets=n_targets,
            hidden_dim=cfg.hidden_dim,
            num_layers=max(1, cfg.num_layers),
            dropout=cfg.dropout,
        )
    if model_name == "transformer":
        return seq_module.TransformerRegressor(
            n_features=n_features,
            n_targets=n_targets,
            seq_len=seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def get_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    if hasattr(model, "regression_head"):
        return list(model.regression_head.parameters())  # type: ignore[attr-defined]
    if hasattr(model, "head"):
        return list(model.head.parameters())  # type: ignore[attr-defined]
    raise ValueError(f"Model {type(model).__name__} has no recognized head parameters")


def freeze_head_only(model: nn.Module) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    head_params = get_head_parameters(model)
    for p in head_params:
        p.requires_grad = True
    return [p for p in head_params if p.requires_grad]


def subject_indices_for_split(meta: pd.DataFrame, split_idx: np.ndarray, subject_id: str) -> np.ndarray:
    split_meta = meta.iloc[split_idx].copy()
    mask = split_meta["subject_id"].astype(str).to_numpy(dtype=object) == str(subject_id)
    idx = split_idx[np.flatnonzero(mask)]
    if len(idx) == 0:
        return idx

    # Chronological order is important for calibration.
    order_values = meta.iloc[idx]["sequence_end"].to_numpy()
    order = np.argsort(order_values, kind="mergesort")
    return idx[order]


def select_subjects(meta: pd.DataFrame, split_idx: np.ndarray, cfg: BaselineConfig) -> list[str]:
    split_meta = meta.iloc[split_idx].copy()
    counts = split_meta["subject_id"].astype(str).value_counts()
    counts = counts[counts >= cfg.min_subject_sequences]
    if counts.empty:
        return []
    if cfg.subject_selection == "largest":
        subjects = counts.sort_values(ascending=False).index.tolist()
    elif cfg.subject_selection == "random":
        rng = np.random.default_rng(cfg.random_state)
        arr = counts.index.to_numpy(dtype=object)
        rng.shuffle(arr)
        subjects = [str(x) for x in arr]
    else:
        subjects = sorted([str(x) for x in counts.index.tolist()])
    if cfg.max_subjects is not None and cfg.max_subjects > 0:
        subjects = subjects[: cfg.max_subjects]
    return [str(s) for s in subjects]


def split_calibration_eval_indices(subject_idx: np.ndarray, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(subject_idx))
    n_cal = int(math.floor(n * cfg.calibration_frac))
    n_cal = max(0, min(n_cal, n - cfg.min_eval_sequences))
    if n_cal <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), subject_idx.copy()

    cal_pool = subject_idx[:n_cal]
    eval_idx = subject_idx[n_cal:]
    if len(eval_idx) < cfg.min_eval_sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if len(cal_pool) >= 4 and cfg.calibration_val_frac > 0:
        n_val = max(1, int(round(len(cal_pool) * cfg.calibration_val_frac)))
        n_val = min(n_val, len(cal_pool) - 1)
        cal_train = cal_pool[:-n_val]
        cal_val = cal_pool[-n_val:]
    else:
        cal_train = cal_pool
        cal_val = np.array([], dtype=np.int64)

    return cal_train.astype(np.int64), cal_val.astype(np.int64), eval_idx.astype(np.int64)


def calibrate_subject(
    base_model: nn.Module,
    base_state: dict,
    x: np.ndarray,
    y: np.ndarray,
    subject_idx: np.ndarray,
    cfg: BaselineConfig,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, dict]:
    model = copy.deepcopy(base_model)
    model.load_state_dict(base_state)
    model.to(device)

    cal_train, cal_val, eval_idx = split_calibration_eval_indices(subject_idx, cfg)
    if len(eval_idx) == 0:
        return model, eval_idx, {"epochs_ran": 0, "best_val_loss": float("nan"), "n_cal_train": 0, "n_cal_val": 0}

    if len(cal_train) == 0 or cfg.no_calibration:
        return model, eval_idx, {
            "epochs_ran": 0,
            "best_val_loss": float("nan"),
            "n_cal_train": int(len(cal_train)),
            "n_cal_val": int(len(cal_val)),
        }

    params = freeze_head_only(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.calibration_lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    train_loader = make_loader(x, y, cal_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x, y, cal_val, cfg.batch_size, shuffle=False) if len(cal_val) else None

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_left = cfg.calibration_patience
    epochs_ran = 0

    for epoch in range(1, cfg.calibration_epochs + 1):
        model.train()
        train_losses = []
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

        epochs_ran = epoch
        if val_loader is not None:
            current = evaluate_loss(model, val_loader, criterion, device)
        else:
            current = float(np.mean(train_losses)) if train_losses else float("inf")

        if current < best_loss - 1e-7:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.calibration_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, eval_idx, {
        "epochs_ran": int(epochs_ran),
        "best_val_loss": float(best_loss),
        "n_cal_train": int(len(cal_train)),
        "n_cal_val": int(len(cal_val)),
    }


def evaluate_zero_full(
    model_name: str,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    cfg: BaselineConfig,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for split, idx in split_indices.items():
        pred = predict_model(model, x, idx, cfg.batch_size, device)
        metrics = compute_metrics(y[idx], pred, cfg.targets, {"model": model_name, "eval_split": split, "phase": "zero_full"})
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_subject_calibration(
    model_name: str,
    base_model: nn.Module,
    base_state: dict,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    cfg: BaselineConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[pd.DataFrame] = []
    pred_rows: list[pd.DataFrame] = []

    for split, split_idx in split_indices.items():
        subjects = select_subjects(meta, split_idx, cfg)
        for subject in subjects:
            subj_idx = subject_indices_for_split(meta, split_idx, subject)
            if len(subj_idx) < cfg.min_subject_sequences:
                continue

            cal_model, eval_idx, info = calibrate_subject(base_model, base_state, x, y, subj_idx, cfg, device)
            if len(eval_idx) < cfg.min_eval_sequences:
                continue

            zero_pred = predict_model(base_model, x, eval_idx, cfg.batch_size, device)
            cal_pred = predict_model(cal_model, x, eval_idx, cfg.batch_size, device)
            y_eval = y[eval_idx]

            prefix_base = {
                "model": model_name,
                "eval_split": split,
                "subject_id": subject,
                "phase": "zero_post_calibration_window",
                "calibration_lr": cfg.calibration_lr,
                "calibration_frac": cfg.calibration_frac,
                "n_eval": int(len(eval_idx)),
                **info,
            }
            prefix_cal = {
                **prefix_base,
                "phase": "calibrated",
            }
            metric_rows.append(compute_metrics(y_eval, zero_pred, cfg.targets, prefix_base))
            metric_rows.append(compute_metrics(y_eval, cal_pred, cfg.targets, prefix_cal))

            # Save compact prediction rows for optional further analysis.
            meta_eval = meta.iloc[eval_idx].reset_index(drop=True)
            for phase, pred in [("zero_post_calibration_window", zero_pred), ("calibrated", cal_pred)]:
                tmp = meta_eval[["source", "subject_id", "record_id", "sequence_start", "sequence_end"]].copy()
                tmp.insert(0, "model", model_name)
                tmp.insert(1, "eval_split", split)
                tmp.insert(2, "phase", phase)
                for j, target in enumerate(cfg.targets):
                    tmp[f"y_true_{target}"] = y_eval[:, j]
                    tmp[f"y_pred_{target}"] = pred[:, j]
                pred_rows.append(tmp)

    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return metrics, preds


def build_pairwise_gains(per_subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if per_subject_metrics.empty:
        return pd.DataFrame()
    keys = ["model", "eval_split", "subject_id", "target"]
    zero = per_subject_metrics[per_subject_metrics["phase"] == "zero_post_calibration_window"].copy()
    cal = per_subject_metrics[per_subject_metrics["phase"] == "calibrated"].copy()
    keep = keys + ["n", "mae", "rmse", "r2", "spearman", "n_eval", "n_cal_train", "n_cal_val", "epochs_ran", "best_val_loss"]
    for col in keep:
        if col not in zero.columns:
            zero[col] = np.nan
        if col not in cal.columns:
            cal[col] = np.nan
    paired = zero[keep].merge(cal[keep], on=keys, suffixes=("_zero", "_calibrated"), how="inner")
    if paired.empty:
        return paired
    paired["r2_gain"] = paired["r2_calibrated"] - paired["r2_zero"]
    paired["spearman_gain"] = paired["spearman_calibrated"] - paired["spearman_zero"]
    paired["mae_reduction"] = paired["mae_zero"] - paired["mae_calibrated"]
    paired["rmse_reduction"] = paired["rmse_zero"] - paired["rmse_calibrated"]
    return paired


def build_calibration_summary(pairwise_gains: pd.DataFrame) -> pd.DataFrame:
    if pairwise_gains.empty:
        return pd.DataFrame()
    rows = []
    for (model, split), g in pairwise_gains.groupby(["model", "eval_split"], sort=True):
        subj = g.groupby("subject_id")["r2_gain"].mean()
        rows.append(
            {
                "model": model,
                "eval_split": split,
                "n_subjects": int(g["subject_id"].nunique()),
                "n_subject_target_pairs": int(len(g)),
                "mean_r2_zero_post": float(g["r2_zero"].mean()),
                "mean_r2_calibrated": float(g["r2_calibrated"].mean()),
                "mean_r2_gain": float(g["r2_gain"].mean()),
                "median_r2_gain": float(g["r2_gain"].median()),
                "subject_target_r2_positive_rate": float((g["r2_gain"] > 0).mean()),
                "subject_mean_r2_positive_rate": float((subj > 0).mean()) if len(subj) else float("nan"),
                "mean_spearman_zero_post": float(g["spearman_zero"].mean()),
                "mean_spearman_calibrated": float(g["spearman_calibrated"].mean()),
                "mean_spearman_gain": float(g["spearman_gain"].mean()),
                "subject_target_spearman_positive_rate": float((g["spearman_gain"] > 0).mean()),
                "mean_mae_reduction": float(g["mae_reduction"].mean()),
                "mean_rmse_reduction": float(g["rmse_reduction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_model_comparison(model_summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or model_summary.empty:
        return
    data = model_summary[(model_summary["eval_split"] == "test") & (model_summary["phase"] == "calibrated")].copy()
    if data.empty:
        data = model_summary[(model_summary["eval_split"] == "test") & (model_summary["phase"] == "zero_full")].copy()
    if data.empty:
        return
    data = data.sort_values("mean_r2")
    plt.figure(figsize=(9, 4.8))
    plt.barh(data["model"], data["mean_r2"])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Mean test RВІ")
    plt.ylabel("Model")
    plt.title("Temporal baseline comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "test_mean_r2_by_model.png", dpi=180)
    plt.close()


def build_report(
    output_dir: Path,
    cfg: BaselineConfig,
    model_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    train_info: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Temporal baseline comparison report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Compare the final Transformer temporal model against simpler temporal baselines "
        "under the same feature set, split, targets, and personal calibration protocol."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Training summary")
    lines.append("")
    lines.append(train_info.to_markdown(index=False) if not train_info.empty else "No training info.")
    lines.append("")

    lines.append("## Model summary")
    lines.append("")
    lines.append(model_summary.to_markdown(index=False) if not model_summary.empty else "No model summary.")
    lines.append("")

    if not calibration_summary.empty:
        lines.append("## Calibration summary")
        lines.append("")
        lines.append(calibration_summary.to_markdown(index=False))
        lines.append("")

    test_cal = model_summary[(model_summary["eval_split"] == "test") & (model_summary["phase"] == "calibrated")].copy()
    if not test_cal.empty:
        best = test_cal.sort_values("mean_r2", ascending=False).iloc[0]
        lines.append("## Best calibrated test model")
        lines.append("")
        lines.append(
            f"Best calibrated test mean RВІ: `{best['model']}` with "
            f"mean RВІ={best['mean_r2']:.4f}, mean Spearman={best['mean_spearman']:.4f}."
        )
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `zero_full` evaluates a model on the full validation/test split without subject adaptation.")
    lines.append("- `zero_post_calibration_window` evaluates zero-shot predictions on the same post-calibration window used for calibrated evaluation.")
    lines.append("- `calibrated` freezes the temporal encoder/backbone and fine-tunes only the regression head for each held-out subject.")
    lines.append("- The key comparison for deployment is usually the calibrated test row.")
    lines.append("")

    (output_dir / "temporal_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train temporal baselines for EEG latent states.")
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet")
    p.add_argument("--output-dir", default="reports/temporal_baselines/pow_plus_eeg_seq8_pca123")
    p.add_argument("--models", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_MODELS))
    p.add_argument("--feature-set", default="pow_plus_eeg")
    p.add_argument("--max-features", type=int, default=448)
    p.add_argument("--targets", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_TARGETS))
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--split-level", choices=["subject", "record", "sequence"], default="subject")
    p.add_argument("--train-size", type=float, default=0.70)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--calibration-lr", type=float, default=1e-4)
    p.add_argument("--calibration-frac", type=float, default=0.20)
    p.add_argument("--calibration-epochs", type=int, default=40)
    p.add_argument("--calibration-patience", type=int, default=6)
    p.add_argument("--calibration-val-frac", type=float, default=0.25)
    p.add_argument("--max-subjects", type=int, default=30)
    p.add_argument("--min-subject-sequences", type=int, default=80)
    p.add_argument("--min-eval-sequences", type=int, default=20)
    p.add_argument("--subject-selection", choices=["largest", "random", "sorted"], default="largest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = BaselineConfig(**vars(args))

    root = Path(cfg.root).resolve()
    output_dir = repo_path(root, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    save_json(output_dir / "temporal_baseline_config.json", asdict(cfg))
    logger.info("Saved config: %s", output_dir / "temporal_baseline_config.json")

    set_seed(cfg.random_state)
    seq_module = load_module(root / "src" / "44_run_seq_len_sensitivity.py", "seq_len_module_for_temporal_baselines")

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if cfg.dry_run:
        logger.info("Dry run requested; exiting before data loading.")
        return

    df = seq_module.read_table(repo_path(root, cfg.dataset))
    id_cols = seq_module.detect_id_columns(df)
    logger.info("Loaded dataset rows: %d", len(df))
    logger.info("ID columns: %s", id_cols)

    feature_cols = call_select_feature_columns(seq_module, df, cfg, logger)
    logger.info("Feature set=%s selected features=%d", cfg.feature_set, len(feature_cols))

    x_raw, y, meta = seq_module.build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=cfg.targets,
        id_cols=id_cols,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        logger=logger,
    )

    train_idx, val_idx, test_idx, split_meta = seq_module.split_indices(
        meta=meta,
        split_level=cfg.split_level,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    save_json(output_dir / "split_meta.json", split_meta)
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    x, imputer, scaler = seq_module.fit_transform_features(x_raw, train_idx)
    with open(output_dir / "preprocessing.pkl", "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "targets": cfg.targets, "imputer": imputer, "scaler": scaler}, f)

    split_map = {"val": val_idx, "test": test_idx}
    n_features = int(x.shape[-1])
    n_targets = int(y.shape[-1])

    all_zero_metrics: list[pd.DataFrame] = []
    all_subject_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    train_rows: list[dict] = []

    for model_name in cfg.models:
        if model_name not in {"last_window_mlp", "mean_pool_mlp", "gru", "transformer"}:
            raise ValueError(f"Unsupported model in --models: {model_name}")

        logger.info("================================================================================")
        logger.info("Training model: %s", model_name)
        logger.info("================================================================================")

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "best_model.pt"
        history_path = model_dir / "train_history.csv"
        train_info_path = model_dir / "train_info.json"

        model = build_model(model_name, n_features, n_targets, cfg.seq_len, cfg, seq_module)

        if cfg.reuse_existing and checkpoint_path.exists():
            logger.info("Loading existing checkpoint: %s", checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            train_info = json.loads(train_info_path.read_text(encoding="utf-8")) if train_info_path.exists() else {}
            history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        else:
            model, history, train_info = train_base_model(model, x, y, train_idx, val_idx, cfg, device, logger)
            torch.save(model.state_dict(), checkpoint_path)
            history.to_csv(history_path, index=False)
            save_json(train_info_path, train_info)

        train_rows.append({"model": model_name, **train_info})
        base_state = copy.deepcopy(model.state_dict())
        model.to(device)

        zero_metrics = evaluate_zero_full(model_name, model, x, y, split_map, cfg, device)
        zero_metrics.to_csv(model_dir / "zero_full_metrics.csv", index=False)
        all_zero_metrics.append(zero_metrics)

        if not cfg.no_calibration:
            subj_metrics, preds = evaluate_subject_calibration(
                model_name=model_name,
                base_model=model,
                base_state=base_state,
                x=x,
                y=y,
                meta=meta,
                split_indices=split_map,
                cfg=cfg,
                device=device,
            )
            subj_metrics.to_csv(model_dir / "per_subject_calibration_metrics.csv", index=False)
            preds.to_csv(model_dir / "predictions.csv", index=False)
            all_subject_metrics.append(subj_metrics)
            all_predictions.append(preds)

    train_info_df = pd.DataFrame(train_rows)
    zero_metrics_df = pd.concat(all_zero_metrics, ignore_index=True) if all_zero_metrics else pd.DataFrame()
    subject_metrics_df = pd.concat(all_subject_metrics, ignore_index=True) if all_subject_metrics else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    pairwise_gains = build_pairwise_gains(subject_metrics_df)
    calibration_summary = build_calibration_summary(pairwise_gains)

    zero_summary = mean_metric_summary(zero_metrics_df, ["model", "eval_split", "phase"])
    calibrated_metrics = subject_metrics_df[subject_metrics_df["phase"] == "calibrated"].copy() if not subject_metrics_df.empty else pd.DataFrame()
    calibrated_summary = mean_metric_summary(calibrated_metrics, ["model", "eval_split", "phase"])
    model_summary = pd.concat([zero_summary, calibrated_summary], ignore_index=True) if not calibrated_summary.empty else zero_summary

    train_info_df.to_csv(output_dir / "train_info.csv", index=False)
    zero_metrics_df.to_csv(output_dir / "zero_full_metrics.csv", index=False)
    subject_metrics_df.to_csv(output_dir / "per_subject_calibration_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    pairwise_gains.to_csv(output_dir / "pairwise_calibration_gains.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)

    plot_model_comparison(model_summary, output_dir)
    build_report(output_dir, cfg, model_summary, calibration_summary, train_info_df)

    logger.info("Saved outputs to: %s", output_dir)
    logger.info("Main summary: %s", output_dir / "model_summary.csv")
    logger.info("Report: %s", output_dir / "temporal_baseline_report.md")


if __name__ == "__main__":
    main()

