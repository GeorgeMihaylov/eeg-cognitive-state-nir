# -*- coding: utf-8 -*-
"""
11_train_multihead_attention_short.py

Multi-head self-attention baseline по соседним EEG/POW-окнам.

Идея:
    [X_{t-1}, X_t, X_{t+1}] -> TransformerEncoder -> prediction_t

Поддерживает:
    --pm-target focus
    --pm-target excitement
    ...
    --pm-target all

Каждый запуск сохраняется в:
    reports/runs/<run_id>/

Если --pm-target all, то для каждой PM-метрики создается:
    reports/runs/<run_id>/targets/<target>/

Ключевые файлы:
    config.json
    train.log
    all_targets_summary.csv
    targets/<target>/fold_metrics.csv
    targets/<target>/aggregated_metrics.csv
    targets/<target>/predictions.parquet
    targets/<target>/report.md

Пример короткого запуска:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\11_train_multihead_attention_short.py ^
      --dataset data\\processed\\windowed_eeg_pm_dataset_w10.parquet ^
      --pm-target focus ^
      --feature-set pow_plus_eeg ^
      --feature-mode log_pow ^
      --seq-len 3 ^
      --max-samples 10000 ^
      --fold-limit 2 ^
      --epochs 12 ^
      --batch-size 128 ^
      --d-model 64 ^
      --nhead 4 ^
      --num-layers 1 ^
      --run-name mha_focus_short

Пример полного запуска по всем PM:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\11_train_multihead_attention_short.py ^
      --dataset data\\processed\\windowed_eeg_pm_dataset_w10.parquet ^
      --pm-target all ^
      --feature-set pow_plus_eeg ^
      --feature-mode log_pow ^
      --seq-len 3 ^
      --fold-limit 0 ^
      --epochs 12 ^
      --batch-size 128 ^
      --d-model 64 ^
      --nhead 4 ^
      --num-layers 1 ^
      --run-name mha_all_pm_full
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


RANDOM_STATE = 42

PM_TARGETS = {
    "attention": "PM.Attention.Scaled__mean",
    "engagement": "PM.Engagement.Scaled__mean",
    "excitement": "PM.Excitement.Scaled__mean",
    "stress": "PM.Stress.Scaled__mean",
    "relaxation": "PM.Relaxation.Scaled__mean",
    "interest": "PM.Interest.Scaled__mean",
    "focus": "PM.Focus.Scaled__mean",
}


@dataclass
class Config:
    root: str
    dataset: str
    run_name: str
    run_id: str
    pm_target: str
    target_col: Optional[str]
    feature_set: str
    feature_mode: str
    seq_len: int
    validation: str
    n_splits: int
    fold_limit: int
    max_samples: Optional[int]
    min_windows_per_subject: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    scaler: str
    center_pool: bool
    device: str
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mha_pm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def infer_pow_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("POW.") and pd.api.types.is_numeric_dtype(df[c])
    ]


def infer_eeg_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("EEG.") and "__" in c and pd.api.types.is_numeric_dtype(df[c])
    ]


def transform_pow(df: pd.DataFrame, pow_cols: List[str], feature_mode: str) -> pd.DataFrame:
    if not pow_cols:
        return pd.DataFrame(index=df.index)

    x_raw = df[pow_cols].copy()

    if feature_mode == "raw_pow":
        return x_raw

    if feature_mode == "log_pow":
        x_log = np.log1p(x_raw.clip(lower=0))
        x_log.columns = [f"log1p_{c}" for c in pow_cols]
        return x_log

    if feature_mode == "raw_plus_log_pow":
        x_log = np.log1p(x_raw.clip(lower=0))
        x_log.columns = [f"log1p_{c}" for c in pow_cols]
        return pd.concat([x_raw, x_log], axis=1)

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_feature_frame(
    df: pd.DataFrame,
    feature_set: str,
    feature_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pow_cols = infer_pow_cols(df)
    eeg_cols = infer_eeg_cols(df)

    parts = []

    if feature_set in {"pow", "pow_plus_eeg"}:
        if not pow_cols:
            raise RuntimeError("No POW.* features found.")
        parts.append(transform_pow(df, pow_cols, feature_mode))

    if feature_set in {"eeg", "pow_plus_eeg"}:
        if not eeg_cols:
            raise RuntimeError("No EEG.* feature columns found.")
        parts.append(df[eeg_cols].copy())

    if not parts:
        raise RuntimeError(f"No features selected for feature_set={feature_set}")

    x = pd.concat(parts, axis=1)

    leakage = [
        c for c in x.columns
        if c.startswith("PM.") or c.startswith("target_") or c.startswith("label_")
    ]
    if leakage:
        raise RuntimeError(f"Leakage columns found in features: {leakage[:20]}")

    info = {
        "pow_available": len(pow_cols),
        "eeg_available": len(eeg_cols),
        "features_used": x.shape[1],
        "feature_set": feature_set,
        "feature_mode": feature_mode,
    }

    return x, info


def build_sequence_meta(
    df: pd.DataFrame,
    target_col: str,
    seq_len: int,
    min_windows_per_subject: int,
) -> pd.DataFrame:
    """
    Строит sequence metadata.

    Важно:
        - Последовательности строятся только внутри одного record_id.
        - Target берется из центрального окна.
        - После сортировки по времени сохраняются original_row_idx исходного df.
          Именно эти индексы затем используются для x_frame.iloc.
    """
    if seq_len % 2 == 0:
        raise ValueError("seq_len must be odd: 3, 5, 7, ...")

    required = ["record_id", "subject_id", "source", "day", "t_center", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data = data[data[target_col].notna()].copy()

    counts = data["subject_id"].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    data = data[data["subject_id"].isin(keep_subjects)].copy()

    data = data.reset_index(drop=False)
    data = data.rename(columns={"index": "original_row_idx"})
    data = data.sort_values(["record_id", "t_center"]).reset_index(drop=True)

    half = seq_len // 2
    rows = []
    sample_id = 0

    for record_id, group in data.groupby("record_id", sort=False):
        group = group.sort_values("t_center")
        group_idx = group.index.to_numpy()

        if len(group_idx) < seq_len:
            continue

        for local_i in range(half, len(group_idx) - half):
            seq_idx = group_idx[local_i - half: local_i + half + 1]
            center_idx = group_idx[local_i]
            center = data.loc[center_idx]

            if data.loc[seq_idx, "record_id"].nunique() != 1:
                continue

            seq_original_idx = data.loc[seq_idx, "original_row_idx"].astype(int).tolist()
            center_original_idx = int(center["original_row_idx"])

            rows.append(
                {
                    "sample_id": sample_id,
                    "sequence_row_indices": json.dumps(seq_original_idx),
                    "center_row_idx": center_original_idx,
                    "record_id": center["record_id"],
                    "subject_id": center["subject_id"],
                    "source": center["source"],
                    "day": center["day"],
                    "t_center": float(center["t_center"]),
                    "target": float(center[target_col]),
                }
            )
            sample_id += 1

    return pd.DataFrame(rows)


def sequence_meta_to_arrays(
    x_frame: pd.DataFrame,
    seq_meta: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    if seq_meta.empty:
        raise RuntimeError("sequence_meta is empty.")

    first_indices = json.loads(seq_meta.iloc[0]["sequence_row_indices"])
    seq_len = len(first_indices)
    n_features = x_frame.shape[1]
    n_samples = len(seq_meta)

    x = np.empty((n_samples, seq_len, n_features), dtype=np.float32)

    for i, index_json in enumerate(seq_meta["sequence_row_indices"]):
        idx = json.loads(index_json)
        x[i] = x_frame.iloc[idx].to_numpy(dtype=np.float32)

    y = seq_meta["target"].to_numpy(dtype=np.float32)

    return x, y


def fit_transform_sequences(
    x_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = x_all[train_idx]
    x_val = x_all[val_idx]

    n_train, seq_len, n_features = x_train.shape
    n_val = x_val.shape[0]

    x_train_2d = x_train.reshape(-1, n_features)
    x_val_2d = x_val.reshape(-1, n_features)

    imputer = SimpleImputer(strategy="median")
    x_train_2d = imputer.fit_transform(x_train_2d)
    x_val_2d = imputer.transform(x_val_2d)

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    x_train_2d = scaler.fit_transform(x_train_2d)
    x_val_2d = scaler.transform(x_val_2d)

    x_train = x_train_2d.reshape(n_train, seq_len, n_features).astype(np.float32)
    x_val = x_val_2d.reshape(n_val, seq_len, n_features).astype(np.float32)

    return x_train, x_val


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MultiHeadWindowAttentionRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        center_pool: bool,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.center_pool = center_pool

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z + self.pos_embed
        h = self.encoder(z)

        if self.center_pool:
            center = self.seq_len // 2
            pooled = h[:, center, :]
        else:
            pooled = h.mean(dim=1)

        return self.head(pooled)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)

    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if pearsonr is not None:
        try:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            out["pearson"] = np.nan
    else:
        out["pearson"] = np.nan

    if spearmanr is not None:
        try:
            out["spearman"] = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            out["spearman"] = np.nan
    else:
        out["spearman"] = np.nan

    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    losses = []
    y_true_parts = []
    y_pred_parts = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)
            loss = loss_fn(out, yb)

            losses.append(float(loss.item()))
            y_true_parts.append(yb.detach().cpu().numpy().reshape(-1))
            y_pred_parts.append(out.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    return float(np.mean(losses)), regression_metrics(y_true, y_pred), y_true, y_pred


def train_fold(
    fold_id: int,
    validation_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_meta: pd.DataFrame,
    config: Config,
    target_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)

    train_loader = DataLoader(
        SeqDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        SeqDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MultiHeadWindowAttentionRegressor(
        n_features=x_train.shape[-1],
        seq_len=config.seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        center_pool=config.center_pool,
    ).to(device)

    loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    checkpoints_dir = target_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    best_rmse = np.inf
    best_epoch = -1
    bad_epochs = 0

    history_rows = []

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(val_loss)

        current_rmse = val_metrics["rmse"]
        improved = current_rmse < best_rmse

        if improved:
            best_rmse = current_rmse
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "fold_id": fold_id,
                    "validation": validation_name,
                    "epoch": epoch,
                    "best_rmse": best_rmse,
                },
                checkpoints_dir / f"fold_{fold_id}_best.pt",
            )
        else:
            bad_epochs += 1

        lr = optimizer.param_groups[0]["lr"]
        elapsed_s = time.time() - start_time

        row = {
            "fold": fold_id,
            "validation": validation_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "best_rmse": best_rmse,
            "bad_epochs": bad_epochs,
            "lr": lr,
            "elapsed_s": elapsed_s,
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history_rows.append(row)

        logger.info(
            "[%s] fold=%s epoch=%03d train_loss=%.6f val_loss=%.6f rmse=%.6f r2=%.6f spearman=%.6f best_rmse=%.6f bad=%d lr=%.2e",
            config.pm_target,
            fold_id,
            epoch,
            train_loss,
            val_loss,
            val_metrics["rmse"],
            val_metrics["r2"],
            val_metrics["spearman"],
            best_rmse,
            bad_epochs,
            lr,
        )

        if bad_epochs >= config.patience:
            logger.info("[%s] fold=%s early stopping at epoch=%d", config.pm_target, fold_id, epoch)
            break

    checkpoint_path = checkpoints_dir / f"fold_{fold_id}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, metrics, y_true, y_pred = evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    fold_metrics = {
        "target": config.pm_target,
        "fold": fold_id,
        "validation": validation_name,
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_val_subjects": val_meta["subject_id"].nunique(),
    }
    fold_metrics.update(metrics)

    pred = val_meta.copy().reset_index(drop=True)
    pred["target_name"] = config.pm_target
    pred["fold"] = fold_id
    pred["validation"] = validation_name
    pred["y_true"] = y_true
    pred["y_pred"] = y_pred

    return pd.DataFrame(history_rows), pd.DataFrame([fold_metrics]), pred


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "r2", "pearson", "spearman"]

    rows = []

    group_cols = ["target", "validation"]
    for (target, validation), group in fold_metrics.groupby(group_cols, dropna=False):
        row = {
            "target": target,
            "validation": validation,
            "folds": len(group),
            "n_val_total": int(group["n_val"].sum()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std()) if len(group) > 1 else 0.0
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_target(agg_metrics: pd.DataFrame) -> Dict[str, Any]:
    if agg_metrics.empty:
        return {}

    # В текущем режиме обычно есть один validation group, но оставляем общий случай.
    row = agg_metrics.iloc[0].to_dict()
    return row


def plot_loss(history: pd.DataFrame, figures_dir: Path) -> List[Path]:
    paths = []

    if history.empty:
        return paths

    for fold, group in history.groupby("fold"):
        path = figures_dir / f"loss_fold_{fold}.png"

        plt.figure(figsize=(8, 5))
        plt.plot(group["epoch"], group["train_loss"], label="train_loss")
        plt.plot(group["epoch"], group["val_loss"], label="val_loss")
        plt.title(f"Loss curve fold {fold}")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        save_plot(path)
        paths.append(path)

    return paths


def plot_scatter(predictions: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if predictions.empty:
        return None

    sample = predictions.copy()
    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=RANDOM_STATE)

    path = figures_dir / "prediction_scatter.png"

    plt.figure(figsize=(7, 6))
    plt.scatter(sample["y_true"], sample["y_pred"], s=8, alpha=0.4)
    plt.title("Multi-head attention prediction")
    plt.xlabel("true")
    plt.ylabel("predicted")

    vmin = min(sample["y_true"].min(), sample["y_pred"].min())
    vmax = max(sample["y_true"].max(), sample["y_pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    save_plot(path)
    return path


def make_target_report(
    report_path: Path,
    config: Config,
    dataset_info: Dict[str, Any],
    feature_info: Dict[str, Any],
    sequence_info: Dict[str, Any],
    fold_metrics: pd.DataFrame,
    agg_metrics: pd.DataFrame,
    figures: List[Path],
    target_dir: Path,
) -> None:
    lines = []

    lines.append("# Multi-head attention PM target report")
    lines.append("")
    lines.append(f"Run ID: `{config.run_id}`")
    lines.append(f"PM target: `{config.pm_target}`")
    lines.append(f"Target column: `{config.target_col}`")
    lines.append(f"Target directory: `{target_dir}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    cfg = pd.DataFrame([asdict(config)]).T.reset_index()
    cfg.columns = ["parameter", "value"]
    lines.append(df_to_markdown_safe(cfg))
    lines.append("")

    lines.append("## Dataset info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([dataset_info])))
    lines.append("")

    lines.append("## Feature info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([feature_info])))
    lines.append("")

    lines.append("## Sequence info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([sequence_info])))
    lines.append("")

    lines.append("## Fold metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(fold_metrics) if not fold_metrics.empty else "_No fold metrics._")
    lines.append("")

    lines.append("## Aggregated metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(agg_metrics) if not agg_metrics.empty else "_No aggregated metrics._")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        try:
            rel = fig.relative_to(target_dir)
        except Exception:
            rel = fig
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("The model receives neighboring windows from the same `record_id` and predicts the PM value of the center window.")
    lines.append("The main comparison should be against the tabular GroupKFold baseline for the same PM target.")
    lines.append("Improvement is supported if RMSE decreases and R2/Spearman increase.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_target(
    base_config: Config,
    df: pd.DataFrame,
    x_frame: pd.DataFrame,
    feature_info: Dict[str, Any],
    pm_target: str,
    target_col: str,
    run_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_config = Config(
        **{
            **asdict(base_config),
            "pm_target": pm_target,
            "target_col": target_col,
        }
    )

    target_dir = run_dir / "targets" / pm_target
    figures_dir = target_dir / "figures"
    ensure_dir(target_dir)
    ensure_dir(figures_dir)

    logger.info("=" * 80)
    logger.info("Target: %s | column: %s", pm_target, target_col)
    logger.info("=" * 80)

    dataset_info = {
        "rows": len(df),
        "columns": df.shape[1],
        "records": df["record_id"].nunique(),
        "subjects": df["subject_id"].nunique(),
        "sources": json.dumps(df["source"].value_counts().to_dict(), ensure_ascii=False),
        "target_col": target_col,
        "target_non_null": int(df[target_col].notna().sum()),
    }

    seq_meta = build_sequence_meta(
        df=df,
        target_col=target_col,
        seq_len=target_config.seq_len,
        min_windows_per_subject=target_config.min_windows_per_subject,
    )

    if seq_meta.empty:
        raise RuntimeError(f"No sequences built for target={pm_target}")

    if target_config.max_samples is not None and len(seq_meta) > target_config.max_samples:
        seq_meta = (
            seq_meta
            .sample(target_config.max_samples, random_state=target_config.seed)
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

    sequence_info = {
        "target": pm_target,
        "sequences": len(seq_meta),
        "seq_len": target_config.seq_len,
        "subjects": seq_meta["subject_id"].nunique(),
        "records": seq_meta["record_id"].nunique(),
        "sources": json.dumps(seq_meta["source"].value_counts().to_dict(), ensure_ascii=False),
        "target_mean": float(seq_meta["target"].mean()),
        "target_std": float(seq_meta["target"].std()),
        "target_min": float(seq_meta["target"].min()),
        "target_median": float(seq_meta["target"].median()),
        "target_max": float(seq_meta["target"].max()),
    }

    logger.info("[%s] Dataset info: %s", pm_target, dataset_info)
    logger.info("[%s] Sequence info: %s", pm_target, sequence_info)

    seq_meta.to_csv(target_dir / "sequence_metadata.csv", index=False, encoding="utf-8-sig")

    logger.info("[%s] Building arrays...", pm_target)
    x_all, y_all = sequence_meta_to_arrays(x_frame, seq_meta)
    logger.info("[%s] X shape: %s | y shape: %s", pm_target, x_all.shape, y_all.shape)

    groups = seq_meta["subject_id"].astype(str).to_numpy()
    n_splits_eff = min(target_config.n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits_eff)
    splits = list(gkf.split(seq_meta, y_all, groups=groups))

    if target_config.fold_limit and target_config.fold_limit > 0:
        splits = splits[:target_config.fold_limit]

    all_history = []
    all_fold_metrics = []
    all_predictions = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        validation_name = "groupkfold_subject"

        logger.info(
            "[%s] Fold %d/%d | n_train=%d | n_val=%d",
            pm_target,
            fold_id,
            len(splits),
            len(train_idx),
            len(val_idx),
        )

        x_train, x_val = fit_transform_sequences(
            x_all=x_all,
            train_idx=train_idx,
            val_idx=val_idx,
            scaler_name=target_config.scaler,
        )

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        val_meta = seq_meta.iloc[val_idx].copy().reset_index(drop=True)

        history_df, fold_metrics_df, predictions_df = train_fold(
            fold_id=fold_id,
            validation_name=validation_name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            val_meta=val_meta,
            config=target_config,
            target_dir=target_dir,
            logger=logger,
        )

        all_history.append(history_df)
        all_fold_metrics.append(fold_metrics_df)
        all_predictions.append(predictions_df)

    history = pd.concat(all_history, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    agg_metrics = aggregate_metrics(fold_metrics)

    history.to_csv(target_dir / "epoch_history.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(target_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    agg_metrics.to_csv(target_dir / "aggregated_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(target_dir / "predictions.parquet", index=False)

    figures = []
    figures.extend(plot_loss(history, figures_dir))

    scatter_path = plot_scatter(predictions, figures_dir)
    if scatter_path is not None:
        figures.append(scatter_path)

    make_target_report(
        report_path=target_dir / "report.md",
        config=target_config,
        dataset_info=dataset_info,
        feature_info=feature_info,
        sequence_info=sequence_info,
        fold_metrics=fold_metrics,
        agg_metrics=agg_metrics,
        figures=figures,
        target_dir=target_dir,
    )

    logger.info("[%s] Aggregated metrics:\n%s", pm_target, agg_metrics.to_string(index=False))

    return fold_metrics, agg_metrics, predictions


def make_all_targets_report(
    run_dir: Path,
    config: Config,
    all_summary: pd.DataFrame,
    feature_info: Dict[str, Any],
) -> None:
    lines = []

    lines.append("# Multi-head attention all-PM summary")
    lines.append("")
    lines.append(f"Run ID: `{config.run_id}`")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    cfg = pd.DataFrame([asdict(config)]).T.reset_index()
    cfg.columns = ["parameter", "value"]
    lines.append(df_to_markdown_safe(cfg))
    lines.append("")

    lines.append("## Feature info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([feature_info])))
    lines.append("")

    lines.append("## All targets summary")
    lines.append("")
    lines.append(df_to_markdown_safe(all_summary) if not all_summary.empty else "_No summary._")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("Targets should be ranked primarily by R2 and Spearman under GroupKFold by subject.")
    lines.append("The result should be compared with the tabular multi-PM baseline using the same target definitions.")
    lines.append("If MHA improves R2/Spearman and lowers RMSE for a PM target, temporal context is useful for that target.")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=r"D:\PycharmProjects\eeg-cognitive-state-nir")
    parser.add_argument("--dataset", type=str, default=r"data\processed\windowed_eeg_pm_dataset_w10.parquet")
    parser.add_argument("--run-name", type=str, default="mha_pm")

    parser.add_argument(
        "--pm-target",
        type=str,
        choices=list(PM_TARGETS.keys()) + ["all"],
        default="focus",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Custom target column. Allowed only when --pm-target is not all.",
    )

    parser.add_argument("--feature-set", type=str, choices=["pow", "eeg", "pow_plus_eeg"], default="pow_plus_eeg")
    parser.add_argument("--feature-mode", type=str, choices=["raw_pow", "log_pow", "raw_plus_log_pow"], default="log_pow")

    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--validation", type=str, choices=["groupkfold"], default="groupkfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0, help="0 means all folds.")
    parser.add_argument("--max-samples", type=int, default=None, help="If omitted, all sequences are used.")
    parser.add_argument("--min-windows-per-subject", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scaler", type=str, choices=["standard", "robust"], default="standard")
    parser.add_argument("--mean-pool", action="store_true", help="Use mean pooling instead of center token.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pm_target == "all" and args.target_col is not None:
        raise ValueError("--target-col cannot be used with --pm-target all")

    set_seed(args.seed)

    root = Path(args.root).resolve()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.run_name}_{args.pm_target}_{args.feature_set}_len{args.seq_len}"
    run_id = run_id.replace(" ", "_")

    run_dir = root / "reports" / "runs" / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "targets")

    logger = setup_logger(run_dir)

    base_config = Config(
        root=str(root),
        dataset=str(dataset_path),
        run_name=args.run_name,
        run_id=run_id,
        pm_target=args.pm_target,
        target_col=args.target_col,
        feature_set=args.feature_set,
        feature_mode=args.feature_mode,
        seq_len=args.seq_len,
        validation=args.validation,
        n_splits=args.n_splits,
        fold_limit=args.fold_limit,
        max_samples=args.max_samples,
        min_windows_per_subject=args.min_windows_per_subject,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        scaler=args.scaler,
        center_pool=not args.mean_pool,
        device=args.device,
        seed=args.seed,
    )

    (run_dir / "config.json").write_text(
        json.dumps(asdict(base_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("Multi-head attention PM baseline")
    logger.info("=" * 80)
    logger.info("Run dir: %s", run_dir)
    logger.info("Dataset: %s", dataset_path)
    logger.info("PM target mode: %s", args.pm_target)
    logger.info("Device: %s", args.device)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    df = pd.read_parquet(dataset_path).reset_index(drop=True)

    logger.info("Loaded dataset: rows=%d columns=%d", len(df), df.shape[1])

    x_frame, feature_info = build_feature_frame(
        df=df,
        feature_set=args.feature_set,
        feature_mode=args.feature_mode,
    )

    logger.info("Feature info: %s", feature_info)

    if args.pm_target == "all":
        targets_to_run = list(PM_TARGETS.items())
    else:
        target_col = args.target_col or PM_TARGETS[args.pm_target]
        targets_to_run = [(args.pm_target, target_col)]

    all_fold_metrics = []
    all_agg_metrics = []
    summary_rows = []

    for pm_target, target_col in targets_to_run:
        if target_col not in df.columns:
            logger.warning("Skipping target=%s because column is missing: %s", pm_target, target_col)
            continue

        fold_metrics, agg_metrics, _ = run_single_target(
            base_config=base_config,
            df=df,
            x_frame=x_frame,
            feature_info=feature_info,
            pm_target=pm_target,
            target_col=target_col,
            run_dir=run_dir,
            logger=logger,
        )

        all_fold_metrics.append(fold_metrics)
        all_agg_metrics.append(agg_metrics)

        if not agg_metrics.empty:
            row = agg_metrics.iloc[0].to_dict()
            row["target_col"] = target_col
            summary_rows.append(row)

    if all_fold_metrics:
        all_fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    else:
        all_fold_metrics_df = pd.DataFrame()

    if all_agg_metrics:
        all_agg_metrics_df = pd.concat(all_agg_metrics, ignore_index=True)
    else:
        all_agg_metrics_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        sort_cols = []
        ascending = []

        if "r2_mean" in summary_df.columns:
            sort_cols.append("r2_mean")
            ascending.append(False)
        if "spearman_mean" in summary_df.columns:
            sort_cols.append("spearman_mean")
            ascending.append(False)
        if "rmse_mean" in summary_df.columns:
            sort_cols.append("rmse_mean")
            ascending.append(True)

        summary_df = summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    all_fold_metrics_df.to_csv(run_dir / "all_targets_fold_metrics.csv", index=False, encoding="utf-8-sig")
    all_agg_metrics_df.to_csv(run_dir / "all_targets_aggregated_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "all_targets_summary.csv", index=False, encoding="utf-8-sig")

    make_all_targets_report(
        run_dir=run_dir,
        config=base_config,
        all_summary=summary_df,
        feature_info=feature_info,
    )

    logger.info("=" * 80)
    logger.info("Finished all targets")
    logger.info("=" * 80)
    logger.info("Saved run dir: %s", run_dir)

    if not summary_df.empty:
        logger.info("All targets summary:\n%s", summary_df.to_string(index=False))
    else:
        logger.info("No target summary produced.")


if __name__ == "__main__":
    main()