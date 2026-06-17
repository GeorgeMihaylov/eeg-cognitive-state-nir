#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seq_len sensitivity experiment with explicit train / validation / test split.

This script is intended for the NIR EEG project. It replaces the older
orchestrator-style seq_len script that only launched scripts 42/43.

Main differences:
  - builds EEG/POW sequences inside this script;
  - splits data explicitly into train / validation / test;
  - default split is subject-wise: no subject overlap between train/val/test;
  - trains a TransformerEncoder with early stopping on validation;
  - evaluates zero-shot quality on test subjects;
  - optionally performs head-only personal calibration on a small fraction
    of each test subject's data and evaluates on the remaining data;
  - saves per-seq_len reports and global summary tables.

Example:
  python src/44_run_seq_len_sensitivity.py ^
    --root . ^
    --dataset reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet ^
    --output-dir reports/seq_len_sensitivity/pm_w10_classic_split ^
    --seq-lens 4,8,16 ^
    --split-level subject ^
    --train-size 0.70 ^
    --val-size 0.15 ^
    --test-size 0.15 ^
    --mode all ^
    --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pickle
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for this script. Install torch in the eeg_nir environment."
    ) from exc


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3,slow_pca_4"

PM_NAME_PARTS = [
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
]

META_NAME_PARTS = [
    "source",
    "subject",
    "record",
    "session",
    "file",
    "path",
    "timestamp",
    "time",
    "date",
    "marker",
    "annotation",
    "label",
    "target",
    "fold",
    "split",
    "index",
    "idx",
    "pca",
    "latent",
    "delta",
    "trend",
    "slow",
    "fast",
    "absolute",
]


@dataclass
class ExperimentConfig:
    root: str
    dataset: str
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
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    calibration_fracs: list[float]
    calibration_epochs: int
    calibration_lr: float
    calibration_mode: str
    calibration_seed: int
    max_subjects: int | None
    min_subject_sequences: int
    subject_selection: str
    min_eval_sequences: int
    device: str
    mode: str
    skip_existing: bool
    dry_run: bool


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")

        self.pooling = pooling
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x) + self.pos_embedding[:, : x.shape[1], :]
        z = self.encoder(z)
        if self.pooling == "mean":
            z = z.mean(dim=1)
        else:
            z = z[:, -1, :]
        z = self.norm(z)
        return self.regression_head(z)


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated integer list")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("All integer values must be positive")
    return out


def parse_csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated string list")
    return out


def parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated float list")
    if any(x < 0 or x >= 1 for x in out):
        raise argparse.ArgumentTypeError("Calibration fractions must be in [0, 1)")
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
    logger = logging.getLogger("seq_len_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "seq_len_sensitivity.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def detect_id_columns(df: pd.DataFrame) -> dict[str, str | None]:
    candidates = {
        "source": ["source", "dataset", "data_source"],
        "subject_id": ["subject_id", "subject", "participant", "participant_id", "user_id"],
        "record_id": ["record_id", "record", "session", "session_id", "file_id", "trial_id"],
        "window_start": ["t_start", "window_start", "start_time", "start", "time_start"],
        "window_end": ["t_end", "window_end", "end_time", "end", "time_end"],
    }
    lower_to_col = {str(c).lower(): c for c in df.columns}
    found: dict[str, str | None] = {}
    for key, names in candidates.items():
        found[key] = None
        for name in names:
            if name.lower() in lower_to_col:
                found[key] = lower_to_col[name.lower()]
                break
    if found["source"] is None:
        df["source"] = "unknown_source"
        found["source"] = "source"
    if found["subject_id"] is None:
        raise ValueError(
            "Could not detect subject column. Expected one of: subject_id, subject, participant, user_id."
        )
    if found["record_id"] is None:
        df["record_id"] = "record_0"
        found["record_id"] = "record_id"
    if found["window_start"] is None:
        df["_row_order"] = np.arange(len(df), dtype=np.int64)
        found["window_start"] = "_row_order"
    return found


def should_exclude_feature_col(col: str, targets: Iterable[str]) -> bool:
    lower = str(col).lower()

    target_set = {str(t).lower() for t in targets}
    if lower in target_set:
        return True

    # Exclude all latent/target/PM/dynamics columns to reduce leakage risk.
    for part in META_NAME_PARTS:
        if part in lower:
            return True
    for part in PM_NAME_PARTS:
        if part in lower:
            return True
    return False



def select_feature_columns(
    df: pd.DataFrame,
    targets: list[str],
    max_features: int,
    logger: logging.Logger,
    feature_set: str = "pow_plus_eeg",
) -> list[str]:
    # Select input feature columns according to feature_set.
    # Supported: numeric, pow, eeg, pow_plus_eeg.
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in dataset: {missing_targets}")

    feature_set = str(feature_set).lower().strip()
    if feature_set not in {"numeric", "pow", "eeg", "pow_plus_eeg"}:
        raise ValueError(
            f"Unsupported feature_set={feature_set!r}. "
            "Expected one of: numeric, pow, eeg, pow_plus_eeg."
        )

    numeric_cols = df.select_dtypes(include=[np.number, "number"]).columns.tolist()
    candidates = [c for c in numeric_cols if not should_exclude_feature_col(c, targets)]

    if not candidates:
        raise ValueError("No feature columns detected after excluding target/meta columns.")

    eeg_channels = {
        "af3", "f7", "f3", "fc5", "t7", "p7", "o1",
        "o2", "p8", "t8", "fc6", "f4", "f8", "af4",
        "fp1", "fp2", "fz", "cz", "pz", "c3", "c4",
        "p3", "p4", "fcz", "cpz", "oz",
    }

    pow_markers = {
        "pow", "power", "bandpower", "band_power", "psd", "welch",
        "delta", "theta", "alpha", "beta", "gamma",
        "lowbeta", "highbeta", "low_beta", "high_beta",
        "beta_low", "beta_high",
    }

    def normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    def has_token_or_substring(norm: str, token_set: set[str]) -> bool:
        tokens = set(norm.split("_"))
        if tokens & token_set:
            return True
        return any(tok in norm for tok in token_set)

    pow_cols: list[str] = []
    eeg_cols: list[str] = []

    for col in candidates:
        norm = normalize_name(col)
        has_channel = has_token_or_substring(norm, eeg_channels)
        has_pow = has_token_or_substring(norm, pow_markers)

        is_pow = bool(has_pow)
        is_eeg = bool(has_channel and not is_pow)

        if is_pow:
            pow_cols.append(col)
        if is_eeg:
            eeg_cols.append(col)

    if feature_set == "numeric":
        selected_pool = candidates
    elif feature_set == "pow":
        selected_pool = pow_cols
    elif feature_set == "eeg":
        selected_pool = eeg_cols
        if not selected_pool:
            selected_pool = [c for c in candidates if c not in set(pow_cols)]
    elif feature_set == "pow_plus_eeg":
        selected_pool = list(dict.fromkeys(pow_cols + eeg_cols))
        if not selected_pool:
            selected_pool = candidates
    else:
        selected_pool = candidates

    if not selected_pool:
        raise ValueError(
            f"No features found for feature_set={feature_set!r}. "
            f"Candidates={len(candidates)}, pow={len(pow_cols)}, eeg={len(eeg_cols)}."
        )

    variances = (
        df[selected_pool]
        .var(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    selected = variances.sort_values(ascending=False).index.tolist()[:max_features]

    logger.info(
        "Feature set=%s | candidates=%d | pow_candidates=%d | eeg_candidates=%d | selected=%d",
        feature_set,
        len(candidates),
        len(pow_cols),
        len(eeg_cols),
        len(selected),
    )
    logger.info("Selected feature examples: %s", selected[:12])

    if len(selected) < max_features:
        logger.warning(
            "Selected only %d features for feature_set=%s; requested max_features=%d.",
            len(selected),
            feature_set,
            max_features,
        )

    return selected



def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    id_cols: dict[str, str | None],
    seq_len: int,
    stride: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    group_cols = [id_cols["source"], id_cols["subject_id"], id_cols["record_id"]]
    sort_col = id_cols["window_start"]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_rows: list[dict] = []

    # Stable sorting helps calibration use chronological first-fraction data later.
    df2 = df.sort_values(group_cols + [sort_col]).reset_index(drop=True)

    for group_key, g in df2.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values(sort_col)
        if len(g) < seq_len:
            continue

        xg = g[feature_cols].to_numpy(dtype=np.float32, copy=True)
        yg = g[target_cols].to_numpy(dtype=np.float32, copy=True)

        if np.all(np.isnan(yg)):
            continue

        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            y_last = yg[end - 1]
            if np.any(~np.isfinite(y_last)):
                continue

            x_parts.append(xg[start:end])
            y_parts.append(y_last)

            last_row = g.iloc[end - 1]
            first_row = g.iloc[start]
            meta = {
                "source": str(last_row[id_cols["source"]]),
                "subject_id": str(last_row[id_cols["subject_id"]]),
                "record_id": str(last_row[id_cols["record_id"]]),
                "sequence_start": first_row[sort_col],
                "sequence_end": last_row[sort_col],
                "group_key": "|".join(str(x) for x in group_key),
            }
            meta_rows.append(meta)

    if not x_parts:
        raise ValueError(f"No sequences created for seq_len={seq_len}.")

    x = np.stack(x_parts).astype(np.float32)
    y = np.stack(y_parts).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)
    logger.info("Created sequences: X=%s y=%s", x.shape, y.shape)
    return x, y, meta_df


def split_indices(
    meta: pd.DataFrame,
    split_level: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"train_size + val_size + test_size must be 1.0, got {total}")

    if split_level == "subject":
        split_key = "subject_id"
    elif split_level == "record":
        split_key = "group_key"
    elif split_level == "sequence":
        rng = np.random.default_rng(random_state)
        all_idx = np.arange(len(meta), dtype=np.int64)
        rng.shuffle(all_idx)
        n_train = int(round(train_size * len(all_idx)))
        n_val = int(round(val_size * len(all_idx)))
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train : n_train + n_val]
        test_idx = all_idx[n_train + n_val :]
        split_meta = {
            "split_level": split_level,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_groups_train": None,
            "n_groups_val": None,
            "n_groups_test": None,
        }
        return train_idx, val_idx, test_idx, split_meta
    else:
        raise ValueError(f"Unsupported split_level: {split_level}")

    # Important: convert PyArrow-backed pandas arrays to a normal NumPy object array.
    groups = np.asarray(meta[split_key].astype(str).dropna().unique().tolist(), dtype=object)
    if len(groups) < 3:
        raise ValueError(
            f"Need at least 3 groups for {split_level}-wise train/val/test split, got {len(groups)}."
        )

    train_groups, temp_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (val_size + test_size)
    val_groups, test_groups = train_test_split(
        np.asarray(temp_groups, dtype=object),
        train_size=relative_val_size,
        random_state=random_state + 1,
        shuffle=True,
    )

    group_values = meta[split_key].astype(str).to_numpy(dtype=object)
    train_mask = np.isin(group_values, np.asarray(train_groups, dtype=object))
    val_mask = np.isin(group_values, np.asarray(val_groups, dtype=object))
    test_mask = np.isin(group_values, np.asarray(test_groups, dtype=object))

    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    test_idx = np.flatnonzero(test_mask).astype(np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Empty split produced: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}."
        )

    split_meta = {
        "split_level": split_level,
        "split_key": split_key,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": int(len(train_groups)),
        "n_groups_val": int(len(val_groups)),
        "n_groups_test": int(len(test_groups)),
        "train_groups": [str(x) for x in train_groups],
        "val_groups": [str(x) for x in val_groups],
        "test_groups": [str(x) for x in test_groups],
    }
    return train_idx, val_idx, test_idx, split_meta


def fit_transform_features(
    x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    n, seq_len, n_features = x.shape
    x_train_flat = x[train_idx].reshape(-1, n_features)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputer.fit(x_train_flat)
    x_train_imp = imputer.transform(x_train_flat)
    scaler.fit(x_train_imp)

    x_flat = x.reshape(-1, n_features)
    x_flat = imputer.transform(x_flat)
    x_flat = scaler.transform(x_flat)
    x_scaled = x_flat.reshape(n, seq_len, n_features).astype(np.float32)
    return x_scaled, imputer, scaler


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = SequenceDataset(x[indices], y[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: ExperimentConfig,
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
    train_info = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val)}
    return model, history, train_info


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        losses.append(float(criterion(pred, yb).detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_model(
    model: nn.Module,
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SequenceDataset(x[indices], np.zeros((len(indices), 1), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    model.eval()
    preds = []
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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    prefix: dict | None = None,
) -> pd.DataFrame:
    rows = []
    prefix = prefix or {}
    for j, target in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        row = {
            **prefix,
            "target": target,
            "n": int(len(yt)),
            "mae": float(mean_absolute_error(yt, yp)) if len(yt) else float("nan"),
            "rmse": float(math.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan"),
            "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
            "spearman": safe_spearman(yt, yp),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_calibration_subjects(
    meta: pd.DataFrame,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
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
        subjects = counts.index.to_numpy(dtype=object)
        rng.shuffle(subjects)
        subjects = [str(x) for x in subjects]
    else:
        subjects = counts.index.sort_values().tolist()

    if cfg.max_subjects is not None:
        subjects = subjects[: cfg.max_subjects]
    return [str(s) for s in subjects]


def freeze_for_calibration(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head_only":
        for p in model.regression_head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported calibration_mode={mode}. Currently supported: head_only")


def calibrate_one_subject(
    base_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    cal_idx: np.ndarray,
    eval_idx: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    model = copy.deepcopy(base_model)
    model.to(device)
    freeze_for_calibration(model, cfg.calibration_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.calibration_lr)
    criterion = nn.MSELoss()

    loader = make_loader(x, y, cal_idx, cfg.batch_size, shuffle=True)

    for _epoch in range(cfg.calibration_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return predict_model(model, x, eval_idx, cfg.batch_size, device)


def run_calibration(
    base_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
    seq_len: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = select_calibration_subjects(meta, test_idx, cfg)
    logger.info("Calibration subjects: %d", len(subjects))

    per_subject_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []

    test_meta = meta.iloc[test_idx].copy()
    test_meta["_seq_idx"] = test_idx

    for subject in subjects:
        sm = test_meta[test_meta["subject_id"].astype(str) == str(subject)].copy()
        sm = sm.sort_values(["source", "record_id", "sequence_start", "sequence_end"])
        subject_indices = sm["_seq_idx"].to_numpy(dtype=np.int64)
        n_subject = len(subject_indices)
        if n_subject < cfg.min_subject_sequences:
            continue

        for frac in cfg.calibration_fracs:
            if frac == 0:
                eval_idx = subject_indices
                if len(eval_idx) < cfg.min_eval_sequences:
                    continue
                y_pred = predict_model(base_model, x, eval_idx, cfg.batch_size, device)
                cal_n = 0
            else:
                cal_n = max(1, int(math.floor(n_subject * frac)))
                cal_n = min(cal_n, n_subject - cfg.min_eval_sequences)
                if cal_n <= 0:
                    continue
                cal_idx = subject_indices[:cal_n]
                eval_idx = subject_indices[cal_n:]
                if len(eval_idx) < cfg.min_eval_sequences:
                    continue
                y_pred = calibrate_one_subject(base_model, x, y, cal_idx, eval_idx, cfg, device)

            y_true = y[eval_idx]
            metrics = compute_metrics(
                y_true,
                y_pred,
                cfg.targets,
                prefix={
                    "seq_len": seq_len,
                    "subject_id": subject,
                    "calibration_frac": float(frac),
                    "n_subject_sequences": int(n_subject),
                    "n_calibration": int(cal_n),
                    "n_eval": int(len(eval_idx)),
                },
            )
            per_subject_parts.append(metrics)

            # Keep compact prediction output for optional error analysis.
            for local_i, global_idx in enumerate(eval_idx):
                base_row = {
                    "seq_len": seq_len,
                    "subject_id": subject,
                    "calibration_frac": float(frac),
                    "global_sequence_idx": int(global_idx),
                }
                for j, target in enumerate(cfg.targets):
                    base_row[f"y_true_{target}"] = float(y_true[local_i, j])
                    base_row[f"y_pred_{target}"] = float(y_pred[local_i, j])
                pred_parts.append(pd.DataFrame([base_row]))

    if per_subject_parts:
        per_subject = pd.concat(per_subject_parts, ignore_index=True)
        summary = (
            per_subject.groupby(["seq_len", "target", "calibration_frac"], as_index=False)[
                ["mae", "rmse", "r2", "spearman"]
            ]
            .mean(numeric_only=True)
        )
        count_df = (
            per_subject.groupby(["seq_len", "target", "calibration_frac"], as_index=False)
            .agg(n_subjects=("subject_id", "nunique"), n_eval_total=("n_eval", "sum"))
        )
        summary = summary.merge(count_df, on=["seq_len", "target", "calibration_frac"], how="left")
    else:
        per_subject = pd.DataFrame()
        summary = pd.DataFrame()

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return per_subject, summary, predictions


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def output_complete(seq_dir: Path) -> bool:
    return (seq_dir / "zero_shot_test_metrics.csv").exists() and (seq_dir / "calibration_summary.csv").exists()


def process_seq_len(cfg: ExperimentConfig, root: Path, seq_len: int, logger: logging.Logger) -> None:
    out_root = repo_path(root, cfg.output_dir)
    seq_dir = out_root / f"seq{seq_len}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    if cfg.skip_existing and output_complete(seq_dir) and cfg.mode in {"all", "transformer", "calibration"}:
        logger.info("[skip] seq_len=%d already has output in %s", seq_len, seq_dir)
        return

    if cfg.dry_run:
        logger.info("[dry-run] would process seq_len=%d -> %s", seq_len, seq_dir)
        return

    dataset_path = repo_path(root, cfg.dataset)
    df = read_table(dataset_path)
    logger.info("Loaded dataset rows: %d", len(df))

    id_cols = detect_id_columns(df)
    logger.info("ID columns: %s", id_cols)

    feature_cols = select_feature_columns(df, cfg.targets, cfg.max_features, logger, feature_set=cfg.feature_set)
    x_raw, y, meta = build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=cfg.targets,
        id_cols=id_cols,
        seq_len=seq_len,
        stride=cfg.stride,
        logger=logger,
    )

    train_idx, val_idx, test_idx, split_meta = split_indices(
        meta=meta,
        split_level=cfg.split_level,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.random_state + seq_len,
    )
    logger.info(
        "Split: train=%d val=%d test=%d level=%s",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        cfg.split_level,
    )

    x, imputer, scaler = fit_transform_features(x_raw, train_idx)

    split_meta_path = seq_dir / "split_meta.json"
    save_json(split_meta_path, split_meta)

    with open(seq_dir / "preprocessing.pkl", "wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, f)

    meta_out = meta.copy()
    split_col = np.full(len(meta_out), "unused", dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "validation"
    split_col[test_idx] = "test"
    meta_out["split"] = split_col
    meta_out.to_csv(seq_dir / "sequence_meta.csv", index=False)

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    if str(device) != cfg.device:
        logger.warning("Requested device=%s, but CUDA is unavailable. Using CPU.", cfg.device)

    set_seed(cfg.random_state + seq_len)

    model = TransformerRegressor(
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

    if cfg.mode in {"all", "transformer"}:
        t0 = time.time()
        model, history, train_info = train_model(
            model=model,
            x=x,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            cfg=cfg,
            device=device,
            logger=logger,
        )
        elapsed = time.time() - t0
        train_info["elapsed_sec"] = float(elapsed)

        history.to_csv(seq_dir / "training_history.csv", index=False)
        save_json(seq_dir / "training_info.json", train_info)
        torch.save(model.state_dict(), seq_dir / "best_model.pt")

        # zero-shot test evaluation
        y_pred_test = predict_model(model, x, test_idx, cfg.batch_size, device)
        zero_metrics = compute_metrics(
            y[test_idx],
            y_pred_test,
            cfg.targets,
            prefix={"seq_len": seq_len, "split": "test", "mode": "zero_shot"},
        )
        zero_metrics.to_csv(seq_dir / "zero_shot_test_metrics.csv", index=False)

        preds = pd.DataFrame({"global_sequence_idx": test_idx})
        for j, target in enumerate(cfg.targets):
            preds[f"y_true_{target}"] = y[test_idx, j]
            preds[f"y_pred_{target}"] = y_pred_test[:, j]
        preds.to_csv(seq_dir / "zero_shot_test_predictions.csv", index=False)

    else:
        checkpoint = seq_dir / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for seq_len={seq_len}: {checkpoint}. Run --mode transformer or --mode all first."
            )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)

    if cfg.mode in {"all", "calibration"}:
        per_subject, cal_summary, cal_predictions = run_calibration(
            base_model=model,
            x=x,
            y=y,
            meta=meta,
            test_idx=test_idx,
            cfg=cfg,
            device=device,
            seq_len=seq_len,
            logger=logger,
        )
        per_subject.to_csv(seq_dir / "calibration_per_subject_metrics.csv", index=False)
        cal_summary.to_csv(seq_dir / "calibration_summary.csv", index=False)
        if not cal_predictions.empty:
            cal_predictions.to_csv(seq_dir / "calibration_predictions.csv", index=False)

        gain_df = build_calibration_gains(cal_summary)
        gain_df.to_csv(seq_dir / "calibration_gain_summary.csv", index=False)

    build_seq_report(seq_dir, cfg, seq_len)


def build_calibration_gains(cal_summary: pd.DataFrame) -> pd.DataFrame:
    if cal_summary.empty:
        return pd.DataFrame()
    rows = []
    for (seq_len, target), g in cal_summary.groupby(["seq_len", "target"]):
        zero = g[g["calibration_frac"] == 0]
        if zero.empty:
            continue
        zero_r2 = float(zero["r2"].iloc[0])
        for _, row in g.iterrows():
            frac = float(row["calibration_frac"])
            rows.append(
                {
                    "seq_len": int(seq_len),
                    "target": target,
                    "calibration_frac": frac,
                    "zero_shot_r2": zero_r2,
                    "calibrated_r2": float(row["r2"]),
                    "r2_gain": float(row["r2"] - zero_r2),
                    "spearman": float(row["spearman"]) if "spearman" in row else np.nan,
                    "n_subjects": int(row.get("n_subjects", 0)),
                }
            )
    return pd.DataFrame(rows)


def build_seq_report(seq_dir: Path, cfg: ExperimentConfig, seq_len: int) -> None:
    lines = []
    lines.append(f"# Seq_len={seq_len} report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Split")
    lines.append("")
    split_path = seq_dir / "split_meta.json"
    if split_path.exists():
        lines.append("```json")
        lines.append(split_path.read_text(encoding="utf-8"))
        lines.append("```")
        lines.append("")

    zero_path = seq_dir / "zero_shot_test_metrics.csv"
    if zero_path.exists():
        zero_df = pd.read_csv(zero_path)
        lines.append("## Zero-shot test metrics")
        lines.append("")
        lines.append(zero_df.to_markdown(index=False))
        lines.append("")

    cal_path = seq_dir / "calibration_summary.csv"
    if cal_path.exists():
        cal_df = pd.read_csv(cal_path)
        if not cal_df.empty:
            lines.append("## Calibration summary")
            lines.append("")
            lines.append(cal_df.to_markdown(index=False))
            lines.append("")

    gain_path = seq_dir / "calibration_gain_summary.csv"
    if gain_path.exists():
        gain_df = pd.read_csv(gain_path)
        if not gain_df.empty:
            lines.append("## Calibration gains")
            lines.append("")
            lines.append(gain_df.to_markdown(index=False))
            lines.append("")

    (seq_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def collect_existing_summaries(out_dir: Path, seq_lens: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zero_parts = []
    cal_parts = []
    gain_parts = []
    for seq_len in seq_lens:
        seq_dir = out_dir / f"seq{seq_len}"
        zero_path = seq_dir / "zero_shot_test_metrics.csv"
        cal_path = seq_dir / "calibration_summary.csv"
        gain_path = seq_dir / "calibration_gain_summary.csv"

        if zero_path.exists():
            z = pd.read_csv(zero_path)
            z["seq_len"] = seq_len
            zero_parts.append(z)
        if cal_path.exists():
            c = pd.read_csv(cal_path)
            c["seq_len"] = seq_len
            cal_parts.append(c)
        if gain_path.exists():
            g = pd.read_csv(gain_path)
            g["seq_len"] = seq_len
            gain_parts.append(g)

    zero = pd.concat(zero_parts, ignore_index=True) if zero_parts else pd.DataFrame()
    cal = pd.concat(cal_parts, ignore_index=True) if cal_parts else pd.DataFrame()
    gain = pd.concat(gain_parts, ignore_index=True) if gain_parts else pd.DataFrame()
    return zero, cal, gain


def summarize_all(cfg: ExperimentConfig, root: Path, logger: logging.Logger) -> None:
    out_dir = repo_path(root, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zero, cal, gain = collect_existing_summaries(out_dir, cfg.seq_lens)

    if not zero.empty:
        zero_path = out_dir / "seq_len_zero_shot_summary.csv"
        zero.to_csv(zero_path, index=False)
        logger.info("Saved: %s", zero_path)

        mean_zero = (
            zero.groupby("seq_len", as_index=False)[["mae", "rmse", "r2", "spearman"]]
            .mean(numeric_only=True)
            .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman"})
        )
        mean_zero.to_csv(out_dir / "seq_len_zero_shot_mean.csv", index=False)

    if not cal.empty:
        cal_path = out_dir / "seq_len_calibration_summary.csv"
        cal.to_csv(cal_path, index=False)
        logger.info("Saved: %s", cal_path)

        mean_cal = (
            cal.groupby(["seq_len", "calibration_frac"], as_index=False)[["mae", "rmse", "r2", "spearman"]]
            .mean(numeric_only=True)
            .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman"})
        )
        mean_cal.to_csv(out_dir / "seq_len_calibration_mean.csv", index=False)

    if not gain.empty:
        gain_path = out_dir / "seq_len_calibration_gain_summary.csv"
        gain.to_csv(gain_path, index=False)
        logger.info("Saved: %s", gain_path)

        mean_gain = (
            gain.groupby(["seq_len", "calibration_frac"], as_index=False)[["r2_gain", "calibrated_r2"]]
            .mean(numeric_only=True)
        )
        mean_gain.to_csv(out_dir / "seq_len_calibration_gain_mean.csv", index=False)

    build_global_report(out_dir, cfg, zero, cal, gain)


def build_global_report(
    out_dir: Path,
    cfg: ExperimentConfig,
    zero: pd.DataFrame,
    cal: pd.DataFrame,
    gain: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Seq_len sensitivity experiment report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Evaluate how Transformer input sequence length affects latent state prediction "
        "under an explicit train/validation/test split. The default split is subject-wise, "
        "so train, validation and test subjects do not overlap."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    if not zero.empty:
        lines.append("## Zero-shot test summary")
        lines.append("")
        show = zero[[c for c in ["seq_len", "target", "n", "mae", "rmse", "r2", "spearman"] if c in zero.columns]]
        lines.append(show.sort_values(["seq_len", "target"]).to_markdown(index=False))
        lines.append("")

        mean_zero = (
            zero.groupby("seq_len", as_index=False)[["r2", "spearman"]]
            .mean(numeric_only=True)
            .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman"})
        )
        lines.append("### Mean zero-shot metrics by seq_len")
        lines.append("")
        lines.append(mean_zero.to_markdown(index=False))
        lines.append("")

    if not cal.empty:
        lines.append("## Calibration summary")
        lines.append("")
        show = cal[
            [
                c
                for c in [
                    "seq_len",
                    "target",
                    "calibration_frac",
                    "n_subjects",
                    "n_eval_total",
                    "mae",
                    "rmse",
                    "r2",
                    "spearman",
                ]
                if c in cal.columns
            ]
        ]
        lines.append(show.sort_values(["seq_len", "target", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

        mean_cal = (
            cal.groupby(["seq_len", "calibration_frac"], as_index=False)[["r2", "spearman"]]
            .mean(numeric_only=True)
            .rename(columns={"r2": "mean_r2", "spearman": "mean_spearman"})
        )
        lines.append("### Mean calibration metrics by seq_len")
        lines.append("")
        lines.append(mean_cal.to_markdown(index=False))
        lines.append("")

    if not gain.empty:
        lines.append("## Calibration gains")
        lines.append("")
        show = gain[
            [
                c
                for c in [
                    "seq_len",
                    "target",
                    "calibration_frac",
                    "zero_shot_r2",
                    "calibrated_r2",
                    "r2_gain",
                    "n_subjects",
                ]
                if c in gain.columns
            ]
        ]
        lines.append(show.sort_values(["seq_len", "target", "calibration_frac"]).to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `seq_len=1` is close to a single-window model.")
    lines.append("- Larger `seq_len` increases temporal context, but may also smooth local dynamics.")
    lines.append("- The train/validation/test split is explicit; by default it is subject-wise.")
    lines.append("- Zero-shot test quality and personalized calibration quality should be interpreted separately.")
    lines.append("- The best practical setting is the one that is stable after personal calibration, not necessarily the best zero-shot setting.")
    lines.append("")

    report_path = out_dir / "seq_len_sensitivity_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seq_len sensitivity experiments with explicit train/val/test split."
    )
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--dataset",
        default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet",
        help="Path to slow latent states dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/seq_len_sensitivity/pm_w10_classic_split",
        help="Directory for seq_len sensitivity outputs.",
    )
    parser.add_argument("--seq-lens", type=parse_csv_ints, default=parse_csv_ints("4,8,16"))
    parser.add_argument("--targets", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_TARGETS))
    parser.add_argument("--feature-set", default="pow_plus_eeg", help="Kept for compatibility; feature columns are auto-selected.")
    parser.add_argument("--max-features", type=int, default=448)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--split-level",
        choices=["subject", "record", "sequence"],
        default="subject",
        help="How to split train/validation/test. Subject split is recommended for EEG.",
    )
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--calibration-fracs", type=parse_csv_floats, default=parse_csv_floats("0,0.05,0.10,0.20"))
    parser.add_argument("--calibration-epochs", type=int, default=20)
    parser.add_argument("--calibration-lr", type=float, default=1e-3)
    parser.add_argument("--calibration-mode", default="head_only")
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument("--max-subjects", type=int, default=30)
    parser.add_argument("--min-subject-sequences", type=int, default=80)
    parser.add_argument("--subject-selection", choices=["largest", "random", "sorted"], default="largest")
    parser.add_argument("--min-eval-sequences", type=int, default=20)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["transformer", "calibration", "summarize", "all"], default="all")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-run seq_len if outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = ExperimentConfig(
        root=args.root,
        dataset=args.dataset,
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
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        calibration_fracs=args.calibration_fracs,
        calibration_epochs=args.calibration_epochs,
        calibration_lr=args.calibration_lr,
        calibration_mode=args.calibration_mode,
        calibration_seed=args.calibration_seed,
        max_subjects=args.max_subjects,
        min_subject_sequences=args.min_subject_sequences,
        subject_selection=args.subject_selection,
        min_eval_sequences=args.min_eval_sequences,
        device=args.device,
        mode=args.mode,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )

    root = Path(cfg.root).resolve()
    out_dir = repo_path(root, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info("Saved config: %s", out_dir / "seq_len_sensitivity_config.json")
    save_json(out_dir / "seq_len_sensitivity_config.json", asdict(cfg))

    if cfg.mode in {"transformer", "calibration"}:
        # For this self-contained script, calibration requires the trained checkpoint.
        # The processing function handles checkpoint loading in calibration mode.
        pass

    if cfg.mode in {"transformer", "calibration", "all"}:
        for seq_len in cfg.seq_lens:
            logger.info("=" * 90)
            logger.info("seq_len=%d", seq_len)
            logger.info("=" * 90)
            process_seq_len(cfg, root, seq_len, logger)

    if cfg.mode in {"summarize", "all"} and not cfg.dry_run:
        summarize_all(cfg, root, logger)


if __name__ == "__main__":
    main()

