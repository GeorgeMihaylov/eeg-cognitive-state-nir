from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_CANDIDATES = {
    "source": ["source", "dataset", "data_source"],
    "subject_id": ["subject_id", "subject", "participant_id", "user_id"],
    "record_id": ["record_id", "record", "session_id", "file_id"],
    "window_start": ["t_start", "window_start", "start_time", "start"],
    "window_end": ["t_end", "window_end", "end_time", "end"],
}

NON_FEATURE_KEYWORDS = [
    "pm.", "pm_", "slow_pm_", "slow_pca_", "latent", "target", "label", "class",
    "fold", "split", "source", "subject", "record", "session", "window", "file", "path",
    "timestamp", "datetime", "date", "time", "annotation", "marker",
]


@dataclass
class Config:
    dataset: Path
    output_dir: Path
    run_name: str
    targets: list[str]
    feature_set: str
    max_features: int | None
    max_rows: int | None
    max_sequences: int | None
    seq_len: int
    stride: int
    max_subjects: int | None
    min_subject_sequences: int
    subject_selection: str
    calibration_fracs: list[float]
    test_size_for_base_val: float
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    pooling: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    min_delta: float
    calibration_mode: str
    calibration_epochs: int
    calibration_lr: float
    calibration_weight_decay: float
    calibration_patience: int
    device: str
    random_state: int
    save_models: bool


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
    ):
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(x)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("transformer_latent_calibration")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def find_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def detect_id_columns(columns: list[str]) -> dict[str, str]:
    found = {}
    for logical_name, candidates in ID_CANDIDATES.items():
        col = find_first_existing(columns, candidates)
        if col is not None:
            found[logical_name] = col
    return found


def is_feature_column(col: str, id_cols: dict[str, str]) -> bool:
    if col in set(id_cols.values()):
        return False
    low = col.lower()
    return not any(keyword in low for keyword in NON_FEATURE_KEYWORDS)


def select_feature_columns(df: pd.DataFrame, id_cols: dict[str, str], feature_set: str, max_features: int | None) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if is_feature_column(c, id_cols)]
    if feature_set == "numeric":
        selected = candidates
    elif feature_set == "pow":
        selected = [c for c in candidates if any(k in c.lower() for k in ["pow", "band", "delta", "theta", "alpha", "beta", "gamma"])]
    elif feature_set == "eeg":
        selected = [c for c in candidates if "eeg" in c.lower() and "pow" not in c.lower() and "band" not in c.lower()]
    elif feature_set == "pow_plus_eeg":
        selected = [c for c in candidates if any(k in c.lower() for k in ["eeg", "pow", "band", "delta", "theta", "alpha", "beta", "gamma"])]
        if len(selected) < 10:
            selected = candidates
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    selected = list(dict.fromkeys(selected))
    if max_features is not None and len(selected) > max_features:
        variances = df[selected].var(numeric_only=True).sort_values(ascending=False)
        selected = variances.head(max_features).index.tolist()
    return selected


def choose_group_columns(id_cols: dict[str, str]) -> list[str]:
    group_cols = [id_cols[k] for k in ["source", "subject_id", "record_id"] if k in id_cols]
    if not group_cols:
        raise ValueError("No group columns found. Need at least source/subject/record.")
    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    return [id_cols["window_start"]] if "window_start" in id_cols else []


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    id_cols: dict[str, str],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)
    x_parts, y_parts, meta_rows = [], [], []
    sequence_id = 0
    for group_key, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group = group.copy()
        if sort_cols:
            group = group.sort_values(sort_cols)
        group = group.reset_index(drop=False).rename(columns={"index": "__row_index"})
        if len(group) < seq_len:
            continue
        x_values = group[feature_cols].to_numpy(dtype=np.float32)
        y_values = group[target_cols].to_numpy(dtype=np.float32)
        group_meta = dict(zip(group_cols, group_key))
        for start in range(0, len(group) - seq_len + 1, stride):
            end = start + seq_len
            target_row = end - 1
            y = y_values[target_row]
            if not np.all(np.isfinite(y)):
                continue
            x_parts.append(x_values[start:end])
            y_parts.append(y)
            last = group.iloc[target_row]
            meta = {
                "sequence_id": sequence_id,
                "start_row_index": int(group.iloc[start]["__row_index"]),
                "end_row_index": int(last["__row_index"]),
                "seq_start_position": int(start),
                "seq_end_position": int(end - 1),
            }
            for logical_name, col in id_cols.items():
                if col in last:
                    meta[logical_name] = last[col]
            for col, value in group_meta.items():
                meta[f"group_{col}"] = value
            meta_rows.append(meta)
            sequence_id += 1
    if not x_parts:
        raise RuntimeError("No valid sequences were created.")
    return np.stack(x_parts).astype(np.float32), np.stack(y_parts).astype(np.float32), pd.DataFrame(meta_rows)


def fit_transform_x(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    n_train, seq_len, n_features = x_train.shape
    n_val = x_val.shape[0]
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x_train_scaled = pipe.fit_transform(x_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    x_val_scaled = pipe.transform(x_val.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)
    return x_train_scaled.astype(np.float32), x_val_scaled.astype(np.float32), pipe


def transform_x_with_pipeline(x: np.ndarray, pipe: Pipeline) -> np.ndarray:
    n, seq_len, n_features = x.shape
    return pipe.transform(x.reshape(-1, n_features)).reshape(n, seq_len, n_features).astype(np.float32)


def fit_transform_y(y_train: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(y_train).astype(np.float32), scaler.transform(y_val).astype(np.float32), scaler


def transform_y_with_scaler(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(y).astype(np.float32)


def inverse_y(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled)


def make_model(config: Config, input_dim: int, output_dim: int) -> TransformerRegressor:
    return TransformerRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        pooling=config.pooling,
    )


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    device: torch.device,
    trainable_mode: str = "full",
) -> tuple[nn.Module, pd.DataFrame]:
    if trainable_mode == "head_only":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")
    elif trainable_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown trainable_mode: {trainable_mode}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters.")
    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state, best_val_loss, best_epoch, bad_epochs = None, float("inf"), 0, 0
    history_rows = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu().item()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": time.perf_counter() - started,
            "trainable_mode": trainable_mode,
            "best_epoch_before_update": best_epoch,
        })
        if val_loss < best_val_loss - min_delta:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    history["best_val_loss"] = best_val_loss
    history["final_best_epoch"] = best_epoch
    return model, history


def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def regression_metrics_by_target(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for i, target in enumerate(target_cols):
        yt, yp = y_true[:, i], y_pred[:, i]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]
        if len(yt) == 0:
            row = {"target": target, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
        else:
            row = {
                "target": target,
                "mae": mean_absolute_error(yt, yp),
                "rmse": math.sqrt(mean_squared_error(yt, yp)),
                "r2": r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
                "pearson": pd.Series(yt).corr(pd.Series(yp), method="pearson"),
                "spearman": pd.Series(yt).corr(pd.Series(yp), method="spearman"),
            }
        rows.append(row)
    return rows


def select_subjects(meta: pd.DataFrame, config: Config) -> list[str]:
    counts = meta["subject_id"].astype(str).value_counts()
    eligible = counts[counts >= config.min_subject_sequences].index.tolist()
    if config.subject_selection == "largest":
        eligible = counts.loc[eligible].sort_values(ascending=False).index.tolist()
    elif config.subject_selection == "random":
        rng = np.random.default_rng(config.random_state)
        eligible = list(rng.permutation(eligible))
    else:
        raise ValueError(f"Unknown subject_selection: {config.subject_selection}")
    if config.max_subjects is not None:
        eligible = eligible[: config.max_subjects]
    return [str(x) for x in eligible]


def split_base_train_val(train_indices: np.ndarray, meta: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    subjects = meta.iloc[train_indices]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) >= 3:
        train_subjects, val_subjects = train_test_split(unique_subjects, test_size=config.test_size_for_base_val, random_state=config.random_state, shuffle=True)
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        base_train_idx, base_val_idx = train_indices[train_mask], train_indices[val_mask]
        if len(base_train_idx) > 0 and len(base_val_idx) > 0:
            return base_train_idx, base_val_idx
    base_train_idx, base_val_idx = train_test_split(train_indices, test_size=config.test_size_for_base_val, random_state=config.random_state, shuffle=True)
    return np.asarray(base_train_idx), np.asarray(base_val_idx)


def split_calibration_test(heldout_indices: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    heldout_indices = np.asarray(heldout_indices)
    if frac <= 0:
        return np.asarray([], dtype=int), heldout_indices
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(heldout_indices)
    n_cal = int(round(len(shuffled) * frac))
    n_cal = max(1, min(n_cal, len(shuffled) - 1))
    return np.sort(shuffled[:n_cal]), np.sort(shuffled[n_cal:])


def evaluate_and_store(
    *,
    model: nn.Module,
    x_scaled: np.ndarray,
    y_true_raw: np.ndarray,
    y_scaler: StandardScaler,
    batch_size: int,
    device: torch.device,
    target_cols: list[str],
    meta_subset: pd.DataFrame,
    mode: str,
    subject_id: str,
    calibration_frac: float,
    n_train: int,
    n_calibration: int,
    all_metric_rows: list[dict[str, object]],
    all_prediction_rows: list[pd.DataFrame],
) -> None:
    pred = inverse_y(predict_model(model, x_scaled, batch_size, device), y_scaler)
    for row in regression_metrics_by_target(y_true_raw, pred, target_cols):
        row.update({"mode": mode, "subject_id": subject_id, "calibration_frac": calibration_frac, "n_train": int(n_train), "n_calibration": int(n_calibration), "n_test": int(len(y_true_raw))})
        all_metric_rows.append(row)
    pred_df = meta_subset.reset_index(drop=True).copy()
    pred_df.insert(0, "mode", mode)
    pred_df.insert(1, "subject_id_eval", subject_id)
    pred_df.insert(2, "calibration_frac", calibration_frac)
    for i, target in enumerate(target_cols):
        pred_df[f"true_{target}"] = y_true_raw[:, i]
        pred_df[f"pred_{target}"] = pred[:, i]
    all_prediction_rows.append(pred_df)


def aggregate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in metrics.groupby(["mode", "calibration_frac", "target"], dropna=False):
        row = dict(zip(["mode", "calibration_frac", "target"], keys))
        row["subjects"] = int(sub["subject_id"].nunique())
        row["n_test_total"] = int(pd.to_numeric(sub["n_test"], errors="coerce").sum())
        row["n_calibration_mean"] = pd.to_numeric(sub["n_calibration"], errors="coerce").mean()
        for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
            values = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "calibration_frac", "mode"]).reset_index(drop=True)


def build_gain_vs_zero(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["subject_id", "calibration_frac", "target"]
    zero = metrics[metrics["mode"] == "zero_shot_matched_test"].copy()
    calibrated = metrics[metrics["mode"].str.contains("calibrated", na=False)].copy()
    if zero.empty or calibrated.empty:
        return pd.DataFrame()
    zero = zero.rename(columns={"mae": "zero_mae", "rmse": "zero_rmse", "r2": "zero_r2", "pearson": "zero_pearson", "spearman": "zero_spearman"})
    for _, cal_row in calibrated.iterrows():
        mask = np.ones(len(zero), dtype=bool)
        for col in key_cols:
            mask &= zero[col].astype(str).to_numpy() == str(cal_row[col])
        match = zero[mask]
        if match.empty:
            continue
        z = match.iloc[0]
        rows.append({
            "subject_id": cal_row["subject_id"],
            "calibration_frac": cal_row["calibration_frac"],
            "target": cal_row["target"],
            "calibrated_mode": cal_row["mode"],
            "zero_r2": z["zero_r2"],
            "calibrated_r2": cal_row["r2"],
            "r2_gain": cal_row["r2"] - z["zero_r2"],
            "zero_spearman": z["zero_spearman"],
            "calibrated_spearman": cal_row["spearman"],
            "spearman_gain": cal_row["spearman"] - z["zero_spearman"],
            "zero_mae": z["zero_mae"],
            "calibrated_mae": cal_row["mae"],
            "mae_gain": z["zero_mae"] - cal_row["mae"],
            "n_calibration": cal_row["n_calibration"],
            "n_test": cal_row["n_test"],
        })
    return pd.DataFrame(rows)


def build_per_subject_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in metrics.groupby(["subject_id", "mode", "calibration_frac"], dropna=False):
        rows.append({
            "subject_id": keys[0],
            "mode": keys[1],
            "calibration_frac": keys[2],
            "targets": int(sub["target"].nunique()),
            "r2_mean_across_targets": pd.to_numeric(sub["r2"], errors="coerce").mean(),
            "spearman_mean_across_targets": pd.to_numeric(sub["spearman"], errors="coerce").mean(),
            "mae_mean_across_targets": pd.to_numeric(sub["mae"], errors="coerce").mean(),
            "n_calibration": int(pd.to_numeric(sub["n_calibration"], errors="coerce").max()),
            "n_test": int(pd.to_numeric(sub["n_test"], errors="coerce").max()),
        })
    return pd.DataFrame(rows).sort_values(["subject_id", "calibration_frac", "mode"]).reset_index(drop=True)


def write_report(output_dir: Path, config: Config, dataset_info: dict[str, object], summary: pd.DataFrame, gain: pd.DataFrame, per_subject: pd.DataFrame) -> None:
    lines = []
    lines.append("# Transformer latent trajectory calibration report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Estimate whether a Transformer latent trajectory model can be personalized to a held-out subject with a small calibration subset.")
    lines.append("")
    lines.append("Main formulation:")
    lines.append("")
    lines.append("```text")
    lines.append("EEG/POW sequence -> slow_pca latent state at the last window")
    lines.append("```")
    lines.append("")
    lines.append("Calibration setting:")
    lines.append("")
    lines.append("```text")
    lines.append("train on other subjects -> evaluate held-out subject -> fine-tune on 5/10/20% of held-out subject -> evaluate remaining data")
    lines.append("```")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    for key, value in dataset_info.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(f"- d_model: `{config.d_model}`")
    lines.append(f"- n_heads: `{config.n_heads}`")
    lines.append(f"- num_layers: `{config.num_layers}`")
    lines.append(f"- dim_feedforward: `{config.dim_feedforward}`")
    lines.append(f"- dropout: `{config.dropout}`")
    lines.append(f"- pooling: `{config.pooling}`")
    lines.append(f"- base epochs: `{config.epochs}`")
    lines.append(f"- calibration mode: `{config.calibration_mode}`")
    lines.append(f"- calibration epochs: `{config.calibration_epochs}`")
    lines.append("")
    lines.append("## Calibration summary")
    lines.append("")
    if summary.empty:
        lines.append("No summary data.")
    else:
        cols = ["mode", "calibration_frac", "target", "subjects", "n_test_total", "n_calibration_mean", "mae_mean", "rmse_mean", "r2_mean", "spearman_mean"]
        cols = [c for c in cols if c in summary.columns]
        lines.append(summary[cols].to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Gain versus zero-shot")
    lines.append("")
    if gain.empty:
        lines.append("No gain table was produced.")
    else:
        gain_summary = gain.groupby(["calibrated_mode", "calibration_frac", "target"], dropna=False).agg(
            subjects=("subject_id", "nunique"),
            r2_gain_mean=("r2_gain", "mean"),
            r2_gain_std=("r2_gain", "std"),
            spearman_gain_mean=("spearman_gain", "mean"),
            mae_gain_mean=("mae_gain", "mean"),
        ).reset_index()
        lines.append(gain_summary.to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Positive `r2_gain` means that subject-specific calibration improves absolute latent-state prediction.")
    lines.append("- Positive `spearman_gain` means that calibration improves ranking/order of latent state intensity within the subject.")
    lines.append("- `head_only` calibration is the safest first personalization strategy because it keeps the temporal encoder fixed and adapts only the regression head.")
    lines.append("- If gains are unstable, the next step is reliability filtering, lower learning rate, or more conservative calibration with only `slow_pca_1` and `slow_pca_2`.")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list_arg(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Transformer latent trajectory model with held-out subject calibration.")
    parser.add_argument("--dataset", type=Path, default=Path("reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/latent_trajectory_transformer_calibration"))
    parser.add_argument("--run-name", type=str, default="latent_trajectory_transformer_calibration")
    parser.add_argument("--targets", type=str, default="slow_pca_1,slow_pca_2,slow_pca_3,slow_pca_4")
    parser.add_argument("--feature-set", type=str, default="pow_plus_eeg", choices=["numeric", "pow", "eeg", "pow_plus_eeg"])
    parser.add_argument("--max-features", type=int, default=448)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-subjects", type=int, default=12)
    parser.add_argument("--min-subject-sequences", type=int, default=80)
    parser.add_argument("--subject-selection", type=str, default="largest", choices=["largest", "random"])
    parser.add_argument("--calibration-fracs", type=str, default="0,0.10,0.20")
    parser.add_argument("--test-size-for-base-val", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--calibration-mode", type=str, default="head_only", choices=["head_only", "full"])
    parser.add_argument("--calibration-epochs", type=int, default=20)
    parser.add_argument("--calibration-lr", type=float, default=0.001)
    parser.add_argument("--calibration-weight-decay", type=float, default=0.0001)
    parser.add_argument("--calibration-patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    return Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        targets=parse_list_arg(args.targets),
        feature_set=args.feature_set,
        max_features=args.max_features,
        max_rows=args.max_rows,
        max_sequences=args.max_sequences,
        seq_len=args.seq_len,
        stride=args.stride,
        max_subjects=args.max_subjects,
        min_subject_sequences=args.min_subject_sequences,
        subject_selection=args.subject_selection,
        calibration_fracs=parse_float_list_arg(args.calibration_fracs),
        test_size_for_base_val=args.test_size_for_base_val,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        calibration_mode=args.calibration_mode,
        calibration_epochs=args.calibration_epochs,
        calibration_lr=args.calibration_lr,
        calibration_weight_decay=args.calibration_weight_decay,
        calibration_patience=args.calibration_patience,
        device=args.device,
        random_state=args.random_state,
        save_models=args.save_models,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()
    set_seed(config.random_state)
    config.dataset = config.dataset.resolve()
    config.output_dir = config.output_dir.resolve()
    models_dir = config.output_dir / "models"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
    logger.info("=" * 80)
    logger.info("Train Transformer latent trajectory model with calibration")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Device: %s", device)
    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")
    df = read_dataset(config.dataset)
    rows_loaded = len(df)
    missing_targets = [t for t in config.targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    id_cols = detect_id_columns(list(df.columns))
    if "subject_id" not in id_cols:
        raise ValueError("subject_id column is required for held-out subject calibration.")
    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows before sequence creation: %d", len(df))
    feature_cols = select_feature_columns(df=df, id_cols=id_cols, feature_set=config.feature_set, max_features=config.max_features)
    if not feature_cols:
        raise ValueError("No feature columns were selected.")
    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Selected features: %d", len(feature_cols))
    logger.info("Targets: %s", config.targets)
    x, y, meta = build_sequences(df=df, feature_cols=feature_cols, target_cols=config.targets, id_cols=id_cols, seq_len=config.seq_len, stride=config.stride)
    if config.max_sequences is not None and len(x) > config.max_sequences:
        rng = np.random.default_rng(config.random_state)
        idx = np.sort(rng.choice(np.arange(len(x)), size=config.max_sequences, replace=False))
        x, y, meta = x[idx], y[idx], meta.iloc[idx].reset_index(drop=True)
        logger.info("Sampled sequences: %d", len(x))
    meta["subject_id"] = meta["subject_id"].astype(str)
    subjects = select_subjects(meta, config)
    if not subjects:
        raise RuntimeError("No eligible subjects for calibration experiment.")
    logger.info("Created sequences: X=%s y=%s", x.shape, y.shape)
    logger.info("Selected held-out subjects: %s", subjects)
    save_json(config.output_dir / "feature_columns.json", {"feature_set": config.feature_set, "n_features": len(feature_cols), "feature_columns": feature_cols})
    save_json(config.output_dir / "model_config.json", {"run_name": config.run_name, "targets": config.targets, "seq_len": config.seq_len, "stride": config.stride, "d_model": config.d_model, "n_heads": config.n_heads, "num_layers": config.num_layers, "dim_feedforward": config.dim_feedforward, "dropout": config.dropout, "pooling": config.pooling, "batch_size": config.batch_size, "epochs": config.epochs, "lr": config.lr, "weight_decay": config.weight_decay, "patience": config.patience, "calibration_mode": config.calibration_mode, "calibration_epochs": config.calibration_epochs, "calibration_lr": config.calibration_lr, "calibration_fracs": config.calibration_fracs, "device": str(device)})
    all_metric_rows, all_prediction_rows, all_history_rows = [], [], []
    indices = np.arange(len(meta))
    for subject_no, subject_id in enumerate(subjects, start=1):
        logger.info("=" * 80)
        logger.info("Held-out subject %d/%d: %s", subject_no, len(subjects), subject_id)
        logger.info("=" * 80)
        heldout_idx = indices[meta["subject_id"].astype(str).to_numpy() == subject_id]
        other_idx = indices[meta["subject_id"].astype(str).to_numpy() != subject_id]
        if len(heldout_idx) < config.min_subject_sequences:
            logger.info("Skip subject %s: not enough sequences (%d)", subject_id, len(heldout_idx))
            continue
        base_train_idx, base_val_idx = split_base_train_val(other_idx, meta, config)
        x_train_raw, y_train_raw = x[base_train_idx], y[base_train_idx]
        x_val_raw, y_val_raw = x[base_val_idx], y[base_val_idx]
        x_train, x_val, x_pipe = fit_transform_x(x_train_raw, x_val_raw)
        y_train, y_val, y_scaler = fit_transform_y(y_train_raw, y_val_raw)
        model = make_model(config, input_dim=x_train.shape[-1], output_dim=len(config.targets))
        started = time.perf_counter()
        model, base_history = train_model(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, batch_size=config.batch_size, epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay, patience=config.patience, min_delta=config.min_delta, device=device, trainable_mode="full")
        base_fit_time = time.perf_counter() - started
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        base_history = base_history.copy()
        base_history.insert(0, "stage", "base_train")
        base_history.insert(1, "subject_id", subject_id)
        base_history.insert(2, "calibration_frac", np.nan)
        all_history_rows.append(base_history)
        if config.save_models:
            torch.save({"model_state_dict": base_state, "targets": config.targets, "feature_columns": feature_cols, "subject_id": subject_id}, models_dir / f"base_excluding_subject_{subject_id}.pt")
        logger.info("Base model trained in %.1f sec", base_fit_time)
        for frac in config.calibration_fracs:
            cal_idx, test_idx = split_calibration_test(heldout_indices=heldout_idx, frac=frac, seed=config.random_state + subject_no + int(round(frac * 1000)))
            if len(test_idx) == 0:
                logger.info("Skip subject=%s frac=%.3f: empty test split", subject_id, frac)
                continue
            x_test_scaled = transform_x_with_pipeline(x[test_idx], x_pipe)
            y_test_raw = y[test_idx]
            evaluate_and_store(model=model, x_scaled=x_test_scaled, y_true_raw=y_test_raw, y_scaler=y_scaler, batch_size=config.batch_size, device=device, target_cols=config.targets, meta_subset=meta.iloc[test_idx], mode="zero_shot" if frac == 0 else "zero_shot_matched_test", subject_id=subject_id, calibration_frac=frac, n_train=len(base_train_idx), n_calibration=0, all_metric_rows=all_metric_rows, all_prediction_rows=all_prediction_rows)
            if frac <= 0:
                continue
            if len(cal_idx) < 2:
                logger.info("Skip calibration subject=%s frac=%.3f: calibration split too small", subject_id, frac)
                continue
            x_cal_scaled = transform_x_with_pipeline(x[cal_idx], x_pipe)
            y_cal_scaled = transform_y_with_scaler(y[cal_idx], y_scaler)
            if len(cal_idx) >= 8:
                cal_train_idx, cal_val_idx = train_test_split(np.arange(len(cal_idx)), test_size=0.25, random_state=config.random_state, shuffle=True)
            else:
                cal_train_idx = np.arange(len(cal_idx))
                cal_val_idx = np.arange(len(cal_idx))
            cal_model = make_model(config, input_dim=x_train.shape[-1], output_dim=len(config.targets))
            cal_model.load_state_dict(base_state)
            cal_model.to(device)
            cal_started = time.perf_counter()
            cal_model, cal_history = train_model(model=cal_model, x_train=x_cal_scaled[cal_train_idx], y_train=y_cal_scaled[cal_train_idx], x_val=x_cal_scaled[cal_val_idx], y_val=y_cal_scaled[cal_val_idx], batch_size=min(config.batch_size, max(1, len(cal_train_idx))), epochs=config.calibration_epochs, lr=config.calibration_lr, weight_decay=config.calibration_weight_decay, patience=config.calibration_patience, min_delta=config.min_delta, device=device, trainable_mode=config.calibration_mode)
            cal_fit_time = time.perf_counter() - cal_started
            cal_history = cal_history.copy()
            cal_history.insert(0, "stage", f"calibration_{config.calibration_mode}")
            cal_history.insert(1, "subject_id", subject_id)
            cal_history.insert(2, "calibration_frac", frac)
            all_history_rows.append(cal_history)
            logger.info("Calibrated subject=%s frac=%.3f n_cal=%d in %.1f sec", subject_id, frac, len(cal_idx), cal_fit_time)
            evaluate_and_store(model=cal_model, x_scaled=x_test_scaled, y_true_raw=y_test_raw, y_scaler=y_scaler, batch_size=config.batch_size, device=device, target_cols=config.targets, meta_subset=meta.iloc[test_idx], mode=f"calibrated_{config.calibration_mode}", subject_id=subject_id, calibration_frac=frac, n_train=len(base_train_idx), n_calibration=len(cal_idx), all_metric_rows=all_metric_rows, all_prediction_rows=all_prediction_rows)
    metrics = pd.DataFrame(all_metric_rows)
    predictions = pd.concat(all_prediction_rows, ignore_index=True) if all_prediction_rows else pd.DataFrame()
    history = pd.concat(all_history_rows, ignore_index=True) if all_history_rows else pd.DataFrame()
    summary = aggregate_summary(metrics)
    gain = build_gain_vs_zero(metrics)
    per_subject = build_per_subject_summary(metrics)
    metrics.to_csv(config.output_dir / "calibration_fold_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "calibration_summary.csv", index=False)
    gain.to_csv(config.output_dir / "calibration_gain_vs_zero_shot.csv", index=False)
    per_subject.to_csv(config.output_dir / "per_subject_calibration.csv", index=False)
    history.to_csv(config.output_dir / "training_history.csv", index=False)
    predictions.to_csv(config.output_dir / "predictions.csv", index=False)
    dataset_info = {"dataset": str(config.dataset), "rows_loaded": int(rows_loaded), "rows_used": int(len(df)), "n_sequences": int(len(x)), "n_features": int(len(feature_cols)), "targets": config.targets, "heldout_subjects": subjects, "n_heldout_subjects": int(len(subjects)), "calibration_fracs": config.calibration_fracs}
    save_json(config.output_dir / "summary.json", {"run_name": config.run_name, "output_dir": str(config.output_dir), **dataset_info, "n_metric_rows": int(len(metrics)), "n_summary_rows": int(len(summary)), "n_gain_rows": int(len(gain)), "n_prediction_rows": int(len(predictions))})
    write_report(output_dir=config.output_dir, config=config, dataset_info=dataset_info, summary=summary, gain=gain, per_subject=per_subject)
    logger.info("=" * 80)
    logger.info("Saved Transformer calibration outputs")
    logger.info("=" * 80)
    logger.info("Metrics: %s", config.output_dir / "calibration_fold_metrics.csv")
    logger.info("Summary: %s", config.output_dir / "calibration_summary.csv")
    logger.info("Gain: %s", config.output_dir / "calibration_gain_vs_zero_shot.csv")
    logger.info("Per-subject: %s", config.output_dir / "per_subject_calibration.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 180):
        if not summary.empty:
            logger.info("Summary:\n%s", summary.to_string(index=False))
        if not gain.empty:
            gain_summary = gain.groupby(["calibrated_mode", "calibration_frac", "target"], dropna=False).agg(subjects=("subject_id", "nunique"), r2_gain_mean=("r2_gain", "mean"), spearman_gain_mean=("spearman_gain", "mean"), mae_gain_mean=("mae_gain", "mean")).reset_index()
            logger.info("Gain summary:\n%s", gain_summary.to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
