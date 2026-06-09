from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
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
    "pm.",
    "pm_",
    "slow_pm_",
    "slow_pca_",
    "latent",
    "target",
    "label",
    "class",
    "fold",
    "split",
    "source",
    "subject",
    "record",
    "session",
    "window",
    "file",
    "path",
    "timestamp",
    "datetime",
    "date",
    "time",
    "annotation",
    "marker",
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
    validation_modes: list[str]
    n_splits: int
    test_size: float

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

    run_baseline: bool
    device: str
    random_state: int
    no_plots: bool


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

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

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
    return logging.getLogger("latent_trajectory_transformer")


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

    for keyword in NON_FEATURE_KEYWORDS:
        if keyword in low:
            return False

    return True


def select_feature_columns(
    df: pd.DataFrame,
    id_cols: dict[str, str],
    feature_set: str,
    max_features: int | None,
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if is_feature_column(c, id_cols)]

    if feature_set == "numeric":
        selected = candidates

    elif feature_set == "pow":
        selected = [
            c
            for c in candidates
            if "pow" in c.lower()
            or "band" in c.lower()
            or "delta" in c.lower()
            or "theta" in c.lower()
            or "alpha" in c.lower()
            or "beta" in c.lower()
            or "gamma" in c.lower()
        ]

    elif feature_set == "eeg":
        selected = [
            c
            for c in candidates
            if "eeg" in c.lower()
            and "pow" not in c.lower()
            and "band" not in c.lower()
        ]

    elif feature_set == "pow_plus_eeg":
        selected = [
            c
            for c in candidates
            if "eeg" in c.lower()
            or "pow" in c.lower()
            or "band" in c.lower()
            or "delta" in c.lower()
            or "theta" in c.lower()
            or "alpha" in c.lower()
            or "beta" in c.lower()
            or "gamma" in c.lower()
        ]

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
    group_cols = []

    for key in ["source", "subject_id", "record_id"]:
        if key in id_cols:
            group_cols.append(id_cols[key])

    if not group_cols:
        raise ValueError("No group columns found. Need at least source/subject/record.")

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    if "window_start" in id_cols:
        return [id_cols["window_start"]]
    return []


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

    x_parts = []
    y_parts = []
    meta_rows = []

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

            x = x_values[start:end]

            x_parts.append(x)
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

    x_arr = np.stack(x_parts).astype(np.float32)
    y_arr = np.stack(y_parts).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return x_arr, y_arr, meta_df


def make_splits(
    meta: pd.DataFrame,
    validation_mode: str,
    id_cols: dict[str, str],
    n_splits: int,
    test_size: float,
    random_state: int,
):
    indices = np.arange(len(meta))

    if validation_mode == "random_split":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        yield {
            "fold": 1,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_source": "mixed",
            "test_source": "mixed",
        }
        return

    if validation_mode == "groupkfold_subject":
        if "subject_id" not in meta.columns:
            raise ValueError("subject_id is required in sequence metadata for groupkfold_subject.")

        groups = meta["subject_id"].astype(str).fillna("unknown").to_numpy()
        actual_splits = min(n_splits, len(np.unique(groups)))

        if actual_splits < 2:
            raise ValueError("Not enough unique subjects for GroupKFold.")

        splitter = GroupKFold(n_splits=actual_splits)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
            yield {
                "fold": fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_source": "mixed",
                "test_source": "mixed",
            }
        return

    if validation_mode == "cross_source":
        if "source" not in meta.columns:
            raise ValueError("source is required in sequence metadata for cross_source.")

        sources = sorted(meta["source"].dropna().astype(str).unique().tolist())

        if len(sources) < 2:
            raise ValueError(f"Need at least two sources for cross_source. Found: {sources}")

        fold = 1

        for train_source in sources:
            for test_source in sources:
                if train_source == test_source:
                    continue

                train_idx = indices[meta["source"].astype(str).to_numpy() == train_source]
                val_idx = indices[meta["source"].astype(str).to_numpy() == test_source]

                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                yield {
                    "fold": fold,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "train_source": train_source,
                    "test_source": test_source,
                }
                fold += 1
        return

    raise ValueError(f"Unknown validation mode: {validation_mode}")


def fit_transform_x(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    n_train, seq_len, n_features = x_train.shape
    n_val = x_val.shape[0]

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    x_train_2d = x_train.reshape(-1, n_features)
    x_val_2d = x_val.reshape(-1, n_features)

    x_train_scaled = pipe.fit_transform(x_train_2d).reshape(n_train, seq_len, n_features)
    x_val_scaled = pipe.transform(x_val_2d).reshape(n_val, seq_len, n_features)

    return x_train_scaled.astype(np.float32), x_val_scaled.astype(np.float32), pipe


def fit_transform_y(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    return y_train_scaled.astype(np.float32), y_val_scaled.astype(np.float32), scaler


def inverse_y(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled)


def regression_metrics_by_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
) -> list[dict[str, float | str]]:
    rows = []

    for i, target in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        valid = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[valid]
        yp = yp[valid]

        if len(yt) == 0:
            row = {
                "target": target,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "pearson": np.nan,
                "spearman": np.nan,
            }
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


def train_one_transformer_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, np.ndarray]:
    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TransformerRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        pooling=config.pooling,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        epoch_time = time.perf_counter() - started

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_sec": epoch_time,
                "best_epoch": best_epoch,
            }
        )

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if bad_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = predict_transformer(model, x_val, config.batch_size, device)

    history = pd.DataFrame(history_rows)
    history["best_val_loss"] = best_val_loss
    history["final_best_epoch"] = best_epoch

    return model, history, preds


def predict_transformer(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []

    model.eval()

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_ridge_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
) -> np.ndarray:
    x_train_last = x_train[:, -1, :]
    x_val_last = x_val[:, -1, :]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    model.fit(x_train_last, y_train)
    return model.predict(x_val_last)


def aggregate_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()

    group_cols = ["model", "validation", "target"]

    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    ]

    rows = []

    for keys, sub in fold_metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(sub["fold"].nunique())
        row["n_val_total"] = int(sub["n_val"].sum())

        for metric in metric_cols:
            row[f"{metric}_mean"] = pd.to_numeric(sub[metric], errors="coerce").mean()
            row[f"{metric}_std"] = pd.to_numeric(sub[metric], errors="coerce").std()
            row[f"{metric}_min"] = pd.to_numeric(sub[metric], errors="coerce").min()
            row[f"{metric}_max"] = pd.to_numeric(sub[metric], errors="coerce").max()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["validation", "model", "r2_mean"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def plot_training_loss(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for key, sub in history.groupby(["validation", "fold"], dropna=False):
        label = f"{key[0]} fold {key[1]}"
        ax.plot(sub["epoch"], sub["train_loss"], alpha=0.5, linestyle="--", label=f"{label} train")
        ax.plot(sub["epoch"], sub["val_loss"], alpha=0.8, label=f"{label} val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Transformer training history")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_r2_by_target(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty or "r2_mean" not in summary.columns:
        return

    pivot = summary.pivot_table(
        index="target",
        columns=["validation", "model"],
        values="r2_mean",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("R² by target")
    ax.set_xlabel("Target")
    ax.set_ylabel("R²")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_predicted_vs_true(
    predictions: pd.DataFrame,
    target_cols: list[str],
    output_dir: Path,
) -> None:
    if predictions.empty:
        return

    for target in target_cols:
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"

        if true_col not in predictions.columns or pred_col not in predictions.columns:
            continue

        sub = predictions[[true_col, pred_col]].dropna()
        if sub.empty:
            continue

        if len(sub) > 5000:
            sub = sub.sample(n=5000, random_state=42)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sub[true_col], sub[pred_col], alpha=0.25, s=8)

        low = min(sub[true_col].min(), sub[pred_col].min())
        high = max(sub[true_col].max(), sub[pred_col].max())
        ax.plot([low, high], [low, high], linestyle="--")

        ax.set_title(f"Predicted vs true: {target}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"predicted_vs_true_{target}.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    dataset_info: dict,
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Transformer latent trajectory model report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Train a TransformerEncoder model for the main project formulation:")
    lines.append("")
    lines.append("```text")
    lines.append("sequence of EEG/POW windows -> slow latent state at the last window")
    lines.append("```")
    lines.append("")
    lines.append("This implements the transition from isolated window-level prediction to temporal modeling of latent cognitive-affective trajectories.")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used: `{dataset_info['rows_used']}`")
    lines.append(f"- Sequences created: `{dataset_info['n_sequences']}`")
    lines.append(f"- Feature columns: `{dataset_info['n_features']}`")
    lines.append(f"- Targets: `{config.targets}`")
    lines.append(f"- Sequence length: `{config.seq_len}`")
    lines.append(f"- Stride: `{config.stride}`")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    lines.append(f"- d_model: `{config.d_model}`")
    lines.append(f"- n_heads: `{config.n_heads}`")
    lines.append(f"- num_layers: `{config.num_layers}`")
    lines.append(f"- dim_feedforward: `{config.dim_feedforward}`")
    lines.append(f"- dropout: `{config.dropout}`")
    lines.append(f"- pooling: `{config.pooling}`")
    lines.append(f"- epochs: `{config.epochs}`")
    lines.append(f"- patience: `{config.patience}`")
    lines.append(f"- batch_size: `{config.batch_size}`")
    lines.append(f"- device: `{config.device}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if not summary.empty:
        display_cols = [
            c
            for c in [
                "model",
                "validation",
                "target",
                "folds",
                "n_val_total",
                "mae_mean",
                "rmse_mean",
                "r2_mean",
                "pearson_mean",
                "spearman_mean",
            ]
            if c in summary.columns
        ]
        lines.append(summary[display_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No summary rows.")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If Transformer is better than the Ridge last-window baseline, local temporal dynamics add useful signal.")
    lines.append("- If Transformer is not better, the current latent states may already be mostly captured by the last EEG/POW window or by classical tabular models.")
    lines.append("- GroupKFold results are the main estimate for cross-subject generalization.")
    lines.append("- Cross-source results estimate transfer between `Old_EEG` and `gpn_data`.")
    lines.append("")

    lines.append("## Recommended next checks")
    lines.append("")
    lines.append("1. Compare Transformer against previous HGB slow latent baselines.")
    lines.append("2. Run cross-source validation if not done in the first run.")
    lines.append("3. Add user calibration mode for Transformer.")
    lines.append("4. Add EEG reliability/artifact score as an input feature or sample weight.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train TransformerEncoder model for slow latent state trajectory prediction."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet"),
        help="Input slow latent dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/latent_trajectory_transformer"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="latent_trajectory_transformer",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="slow_pca_1,slow_pca_2,slow_pca_3,slow_pca_4",
        help="Comma-separated target columns.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pow_plus_eeg",
        choices=["numeric", "pow", "eeg", "pow_plus_eeg"],
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row sampling before sequence creation.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional sequence sampling after sequence creation.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--validation-modes",
        type=str,
        default="random_split,groupkfold_subject",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["last", "mean"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run Ridge last-window baseline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

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
        validation_modes=parse_list_arg(args.validation_modes),
        n_splits=args.n_splits,
        test_size=args.test_size,
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
        run_baseline=args.run_baseline,
        device=args.device,
        random_state=args.random_state,
        no_plots=args.no_plots,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()
    set_seed(config.random_state)

    config.dataset = config.dataset.resolve()
    config.output_dir = config.output_dir.resolve()

    figures_dir = config.output_dir / "figures"
    models_dir = config.output_dir / "models"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info("=" * 80)
    logger.info("Train Transformer latent trajectory model")
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

    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows before sequence creation: %d", len(df))

    feature_cols = select_feature_columns(
        df=df,
        id_cols=id_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    if not feature_cols:
        raise ValueError("No feature columns were selected.")

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Selected features: %d", len(feature_cols))
    logger.info("Targets: %s", config.targets)

    x, y, meta = build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=config.targets,
        id_cols=id_cols,
        seq_len=config.seq_len,
        stride=config.stride,
    )

    if config.max_sequences is not None and len(x) > config.max_sequences:
        rng = np.random.default_rng(config.random_state)
        idx = rng.choice(np.arange(len(x)), size=config.max_sequences, replace=False)
        idx = np.sort(idx)
        x = x[idx]
        y = y[idx]
        meta = meta.iloc[idx].reset_index(drop=True)
        logger.info("Sampled sequences: %d", len(x))

    logger.info("Created sequences: X=%s y=%s", x.shape, y.shape)

    save_json(
        config.output_dir / "feature_columns.json",
        {
            "feature_set": config.feature_set,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
        },
    )

    save_json(
        config.output_dir / "model_config.json",
        {
            "run_name": config.run_name,
            "targets": config.targets,
            "seq_len": config.seq_len,
            "stride": config.stride,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
            "pooling": config.pooling,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "device": str(device),
        },
    )

    all_fold_metrics = []
    all_history = []
    all_predictions = []

    for validation_mode in config.validation_modes:
        logger.info("=" * 80)
        logger.info("Validation mode: %s", validation_mode)
        logger.info("=" * 80)

        for split in make_splits(
            meta=meta,
            validation_mode=validation_mode,
            id_cols=id_cols,
            n_splits=config.n_splits,
            test_size=config.test_size,
            random_state=config.random_state,
        ):
            fold = split["fold"]
            train_idx = split["train_idx"]
            val_idx = split["val_idx"]

            logger.info(
                "Fold %s | train=%d val=%d | train_source=%s test_source=%s",
                fold,
                len(train_idx),
                len(val_idx),
                split["train_source"],
                split["test_source"],
            )

            x_train_raw = x[train_idx]
            y_train_raw = y[train_idx]
            x_val_raw = x[val_idx]
            y_val_raw = y[val_idx]

            x_train, x_val, _ = fit_transform_x(x_train_raw, x_val_raw)
            y_train, y_val, y_scaler = fit_transform_y(y_train_raw, y_val_raw)

            started = time.perf_counter()

            model, history, pred_scaled = train_one_transformer_fold(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=config,
                input_dim=x_train.shape[-1],
                output_dim=len(config.targets),
                device=device,
            )

            fit_time = time.perf_counter() - started

            pred = inverse_y(pred_scaled, y_scaler)
            y_true = y_val_raw

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": x_train.shape[-1],
                        "output_dim": len(config.targets),
                        "seq_len": config.seq_len,
                        "d_model": config.d_model,
                        "n_heads": config.n_heads,
                        "num_layers": config.num_layers,
                        "dim_feedforward": config.dim_feedforward,
                        "dropout": config.dropout,
                        "pooling": config.pooling,
                    },
                    "targets": config.targets,
                    "feature_columns": feature_cols,
                },
                models_dir / f"transformer_{validation_mode}_fold{fold}.pt",
            )

            history = history.copy()
            history.insert(0, "model", "transformer")
            history.insert(1, "validation", validation_mode)
            history.insert(2, "fold", fold)
            all_history.append(history)

            metric_rows = regression_metrics_by_target(y_true, pred, config.targets)

            for row in metric_rows:
                row.update(
                    {
                        "model": "transformer",
                        "validation": validation_mode,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "train_source": split["train_source"],
                        "test_source": split["test_source"],
                        "fit_time_sec": fit_time,
                    }
                )
                all_fold_metrics.append(row)

            pred_df = meta.iloc[val_idx].reset_index(drop=True).copy()
            pred_df.insert(0, "model", "transformer")
            pred_df.insert(1, "validation", validation_mode)
            pred_df.insert(2, "fold", fold)

            for i, target in enumerate(config.targets):
                pred_df[f"true_{target}"] = y_true[:, i]
                pred_df[f"pred_{target}"] = pred[:, i]

            all_predictions.append(pred_df)

            if config.run_baseline:
                baseline_started = time.perf_counter()
                baseline_pred = run_ridge_baseline(
                    x_train=x_train_raw,
                    y_train=y_train_raw,
                    x_val=x_val_raw,
                )
                baseline_fit_time = time.perf_counter() - baseline_started

                baseline_metric_rows = regression_metrics_by_target(
                    y_true,
                    baseline_pred,
                    config.targets,
                )

                for row in baseline_metric_rows:
                    row.update(
                        {
                            "model": "ridge_last_window",
                            "validation": validation_mode,
                            "fold": fold,
                            "n_train": int(len(train_idx)),
                            "n_val": int(len(val_idx)),
                            "train_source": split["train_source"],
                            "test_source": split["test_source"],
                            "fit_time_sec": baseline_fit_time,
                        }
                    )
                    all_fold_metrics.append(row)

                baseline_pred_df = meta.iloc[val_idx].reset_index(drop=True).copy()
                baseline_pred_df.insert(0, "model", "ridge_last_window")
                baseline_pred_df.insert(1, "validation", validation_mode)
                baseline_pred_df.insert(2, "fold", fold)

                for i, target in enumerate(config.targets):
                    baseline_pred_df[f"true_{target}"] = y_true[:, i]
                    baseline_pred_df[f"pred_{target}"] = baseline_pred[:, i]

                all_predictions.append(baseline_pred_df)

    fold_metrics = pd.DataFrame(all_fold_metrics)
    history_df = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = aggregate_summary(fold_metrics)

    fold_metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    history_df.to_csv(config.output_dir / "training_history.csv", index=False)
    predictions_df.to_csv(config.output_dir / "predictions.csv", index=False)

    dataset_info = {
        "dataset": str(config.dataset),
        "rows_loaded": int(rows_loaded),
        "rows_used": int(len(df)),
        "n_sequences": int(len(x)),
        "n_features": int(len(feature_cols)),
        "targets": config.targets,
        "id_columns": id_cols,
        "validation_modes": config.validation_modes,
    }

    save_json(
        config.output_dir / "summary.json",
        {
            "run_name": config.run_name,
            "output_dir": str(config.output_dir),
            **dataset_info,
            "n_fold_metric_rows": int(len(fold_metrics)),
            "n_summary_rows": int(len(summary)),
            "n_prediction_rows": int(len(predictions_df)),
        },
    )

    if not config.no_plots:
        plot_training_loss(history_df, figures_dir / "training_loss.png")
        plot_r2_by_target(summary, figures_dir / "r2_by_target.png")
        plot_predicted_vs_true(predictions_df[predictions_df["model"] == "transformer"], config.targets, figures_dir)

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=dataset_info,
        summary=summary,
        fold_metrics=fold_metrics,
    )

    logger.info("=" * 80)
    logger.info("Saved Transformer latent trajectory outputs")
    logger.info("=" * 80)
    logger.info("Fold metrics: %s", config.output_dir / "fold_metrics.csv")
    logger.info("Summary: %s", config.output_dir / "summary.csv")
    logger.info("Predictions: %s", config.output_dir / "predictions.csv")
    logger.info("History: %s", config.output_dir / "training_history.csv")
    logger.info("Report: %s", config.output_dir / "report.md")

    with pd.option_context("display.max_rows", 40, "display.max_columns", 20, "display.width", 180):
        logger.info("Summary:\n%s", summary.to_string(index=False))

    logger.info("Done.")


if __name__ == "__main__":
    main()